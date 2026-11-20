# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 The FlashMask Authors. All rights reserved.

"""Paddle smoke test for the SM100 SwapAB decode kernel.

Covers the paddle backend of ``_flash_attn_decode_swapab`` (allocation, strides, the split-KV
combine) and the ``flashmask_attention`` dispatch of seqlen_q = 1 calls. The kernel itself is
tested in depth by tests/torch/test_decode_swapab.py.
"""

import math
from unittest import mock

import paddle
import pytest

import flashmask.frontend.paddle as fm
from flashmask.cute.ops.decode import _flash_attn_decode_swapab
from flashmask.cute.ops.fwd import _flash_attn_fwd


def _sm100_family():
    if not paddle.device.is_compiled_with_cuda():
        return False
    return tuple(paddle.device.cuda.get_device_capability()) in ((10, 0), (10, 3))


pytestmark = pytest.mark.skipif(not _sm100_family(), reason="SwapAB decode needs SM100/SM103")


def _reference(q, k, v, scale, keep=None):
    """(out (b, 1, h, dv), lse (b, h, 1)) in fp32; keep (b, h, s_k) masks keys."""
    group = q.shape[2] // k.shape[2]
    qf = q.astype("float32").transpose([0, 2, 1, 3])
    kf = paddle.repeat_interleave(k.astype("float32"), group, axis=2).transpose([0, 2, 1, 3])
    vf = paddle.repeat_interleave(v.astype("float32"), group, axis=2).transpose([0, 2, 1, 3])
    scores = paddle.matmul(qf, kf, transpose_y=True) * scale  # (b, h, 1, s_k)
    if keep is not None:
        scores = paddle.where(keep.unsqueeze(2), scores, paddle.full_like(scores, -math.inf))
    lse = paddle.logsumexp(scores, axis=-1)
    probs = paddle.nan_to_num(paddle.nn.functional.softmax(scores, axis=-1), 0.0)
    return paddle.matmul(probs, vf).transpose([0, 2, 1, 3]), lse


def _max_err(a, b):
    return (a.astype("float32") - b).abs().max().item()


@pytest.mark.parametrize("head_dim,head_dim_v", [(128, 128), (576, 512)])
@pytest.mark.parametrize("num_splits", [1, 3])
def test_decode_swapab_dense(head_dim, head_dim_v, num_splits):
    paddle.seed(0)
    q = paddle.randn([2, 1, 16, head_dim]).astype("bfloat16")
    k = paddle.randn([2, 1000, 4, head_dim]).astype("bfloat16")
    v = paddle.randn([2, 1000, 4, head_dim_v]).astype("bfloat16")
    out, lse = _flash_attn_decode_swapab(q, k, v, num_splits=num_splits, return_lse=True)
    out_ref, lse_ref = _reference(q, k, v, 1.0 / math.sqrt(head_dim))
    assert out.shape == [2, 1, 16, head_dim_v] and out.dtype == paddle.bfloat16
    assert _max_err(out, out_ref) < 2e-2
    assert _max_err(lse, lse_ref) < 1e-3


@pytest.mark.parametrize("num_vecs,causal", [(1, True), (2, False), (4, False)])
def test_flashmask_attention_decode_dispatch(num_vecs, causal):
    """seqlen_q = 1 with stop_gradient inputs runs SwapAB, matching the main kernel."""
    paddle.seed(0)
    batch, seqlen_k, num_heads, num_heads_kv = 3, 3000, 16, 4
    q = paddle.randn([batch, 1, num_heads, 128]).astype("bfloat16")
    k = paddle.randn([batch, seqlen_k, num_heads_kv, 128]).astype("bfloat16")
    v = paddle.randn([batch, seqlen_k, num_heads_kv, 128]).astype("bfloat16")
    # Bounds in {0, 1, 2}, each pair ordered (non-causal num_vecs 2 descending, UTE <= LTS):
    # row 0 sees a good share of the keys.
    m = paddle.randint(0, 3, [batch, num_heads_kv, seqlen_k, num_vecs])
    if num_vecs >= 2:
        pair = paddle.sort(m[..., :2], axis=-1, descending=num_vecs == 2 and not causal)
        m = paddle.concat([pair, m[..., 2:]], axis=-1)
    if num_vecs == 4:
        m = paddle.concat([m[..., :2], paddle.sort(m[..., 2:], axis=-1)], axis=-1)
    mask = m.astype("int32").contiguous()

    spy = mock.patch.object(fm, "_flash_attn_decode_swapab", wraps=fm._flash_attn_decode_swapab)
    with spy as called:
        out, lse = fm.flashmask_attention(
            q, k, v, mask, causal=causal, return_softmax_lse=True
        )
    assert called.call_count == 1
    out_main, lse_main = _flash_attn_fwd(
        q, k, v, causal=causal, return_lse=True, startend_row_indices=mask, pack_gqa=False
    )

    row = 0
    lts = mask[..., 0]
    masked = row >= lts
    if num_vecs == 2:
        masked = masked | (row < mask[..., 1])
    elif num_vecs == 4:
        masked = (masked & (row < mask[..., 1])) | ((row >= mask[..., 2]) & (row < mask[..., 3]))
    # repeat_interleave has no bool kernel.
    keep = paddle.repeat_interleave((~masked).astype("int32"), num_heads // num_heads_kv, axis=1)
    keep = keep.astype("bool")
    out_ref, lse_ref = _reference(q, k, v, 1.0 / math.sqrt(128), keep)
    assert _max_err(out, out_ref) < 2e-2
    assert _max_err(out_main, out_ref) < 2e-2
    assert _max_err(lse, lse_ref) < 1e-3
    assert _max_err(lse_main, lse_ref) < 1e-3
