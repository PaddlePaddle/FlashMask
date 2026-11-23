# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 The FlashMask Authors. All rights reserved.

"""FlashMask functional test (paddle).

Runs ``flashmask_attention`` against an eager fp32 reference and allows it the
error a naive bf16 implementation makes -- concretely
``|out - ref| <= 2 * |naive_bf16 - ref| + atol`` -- so the bound scales with the
configuration instead of being a hand-picked constant. Covers forward, dq/dk/dv,
and dsink across the mask library, split/shared KV, and head dims up to 576.
The ``test_decode_swapab*`` / ``test_flashmask_attention_decode_dispatch*`` tests
cover the SM100 SwapAB decode branch (seqlen_q = 1, forward only).

FlashMask only ships the FA-v4 CuTe DSL kernels (SM90 / SM100), so there is no
flash-attention version axis here.
"""

import contextlib
import math
from functools import partial
from unittest import mock

import paddle
import pytest

import flashmask.cute.ops.fwd as fwd_ops
from flashmask.cute.interface import flashmask_attention
from flashmask.cute.ops.fwd import _flash_attn_fwd

from tests.paddle.attention_ref import attention_ref
from tests.paddle.generate_startend_row_indices import (
    startend_row_indices_to_attn_bias,
    generate_none_mask,
    generate_sliding_window_mask,
    generate_causal_document_mask,
    generate_document_mask,
    generate_share_question_mask,
    generate_global_sliding_window_mask,
    generate_causal_blockwise_mask,
    generate_prefix_lm_document_mask,
    generate_prefix_lm_causal_mask,
    generate_qk_sparse_mask,
    generate_random_eviction_mask,
    generate_empty_mask,
)

# KV shared: k and v are ONE buffer (the MLA convention, v = k[..., :dv]). Only the
# big-headdim backward merges dK/dV, and only for these (d, dv) pairs -- anywhere
# else "shared" would just re-run the split path with a different allocation.
KV_SHARED_D_DV = ((512, 512), (576, 512))


def kv_shared_detected(k, v):
    """Whether the backward will take its kv-shared path for these two tensors.

    Mirrors the predicate in flashmask/cute/interface.py minus the (d, dv) gate, so
    a test can ASSERT which path it exercised: if paddle ever materialises
    ``k[..., :dv]`` as a copy, the merge would silently not run and the test would
    pass while covering nothing.
    """
    try:
        same_storage = k.data_ptr() == v.data_ptr()
    except (AttributeError, RuntimeError):
        return False
    return (
        same_storage
        and k.dtype == v.dtype
        and list(k.shape[:-1]) == list(v.shape[:-1])
        and v.shape[-1] <= k.shape[-1]
        and tuple(k.strides) == tuple(v.strides)
    )

# batch_size, seqlen_q, seqlen_k, nheads, nheads_kv
shape_cases = (
    [
        (2840, 32, 32, 16, 4),
        (1, 300, 300, 16, 16),
        (2, 16383, 16385, 4, 1),
    ]
)

_shape_cases_before = len(shape_cases)
shape_cases = list(dict.fromkeys(shape_cases))  # 保序去重
print(f"{'='*60}")
print(f"[test_flashmask] Shape Cases Summary:")
print(f"  - Original Count: {_shape_cases_before}")
print(f"  - Unique Count:   {len(shape_cases)}")
print(f"  - Removed:        {_shape_cases_before - len(shape_cases)}")
print(f"{'='*60}")

d_dv_cases = [
    (80, 80),
    (128, 128),
    (192, 128),
    (256, 256),
    (512, 512),
    (576, 512),
]

# sink dtype as a string tag (resolved to a paddle dtype in the test): pytest
# cannot stringify a paddle dtype, so a raw dtype shows up in the id as an opaque
# "sink_dtype4" instead of "fp16".
_SINK_TAGS = ["bf16", "fp16", "fp32"]
_SINK_DTYPE = {"bf16": paddle.bfloat16, "fp16": paddle.float16, "fp32": paddle.float32}

# Generate all combinations for second param
def generate_shapes():
    # sink dtype is round-robined over the generated cases: exactly uniform (the case
    # count is a multiple of len(_SINK_TAGS)), deterministic, adds no case, and shows
    # up in the pytest id so a failure tells you the dtype without reading the log.
    for i, (batch_size, seqlen_q, seqlen_k, nheads, nheads_kv, nheads_startend_row_indices) in (
        enumerate(_generate_shape_params())
    ):
        yield (
            batch_size, seqlen_q, seqlen_k, nheads, nheads_kv, nheads_startend_row_indices,
            _SINK_TAGS[i % len(_SINK_TAGS)],
        )


def _generate_shape_params():
    for batch_size, seqlen_q, seqlen_k, nheads, nheads_kv in shape_cases:
        if nheads_kv == 1:
          nheads_startend_row_indices_values = [1]
        else:
          nheads_startend_row_indices_values = [1, nheads_kv]
        for nheads_startend_row_indices in nheads_startend_row_indices_values:
            yield (
                batch_size, seqlen_q, seqlen_k, nheads, nheads_kv, nheads_startend_row_indices
            )


@pytest.mark.parametrize("dtype", [paddle.bfloat16], ids=["bf16"])
@pytest.mark.parametrize("kv_mode", ["split", "shared"])
@pytest.mark.parametrize(
    "d, dv",
    d_dv_cases,
    ids=[f"d{c[0]}-dv{c[1]}" for c in d_dv_cases]
)
@pytest.mark.parametrize(
    "batch_size, seqlen_q, seqlen_k, nheads, nheads_kv, nheads_startend_row_indices, sink_dtype",
    list(generate_shapes()),
)
@pytest.mark.parametrize(
    "gen_startend_row_indices",
    [
        partial(generate_none_mask, causal=False), # full
        partial(generate_none_mask, causal=True), # causal
        partial(generate_sliding_window_mask), # sliding window
        partial(generate_causal_document_mask), # causal document mask
        partial(generate_document_mask), # document mask
        partial(generate_share_question_mask), # share question mask
        partial(generate_global_sliding_window_mask), # global sliding window
        partial(generate_causal_blockwise_mask), # causal blockwise mask
        partial(generate_prefix_lm_document_mask), # prefix lm document mask
        partial(generate_prefix_lm_causal_mask), # prefix lm causal mask
        partial(generate_qk_sparse_mask), # qk-sparse mask
        partial(generate_random_eviction_mask), # random eviction mask
        partial(generate_empty_mask),
    ],
)
def test_flashmask(
    batch_size, seqlen_q, seqlen_k, nheads, nheads_kv, d, dv, nheads_startend_row_indices, sink_dtype, kv_mode, dtype, gen_startend_row_indices, softcap=0.0
):
    paddle.seed(2026)
    assert nheads % nheads_kv == 0
    # tag -> paddle dtype (see _SINK_TAGS for why it arrives as a string)
    sink_dtype = _SINK_DTYPE[sink_dtype]

    # SM90 kernels only support head dims up to 256.
    if paddle.device.cuda.get_device_capability()[0] == 9 and max(d, dv) > 256:
        pytest.skip(f"Skipping d{d}-dv{dv} on SM90: head dim > 256 not supported")

    # Only the big-headdim backward merges dK/dV (see KV_SHARED_D_DV); anywhere
    # else "shared" would just re-run the split path with a different allocation.
    kv_shared = kv_mode == "shared"
    if kv_shared and (d, dv) not in KV_SHARED_D_DV:
        pytest.skip(f"Skipping kv shared for d{d}-dv{dv}: merge not implemented")

    startend_row_indices, causal = gen_startend_row_indices(batch_size, seqlen_q, seqlen_k, nheads_startend_row_indices)

    # d192/dv192 has no causal kernel for seqlen_q != seqlen_k (the cutedsl gate
    # allows 192/192/128, not 192/192/192).
    if (d == 192 and dv == 192) and seqlen_q != seqlen_k and causal:
        pytest.skip("Skipping d192-dv192 causal when seqlen_q != seqlen_k")

    use_sink = not (d == 192 and dv == 192)

    q_ref = paddle.randn(shape=[batch_size, seqlen_q, nheads, d], dtype=dtype)
    k_ref = paddle.randn(shape=[batch_size, seqlen_k, nheads_kv, d], dtype=dtype)

    q_ref.stop_gradient = False
    k_ref.stop_gradient = False

    def make_qkv(q_src, k_src, v_src):
        """One (q, k, v) triplet. In shared mode v is a stride-1 view onto k.

        The view is taken AFTER stop_gradient is cleared, so k is the leaf and
        autograd folds dV back into k.grad the same way the kernel's merged
        accumulator does.
        """
        q = q_src.detach().clone()
        k = k_src.detach().clone()
        q.stop_gradient = False
        k.stop_gradient = False
        if kv_shared:
            v = k[..., :dv]
        else:
            v = v_src.detach().clone()
            v.stop_gradient = False
        return q, k, v

    if kv_shared:
        v_ref = k_ref[..., :dv]
        assert kv_shared_detected(k_ref, v_ref), (
            "v is not a stride-1 view onto k -- paddle materialised the slice, "
            "so the kernel's kv-shared path would not be exercised"
        )
    else:
        v_ref = paddle.randn(shape=[batch_size, seqlen_k, nheads_kv, dv], dtype=dtype)
        v_ref.stop_gradient = False

    q_bf16, k_bf16, v_bf16 = make_qkv(q_ref, k_ref, v_ref)
    q, k, v = make_qkv(q_ref, k_ref, v_ref)

    if kv_shared:
        assert kv_shared_detected(k, v)

    attn_bias = startend_row_indices_to_attn_bias(startend_row_indices, seqlen_q, nheads, dtype, causal)

    if use_sink:
        # sink_dtype comes from the shape parametrize (round-robined in generate_shapes),
        # so it is independent of the q/k/v dtype and costs no extra case.
        print(f"learnable_sink dtype: {sink_dtype}")
        sink_ref = paddle.randn(shape=[nheads], dtype=paddle.float32).astype(sink_dtype)
        sink_bf16 = sink_ref.detach().clone()
        sink = sink_ref.detach().clone()

        sink_bf16.stop_gradient = False
        sink_ref.stop_gradient = False
        sink.stop_gradient = False
    else:
        sink_ref = None
        sink_bf16 = None
        sink = None

    out_ref, attn_ref = attention_ref(
        q_ref,
        k_ref,
        v_ref,
        causal=causal,
        attn_bias=attn_bias,
        learnable_sink=sink_ref,
    )

    out_bf16, attn_bf16 = attention_ref(
        q_bf16,
        k_bf16,
        v_bf16,
        causal=causal,
        attn_bias=attn_bias,
        upcast=False,
        reorder_ops=True,
        learnable_sink=sink_bf16,
    )

    # # Numerical error if we just do any arithmetic on out_ref
    fwd_atol = 2 * (out_ref + 0.3 - 0.3 - out_ref).abs().max().item()
    assert softcap == 0.0
    rtol = 2 if softcap == 0.0 else 3

    print(f"Paddle naive bf16 Output max diff: {(out_bf16 - out_ref).abs().max().item()}")
    print(f"Paddle naive bf16 Output mean diff: {(out_bf16 - out_ref).abs().mean().item()}")

    out, lse = flashmask_attention(
        q,
        k,
        v,
        startend_row_indices=startend_row_indices,
        causal=causal,
        return_softmax_lse=True,
        learnable_sink=sink,
    )
    print(f"flashmask Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"flashmask Output mean diff: {(out - out_ref).abs().mean().item()}")
    # if not causal:
    #     print(f"LSE max diff: {(lse - lse_ref).abs().max().item()}")
    # breakpoint()

    # Check that FlashAttention's numerical error is at most twice the numerical error
    # of a Pytorch implementation.

    assert (out - out_ref).abs().max().item() <= rtol * (out_bf16 - out_ref).abs().max().item() + fwd_atol

    g = paddle.randn(shape=out.shape, dtype=out.dtype)
    out.backward(g)
    out_ref.backward(g)
    out_bf16.backward(g)

    print(f"flashmask dQ max diff: {(q.grad - q_ref.grad).abs().max().item()}")
    print(f"flashmask dK max diff: {(k.grad - k_ref.grad).abs().max().item()}")
    print(f"flashmask dQ mean diff: {(q.grad - q_ref.grad).abs().mean().item()}")
    print(f"flashmask dK mean diff: {(k.grad - k_ref.grad).abs().mean().item()}")

    print(f"Paddle naive bf16 dQ max diff: {(q_bf16.grad - q_ref.grad).abs().max().item()}")
    print(f"Paddle naive bf16 dK max diff: {(k_bf16.grad - k_ref.grad).abs().max().item()}")
    print(f"Paddle naive bf16 dQ mean diff: {(q_bf16.grad - q_ref.grad).abs().mean().item()}")
    print(f"Paddle naive bf16 dK mean diff: {(k_bf16.grad - k_ref.grad).abs().mean().item()}")

    if not kv_shared:
        print(f"flashmask dV max diff: {(v.grad - v_ref.grad).abs().max().item()}")
        print(f"flashmask dV mean diff: {(v.grad - v_ref.grad).abs().mean().item()}")
        print(f"Paddle naive bf16 dV max diff: {(v_bf16.grad - v_ref.grad).abs().max().item()}")
        print(f"Paddle naive bf16 dV mean diff: {(v_bf16.grad - v_ref.grad).abs().mean().item()}")

    dq_atol = 2 * (q_ref.grad + 0.3 - 0.3 - q_ref.grad).abs().max().item() + (0 if softcap == 0 else 3e-4)
    assert (q.grad - q_ref.grad).abs().max().item() <= rtol * (q_bf16.grad - q_ref.grad).abs().max().item() + dq_atol
    # In shared mode k is the only leaf, so k.grad already carries dK + dV: the
    # kernel merges them into one accumulator and returns an all-zero dv, and
    # autograd's slice_grad scatters that zero dv back into the same buffer. The
    # dK check below is therefore the dKV check; v.grad does not exist.
    dk_atol = 2 * (k_ref.grad + 0.3 - 0.3 - k_ref.grad).abs().max().item() + (0 if softcap == 0 else 3e-4)
    assert (k.grad - k_ref.grad).abs().max().item() <= rtol * (k_bf16.grad - k_ref.grad).abs().max().item() + dk_atol
    if not kv_shared:
        dv_atol = 2 * (v_ref.grad + 0.3 - 0.3 - v_ref.grad).abs().max().item() + (0 if softcap == 0 else 3e-4)
        assert (v.grad - v_ref.grad).abs().max().item() <= rtol * (v_bf16.grad - v_ref.grad).abs().max().item() + dv_atol

    if use_sink:
        assert sink.grad.dtype == sink_dtype, (
            f"dsink dtype {sink.grad.dtype} != sink dtype {sink_dtype}"
        )
        assert bool(paddle.isfinite(sink.grad.astype(paddle.float32)).all().item()), (
            f"dsink has inf/nan (sink dtype={sink_dtype}): {sink.grad}"
        )
        print(f"flashmask dSink max diff: {(sink.grad - sink_ref.grad).abs().max().item()}")
        print(f"flashmask dSink mean diff: {(sink.grad - sink_ref.grad).abs().mean().item()}")
        print(f"Paddle naive bf16 dSink max diff: {(sink_bf16.grad - sink_ref.grad).abs().max().item()}")
        print(f"Paddle naive bf16 dSink mean diff: {(sink_bf16.grad - sink_ref.grad).abs().mean().item()}")

        delta = (out_ref * g).sum(-1)                                      # (b, sq, h)
        delta = delta.transpose([0, 2, 1])                                 # (b, h, sq)
        p_sink = 1.0 - attn_ref.sum(-1)

        err_scale = (p_sink * delta.abs()).sum(axis=[0, 2]).astype(paddle.float32)   # (h,)

        # Compare in fp32: sink may be fp16/bf16/fp32, err_scale comes from bf16 attn_ref
        dsink_diff = (
            sink.grad.astype(paddle.float32) - sink_ref.grad.astype(paddle.float32)
        ).abs()
        dsink_tol = 1e-2 + rtol * 2**-8 * err_scale
        assert bool((dsink_diff <= dsink_tol).all().item()), (
            f"dsink mismatch (sink dtype={sink_dtype}): "
            f"max_diff={dsink_diff.max().item():.6f}, "
            f"max_tol={dsink_tol.max().item():.6f}, "
            f"err_scale_max={err_scale.max().item():.4f}"
        )


# ---------------------------------------------------------------------------
# SwapAB decode (SM100 / SM103): the paddle backend of the SwapAB path (allocation, strides,
# the split-KV combine) through the ``_flash_attn_fwd`` branch that seqlen_q = 1 calls of
# ``flashmask_attention`` take. tests/torch/test_flashmask.py tests the kernel in depth.
# ---------------------------------------------------------------------------


def _sm100_family():
    if not paddle.device.is_compiled_with_cuda():
        return False
    return tuple(paddle.device.cuda.get_device_capability()) in ((10, 0), (10, 3))


_needs_swapab = pytest.mark.skipif(not _sm100_family(), reason="SwapAB decode needs SM100/SM103")


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


def _attn(q, k, v, mask=None, causal=False, num_splits=0):
    """flashmask_attention, asserting it ran SwapAB; num_splits > 0 replaces the heuristic."""
    taken = []
    real = fwd_ops.swapab_decode_supported

    def supported(*args):
        taken.append(real(*args))
        return taken[-1]

    spy = mock.patch.object(fwd_ops, "swapab_decode_supported", supported)
    splits = (
        mock.patch.object(fwd_ops, "swapab_num_splits", return_value=num_splits)
        if num_splits
        else contextlib.nullcontext()
    )
    with spy, splits:
        out, lse = flashmask_attention(q, k, v, mask, causal=causal, return_softmax_lse=True)
    assert taken == [True], "flashmask_attention did not run the SwapAB kernel"
    return out, lse


@pytest.mark.parametrize("head_dim,head_dim_v", [(128, 128), (576, 512)])
@pytest.mark.parametrize("num_splits", [1, 3])
@_needs_swapab
def test_decode_swapab_dense(head_dim, head_dim_v, num_splits):
    paddle.seed(0)
    q = paddle.randn([2, 1, 16, head_dim]).astype("bfloat16")
    k = paddle.randn([2, 1000, 4, head_dim]).astype("bfloat16")
    v = paddle.randn([2, 1000, 4, head_dim_v]).astype("bfloat16")
    out, lse = _attn(q, k, v, num_splits=num_splits)
    out_ref, lse_ref = _reference(q, k, v, 1.0 / math.sqrt(head_dim))
    assert out.shape == [2, 1, 16, head_dim_v] and out.dtype == paddle.bfloat16
    assert _max_err(out, out_ref) < 2e-2
    assert _max_err(lse, lse_ref) < 1e-3


@pytest.mark.parametrize("num_vecs,causal", [(1, True), (2, False), (4, False)])
@_needs_swapab
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

    out, lse = _attn(q, k, v, mask, causal)
    # A q that requires grad keeps _flash_attn_fwd on the main kernel.
    q_main = q.detach()
    q_main.stop_gradient = False
    out_main, lse_main = _flash_attn_fwd(
        q_main, k, v, causal=causal, return_lse=True, startend_row_indices=mask, pack_gqa=False
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
