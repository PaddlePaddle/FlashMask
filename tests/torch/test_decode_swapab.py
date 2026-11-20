# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 The FlashMask Authors. All rights reserved.

"""SM100 SwapAB decode kernel test (torch): one query token, optional FlashMask column mask.

The tests call ``flashmask_attention`` and check that it ran the SwapAB kernel; split count
and occupancy are forced by patching their heuristics. Only the tuning arguments it cannot
pass (in-place masking, per-query-head masks, mask_row, out, block_list, empty KV) call
``_flashmask_decode_swapab`` directly.

Most tests use head_dim 128; the head_dim tests cover the other (head_dim, head_dim_v) pairs.

The output must stay within ``2 * |naive_lowp - ref| + atol`` of an fp32 reference, where the
naive low-precision run casts its softmax probabilities to the input dtype as the kernel does.
"""

import contextlib
import math
from unittest import mock

import pytest
import torch

import flashmask.cute.ops.decode as decode_ops
import flashmask.frontend.torch as fm
from flashmask.cute.flash_fwd_decode_sm100 import swapab_occupancy_fits, swapab_tmem_cols
from flashmask.cute.ops.decode import _flashmask_decode_swapab, swapab_decode_block_list


def _is_sm100_family():
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return (major, minor) in ((10, 0), (10, 3))


pytestmark = pytest.mark.skipif(not _is_sm100_family(), reason="SwapAB decode needs SM100/SM103")


def _reference(q, k, v, scale, upcast=True):
    """(out (b, 1, h, d), lse (b, h, 1)); upcast=False mimics a low-precision P."""
    qhpk = q.shape[2] // k.shape[2]
    k = k.repeat_interleave(qhpk, dim=2)
    v = v.repeat_interleave(qhpk, dim=2)
    qf, kf, vf = q.float(), k.float(), v.float()
    scores = torch.einsum("bqhd,bkhd->bhqk", qf, kf) * scale
    lse = torch.logsumexp(scores, dim=-1)  # (b, h, 1)
    probs = torch.softmax(scores, dim=-1)
    if not upcast:
        probs = probs.to(q.dtype).float()
    out = torch.einsum("bhqk,bkhd->bqhd", probs, vf)
    return out, lse


def _attn(q, k, v, mask=None, causal=False, softmax_scale=None, num_splits=0, occupancy=0,
          return_lse=True):
    """flashmask_attention, asserting it ran SwapAB. num_splits / occupancy > 0 replace the
    kernel's heuristics. Returns (out, lse); lse is None unless return_lse."""
    with contextlib.ExitStack() as stack:
        called = stack.enter_context(mock.patch.object(
            fm, "_flashmask_decode_swapab", wraps=fm._flashmask_decode_swapab
        ))
        if num_splits:
            stack.enter_context(
                mock.patch.object(decode_ops, "swapab_num_splits", return_value=num_splits)
            )
        if occupancy:
            stack.enter_context(
                mock.patch.object(decode_ops, "swapab_occupancy", return_value=occupancy)
            )
        res = fm.flashmask_attention(
            q, k, v, mask, causal=causal, softmax_scale=softmax_scale, return_lse=return_lse
        )
    torch.cuda.synchronize()
    assert called.call_count == 1, "flashmask_attention did not run the SwapAB kernel"
    return res if return_lse else (res, None)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "batch,seqlen_k,num_heads,num_heads_kv",
    [
        (1, 1, 8, 1),
        (1, 100, 8, 8),
        (2, 256, 16, 2),
        (3, 257, 32, 8),
        (4, 1000, 28, 4),
        (2, 4096, 32, 1),
        (1, 8192, 64, 8),
        (8, 2049, 8, 2),
        (16, 512, 40, 8),
    ],
)
@pytest.mark.parametrize("num_splits", [1, 0, 3])
@pytest.mark.parametrize("return_lse", [False, True])
def test_decode_swapab(dtype, batch, seqlen_k, num_heads, num_heads_kv, num_splits, return_lse):
    torch.manual_seed(0)
    head_dim = 128
    device = "cuda"
    q = torch.randn(batch, 1, num_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch, seqlen_k, num_heads_kv, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch, seqlen_k, num_heads_kv, head_dim, device=device, dtype=dtype)
    scale = 1.0 / math.sqrt(head_dim)

    out, lse = _attn(q, k, v, num_splits=num_splits, return_lse=return_lse)
    out_ref, lse_ref = _reference(q, k, v, scale)
    out_naive, _ = _reference(q, k, v, scale, upcast=False)

    assert out.shape == q.shape and out.dtype == dtype
    assert torch.isfinite(out.float()).all()
    err = (out.float() - out_ref).abs().max().item()
    naive_err = (out_naive.to(dtype).float() - out_ref).abs().max().item()
    assert err <= 2 * naive_err + 1e-3, f"max err {err} vs naive {naive_err}"
    if return_lse:
        assert lse.shape == (batch, num_heads, 1)
        torch.testing.assert_close(lse, lse_ref, atol=1e-3, rtol=1e-3)
    else:
        assert lse is None


def test_decode_swapab_strided_kv():
    """K/V as a slice of a larger cache: the kernel reads the real strides."""
    torch.manual_seed(0)
    dtype, head_dim = torch.bfloat16, 128
    batch, seqlen_k, cap, num_heads, num_heads_kv = 2, 700, 1024, 16, 4
    q = torch.randn(batch, 1, num_heads, head_dim, device="cuda", dtype=dtype)
    k_cache = torch.randn(batch, cap, num_heads_kv, head_dim, device="cuda", dtype=dtype)
    v_cache = torch.randn(batch, cap, num_heads_kv, head_dim, device="cuda", dtype=dtype)
    k, v = k_cache[:, :seqlen_k], v_cache[:, :seqlen_k]
    out, _ = _attn(q, k, v)
    out_ref, _ = _reference(q, k, v, 1.0 / math.sqrt(head_dim))
    out_naive, _ = _reference(q, k, v, 1.0 / math.sqrt(head_dim), upcast=False)
    err = (out.float() - out_ref).abs().max().item()
    naive_err = (out_naive.to(dtype).float() - out_ref).abs().max().item()
    assert err <= 2 * naive_err + 1e-3, f"max err {err} vs naive {naive_err}"


def _col_mask_keep(mask, causal, row=None):
    """(b, h_m, s_k) True where the decode query (mask row ``row``, default s_k - 1) may attend
    the key."""
    if row is None:
        row = mask.shape[2] - 1
    num_vecs = mask.shape[3]
    lts = mask[..., 0]
    masked = row >= lts
    if num_vecs == 2:
        bound = mask[..., 1]
        masked = (masked & (row < bound)) if causal else (masked | (row < bound))
    elif num_vecs == 4:
        lte, uts, ute = mask[..., 1], mask[..., 2], mask[..., 3]
        masked = (masked & (row < lte)) | ((row >= uts) & (row < ute))
    return ~masked


def _masked_reference(q, k, v, scale, keep, upcast=True):
    """keep: (b, h, s_k). Fully masked heads give O = 0 and LSE = -inf."""
    qhpk = q.shape[2] // k.shape[2]
    k = k.repeat_interleave(qhpk, dim=2)
    v = v.repeat_interleave(qhpk, dim=2)
    scores = torch.einsum("bqhd,bkhd->bhqk", q.float(), k.float()) * scale
    scores = scores.masked_fill(~keep[:, :, None, :], float("-inf"))
    lse = torch.logsumexp(scores, dim=-1)
    probs = torch.softmax(scores, dim=-1).nan_to_num(0.0)
    if not upcast:
        probs = probs.to(q.dtype).float()
    out = torch.einsum("bhqk,bkhd->bqhd", probs, v.float())
    return out, lse


def _random_mask(batch, mask_heads, seqlen_k, num_vecs, device, causal=True, high=2):
    """Random bounds in [0, high] (default 2, so row 0 sees part of the keys), each pair
    ordered. Non-causal num_vecs 2 masks [LTS, inf) and [0, UTE), so keys are visible only
    with UTE <= LTS: that pair is sorted descending."""
    m = torch.randint(0, high + 1, (batch, mask_heads, seqlen_k, num_vecs), device=device)
    if num_vecs >= 2:
        m[..., :2] = m[..., :2].sort(dim=-1, descending=num_vecs == 2 and not causal).values
    if num_vecs == 4:
        m[..., 2:] = m[..., 2:].sort(dim=-1).values
    return m.to(torch.int32).contiguous()


def _run_masked(
    q, k, v, mask, causal, num_splits=0, block_skip=True, mask_row=0, occupancy=0,
):
    """Check one masked decode call against the reference; the query is mask row mask_row.

    Goes through flashmask_attention (whose query is mask row 0) unless the call needs an
    argument it cannot pass: in-place masking, a mask head per query head under GQA, or
    another mask_row. Those call _flashmask_decode_swapab directly.
    """
    num_heads = q.shape[2]
    scale = 1.0 / math.sqrt(q.shape[-1])
    if block_skip and mask_row == 0 and mask.shape[1] in (1, k.shape[2]):
        out, lse = _attn(q, k, v, mask, causal, num_splits=num_splits, occupancy=occupancy)
    else:
        out, lse = _flashmask_decode_swapab(
            q, k, v, num_splits=num_splits, return_lse=True,
            startend_row_indices=mask, causal=causal, block_skip=block_skip, mask_row=mask_row,
            occupancy=occupancy,
        )
        torch.cuda.synchronize()
    keep = _col_mask_keep(mask, causal, mask_row)
    keep = keep.repeat_interleave(num_heads // mask.shape[1], dim=1)
    out_ref, lse_ref = _masked_reference(q, k, v, scale, keep)
    out_naive, _ = _masked_reference(q, k, v, scale, keep, upcast=False)
    assert torch.isfinite(out.float()).all()
    err = (out.float() - out_ref).abs().max().item()
    naive_err = (out_naive.to(q.dtype).float() - out_ref).abs().max().item()
    assert err <= 2 * naive_err + 1e-3, f"max err {err} vs naive {naive_err}"
    torch.testing.assert_close(lse, lse_ref, atol=1e-3, rtol=1e-3)
    return keep


# Layouts flashmask_attention accepts: num_vecs 1 / 2 with causal, 2 / 4 without. causal only
# changes the kernel for num_vecs 2, so (1, False) would add nothing.
_MASK_KINDS = [(1, True), (2, True), (2, False), (4, False)]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("num_vecs,causal", _MASK_KINDS)
@pytest.mark.parametrize("mask_heads", ["one", "kv", "all"])
@pytest.mark.parametrize(
    "batch,seqlen_k,num_heads,num_heads_kv",
    [
        (2, 300, 16, 2),
        (4, 129, 28, 4),
        (3, 1000, 8, 8),
        (1, 4096, 32, 8),
    ],
)
@pytest.mark.parametrize("num_splits", [1, 0, 3])
def test_decode_swapab_mask(
    dtype, num_vecs, causal, mask_heads, batch, seqlen_k, num_heads, num_heads_kv, num_splits
):
    torch.manual_seed(0)
    head_dim = 128
    h_m = {"one": 1, "kv": num_heads_kv, "all": num_heads}[mask_heads]
    q = torch.randn(batch, 1, num_heads, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(batch, seqlen_k, num_heads_kv, head_dim, device="cuda", dtype=dtype)
    v = torch.randn(batch, seqlen_k, num_heads_kv, head_dim, device="cuda", dtype=dtype)
    mask = _random_mask(batch, h_m, seqlen_k, num_vecs, "cuda", causal)
    _run_masked(q, k, v, mask, causal, num_splits)


@pytest.mark.parametrize("mask_heads", ["one", "all"])
@pytest.mark.parametrize("num_splits", [1, 4])
def test_decode_swapab_mask_fully_masked(mask_heads, num_splits):
    """Fully masked heads and fully masked splits: O = 0, LSE = -inf, no NaN."""
    torch.manual_seed(0)
    dtype, head_dim = torch.bfloat16, 128
    batch, seqlen_k, num_heads, num_heads_kv = 3, 2048, 16, 2
    h_m = 1 if mask_heads == "one" else num_heads
    q = torch.randn(batch, 1, num_heads, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(batch, seqlen_k, num_heads_kv, head_dim, device="cuda", dtype=dtype)
    v = torch.randn(batch, seqlen_k, num_heads_kv, head_dim, device="cuda", dtype=dtype)
    # num_vecs 1: a key is masked iff LTS <= 0 (the query is row 0).
    mask = torch.full((batch, h_m, seqlen_k, 1), seqlen_k, device="cuda", dtype=torch.int32)
    mask[0] = 0  # batch 0: every key masked
    mask[1, :, : seqlen_k * 3 // 4] = 0  # batch 1: the first three splits see only masked keys
    mask[2, 0] = 0  # batch 2: mask head 0 fully masked
    mask[2, :, 7] = 0
    keep = _run_masked(q, k, v, mask, True, num_splits)
    assert not keep[0].any() and keep[1].any()


def _block_sparse_mask(batch, mask_heads, seqlen_k, num_vecs, causal, device):
    """Mask with whole KV units hidden, visible or mixed, so block skipping matters.

    Hidden keys alternate between the two masking intervals where the layout has two.
    """
    s = seqlen_k
    nblk = -(-s // 128)
    state = torch.randint(0, 3, (batch, mask_heads, nblk), device=device)  # 0 hide 1 show 2 mix
    state = state.repeat_interleave(128, dim=-1)[..., :s]
    hidden = (state == 0) | ((state == 2) & (torch.rand(state.shape, device=device) < 0.5))
    odd = (torch.arange(s, device=device) % 2 == 1).expand_as(hidden)
    m = torch.full((batch, mask_heads, s, num_vecs), s, dtype=torch.int32, device=device)
    if num_vecs == 1:
        m[..., 0][hidden] = 0
    elif num_vecs == 2 and causal:
        m[..., 0][hidden] = 0  # [LTS, LTE) = [0, s)
    elif num_vecs == 2:
        m[..., 1] = 0
        m[..., 0][hidden & odd] = 0  # row >= LTS
        m[..., 1][hidden & ~odd] = s  # row < UTE
    else:
        m[..., 0][hidden & odd] = 0  # [LTS, LTE) = [0, s)
        m[..., 2][hidden & ~odd] = 0  # [UTS, UTE) = [0, s)
    return m.contiguous()


@pytest.mark.parametrize("num_vecs,causal", _MASK_KINDS)
@pytest.mark.parametrize("mask_heads", ["one", "kv"])
@pytest.mark.parametrize("batch,seqlen_k,num_heads_kv", [(3, 4096, 8), (2, 1000, 2), (1, 9000, 4)])
def test_decode_block_list(num_vecs, causal, mask_heads, batch, seqlen_k, num_heads_kv):
    """The block list holds exactly the KV units with a visible key."""
    torch.manual_seed(0)
    h_m = 1 if mask_heads == "one" else num_heads_kv
    mask = _block_sparse_mask(batch, h_m, seqlen_k, num_vecs, causal, "cuda")
    cnt, idx = swapab_decode_block_list(mask, num_heads_kv, causal)
    torch.cuda.synchronize()

    keep = _col_mask_keep(mask, causal)  # (b, h_m, s)
    nk = -(-seqlen_k // 128)
    pad = torch.zeros(batch, h_m, nk * 128, dtype=torch.bool, device="cuda")
    pad[..., :seqlen_k] = keep
    vis = pad.view(batch, h_m, nk, 128).any(-1)
    group = num_heads_kv // h_m
    for b in range(batch):
        for kvh in range(num_heads_kv):
            blocks = vis[b, kvh // group].nonzero().flatten()
            assert cnt[b, kvh, 0].item() == len(blocks)
            assert torch.equal(idx[b, kvh, 0, : len(blocks)], blocks.int())


@pytest.mark.parametrize("num_vecs,causal", _MASK_KINDS)
@pytest.mark.parametrize("num_splits", [1, 0, 5])
@pytest.mark.parametrize("mask_row", [None, 0])
def test_decode_swapab_prebuilt_block_list(num_vecs, causal, num_splits, mask_row):
    """A block list built once serves several calls (layers) that share the mask."""
    torch.manual_seed(0)
    batch, seqlen_k, num_heads, num_heads_kv, d = 3, 4096, 32, 8, 128
    mask = _block_sparse_mask(batch, 1, seqlen_k, num_vecs, causal, "cuda")
    block_list = swapab_decode_block_list(mask, num_heads_kv, causal, mask_row)
    for _layer in range(2):
        q = torch.randn(batch, 1, num_heads, d, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(batch, seqlen_k, num_heads_kv, d, device="cuda", dtype=torch.bfloat16)
        v = torch.randn_like(k)
        kw = dict(startend_row_indices=mask, causal=causal, num_splits=num_splits,
                  return_lse=True, mask_row=mask_row)
        out, lse = _flashmask_decode_swapab(q, k, v, block_list=block_list, **kw)
        out_ref, lse_ref = _flashmask_decode_swapab(q, k, v, **kw)
        assert torch.equal(out, out_ref)
        assert torch.equal(lse, lse_ref)


@pytest.mark.parametrize("num_vecs,causal", _MASK_KINDS)
@pytest.mark.parametrize("mask_heads", ["one", "kv", "all"])
@pytest.mark.parametrize(
    "batch,seqlen_k,num_heads,num_heads_kv", [(3, 4096, 32, 8), (2, 1000, 16, 2)]
)
@pytest.mark.parametrize("num_splits", [1, 0, 5])
@pytest.mark.parametrize("block_skip", [True, False])
def test_decode_swapab_mask_block_sparse(
    num_vecs, causal, mask_heads, batch, seqlen_k, num_heads, num_heads_kv, num_splits,
    block_skip,
):
    torch.manual_seed(0)
    dtype, head_dim = torch.bfloat16, 128
    h_m = {"one": 1, "kv": num_heads_kv, "all": num_heads}[mask_heads]
    q = torch.randn(batch, 1, num_heads, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(batch, seqlen_k, num_heads_kv, head_dim, device="cuda", dtype=dtype)
    v = torch.randn(batch, seqlen_k, num_heads_kv, head_dim, device="cuda", dtype=dtype)
    mask = _block_sparse_mask(batch, h_m, seqlen_k, num_vecs, causal, "cuda")
    _run_masked(q, k, v, mask, causal, num_splits, block_skip=block_skip)


_HEAD_DIMS = [(192, 128), (256, 256), (512, 512), (576, 512)]


@pytest.mark.parametrize("head_dim,head_dim_v", _HEAD_DIMS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "batch,seqlen_k,num_heads,num_heads_kv",
    [(1, 1, 8, 1), (3, 257, 32, 8), (2, 4096, 16, 16), (16, 512, 8, 2)],
)
@pytest.mark.parametrize("num_splits", [1, 0, 3])
def test_decode_swapab_head_dim(
    head_dim, head_dim_v, dtype, batch, seqlen_k, num_heads, num_heads_kv, num_splits
):
    torch.manual_seed(0)
    q = torch.randn(batch, 1, num_heads, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(batch, seqlen_k, num_heads_kv, head_dim, device="cuda", dtype=dtype)
    v = torch.randn(batch, seqlen_k, num_heads_kv, head_dim_v, device="cuda", dtype=dtype)
    scale = 1.0 / math.sqrt(head_dim)
    out, lse = _attn(q, k, v, num_splits=num_splits)
    out_ref, lse_ref = _reference(q, k, v, scale)
    out_naive, _ = _reference(q, k, v, scale, upcast=False)
    assert out.shape == (batch, 1, num_heads, head_dim_v) and out.dtype == dtype
    err = (out.float() - out_ref).abs().max().item()
    naive_err = (out_naive.to(dtype).float() - out_ref).abs().max().item()
    assert err <= 2 * naive_err + 1e-3, f"max err {err} vs naive {naive_err}"
    torch.testing.assert_close(lse, lse_ref, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("head_dim,head_dim_v", _HEAD_DIMS)
@pytest.mark.parametrize("num_vecs,causal", _MASK_KINDS)
@pytest.mark.parametrize("mask_heads", ["one", "kv", "all"])
@pytest.mark.parametrize("num_splits", [1, 0, 5])
@pytest.mark.parametrize("block_skip", [True, False])
def test_decode_swapab_head_dim_mask(
    head_dim, head_dim_v, num_vecs, causal, mask_heads, num_splits, block_skip
):
    torch.manual_seed(0)
    dtype = torch.bfloat16
    batch, seqlen_k, num_heads, num_heads_kv = 3, 3000, 16, 4
    h_m = {"one": 1, "kv": num_heads_kv, "all": num_heads}[mask_heads]
    q = torch.randn(batch, 1, num_heads, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(batch, seqlen_k, num_heads_kv, head_dim, device="cuda", dtype=dtype)
    v = torch.randn(batch, seqlen_k, num_heads_kv, head_dim_v, device="cuda", dtype=dtype)
    mask = _block_sparse_mask(batch, h_m, seqlen_k, num_vecs, causal, "cuda")
    _run_masked(q, k, v, mask, causal, num_splits, block_skip=block_skip)


@pytest.mark.parametrize("head_dim,head_dim_v", _HEAD_DIMS + [(128, 128)])
@pytest.mark.parametrize(
    "batch,seqlen_k,num_heads,num_heads_kv",
    [(1, 1, 8, 1), (3, 257, 32, 8), (2, 4096, 16, 16), (16, 512, 8, 2), (4, 1000, 64, 2)],
)
@pytest.mark.parametrize("num_splits", [1, 0, 3])
@pytest.mark.parametrize("occupancy", [1, 2])
def test_decode_swapab_occupancy(
    head_dim, head_dim_v, batch, seqlen_k, num_heads, num_heads_kv, num_splits, occupancy
):
    """Forced occupancy: two CTAs per SM (smaller TMEM allocation, own SMEM share) match the
    reference, and match one CTA per SM bit for bit."""
    qhead_per_kvhead = num_heads // num_heads_kv
    if swapab_tmem_cols(qhead_per_kvhead, head_dim_v) > 512 or not swapab_occupancy_fits(
        qhead_per_kvhead, head_dim_v, occupancy
    ):
        pytest.skip("does not fit TMEM")
    torch.manual_seed(0)
    dtype = torch.bfloat16
    q = torch.randn(batch, 1, num_heads, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(batch, seqlen_k, num_heads_kv, head_dim, device="cuda", dtype=dtype)
    v = torch.randn(batch, seqlen_k, num_heads_kv, head_dim_v, device="cuda", dtype=dtype)
    scale = 1.0 / math.sqrt(head_dim)
    kw = dict(num_splits=num_splits)
    out, lse = _attn(q, k, v, occupancy=occupancy, **kw)
    _assert_within_naive(out, q, k, v, scale)
    torch.testing.assert_close(lse, _reference(q, k, v, scale)[1], atol=1e-3, rtol=1e-3)
    out1, lse1 = _attn(q, k, v, occupancy=1, **kw)
    assert torch.equal(out, out1) and torch.equal(lse, lse1)


@pytest.mark.parametrize("head_dim,head_dim_v", [(128, 128), (256, 256)])
@pytest.mark.parametrize("num_vecs,causal", _MASK_KINDS)
@pytest.mark.parametrize("mask_heads", ["one", "kv", "all"])
@pytest.mark.parametrize("num_splits", [1, 0, 5])
@pytest.mark.parametrize("block_skip", [True, False])
def test_decode_swapab_occupancy_mask(
    head_dim, head_dim_v, num_vecs, causal, mask_heads, num_splits, block_skip
):
    torch.manual_seed(0)
    dtype = torch.bfloat16
    batch, seqlen_k, num_heads, num_heads_kv = 3, 3000, 16, 4
    h_m = {"one": 1, "kv": num_heads_kv, "all": num_heads}[mask_heads]
    q = torch.randn(batch, 1, num_heads, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(batch, seqlen_k, num_heads_kv, head_dim, device="cuda", dtype=dtype)
    v = torch.randn(batch, seqlen_k, num_heads_kv, head_dim_v, device="cuda", dtype=dtype)
    mask = _block_sparse_mask(batch, h_m, seqlen_k, num_vecs, causal, "cuda")
    _run_masked(q, k, v, mask, causal, num_splits, block_skip=block_skip, occupancy=2)


def test_decode_swapab_occupancy_too_big():
    """16 q heads per KV head at head_dim_v 512 need 288 TMEM columns: two CTAs per SM are
    refused."""
    q = torch.randn(1, 1, 16, 512, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 256, 1, 512, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(AssertionError, match="do not fit TMEM"):
        _attn(q, k, k, occupancy=2)


@pytest.mark.parametrize(
    "batch,seqlen_k,num_heads,num_heads_kv", [(2, 4096, 8, 8), (3, 3000, 8, 1), (4, 1000, 16, 1)]
)
@pytest.mark.parametrize("num_splits", [1, 0])
@pytest.mark.parametrize("masked", [False, True])
def test_decode_swapab_mla_shared_kv(batch, seqlen_k, num_heads, num_heads_kv, num_splits, masked):
    """MLA layout: one 576-dim latent cache, V is its first 512 dims (a strided view of K)."""
    torch.manual_seed(0)
    dtype = torch.bfloat16
    q = torch.randn(batch, 1, num_heads, 576, device="cuda", dtype=dtype)
    k = torch.randn(batch, seqlen_k, num_heads_kv, 576, device="cuda", dtype=dtype)
    v = k[..., :512]
    if masked:
        mask = _block_sparse_mask(batch, num_heads_kv, seqlen_k, 1, True, "cuda")
        _run_masked(q, k, v, mask, True, num_splits)
        return
    scale = 1.0 / math.sqrt(576)
    out, lse = _attn(q, k, v, num_splits=num_splits)
    out_ref, lse_ref = _reference(q, k, v, scale)
    out_naive, _ = _reference(q, k, v, scale, upcast=False)
    err = (out.float() - out_ref).abs().max().item()
    naive_err = (out_naive.to(dtype).float() - out_ref).abs().max().item()
    assert err <= 2 * naive_err + 1e-3, f"max err {err} vs naive {naive_err}"
    torch.testing.assert_close(lse, lse_ref, atol=1e-3, rtol=1e-3)


def _assert_within_naive(out, q, k, v, scale):
    out_ref, _ = _reference(q, k, v, scale)
    out_naive, _ = _reference(q, k, v, scale, upcast=False)
    err = (out.float() - out_ref).abs().max().item()
    naive_err = (out_naive.to(q.dtype).float() - out_ref).abs().max().item()
    assert err <= 2 * naive_err + 1e-3, f"max err {err} vs naive {naive_err}"


@pytest.mark.parametrize("num_splits", [1, 3])
@pytest.mark.parametrize("strided", [False, True])
def test_decode_swapab_out_buffer(num_splits, strided):
    """out= is written in place, also as a strided view of a wider buffer."""
    torch.manual_seed(0)
    dtype = torch.bfloat16
    q = torch.randn(2, 1, 16, 128, device="cuda", dtype=dtype)
    k = torch.randn(2, 1000, 2, 128, device="cuda", dtype=dtype)
    v = torch.randn(2, 1000, 2, 128, device="cuda", dtype=dtype)
    if strided:
        big = torch.zeros(2, 1, 16, 256, device="cuda", dtype=dtype)
        out = big[..., :128]
    else:
        out = torch.empty(2, 1, 16, 128, device="cuda", dtype=dtype)
    o, _ = _flashmask_decode_swapab(q, k, v, out=out, num_splits=num_splits)
    torch.cuda.synchronize()
    assert o.data_ptr() == out.data_ptr()
    _assert_within_naive(out, q, k, v, 1.0 / math.sqrt(128))
    if strided:
        assert (big[..., 128:] == 0).all()


@pytest.mark.parametrize("num_splits", [1, 3])
def test_decode_swapab_softmax_scale(num_splits):
    torch.manual_seed(0)
    dtype = torch.bfloat16
    q = torch.randn(2, 1, 16, 128, device="cuda", dtype=dtype)
    k = torch.randn(2, 1000, 2, 128, device="cuda", dtype=dtype)
    v = torch.randn(2, 1000, 2, 128, device="cuda", dtype=dtype)
    out, lse = _attn(q, k, v, softmax_scale=0.3, num_splits=num_splits)
    _assert_within_naive(out, q, k, v, 0.3)
    torch.testing.assert_close(lse, _reference(q, k, v, 0.3)[1], atol=1e-3, rtol=1e-3)


def test_decode_swapab_empty_kv():
    """seqlen_k = 0: O = 0 and LSE = -inf."""
    dtype = torch.bfloat16
    q = torch.randn(2, 1, 16, 128, device="cuda", dtype=dtype)
    k = torch.randn(2, 0, 2, 128, device="cuda", dtype=dtype)
    v = torch.randn(2, 0, 2, 128, device="cuda", dtype=dtype)
    out, lse = _flashmask_decode_swapab(q, k, v, return_lse=True)
    torch.cuda.synchronize()
    assert (out == 0).all()
    assert (lse == float("-inf")).all()


@pytest.mark.parametrize("num_vecs,causal", _MASK_KINDS)
@pytest.mark.parametrize("mask_row", [0, 517])
@pytest.mark.parametrize("block_skip", [True, False])
def test_decode_swapab_mask_row(num_vecs, causal, mask_row, block_skip):
    """An explicit mask_row moves the query's row in the column mask."""
    torch.manual_seed(0)
    dtype = torch.bfloat16
    batch, seqlen_k, num_heads, num_heads_kv = 2, 1000, 16, 2
    q = torch.randn(batch, 1, num_heads, 128, device="cuda", dtype=dtype)
    k = torch.randn(batch, seqlen_k, num_heads_kv, 128, device="cuda", dtype=dtype)
    v = torch.randn(batch, seqlen_k, num_heads_kv, 128, device="cuda", dtype=dtype)
    # Bounds in [0, 2 * mask_row + 2] so a good share of the keys is hidden from mask_row.
    mask = _random_mask(batch, num_heads_kv, seqlen_k, num_vecs, "cuda", causal, 2 * mask_row + 2)
    _run_masked(q, k, v, mask, causal, 0, block_skip=block_skip, mask_row=mask_row)


# _MASK_KINDS plus no mask.
_FRONTEND_MASK_KINDS = [(None, False), (None, True)] + _MASK_KINDS


@pytest.mark.parametrize("head_dim,head_dim_v", [(128, 128), (256, 256), (576, 512)])
@pytest.mark.parametrize("num_vecs,causal", _FRONTEND_MASK_KINDS)
@pytest.mark.parametrize("mask_heads", ["one", "kv"])
def test_flashmask_attention_decode_dispatch(head_dim, head_dim_v, num_vecs, causal, mask_heads):
    """seqlen_q = 1 without autograd runs SwapAB with the main kernel's mask semantics."""
    from flashmask.cute.ops.fwd import _flash_attn_fwd

    if num_vecs is None and mask_heads == "kv":
        pytest.skip("no mask")
    torch.manual_seed(0)
    dtype = torch.bfloat16
    batch, seqlen_k, num_heads, num_heads_kv = 3, 3000, 16, 4
    q = torch.randn(batch, 1, num_heads, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(batch, seqlen_k, num_heads_kv, head_dim, device="cuda", dtype=dtype)
    v = torch.randn(batch, seqlen_k, num_heads_kv, head_dim_v, device="cuda", dtype=dtype)
    mask = None
    if num_vecs is not None:
        h_m = 1 if mask_heads == "one" else num_heads_kv
        # Bounds in {0, 1, 2}: row 0 and row seqlen_k - 1 see different keys.
        mask = _random_mask(batch, h_m, seqlen_k, num_vecs, "cuda", causal, 2)
        mask[0, 0, :, 0] = 0  # sequence 0, mask head 0: every key hidden from row 0
        if num_vecs >= 2:
            mask[0, 0, :, 1] = 1

    with torch.no_grad():
        out, lse = _attn(q, k, v, mask, causal)
    out_main, lse_main = _flash_attn_fwd(
        q, k, v, causal=causal, return_lse=True, startend_row_indices=mask, pack_gqa=False
    )
    torch.cuda.synchronize()
    assert out.shape == out_main.shape and lse.shape == lse_main.shape

    scale = 1.0 / math.sqrt(head_dim)
    if mask is None:
        keep = torch.ones(batch, num_heads, seqlen_k, dtype=torch.bool, device="cuda")
    else:
        keep = _col_mask_keep(mask, causal, row=0)
        keep = keep.repeat_interleave(num_heads // mask.shape[1], dim=1)
    out_ref, lse_ref = _masked_reference(q, k, v, scale, keep)
    out_naive, _ = _masked_reference(q, k, v, scale, keep, upcast=False)
    naive_err = (out_naive.to(dtype).float() - out_ref).abs().max().item()
    for o in (out, out_main):
        err = (o.float() - out_ref).abs().max().item()
        assert err <= 2 * naive_err + 1e-3, f"max err {err} vs naive {naive_err}"
    torch.testing.assert_close(lse, lse_ref, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(lse, lse_main, atol=1e-3, rtol=1e-3)


def test_flashmask_attention_decode_dispatch_skipped_with_grad():
    """A call that needs gradients keeps the main kernel."""
    torch.manual_seed(0)
    q = torch.randn(2, 1, 8, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(2, 512, 8, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(2, 512, 8, 128, device="cuda", dtype=torch.bfloat16)
    with mock.patch.object(fm, "_flashmask_decode_swapab") as called:
        out = fm.flashmask_attention(q, k, v)
    assert called.call_count == 0
    out.float().sum().backward()
    assert q.grad is not None
