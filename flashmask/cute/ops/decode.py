# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025, Tri Dao.
# Copyright (c) 2026 The FlashMask Authors. Modifications for FlashMask.

"""SM100 SwapAB single-token decode orchestration (standalone, (head_dim, head_dim_v) in
SWAPAB_HEAD_DIMS, optional FlashMask column mask).

Framework-agnostic: allocation, dtype and stream access go through a
`flashmask.backend` adapter (``be``).
"""

import math
from typing import Optional, Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from ...backend import Tensor, get_backend
from ..flash_fwd_decode_sm100 import (
    SWAPAB_HEAD_DIMS,
    SWAPAB_MAX_QHEAD_PER_KVHEAD,
    SWAPAB_OCCUPANCIES,
    SWAPAB_SKIP_UNIT,
    SWAPAB_TILE_N,
    FlashAttentionDecodeSm100SwapAB,
    build_decode_block_list,
    swapab_occupancy_fits,
    swapab_tile_hq,
    swapab_tmem_cols,
)
from .combine import _flash_attn_fwd_combine

__all__ = [
    "_flash_attn_decode_swapab",
    "swapab_decode_block_list",
    "swapab_decode_supported",
    "swapab_num_splits",
    "swapab_occupancy",
]

CUDA_MAX_GRID_DIM_YZ = 65535

# Static split policy, as upstream SwapABPolicy (fitted on B200 / B300 decode sweeps): decode is
# bandwidth bound and DRAM saturates with about this many CTAs. B300 bf16 with head_dim_v 512
# needs the larger target at any batch: unmasked, batch 1-2, 6.5 -> 6.8 TB/s (576/512, 8 KV
# heads, 131072 tokens: 351 -> 336 us); head_dim_v 128 / 256 gain nothing from more splits.
_TARGET_CTAS = 64
_SM103_BF16_TARGET_CTAS = 128
_SM103_BF16_MIN_BATCH = 4
_SM103_BF16_MIN_HEADS = 16
_SM103_BF16_WIDE_HEAD_DIM_V = 512
_MAX_SPLITS = 32
_MIN_TOKENS_PER_SPLIT = 512


def swapab_num_splits(
    batch_size: int,
    num_heads_kv: int,
    seqlen_k: int,
    is_sm103: bool,
    is_bf16: bool,
    head_dim_v: int,
) -> int:
    """Upstream's static split count: a power of two near target CTAs / (batch * KV heads)."""
    bh = batch_size * num_heads_kv
    if bh == 0:
        return 1
    target = _TARGET_CTAS
    if is_sm103 and is_bf16 and (
        (batch_size >= _SM103_BF16_MIN_BATCH and bh >= _SM103_BF16_MIN_HEADS)
        or head_dim_v >= _SM103_BF16_WIDE_HEAD_DIM_V
    ):
        target = _SM103_BF16_TARGET_CTAS
    pow2_splits = 1 << (max(1, target // bh).bit_length() - 1)
    return min(pow2_splits, _MAX_SPLITS, max(1, seqlen_k // _MIN_TOKENS_PER_SPLIT))


# Two CTAs per SM overlap one CTA's prologue (Q load, first QK) and epilogue (O store) with the
# other's KV stream, but each gets half the SMEM, so a shallower K/V ring. B300 sweep (clean L2,
# no mask, one split, head_dim 128 / 192 / 256): with at least 1.5 CTAs per SM and at most 1024
# tokens per CTA, two per SM is 3-20% faster (222 CTAs x 512 tokens 20.3 -> 16.4 us); with 2048+
# tokens per CTA it is 0-3% slower; below one CTA per SM it is up to 80% slower (one CTA, 16K
# tokens 53 -> 96 us). Masked cases skip invisible blocks, so their per-CTA work is usually far
# below seqlen_k / splits; the D=128 suite masks run 5-25% faster with two per SM.
_OCC2_MIN_CTAS_PER_SM = 1.5
_OCC2_MAX_TOKENS_PER_CTA = 1024
_OCC2_MAX_HEAD_DIM_V = 256


def swapab_occupancy(
    num_ctas: int,
    num_sms: int,
    tokens_per_cta: int,
    block_skip: bool,
    qhead_per_kvhead: int,
    head_dim_v: int,
) -> int:
    """CTAs per SM: 2 for many CTAs with little KV each (or block skipping), else 1."""
    if (
        num_ctas >= _OCC2_MIN_CTAS_PER_SM * num_sms
        and (tokens_per_cta <= _OCC2_MAX_TOKENS_PER_CTA or block_skip)
        and head_dim_v <= _OCC2_MAX_HEAD_DIM_V
        and swapab_occupancy_fits(qhead_per_kvhead, head_dim_v, 2)
    ):
        return 2
    return 1


def _to_cute(t: Tensor, align: int):
    return from_dlpack(t.detach(), assumed_align=align).mark_layout_dynamic(
        leading_dim=t.ndim - 1
    )


def _is_contiguous(t: Tensor, be) -> bool:
    """Row-major strides, ignoring modes of extent 1."""
    expect = 1
    for st, n in reversed(list(zip(be.strides(t), t.shape))):
        if n > 1 and st != expect:
            return False
        expect *= n
    return True


def swapab_decode_supported(q, k, v, startend_row_indices, causal, be) -> bool:
    """Whether ``_flash_attn_decode_swapab`` accepts this forward call (no autograd checks).

    Used by the frontends to send single-token decode calls to the SwapAB kernel. Besides the
    kernel's own asserts this requires the TMA alignment (16-byte base addresses and strides)
    that the wrapper would otherwise only hit at launch.
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        return False
    batch_size, seqlen_q, num_heads, head_dim = q.shape
    _, seqlen_k, num_heads_kv, _ = k.shape
    head_dim_v = v.shape[-1]
    if seqlen_q != 1 or seqlen_k < 1 or (head_dim, head_dim_v) not in SWAPAB_HEAD_DIMS:
        return False
    if q.dtype not in (be.float16, be.bfloat16) or not (q.dtype == k.dtype == v.dtype):
        return False
    major, minor = be.device_capability()
    if major * 10 + minor not in (100, 103):
        return False
    if num_heads_kv < 1 or num_heads % num_heads_kv != 0:
        return False
    qhead_per_kvhead = num_heads // num_heads_kv
    if qhead_per_kvhead > SWAPAB_MAX_QHEAD_PER_KVHEAD:
        return False
    if swapab_tmem_cols(qhead_per_kvhead, head_dim_v) > 512:
        return False
    if batch_size > CUDA_MAX_GRID_DIM_YZ or num_heads_kv * _MAX_SPLITS > CUDA_MAX_GRID_DIM_YZ:
        return False
    for t in (q, k, v):
        strides = be.strides(t)
        if strides[-1] != 1 or be.data_ptr(t) % 16 != 0:
            return False
        if any(st % 8 != 0 for st, n in zip(strides[:-1], t.shape[:-1]) if n > 1):
            return False
    if startend_row_indices is not None:
        mask = startend_row_indices
        if mask.ndim != 4 or mask.dtype != be.int32 or not _is_contiguous(mask, be):
            return False
        mask_batch, mask_heads, mask_seqlen, num_vecs = mask.shape
        if mask_batch != batch_size or mask_seqlen != seqlen_k or num_vecs not in (1, 2, 4):
            return False
        if (causal and num_vecs == 4) or mask_heads < 1 or num_heads % mask_heads != 0:
            return False
    return True


def _decode_block_list(
    startend_row_indices, mask_tensor, num_heads_kv, is_causal, stream, be, mask_row=None
):
    """(cnt (B, H_kv, 1), idx (B, H_kv, 1, nk)) int32 for the block-skipping SwapAB kernel.

    The list holds 128-token KV units visible from query row ``mask_row`` (default
    seqlen_k - 1). Entries past cnt are left uninitialized; the kernel never reads them.
    """
    batch_size, mask_heads, seqlen_k, num_vecs = startend_row_indices.shape
    if mask_row is None:
        mask_row = seqlen_k - 1
    nk = -(-seqlen_k // SWAPAB_SKIP_UNIT)
    flags = be.empty([batch_size, mask_heads, nk], dtype=be.int32)
    cnt = be.empty([batch_size, num_heads_kv, 1], dtype=be.int32)
    idx = be.empty([batch_size, num_heads_kv, 1, nk], dtype=be.int32)
    tensors = [_to_cute(t, 4) for t in (flags, cnt, idx)]
    cache = _decode_block_list.compile_cache
    key = (num_vecs, is_causal)
    if key not in cache:
        cache[key] = cute.compile(
            build_decode_block_list, mask_tensor, *tensors, cutlass.Int32(mask_row), num_vecs,
            is_causal, stream,
        )
    cache[key](mask_tensor, *tensors, cutlass.Int32(mask_row), stream)
    return cnt, idx


_decode_block_list.compile_cache = {}


def _check_mask(mask, causal, be):
    assert mask.ndim == 4, "startend_row_indices must be (batch, H_m, seqlen_k, num_vecs)"
    assert mask.dtype == be.int32, "startend_row_indices must be int32"
    assert be.is_gpu(mask), "startend_row_indices must be on CUDA device"
    assert mask.shape[3] in (1, 2, 4), "num_vecs must be 1, 2 or 4"
    assert not (causal and mask.shape[3] == 4), "num_vecs 4 needs causal=False"
    assert _is_contiguous(mask, be), "startend_row_indices must be contiguous"


def swapab_decode_block_list(
    startend_row_indices: Tensor,
    num_heads_kv: int,
    causal: bool = False,
    mask_row: Optional[int] = None,
) -> Tuple[Tensor, Tensor]:
    """Build the SwapAB block list once, to pass as ``block_list`` to several decode calls.

    The list depends only on the mask, ``num_heads_kv``, ``causal`` and ``mask_row``, so the
    layers of one decode step can share it. Each call that takes it must use the same mask and
    the same values of those arguments; the kernel does not check. ``num_heads_kv`` must be a
    multiple of the mask head count (otherwise the decode call masks in place and takes no list).

    Returns (cnt (batch, num_heads_kv, 1), idx (batch, num_heads_kv, 1, ceil(seqlen_k / 128)))
    int32.
    """
    be = get_backend(startend_row_indices)
    mask = startend_row_indices
    _check_mask(mask, causal, be)
    assert num_heads_kv >= 1 and num_heads_kv % mask.shape[1] == 0, (
        "num_heads_kv must be a multiple of the mask head count"
    )
    mask_tensor = from_dlpack(mask.detach(), assumed_align=4).mark_layout_dynamic(leading_dim=3)
    return _decode_block_list(
        mask, mask_tensor, num_heads_kv, bool(causal) and mask.shape[3] == 2,
        cuda.CUstream(be.cuda_stream()), be, mask_row,
    )


def _flash_attn_decode_swapab(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    softmax_scale: Optional[float] = None,
    num_splits: int = 0,
    return_lse: bool = False,
    out: Optional[Tensor] = None,
    startend_row_indices: Optional[Tensor] = None,
    causal: bool = False,
    block_skip: bool = True,
    mask_row: Optional[int] = None,
    block_list: Optional[Tuple[Tensor, Tensor]] = None,
    occupancy: int = 0,
) -> Tuple[Tensor, Optional[Tensor]]:
    """Single-token decode attention with the SM100 SwapAB kernel.

    Args:
        q: (batch, 1, num_heads, head_dim), fp16 / bf16, last dim contiguous.
        k: (batch, seqlen_k, num_heads_kv, head_dim), v: (..., head_dim_v), same dtype, last
            dim contiguous. (head_dim, head_dim_v) must be in SWAPAB_HEAD_DIMS.
        softmax_scale: defaults to 1 / sqrt(head_dim).
        num_splits: KV splits; 0 picks upstream's static heuristic.
        return_lse: also return the (batch, num_heads, 1) fp32 natural-log LSE.
        out: optional (batch, 1, num_heads, head_dim_v) output buffer.
        startend_row_indices: optional FlashMask column mask (batch, H_m, seqlen_k, num_vecs)
            int32, contiguous, num_heads % H_m == 0. The query is mask row ``mask_row``.
            Heads with every key masked get O = 0, LSE = -inf.
        causal: selects the num_vecs = 2 layout ([LTS, LTE) if causal, else LTS and UTE);
            bottom-right causal itself masks nothing for the last row.
        block_skip: with a mask whose heads are shared by the query heads of each KV head
            (num_heads_kv % H_m == 0), first build the list of KV blocks that are not fully
            masked and split only those. Otherwise every block is loaded and masked in place.
        mask_row: the row the query is compared as against the mask bounds. Defaults to
            seqlen_k - 1 (bottom-right aligned, the query is the newest token); 0 matches
            ``_flash_attn_fwd`` with seqlen_q = 1, whose mask rows count from the first query.
        block_list: (cnt, idx) from ``swapab_decode_block_list`` for this mask, num_heads_kv,
            causal and mask_row, to skip building it here (block_skip must be on). Reuse it
            across the layers that share the mask.
        occupancy: CTAs resident per SM, 1 or 2; 0 picks ``swapab_occupancy``.

    Returns:
        (out, lse), lse is None unless return_lse.
    """
    be = get_backend(q, k, v)
    assert q.ndim == 4 and k.ndim == 4 and v.ndim == 4, "q, k, v must be 4-D (b, s, h, d)"
    batch_size, seqlen_q, num_heads, head_dim = q.shape
    _, seqlen_k, num_heads_kv, _ = k.shape
    head_dim_v = v.shape[-1]
    assert seqlen_q == 1, "SwapAB decode needs exactly one query token per sequence"
    assert (head_dim, head_dim_v) in SWAPAB_HEAD_DIMS, (
        f"SwapAB decode supports (head_dim, head_dim_v) in {SWAPAB_HEAD_DIMS}"
    )
    assert tuple(k.shape) == (batch_size, seqlen_k, num_heads_kv, head_dim)
    assert tuple(v.shape) == (batch_size, seqlen_k, num_heads_kv, head_dim_v)
    assert q.dtype in (be.float16, be.bfloat16), "q must be fp16 or bf16"
    assert q.dtype == k.dtype == v.dtype, "q, k, v must share a dtype"
    assert all(be.is_gpu(t) for t in (q, k, v)), "tensors must be on CUDA device"
    for t, name in ((q, "q"), (k, "k"), (v, "v")):
        assert be.strides(t)[-1] == 1, f"{name} must be contiguous in the last dimension"
    assert num_heads % num_heads_kv == 0, "num_heads must be a multiple of num_heads_kv"
    qhead_per_kvhead = num_heads // num_heads_kv
    assert qhead_per_kvhead <= SWAPAB_MAX_QHEAD_PER_KVHEAD, (
        f"SwapAB decode supports at most {SWAPAB_MAX_QHEAD_PER_KVHEAD} q heads per KV head"
    )
    assert swapab_tmem_cols(qhead_per_kvhead, head_dim_v) <= 512, (
        f"{qhead_per_kvhead} q heads per KV head do not fit TMEM at head_dim_v {head_dim_v}"
    )
    assert batch_size <= CUDA_MAX_GRID_DIM_YZ
    major, minor = be.device_capability()
    arch = major * 10 + minor
    assert arch in (100, 103), "SwapAB decode needs SM100 or SM103"

    mask_num_vecs, mask_per_head = 0, False
    if startend_row_indices is not None:
        mask = startend_row_indices
        _check_mask(mask, causal, be)
        mask_batch, mask_heads, mask_seqlen, mask_num_vecs = mask.shape
        assert mask_batch == batch_size and mask_seqlen == seqlen_k, (
            "startend_row_indices must be (batch, H_m, seqlen_k, num_vecs)"
        )
        assert mask_heads >= 1 and num_heads % mask_heads == 0, (
            "num_heads must be a multiple of the mask head count"
        )
        # The query heads of one KV head share a mask head iff H_m divides num_heads_kv.
        mask_per_head = num_heads_kv % mask_heads != 0
    if block_list is not None:
        assert startend_row_indices is not None and block_skip and not mask_per_head, (
            "block_list needs a mask shared by the query heads of each KV head and block_skip"
        )
        nk = -(-seqlen_k // SWAPAB_SKIP_UNIT)
        blk_cnt, blk_idx = block_list
        assert tuple(blk_cnt.shape) == (batch_size, num_heads_kv, 1), "block_list cnt shape"
        assert tuple(blk_idx.shape) == (batch_size, num_heads_kv, 1, nk), "block_list idx shape"
        for t in (blk_cnt, blk_idx):
            assert t.dtype == be.int32 and be.is_gpu(t) and _is_contiguous(t, be), (
                "block_list tensors must be contiguous int32 on the CUDA device"
            )

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)
    if mask_row is None:
        mask_row = seqlen_k - 1
    if num_splits < 1:
        num_splits = swapab_num_splits(
            batch_size, num_heads_kv, seqlen_k, arch == 103, q.dtype == be.bfloat16, head_dim_v
        )
    # More splits than KV blocks would only add empty CTAs.
    num_splits = max(1, min(num_splits, -(-seqlen_k // SWAPAB_TILE_N), 256))
    assert num_heads_kv * num_splits <= CUDA_MAX_GRID_DIM_YZ


    if out is None:
        out = be.empty([batch_size, 1, num_heads, head_dim_v], dtype=q.dtype)
    else:
        assert tuple(out.shape) == (batch_size, 1, num_heads, head_dim_v)
        assert out.dtype == q.dtype and be.strides(out)[-1] == 1
    lse = be.empty([batch_size, num_heads, 1], dtype=be.float32) if return_lse else None

    is_split_kv = num_splits > 1
    if is_split_kv:
        out_partial = be.empty(
            [num_splits, batch_size, 1, num_heads, head_dim_v], dtype=be.float32
        )
        lse_partial = be.empty([num_splits, batch_size, num_heads, 1], dtype=be.float32)
        o_tensor = _to_cute(out_partial, 16)
        lse_tensor = from_dlpack(lse_partial.detach(), assumed_align=4).mark_layout_dynamic(
            leading_dim=lse_partial.ndim - 1
        )
    else:
        o_tensor = _to_cute(out, 16)
        lse_tensor = (
            from_dlpack(lse.detach(), assumed_align=4).mark_layout_dynamic(leading_dim=lse.ndim - 1)
            if lse is not None
            else None
        )
    q_tensor, k_tensor, v_tensor = [_to_cute(t, 16) for t in (q, k, v)]
    mask_tensor = (
        from_dlpack(startend_row_indices.detach(), assumed_align=4).mark_layout_dynamic(
            leading_dim=3
        )
        if startend_row_indices is not None
        else None
    )
    current_stream = cuda.CUstream(be.cuda_stream())

    dtype = be.to_cute_dtype(q.dtype)
    tile_hq = swapab_tile_hq(qhead_per_kvhead)
    # causal only changes the code for num_vecs 2.
    mask_is_causal = bool(causal) and mask_num_vecs == 2

    # Block skipping: per (sequence, KV head), the 128-token units in which row mask_row
    # sees at least one key. The list kernel addresses mask head kv_head // (num_heads_kv / H_m),
    # so the KV heads must share mask heads.
    use_block_list = block_skip and startend_row_indices is not None and not mask_per_head
    if occupancy < 1:
        occupancy = swapab_occupancy(
            batch_size * num_heads_kv * num_splits,
            be.multi_processor_count(),
            -(-seqlen_k // num_splits),
            use_block_list,
            qhead_per_kvhead,
            head_dim_v,
        )
    assert occupancy in SWAPAB_OCCUPANCIES, f"occupancy must be in {SWAPAB_OCCUPANCIES}"
    assert swapab_occupancy_fits(qhead_per_kvhead, head_dim_v, occupancy), (
        f"{occupancy} CTAs per SM do not fit TMEM with {qhead_per_kvhead} q heads per KV head"
    )
    blk_cnt_tensor = blk_idx_tensor = None
    if use_block_list:
        if block_list is None:
            blk_cnt, blk_idx = _decode_block_list(
                startend_row_indices, mask_tensor, num_heads_kv, mask_is_causal, current_stream,
                be, mask_row,
            )
        blk_cnt_tensor = from_dlpack(blk_cnt.detach(), assumed_align=4).mark_layout_dynamic(
            leading_dim=2
        )
        blk_idx_tensor = from_dlpack(blk_idx.detach(), assumed_align=4).mark_layout_dynamic(
            leading_dim=3
        )

    compile_key = (
        dtype,
        head_dim,
        head_dim_v,
        tile_hq,
        qhead_per_kvhead,
        is_split_kv,
        lse_tensor is not None,
        mask_num_vecs,
        mask_is_causal,
        mask_per_head,
        use_block_list,
        occupancy,
    )
    cache = _flash_attn_decode_swapab.compile_cache
    if compile_key not in cache:
        kernel = FlashAttentionDecodeSm100SwapAB(
            head_dim,
            head_dim_v,
            qhead_per_kvhead,
            is_split_kv,
            mask_num_vecs=mask_num_vecs,
            mask_is_causal=mask_is_causal,
            mask_per_head=mask_per_head,
            occupancy=occupancy,
        )
        cache[compile_key] = cute.compile(
            kernel,
            q_tensor,
            k_tensor,
            v_tensor,
            o_tensor,
            lse_tensor,
            mask_tensor,
            blk_cnt_tensor,
            blk_idx_tensor,
            cutlass.Float32(softmax_scale),
            cutlass.Int32(mask_row),
            current_stream,
        )
    cache[compile_key](
        q_tensor,
        k_tensor,
        v_tensor,
        o_tensor,
        lse_tensor,
        mask_tensor,
        blk_cnt_tensor,
        blk_idx_tensor,
        cutlass.Float32(softmax_scale),
        cutlass.Int32(mask_row),
        current_stream,
    )
    if is_split_kv:
        _flash_attn_fwd_combine(
            out_partial,
            lse_partial.transpose(-1, -2),
            out,
            lse.transpose(-1, -2) if lse is not None else None,
        )
    return out, lse


_flash_attn_decode_swapab.compile_cache = {}
