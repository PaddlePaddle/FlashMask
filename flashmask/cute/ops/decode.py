# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025, Tri Dao.
# Copyright (c) 2026 The FlashMask Authors. Modifications for FlashMask.

"""Host-side helpers of the SM100 SwapAB single-token decode kernel.

``_flash_attn_fwd`` runs the kernel when ``swapab_decode_supported`` accepts the call; this
module holds its split / occupancy heuristics and the block-list builder. Framework-agnostic:
allocation goes through a `flashmask.backend` adapter (``be``).
"""

from typing import Optional, Tuple

import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
from cutlass.cute.runtime import from_dlpack

from ...backend import Tensor, get_backend
from ..flash_fwd_decode_sm100 import (
    SWAPAB_HEAD_DIMS,
    SWAPAB_MAX_QHEAD_PER_KVHEAD,
    SWAPAB_SKIP_UNIT,
    build_decode_block_list,
    swapab_occupancy_fits,
    swapab_tmem_cols,
)

__all__ = [
    "swapab_decode_block_list",
    "swapab_decode_supported",
    "swapab_num_splits",
    "swapab_occupancy",
]

CUDA_MAX_GRID_DIM_YZ = 65535

# Static split policy, as upstream SwapABPolicy: decode is bandwidth bound and DRAM saturates
# once there are about _TARGET_CTAS CTAs. SM103 bf16 reaches higher bandwidth with more CTAs; the
# widest head_dim_v needs the larger target at any batch, narrower ones gain nothing from it.
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
# other's KV stream, but each gets half the SMEM, so a shallower K/V ring. That pays off when
# there are enough CTAs to fill both slots and each CTA has little KV; with long KV per CTA the
# shallower ring costs more than the overlap gains, and with too few CTAs it leaves SMs idle.
# Masked cases skip invisible blocks, so their per-CTA work is usually far below
# seqlen_k / splits and they also benefit.
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


def swapab_decode_supported(
    q, k, v, startend_row_indices, causal, be, seqused_k=None, page_table=None
) -> bool:
    """Whether the SwapAB kernel supports this forward call (no autograd checks).

    ``_flash_attn_fwd`` sends single-token calls that pass to the SwapAB kernel. Besides the
    kernel's own asserts this requires the TMA alignment (16-byte base addresses and strides)
    that would otherwise only fail at launch.

    ``seqused_k`` (int32 (B,), device) optionally gives each sequence's KV length; ``page_table``
    (int32 (B, max_pages), device) switches K / V to paged pools with page_size ==
    SWAPAB_SKIP_UNIT.
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        return False
    batch_size, seqlen_q, num_heads, head_dim = q.shape
    _, page_size, num_heads_kv, _ = k.shape
    head_dim_v = v.shape[-1]
    if page_table is not None:
        # One page per 128-token block: a page is exactly one TMA tile.
        if page_size != SWAPAB_SKIP_UNIT:
            return False
        max_seqlen_k = page_table.shape[1] * page_size
        if page_table.dtype != be.int32 or page_table.ndim != 2:
            return False
        if page_table.shape[0] != batch_size or page_table.shape[1] < 1:
            return False
        if not _is_contiguous(page_table, be) or be.data_ptr(page_table) % 4 != 0:
            return False
    else:
        max_seqlen_k = k.shape[1]
    if seqlen_q != 1 or max_seqlen_k < 1 or (head_dim, head_dim_v) not in SWAPAB_HEAD_DIMS:
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
    if seqused_k is not None:
        if seqused_k.dtype != be.int32 or seqused_k.ndim != 1:
            return False
        if seqused_k.shape[0] != batch_size or not _is_contiguous(seqused_k, be):
            return False
        if be.data_ptr(seqused_k) % 4 != 0:
            return False
    if startend_row_indices is not None:
        mask = startend_row_indices
        if mask.ndim != 4 or mask.dtype != be.int32 or not _is_contiguous(mask, be):
            return False
        mask_batch, mask_heads, mask_seqlen, num_vecs = mask.shape
        if mask_batch != batch_size or mask_seqlen != max_seqlen_k or num_vecs not in (1, 2, 4):
            return False
        if (causal and num_vecs == 4) or mask_heads < 1 or num_heads % mask_heads != 0:
            return False
    return True


def _decode_block_list(
    startend_row_indices, mask_tensor, num_heads_kv, is_causal, stream, be, mask_row,
    seqused_k=None, buffers=None,
):
    """(cnt (B, H_kv, 1), idx (B, H_kv, 1, nk)) int32 for the block-skipping SwapAB kernel.

    The list holds the KV units (SWAPAB_SKIP_UNIT tokens each) visible from query row
    ``mask_row``. Entries past cnt are left uninitialized; the kernel never reads them.
    ``seqused_k`` (int32 (B,), optional) truncates each sequence: units past its length are
    never listed.
    ``buffers`` optionally supplies (flags (B, H_m, nk), cnt, idx) int32 tensors, e.g. from a
    FwdWorkspace, so the call allocates nothing.
    """
    batch_size, mask_heads, seqlen_k, num_vecs = startend_row_indices.shape
    nk = -(-seqlen_k // SWAPAB_SKIP_UNIT)
    if buffers is None:
        flags = be.empty([batch_size, mask_heads, nk], dtype=be.int32)
        cnt = be.empty([batch_size, num_heads_kv, 1], dtype=be.int32)
        idx = be.empty([batch_size, num_heads_kv, 1, nk], dtype=be.int32)
    else:
        flags, cnt, idx = buffers
        assert list(flags.shape) == [batch_size, mask_heads, nk], (
            f"flags shape {list(flags.shape)} != [{batch_size}, {mask_heads}, {nk}]"
        )
        assert list(cnt.shape) == [batch_size, num_heads_kv, 1], (
            f"cnt shape {list(cnt.shape)} != [{batch_size}, {num_heads_kv}, 1]"
        )
        assert list(idx.shape) == [batch_size, num_heads_kv, 1, nk], (
            f"idx shape {list(idx.shape)} != [{batch_size}, {num_heads_kv}, 1, {nk}]"
        )
    seqused_tensor = _to_cute(seqused_k, 4) if seqused_k is not None else None
    tensors = [_to_cute(t, 4) for t in (flags, cnt, idx)]
    cache = _decode_block_list.compile_cache
    key = (num_vecs, is_causal, seqused_k is not None)
    if key not in cache:
        cache[key] = cute.compile(
            build_decode_block_list, mask_tensor, seqused_tensor, *tensors,
            cutlass.Int32(mask_row), num_vecs, is_causal, stream,
        )
    cache[key](mask_tensor, seqused_tensor, *tensors, cutlass.Int32(mask_row), stream)
    return cnt, idx


def swapab_decode_block_list(
    startend_row_indices: Tensor,
    num_heads_kv: int,
    *,
    causal: bool = False,
    mask_row: int = 0,
    seqused_k: Optional[Tensor] = None,
    workspace=None,
) -> Tuple[Tensor, Tensor]:
    """Build the SwapAB decode block list once per step.

    Returns (cnt, idx) int32 tensors shaped (B, H_kv, 1) and (B, H_kv, 1, ceil(S/128)).
    Pass ``block_list=(cnt, idx)`` to every forward call of the step so the list is built
    once and shared by all layers.

    With ``workspace`` (a FwdWorkspace built with the mask's num_vecs), the flags / cnt /
    idx buffers come from the workspace and nothing is allocated. Otherwise they are
    allocated here.
    """
    be = get_backend(startend_row_indices)
    mask_tensor = from_dlpack(
        startend_row_indices.detach(), assumed_align=4
    ).mark_layout_dynamic(leading_dim=3)
    # causal only changes the code for num_vecs 2.
    is_causal = causal and startend_row_indices.shape[3] == 2
    buffers = None
    if workspace is not None:
        batch_size, mask_heads, seqlen_k, _ = startend_row_indices.shape
        buffers = workspace.swapab_block_list_views(
            batch_size, mask_heads, num_heads_kv, seqlen_k
        )
    return _decode_block_list(
        startend_row_indices,
        mask_tensor,
        num_heads_kv,
        is_causal,
        cuda.CUstream(be.cuda_stream()),
        be,
        mask_row=mask_row,
        seqused_k=seqused_k,
        buffers=buffers,
    )


_decode_block_list.compile_cache = {}
