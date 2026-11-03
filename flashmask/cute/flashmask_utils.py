# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 The FlashMask Authors. All rights reserved.

"""
- flashmask v3 (SM90/SM100 era) and computes per-
  block (kBlockN) maxima and minima across a 1D input sequence then stores
  them in output buffers laid out either continuously or in aligned "chunks".
"""

from typing import Optional, NamedTuple
from dataclasses import dataclass
import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
from cutlass.cute.runtime import from_dlpack
from . import utils
from ..backend import Tensor, get_backend
import operator

__all__ = [
    "prepare_block_maxmin",
    "FlashMaskInfoPaddle"
]


# Keep the same constant name for clarity with the CUDA source.
flashmask_buffer_length = 16 * 1024


class FlashMaskInfo(NamedTuple):
    is_causal: bool
    startend_row_indices: cute.Tensor
    LTS_nblock_max: Optional[cute.Tensor]
    LTS_nblock_min: Optional[cute.Tensor]
    LTE_nblock_max: Optional[cute.Tensor]
    LTE_nblock_min: Optional[cute.Tensor]
    UTS_nblock_max: Optional[cute.Tensor]
    UTS_nblock_min: Optional[cute.Tensor]
    UTE_nblock_max: Optional[cute.Tensor]
    UTE_nblock_min: Optional[cute.Tensor]
    valid_block_count: Optional[cute.Tensor]

    def __new_from_mlir_values__(self, values):
        if len(values) == 3:
            values = (self.is_causal, *values, None, None, None, None, None, None, None)
        elif len(values) == 4:
            values = (self.is_causal, *values[:3], None, None, None, None, None, None, *values[3:])
        elif self.is_causal and len(values) == 5:
            values = (self.is_causal, *values, None, None, None, None, None)
        elif self.is_causal and len(values) == 6:
            values = (self.is_causal, *values[:5], None, None, None, None, *values[5:])
        elif not self.is_causal and len(values) == 5:
            values = (self.is_causal, *values[:3], None, None, None, None, *values[3:], None)
        elif not self.is_causal and len(values) == 6:
            values = (self.is_causal, *values[:3], None, None, None, None, *values[3:5], *values[5:])
        elif len(values) == 9:
            values = (self.is_causal, *values, None)
        else:
            values = (self.is_causal, *values)
        return FlashMaskInfo(*values)


class OverlapInfo(NamedTuple):
    """Carries the comm-side readiness handle into the fwd load warp.

    write_ptr is a 1-D int32 device tensor (one entry, but laid out per batch by
    the comm kernel: stride is (s_total - kv_chunk_size) rows per batch). The
    load warp spins on it before each remote KV tile. kv_chunk_size is the local
    chunk row count (== s_local); rows in the last kv_chunk_size of SRBuffer are
    local and never remote-fetched, so they skip the wait. spin_timeout_ns is the
    watchdog budget (0 disables).
    """
    write_ptr: cute.Tensor
    kv_chunk_size: cutlass.Constexpr[int]
    spin_timeout_ns: cutlass.Int64

    def __new_from_mlir_values__(self, values):
        # kv_chunk_size is a compile-time constant; the other fields are MLIR values.
        return OverlapInfo(values[0], self.kv_chunk_size, cutlass.Int64(values[1]))


@dataclass
class FlashMaskInfoPaddle:
    is_causal: bool
    startend_row_indices: Tensor
    LTS_nblock_max: Optional[Tensor] = None
    LTS_nblock_min: Optional[Tensor] = None
    LTE_nblock_max: Optional[Tensor] = None
    LTE_nblock_min: Optional[Tensor] = None
    UTS_nblock_max: Optional[Tensor] = None
    UTS_nblock_min: Optional[Tensor] = None
    UTE_nblock_max: Optional[Tensor] = None
    UTE_nblock_min: Optional[Tensor] = None
    valid_block_count: Optional[Tensor] = None


def _compute_nblock_seqlen(seqlen_k: int, kBlockN: int) -> int:
    """Compute the padded number of blocks (the same formula as original).

    Uses: ((n + kBlockN - 1) / kBlockN + 3) & 0xfffffffc
    The +3 then & ~3 is to make padding to a multiple of 4 (umising: int4 load)
    """
    nblock = (seqlen_k + kBlockN - 1) // kBlockN
    return (nblock + 3) & 0xfffffffc


@cute.kernel
def scan_max_min_kernel(
    mInput: cute.Tensor,  # expected shape (b, n)
    b: cutlass.Int32,
    n: cutlass.Int32,
    kBlockN: cutlass.Int32,
    mMaxO: cute.Tensor,
    mMinO: cute.Tensor,
):
    # thread / block indices
    tidx, tidy, tidz = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    bd_x, bd_y, bd_z = cute.arch.block_dim()

    nblock = (n + kBlockN - 1) // kBlockN
    nblock_seqlen = ((nblock + 3) // 4) * 4

    mInput = cute.make_tensor(mInput.iterator, cute.make_layout((cutlass.Int32(b), cutlass.Int32(n)), stride=(cutlass.Int32(n), cutlass.Int32(1))))
    mMaxO = cute.make_tensor(mMaxO.iterator, cute.make_layout((cutlass.Int32(b * nblock_seqlen)), stride=(cutlass.Int32(1))))
    mMinO = cute.make_tensor(mMinO.iterator, cute.make_layout((cutlass.Int32(b * nblock_seqlen)), stride=(cutlass.Int32(1))))

    # compute batch row id
    bid_row = tidy + bidy * bd_y
    if bid_row < b:
        nblock_idx = bidx

        # lane id within warp
        lane_id = tidx % cute.arch.WARP_SIZE

        # number of 32-element strides per thread required to cover kBlockN
        nums = (kBlockN + 31) // 32

        # Per-thread partial max/min initial values
        maxv = cutlass.Int32(0)
        minv = cutlass.Int32(0x7FFFFFFF)

        # Per-thread starting global element index
        idx = nblock_idx * kBlockN + tidx
        for i in cutlass.range(nums, unroll=1):
            local_pos = lane_id + i * cute.arch.WARP_SIZE
            if (local_pos < kBlockN) and (idx < n):
                # load element (mInput is (b, n))
                # Use domain_offset-style indexing: mInput[bid_row, idx]
                val = cutlass.Int32(mInput[bid_row, idx])
                # update per-thread min/max
                maxv = cutlass.max(maxv, val)
                minv = cutlass.min(minv, val)
            idx = idx + 32

        cute.arch.sync_warp()
        # Warp-level reduction: reduce across the 32 lanes in the warp
        warp_max = utils.warp_reduce(maxv, lambda x, y: cutlass.max(x, y), width=cute.arch.WARP_SIZE)
        warp_min = utils.warp_reduce(minv, lambda x, y: cutlass.min(x, y), width=cute.arch.WARP_SIZE)

        # lane 0 writes the reduced result (one writer per warp)
        if lane_id == 0:
            # compute storage layout indexes similar to CUDA code
            # nblock_seqlen = ((n + kBlockN - 1) / kBlockN + 3) & 0xfffffffc  --> round up to multiple of 4

            dest_idx = bid_row * nblock_seqlen + nblock_idx

            # store to output tensors
            mMaxO[dest_idx] = warp_max
            mMinO[dest_idx] = warp_min

@cute.jit
def scan_max_min_cute(
    mInput: cute.Tensor,  # expected shape (b, h, s)
    b: cutlass.Int32,
    n: cutlass.Int32,
    kBlockN: cutlass.Int32,
    stream: cuda.CUstream,
    mMaxO: cute.Tensor,
    mMinO: cute.Tensor,
):
    scan_max_min_kernel(
        mInput,
        b, n, kBlockN,
        mMaxO,
        mMinO,
    ).launch(
        grid=[(n + kBlockN - 1) // kBlockN, (b + 3) // 4, cutlass.Int32(1)],
        block=[cutlass.Int32(32), cutlass.Int32(4), cutlass.Int32(1)],
        stream=stream,
    )

def _scan_max_min(
    mInput: Tensor,
    b: int,
    n: int,
    mMaxO: Tensor,
    mMinO: Tensor,
    kBlockN: int,
):
    be = get_backend(mInput, mMaxO, mMinO)
    input_tensor = from_dlpack(mInput.contiguous(), assumed_align=4).mark_layout_dynamic(leading_dim=2)
    max_tensor = from_dlpack(mMaxO, assumed_align=4).mark_layout_dynamic(leading_dim=2)
    min_tensor = from_dlpack(mMinO, assumed_align=4).mark_layout_dynamic(leading_dim=2)

    current_stream = cuda.CUstream(be.cuda_stream())

    compile_key = (kBlockN,)
    if compile_key not in _scan_max_min.compile_cache:
        _scan_max_min.compile_cache[compile_key] = cute.compile(
            scan_max_min_cute,
            input_tensor,
            cutlass.Int32(b), cutlass.Int32(n), cutlass.Int32(kBlockN),
            current_stream,
            max_tensor,
            min_tensor,
        )
    _scan_max_min.compile_cache[compile_key](
        input_tensor,
        cutlass.Int32(b), cutlass.Int32(n), cutlass.Int32(kBlockN),
        current_stream,
        max_tensor,
        min_tensor,
    )

_scan_max_min.compile_cache = {}

def prepare_block_maxmin(flashmask_info: FlashMaskInfoPaddle, kBlockN: int = 128):
    """Prepare block-sparse max/min tensors for flashmask.

    The function will compute derived pointers/offsets and call scanMaxMinGpu
    for each existing input pointer.
    """
    be = get_backend(flashmask_info.startend_row_indices)

    batch, heads, seqlen_k, num_vecs = flashmask_info.startend_row_indices.shape
    nblocks = _compute_nblock_seqlen(seqlen_k, kBlockN)

    if num_vecs == 1 and flashmask_info.LTS_nblock_max is None and flashmask_info.LTS_nblock_min is None:
        flashmask_info.LTS_nblock_max = be.zeros([batch, heads, nblocks], dtype=be.int32)
        flashmask_info.LTS_nblock_min = be.zeros([batch, heads, nblocks], dtype=be.int32)
        _scan_max_min(flashmask_info.startend_row_indices[..., 0], batch * heads, seqlen_k, flashmask_info.LTS_nblock_max, flashmask_info.LTS_nblock_min, kBlockN)
    elif num_vecs == 2 and flashmask_info.is_causal and (
            flashmask_info.LTS_nblock_max is None and flashmask_info.LTS_nblock_min is None and
            flashmask_info.LTE_nblock_max is None and flashmask_info.LTE_nblock_min is None
    ):
        flashmask_info.LTS_nblock_max = be.zeros([batch, heads, nblocks], dtype=be.int32)
        flashmask_info.LTS_nblock_min = be.zeros([batch, heads, nblocks], dtype=be.int32)
        flashmask_info.LTE_nblock_max = be.zeros([batch, heads, nblocks], dtype=be.int32)
        flashmask_info.LTE_nblock_min = be.zeros([batch, heads, nblocks], dtype=be.int32)
        _scan_max_min(flashmask_info.startend_row_indices[..., 0], batch * heads, seqlen_k, flashmask_info.LTS_nblock_max, flashmask_info.LTS_nblock_min, kBlockN)
        _scan_max_min(flashmask_info.startend_row_indices[..., 1], batch * heads, seqlen_k, flashmask_info.LTE_nblock_max, flashmask_info.LTE_nblock_min, kBlockN)
    elif num_vecs == 2 and not flashmask_info.is_causal and (
            flashmask_info.LTS_nblock_max is None and flashmask_info.LTS_nblock_min is None and
            flashmask_info.UTE_nblock_max is None and flashmask_info.UTE_nblock_min is None
    ):
        flashmask_info.LTS_nblock_max = be.zeros([batch, heads, nblocks], dtype=be.int32)
        flashmask_info.LTS_nblock_min = be.zeros([batch, heads, nblocks], dtype=be.int32)
        flashmask_info.UTE_nblock_max = be.zeros([batch, heads, nblocks], dtype=be.int32)
        flashmask_info.UTE_nblock_min = be.zeros([batch, heads, nblocks], dtype=be.int32)
        _scan_max_min(flashmask_info.startend_row_indices[..., 0], batch * heads, seqlen_k, flashmask_info.LTS_nblock_max, flashmask_info.LTS_nblock_min, kBlockN)
        _scan_max_min(flashmask_info.startend_row_indices[..., 1], batch * heads, seqlen_k, flashmask_info.UTE_nblock_max, flashmask_info.UTE_nblock_min, kBlockN)
    elif num_vecs == 4 and (
            flashmask_info.LTS_nblock_max is None and flashmask_info.LTS_nblock_min is None and
            flashmask_info.LTE_nblock_max is None and flashmask_info.LTE_nblock_min is None and
            flashmask_info.UTS_nblock_max is None and flashmask_info.UTS_nblock_min is None and
            flashmask_info.UTE_nblock_max is None and flashmask_info.UTE_nblock_min is None
            
    ):
        flashmask_info.LTS_nblock_max = be.zeros([batch, heads, nblocks], dtype=be.int32)
        flashmask_info.LTS_nblock_min = be.zeros([batch, heads, nblocks], dtype=be.int32)
        flashmask_info.LTE_nblock_max = be.zeros([batch, heads, nblocks], dtype=be.int32)
        flashmask_info.LTE_nblock_min = be.zeros([batch, heads, nblocks], dtype=be.int32)
        flashmask_info.UTS_nblock_max = be.zeros([batch, heads, nblocks], dtype=be.int32)
        flashmask_info.UTS_nblock_min = be.zeros([batch, heads, nblocks], dtype=be.int32)
        flashmask_info.UTE_nblock_max = be.zeros([batch, heads, nblocks], dtype=be.int32)
        flashmask_info.UTE_nblock_min = be.zeros([batch, heads, nblocks], dtype=be.int32)
        _scan_max_min(flashmask_info.startend_row_indices[..., 0], batch * heads, seqlen_k, flashmask_info.LTS_nblock_max, flashmask_info.LTS_nblock_min, kBlockN)
        _scan_max_min(flashmask_info.startend_row_indices[..., 1], batch * heads, seqlen_k, flashmask_info.LTE_nblock_max, flashmask_info.LTE_nblock_min, kBlockN)
        _scan_max_min(flashmask_info.startend_row_indices[..., 2], batch * heads, seqlen_k, flashmask_info.UTS_nblock_max, flashmask_info.UTS_nblock_min, kBlockN)
        _scan_max_min(flashmask_info.startend_row_indices[..., 3], batch * heads, seqlen_k, flashmask_info.UTE_nblock_max, flashmask_info.UTE_nblock_min, kBlockN)
    else:
        raise ValueError(f"Unsupported num_vecs={num_vecs} in flashmask_info")
    

def is_flashmask_enabled(flashmask_info: FlashMaskInfoPaddle) -> bool:
    return any(t is not None for t in (
        flashmask_info.LTS_nblock_max,
        flashmask_info.LTS_nblock_min,
        flashmask_info.LTE_nblock_max,
        flashmask_info.LTE_nblock_min,
        flashmask_info.UTS_nblock_max,
        flashmask_info.UTS_nblock_min,
        flashmask_info.UTE_nblock_max,
        flashmask_info.UTE_nblock_min,
    ))

def to_cute_flashmask_info(flashmask_info: FlashMaskInfoPaddle) -> Optional[FlashMaskInfo]:
    if not is_flashmask_enabled(flashmask_info):
        return None

    batch, heads, seqlen_k, num_vecs = flashmask_info.startend_row_indices.shape

    startend_row_indices_tensor = from_dlpack(flashmask_info.startend_row_indices, assumed_align=4).mark_layout_dynamic(leading_dim=3)
    LTS_nblock_max_tensor = None
    LTS_nblock_min_tensor = None
    LTE_nblock_max_tensor = None
    LTE_nblock_min_tensor = None
    UTS_nblock_max_tensor = None
    UTS_nblock_min_tensor = None
    UTE_nblock_max_tensor = None
    UTE_nblock_min_tensor = None

    if num_vecs == 1:
        LTS_nblock_max_tensor = from_dlpack(flashmask_info.LTS_nblock_max, assumed_align=4).mark_layout_dynamic(leading_dim=2)
        LTS_nblock_min_tensor = from_dlpack(flashmask_info.LTS_nblock_min, assumed_align=4).mark_layout_dynamic(leading_dim=2)
    elif num_vecs == 2 and flashmask_info.is_causal:
        LTS_nblock_max_tensor = from_dlpack(flashmask_info.LTS_nblock_max, assumed_align=4).mark_layout_dynamic(leading_dim=2)
        LTS_nblock_min_tensor = from_dlpack(flashmask_info.LTS_nblock_min, assumed_align=4).mark_layout_dynamic(leading_dim=2)
        LTE_nblock_max_tensor = from_dlpack(flashmask_info.LTE_nblock_max, assumed_align=4).mark_layout_dynamic(leading_dim=2)
        LTE_nblock_min_tensor = from_dlpack(flashmask_info.LTE_nblock_min, assumed_align=4).mark_layout_dynamic(leading_dim=2)
    elif num_vecs == 2 and not flashmask_info.is_causal:
        LTS_nblock_max_tensor = from_dlpack(flashmask_info.LTS_nblock_max, assumed_align=4).mark_layout_dynamic(leading_dim=2)
        LTS_nblock_min_tensor = from_dlpack(flashmask_info.LTS_nblock_min, assumed_align=4).mark_layout_dynamic(leading_dim=2)
        UTE_nblock_max_tensor = from_dlpack(flashmask_info.UTE_nblock_max, assumed_align=4).mark_layout_dynamic(leading_dim=2)
        UTE_nblock_min_tensor = from_dlpack(flashmask_info.UTE_nblock_min, assumed_align=4).mark_layout_dynamic(leading_dim=2)
    elif num_vecs == 4:
        LTS_nblock_max_tensor = from_dlpack(flashmask_info.LTS_nblock_max, assumed_align=4).mark_layout_dynamic(leading_dim=2)
        LTS_nblock_min_tensor = from_dlpack(flashmask_info.LTS_nblock_min, assumed_align=4).mark_layout_dynamic(leading_dim=2)
        LTE_nblock_max_tensor = from_dlpack(flashmask_info.LTE_nblock_max, assumed_align=4).mark_layout_dynamic(leading_dim=2)
        LTE_nblock_min_tensor = from_dlpack(flashmask_info.LTE_nblock_min, assumed_align=4).mark_layout_dynamic(leading_dim=2)
        UTS_nblock_max_tensor = from_dlpack(flashmask_info.UTS_nblock_max, assumed_align=4).mark_layout_dynamic(leading_dim=2)
        UTS_nblock_min_tensor = from_dlpack(flashmask_info.UTS_nblock_min, assumed_align=4).mark_layout_dynamic(leading_dim=2)
        UTE_nblock_max_tensor = from_dlpack(flashmask_info.UTE_nblock_max, assumed_align=4).mark_layout_dynamic(leading_dim=2)
        UTE_nblock_min_tensor = from_dlpack(flashmask_info.UTE_nblock_min, assumed_align=4).mark_layout_dynamic(leading_dim=2)
    else:
        raise ValueError(f"Unsupported num_vecs={num_vecs} in flashmask_info")

    if flashmask_info.valid_block_count is not None:
        valid_block_count = from_dlpack(flashmask_info.valid_block_count, assumed_align=4).mark_layout_dynamic(leading_dim=2)
    else:
        valid_block_count = None

    return FlashMaskInfo(
        flashmask_info.is_causal,
        startend_row_indices_tensor,
        LTS_nblock_max_tensor,
        LTS_nblock_min_tensor,
        LTE_nblock_max_tensor,
        LTE_nblock_min_tensor,
        UTS_nblock_max_tensor,
        UTS_nblock_min_tensor,
        UTE_nblock_max_tensor,
        UTE_nblock_min_tensor,
        valid_block_count
    )

@cute.kernel
def reduce_block_count_kernel(
    LTS_nblock_max: cute.Tensor, # [b, h, sk/kBlockN]
    LTE_nblock_min: cute.Tensor,
    UTS_nblock_max: cute.Tensor,
    UTE_nblock_min: cute.Tensor,
    valid_block_count: cute.Tensor, # [b, h, sQ/kBlockM] valid_block_count means how many block are not fully masked in each row
    num_blocks_row: cutlass.Int32,
    num_blocks_col: cutlass.Int32, # num_blocks means how many blocks in a row, note that the padding region of the max/min tensor is not count
    is_causal: cutlass.Constexpr[bool],
    has_lte: cutlass.Constexpr[bool],
    has_uts: cutlass.Constexpr[bool],
    has_ute: cutlass.Constexpr[bool],
    batch_size: cutlass.Int32,
    h_flashmask: cutlass.Int32,
    kBlockM: cutlass.Int32,
    kBlockN: cutlass.Int32,
    seqlen_q: cutlass.Int32,
    seqlen_k: cutlass.Int32,
):
    # one warp per block row
    tidx = cute.arch.thread_idx()[0]
    bidx = cute.arch.block_idx()[0]
    bdimx = cute.arch.block_dim()[0]
    warp_per_block = bdimx >> 5
    # make sure num of threads is multiple of 32
    lane_id = tidx & 31
    warp_id = tidx >> 5

    global_warp_id = warp_id + (bidx * bdimx >> 5)
    block_row_idx = global_warp_id % num_blocks_row
    head_idx = (global_warp_id // num_blocks_row) % h_flashmask
    batch_idx = global_warp_id // (h_flashmask * num_blocks_row)

    batch_head_idx = batch_idx * h_flashmask + head_idx

    global_block_row_idx = block_row_idx + head_idx * num_blocks_row + batch_idx * (h_flashmask * num_blocks_row)
    total_num_blocks_row = batch_size * h_flashmask * num_blocks_row

    if global_block_row_idx < total_num_blocks_row:
        row_idx_start = block_row_idx * kBlockM
        row_idx_end = min(row_idx_start + kBlockM, seqlen_q)
        n_block_max = num_blocks_col
        if is_causal:
            # make sure window_size_right is 0
            n_idx_right = row_idx_end + seqlen_k - seqlen_q
            n_block_max = min(n_block_max, (n_idx_right + kBlockN - 1) // kBlockN)
        loop_num = (n_block_max + 31) >> 5
        local_sum = 0
        for i in cutlass.range(loop_num):
            block_col_idx = i * 32 + lane_id
            if block_col_idx < n_block_max:
                if cutlass.const_expr(has_uts):
                    if not ((row_idx_start >= LTS_nblock_max[batch_idx, head_idx, block_col_idx] and row_idx_end <= LTE_nblock_min[batch_idx, head_idx, block_col_idx]) or (row_idx_start >= UTS_nblock_max[batch_idx, head_idx, block_col_idx] and row_idx_end <= UTE_nblock_min[batch_idx, head_idx, block_col_idx])):
                        local_sum += 1
                elif cutlass.const_expr(has_lte):
                    if not (row_idx_start >= LTS_nblock_max[batch_idx, head_idx, block_col_idx] and row_idx_end <= LTE_nblock_min[batch_idx, head_idx, block_col_idx]):
                        local_sum += 1
                elif cutlass.const_expr(has_ute):
                    if not (row_idx_start >= LTS_nblock_max[batch_idx, head_idx, block_col_idx] or row_idx_end <= UTE_nblock_min[batch_idx, head_idx, block_col_idx]):
                        local_sum += 1
                else:
                    if not (row_idx_start >= LTS_nblock_max[batch_idx, head_idx, block_col_idx]):
                        local_sum += 1

        warp_sum = utils.warp_reduce(local_sum, operator.add)
        if lane_id == 0:
            valid_block_count[batch_idx, head_idx, block_row_idx] = warp_sum

@cute.jit
def reduce_block_count_cute(
    LTS_nblock_max: cute.Tensor, # [b, h_fm, sk/kBlockN]
    LTE_nblock_min: cute.Tensor,
    UTS_nblock_max: cute.Tensor,
    UTE_nblock_min: cute.Tensor,
    valid_block_count: cute.Tensor, # [b,h_fm, sQ/kBlockM] valid_block_count means how many block are not fully masked in each row
    num_blocks_row: cutlass.Int32,
    num_blocks_col: cutlass.Int32, # num_blocks means how many blocks in a row, note that the padding region of the max/min tensor is not count
    is_causal: cutlass.Constexpr[bool],
    has_lte: cutlass.Constexpr[bool],
    has_uts: cutlass.Constexpr[bool],
    has_ute: cutlass.Constexpr[bool],
    batch_size: cutlass.Int32,
    h_flashmask: cutlass.Int32,
    kBlockM: cutlass.Int32,
    kBlockN: cutlass.Int32,
    seqlen_q: cutlass.Int32,
    seqlen_k: cutlass.Int32,
    stream: cuda.CUstream,
):
    reduce_block_count_kernel(
        LTS_nblock_max if LTS_nblock_max is not None else None,
        LTE_nblock_min if LTE_nblock_min is not None else None,
        UTS_nblock_max if UTS_nblock_max is not None else None,
        UTE_nblock_min if UTE_nblock_min is not None else None,
        valid_block_count,
        num_blocks_row,
        num_blocks_col,
        is_causal,
        has_lte,
        has_uts,
        has_ute,
        batch_size,
        h_flashmask,
        kBlockM,
        kBlockN,
        seqlen_q,
        seqlen_k
    ).launch(
        grid=[(num_blocks_row * batch_size * h_flashmask + 3) >> 2, 1, 1],
        block=[32 * 4, 1, 1],
        stream=stream,
    )

# must be called after scan_max_min
def reduce_block_count(
    flashmask_info: FlashMaskInfo,
    is_causal: bool,
    kBlockM: int,
    kBlockN: int,
    seqlen_q: int,
    be,
):
    # Unlike the other entry points here, `flashmask_info` is the *cute*
    # FlashMaskInfo -- its fields are cute.Tensor, so the backend cannot be
    # derived from them and has to come from the caller.
    batch, heads, seqlen_k, num_vecs = flashmask_info.startend_row_indices.shape
    num_blocks_row = (seqlen_q + kBlockM - 1) // kBlockM
    num_blocks_col = (seqlen_k + kBlockN - 1) // kBlockN
    if num_vecs == 4:
        has_lte = True
        has_uts = True
        has_ute = True
    elif num_vecs == 2:
        if flashmask_info.is_causal:
            has_lte = True
            has_uts = False
            has_ute = False
        else:
            has_lte = False
            has_uts = False
            has_ute = True
    else:
        has_lte = False
        has_uts = False
        has_ute = False

    current_stream = cuda.CUstream(be.cuda_stream())

    # TODO: are all of these compile keys necessary?
    compile_key = (is_causal, kBlockM, kBlockN, has_lte, has_uts, has_ute)
    if compile_key not in reduce_block_count.compile_cache:
        reduce_block_count.compile_cache[compile_key] = cute.compile(
            reduce_block_count_cute,
            flashmask_info.LTS_nblock_max,
            flashmask_info.LTE_nblock_min,
            flashmask_info.UTS_nblock_max,
            flashmask_info.UTE_nblock_min,
            flashmask_info.valid_block_count,
            num_blocks_row,
            num_blocks_col,
            is_causal,
            has_lte,
            has_uts,
            has_ute,
            batch,
            heads,
            kBlockM,
            kBlockN,
            seqlen_q,
            seqlen_k,
            current_stream
        )
    reduce_block_count.compile_cache[compile_key](
        flashmask_info.LTS_nblock_max,
        flashmask_info.LTE_nblock_min,
        flashmask_info.UTS_nblock_max,
        flashmask_info.UTE_nblock_min,
        flashmask_info.valid_block_count,
        num_blocks_row,
        num_blocks_col,
        # has_lte,
        # has_uts,
        # has_ute,
        batch,
        heads,
        kBlockM,
        kBlockN,
        seqlen_q,
        seqlen_k,
        current_stream
    )

reduce_block_count.compile_cache = {}


def _flashmask_config_flags(num_vecs: int, is_causal: bool):
    """Map (num_vecs, is_causal) to the has_lte/has_uts/has_ute flags used by the
    fully-masked / partial predicates (mirrors reduce_block_count_kernel)."""
    if num_vecs == 4:
        return True, True, True  # has_lte, has_uts, has_ute
    elif num_vecs == 2:
        if is_causal:
            return True, False, False
        else:
            return False, False, True
    else:
        return False, False, False


def compute_flashmask_block_lists(
    flashmask_info: "FlashMaskInfoPaddle",
    is_causal: bool,
    kBlockM: int,
    kBlockN: int,
    seqlen_q: int,
    seqlen_k: int,
    num_heads: int,
    split_full: bool = False,
):
    """Host-side generation of the per-(batch, head, m_block) compact list of
    surviving KV blocks for flashmask, split into `full` (fully visible, no
    element mask needed) and `mask` (partial, needs the flashmask element mask
    and/or causal/seqlen mask) lists. Fully-masked KV blocks are skipped (absent
    from both lists). This mirrors the v3 forward "n_block index list" so the
    kernel iterates only surviving blocks (arbitrary/mid-range skipping) instead
    of a contiguous [n_block_min, n_block_max) range.

    Predicates replicate reduce_block_count_kernel (fully-masked) and
    _flashmask_block_partial (partial), combined with the causal / seqlen block
    geometry. Returns a BlockSparseTensorsPaddle-compatible tuple:
        (full_block_cnt, full_block_idx, mask_block_cnt, mask_block_idx)
    with cnt shape (b, num_heads, nq) and idx shape (b, num_heads, nq, nk),
    int32, on the same device as startend_row_indices. Indices are stored in
    ascending KV-block order (the consumer walks them high->low).
    """
    be = get_backend(flashmask_info.startend_row_indices)
    srow = flashmask_info.startend_row_indices
    b, h_fm, _, num_vecs = srow.shape
    nq = (seqlen_q + kBlockM - 1) // kBlockM
    nk = (seqlen_k + kBlockN - 1) // kBlockN
    has_lte, has_uts, has_ute = _flashmask_config_flags(num_vecs, is_causal)

    def _sl(t):
        # (b, h_fm, padded_nblocks) -> (b, h_fm, 1, nk)
        return be.astype(t[:, :, :nk].reshape([b, h_fm, 1, nk]), "int32")

    lts_max = _sl(flashmask_info.LTS_nblock_max)
    lts_min = _sl(flashmask_info.LTS_nblock_min)
    lte_max = _sl(flashmask_info.LTE_nblock_max) if has_lte or has_uts else None
    lte_min = _sl(flashmask_info.LTE_nblock_min) if has_lte or has_uts else None
    uts_max = _sl(flashmask_info.UTS_nblock_max) if has_uts else None
    uts_min = _sl(flashmask_info.UTS_nblock_min) if has_uts else None
    ute_max = _sl(flashmask_info.UTE_nblock_max) if has_uts or has_ute else None
    ute_min = _sl(flashmask_info.UTE_nblock_min) if has_uts or has_ute else None

    m_idx = be.arange(nq, dtype="int32").reshape([1, 1, nq, 1])
    m_start = m_idx * kBlockM
    m_end = be.minimum(m_start + kBlockM, be.to_tensor(seqlen_q, dtype="int32"))

    n_idx = be.arange(nk, dtype="int32").reshape([1, 1, 1, nk])
    n_start = n_idx * kBlockN
    n_end = be.minimum(n_start + kBlockN, be.to_tensor(seqlen_k, dtype="int32"))

    # ---- flashmask fully-masked (skip) ----
    if has_uts:
        fm_fully = ((m_start >= lts_max) & (m_end <= lte_min)) | (
            (m_start >= uts_max) & (m_end <= ute_min)
        )
    elif has_lte:
        fm_fully = (m_start >= lts_max) & (m_end <= lte_min)
    elif has_ute:
        fm_fully = (m_start >= lts_max) | (m_end <= ute_min)
    else:
        fm_fully = m_start >= lts_max

    # ---- flashmask partial (needs element mask) ----
    if has_uts:
        fm_partial = ((m_start < lte_max) & (m_end > lts_min)) | (
            (m_start < ute_max) & (m_end > uts_min)
        )
    elif has_lte:
        fm_partial = (m_start < lte_max) & (m_end > lts_min)
    elif has_ute:
        fm_partial = (m_end > lts_min) | (m_start < ute_max)
    else:
        fm_partial = m_end > lts_min

    # ---- causal / seqlen geometry ----
    offset = seqlen_k - seqlen_q
    if is_causal:
        causal_empty = n_start > (m_end - 1 + offset)  # entirely in the future
        causal_full = (n_end - 1) <= (m_start + offset)  # entirely below diagonal
        causal_partial = (~causal_empty) & (~causal_full)
    else:
        causal_empty = be.zeros([1, 1, nq, nk], dtype="bool")
        causal_partial = be.zeros([1, 1, nq, nk], dtype="bool")
    # last KV block may be shorter than kBlockN -> needs seqlen masking
    seqlen_boundary = n_end < (n_start + kBlockN)

    # ---- classify ----
    skip = causal_empty | fm_fully
    needs_mask = causal_partial | seqlen_boundary | fm_partial
    if split_full:
        # Optimization: fully-visible blocks go to the `full` list (no element
        # mask applied). Correct only if the split predicate is exact.
        is_mask = (~skip) & needs_mask
        is_full = (~skip) & (~needs_mask)
    else:
        # Correctness-first: route every surviving block through the `mask` list.
        # The causal/seqlen/flashmask masks applied on a fully-visible block are
        # no-ops, so this is always correct (just skips fewer mask evaluations).
        is_mask = ~skip
        is_full = be.zeros([1, 1, nq, nk], dtype="bool") & skip

    # broadcast (b, h_fm, nq, nk)
    is_mask = be.broadcast_to(is_mask, [b, h_fm, nq, nk])
    is_full = be.broadcast_to(is_full, [b, h_fm, nq, nk])
    n_full = be.broadcast_to(n_idx, [b, h_fm, nq, nk])

    def _pack(sel):
        # Compact the selected KV-block indices to the front in ascending order.
        # Equivalent to the previous sort-based packing (key = n where selected
        # else nk, then sort), but ~20x cheaper: a cumsum prefix-sum + scatter
        # instead of paddle.sort over (b, h_fm, nq, nk). paddle.sort dominated the
        # per-call host prologue that is included in the fwd benchmark timing.
        seli = be.astype(sel, "int32")
        cnt = be.sum(seli, axis=-1)
        # 0-based rank of each selected column among the selected ones, scanning
        # low->high n so the packed output stays ascending (matches the sort).
        rank = be.cumsum(seli, axis=-1) - 1
        # Selected columns scatter their n (= column index) to their rank slot;
        # unselected columns scatter to a trash slot at index nk that is dropped,
        # so they never collide with a valid rank in [0, cnt).
        target = be.astype(be.where(sel, rank, be.full_like(rank, nk)), "int64")
        idx_ext = be.zeros([b, h_fm, nq, nk + 1], dtype="int32")
        idx_ext = be.put_along_axis(idx_ext, target, n_full, axis=-1)
        idx = idx_ext[..., :nk]
        return be.astype(cnt, "int32"), be.astype(idx, "int32")

    full_cnt, full_idx = _pack(is_full)
    mask_cnt, mask_idx = _pack(is_mask)

    # broadcast flashmask heads (h_fm) -> num_heads (GQA / shared flashmask head)
    if h_fm != num_heads:
        rep = num_heads // h_fm
        full_cnt = be.repeat_interleave(full_cnt, rep, axis=1)
        full_idx = be.repeat_interleave(full_idx, rep, axis=1)
        mask_cnt = be.repeat_interleave(mask_cnt, rep, axis=1)
        mask_idx = be.repeat_interleave(mask_idx, rep, axis=1)

    return full_cnt, full_idx, mask_cnt, mask_idx


@cute.kernel
def build_flashmask_block_lists_kernel(
    LTS_nblock_max: cute.Tensor,  # [b, h_fm, nblocks_padded]
    LTS_nblock_min: cute.Tensor,
    LTE_nblock_max: cute.Tensor,
    LTE_nblock_min: cute.Tensor,
    UTS_nblock_max: cute.Tensor,
    UTS_nblock_min: cute.Tensor,
    UTE_nblock_max: cute.Tensor,
    UTE_nblock_min: cute.Tensor,
    full_cnt: cute.Tensor,  # [b, num_heads, nq]
    full_idx: cute.Tensor,  # [b, num_heads, nq, nk]
    mask_cnt: cute.Tensor,  # [b, num_heads, nq]
    mask_idx: cute.Tensor,  # [b, num_heads, nq, nk]
    num_blocks_row: cutlass.Int32,  # nq
    num_blocks_col: cutlass.Int32,  # nk
    is_causal: cutlass.Constexpr[bool],
    has_lte: cutlass.Constexpr[bool],
    has_uts: cutlass.Constexpr[bool],
    has_ute: cutlass.Constexpr[bool],
    split_full: cutlass.Constexpr[bool],
    batch_size: cutlass.Int32,
    num_heads: cutlass.Int32,
    h_flashmask: cutlass.Int32,
    kBlockM: cutlass.Int32,
    kBlockN: cutlass.Int32,
    seqlen_q: cutlass.Int32,
    seqlen_k: cutlass.Int32,
):
    """In-kernel generation of the per-(b, head, m_block) compact surviving-block
    lists. One warp per (batch, head, m_block) row scans all n_blocks in chunks of
    32; each lane classifies its n_block (skip / full / mask) with the identical
    predicates as compute_flashmask_block_lists, then a warp prefix-sum compacts
    the surviving indices (ascending) into full_idx / mask_idx with counts in
    full_cnt / mask_cnt. Replaces the ~40 paddle eager ops (host prologue) with a
    single kernel. idx arrays must be zero-initialized by the caller (unused
    positions stay 0)."""
    tidx = cute.arch.thread_idx()[0]
    bidx = cute.arch.block_idx()[0]
    bdimx = cute.arch.block_dim()[0]
    lane_id = tidx & 31
    warp_id = tidx >> 5
    global_warp_id = warp_id + (bidx * bdimx >> 5)
    total_num_blocks_row = batch_size * num_heads * num_blocks_row

    if global_warp_id < total_num_blocks_row:
        block_row_idx = global_warp_id % num_blocks_row
        head_idx = (global_warp_id // num_blocks_row) % num_heads
        batch_idx = global_warp_id // (num_heads * num_blocks_row)
        fm_head_idx = head_idx // (num_heads // h_flashmask)

        m_start = block_row_idx * kBlockM
        m_end = cutlass.min(m_start + kBlockM, seqlen_q)
        offset = seqlen_k - seqlen_q

        loop_num = (num_blocks_col + 31) >> 5
        run_mask = cutlass.Int32(0)
        run_full = cutlass.Int32(0)
        for i in cutlass.range(loop_num):
            block_col_idx = i * 32 + lane_id
            is_mask = cutlass.Int32(0)
            is_full = cutlass.Int32(0)
            # fm_partial flag for mask-list blocks: encoded into the high bit of
            # the stored index so the fwd producer/consumer can decide whether a
            # per-element flashmask apply is needed without recomputing the
            # (gmem-read) partial predicate per block.
            mask_is_fm_partial = cutlass.Int32(0)
            if block_col_idx < num_blocks_col:
                n_start = block_col_idx * kBlockN
                n_end = cutlass.min(n_start + kBlockN, seqlen_k)
                lts_max = LTS_nblock_max[batch_idx, fm_head_idx, block_col_idx]
                lts_min = LTS_nblock_min[batch_idx, fm_head_idx, block_col_idx]
                if cutlass.const_expr(has_uts):
                    lte_max = LTE_nblock_max[batch_idx, fm_head_idx, block_col_idx]
                    lte_min = LTE_nblock_min[batch_idx, fm_head_idx, block_col_idx]
                    uts_max = UTS_nblock_max[batch_idx, fm_head_idx, block_col_idx]
                    uts_min = UTS_nblock_min[batch_idx, fm_head_idx, block_col_idx]
                    ute_max = UTE_nblock_max[batch_idx, fm_head_idx, block_col_idx]
                    ute_min = UTE_nblock_min[batch_idx, fm_head_idx, block_col_idx]
                    fm_fully = ((m_start >= lts_max) and (m_end <= lte_min)) or (
                        (m_start >= uts_max) and (m_end <= ute_min)
                    )
                    fm_partial = ((m_start < lte_max) and (m_end > lts_min)) or (
                        (m_start < ute_max) and (m_end > uts_min)
                    )
                elif cutlass.const_expr(has_lte):
                    lte_max = LTE_nblock_max[batch_idx, fm_head_idx, block_col_idx]
                    lte_min = LTE_nblock_min[batch_idx, fm_head_idx, block_col_idx]
                    fm_fully = (m_start >= lts_max) and (m_end <= lte_min)
                    fm_partial = (m_start < lte_max) and (m_end > lts_min)
                elif cutlass.const_expr(has_ute):
                    ute_max = UTE_nblock_max[batch_idx, fm_head_idx, block_col_idx]
                    ute_min = UTE_nblock_min[batch_idx, fm_head_idx, block_col_idx]
                    fm_fully = (m_start >= lts_max) or (m_end <= ute_min)
                    fm_partial = (m_end > lts_min) or (m_start < ute_max)
                else:
                    fm_fully = m_start >= lts_max
                    fm_partial = m_end > lts_min

                if cutlass.const_expr(is_causal):
                    causal_empty = n_start > (m_end - 1 + offset)
                    causal_full = (n_end - 1) <= (m_start + offset)
                    causal_partial = (not causal_empty) and (not causal_full)
                    skip = causal_empty or fm_fully
                    needs_mask = causal_partial or (n_end < (n_start + kBlockN)) or fm_partial
                else:
                    skip = fm_fully
                    needs_mask = (n_end < (n_start + kBlockN)) or fm_partial

                if cutlass.const_expr(split_full):
                    if (not skip) and needs_mask:
                        is_mask = cutlass.Int32(1)
                        if fm_partial:
                            mask_is_fm_partial = cutlass.Int32(1)
                    elif not skip:
                        is_full = cutlass.Int32(1)
                else:
                    if not skip:
                        is_mask = cutlass.Int32(1)
                        if fm_partial:
                            mask_is_fm_partial = cutlass.Int32(1)

            # Warp-compact the surviving indices (ascending) with a prefix sum.
            incl_mask = utils.warp_prefix_sum(is_mask, lane_id)
            incl_full = utils.warp_prefix_sum(is_full, lane_id)
            excl_mask = incl_mask - is_mask
            excl_full = incl_full - is_full
            tot_mask = cute.arch.shuffle_sync(incl_mask, cute.arch.WARP_SIZE - 1)
            tot_full = cute.arch.shuffle_sync(incl_full, cute.arch.WARP_SIZE - 1)
            if is_mask != 0:
                # High bit (1<<30) marks blocks that actually overlap the flashmask
                # region (need per-element apply). nk << 2^30 so bit 30 is free.
                encoded_idx = block_col_idx
                if mask_is_fm_partial != 0:
                    encoded_idx = block_col_idx | (cutlass.Int32(1) << 30)
                mask_idx[batch_idx, head_idx, block_row_idx, run_mask + excl_mask] = encoded_idx
            if is_full != 0:
                full_idx[batch_idx, head_idx, block_row_idx, run_full + excl_full] = block_col_idx
            run_mask = run_mask + tot_mask
            run_full = run_full + tot_full

        if lane_id == 0:
            mask_cnt[batch_idx, head_idx, block_row_idx] = run_mask
            full_cnt[batch_idx, head_idx, block_row_idx] = run_full


@cute.jit
def build_flashmask_block_lists_cute(
    LTS_nblock_max: cute.Tensor,
    LTS_nblock_min: cute.Tensor,
    LTE_nblock_max: cute.Tensor,
    LTE_nblock_min: cute.Tensor,
    UTS_nblock_max: cute.Tensor,
    UTS_nblock_min: cute.Tensor,
    UTE_nblock_max: cute.Tensor,
    UTE_nblock_min: cute.Tensor,
    full_cnt: cute.Tensor,
    full_idx: cute.Tensor,
    mask_cnt: cute.Tensor,
    mask_idx: cute.Tensor,
    num_blocks_row: cutlass.Int32,
    num_blocks_col: cutlass.Int32,
    is_causal: cutlass.Constexpr[bool],
    has_lte: cutlass.Constexpr[bool],
    has_uts: cutlass.Constexpr[bool],
    has_ute: cutlass.Constexpr[bool],
    split_full: cutlass.Constexpr[bool],
    batch_size: cutlass.Int32,
    num_heads: cutlass.Int32,
    h_flashmask: cutlass.Int32,
    kBlockM: cutlass.Int32,
    kBlockN: cutlass.Int32,
    seqlen_q: cutlass.Int32,
    seqlen_k: cutlass.Int32,
    stream: cuda.CUstream,
):
    build_flashmask_block_lists_kernel(
        LTS_nblock_max,
        LTS_nblock_min,
        LTE_nblock_max if LTE_nblock_max is not None else None,
        LTE_nblock_min if LTE_nblock_min is not None else None,
        UTS_nblock_max if UTS_nblock_max is not None else None,
        UTS_nblock_min if UTS_nblock_min is not None else None,
        UTE_nblock_max if UTE_nblock_max is not None else None,
        UTE_nblock_min if UTE_nblock_min is not None else None,
        full_cnt,
        full_idx,
        mask_cnt,
        mask_idx,
        num_blocks_row,
        num_blocks_col,
        is_causal,
        has_lte,
        has_uts,
        has_ute,
        split_full,
        batch_size,
        num_heads,
        h_flashmask,
        kBlockM,
        kBlockN,
        seqlen_q,
        seqlen_k,
    ).launch(
        grid=[(num_blocks_row * batch_size * num_heads + 3) >> 2, 1, 1],
        block=[32 * 4, 1, 1],
        stream=stream,
    )


def build_flashmask_block_lists(
    flashmask_info: "FlashMaskInfoPaddle",
    is_causal: bool,
    kBlockM: int,
    kBlockN: int,
    seqlen_q: int,
    seqlen_k: int,
    num_heads: int,
    split_full: bool = True,
):
    """cute-kernel replacement for compute_flashmask_block_lists: produces the same
    (full_cnt, full_idx, mask_cnt, mask_idx) tensors (cnt shape (b, num_heads, nq),
    idx shape (b, num_heads, nq, nk), int32, ascending, unused positions 0) via a
    single kernel launch instead of ~40 paddle eager ops. Requires the *_nblock
    max/min arrays to already be filled (prepare_block_maxmin)."""
    be = get_backend(flashmask_info.startend_row_indices)
    srow = flashmask_info.startend_row_indices
    b, h_fm, _, num_vecs = srow.shape
    nq = (seqlen_q + kBlockM - 1) // kBlockM
    nk = (seqlen_k + kBlockN - 1) // kBlockN
    has_lte, has_uts, has_ute = _flashmask_config_flags(num_vecs, is_causal)

    full_cnt = be.zeros([b, num_heads, nq], dtype=be.int32)
    mask_cnt = be.zeros([b, num_heads, nq], dtype=be.int32)
    full_idx = be.zeros([b, num_heads, nq, nk], dtype=be.int32)
    mask_idx = be.zeros([b, num_heads, nq, nk], dtype=be.int32)

    def _c3(t):
        if t is None:
            return None
        return from_dlpack(t, assumed_align=4).mark_layout_dynamic(leading_dim=2)

    def _c4(t):
        return from_dlpack(t, assumed_align=4).mark_layout_dynamic(leading_dim=3)

    lts_max_t = _c3(flashmask_info.LTS_nblock_max)
    lts_min_t = _c3(flashmask_info.LTS_nblock_min)
    lte_max_t = _c3(flashmask_info.LTE_nblock_max)
    lte_min_t = _c3(flashmask_info.LTE_nblock_min)
    uts_max_t = _c3(flashmask_info.UTS_nblock_max)
    uts_min_t = _c3(flashmask_info.UTS_nblock_min)
    ute_max_t = _c3(flashmask_info.UTE_nblock_max)
    ute_min_t = _c3(flashmask_info.UTE_nblock_min)
    full_cnt_t = _c3(full_cnt)
    mask_cnt_t = _c3(mask_cnt)
    full_idx_t = _c4(full_idx)
    mask_idx_t = _c4(mask_idx)

    current_stream = cuda.CUstream(be.cuda_stream())

    compile_key = (is_causal, has_lte, has_uts, has_ute, split_full, kBlockM, kBlockN)
    if compile_key not in build_flashmask_block_lists.compile_cache:
        build_flashmask_block_lists.compile_cache[compile_key] = cute.compile(
            build_flashmask_block_lists_cute,
            lts_max_t,
            lts_min_t,
            lte_max_t,
            lte_min_t,
            uts_max_t,
            uts_min_t,
            ute_max_t,
            ute_min_t,
            full_cnt_t,
            full_idx_t,
            mask_cnt_t,
            mask_idx_t,
            cutlass.Int32(nq),
            cutlass.Int32(nk),
            is_causal,
            has_lte,
            has_uts,
            has_ute,
            split_full,
            cutlass.Int32(b),
            cutlass.Int32(num_heads),
            cutlass.Int32(h_fm),
            cutlass.Int32(kBlockM),
            cutlass.Int32(kBlockN),
            cutlass.Int32(seqlen_q),
            cutlass.Int32(seqlen_k),
            current_stream,
        )
    build_flashmask_block_lists.compile_cache[compile_key](
        lts_max_t,
        lts_min_t,
        lte_max_t,
        lte_min_t,
        uts_max_t,
        uts_min_t,
        ute_max_t,
        ute_min_t,
        full_cnt_t,
        full_idx_t,
        mask_cnt_t,
        mask_idx_t,
        cutlass.Int32(nq),
        cutlass.Int32(nk),
        cutlass.Int32(b),
        cutlass.Int32(num_heads),
        cutlass.Int32(h_fm),
        cutlass.Int32(kBlockM),
        cutlass.Int32(kBlockN),
        cutlass.Int32(seqlen_q),
        cutlass.Int32(seqlen_k),
        current_stream,
    )
    return full_cnt, full_idx, mask_cnt, mask_idx


build_flashmask_block_lists.compile_cache = {}
