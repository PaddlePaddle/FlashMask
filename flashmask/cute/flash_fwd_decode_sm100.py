# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025, Tri Dao.
# Copyright (c) 2026 The FlashMask Authors. Modifications for FlashMask.
#
# SM100 swap-AB single-token decode, ported from the upstream FlashAttention CuTe decode kernel,
# which adapts the NVIDIA CUTLASS GQA decode example.
# It computes S^T = K Q^T and O^T = V^T P^T, so KV tokens fill the MMA M dimension and the
# query heads of one KV head fill the small N dimension.
#
# First FlashMask version: dense KV, no score_mod / local window, (head_dim, head_dim_v) in
# SWAPAB_HEAD_DIMS, no varlen Q. Static split count only.
#
# Optional seqused_k (B,) gives each sequence's KV length; tokens past it are masked out and
# its block-list entries are dropped. Optional paged KV with page_size == SWAPAB_SKIP_UNIT:
# K / V are (num_pages, page_size, H_kv, D) pools and page_table (B, max_pages) picks the
# pool page of each 128-token unit (out-of-range units read page -1 and are TMA zero-filled).
#
# Optional FlashMask column mask startend_row_indices (B, H_m, S_k, num_vecs), applied element
# by element. With a block list (build_decode_block_list), fully masked KV units are skipped.
# The single query is compared as mask row ``mask_row`` (a runtime argument): seqlen_k - 1
# aligns the query bottom-right (it is the newest token), 0 matches the main kernel's Sq = 1
# semantics (flashmask_attention dispatch).
#
# Core contract: Q is (B,H,D), K and V are (B,S,Hkv,D), O is (splits,B,H,D), optional LSE is
# (splits,B,H) fp32 natural log. Empty splits write O=0 and, when requested, LSE=-inf.
# The caller launches the combine kernel if splits>1.
# Warp roles in one CTA: warp 0 QK MMA, warp 1 PV MMA, warp 2 K/V load + final statistics,
# warp 3 Q load + O store; WG1 / WG2 alternate softmax, WG3 corrects O.

import math
from functools import partial
from typing import Optional, Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.pipeline as pipeline
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass import cute, const_expr
from cutlass.cute.nvgpu import OperandMajorMode, cpasync, tcgen05
from cutlass.cute.typing import Float32, Int32
from cutlass.memory import SmemAllocator, TmemAllocator, get_smem_capacity_in_bytes

from . import copy_utils
from .cute_dsl_utils import assume_tensor_aligned
from .named_barrier import NamedBarrierFwdSm100SwapAB
from .pack_gqa import pack_gqa_layout
from .pipeline import NamedBarrier


exp2_fast = partial(cute.math.exp2, fastmath=True)
warp_reduce_max = partial(cute.arch.warp_redux_sync, kind="fmax", nan=True)
smem_atomic_max = partial(cute.arch.atomic_fmax, sem="relaxed", scope="cta")

WARP_SIZE = 32
WARPS_PER_WG = 4
WG_SIZE = WARPS_PER_WG * WARP_SIZE
MMA_N_MIN = 8
NUM_SOFTMAX_WG = 2
NUM_WG = NUM_SOFTMAX_WG + 2
LOG2_E = math.log2(math.e)

# Kernel shape constants (see SwapABPolicy upstream).
SWAPAB_TILE_N = 256
# Supported (head_dim, head_dim_v): QK and PV both tile their dims to the MMA shapes below.
SWAPAB_HEAD_DIMS = ((128, 128), (192, 128), (256, 256), (512, 512), (576, 512))
SWAPAB_MAX_QHEAD_PER_KVHEAD = 32
_TMEM_COLS = 512
_NUM_STAGES_S, _NUM_STAGES_O = 4, 2
# SM100 shared memory per SM, and what the hardware reserves per resident CTA.
_SM_SMEM_BYTES = 228 * 1024
_SMEM_RESERVED_PER_CTA = 1024
SWAPAB_OCCUPANCIES = (1, 2)


def swapab_tile_hq(qhead_per_kvhead: int) -> int:
    """Query heads per CTA: the next power of two, so one head block covers the group."""
    return 1 << (qhead_per_kvhead - 1).bit_length()


def swapab_tmem_cols(qhead_per_kvhead: int, head_dim_v: int) -> int:
    """TMEM columns the kernel needs: S stages x 2 sub-tiles, L mailboxes, O slots x d tiles."""
    tile_hq_mma = max(MMA_N_MIN, swapab_tile_hq(qhead_per_kvhead))
    num_d_tiles = -(-head_dim_v // 128)
    num_tiles = _NUM_STAGES_S * (SWAPAB_TILE_N // 128) + NUM_SOFTMAX_WG
    return tile_hq_mma * (num_tiles + _NUM_STAGES_O * num_d_tiles)


def swapab_tmem_alloc_cols(qhead_per_kvhead: int, head_dim_v: int, occupancy: int) -> int:
    """TMEM columns each CTA allocates: all 512 alone on an SM, else the power of two it needs."""
    if occupancy == 1:
        return _TMEM_COLS
    return max(32, 1 << (swapab_tmem_cols(qhead_per_kvhead, head_dim_v) - 1).bit_length())


def swapab_occupancy_fits(qhead_per_kvhead: int, head_dim_v: int, occupancy: int) -> bool:
    """Whether occupancy CTAs of this shape fit in one SM's TMEM."""
    return occupancy * swapab_tmem_alloc_cols(qhead_per_kvhead, head_dim_v, occupancy) <= _TMEM_COLS


@cute.jit
def load_kv_stage(
    pipeline_kv: pipeline.PipelineTmaUmma,
    kv_state: pipeline.PipelineState,
    kv_token: cutlass.Boolean,
    tma_atom: cute.CopyAtom,
    tXgX: cute.Tensor,
    tXsX: cute.Tensor,
) -> Tuple[pipeline.PipelineState, cutlass.Boolean]:
    """TMA one K or V^T tile tXgX into the next ring stage of tXsX (staged by its last mode).

    Acquires the stage (skipping the wait if kv_token says it is already free), issues the copy
    on its full barrier, and returns the advanced producer state with the next stage's token.
    """
    pipeline_kv.producer_acquire(kv_state, kv_token)
    barrier = pipeline_kv.producer_get_barrier(kv_state)
    cute.copy(tma_atom, tXgX, tXsX[None, kv_state.index], tma_bar_ptr=barrier)
    pipeline_kv.producer_commit(kv_state)
    kv_state.advance()
    return kv_state, pipeline_kv.producer_try_acquire(kv_state)


@cute.jit
def flashmask_col_masked(
    mStartEnd: cute.Tensor,
    batch_idx: Int32,
    mask_head_idx: Int32,
    token: Int32,
    row: Int32,
    num_vecs: cutlass.Constexpr[int],
    is_causal: cutlass.Constexpr[bool],
) -> cutlass.Boolean:
    """Whether the column mask hides key ``token`` from query ``row`` (as mask.py).

    num_vecs 1: [LTS, inf); 2 causal: [LTS, LTE); 2 non-causal: [LTS, inf) and [0, UTE);
    4: [LTS, LTE) and [UTS, UTE).
    """
    vecs = []
    for v in cutlass.range_constexpr(num_vecs):
        vecs.append(mStartEnd[batch_idx, mask_head_idx, token, v])
    return col_masked_by(vecs, row, num_vecs, is_causal)


def col_masked_by(vecs, row, num_vecs, is_causal):
    """flashmask_col_masked from the token's num_vecs mask entries ``vecs``, already loaded."""
    masked = row >= vecs[0]
    if const_expr(num_vecs == 2):
        if const_expr(is_causal):
            masked = masked & (row < vecs[1])
        else:
            masked = masked | (row < vecs[1])
    elif const_expr(num_vecs == 4):
        masked = (masked & (row < vecs[1])) | ((row >= vecs[2]) & (row < vecs[3]))
    return masked


# Block list for decode block skipping, at the granularity of one QK MMA tile of
# SWAPAB_SKIP_UNIT tokens (a "unit"; a KV block of SWAPAB_TILE_N tokens holds several). Kernel 1
# flags every (sequence, mask head, unit) in which the query's mask row sees at least one key,
# from the mask entries themselves. Kernel 2 compacts the flagged units of each (sequence, KV
# head) into an ascending list. Only list positions below the count are written. The attention
# kernel packs consecutive list entries into its blocks, so units need not be adjacent. Listed
# units are still masked per token: the in-kernel mask read is cheap, so the list does not mark
# which units are fully visible.
SWAPAB_SKIP_UNIT = 128
BLOCK_LIST_WARPS = 4
# Units each flags warp tests, and flags each compact lane loads per pass: both kernels are
# bound by load latency, so each warp keeps one batch of loads in flight (even a long sequence
# compacts in few passes).
BLOCK_FLAGS_UNITS = 4
BLOCK_COMPACT_UNROLL = 32


@cute.jit
def decode_kv_unit(
    mBlockIdx: cute.Tensor,
    batch_idx: Int32,
    kv_head_idx: Int32,
    pos: Int32,
    num_listed: Int32,
    oob_unit: Int32,
) -> Int32:
    """The KV unit at block-list position ``pos``.

    Positions past the list (the second half of an odd-length list's last block) map to
    ``oob_unit``, which lies wholly past seqlen_k: TMA zero-fills it without reading memory and
    the tail mask hides its rows.
    """
    unit = oob_unit
    if pos < num_listed:
        unit = mBlockIdx[batch_idx, kv_head_idx, 0, pos]
    return unit


@cute.kernel
def _block_flags_kernel(
    mStartEnd: cute.Tensor,  # (B, H_m, S, num_vecs)
    mSeqUsedK: Optional[cute.Tensor],  # (B,) int32: only the first seqused_k columns count
    mFlags: cute.Tensor,  # (B, H_m, nk)
    row: Int32,
    num_vecs: cutlass.Constexpr[int],
    is_causal: cutlass.Constexpr[bool],
):
    tidx = cute.arch.thread_idx()[0]
    bidx, bidy, _ = cute.arch.block_idx()
    # Let the compact kernel launch now; it waits for this grid before reading the flags.
    cute.arch.griddepcontrol_launch_dependents()
    lane = tidx % WARP_SIZE
    blk0 = (bidx * BLOCK_LIST_WARPS + tidx // WARP_SIZE) * BLOCK_FLAGS_UNITS
    num_mask_heads, max_seqlen_k = mStartEnd.shape[1], mStartEnd.shape[2]
    batch_idx = bidy // num_mask_heads
    mask_head_idx = bidy % num_mask_heads
    seq_len = max_seqlen_k
    if const_expr(mSeqUsedK is not None):
        seq_len = mSeqUsedK[batch_idx]
    tokens_per_lane = SWAPAB_SKIP_UNIT // WARP_SIZE
    # Issue the loads of all the warp's units before testing any: one memory round trip.
    vals = []
    for t in cutlass.range_constexpr(BLOCK_FLAGS_UNITS * tokens_per_lane):
        token = blk0 * SWAPAB_SKIP_UNIT + t * WARP_SIZE + lane
        for v in cutlass.range_constexpr(num_vecs):
            x = Int32(0)
            if token < seq_len:
                x = mStartEnd[batch_idx, mask_head_idx, token, v]
            vals.append(x)
    for u in cutlass.range_constexpr(BLOCK_FLAGS_UNITS):
        blk = blk0 + u
        if blk < mFlags.shape[2]:
            any_visible = Int32(0)
            for i in cutlass.range_constexpr(tokens_per_lane):
                t = u * tokens_per_lane + i
                token = blk0 * SWAPAB_SKIP_UNIT + t * WARP_SIZE + lane
                vecs = vals[t * num_vecs:(t + 1) * num_vecs]
                if token < seq_len:
                    if not col_masked_by(vecs, row, num_vecs, is_causal):
                        any_visible = Int32(1)
            visible = cute.arch.vote_any_sync(any_visible != 0)
            if lane == 0:
                flag = Int32(0)
                if visible:
                    flag = Int32(1)
                mFlags[batch_idx, mask_head_idx, blk] = flag


@cute.kernel
def _block_compact_kernel(
    mFlags: cute.Tensor,  # (B, H_m, nk)
    mBlockCnt: cute.Tensor,  # (B, H_kv, 1)
    mBlockIdx: cute.Tensor,  # (B, H_kv, 1, nk)
):
    tidx = cute.arch.thread_idx()[0]
    bidx = cute.arch.block_idx()[0]
    # Launched with PDL after the flags kernel; the attention kernel waits for this grid in turn.
    cute.arch.griddepcontrol_wait()
    cute.arch.griddepcontrol_launch_dependents()
    lane = tidx % WARP_SIZE
    batch_size, num_heads_kv = mBlockCnt.shape[0], mBlockCnt.shape[1]
    num_mask_heads, nk = mFlags.shape[1], mFlags.shape[2]
    row_id = bidx * BLOCK_LIST_WARPS + tidx // WARP_SIZE
    if row_id < batch_size * num_heads_kv:
        batch_idx = row_id // num_heads_kv
        kv_head_idx = row_id % num_heads_kv
        mask_head_idx = kv_head_idx // (num_heads_kv // num_mask_heads)
        count = Int32(0)
        lanes_below = (Int32(1) << lane) - Int32(1)
        chunk = WARP_SIZE * BLOCK_COMPACT_UNROLL
        # Issue a chunk's flag loads together, then compact them with one ballot per warp-width
        # of units: the loop is bound by load latency, and a long sequence has many units.
        for i in cutlass.range((nk + chunk - 1) // chunk, unroll=1):
            flags = []
            for j in cutlass.range_constexpr(BLOCK_COMPACT_UNROLL):
                blk = i * chunk + j * WARP_SIZE + lane
                flag = Int32(0)
                if blk < nk:
                    flag = mFlags[batch_idx, mask_head_idx, blk]
                flags.append(flag)
            for j in cutlass.range_constexpr(BLOCK_COMPACT_UNROLL):
                blk = i * chunk + j * WARP_SIZE + lane
                keep = flags[j] != 0
                ballot = cute.arch.vote_ballot_sync(keep)
                if keep:
                    pos = count + cute.arch.popc(ballot & lanes_below)
                    mBlockIdx[batch_idx, kv_head_idx, 0, pos] = blk
                count += cute.arch.popc(ballot)
        if lane == 0:
            mBlockCnt[batch_idx, kv_head_idx, 0] = count


@cute.jit
def build_decode_block_list(
    mStartEnd: cute.Tensor,
    mSeqUsedK: Optional[cute.Tensor],
    mFlags: cute.Tensor,
    mBlockCnt: cute.Tensor,
    mBlockIdx: cute.Tensor,
    mask_row: Int32,
    num_vecs: cutlass.Constexpr[int],
    is_causal: cutlass.Constexpr[bool],
    stream: cuda.CUstream,
):
    """Fill mBlockCnt / mBlockIdx for decode block skipping (two launches, no memset).

    The compact kernel is a programmatic dependent launch of the flags kernel, and triggers the
    next launch on entry, so an attention kernel launched with PDL can start its prologue early.
    """
    batch_size, num_mask_heads, nk = mFlags.shape
    units_per_cta = BLOCK_LIST_WARPS * BLOCK_FLAGS_UNITS
    _block_flags_kernel(mStartEnd, mSeqUsedK, mFlags, mask_row, num_vecs, is_causal).launch(
        grid=[cute.ceil_div(nk, units_per_cta), batch_size * num_mask_heads, 1],
        block=[BLOCK_LIST_WARPS * WARP_SIZE, 1, 1],
        stream=stream,
    )
    num_rows = mBlockCnt.shape[0] * mBlockCnt.shape[1]
    _block_compact_kernel(mFlags, mBlockCnt, mBlockIdx).launch(
        grid=[cute.ceil_div(num_rows, BLOCK_LIST_WARPS), 1, 1],
        block=[BLOCK_LIST_WARPS * WARP_SIZE, 1, 1],
        stream=stream,
        use_pdl=True,
    )


def _split_block_range(n_block_total, split_id, num_splits):
    """(n_block_min, n_block_max, count) of this split: equal contiguous runs of the blocks, as
    in upstream BlockInfo.get_n_block_min_max."""
    per_split = (n_block_total + num_splits - 1) // num_splits
    n_block_min = split_id * per_split
    n_block_max = cutlass.min(n_block_min + per_split, n_block_total)
    return n_block_min, n_block_max, cutlass.max(n_block_max - n_block_min, 0)


def select_modes(T, order):
    """``T`` with its modes permuted to ``order`` (a full permutation), same data."""
    assert sorted(order) == list(range(cute.rank(T))), f"{order} is not a permutation of modes"
    return cute.make_tensor(T.iterator, cute.select(T.layout, mode=order))


def gemm_view(mX, order, head_idx=None, qhead_per_kvhead=None, num_heads_kv=None):
    """mX with its modes permuted to order and the (kv head, batch) modes 2 and 3 grouped.

    With head_idx, the head mode is first split into (qhead_per_kvhead, num_heads_kv) as in
    pack_gqa_layout, so order indexes (qhead_per_kvhead, mode 0, ..., num_heads_kv, ...).
    """
    if head_idx is not None:
        mX = cute.flatten(pack_gqa_layout(mX, qhead_per_kvhead, num_heads_kv, head_idx))
    return cute.group_modes(select_modes(mX, order), 2, 4)


class _SwapABCore:
    def __init__(
        self,
        tile_hq,
        tile_n,
        head_dim,
        head_dim_v,
        dtype,
        mask_num_vecs=0,
        mask_is_causal=False,
        mask_per_head=False,
        occupancy=1,
    ):
        self.tile_hq = tile_hq
        self.tile_n = tile_n
        self.head_dim = head_dim
        self.head_dim_v = head_dim_v
        self.dtype = dtype
        self.qk_acc_dtype = cutlass.Float32
        # Column mask: num_vecs (0 = none); is_causal picks LTE vs UTE for num_vecs 2;
        # per_head when the query heads of one KV head use different mask heads.
        self.mask_num_vecs = mask_num_vecs
        self.mask_is_causal = mask_is_causal
        self.mask_per_head = mask_per_head
        # Resident CTAs per SM: splits the SMEM and TMEM budget and sets the launch bound.
        self.occupancy = occupancy

        assert (head_dim, head_dim_v) in SWAPAB_HEAD_DIMS
        assert tile_hq in (1, 2, 4, 8, 16, 32)
        assert tile_n > 0 and tile_n % 128 == 0 and tile_n <= 256
        assert mask_num_vecs in (0, 1, 2, 4)
        assert occupancy in SWAPAB_OCCUPANCIES

    def make_shared_storage(self, sQ_layout, sK_layout, sP_layout):
        tile_hq_mma = self.mma_tiler[1]

        @cute.struct
        class SharedStorage:
            mbar_q: cute.struct.MemRange[cutlass.Int64, 1]
            mbar_kv: cute.struct.MemRange[cutlass.Int64, 2 * self.num_stages_kv]
            mbar_s: cute.struct.MemRange[cutlass.Int64, 2 * self.num_stages_s]
            mbar_p: cute.struct.MemRange[cutlass.Int64, 2 * self.num_stages_p]
            mbar_o: cute.struct.MemRange[cutlass.Int64, 2 * self.num_stages_o]
            tmem_holding_buf: Int32
            # Per-head running max for softmax/correction; split stats later publish 1/L here.
            sRowMax: cute.struct.Align[cute.struct.MemRange[Float32, tile_hq_mma], 128]
            # Per-warp row sums from the correction WG for the split statistics.
            sRowSumWarp: cute.struct.Align[
                cute.struct.MemRange[Float32, WARPS_PER_WG * self.tile_hq], 128
            ]
            sP: cute.struct.Align[cute.struct.MemRange[self.dtype, cute.cosize(sP_layout)], 1024]
            sQ: cute.struct.Align[cute.struct.MemRange[self.dtype, cute.cosize(sQ_layout)], 1024]
            # One ring for both K and V^T tiles (same bytes, two swizzled views), reused for O.
            sKV: cute.struct.Align[cute.struct.MemRange[self.dtype, cute.cosize(sK_layout)], 1024]

        return SharedStorage

    def plan_kv_ring(self, tiled_mma_qk, sQ_layout, sP_layout):
        """Size the K/V ring and set the stages and storage.

        Alone on an SM the ring is one stage below the deepest that fits: upstream found the
        deepest ring slower for small-head_dim MHA decode, and the decode heuristics were tuned
        at this depth. With several CTAs per SM each gets an equal share of the SM and the
        deepest ring in it, since one stage fewer was measurably slower there.
        """

        def storage_for(num_stages_kv):
            self.num_stages_kv = num_stages_kv
            sK_layout = sm100_utils.make_smem_layout_a(
                tiled_mma_qk, self.mma_tiler, self.dtype, num_stages_kv
            )
            self.shared_storage = self.make_shared_storage(sQ_layout, sK_layout, sP_layout)
            return sK_layout

        smem_capacity = get_smem_capacity_in_bytes("sm_100")
        if const_expr(self.occupancy > 1):
            smem_capacity = _SM_SMEM_BYTES // self.occupancy - _SMEM_RESERVED_PER_CTA
        tile_m, _, tile_k = self.mma_tiler
        num_stages_kv = smem_capacity // (tile_m * tile_k * self.dtype.width // 8)
        storage_for(num_stages_kv)
        while self.shared_storage.size_in_bytes() > smem_capacity:
            num_stages_kv -= 1
            storage_for(num_stages_kv)
        assert num_stages_kv > 1
        if const_expr(self.occupancy == 1):
            num_stages_kv -= 1
        return storage_for(num_stages_kv)

    @cute.jit
    def __call__(
        self,
        problem_size: Tuple[Int32, Int32, Int32, Int32],
        mQ: cute.Tensor,  # (batch, num_heads, head_dim), head_dim contiguous
        mK: cute.Tensor,  # (batch, seqlen_k, num_heads_kv, head_dim), head_dim contiguous
        mV: cute.Tensor,  # (batch, seqlen_k, num_heads_kv, head_dim_v), last dim contiguous
        mO_partial: cute.Tensor,  # (num_splits, batch, num_heads, head_dim_v)
        mLSE_partial: cute.Tensor | None,  # (num_splits, batch, num_heads), natural log
        mStartEnd: cute.Tensor | None,  # (batch, H_m, seqlen_k, num_vecs) int32 column mask
        mBlockCnt: cute.Tensor | None,  # (batch, num_heads_kv, 1) int32 visible-unit count
        mBlockIdx: cute.Tensor | None,  # (batch, num_heads_kv, 1, n_units) int32 unit list
        mSeqUsedK: cute.Tensor | None,  # (batch,) int32: KV length of each sequence
        mPageTable: cute.Tensor | None,  # (batch, max_pages) int32 paged-KV table
        mNumSplitsDynamic: cute.Tensor | None,  # (batch,) int32: per-batch split count
        softmax_scale: Float32,
        mask_row: Int32,  # the query's row in the column mask
        stream: cuda.CUstream,
    ):
        hdim = self.head_dim
        hdim_v = self.head_dim_v
        batch, num_heads, num_heads_kv, _ = problem_size
        softmax_scale_log2 = softmax_scale * LOG2_E
        qhead_per_kvhead = num_heads // num_heads_kv
        tile_n = self.tile_n

        # GEMM views over the (B, S, H, D) layouts with the real strides (so e.g. a KV cache slice
        # works without a copy). The (kv head, batch) pair is one hierarchical mode, indexed by
        # bh_idx = batch_idx * num_heads_kv + kv_head_idx.
        # QK: K is operand A (M = tokens, K = head_dim), Q is operand B (N = query heads of this
        # KV head). PV: V^T is operand A (M = head_dim_v, K = tokens).
        # Paged KV: mK / mV are (num_pages, page_size, H_kv, D) pools. page_size == 128 (one
        # SWAPAB_SKIP_UNIT) so a unit is exactly one page; the page is an extra free mode next to
        # the KV head rather than part of the (h_kv, b) group.
        if const_expr(mPageTable is None):
            mK_qk = gemm_view(mK, [1, 3, 2, 0])  # (seqlen_k, d, (h_kv, b))
            mVt_pv = gemm_view(mV, [3, 1, 2, 0])  # (d_v, seqlen_k, (h_kv, b))
        else:
            mK_qk = select_modes(mK, [1, 3, 2, 0])  # (page_size, d, h_kv, num_pages)
            mVt_pv = select_modes(mV, [3, 1, 2, 0])  # (d_v, page_size, h_kv, num_pages)
        gqa = (qhead_per_kvhead, num_heads_kv)
        mQ_qk = gemm_view(mQ, [0, 3, 2, 1], 1, *gqa)  # (qpk, d, (h_kv, b))
        # (d_v, qpk, (h_kv, b), ns)
        mO_partial_tma = gemm_view(mO_partial, [4, 0, 3, 2, 1], 2, *gqa)
        mLSE_partial_fwd = None
        if const_expr(mLSE_partial is not None):
            # (ns, qpk, (h_kv, b))
            mLSE_partial_fwd = gemm_view(mLSE_partial, [1, 0, 3, 2], 2, *gqa)

        # MMA tile: M is the full tcgen05 M so S and O use the canonical TMEM accumulator layout.
        # The K-step is the same for both GEMMs so that a K tile (tokens x dims) and a V^T tile
        # (dims x tokens) have the same size and can share one SMEM ring.
        tile_m = 128
        tile_hq_mma = max(MMA_N_MIN, self.tile_hq)
        tile_k = 64
        assert tile_hq_mma % self.tile_hq == 0
        num_s_subtiles = cute.ceil_div(tile_n, tile_m)
        num_k_tiles = cute.ceil_div(hdim, tile_k)
        num_d_tiles = cute.ceil_div(hdim_v, tile_m)
        num_pv_k_tiles = cute.ceil_div(tile_n, tile_k)
        self.mma_tiler = (tile_m, tile_hq_mma, tile_k)
        self.subtile_counts = (num_s_subtiles, num_k_tiles, num_d_tiles, num_pv_k_tiles)

        # Pipeline depths. S: TMEM score buffers, P: SMEM buffers, O: two TMEM slots (even / odd
        # blocks), L: one row-sum mailbox per softmax WG. The K/V ring gets the SMEM left over.
        self.num_stages_q = num_k_tiles
        self.num_stages_p = 4
        self.num_stages_s = _NUM_STAGES_S
        self.num_stages_l = NUM_SOFTMAX_WG
        self.num_stages_o = _NUM_STAGES_O

        # K and Q are both contiguous along head_dim (the QK reduction dim) -> K-major.
        tiled_mma_qk = sm100_utils.make_trivial_tiled_mma(
            self.dtype,
            OperandMajorMode.K,
            OperandMajorMode.K,
            self.qk_acc_dtype,
            tcgen05.CtaGroup.ONE,
            self.mma_tiler[:2],
        )
        # The PV reduction runs over tokens, the contiguous dim of neither V nor P -> MN-major.
        tiled_mma_pv = sm100_utils.make_trivial_tiled_mma(
            self.dtype,
            OperandMajorMode.MN,
            OperandMajorMode.MN,
            self.qk_acc_dtype,
            tcgen05.CtaGroup.ONE,
            self.mma_tiler[:2],
        )

        sQ_layout = sm100_utils.make_smem_layout_b(
            tiled_mma_qk, self.mma_tiler, self.dtype, self.num_stages_q
        )
        sP_layout = sm100_utils.make_smem_layout_b(
            tiled_mma_pv, (None, tile_hq_mma, tile_n), self.dtype, self.num_stages_p
        )
        sO_layout_atom = tcgen05.make_smem_layout_atom(
            tcgen05.mma.SmemLayoutAtomKind.MN_SW128, mO_partial_tma.element_type
        )

        # O staging tile uses the output dtype and reuses the K/V ring SMEM at the end.
        O_shape = (num_d_tiles * tile_m, tile_hq_mma)
        sO_layout = cute.tile_to_shape(sO_layout_atom, O_shape, order=(1, 0))
        sO_layout = cute.flat_divide(sO_layout, (tile_m, tile_hq_mma))

        sK_layout = self.plan_kv_ring(tiled_mma_qk, sQ_layout, sP_layout)
        sVt_layout = sm100_utils.make_smem_layout_a(
            tiled_mma_pv, self.mma_tiler, self.dtype, self.num_stages_kv
        )
        # The V^T view and the O staging tile alias the K ring. A V^T stage must span exactly the
        # bytes of a K stage: the KV pipeline arms one transaction count for both.
        ring_bytes = cute.size_in_bytes(self.dtype, sK_layout)
        k_stage_bytes = cute.size_in_bytes(self.dtype, cute.select(sK_layout, mode=[0, 1, 2]))
        vt_stage_bytes = cute.size_in_bytes(self.dtype, cute.select(sVt_layout, mode=[0, 1, 2]))
        assert vt_stage_bytes == k_stage_bytes
        assert cute.size_in_bytes(self.dtype, sVt_layout) <= ring_bytes
        o_bytes = cute.size_in_bytes(mO_partial_tma.element_type, sO_layout)
        assert o_bytes <= ring_bytes

        # TMA atoms; the returned coordinate tensors replace the GEMM views in the kernel.
        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        cluster_shape_vmnk = (1, 1, 1, 1)
        tma_atom_Q, mQ_tma = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            mQ_qk,
            cute.select(sQ_layout, mode=[0, 1, 2]),
            self.mma_tiler,
            tiled_mma_qk,
            cluster_shape_vmnk,
        )
        tma_atom_K, mK_tma = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            mK_qk,
            cute.select(sK_layout, mode=[0, 1, 2]),
            self.mma_tiler,
            tiled_mma_qk,
            cluster_shape_vmnk,
        )
        tma_atom_V, mVt_tma = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            mVt_pv,
            cute.select(sVt_layout, mode=[0, 1, 2]),
            self.mma_tiler,
            tiled_mma_pv,
            cluster_shape_vmnk,
        )
        tma_atom_O, mO_tma = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            mO_partial_tma,
            cute.select(sO_layout, mode=[0, 1]),
            (tile_m, tile_hq_mma),
        )

        # One CTA per (split, KV head, sequence); tile_hq covers every query head of a KV head
        # (host-checked). Splits vary fastest in grid Y, as in the upstream tile scheduler.
        num_splits = mO_partial.shape[0]
        grid = (1, num_heads_kv * num_splits, batch)
        self.attn_kernel(
            problem_size,
            tma_atom_Q,
            mQ_tma,
            tma_atom_K,
            mK_tma,
            tma_atom_V,
            mVt_tma,
            tma_atom_O,
            mO_tma,
            mO_partial,
            mLSE_partial_fwd,
            mStartEnd,
            mBlockCnt,
            mBlockIdx,
            mSeqUsedK,
            mPageTable,
            mNumSplitsDynamic,
            softmax_scale_log2,
            mask_row,
            tiled_mma_qk,
            tiled_mma_pv,
            sQ_layout,
            sK_layout,
            sVt_layout,
            sO_layout,
            sP_layout,
            num_splits,
        ).launch(
            grid=grid,
            block=(NUM_WG * WG_SIZE, 1, 1),
            cluster=(1, 1, 1),
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=self.occupancy,
            use_pdl=mBlockIdx is not None,
        )

    @cute.kernel
    def attn_kernel(
        self,
        problem_size: Tuple[Int32, Int32, Int32, Int32],
        tma_atom_Q: cute.CopyAtom,
        mQ: cute.Tensor,
        tma_atom_K: cute.CopyAtom,
        mK: cute.Tensor,
        tma_atom_V: cute.CopyAtom,
        mV: cute.Tensor,
        tma_atom_O: cute.CopyAtom,
        mO_partial_tma: cute.Tensor,
        mO_partial_raw: cute.Tensor,  # direct stores; mO_partial_tma holds TMA coordinates
        mLSE_partial: cute.Tensor | None,
        mStartEnd: cute.Tensor | None,
        mBlockCnt: cute.Tensor | None,
        mBlockIdx: cute.Tensor | None,
        mSeqUsedK: cute.Tensor | None,
        mPageTable: cute.Tensor | None,
        mNumSplitsDynamic: cute.Tensor | None,
        softmax_scale_log2: Float32,
        mask_row: Int32,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sVt_layout: cute.ComposedLayout,
        sO_layout: cute.ComposedLayout,
        sP_layout: cute.ComposedLayout,
        num_splits: Int32,
    ):
        """One split CTA: its contiguous run of KV blocks (or of block-list entries) of bh_idx."""
        # Warp roles (see header): WG0 = MMA + TMA warps, WG1 / WG2 = softmax for even / odd KV
        # blocks, WG3 = correction. Warp 2 publishes this split's LSE and reciprocal row sum.
        mma_qk_warp_id, mma_pv_warp_id, load_kv_warp_id, load_q_store_o_warp_id = 0, 1, 2, 3
        softmax_wg_ids = tuple(range(1, 1 + NUM_SOFTMAX_WG))
        correction_wg_id = 1 + NUM_SOFTMAX_WG
        split_stats_warp_id = load_kv_warp_id
        thread_id, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        lane_idx = cute.arch.lane_idx()
        wg_idx = warp_idx // WARPS_PER_WG
        wg_warp_idx = warp_idx % WARPS_PER_WG

        _, head_split_idx, batch_idx = cute.arch.block_idx()
        kv_head_idx = head_split_idx // num_splits
        split_id = head_split_idx - kv_head_idx * num_splits

        hdim = self.head_dim
        hdim_v = self.head_dim_v
        _b, num_heads, num_heads_kv, seqlen_k = problem_size
        bh_idx = batch_idx * num_heads_kv + kv_head_idx
        # Per-batch KV length: the static problem length, or seqused_k[b] when the caller
        # provides one (dense or paged). Tokens at or past it are masked and, on the paged
        # path, their pages are never loaded.
        seq_len_b = seqlen_k
        if const_expr(mSeqUsedK is not None):
            seq_len_b = mSeqUsedK[batch_idx]
        num_units = cute.ceil_div(seq_len_b, SWAPAB_SKIP_UNIT)
        tile_n = self.tile_n
        tile_hq = self.tile_hq
        qhead_per_kvhead = num_heads // num_heads_kv
        # Column mask: the query is row mask_row. Query head h reads mask head
        # h // mask_head_group; without per_head, all heads of this KV head share mask_head_base.
        mask_head_group = Int32(1)
        mask_head_base = Int32(0)
        if const_expr(mStartEnd is not None):
            mask_head_group = num_heads // mStartEnd.shape[1]
            mask_head_base = kv_head_idx * qhead_per_kvhead // mask_head_group
        # This split's KV blocks are [n_block_min, n_block_max): the blocks are dealt out in
        # equal contiguous runs, as in upstream BlockInfo.get_n_block_min_max. With a block
        # list, block j holds the units at list positions j * units_per_block + i.
        # The block-list count is read only after the on-chip prologue (see the PDL wait below).
        # Per-batch split count (#45): the grid keeps num_splits (the maximum) and batch b
        # uses num_splits_dynamic[b] of them; splits at or past it get an empty range.
        units_per_block = tile_n // SWAPAB_SKIP_UNIT
        num_listed = Int32(0)
        oob_unit = num_units
        num_splits_b = num_splits
        if const_expr(mBlockIdx is None):
            if const_expr(mNumSplitsDynamic is not None):
                num_splits_b = mNumSplitsDynamic[batch_idx]
            n_block_min, n_block_max, num_n_blocks = _split_block_range(
                cute.ceil_div(seq_len_b, tile_n), split_id, num_splits_b
            )
            is_empty_split = num_n_blocks == 0
        # Q, O and LSE rows are the sequence index.
        num_q_rows = mO_partial_raw.shape[1]
        q_row = batch_idx
        q_bh_idx = bh_idx

        mma_tiler = self.mma_tiler
        tile_m, tile_hq_mma, tile_k = self.mma_tiler
        num_s_subtiles, num_k_tiles, num_d_tiles, num_pv_k_tiles = self.subtile_counts

        mma_kq_instr_shape_k = cute.size(tiled_mma_qk.shape_mnk, mode=[2])
        mma_vp_instr_shape_k = cute.size(tiled_mma_pv.shape_mnk, mode=[2])
        qk_k_blocks = tile_k // mma_kq_instr_shape_k
        pv_k_blocks = tile_k // mma_vp_instr_shape_k

        # The largest head tile needs redistribution for its softmax / correction row state.
        # Keep sufficient registers in the MMA / load warps as well.
        setmaxreg = tile_hq > 16
        # Per-thread setmaxnreg targets: one MMA / load WG, two softmax WGs and one correction WG
        # together use the whole register file.
        num_regs_mma_load, num_regs_softmax, num_regs_correction = 64, 120, 208

        if warp_idx == load_kv_warp_id:
            for tma_atom in (tma_atom_Q, tma_atom_K, tma_atom_V, tma_atom_O):
                cpasync.prefetch_descriptor(tma_atom)

        storage = SmemAllocator().allocate(self.shared_storage)
        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sP = storage.sP.get_tensor(sP_layout.outer, swizzle=sP_layout.inner)

        sK = storage.sKV.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        sVt_iter = cute.recast_ptr(sK.iterator, sVt_layout.inner, dtype=self.dtype)
        sVt = cute.make_tensor(sVt_iter, sVt_layout.outer)

        # The O staging tile is only written after the last PV MMA, so it can reuse the ring.
        o_dtype = mO_partial_raw.element_type
        sO_iter = cute.recast_ptr(sK.iterator, sO_layout.inner, dtype=o_dtype)
        sO = cute.make_tensor(sO_iter, sO_layout.outer)

        thr_mma_pv = tiled_mma_pv.get_slice(0)
        thr_mma_qk = tiled_mma_qk.get_slice(0)

        # TMEM columns: S (stages x sub-tiles) first, L (one row-sum mailbox per softmax WG),
        # then O (slots x head_dim_v tiles). The views are built here for their layouts and
        # rebased onto the allocator's pointer once it is known.
        acc_shape = thr_mma_qk.partition_shape_C(mma_tiler[:2])
        num_s_cols = self.num_stages_s * num_s_subtiles * tile_hq_mma
        num_l_cols = self.num_stages_l * tile_hq_mma
        num_o_cols = self.num_stages_o * num_d_tiles * tile_hq_mma
        num_tmem_cols = cute.arch.get_max_tmem_alloc_cols("sm_100")
        if const_expr(self.occupancy > 1):
            num_tmem_cols = max(32, 1 << (num_s_cols + num_l_cols + num_o_cols - 1).bit_length())
            assert num_tmem_cols * self.occupancy <= _TMEM_COLS
        assert num_s_cols + num_l_cols + num_o_cols <= num_tmem_cols
        # ((tile_m, tile_hq_mma), 1, 1, subtile, stage)
        tStS = thr_mma_qk.make_fragment_C(
            cute.append(cute.append(acc_shape, num_s_subtiles), self.num_stages_s)
        )
        tLtL = thr_mma_qk.make_fragment_C(cute.append(acc_shape, self.num_stages_l))
        tLtL = cute.make_tensor(tStS.iterator + num_s_cols, tLtL.layout)
        # ((tile_m, tile_hq_mma), 1, 1, d tile, slot)
        tOtO = thr_mma_pv.make_fragment_C(
            cute.append(cute.append(acc_shape, num_d_tiles), self.num_stages_o)
        )
        tOtO = cute.make_tensor(tStS.iterator + num_s_cols + num_l_cols, tOtO.layout)

        tLtL_0 = tLtL[(None, None), 0, 0, 0]
        t2r_rep = tcgen05.Repetition(tile_hq)
        t2r_atom = cute.make_copy_atom(
            tcgen05.Ld32x32bOp(t2r_rep, tcgen05.Pack.NONE), self.qk_acc_dtype
        )
        r2t_atom = cute.make_copy_atom(
            tcgen05.St32x32bOp(t2r_rep, tcgen05.Unpack.NONE), self.qk_acc_dtype
        )
        r2s_atom_P = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.dtype)
        r2s_atom_O = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), o_dtype)
        tiled_t2r = tcgen05.make_tmem_copy(t2r_atom, tLtL_0)
        tiled_r2t = tcgen05.make_tmem_copy(r2t_atom, tLtL_0)
        tiled_r2s_O = cute.make_tiled_copy_D(r2s_atom_O, tiled_t2r)
        thr_in_wg = thread_id % WG_SIZE
        thr_t2r = tiled_t2r.get_slice(thr_in_wg)
        thr_r2t = tiled_r2t.get_slice(thr_in_wg)
        thr_r2s_O = tiled_r2s_O.get_slice(thr_in_wg)
        vec_layout = thr_t2r.partition_D(tLtL_0).layout
        vec_size = cute.cosize(vec_layout)
        assert vec_size == tile_hq_mma
        assert vec_layout.shape == ((tile_hq, 1), 1, tile_hq_mma // tile_hq)

        sRowMax = storage.sRowMax.get_tensor(vec_layout)
        sRowSumWarp = storage.sRowSumWarp.get_tensor(
            cute.make_layout(shape=(WARPS_PER_WG, tile_hq), stride=(tile_hq, 1))
        )
        lane_owns_head = lane_idx < tile_hq
        if warp_idx == load_q_store_o_warp_id:
            if lane_owns_head:
                sRowMax[lane_idx] = -Float32.inf

        # The KV ring carries K two blocks ahead of V: K(j) is loaded together with V(j - 2).
        # Both MMA warps read the ring with their own cursors: each skips the other's stages and
        # syncs on bar_pv_issued, so the cursors stay within one window and mbarrier phases are
        # never misread.
        kv_lookahead = 2

        # Named barriers (hardware barriers that only count arriving threads):
        #   bar_max_full / bar_max_empty   softmax WG <-> correction WG hand-off of the running max
        #   bar_sum_full / bar_sum_empty   row-sum mailbox hand-off, one pair per softmax WG
        #                                  (+phase)
        #   bar_max_turn                   token passed between the two softmax WGs (see softmax)
        #   bar_qk_issued / bar_pv_issued  keep the two MMA warps in K/V ring order
        #   bar_max_done / bar_sum_done    final max / sum ready for the split-stats warp
        #   bar_o_done                     final O staged in SMEM, ready for the TMA store
        B = NamedBarrierFwdSm100SwapAB
        # Softmax and correction wait at different sites: unaligned wait.
        bar_max_full = NamedBarrier(int(B.MaxFull), 2 * WG_SIZE)
        bar_max_empty = NamedBarrier(int(B.MaxEmpty), 2 * WG_SIZE)
        bar_sum_full = NamedBarrier(int(B.SumFullWG1), 2 * WG_SIZE)
        bar_sum_empty = NamedBarrier(int(B.SumEmptyWG1), 2 * WG_SIZE)
        bar_max_done = NamedBarrier(int(B.MaxDone), WG_SIZE + WARP_SIZE)
        bar_stats_ready = NamedBarrier(int(B.StatsReady), WG_SIZE + WARP_SIZE)
        bar_sum_done = NamedBarrier(int(B.SumDone), WG_SIZE + WARP_SIZE)
        bar_qk_issued = NamedBarrier(int(B.QKIssued), 2 * WARP_SIZE)
        bar_pv_issued = NamedBarrier(int(B.PVIssued), 2 * WARP_SIZE)
        bar_max_turn = NamedBarrier(int(B.MaxTurnWG1), NUM_SOFTMAX_WG * WG_SIZE)
        bar_o_done = NamedBarrier(int(B.ODone), WG_SIZE + WARP_SIZE)

        mbar_q_full = storage.mbar_q.data_ptr()
        if warp_idx == load_q_store_o_warp_id:
            q_tx_bytes = cute.size_in_bytes(self.dtype, sQ_layout)
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(mbar_q_full, 1)
                cute.arch.mbarrier_init_fence()
                cute.arch.mbarrier_arrive_and_expect_tx(mbar_q_full, q_tx_bytes)
        # Q: all query heads of this KV head, loaded once, one tile_k-dim chunk per stage. Only
        # warp 3 initializes and produces this barrier, so without a block list the load starts
        # before the CTA-wide prologue barrier; warp 0 waits on it only after that barrier.
        gQ = cute.local_tile(mQ, (tile_hq_mma, tile_k), (0, None, q_bh_idx))
        load_Q, _, _ = copy_utils.tma_get_copy_fn(
            tma_atom_Q, 0, cute.make_layout(1), thr_mma_qk.partition_B(gQ), sQ
        )
        if const_expr(mBlockIdx is None):
            if warp_idx == load_q_store_o_warp_id:
                if not is_empty_split:
                    for dk in cutlass.range_constexpr(num_k_tiles):
                        load_Q(src_idx=dk, dst_idx=dk, tma_bar_ptr=mbar_q_full)
        # mbarrier pipelines: KV (TMA -> MMA), S (QK MMA -> softmax), P (softmax -> PV MMA),
        # O (PV MMA -> correction). Arrival counts: one elected TMA arrive or tcgen05.commit per
        # MMA/TMA side, one arrive per thread on the softmax / correction side.
        one_thread = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        wg_threads = pipeline.CooperativeGroup(pipeline.Agent.Thread, WG_SIZE)
        kv_tx_bytes = cute.size_in_bytes(self.dtype, cute.select(sK_layout, mode=[0, 1, 2]))
        pipeline_kv = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.mbar_kv.data_ptr(),
            num_stages=self.num_stages_kv,
            producer_group=one_thread,
            consumer_group=one_thread,
            tx_count=kv_tx_bytes,
            defer_sync=True,
        )
        pipeline_s = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.mbar_s.data_ptr(),
            num_stages=self.num_stages_s,
            producer_group=one_thread,
            consumer_group=wg_threads,
            defer_sync=True,
        )
        pipeline_p = pipeline.PipelineAsyncUmma.create(
            barrier_storage=storage.mbar_p.data_ptr(),
            num_stages=self.num_stages_p,
            producer_group=wg_threads,
            consumer_group=one_thread,
            defer_sync=True,
        )
        pipeline_o = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.mbar_o.data_ptr(),
            num_stages=self.num_stages_o,
            producer_group=one_thread,
            consumer_group=wg_threads,
            defer_sync=True,
        )
        Producer, Consumer = pipeline.PipelineUserType.Producer, pipeline.PipelineUserType.Consumer

        # The pipelines initialize their mbarriers from warp 0, so warp 1 (PV MMA) allocates TMEM
        # meanwhile. The TmemPtr barrier publishes both the allocation and the mbarrier inits.
        tmem = TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=NamedBarrier(int(B.TmemPtr), NUM_WG * WG_SIZE),
            allocator_warp_id=mma_pv_warp_id,
        )
        tmem.allocate(num_tmem_cols)
        tmem.relinquish_alloc_permit()
        cute.arch.mbarrier_init_fence()
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(self.qk_acc_dtype)
        tStS = cute.make_tensor(tmem_ptr, tStS.layout)
        tLtL = cute.make_tensor(tmem_ptr + num_s_cols, tLtL.layout)
        tOtO = cute.make_tensor(tmem_ptr + num_s_cols + num_l_cols, tOtO.layout)

        if const_expr(mBlockIdx is not None):
            # Launched with PDL after the block-list kernels (or the caller's kernels, with a
            # prebuilt list): the on-chip prologue above overlaps them. Wait before the first
            # global read, the list count, and before loading Q.
            cute.arch.griddepcontrol_wait()
            num_listed = mBlockCnt[batch_idx, kv_head_idx, 0]
            if const_expr(mNumSplitsDynamic is not None):
                num_splits_b = mNumSplitsDynamic[batch_idx]
            n_block_min, n_block_max, num_n_blocks = _split_block_range(
                (num_listed + units_per_block - 1) // units_per_block, split_id, num_splits_b
            )
            is_empty_split = num_n_blocks == 0
            if warp_idx == load_q_store_o_warp_id:
                if not is_empty_split:
                    for dk in cutlass.range_constexpr(num_k_tiles):
                        load_Q(src_idx=dk, dst_idx=dk, tma_bar_ptr=mbar_q_full)

        # Every empty split must publish O=0 as well as LSE=-inf: the combine may load its O
        # even with a zero weight, and zero times an uninitialized NaN is NaN.
        if is_empty_split:
            if wg_idx == correction_wg_id:
                bar_max_done.arrive()
                if lane_idx < tile_hq:
                    sRowSumWarp[wg_warp_idx, lane_idx] = Float32(0.0)
                bar_sum_done.arrive()
            for o_idx in cutlass.range(thread_id, hdim_v * tile_hq, NUM_WG * WG_SIZE):
                head = o_idx // hdim_v
                if head < qhead_per_kvhead and q_row < num_q_rows:
                    global_head = kv_head_idx * qhead_per_kvhead + head
                    mO_partial_raw[split_id, q_row, global_head, o_idx % hdim_v] = (
                        mO_partial_raw.element_type(0.0)
                    )

        elif warp_idx == load_q_store_o_warp_id:
            if cutlass.const_expr(setmaxreg):
                cute.arch.setmaxregister_decrease(num_regs_mma_load)

            # Wait until the correction WG has staged the final O, then TMA-store this split.
            bar_o_done.arrive_and_wait()
            for dm in cutlass.range_constexpr(num_d_tiles):
                gO = cute.local_tile(
                    mO_partial_tma, (tile_m, tile_hq_mma), (dm, 0, q_bh_idx, split_id)
                )
                store_O, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_O, 0, cute.make_layout(1), sO[None, None, dm, 0], gO, single_stage=True
                )
                store_O()
            cute.arch.cp_async_bulk_commit_group()
            # The ring SMEM must stay valid until the bulk store has read it.
            cute.arch.cp_async_bulk_wait_group(0, read=True)

        elif warp_idx == load_kv_warp_id:
            if cutlass.const_expr(setmaxreg):
                cute.arch.setmaxregister_decrease(num_regs_mma_load)

            # Producer for the shared ring. Loop iteration s pushes K(s) and V(s - lookahead), so
            # the stream is K0 K1 K2 | V0 K3 V1 ... and finally the trailing V tiles. The
            # unit of block j, sub-tile i, is j * num_s_subtiles + i, looked up in the
            # block list if there is one. Paged KV: unit u is page page_table[b, u] (tile 0 of
            # it); units past the sequence length map to page -1, out of the pool, so TMA
            # zero-fills them instead of reading a page the sequence does not own.
            kv_tiler = (tile_m, tile_k)
            kv_state = pipeline.make_pipeline_state(Producer, self.num_stages_kv)
            kv_token = cutlass.Boolean(True)
            for s in cutlass.range(kv_lookahead + num_n_blocks):
                if s < num_n_blocks:
                    k_block = s + n_block_min
                    for sm in cutlass.range_constexpr(num_s_subtiles):
                        k_unit = k_block * num_s_subtiles + sm
                        if const_expr(mBlockIdx is not None):
                            k_unit = decode_kv_unit(
                                mBlockIdx, batch_idx, kv_head_idx, k_unit, num_listed, oob_unit
                            )
                        if const_expr(mPageTable is None):
                            gK = cute.local_tile(mK, (tile_m, hdim), (k_unit, 0, bh_idx))
                        else:
                            k_page = Int32(-1)
                            if k_unit < num_units:
                                k_page = mPageTable[batch_idx, k_unit]
                            # (page_size, d, H_kv, num_pages) -> tile 0 of this KV head's page.
                            gK = cute.local_tile(
                                mK[None, None, kv_head_idx, None],
                                (tile_m, hdim),
                                (0, 0, k_page),
                            )
                        gK_sub = cute.zipped_divide(gK, kv_tiler)[(None, None), (0, None)]
                        tKsK, tKgK = cpasync.tma_partition(
                            tma_atom_K,
                            0,
                            cute.make_layout(1),
                            cute.group_modes(sK, 0, 3),
                            cute.group_modes(thr_mma_qk.partition_A(gK_sub), 0, 3),
                        )
                        for dk in cutlass.range_constexpr(num_k_tiles):
                            kv_state, kv_token = load_kv_stage(
                                pipeline_kv, kv_state, kv_token, tma_atom_K, tKgK[None, dk], tKsK
                            )

                if s >= kv_lookahead:
                    v_block = s - kv_lookahead + n_block_min
                    for sk in cutlass.range_constexpr(num_pv_k_tiles):
                        # Consecutive tile_k-token V^T tiles of the block's units.
                        if cutlass.const_expr(sk % (tile_m // tile_k) == 0):
                            v_unit = v_block * num_s_subtiles + sk // (tile_m // tile_k)
                            if const_expr(mBlockIdx is not None):
                                v_unit = decode_kv_unit(
                                    mBlockIdx, batch_idx, kv_head_idx, v_unit, num_listed, oob_unit
                                )
                            if const_expr(mPageTable is None):
                                gVt_blk = cute.zipped_divide(
                                    cute.local_tile(mV, (hdim_v, tile_m), (0, v_unit, bh_idx)),
                                    kv_tiler,
                                )
                            else:
                                v_page = Int32(-1)
                                if v_unit < num_units:
                                    v_page = mPageTable[batch_idx, v_unit]
                                # (d_v, page_size, H_kv, num_pages) -> tile 0 of this page.
                                gVt_blk = cute.zipped_divide(
                                    cute.local_tile(
                                        mV[None, None, kv_head_idx, None],
                                        (hdim_v, tile_m),
                                        (0, 0, v_page),
                                    ),
                                    kv_tiler,
                                )
                        gVt_sub = gVt_blk[(None, None), (None, sk % (tile_m // tile_k))]
                        tVsV, tVgV = cpasync.tma_partition(
                            tma_atom_V,
                            0,
                            cute.make_layout(1),
                            cute.group_modes(sVt, 0, 3),
                            cute.group_modes(thr_mma_pv.partition_A(gVt_sub), 0, 3),
                        )
                        for dm in cutlass.range_constexpr(num_d_tiles):
                            kv_state, kv_token = load_kv_stage(
                                pipeline_kv, kv_state, kv_token, tma_atom_V, tVgV[None, dm], tVsV
                            )
            # No producer_tail: the cluster is one CTA, and each MMA warp issues its ring releases
            # before the S / O commits that the softmax / correction WGs wait on before exit.

        elif warp_idx == mma_qk_warp_id:
            if cutlass.const_expr(setmaxreg):
                cute.arch.setmaxregister_decrease(num_regs_mma_load)

            pv_tiles_per_iter = num_d_tiles * num_pv_k_tiles
            mma_atom = cute.make_mma_atom(tiled_mma_qk.op)
            tCrK = tiled_mma_qk.make_fragment_A(sK)
            tCrQ = tiled_mma_qk.make_fragment_B(sQ)

            # QK MMA warp: S^T(j) = K(j) Q^T, one MMA per sub-tile of block j.
            cute.arch.mbarrier_wait(mbar_q_full, phase=0)

            kv_state = pipeline.make_pipeline_state(Consumer, self.num_stages_kv)
            s_state = pipeline.make_pipeline_state(Producer, self.num_stages_s)
            for s in cutlass.range(num_n_blocks):
                s_token = pipeline_s.producer_try_acquire(s_state)

                # Skip the PV warp's V(j-3) stages and wait until it has issued them.
                if s >= kv_lookahead + 1:
                    for _ in cutlass.range_constexpr(pv_tiles_per_iter):
                        kv_state.advance()
                bar_pv_issued.arrive_and_wait()
                k_token = pipeline_kv.consumer_try_wait(kv_state)

                pipeline_s.producer_acquire(s_state, s_token)
                for sm in cutlass.range_constexpr(num_s_subtiles):
                    mma_atom.set(tcgen05.Field.ACCUMULATE, False)
                    tStS_slice = tStS[None, None, None, sm, s_state.index]
                    for dk in cutlass.range_constexpr(num_k_tiles):
                        pipeline_kv.consumer_wait(kv_state, k_token)
                        k_stage_idx = kv_state.index
                        for instr_idx in cutlass.range_constexpr(qk_k_blocks):
                            k_slice = tCrK[None, None, instr_idx, k_stage_idx]
                            q_slice = tCrQ[None, None, instr_idx, dk]
                            cute.gemm(mma_atom, tStS_slice, k_slice, q_slice, tStS_slice)
                            mma_atom.set(tcgen05.Field.ACCUMULATE, True)
                        pipeline_kv.consumer_release(kv_state)
                        kv_state.advance()
                        if sm == num_s_subtiles - 1 and dk == num_k_tiles - 1:
                            bar_qk_issued.arrive()
                        else:
                            k_token = pipeline_kv.consumer_try_wait(kv_state)
                pipeline_s.producer_commit(s_state)
                s_state.advance()

            for _ in cutlass.range_constexpr(kv_lookahead):
                bar_pv_issued.arrive_and_wait()
                bar_qk_issued.arrive()

        elif warp_idx == mma_pv_warp_id:
            if cutlass.const_expr(setmaxreg):
                cute.arch.setmaxregister_decrease(num_regs_mma_load)

            qk_tiles_per_iter = num_s_subtiles * num_k_tiles
            mma_atom = cute.make_mma_atom(tiled_mma_pv.op)
            # The correction WG zero-fills both O slots, so every PV MMA accumulates.
            mma_atom.set(tcgen05.Field.ACCUMULATE, True)
            tCrVt = tiled_mma_pv.make_fragment_A(sVt)
            tCrP = tiled_mma_pv.make_fragment_B(sP)

            # PV MMA warp: O^T(slot j % 2) += V^T(j) P^T(j). Warm-up skips K(0), K(1) and
            # releases the QK warp's first bar_pv_issued waits (see kv_lookahead).
            kv_state = pipeline.make_pipeline_state(Consumer, self.num_stages_kv)
            p_state = pipeline.make_pipeline_state(Consumer, self.num_stages_p)
            o_state = pipeline.make_pipeline_state(Producer, self.num_stages_o)
            bar_pv_issued.arrive()
            for prefetch_iter in cutlass.range_constexpr(kv_lookahead):
                if prefetch_iter < num_n_blocks:
                    for _ in cutlass.range_constexpr(qk_tiles_per_iter):
                        kv_state.advance()
                bar_qk_issued.arrive_and_wait()
                bar_pv_issued.arrive()

            for s in cutlass.range(num_n_blocks):
                if s + kv_lookahead < num_n_blocks:
                    for _ in cutlass.range_constexpr(qk_tiles_per_iter):
                        kv_state.advance()
                bar_qk_issued.arrive_and_wait()

                p_token = pipeline_p.consumer_try_wait(p_state)
                v_token = pipeline_kv.consumer_try_wait(kv_state)
                o_token = pipeline_o.producer_try_acquire(o_state)

                pipeline_p.consumer_wait(p_state, p_token)
                p_idx = p_state.index
                # The O slot for block j is free once the correction WG has rescaled what
                # PV(j - 2) left there.
                pipeline_o.producer_acquire(o_state, o_token)
                o_idx = o_state.index
                for dk in cutlass.range_constexpr(num_pv_k_tiles):
                    for dm in cutlass.range_constexpr(num_d_tiles):
                        pipeline_kv.consumer_wait(kv_state, v_token)
                        v_stage_idx = kv_state.index
                        tOtO_slot = tOtO[None, None, None, dm, o_idx]
                        for instr_idx in cutlass.range_constexpr(pv_k_blocks):
                            v_slice = tCrVt[None, None, instr_idx, v_stage_idx]
                            p_slice = tCrP[None, None, dk * pv_k_blocks + instr_idx, p_idx]
                            cute.gemm(mma_atom, tOtO_slot, v_slice, p_slice, tOtO_slot)
                        if dm == num_d_tiles - 1 and dk == num_pv_k_tiles - 1:
                            bar_pv_issued.arrive()
                        pipeline_kv.consumer_release(kv_state)
                        kv_state.advance()
                        if dm != num_d_tiles - 1 or dk != num_pv_k_tiles - 1:
                            v_token = pipeline_kv.consumer_try_wait(kv_state)
                pipeline_p.consumer_release(p_state)
                p_state.advance()
                pipeline_o.producer_commit(o_state)
                o_state.advance()
            # The correction tail always merges two slots; with a single block, commit an empty
            # second one.
            if num_n_blocks == 1:
                pipeline_o.producer_acquire(o_state)
                pipeline_o.producer_commit(o_state)
                o_state.advance()
            pipeline_o.producer_tail(o_state)

        elif wg_idx in softmax_wg_ids:
            if cutlass.const_expr(setmaxreg):
                cute.arch.setmaxregister_decrease(num_regs_softmax)

            # Softmax WGs. Each thread owns one token row of each sub-tile of the block, and
            # tile_hq query-head columns.
            tStS_0 = tStS[(None, None), 0, 0, None, 0]
            t2r_rep_S = tcgen05.Repetition(tile_hq)
            if cutlass.const_expr(tile_hq_mma == tile_hq):
                t2r_rep_S = tcgen05.Repetition(tile_hq * num_s_subtiles)
            t2r_atom_S = cute.make_copy_atom(
                tcgen05.Ld32x32bOp(t2r_rep_S, tcgen05.Pack.NONE),
                self.qk_acc_dtype,
            )
            tiled_t2r_S = tcgen05.make_tmem_copy(t2r_atom_S, tStS_0)
            tiled_r2s_P = cute.make_tiled_copy_D(r2s_atom_P, tiled_t2r_S)
            thr_t2r_S = tiled_t2r_S.get_slice(thr_in_wg)
            thr_r2s_P = tiled_r2s_P.get_slice(thr_in_wg)
            tSrS_layout = thr_t2r_S.partition_D(tStS_0).layout
            assert cute.cosize(tSrS_layout) == tile_hq_mma * num_s_subtiles

            tSrP_layout = cute.make_layout(
                (*vec_layout.shape, num_s_subtiles),
                stride=(*vec_layout.stride, vec_size),
            )
            scores_layout = cute.make_layout(
                shape=(tile_hq, num_s_subtiles),
                stride=(1, vec_size),
            )
            # (token row, head, subtile) -> one P stage's logical (head, token) index.
            sP_view_layout = cute.make_layout(
                shape=(tile_m, tile_hq_mma, num_s_subtiles),
                stride=(tile_hq_mma, 1, tile_m * tile_hq_mma),
            )

            tSrS = cute.make_rmem_tensor(tSrS_layout, self.qk_acc_dtype)
            tSrP = cute.make_rmem_tensor(tSrP_layout, self.dtype)
            tLrL = cute.make_rmem_tensor(vec_layout, self.qk_acc_dtype)
            row_max_r = cute.make_rmem_tensor(vec_layout, self.qk_acc_dtype)
            scores_r = cute.make_tensor(tSrS.iterator, scores_layout)
            probs_r = cute.make_tensor(tSrP.iterator, scores_layout)
            row_sum_valid = tLrL[(None, 0), 0, 0]
            row_max_valid = row_max_r[(None, 0), 0, 0]

            phase = wg_idx - softmax_wg_ids[0]
            tLtL_cur = tLtL[(None, None), 0, 0, phase]

            # WG2 handles the odd blocks: shift its pipeline cursors by one block and give WG1
            # the first turn on the shared running max.
            s_state = pipeline.make_pipeline_state(Consumer, self.num_stages_s)
            p_state = pipeline.make_pipeline_state(Producer, self.num_stages_p)
            if phase == 1:
                s_state.advance()
                p_state.advance()
                bar_max_turn.arrive_w_index(phase ^ 1)
                if num_n_blocks == 1:
                    bar_sum_full.arrive_w_index(phase)

            for n_iter in cutlass.range(phase, num_n_blocks, NUM_SOFTMAX_WG):
                s = n_iter
                s_token = pipeline_s.consumer_try_wait(s_state)
                p_token = pipeline_p.producer_try_acquire(p_state)

                # Column mask, shared mask head: read this thread's token bounds while S(j) is
                # still in flight. Rows past seq_len_b read the last token; the tail mask below
                # hides them anyway.
                # Token of this thread's row in each unit of block s.
                kv_block = s + n_block_min
                unit_token = []
                for sm in cutlass.range_constexpr(num_s_subtiles):
                    unit = kv_block * num_s_subtiles + sm
                    if const_expr(mBlockIdx is not None):
                        unit = decode_kv_unit(
                            mBlockIdx, batch_idx, kv_head_idx, unit, num_listed, oob_unit
                        )
                    unit_token.append(unit * tile_m + thr_in_wg)
                row_masked = []
                if const_expr(mStartEnd is not None and not self.mask_per_head):
                    for sm in cutlass.range_constexpr(num_s_subtiles):
                        token = cutlass.min(unit_token[sm], seq_len_b - 1)
                        row_masked.append(
                            flashmask_col_masked(
                                mStartEnd,
                                batch_idx,
                                mask_head_base,
                                token,
                                mask_row,
                                self.mask_num_vecs,
                                self.mask_is_causal,
                            )
                        )

                pipeline_s.consumer_wait(s_state, s_token)
                row_max_r.fill(-Float32.inf)
                tStS_cur = tStS[(None, None), 0, 0, None, s_state.index]
                cute.copy(tiled_t2r_S, thr_t2r_S.partition_S(tStS_cur), tSrS)
                cute.arch.fence_view_async_tmem_load()
                pipeline_s.consumer_release(s_state)
                s_state.advance()

                # Mask the token rows past seq_len_b (the last unit, and the padding unit).
                for sm in cutlass.range_constexpr(num_s_subtiles):
                    if unit_token[sm] >= seq_len_b:
                        scores_r[None, sm].fill(-Float32.inf)

                if const_expr(mStartEnd is not None):
                    if const_expr(not self.mask_per_head):
                        for sm in cutlass.range_constexpr(num_s_subtiles):
                            if row_masked[sm]:
                                scores_r[None, sm].fill(-Float32.inf)
                    else:
                        # Query head h uses mask head h // (num_heads / H_m). Padding columns past
                        # qhead_per_kvhead are never stored; clamping keeps their reads in bounds.
                        for sm in cutlass.range_constexpr(num_s_subtiles):
                            token = cutlass.min(unit_token[sm], seq_len_b - 1)
                            for jj in cutlass.range_constexpr(tile_hq):
                                q_head = cutlass.min(
                                    kv_head_idx * qhead_per_kvhead + jj, num_heads - 1
                                )
                                if flashmask_col_masked(
                                    mStartEnd,
                                    batch_idx,
                                    q_head // mask_head_group,
                                    token,
                                    mask_row,
                                    self.mask_num_vecs,
                                    self.mask_is_causal,
                                ):
                                    scores_r[jj, sm] = -Float32.inf

                scores = scores_r.load()
                row_max_valid.store(
                    scores.reduce(
                        cute.ReductionOp.MAX,
                        -Float32.inf,
                        (None, 0),
                    ).reshape((tile_hq,))
                )

                # Block max per head: reduce over this warp's rows; lane h then holds head h's max.
                lane_max = -Float32.inf
                for j in cutlass.range_constexpr(tile_hq):
                    row_max_r[j] = warp_reduce_max(row_max_r[j])
                    if j == lane_idx:
                        lane_max = row_max_r[j]
                lane_max *= softmax_scale_log2

                # M(j) = max(M(j-1), blockmax(j)); WG1/WG2 own even/odd j. Arrival counts alone
                # can admit both WGs and make correction miss M(j). bar_max_turn: take -> wait
                # for correction to read M(j-1) -> atomic max -> publish -> pass.
                bar_max_turn.arrive_and_wait_w_index(phase)
                bar_max_empty.arrive_and_wait()
                if lane_owns_head:
                    sRowMax_ptr = sRowMax.iterator + sRowMax.layout(lane_idx)
                    smem_atomic_max(sRowMax_ptr, lane_max)

                bar_max_full.arrive_and_wait_unaligned()
                cute.autovec_copy(sRowMax, row_max_r)
                bar_max_turn.arrive_w_index(phase ^ 1)
                if const_expr(mStartEnd is not None):
                    # A head with every key so far masked has M(j) = -inf: subtract 0 instead,
                    # so its P is exp2(-inf) = 0 rather than NaN.
                    for j in cutlass.range_constexpr(tile_hq):
                        if row_max_valid[j] == -Float32.inf:
                            row_max_valid[j] = Float32(0.0)

                # P = exp2(S * scale * log2(e) - M(j)) in bf16 / fp16 into SMEM for the PV MMA.
                pipeline_p.producer_acquire(p_state, p_token)
                row_max_tile = row_max_valid.load().reshape((tile_hq, 1))
                probs = exp2_fast(softmax_scale_log2 * scores - row_max_tile)

                if cutlass.const_expr(tile_hq < tile_hq_mma):
                    tSrP.fill(self.dtype(0))

                probs_r.store(probs.to(self.dtype))
                sP_view = cute.composition(sP[None, None, None, p_state.index], sP_view_layout)
                cute.autovec_copy(tSrP, thr_r2s_P.partition_D(sP_view))
                cute.arch.fence_view_async_shared()
                pipeline_p.producer_commit(p_state)
                p_state.advance()

                # This WG's TMEM mailbox carries this thread's two-token sums per head, not P.
                # Cross-thread reduction runs once in the correction tail.
                row_sum_valid.store(
                    probs.reduce(
                        cute.ReductionOp.ADD,
                        Float32(0.0),
                        (None, 0),
                    ).reshape((tile_hq,))
                )

                bar_sum_empty.arrive_and_wait_w_index(phase)
                cute.copy(tiled_r2t, tLrL, thr_r2t.partition_D(tLtL_cur))
                cute.arch.fence_view_async_tmem_store()
                bar_sum_full.arrive_w_index(phase)

                # Skip the block that the other softmax WG handles.
                s_state.advance()
                p_state.advance()

        elif wg_idx == correction_wg_id:
            if cutlass.const_expr(setmaxreg):
                cute.arch.setmaxregister_increase(num_regs_correction)

            # Correction WG. Thread t owns TMEM lane t (one head_dim_v row of O, all heads) and
            # keeps one row-sum accumulator per O slot.
            l_slot_shape = (*vec_layout.shape, self.num_stages_o)
            l_slot_stride = (*vec_layout.stride, vec_size)
            row_sum_slot_layout = cute.make_layout(l_slot_shape, stride=l_slot_stride)
            tLrL = cute.make_rmem_tensor(vec_layout, self.qk_acc_dtype)
            row_sum_final = cute.make_rmem_tensor(vec_layout, self.qk_acc_dtype)
            acc_scale = cute.make_rmem_tensor(vec_layout, self.qk_acc_dtype)
            row_sum_acc = cute.make_rmem_tensor(row_sum_slot_layout, self.qk_acc_dtype)

            row_sum_acc.fill(Float32(0.0))

            zeros_r = cute.make_rmem_tensor(vec_layout, self.qk_acc_dtype)
            zeros_r.fill(Float32(0.0))
            for phase in cutlass.range_constexpr(self.num_stages_o):
                for dm in cutlass.range_constexpr(num_d_tiles):
                    tOtO_slot = tOtO[(None, None), 0, 0, dm, phase]
                    cute.copy(tiled_r2t, zeros_r, thr_r2t.partition_D(tOtO_slot))
            tLtL_1 = tLtL[(None, None), 0, 0, 1]
            cute.copy(tiled_r2t, zeros_r, thr_r2t.partition_D(tLtL_1))
            cute.arch.fence_view_async_tmem_store()

            # Initial tokens: the running max and both row-sum mailboxes start out empty.
            bar_max_empty.arrive()
            for phase in cutlass.range_constexpr(NUM_SOFTMAX_WG):
                bar_sum_empty.arrive_w_index(phase)

            # Read M(0) and M(1) up front: rescaling slot j needs M(j) and M(j + 2).
            row_max_prev2, row_max_prev = -Float32.inf, -Float32.inf
            for preload_idx in cutlass.range_constexpr(self.num_stages_o):
                row_max_prev2 = row_max_prev
                if not (preload_idx == 1 and num_n_blocks == 1):
                    bar_max_full.arrive_and_wait_unaligned()
                    row_max_prev = Float32(0.0)
                    if lane_owns_head:
                        row_max_prev = sRowMax[lane_idx]
                    bar_max_empty.arrive()

            # --- main loop, block s uses O slot s % 2 ---
            # 1. collect the partial row sums of block s
            # 2. read M(s + 2); the last time round tell the split-stats warp the max is final
            # 3. once PV(s) has landed, scale the slot by exp2(M(s) - M(s + 2)) and release it,
            #    so PV(s + 2) can accumulate on top.
            o_state = pipeline.make_pipeline_state(Consumer, self.num_stages_o)
            phase = 0
            for s in cutlass.range(num_n_blocks - self.num_stages_o, unroll=self.num_stages_o):
                bar_sum_full.arrive_and_wait_w_index(phase)
                tLtL_cur = tLtL[(None, None), 0, 0, phase]
                cute.copy(tiled_t2r, thr_t2r.partition_S(tLtL_cur), tLrL)
                cute.arch.fence_view_async_tmem_load()
                bar_sum_empty.arrive_w_index(phase)

                bar_max_full.arrive_and_wait_unaligned()
                if s == num_n_blocks - self.num_stages_o - 1:
                    bar_max_done.arrive()
                row_max_cur = Float32(0.0)
                if lane_owns_head:
                    row_max_cur = sRowMax[lane_idx]
                bar_max_empty.arrive()

                pipeline_o.consumer_wait(o_state, pipeline_o.consumer_try_wait(o_state))

                acc_scale_lane = exp2_fast(row_max_prev2 - row_max_cur)
                if const_expr(mStartEnd is not None):
                    # M(s + 2) = -inf means every key so far is masked and the slot holds 0;
                    # keep the scale finite instead of exp2(-inf + inf) = NaN.
                    if row_max_cur == -Float32.inf:
                        acc_scale_lane = Float32(1.0)
                for gi in cutlass.range_constexpr(tile_hq):
                    acc_scale[gi] = cute.arch.shuffle_sync(acc_scale_lane, gi)

                for dm in cutlass.range_constexpr(num_d_tiles):
                    tOtO_slot = tOtO[(None, None), 0, 0, dm, phase]
                    tOrO = cute.make_rmem_tensor(vec_layout, self.qk_acc_dtype)
                    cute.copy(tiled_t2r, thr_t2r.partition_S(tOtO_slot), tOrO)
                    cute.arch.fence_view_async_tmem_load()
                    tOrO.store(tOrO.load() * acc_scale.load())
                    cute.copy(tiled_r2t, tOrO, thr_r2t.partition_D(tOtO_slot))
                cute.arch.fence_view_async_tmem_store()
                pipeline_o.consumer_release(o_state)
                o_state.advance()

                row_sum_acc_cur = row_sum_acc[None, None, None, phase]
                tLrL.store(tLrL.load() * acc_scale.load())
                row_sum_acc_cur.store(row_sum_acc_cur.load() * acc_scale.load() + tLrL.load())

                row_max_prev2, row_max_prev = row_max_prev, row_max_cur
                phase ^= 1

            if num_n_blocks <= self.num_stages_o:
                bar_max_done.arrive()

            # --- tail: merge the two slots ---
            # Blocks n-2, n-1 are relative to M(n-2) and M(n-1). Bring slot (n-2) up to the final
            # max with exp2(M(n-2) - M(n-1)) and add: O = O[n-1] + c * O[n-2], row sums alike.
            acc_scale_lane = exp2_fast(row_max_prev2 - row_max_prev)
            if const_expr(mStartEnd is not None):
                if row_max_prev == -Float32.inf:
                    acc_scale_lane = Float32(1.0)
            for gi in cutlass.range_constexpr(tile_hq):
                acc_scale[gi] = cute.arch.shuffle_sync(acc_scale_lane, gi)

            tail_phase = num_n_blocks % self.num_stages_o
            for phase in cutlass.range_constexpr(self.num_stages_o):
                if tail_phase == phase:
                    final_phase = phase ^ 1

                    bar_sum_full.arrive_and_wait_w_index(phase)
                    tLtL_cur = tLtL[(None, None), 0, 0, phase]
                    cute.copy(tiled_t2r, thr_t2r.partition_S(tLtL_cur), tLrL)
                    cute.arch.fence_view_async_tmem_load()
                    bar_sum_empty.arrive_w_index(phase)

                    bar_sum_full.arrive_and_wait_w_index(final_phase)
                    tLtL_last = tLtL[(None, None), 0, 0, final_phase]
                    cute.copy(tiled_t2r, thr_t2r.partition_S(tLtL_last), row_sum_final)
                    cute.arch.fence_view_async_tmem_load()
                    bar_sum_empty.arrive_w_index(final_phase)

                    tLrL.store(tLrL.load() + row_sum_acc[None, None, None, phase].load())
                    row_sum_final.store(
                        row_sum_final.load()
                        + row_sum_acc[None, None, None, final_phase].load()
                        + acc_scale.load() * tLrL.load()
                    )

            # Sum the row sums across the warp's lanes, then per warp into SMEM for split stats.
            for gi in cutlass.range_constexpr(tile_hq):
                row_sum_final[gi] = cute.arch.warp_reduction_sum(row_sum_final[gi])
                if lane_idx == 0:
                    sRowSumWarp[wg_warp_idx, gi] = row_sum_final[gi]
            bar_sum_done.arrive()

            # StatsReady publishes 1/L after the final max and the per-warp sums are consumed.
            bar_stats_ready.arrive_and_wait()
            cute.autovec_copy(sRowMax, tLrL)

            o_slot_shape = (*vec_layout.shape, num_d_tiles, self.num_stages_o)
            o_slot_stride = (
                *vec_layout.stride,
                vec_size,
                num_d_tiles * vec_size,
            )
            tOrO_slot_layout = cute.make_layout(o_slot_shape, stride=o_slot_stride)
            tOrO_tail = cute.make_rmem_tensor(tOrO_slot_layout, self.qk_acc_dtype)
            for s in cutlass.range_constexpr(self.num_stages_o):
                phase = tail_phase ^ s
                pipeline_o.consumer_wait(o_state, pipeline_o.consumer_try_wait(o_state))
                cute.copy(
                    tiled_t2r,
                    thr_t2r.partition_S(tOtO[(None, None), 0, 0, None, phase]),
                    tOrO_tail[None, None, None, None, s],
                )
                cute.arch.fence_view_async_tmem_load()
                pipeline_o.consumer_release(o_state)
                o_state.advance()

            # Normalize the merged O in fp32, then stage it for warp 3's TMA store.
            tOrO_tail0 = tOrO_tail[None, None, None, None, 0].load()
            tOrO_tail1 = tOrO_tail[None, None, None, None, 1].load()
            tOrO_tail0 = tOrO_tail0.reshape((vec_size, num_d_tiles))
            tOrO_tail1 = tOrO_tail1.reshape((vec_size, num_d_tiles))
            tOrO_final = tOrO_tail1 + acc_scale.load().reshape((vec_size, 1)) * tOrO_tail0
            tOrO_final *= tLrL.load().reshape((vec_size, 1))
            tOsO = thr_r2s_O.partition_D(sO[None, None, None, 0])
            tOsO.store(tOrO_final.to(o_dtype).reshape(tOsO.shape))

            cute.arch.fence_view_async_shared()
            bar_o_done.arrive()

        # After its loads, warp 2 normalizes this split's statistics.
        if warp_idx == split_stats_warp_id:
            self.store_split_stats(
                tile_hq,
                (split_id, q_bh_idx),
                lane_idx,
                is_empty_split,
                bar_max_done,
                bar_sum_done,
                bar_stats_ready,
                sRowMax,
                sRowSumWarp,
                mLSE_partial,
            )

        # Free TMEM once every role is done with it.
        cute.arch.sync_threads()
        tmem.free(tmem_ptr)

    @staticmethod
    @cute.jit
    def store_split_stats(
        tile_hq: int,
        block_idx: Tuple[Int32, Int32],
        lane_idx: Int32,
        is_empty_split: cutlass.Boolean,
        bar_max_done,
        bar_sum_done,
        bar_stats_ready,
        sRowMax: cute.Tensor,
        sRowSumWarp: cute.Tensor,
        mLSE_partial: cute.Tensor | None,
    ):
        split_id, bh_idx = block_idx
        lane_owns_head = lane_idx < tile_hq
        row_max_lane = -Float32.inf

        cute.arch.fence_acq_rel_cta()
        bar_max_done.arrive_and_wait()
        if lane_owns_head:
            row_max_lane = sRowMax[lane_idx]

        bar_sum_done.arrive_and_wait()
        if lane_owns_head:
            row_sum_w01 = sRowSumWarp[0, lane_idx] + sRowSumWarp[1, lane_idx]
            row_sum_w23 = sRowSumWarp[2, lane_idx] + sRowSumWarp[3, lane_idx]
            row_sum = row_sum_w01 + row_sum_w23
            inv_sum = Float32(0.0)
            if row_sum > Float32(0.0):
                inv_sum = cute.arch.rcp_approx(row_sum)
            sRowMax[lane_idx] = inv_sum
            if cutlass.const_expr(mLSE_partial is not None):
                if lane_idx < mLSE_partial.shape[1]:
                    lse = -Float32.inf
                    if row_sum > Float32(0.0):
                        lse = (row_max_lane + cute.math.log2(row_sum, fastmath=True)) * math.log(
                            2.0
                        )
                    num_heads_kv = mLSE_partial.shape[2][0]
                    bh_coord = (bh_idx % num_heads_kv, bh_idx // num_heads_kv)
                    if bh_coord[1] < mLSE_partial.shape[2][1]:
                        mLSE_partial[split_id, lane_idx, bh_coord] = lse

        # Empty CTAs skip StatsReady, as correction skips its wait there. bar_pv_issued,
        # bar_max_turn, bar_max_empty and bar_sum_empty end with unmatched arrivals; that is
        # harmless only because each CTA runs one tile and exits.
        if not is_empty_split:
            bar_stats_ready.arrive()

class FlashMaskDecodeSm100SwapAB:
    """Single-query decode with (B, S, H, D) tensors; the caller owns allocation and combine.

    Split outputs are normalized fp32 O and natural-log LSE; with one split the kernel writes
    the input-dtype output directly. A head whose keys are all masked gets O = 0, LSE = -inf.

    mask_num_vecs (0 = no mask), mask_is_causal and mask_per_head describe startend_row_indices
    as in _SwapABCore; the host picks them, they change the generated code. occupancy is the
    number of CTAs resident per SM (one of SWAPAB_OCCUPANCIES, see swapab_occupancy_fits).

    mBlockCnt / mBlockIdx (optional, shared mask head only) list the KV units of each
    (sequence, KV head) that are not fully masked, from build_decode_block_list. Each KV block
    takes consecutive list entries and splits are dealt out over these blocks, so fully masked
    units are never loaded.
    """

    def __init__(
        self,
        head_dim: int,
        head_dim_v: int,
        qhead_per_kvhead: int,
        is_split_kv: bool,
        mask_num_vecs: int = 0,
        mask_is_causal: bool = False,
        mask_per_head: bool = False,
        occupancy: int = 1,
    ):
        assert (head_dim, head_dim_v) in SWAPAB_HEAD_DIMS
        assert 1 <= qhead_per_kvhead <= SWAPAB_MAX_QHEAD_PER_KVHEAD
        assert swapab_tmem_cols(qhead_per_kvhead, head_dim_v) <= _TMEM_COLS
        assert swapab_occupancy_fits(qhead_per_kvhead, head_dim_v, occupancy)
        self.head_dim = head_dim
        self.head_dim_v = head_dim_v
        self.is_split_kv = is_split_kv
        self.tile_hq = swapab_tile_hq(qhead_per_kvhead)
        self.tile_n = SWAPAB_TILE_N
        self.mask_args = (mask_num_vecs, mask_is_causal, mask_per_head)
        self.occupancy = occupancy

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,  # (b, 1, h, d)
        mK: cute.Tensor,  # (b, s_k, h_k, d)
        mV: cute.Tensor,  # (b, s_k, h_k, d_v)
        mO: cute.Tensor,  # split: out_partial (ns, b, 1, h, d_v) fp32; else out (b, 1, h, d_v)
        mLSE: cute.Tensor | None,  # split: lse_partial (ns, b, h, 1); else lse (b, h, 1) | None
        mStartEnd: cute.Tensor | None,  # (b, h_m, s_k, num_vecs) int32 | None
        mBlockCnt: cute.Tensor | None,  # (b, h_k, 1) int32 | None
        mBlockIdx: cute.Tensor | None,  # (b, h_k, 1, n_units) int32 | None
        mSeqUsedK: cute.Tensor | None,  # (b,) int32 | None
        mPageTable: cute.Tensor | None,  # (b, max_pages) int32 | None
        mNumSplitsDynamic: cute.Tensor | None,  # (b,) int32 | None
        softmax_scale: Float32,
        mask_row: Int32,  # the query's mask row: s_k - 1 bottom-right, 0 as the main kernel
        stream: cuda.CUstream,
    ):
        assert len(mQ.shape) == 4 and len(mK.shape) == 4 and len(mV.shape) == 4
        assert mQ.element_type in (cutlass.Float16, cutlass.BFloat16)
        assert mQ.element_type == mK.element_type == mV.element_type
        if const_expr(self.is_split_kv):
            assert mO.element_type == Float32 and mLSE is not None
        else:
            assert mO.element_type == mQ.element_type
        if const_expr(mLSE is not None):
            assert mLSE.element_type == Float32
        assert (mStartEnd is not None) == (self.mask_args[0] > 0)
        if const_expr(mStartEnd is not None):
            assert len(mStartEnd.shape) == 4 and mStartEnd.element_type == Int32
        assert (mBlockIdx is not None) == (mBlockCnt is not None)
        if const_expr(mBlockIdx is not None):
            assert mStartEnd is not None and not self.mask_args[2]
            assert len(mBlockIdx.shape) == 4 and mBlockIdx.element_type == Int32
            assert len(mBlockCnt.shape) == 3 and mBlockCnt.element_type == Int32
        if const_expr(mSeqUsedK is not None):
            assert len(mSeqUsedK.shape) == 1 and mSeqUsedK.element_type == Int32
        if const_expr(mPageTable is not None):
            assert len(mPageTable.shape) == 2 and mPageTable.element_type == Int32
        if const_expr(mNumSplitsDynamic is not None):
            assert len(mNumSplitsDynamic.shape) == 1 and mNumSplitsDynamic.element_type == Int32

        mQ, mK, mV, mO = [assume_tensor_aligned(t) for t in (mQ, mK, mV, mO)]
        batch, num_heads = mQ.shape[0], mQ.shape[2]
        # Paged KV: mK / mV are (num_pages, page_size, H_kv, D) pools. page_size must be one
        # 128-token unit so a unit is exactly one page; the maximum sequence length is the
        # page table width. Dense KV keeps (batch, seqlen_k, H_kv, D). page_size is dynamic
        # here, so the dispatch check (swapab_decode_supported) enforces it on the host.
        if const_expr(mPageTable is not None):
            num_heads_kv = mK.shape[2]
            seqlen_k = mPageTable.shape[1] * SWAPAB_SKIP_UNIT
        else:
            num_heads_kv, seqlen_k = mK.shape[2], mK.shape[1]
        # The core takes Q as (b, h, d), O as (ns, b, h, d) and LSE as (ns, b, h). Without
        # splits the seqlen_q mode of extent 1 becomes the split mode.
        mLSE_core = None
        mQ_core = mQ[None, 0, None, None]
        if const_expr(self.is_split_kv):
            mO_core = mO[None, None, 0, None, None]
            mLSE_core = mLSE[None, None, None, 0]
        else:
            mO_core = select_modes(mO, [1, 0, 2, 3])
            if const_expr(mLSE is not None):
                mLSE_core = select_modes(mLSE, [2, 0, 1])
        core = _SwapABCore(
            self.tile_hq,
            self.tile_n,
            self.head_dim,
            self.head_dim_v,
            mQ.element_type,
            *self.mask_args,
            occupancy=self.occupancy,
        )
        problem_size = (batch, num_heads, num_heads_kv, seqlen_k)
        core(
            problem_size,
            mQ_core,
            mK,
            mV,
            mO_core,
            mLSE_core,
            mStartEnd,
            mBlockCnt,
            mBlockIdx,
            mSeqUsedK,
            mPageTable,
            mNumSplitsDynamic,
            softmax_scale,
            mask_row,
            stream,
        )
