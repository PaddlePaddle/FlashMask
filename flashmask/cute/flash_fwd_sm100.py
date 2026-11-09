# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 The FlashMask Authors. All rights reserved.

import enum
import math
from typing import Type, Tuple, Callable, Optional, Literal
from functools import partial

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, const_expr
from cutlass.cute.nvgpu import cpasync
import cutlass.cute.nvgpu.tcgen05 as tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils_basic

from . import utils
from . import copy_utils
from .barrier import wait_write_ptr_ge
from . import pipeline
from .mask import AttentionMask
from .softmax import SoftmaxSm100
from .seqlen_info import SeqlenInfoQK
from .block_info import BlockInfo
from .block_sparsity import BlockSparseTensors
from .block_sparse_utils import (
    get_total_block_count,
    produce_block_sparse_loads_sm100,
    softmax_block_sparse_sm100,
    handle_block_sparse_empty_tile_correction_sm100,
)
from .pack_gqa import PackGQA
from . import mma_sm100_desc as sm100_desc
from . import blackwell_helpers as sm100_utils
from cutlass.cute import FastDivmodDivisor
from .tile_scheduler import (
    TileSchedulerArguments,
    SingleTileScheduler,
    StaticPersistentTileScheduler,
    SingleTileLPTScheduler,
    DynamicPersistentTileScheduler,
    ParamsBase,
)
from .flashmask_utils import FlashMaskInfo, OverlapInfo


@cute.jit
def _overlap_gate(
    nblk: Int32,
    tidx: Int32,
    s_total: Int32,
    batch_idx: Int32,
    write_ptr: cute.Pointer,
    spin_timeout_ns: cutlass.Int64,
    n_block_size: cutlass.Constexpr[int],
    local_kv_seqlen: Int32,
):
    """Spin until the comm side has gathered the KV rows of tile ``nblk``.

    All load-warp lanes must call this; the local chunk (``reverse_row < 0``) needs no wait.
    """
    reverse_row = s_total - nblk * n_block_size - local_kv_seqlen
    if reverse_row >= 0:
        if tidx == 0:
            target = batch_idx * (s_total - local_kv_seqlen) + reverse_row
            wait_write_ptr_ge(write_ptr, 0, Int32(target), spin_timeout_ns)
        # Share lane 0's acquire with the whole warp.
        cute.arch.sync_warp()
        # Order the acquire before this lane's TMA (async-proxy) reads.
        cute.arch.fence_proxy("async.global")


class NamedBarrierFwd(enum.IntEnum):
    Epilogue = enum.auto()  # starts from 1 as barrier 0 is reserved for sync_threads()
    GenerateBlock = enum.auto()
    # Folded accumulator (m_block_size == 64) only: the two threads that share a query row
    # (t and t+64, i.e. different warps) exchange their half-row softmax statistics through
    # smem, which needs a barrier over the whole 4-warp softmax group.
    SoftmaxRowExchange = enum.auto()
#     WarpSchedulerWG1 = enum.auto()
#     WarpSchedulerWG2 = enum.auto()
#     WarpSchedulerWG3 = enum.auto()
#     PFull = enum.auto()
#     PEmpty = enum.auto()


class FlashAttentionForwardSm100:
    arch = 100

    def __init__(
        self,
        # dtype: Type[cutlass.Numeric],
        head_dim: int,
        head_dim_v: Optional[int] = None,
        qhead_per_kvhead: cutlass.Constexpr[int] = 1,
        is_causal: bool = False,
        is_local: bool = False,
        is_split_kv: bool = False,
        pack_gqa: bool = False,
        m_block_size: int = 128,
        n_block_size: int = 128,
        is_persistent: bool = True,
        single_q_stage: bool = False,
        has_block_logit: cutlass.Constexpr = False,
        block_size: cutlass.Constexpr[int] = 64,
        has_block_bos: cutlass.Constexpr = False,
        use_2cta_instrs: bool = False,
    ):
        # self.dtype = dtype
        # padding head_dim to a multiple of 16 as k_block_size
        hdim_multiple_of = 16
        self.head_dim_padded = int(math.ceil(head_dim / hdim_multiple_of) * hdim_multiple_of)
        head_dim_v = head_dim_v if head_dim_v is not None else head_dim
        self.same_hdim_kv = head_dim == head_dim_v
        self.head_dim_v_padded = int(math.ceil(head_dim_v / hdim_multiple_of) * hdim_multiple_of)
        self.same_hdim_kv_padded = self.head_dim_padded == self.head_dim_v_padded
        self.check_hdim_oob = head_dim != self.head_dim_padded
        self.check_hdim_v_oob = head_dim_v != self.head_dim_v_padded
        self.m_block_size = m_block_size
        self.n_block_size = n_block_size
        self.single_q_stage = single_q_stage
        # single_q_stage: q_stage must be 1 to fit TMEM (S + O_full = 128 + 256 = 384, with gap total = 512)
        self.q_stage = 1 if single_q_stage else 2
        assert self.q_stage in [1, 2]
        # Number of S/P (softmax) pipeline stages. The non-single_q_stage kernel runs two
        # softmax warpgroups on two DIFFERENT Q tiles (stage == Q tile index), so both
        # QK gemms and both softmax warpgroups do useful work. single_q_stage has a single Q
        # tile, so a second stage would recompute the very same S from the same Q and
        # the same full-d K and then throw the result away: one wasted QK gemm plus one
        # wasted softmax warpgroup per KV tile. Keep exactly q_stage softmax stages.
        self.num_s_stages = self.q_stage

        # 2-CTA (tcgen05 CTA-pair) UMMA. The MMA tiler's M spans the whole pair while
        # each CTA still owns `m_block_size` rows, so a pair covers
        # `cta_group_size * m_block_size` query rows. Everything that indexes a work
        # tile / mask / block list must use that product (the interface computes it as
        # fwd_m_tile_rows), everything that indexes this CTA's own rows must use
        # `m_block_size`.
        self.use_2cta_instrs = use_2cta_instrs
        self.cta_group_size = 2 if use_2cta_instrs else 1

        # 2 Q tile per CTA
        self.cta_tiler = (self.q_stage * m_block_size, n_block_size, self.head_dim_padded)
        self.mma_tiler_qk = (
            self.cta_group_size * m_block_size,
            n_block_size,
            self.head_dim_padded,
        )
        self.mma_tiler_pv = (
            self.cta_group_size * m_block_size,
            self.head_dim_v_padded,
            n_block_size,
        )
        # The tcgen05 MMA atom's N is capped at 256, so head_dim_v > 256 cannot be one
        # instruction. It is expressed as several N-tiles of the SAME accumulator
        # instead: the atom is built with N = pv_atom_n while the MMA *tiler* keeps the
        # full head_dim_v, so partition_shape_C((M, head_dim_v)) yields
        # head_dim_v / pv_atom_n N-tiles that sit side by side in TMEM columns.
        self.pv_atom_n = min(self.head_dim_v_padded, 256)
        assert self.head_dim_v_padded % self.pv_atom_n == 0
        self.qk_acc_dtype = Float32
        self.pv_acc_dtype = Float32
        self.cluster_shape_mn = (self.cta_group_size, 1)
        # tcgen05.commit / remote-mbarrier plumbing for the CTA pair. With cta_group=2 the
        # UMMA is issued by the leader CTA only, so (a) every barrier the MMA warp *drives*
        # must be committed with a cluster multicast mask so the peer CTA's warps wake up,
        # and (b) every barrier the MMA warp *waits on* must be arrived at remotely by the
        # peer CTA's softmax / correction warps (they only exist locally).
        self.mma_commit_mask = None if self.cta_group_size == 1 else (1 << self.cta_group_size) - 1
        self.mma_leader_cta_rank = None if self.cta_group_size == 1 else 0
        if self.use_2cta_instrs:
            # gQ is tiled by the MMA tiler's M (= cta_group_size * m_block_size), so the Q
            # tile index has to equal the work tile index; that only holds for q_stage == 1.
            assert self.q_stage == 1, "2-CTA UMMA currently requires q_stage == 1"
            assert self.num_s_stages == 1
            # A CTA owns m_block_size accumulator rows. Every TMEM copy in
            # softmax/correction/epilogue is a 128-thread tiled copy; its thread->row map is
            # 1:1 (thread t owns row t, holding the full n_block) only when the CTA owns 128
            # rows. With m_block_size == 64 CuTe instead spreads the 64 rows over all 128
            # TMEM lanes by splitting the accumulator's N, so threads t and t+64 share a row
            # and each holds half of it. That FOLDED layout is what makes dv=512 fit in one
            # pass (a 64 x N f32 accumulator costs N/2 TMEM columns, not N -- the same trick
            # FlashMLA's sm100 sparse prefill uses to keep O(64x512) in 256 columns), at the
            # cost of:
            #   - row_max must be combined across the thread pair before exp2 (both halves of
            #     a row must be scaled by the SAME max), see softmax_step's row exchange;
            #   - row_sum is a per-half partial and is summed once per work tile;
            #   - every tidx -> query row mapping (sScale, gLSE) must go through the
            #     accumulator's identity tensor instead of using tidx directly.
            assert m_block_size in (64, 128), (
                "2-CTA UMMA requires m_block_size in (64, 128) (per CTA); "
                f"got {m_block_size}"
            )
        # Folded accumulator: the CTA owns fewer than 128 accumulator rows, so CuTe splits N
        # over the TMEM lanes and a query row is shared by two threads (t, t + m_block_size).
        self.folded_acc = self.m_block_size < 128
        if self.folded_acc:
            # These paths map a thread index straight to a query row (block_logit's
            # `row = m_block * m_block_size + blk_tidx`, PackGQA's per-row q_head_idx) or
            # rely on one thread owning a whole row; none of them is needed by the
            # dv=512 single-pass config that motivates the folded layout.
            assert not has_block_logit, "folded accumulator does not support block_logit"
            assert not pack_gqa, "folded accumulator does not support pack_gqa"
            assert not is_split_kv, "folded accumulator does not support split_kv"
        # sScale holds, per softmax stage, one row_sum and one row_max slot per accumulator
        # row: [stage * m + row] and [stage * m + row + 2 * m]. The folded layout appends
        # 2 * m exchange slots ([exch_offset + half * m + row]) that the two threads sharing
        # a row use to combine their half-row statistics; they are in different warps, so a
        # shuffle cannot reach across and this has to go through smem.
        self.sScale_exch_offset = 2 * self.m_block_size * 2
        self.sScale_size = self.sScale_exch_offset + (
            2 * self.m_block_size if self.folded_acc else 0
        )
        self.is_persistent = is_persistent
        self.is_causal = is_causal
        self.is_local = is_local
        self.qhead_per_kvhead = qhead_per_kvhead
        self.is_split_kv = is_split_kv
        self.pack_gqa = pack_gqa
        if pack_gqa:
            assert m_block_size % self.qhead_per_kvhead == 0, (
                "For PackGQA, m_block_size must be divisible by qhead_per_kvhead"
            )
        assert not (self.is_split_kv and self.head_dim_v_padded >= 192), (
            "SplitKV is not supported for hdim >= 192"
        )
        if single_q_stage:
            # single_q_stage forces q_stage == 1 so that S/P (2 * n_block cols) plus the
            # O accumulator (head_dim_v_padded cols) fit the 512-col TMEM budget. Two
            # configs use it:
            #   - symmetric d == dv == 256 (n_block=128): 2*128 + 256 = 512
            #   - big-d d>256 with a caller-provided dv <= 256 (n_block=32):
            #     2*32 + 256 = 320. dv > 256 cannot use this config at all (dv=512
            #     alone would need all 512 columns) and goes folded instead.
            assert self.head_dim_padded > 192, "single_q_stage requires head_dim > 192"
            assert self.head_dim_v_padded <= 256 or self.folded_acc, (
                "single_q_stage requires head_dim_v <= 256 so O (head_dim_v cols) fits TMEM "
                "alongside S/P (2 * n_block cols); head_dim_v up to 512 needs the folded "
                "(m_block_size == 64) accumulator, which halves every column count"
            )
            assert not self.is_split_kv, "single_q_stage does not support SplitKV"
            assert not self.pack_gqa, "single_q_stage does not support pack_gqa"
        # HySparse block-score fusion: emit per-(query, key-block) max raw logit.
        self.has_block_logit = has_block_logit
        self.block_size = block_size
        # Per-row document-start (bos) input for document-RELATIVE block bucketing
        # (pack-equivalence). When absent we fall back to bos=0, i.e. the original
        # absolute (packed-sequence) bucketing.
        self.has_block_bos = has_block_bos
        if cutlass.const_expr(has_block_logit):
            assert n_block_size % block_size == 0, (
                "block_size must divide n_block_size for the fused block-score"
            )
            # Relative bucketing floor-divides (abs_col - bos) by block_size via an
            # arithmetic right shift, so block_size must be a power of two.
            assert (block_size & (block_size - 1)) == 0, (
                "block_size must be a power of two for relative block bucketing"
            )
            self.block_size_log2: cutlass.Constexpr = block_size.bit_length() - 1
            self.blocks_per_ntile: cutlass.Constexpr = n_block_size // block_size
            # A document-relative bos offset (unaligned to block_size) shifts the
            # 64-wide grid, so one n-tile (blocks_per_ntile blocks) can straddle one
            # extra relative block: allocate blocks_per_ntile + 1 accumulator slots.
            self.n_block_slots: cutlass.Constexpr = self.blocks_per_ntile + 1
        else:
            self.blocks_per_ntile: cutlass.Constexpr = 1
            self.n_block_slots: cutlass.Constexpr = 1
        self.vec_size: cutlass.Constexpr = 2
        # Does S1 need to wait for S0 to finish
        # self.s0_s1_barrier = self.head_dim_padded in [64, 96] and (not self.is_causal and not self.is_local)
        self.s0_s1_barrier = False

        # The softmax warp stores P to TMEM in two chunks (first 3/4, then the last 1/4)
        # so the MMA warp can start the PV gemm on the first chunk and pick up the rest
        # mid-instruction-stream (mbar_P_full_2, see gemm_ptx_partial's mbar_ptr path).
        #
        # The two sides count DIFFERENT things and both split at `count // 4 * 3`:
        #   - softmax: `tStP_r2t.shape[2]` TMEM store chunks. tStP has
        #     tilePlikeFP32 = n_block // 32 * 16 f32-equivalent columns and the store atom is
        #     St32x32b(Repetition(16)), so the chunk count is n_block // 32.
        #   - MMA: `tCrA.shape[2]` PV MMA-K tiles, i.e. n_block // 16 (16 elems per tcgen05
        #     k step for 16-bit).
        # They only describe the same 3/4 of P when the store side has at least 4 chunks,
        # i.e. n_block >= 128. At n_block = 64 the store side degenerates to 4 // 4 * 3 with
        # only 2 chunks -> `2 // 4 * 3 == 0`: the softmax signals P_full before storing ANY
        # of P, while the MMA still runs 3 of its 4 K tiles -> the PV gemm consumes the
        # PREVIOUS KV tile's P. (Invisible with an all-ones input, where every tile's P is
        # identical; with random inputs it showed up as a ~50x output error.)
        self.split_p_store = self.n_block_size // 32 >= 4
        if self.split_p_store:
            # Guard the invariant above: both split points must cover the same columns.
            store_chunks = self.n_block_size // 32
            mma_k_tiles = self.n_block_size // 16
            assert (store_chunks // 4 * 3) * 32 == (mma_k_tiles // 4 * 3) * 16, (
                "split_p_store requires the softmax store split and the MMA K split to "
                f"cover the same part of P (n_block_size={self.n_block_size})"
            )

        if self.num_s_stages == 1:
            # single_q_stage / 2-CTA big-d: there is a single S/P stage, so the second softmax
            # warpgroup has no work at all -- keep it OUT of the CTA instead of letting it
            # idle. The CTA then has 12 warps (384 threads) and ptxas can give every thread
            # 65536/384 = 168 registers instead of 65536/512 = 128, which is what the
            # softmax fragment (n_block f32 of S + P + the flashmask compare temporaries)
            # needs to stay out of local memory. The 128-register build spilled ~16KB per
            # KV tile and spent 83% of its issue slots stalled on those local loads.
            #
            # The softmax / correction groups keep FOUR warps in 2-CTA mode as well: the
            # TMEM copies (Ld32x32b / St32x32b) over a `m_block_size` x n accumulator are
            # 128-thread tiled copies regardless of cta_group, and their thread->row map is
            # 1:1 only at m_block_size == 128. Running 2 warps there left half of every S
            # fragment unread and half of P unwritten (output was zeros plus inf).
            self.softmax0_warp_ids = (0, 1, 2, 3)
            self.softmax1_warp_ids = ()
            self.correction_warp_ids = (4, 5, 6, 7)
            self.mma_warp_id = 8
            self.epilogue_warp_ids = (9,)
            self.load_warp_ids = (10,)
            self.empty_warp_ids = ()
            self.generate_block_warp_ids = (11,)
        else:
            self.softmax0_warp_ids = (0, 1, 2, 3)
            self.softmax1_warp_ids = (4, 5, 6, 7)
            self.correction_warp_ids = (8, 9, 10, 11)
            self.mma_warp_id = 12
            self.epilogue_warp_ids = (13,)
            self.load_warp_ids = (14,)
            self.empty_warp_ids = ()
            self.generate_block_warp_ids = (15,)
        SM100_TMEM_CAPACITY_COLUMNS = 512
        self.tmem_alloc_cols = SM100_TMEM_CAPACITY_COLUMNS

        self.threads_per_cta = cute.arch.WARP_SIZE * len(
            (
                *self.softmax0_warp_ids,
                *self.softmax1_warp_ids,
                *self.correction_warp_ids,
                self.mma_warp_id,
                *self.load_warp_ids,
                *self.epilogue_warp_ids,
                *self.generate_block_warp_ids,
                *self.empty_warp_ids,
            )
        )
        self.num_warps = self.threads_per_cta // cute.arch.WARP_SIZE
        # The mbarrier init work is spread over one warp per barrier group (groups 1..9).
        # The group index is wrapped into the warps that exist so a narrower warp layout
        # still initializes every group -- a warp may handle two groups, but no group is
        # left uninitialized (which would deadlock) and no barrier is initialized twice.
        # With the 12- and 16-warp layouts this is the identity mapping.
        self.mbar_init_warp = tuple(i % self.num_warps for i in range(16))

        self.tmem_s_offset = [0, self.tmem_cols(self.n_block_size)]  # e.g., 0, 128
        self.tmem_o_offset = [
            self.tmem_s_offset[-1]
            + self.tmem_cols(self.n_block_size)
            + i * self.tmem_cols(self.head_dim_v_padded)
            for i in range(self.q_stage)
        ]  # e.g., 256, 384
        self.tmem_total = self.tmem_o_offset[-1] + self.tmem_cols(self.head_dim_v_padded)
        assert self.tmem_total <= SM100_TMEM_CAPACITY_COLUMNS
        # The O accumulator's tmem pointers claim 16B (4 column) alignment so the epilogue's
        # tmem load atom (16 DP / 256 bit / x2 for a 64-row epi tile) accepts them.
        assert all(off % 4 == 0 for off in self.tmem_o_offset)
        self.tmem_s_to_p_offset = self.tmem_cols(self.n_block_size) // 2
        self.tmem_p_offset = [
            self.tmem_s_offset[i] + self.tmem_s_to_p_offset for i in range(2)
        ]  # 0, 128
        # d == dv == 256 on the CTA pair with 128 x 128 tiles:
        # S0/S1 = [0, 128) / [128, 256), O = [256, 512), P over the upper half of its S slot.
        # Each CTA's TMEM lanes hold that CTA's 128 rows of S, P and O, so there is no
        # cross-CTA column sharing to avoid here.
        self.hd256_2cta = (
            self.use_2cta_instrs
            and self.head_dim_padded == 256
            and self.head_dim_v_padded == 256
            and self.m_block_size == 128
            and self.n_block_size == 128
        )
        if self.use_2cta_instrs and not self.hd256_2cta:
            # 2-CTA: give P its own TMEM columns instead of folding it onto S's upper half.
            # Two reasons, both specific to the CTA pair:
            #   - P is the only operand the LEADER's UMMA reads out of the PEER's TMEM
            #     (A-from-TMEM, rows 128-255 live in the odd CTA), so it must not share
            #     columns with anything the pair writes on a different schedule.
            #   - the shared-column offset was n_block/2 (32 columns at n=64), i.e. not
            #     even 64-column aligned; P now starts at a 128-column-aligned address.
            # Costs tilePlikeFP32 = n_block/2 columns (32 at n=64, on top of 384 used by
            # S/O), which the 512-column budget has room for.
            p_cols = self.tmem_cols(self.n_block_size) // 2  # tilePlikeFP32 for 16-bit operands
            self.tmem_p_offset = [self.tmem_total + i * p_cols for i in range(2)]
            # softmax_step derives P's address as tStS + tmem_s_offset[stage] +
            # tmem_s_to_p_offset, so keep the two views consistent. Only stage 0 exists
            # here (2-CTA asserts num_s_stages == 1 above).
            self.tmem_s_to_p_offset = self.tmem_p_offset[0] - self.tmem_s_offset[0]
            assert (
                self.tmem_p_offset[self.num_s_stages - 1] + p_cols
                <= SM100_TMEM_CAPACITY_COLUMNS
            )

        # vec buffer for row_max & row_sum
        self.tmem_vec_offset = self.tmem_s_offset

        if self.head_dim_padded < 96:
            self.num_regs_softmax = 200
            self.num_regs_correction = 64
            self.num_regs_other = 48
        elif self.single_q_stage:
            # 256 + 128 + 64 = 448 <= 512. Only the folded (dv > 256) config is retuned;
            # the symmetric d == dv == 256 single_q_stage config keeps its measured 208/128/32.
            if self.folded_acc:
                self.num_regs_softmax = 256
                self.num_regs_correction = 128
                self.num_regs_other = 64
            else:
                self.num_regs_softmax = 208
                self.num_regs_correction = 128
                self.num_regs_other = 32

        else:
            # self.num_regs_softmax = 192 if self.is_causal or self.is_local else 184
            self.num_regs_softmax = 200
            # self.num_regs_softmax = 176
            # self.num_regs_correction = 96
            # self.num_regs_correction = 80
            # self.num_regs_correction = 64 if self.is_causal or self.is_local else 80
            self.num_regs_correction = 64
            # self.num_regs_other = 32
            # self.num_regs_other = 64
            # self.num_regs_other = 80
            self.num_regs_other = 48
            # self.num_regs_other = 96 if self.is_causal or self.is_local else 80
            # self.num_regs_other = 64 if self.is_causal or self.is_local else 80
        self.num_regs_empty = 24

        self.buffer_align_bytes = 1024

        self.generate_block_incomplete = 0x80000001 # Does cutlass.Int32 store Int32.min() as 0x80000000?
        self.generate_block_finish = 0x80000000

    def _setup_attributes(self):
        """Set up configurations and parameters for the FMHA kernel operation.

        This method initializes and configures various attributes required for the
        execution of the fused multi-head attention kernel, mainly about the pipeline stages:

        - Sets up staging parameters for Q, K, V inputs and accumulator data
        - Configures pipeline stages for softmax, correction, and epilogue operations
        """

        if self.head_dim_padded == 192 and self.head_dim_v_padded == 128:
            self.kv_stage = 2 if self.enable_flashmask else 3
        elif self.single_q_stage:
            # single_q_stage: SMEM budget is tight (d=dv=256: Q=64KB + KV=128KB = 192KB;
            # d=576/dv<=256, n=32: Q=144KB + KV=72KB = 216KB), so keep the shallowest
            # depth that still works. kv_stage MUST be >= 2: V aliases K's SMEM (there
            # is no separate sV field, see `sV = recast_ptr(sK.iterator, ...)`), so with
            # kv_stage=1 a block's K and V map to the same physical slot and V would
            # overwrite K.
            # hd256 on the CTA pair: a stage is half a K/V tile (32 KB), 4 stages = 128 KB.
            self.kv_stage = 4 if self.hd256_2cta else 2
        elif self.q_dtype.width == 8 or self.q_stage == 1:
            self.kv_stage = 4
        else:
            self.kv_stage = 3

        self.acc_stage = 1
        # single_q_stage: reduce epi_stage to 1 to fit SMEM (O=128KB with epi_stage=2 is too large)
        self.epi_stage = 1 if self.single_q_stage else 2
        self.generate_block_stage = 2
        # For hdim 192,128, we don't have enough smem to store all 3 stages of KV:
        # 128 x 192 x 2 bytes x 3 stages = 144KB, and we need 96KB for Q.
        # Instead we store smem as [smem_large, smem_small, smem_large], where smem_large is
        # 128 x 192 and smem_small is 128 x 128. We set the stride between the stages to be
        # 128 * 160, so that indexing the 0th and 2nd stages will get the right address,
        # but for the 1st stage we need to add or subtract (depending on phase) 128 x 64.
        self.uneven_kv_smem = (
            self.head_dim_padded == 192 and self.head_dim_v_padded == 128 and self.kv_stage == 3
        )
        self.uneven_kv_smem_offset = (
            self.m_block_size * (self.head_dim_padded - self.head_dim_v_padded) // 2
            if self.uneven_kv_smem
            else 0
        )
        assert self.uneven_kv_smem_offset % 1024 == 0

    @cute.jit
    def tma_expect_tx(self, mbar_ptr, tx_bytes):
        """Declare the expected transaction bytes for a TMA load into `mbar_ptr`.

        With cta_group=2 both CTAs of the pair issue their own half of the
        `cp.async.bulk.tensor ... .cta_group::2`, but the hardware records every
        `complete_tx` on the *leader* CTA's mbarrier (the barrier address has the CTA-rank
        bits masked off). So only the leader may declare expect_tx, and it declares the
        whole pair's byte count -- which is why `tma_copy_bytes` is scaled by
        `cta_group_size`. If a follower also declared expect_tx, its own barrier would
        accumulate pending transactions that never complete and overflow the mbarrier's
        20-bit tx counter, faulting on the arrive itself (SYNCS.ARRIVE.TRANS64).
        `PipelineTmaUmma.producer_acquire` gates this the same way; these two load helpers
        drive the mbarriers directly and so have to gate it themselves.
        """
        if const_expr(self.cta_group_size == 1):
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(mbar_ptr, tx_bytes)
        else:
            if self.cta_coord_v() == 0:
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(mbar_ptr, tx_bytes)

    def cta_coord_v(self):
        """This CTA's V-mode coordinate inside the MMA's CTA pair (0 for cta_group=1)."""
        if const_expr(self.cta_group_size == 1):
            return Int32(0)
        return (
            cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
            % self.cta_group_size
        )

    def m_tile_pair_base(self, m_block, stage):
        """Row base handed to the mask, in units of `m_block_size`.

        The mask derives each element's row as `tScS_t2r[i][0] + m_block * tile_m`, and
        `tScS` comes from `thr_mma.partition_C(identity(mma_tiler_qk[:2]))`. With cta_group=2
        that identity spans the CTA *pair's* whole M range, so the coordinate already carries
        this CTA's intra-pair row offset -- adding `mma_tile_coord_v` here would double-count
        it. Contrast with `m_tile_index`, which is for tensors that are tiled per CTA (LSE,
        O) and therefore does need the V coordinate.
        """
        base = self.q_stage * m_block + stage
        if const_expr(self.cta_group_size == 1):
            return base
        return base * self.cta_group_size

    @cute.jit
    def gemm_lib(
        self,
        tiled_mma: cute.TiledMma,
        acc: cute.Tensor,
        tCrA: cute.Tensor,
        tCrB: Optional[cute.Tensor] = None,
        sA: Optional[cute.Tensor] = None,
        sB: Optional[cute.Tensor] = None,
        zero_init: bool = False,
        mbar_ptr: Optional[cute.Pointer] = None,
        mbar_phase: Optional[Int32] = None,
    ):
        """`cute.gemm` drop-in for `gemm_ptx_partial`, used by the 2-CTA path.

        Same keyword interface as the hand-written PTX helper (sA/sB are accepted and
        ignored; the library derives both operand descriptors and the cta_group from the
        MMA op). `mbar_ptr` is honoured as a plain wait BEFORE the gemm instead of
        mid-instruction-stream, so the barrier stays balanced; the 2-CTA config keeps
        split_p_store off, so that path is not on the hot loop.

        The ACCUMULATE field is set on a LOCAL mma atom, never on the shared `tiled_mma`:
        `tiled_mma.set` rebinds the object's MLIR value, and since the same `tiled_mma`
        object is sliced once at the top of the kernel and reused by the softmax /
        correction warps, mutating it inside the MMA warp's region makes those later uses
        reference a value defined in a region that does not dominate them ("operand #0
        does not dominate this use" at `thr_mma_qk.partition_C`).
        """
        if const_expr(mbar_ptr is not None):
            cute.arch.mbarrier_wait(mbar_ptr, mbar_phase)
        mma_atom = cute.make_mma_atom(tiled_mma.op)
        for k in cutlass.range_constexpr(cute.size(tCrA.shape[2])):
            mma_atom.set(tcgen05.Field.ACCUMULATE, (k != 0) or (not zero_init))
            cute.gemm(mma_atom, acc, tCrA[None, None, k], tCrB[None, None, k], acc)

    def tmem_cols(self, n: int) -> int:
        """Physical TMEM columns taken by an `m_block_size x n` f32 accumulator.

        m_block_size is this CTA's share of the MMA's M, so the cluster-wide M is
        m_block_size * cta_group_size. The column rule (including folding) lives in
        blackwell_helpers.tmem_cols.
        """
        return sm100_utils.tmem_cols(
            n, self.m_block_size * self.cta_group_size, self.cta_group_size
        )

    def folded_acc_lanes(self) -> int:
        """Number of TMEM lanes the O accumulator occupies (always 128)."""
        return self.m_block_size * (2 if self.folded_acc else 1)

    def folded_o_phys_view(self, t: cute.Tensor) -> cute.Tensor:
        """Re-index a folded O accumulator (or an sO / identity partition of it) by its
        PHYSICAL (lane, column) shape.

        The folded accumulator keeps logical column `n` of row `r` in TMEM lane
        `r + m_block_size * (n // (pv_atom_n // 2))`: the upper half of every N-tile's
        columns lives in lanes 64..127. A tmem copy built over a tile whose M mode is only
        `m_block_size` therefore covers 64 lanes, and `make_tmem_copy` resolves that by
        handing warps 2/3 the SAME coordinates as warps 0/1 (measured TV layout
        ((32,(2,2)),(16,32)):((0,(1,0)),(64,2)) -- warp stride 0). The hardware still points
        those warps at lanes 64..127, so half the threads read the OTHER dv half and store it
        over the correct values. FWD_ONES=v cannot see this (with V == 1 every dv column of O
        equals the row sum, so any dv mix-up still normalizes to 1.0), which is exactly why
        that run passed while real V did not.

        This view makes the fold explicit: mode 0 is (row, lane_half) == 128 lanes and mode 1
        is (columns within a lane half, n_tile). Every copy built over it is a real
        128-thread copy with one lane per thread, and the matching view of sO / of the
        identity tensor carries the logical dv column that belongs to that lane, so the
        epilogue's (row, dv) map follows the fold instead of fighting it.

        `t` must be an O-accumulator-shaped tensor, i.e. ((m_block_size, pv_atom_n), _,
        n_tiles, ...) as produced by make_fragment_C / partition_C.
        """
        assert self.folded_acc
        m, half_n = self.m_block_size, self.pv_atom_n // 2
        n_tiles = self.head_dim_v_padded // self.pv_atom_n
        fold = cute.make_layout(
            ((m, 2), (half_n, n_tiles)),
            stride=((1, m * half_n), (m, m * self.pv_atom_n)),
        )
        return cute.make_tensor(t.iterator, cute.composition(t.layout, fold))

    @cute.jit
    def acc_row_half(self, coord_frg, thr_idx) -> Tuple[Int32, Int32]:
        """Map a thread's accumulator fragment to (row within this CTA, half index).

        Non-folded (m_block_size == 128): every thread owns one full row, so this returns
        (thr_idx, 0) -- bit-identical to the tidx-based indexing it replaces.

        Folded (m_block_size == 64): threads t and t + 64 share row t, each holding one half
        of the columns, so this returns (t, 0) and (t, 1). `coord_frg` is the
        thread-partitioned identity tensor over the accumulator tile, so `coord_frg[0]` is the
        (row, col) coordinate of the thread's first element. The row is taken modulo
        m_block_size because with cta_group=2 the identity spans the CTA pair's whole M range
        and therefore carries this CTA's intra-pair row offset, while every per-CTA buffer
        (sScale, the gLSE tile, sO) is only m_block_size rows tall.
        """
        if const_expr(not self.folded_acc):
            return Int32(thr_idx), Int32(0)
        row = Int32(coord_frg[0][0]) % self.m_block_size
        # col 0 -> low half, col N/2 -> high half. min(col, 1) turns that into 0/1 without
        # depending on N.
        half = cutlass.min(Int32(coord_frg[0][1]), Int32(1))
        return row, half

    @cute.jit
    def pair_exchange(self, sScale, row: Int32, half: Int32, value: Float32, is_max: bool):
        """Combine a half-row softmax statistic with the thread that owns the other half.

        No-op unless the accumulator is folded. The two threads sharing a row are 64 apart,
        i.e. in different warps, so this goes through smem (same reason FlashMLA's sm100
        sparse prefill keeps a `p_exchange_buf`). Both threads return the same value, which
        lets every downstream sScale / gLSE store be a same-address-same-value write.
        """
        if const_expr(not self.folded_acc):
            return value
        num_softmax_threads = len(self.softmax0_warp_ids) * cute.arch.WARP_SIZE
        mine = self.sScale_exch_offset + half * self.m_block_size + row
        peer = self.sScale_exch_offset + (1 - half) * self.m_block_size + row
        sScale[mine] = value
        cute.arch.barrier(
            barrier_id=int(NamedBarrierFwd.SoftmaxRowExchange),
            number_of_threads=num_softmax_threads,
        )
        other = sScale[peer]
        # Second barrier: the next exchange must not overwrite a slot that a partner thread
        # has not read yet.
        cute.arch.barrier(
            barrier_id=int(NamedBarrierFwd.SoftmaxRowExchange),
            number_of_threads=num_softmax_threads,
        )
        if const_expr(is_max):
            return cute.arch.fmax(value, other)
        else:
            return value + other

    def m_tile_index(self, m_block, stage, mma_tile_coord_v=None):
        """Index of the `m_block_size` rows this CTA owns, in units of `m_block_size`.

        `m_block` is the work tile index. One work tile covers
        `q_stage * cta_group_size * m_block_size` rows: `stage` selects the Q stage and, with
        cta_group=2, `mma_tile_coord_v` selects which half of the 2-CTA UMMA's M range this
        CTA owns. Every place that maps the accumulator back to global rows (mask row base,
        LSE rows, O rows) must go through here.
        """
        base = self.q_stage * m_block + stage
        if const_expr(self.cta_group_size == 1):
            return base
        return base * self.cta_group_size + mma_tile_coord_v

    def mma_barrier_arrive(self, mbar_ptr):
        """Arrive on a barrier that the MMA warp waits on.

        With cta_group=2 only the leader CTA runs the MMA warp, so the peer CTA's softmax /
        correction warps have to arrive on the *leader's* copy of the barrier (remote
        mbarrier arrive through `mapa`). For cta_group=1 this degenerates to a plain
        CTA-local arrive.
        """
        cute.arch.mbarrier_arrive(mbar_ptr, self.mma_leader_cta_rank)

    def mma_barrier_commit(self, mbar_ptr):
        """Signal, from the MMA warp, a barrier consumed by every CTA of the pair."""
        tcgen05.commit(mbar_ptr, self.mma_commit_mask, self.mma_cta_group)

    def tmem_dealloc_arrive(self, mbar_ptr):
        """Arrive on the tmem-dealloc barrier -- always CTA-local.

        Unlike the S/P handshakes, this one is NOT routed to the leader: every CTA of the
        pair issues its own `tcgen05.dealloc.cta_group::2` (that instruction is a pair
        collective), so each CTA gates on its own copy of the barrier, signalled by its own
        softmax / correction warps -- the only readers of that CTA's tmem.
        """
        cute.arch.mbarrier_arrive(mbar_ptr)

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,  # (b, s_q, h, d)
        mK: cute.Tensor,  # (b_k, s_k, h_k, d)
        mV: cute.Tensor,  # (b_k, s_k, h_k, dv)
        mO: cute.Tensor,  # (b, s_q, h, dv)
        mLSE: Optional[cute.Tensor],
        softmax_scale: Float32,
        window_size_left: Int32 | int | None = None,
        window_size_right: Int32 | int | None = None,
        learnable_sink: Optional[cute.Tensor] = None,
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        flashmask_info: Optional[FlashMaskInfo] = None,
        mBlockLogit: Optional[cute.Tensor] = None,
        mBlockBos: Optional[cute.Tensor] = None,
        overlap_k_addr: Optional[cutlass.Int64] = None,
        overlap_v_addr: Optional[cutlass.Int64] = None,
        overlap_write_ptr_addr: Optional[cutlass.Int64] = None,
        overlap_b: Optional[cutlass.Int32] = None,
        overlap_s: Optional[cutlass.Int32] = None,
        overlap_h: Optional[cutlass.Int32] = None,
        overlap_d: Optional[cutlass.Int32] = None,
        overlap_spin_timeout_ns: Optional[cutlass.Int64] = None,
        overlap_local_kv_size: Optional[cutlass.Int32] = None,
        overlap_bhsd_layout: cutlass.Constexpr = False,
        # int32[2] ticket / finished-CTA counter of DynamicPersistentTileScheduler (zeroed once,
        # self-resetting); only used when fm_persistent.
        mSchedCounter: Optional[cute.Tensor] = None,
        # Always keep stream as the last parameter (EnvStream: obtained implicitly via TVM FFI).
        stream: cuda.CUstream = None,
    ):
        """Execute the Fused Multi-Head Attention operation on the provided tensors.

        This method prepares the input tensors for processing, validates their shapes and types,
        configures the computation parameters, and launches the CUDA kernel.

        The method handles:
        1. Tensor layout transformations for specific memory access patterns
        2. Validation of tensor shapes and data types
        3. Initialization of hardware-specific parameters and memory layouts
        4. Configuration of TMA (Tensor Memory Access) operations
        5. Grid and work scheduling computation
        6. Kernel launch with appropriate parameters
        """
        # info for flashmask
        self.enable_flashmask = flashmask_info is not None
        # ops/fwd.py routes d == dv == 256 to the CTA pair only without flashmask (a 256-row pair
        # tile covers more KV blocks of a window than a 128-row one); the flashmask paths
        # below assume the 1-CTA kernel for that shape.
        assert not (self.enable_flashmask and self.hd256_2cta)
        self.has_lt_end = const_expr(flashmask_info is not None and flashmask_info.LTE_nblock_max is not None)
        self.has_ut_start = const_expr(flashmask_info is not None and flashmask_info.UTS_nblock_max is not None)
        self.has_ut_end = const_expr(flashmask_info is not None and flashmask_info.UTE_nblock_max is not None)
        # FM-4 overlap: K/V live in the NCCL-window SRBuffer (no Paddle tensor / dlpack
        # capsule), so they arrive as a raw addr + the gathered (B, S_total, H, D)
        # dims as RUNTIME Int32 scalars. Build the views HERE -- make_*_from_addr
        # needs this jit body's MLIR Context, and the Int32 dims give a dynamic
        # layout matching the dense from_dlpack path (its docstring explains why
        # static dims read the wrong bytes). write_ptr is the gate's int32 counter.
        self.enable_overlap = const_expr(overlap_write_ptr_addr is not None)
        self.overlap_bhsd_layout = const_expr(overlap_bhsd_layout)
        if const_expr(self.enable_overlap):
            if const_expr(overlap_bhsd_layout):
                mK = utils.make_bhsd_storage_bshd_from_addr(
                    overlap_k_addr,
                    overlap_b,
                    overlap_s,
                    overlap_h,
                    overlap_d,
                    mQ.element_type,
                    align=16,
                )
                mV = utils.make_bhsd_storage_bshd_from_addr(
                    overlap_v_addr,
                    overlap_b,
                    overlap_s,
                    overlap_h,
                    overlap_d,
                    mQ.element_type,
                    align=16,
                )
            else:
                mK = utils.make_contiguous_bshd_from_addr(
                    overlap_k_addr,
                    overlap_b,
                    overlap_s,
                    overlap_h,
                    overlap_d,
                    mQ.element_type,
                    align=16,
                )
                mV = utils.make_contiguous_bshd_from_addr(
                    overlap_v_addr,
                    overlap_b,
                    overlap_s,
                    overlap_h,
                    overlap_d,
                    mQ.element_type,
                    align=16,
                )
            assert overlap_spin_timeout_ns is not None
            overlap_info = OverlapInfo(
                utils.make_gmem_tensor_from_addr(
                    overlap_write_ptr_addr, (1,), (1,), cutlass.Int32, align=4
                ),
                overlap_local_kv_size,
                overlap_spin_timeout_ns,
            )
        else:
            overlap_info = None

        # setup static attributes before smem/grid/tma computation
        self.q_dtype = mQ.element_type
        self.k_dtype = mK.element_type
        self.v_dtype = mV.element_type
        self.o_dtype = mO.element_type
        # Assume all strides are divisible by 128 bits except the last stride
        new_stride = lambda t: (
            *(cute.assume(s, divby=128 // t.element_type.width) for s in t.stride[:-1]),
            t.stride[-1],
        )
        mQ, mK, mV, mO = [
            cute.make_tensor(t.iterator, cute.make_layout(t.shape, stride=new_stride(t)))
            for t in (mQ, mK, mV, mO)
        ]
        Q_layout_transpose = [1, 3, 2, 0]
        mQ = cute.make_tensor(mQ.iterator, cute.select(mQ.layout, mode=Q_layout_transpose))
        # (s_k, d, h_k, b_k)
        KV_layout_transpose = [1, 3, 2, 0]
        mK, mV = [
            cute.make_tensor(t.iterator, cute.select(t.layout, mode=KV_layout_transpose))
            for t in (mK, mV)
        ]
        if const_expr(self.is_split_kv):
            O_layout_transpose = [2, 4, 3, 1, 0]
            LSE_layout_transpose = [3, 2, 1, 0]
            num_splits = mO.shape[0]
        else:
            O_layout_transpose = [1, 3, 2, 0]
            LSE_layout_transpose = [2, 1, 0]
            num_splits = Int32(1)
        mO = cute.make_tensor(mO.iterator, cute.select(mO.layout, mode=O_layout_transpose))
        mLSE = (
            cute.make_tensor(mLSE.iterator, cute.select(mLSE.layout, mode=LSE_layout_transpose))
            if const_expr(mLSE is not None)
            else None
        )
        # HySparse block-score: (b, h, s_q, nblocks) -> (s_q, h, b, nblocks) so
        # the per-(q_row, head, batch) write mirrors the LSE addressing.
        if const_expr(self.has_block_logit):
            assert not self.is_split_kv, "block-score fusion assumes not split_kv"
            mBlockLogit = cute.make_tensor(
                mBlockLogit.iterator, cute.select(mBlockLogit.layout, mode=[2, 1, 0, 3])
            )
        # (s, d, h, b) -> (d, s, h, b)
        V_layout_transpose = [1, 0, 2, 3]
        mV = cute.make_tensor(mV.iterator, cute.select(mV.layout, mode=V_layout_transpose))

        self.q_major_mode = cutlass.utils.LayoutEnum.from_tensor(mQ).mma_major_mode()
        self.k_major_mode = cutlass.utils.LayoutEnum.from_tensor(mK).mma_major_mode()
        self.v_major_mode = cutlass.utils.LayoutEnum.from_tensor(mV).mma_major_mode()
        self.o_layout = cutlass.utils.LayoutEnum.from_tensor(mO)

        if const_expr(self.q_major_mode != tcgen05.OperandMajorMode.K):
            raise RuntimeError("The layout of mQ is not supported")
        if const_expr(self.k_major_mode != tcgen05.OperandMajorMode.K):
            raise RuntimeError("The layout of mK is not supported")
        if const_expr(self.v_major_mode != tcgen05.OperandMajorMode.MN):
            raise RuntimeError("The layout of mV is not supported")

        # check type consistency
        if const_expr(self.q_dtype != self.k_dtype):
            raise TypeError(f"Type mismatch: {self.q_dtype} != {self.k_dtype}")
        if const_expr(self.q_dtype != self.v_dtype):
            raise TypeError(f"Type mismatch: {self.q_dtype} != {self.v_dtype}")
        self._setup_attributes()
        self.use_tma_O = self.arch >= 90
        # This can be tuned. New apply_exp2_convert interface:
        #   ex2_emu_freq=0 ⇒ all-hardware exp2 (equivalent to old e2e=True without emulation).
        #   ex2_emu_freq>0 ⇒ emulate exp2 every N fragments, starting at ex2_emu_start_frg.
        self.ex2_emu_freq = 16
        self.ex2_emu_start_frg = 0
        if const_expr(
            self.head_dim_padded > 64 and not self.is_causal and not self.is_local and self.pack_gqa
        ):
            self.ex2_emu_freq = 10
            self.ex2_emu_start_frg = 1

        cta_group = tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE
        self.mma_cta_group = cta_group
        # the intermediate tensor p is from tmem & mK-major
        if const_expr(not self.folded_acc):
            p_source = tcgen05.OperandSource.TMEM
        else:
            # Folded accumulator: P CANNOT go through TMEM. make_fragment_A over a TMEM P
            # describes the A operand as "row r lives in lane r, all of the row's columns in
            # that lane" (measured: ((128,16),1,4):((65536,1),0,16), identical to the
            # non-folded config), but the folded softmax has row r split across threads t and
            # t + 64, i.e. across two TMEM LANES -- and a tmem store is lane-local, so the
            # thread holding the upper half of the row cannot write into the lower half's
            # lane. P therefore goes to SMEM and the PV MMA takes its A operand from there,
            # exactly like FlashMLA's sm100 sparse prefill (utcmma_ss with S in smem) and
            # exactly like this kernel's own QK gemm, which already feeds A from each CTA's
            # sQ under cta_group=2.
            p_source = tcgen05.OperandSource.SMEM
        p_major_mode = tcgen05.OperandMajorMode.K
        tiled_mma_qk = sm100_utils_basic.make_trivial_tiled_mma(
            self.q_dtype,
            self.q_major_mode,
            self.k_major_mode,
            self.qk_acc_dtype,
            cta_group,
            self.mma_tiler_qk[:2],
        )
        tiled_mma_pv = sm100_utils_basic.make_trivial_tiled_mma(
            self.v_dtype,
            p_major_mode,
            self.v_major_mode,
            self.pv_acc_dtype,
            cta_group,
            (self.mma_tiler_pv[0], self.pv_atom_n),
            p_source,
        )

        self.cluster_shape_mnk = (*self.cluster_shape_mn, 1)
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (tiled_mma_qk.thr_id.shape,),
        )

        # The epilogue tile is per-CTA: with cta_group=2 the MMA tiler's M spans the pair but
        # sO / the O TMA descriptor / gO all live in one CTA and cover m_block_size rows.
        self.epi_tile = (self.mma_tiler_pv[0] // self.cta_group_size, self.mma_tiler_pv[1])

        sQ_layout = sm100_utils_basic.make_smem_layout_a(
            tiled_mma_qk,
            self.mma_tiler_qk,
            self.q_dtype,
            self.q_stage,
        )
        sK_layout = sm100_utils_basic.make_smem_layout_b(
            tiled_mma_qk,
            self.mma_tiler_qk,
            self.k_dtype,
            self.kv_stage,
        )
        tP_layout = sm100_utils_basic.make_smem_layout_a(
            tiled_mma_pv,
            self.mma_tiler_pv,
            self.q_dtype,
            self.acc_stage,
        )
        sV_layout = sm100_utils_basic.make_smem_layout_b(
            tiled_mma_pv,
            self.mma_tiler_pv,
            self.v_dtype,
            self.kv_stage,
        )
        sO_layout = sm100_utils_basic.make_smem_layout_epi(
            self.o_dtype,
            self.o_layout,
            self.epi_tile,
            self.epi_stage,
        )
        if const_expr(not self.same_hdim_kv_padded):
            # sK and sV are using the same physical smem so we need to adjust the stride so that they line up
            stride_sK = const_expr(
                max(sK_layout.outer.stride[-1], 0)
            )  # take max to turn tuple to Int32
            stride_sV = const_expr(max(sV_layout.outer.stride[-1], 0))
            stage_stride = const_expr(
                max(stride_sK, stride_sV)
                if not self.uneven_kv_smem
                else (stride_sK + stride_sV) // 2
            )
            sK_layout = cute.make_composed_layout(
                sK_layout.inner,
                0,
                cute.make_layout(
                    (*sK_layout.outer.shape[:-1], self.kv_stage),
                    stride=(*sK_layout.outer.stride[:-1], stage_stride),
                ),
            )
            sV_layout = cute.make_composed_layout(
                sV_layout.inner,
                0,
                cute.make_layout(
                    (*sV_layout.outer.shape[:-1], self.kv_stage),
                    stride=(*sV_layout.outer.stride[:-1], stage_stride),
                ),
            )

        if const_expr(self.pack_gqa):
            shape_Q_packed = (
                (self.qhead_per_kvhead, mQ.shape[0]),
                mQ.shape[1],
                mK.shape[2],
                *mQ.shape[3:],
            )
            stride_Q_packed = (
                (mQ.stride[2], mQ.stride[0]),
                mQ.stride[1],
                mQ.stride[2] * self.qhead_per_kvhead,
                *mQ.stride[3:],
            )
            mQ = cute.make_tensor(
                mQ.iterator, cute.make_layout(shape_Q_packed, stride=stride_Q_packed)
            )
            shape_O_packed = (
                (self.qhead_per_kvhead, mO.shape[0]),
                mO.shape[1],
                mK.shape[2],
                *mO.shape[3:],
            )
            stride_O_packed = (
                (mO.stride[2], mO.stride[0]),
                mO.stride[1],
                mO.stride[2] * self.qhead_per_kvhead,
                *mO.stride[3:],
            )
            mO = cute.make_tensor(
                mO.iterator, cute.make_layout(shape_O_packed, stride=stride_O_packed)
            )
            if const_expr(mLSE is not None):
                shape_LSE_packed = (
                    (self.qhead_per_kvhead, mLSE.shape[0]),
                    mK.shape[2],
                    *mLSE.shape[2:],
                )
                stride_LSE_packed = (
                    (mLSE.stride[1], mLSE.stride[0]),
                    mLSE.stride[1] * self.qhead_per_kvhead,
                    *mLSE.stride[2:],
                )
                mLSE = cute.make_tensor(
                    mLSE.iterator, cute.make_layout(shape_LSE_packed, stride=stride_LSE_packed)
                )

        self.tma_copy_bytes = {
            name: cute.size_in_bytes(mX.element_type, cute.select(layout, mode=[0, 1, 2]))
            for name, mX, layout in [
                ("Q", mQ, sQ_layout),
                ("K", mK, sK_layout),
                ("V", mV, sV_layout),
            ]
        }
        if const_expr(self.cta_group_size > 1):
            # With cta_group::2 the TMA instruction issued by each CTA of the pair signals
            # the mbarrier of BOTH CTAs, so every mbarrier sees the *pair's* total bytes
            # while the smem layouts above are per-CTA. Scale the expected tx count.
            for name in ("Q", "K", "V"):
                self.tma_copy_bytes[name] *= self.cta_group_size

        # TMA load for Q
        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(cta_group)
        tma_store_op = cpasync.CopyBulkTensorTileS2GOp()

        tma_atom_Q, mQ = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            mQ,
            cute.select(sQ_layout, mode=[0, 1, 2]),
            self.mma_tiler_qk,
            tiled_mma_qk,
            self.cluster_layout_vmnk.shape,
        )

        # TMA load for K
        tma_atom_K, mK = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            mK,
            cute.select(sK_layout, mode=[0, 1, 2]),
            self.mma_tiler_qk,
            tiled_mma_qk,
            self.cluster_layout_vmnk.shape,
        )
        # TMA load for V
        tma_atom_V, mV = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            mV,
            cute.select(sV_layout, mode=[0, 1, 2]),
            self.mma_tiler_pv,
            tiled_mma_pv,
            self.cluster_layout_vmnk.shape,
        )

        o_cta_v_layout = cute.composition(cute.make_identity_layout(mO.shape), self.epi_tile)

        self.num_epilogue_threads = cute.arch.WARP_SIZE * len(self.epilogue_warp_ids)
        # fm_persistent writes O with plain stores from the correction warps, so it needs the
        # real gmem tensor (mO becomes the TMA coordinate tensor below).
        mO_gmem = mO
        if const_expr(self.use_tma_O):
            tma_atom_O, mO = cpasync.make_tiled_tma_atom(
                tma_store_op,
                mO,
                cute.select(sO_layout, mode=[0, 1]),
                o_cta_v_layout,
            )
            gmem_tiled_copy_O = None
        else:
            tma_atom_O = None
            universal_copy_bits = 128
            async_copy_elems = universal_copy_bits // self.o_dtype.width
            atom_universal_copy = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                self.o_dtype,
                num_bits_per_copy=universal_copy_bits,
            )
            tO_shape_dim_1 = sO_layout.outer.shape[1][0] // async_copy_elems
            tO_layout = cute.make_ordered_layout(
                (self.num_epilogue_threads // tO_shape_dim_1, tO_shape_dim_1),
                order=(1, 0),
            )
            # So that we don't have to check if we overshoot kBlockM when we store O
            assert self.m_block_size % tO_layout.shape[0] == 0
            vO_layout = cute.make_layout((1, async_copy_elems))
            gmem_tiled_copy_O = cute.make_tiled_copy_tv(atom_universal_copy, tO_layout, vO_layout)

        self.overlap_sO_sQ = (
            (self.head_dim_padded == 192 and self.head_dim_v_padded >= 64) or
            (self.head_dim_v_padded >= 128 and self.is_split_kv) or
            self.single_q_stage
        )
        if const_expr(self.enable_flashmask):
            self.overlap_sO_sQ = True
        if const_expr(self.overlap_sO_sQ):
            self.is_persistent = False
        # d == dv == 256 flashmask: persistent CTAs with dynamic (atomic ticket) scheduling so
        # the next tile's block list / Q / K0 / QK0 overlap the current tile's tail instead of
        # paying CTA launch + prologue + cold start per tile. O is written straight from the
        # correction warps' registers to gmem, so sO (aliased with sQ) is never touched and the
        # next Q can land as soon as the last QK of the current tile has read sQ.
        self.fm_persistent = const_expr(
            all((
                self.enable_flashmask,
                mSchedCounter is not None,
                self.single_q_stage,
                not self.use_2cta_instrs,
                not self.folded_acc,
                not self.is_split_kv,
                not self.pack_gqa,
                not self.enable_overlap,
                blocksparse_tensors is None,
                self.use_tma_O,
                self.head_dim_padded == 256,
                self.head_dim_v_padded == 256,
                not self.is_local,
            ))
        )
        self.sched_ring_stages = 4
        # Every warp but the producer (generate_block warp) reads the ticket ring.
        self.sched_consumer_threads = cute.arch.WARP_SIZE * (
            self.num_warps - len(self.generate_block_warp_ids) - len(self.empty_warp_ids)
        )
        if const_expr(self.fm_persistent):
            self.is_persistent = True
            TileScheduler = DynamicPersistentTileScheduler
            # At 32 registers the load / generate_block warps spill their startend LDG
            # address math (sm_103a: STACK 32, 12 STL/LDL in the per-block staging loop).
            # 128 threads x (softmax + correction + other) must fit in 64K registers.
            self.num_regs_softmax, self.num_regs_correction, self.num_regs_other = 208, 128, 64
            assert self.num_regs_softmax + self.num_regs_correction + self.num_regs_other <= 512
        elif const_expr(self.is_causal or self.is_local):
            TileScheduler = SingleTileLPTScheduler
        elif const_expr(self.is_persistent):
            TileScheduler = StaticPersistentTileScheduler
        else:
            TileScheduler = SingleTileScheduler
        # fm_persistent O write-back through a small sO ring (32 columns x 2 stages) + TMA.
        # sQ cannot hold O here (the next tile's Q is already
        # landing), and ~19 KB of SMEM is left, so O goes out in (m_block, cols) chunks: the
        # correction warps stage chunk c, the epilogue warp TMA-stores it and frees the slot.
        # TMEM O is released once the last chunk is in smem instead of after 64 KB of strided
        # st.global (4.2 us tail).
        self.sOc_layout = None
        tma_atom_Oc, mOc = None, None
        if const_expr(self.fm_persistent):
            self.o_chunk_cols, self.o_chunk_stages = 32, 2
            # Per-tile uses of each slot must be even so every tile starts on the same parity.
            assert (self.head_dim_v_padded // (self.o_chunk_cols * self.o_chunk_stages)) % 2 == 0
            self.sOc_layout = sm100_utils_basic.make_smem_layout_epi(
                self.o_dtype, self.o_layout, (self.m_block_size, self.o_chunk_cols), self.o_chunk_stages
            )
            tma_atom_Oc, mOc = cpasync.make_tiled_tma_atom(
                tma_store_op,
                mO_gmem,
                cute.select(self.sOc_layout, mode=[0, 1]),
                cute.composition(
                    cute.make_identity_layout(mO_gmem.shape), (self.m_block_size, self.o_chunk_cols)
                ),
            )

        # A CTA pair covers `cta_group_size * cta_tiler[0]` query rows (the 2-CTA UMMA
        # splits M across the pair), so one work tile spans that many rows.
        self.work_tile_m = self.cta_tiler[0] * self.cta_group_size
        tile_sched_args = TileSchedulerArguments(
            cute.ceil_div(cute.size(mQ.shape[0]), self.work_tile_m),
            cute.size(mQ.shape[2]),
            cute.size(mQ.shape[3]),
            num_splits,
            cute.size(mK.shape[0]),
            mQ.shape[1],
            mV.shape[0],  # Note that this is different from Sm90 since we transpose mV in Sm100
            total_q=cute.size(mQ.shape[0]) * cute.size(mQ.shape[3]),
            tile_shape_mn=(self.work_tile_m, self.cta_tiler[1]),
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
            element_size=self.k_dtype.width // 8,
            is_persistent=self.is_persistent,
            lpt=self.is_causal or self.is_local,
            is_split_kv=self.is_split_kv,
            cluster_shape_mn=self.cluster_shape_mn,
            # num_block above is counted in work_tile_m (= cta_tiler[0] * cta_group_size)
            # rows, i.e. per CTA PAIR, so the whole cluster must resolve the same tile.
            cluster_share_tile=self.cta_group_size > 1,
        )
        tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
        self.tile_scheduler_cls = TileScheduler
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)

        self.mbar_load_q_full_offset = 0
        self.mbar_load_q_empty_offset = self.mbar_load_q_full_offset + self.q_stage
        self.mbar_load_kv_full_offset = self.mbar_load_q_empty_offset + self.q_stage
        self.mbar_load_kv_empty_offset = self.mbar_load_kv_full_offset + self.kv_stage
        self.mbar_P_full_O_rescaled_offset = self.mbar_load_kv_empty_offset + self.kv_stage
        self.mbar_S_full_offset = self.mbar_P_full_O_rescaled_offset + 2
        self.mbar_O_full_offset = self.mbar_S_full_offset + 2
        self.mbar_softmax_corr_full_offset = self.mbar_O_full_offset + 2
        self.mbar_softmax_corr_empty_offset = self.mbar_softmax_corr_full_offset + 2
        self.mbar_corr_epi_full_offset = self.mbar_softmax_corr_empty_offset + 2  # softmax_corr_empty always has 2 barriers
        self.mbar_corr_epi_empty_offset = self.mbar_corr_epi_full_offset + self.q_stage  # corr_epi_full has q_stage barriers
        self.mbar_s0_s1_sequence_offset = self.mbar_corr_epi_empty_offset + 2
        self.mbar_tmem_dealloc_offset = self.mbar_s0_s1_sequence_offset + 8
        self.mbar_P_full_2_offset = self.mbar_tmem_dealloc_offset + 1
        self.mbar_generate_block_full_offset = self.mbar_P_full_2_offset + 2
        # ------------------------------------------------------------------
        # single_q_stage barrier invariants (q_stage == num_s_stages == 1, i.e. d == dv == 256
        # or the big-d d>256 / dv<=256 config):
        #   - Every stage-indexed barrier (load_q_full/empty, S_full, O_full, P_full_2,
        #     P_full_O_rescaled, softmax_corr_full/empty, load_startend_row_indices) is
        #     driven only for stage 0: the MMA warp issues one QK and one PV gemm per KV
        #     tile, softmax1's warps skip softmax_loop entirely, and correction hands the
        #     single stage back to softmax0 instead of cross-signalling stage 1.
        #   - The slots for stage 1 are still allocated (the offsets below reserve 2) but
        #     are neither initialized as consumers nor waited on.
        # If you add a stage-indexed arrive/wait, drive it with range(self.num_s_stages)
        # so both the 1-stage and the 2-stage kernel stay balanced.
        # ------------------------------------------------------------------
        self.mbar_generate_block_empty_offset = self.mbar_generate_block_full_offset + self.generate_block_stage

        self.mbar_load_startend_row_indices_full_offset = self.mbar_generate_block_empty_offset + self.generate_block_stage
        self.mbar_load_startend_row_indices_empty_offset = self.mbar_load_startend_row_indices_full_offset + self.kv_stage * 2

        self.mbar_total = self.mbar_load_startend_row_indices_empty_offset + self.kv_stage * 2
        self.mbar_sched_offset = self.mbar_total  # full[ring_stages], empty[ring_stages]
        if const_expr(self.fm_persistent):
            self.mbar_total = self.mbar_total + 2 * self.sched_ring_stages
        self.mbar_o_chunk_offset = self.mbar_total  # full[o_chunk_stages], empty[o_chunk_stages]
        if const_expr(self.fm_persistent):
            self.mbar_total = self.mbar_total + 2 * self.o_chunk_stages

        sO_size = cute.cosize(sO_layout) if const_expr(not self.overlap_sO_sQ) else 0
        # Folded accumulator: P lives in SMEM (see the p_source comment) in exactly the layout
        # the PV MMA wants for an A-from-SMEM operand, i.e. tP_layout. 8KB at m=64/n=64.
        sP_size = cute.cosize(tP_layout) if const_expr(self.folded_acc) else 0
        sQ_size = (
            cute.cosize(sQ_layout) if const_expr(not self.overlap_sO_sQ) else
            cutlass.max(cute.cosize(sQ_layout), cute.cosize(sO_layout) * self.o_dtype.width // self.q_dtype.width)
        )

        # The flashmask block max/min working buffer is sized in *blocks*
        # (generate_block_seqlen_k / n_block_size), so a small KV tile inflates its smem
        # footprint (4x at n=32 vs n=128). With d=576 (sQ + sK already ~216KB) that
        # overflows the ~227KB per-CTA budget -> CUDA_ERROR_INVALID_VALUE at launch.
        # Keep the buffer at a fixed block count for small tiles. This only caps the
        # generation *window*: generate_block() streams the real seqlen_k in chunks
        # (`num_chunks = ceil(num_blocks / usable_block_count)`), so a large seqlen_k
        # just refills more often.
        if const_expr(self.n_block_size >= 64):
            self.generate_block_seqlen_k = Int32(1024 * 16)
        else:
            self.generate_block_seqlen_k = Int32(64 * self.n_block_size)
        self.generate_block_buffer_block_count = Int32(Int32(((self.generate_block_seqlen_k + self.n_block_size - 1) // self.n_block_size + 31)) & 0xffffffe0)
        self.generate_block_buffer_usable_block_count = Int32(((self.generate_block_seqlen_k + self.n_block_size - 1) // self.n_block_size + 3) // 4 * 4)
        assert self.generate_block_buffer_usable_block_count % (len(self.generate_block_warp_ids) * cute.arch.WARP_SIZE) == 0

        @cute.struct
        class SharedStorage:
            sO: cute.struct.Align[
                cute.struct.MemRange[self.o_dtype, sO_size],
                self.buffer_align_bytes,
            ]
            sQ: cute.struct.Align[
                cute.struct.MemRange[self.q_dtype, sQ_size],
                self.buffer_align_bytes,
            ]
            sK: cute.struct.Align[
                # cute.cosize(sK_layout) is correct even in the case of self.uneven_kv_smem
                cute.struct.MemRange[self.k_dtype, cute.cosize(sK_layout)],
                self.buffer_align_bytes,
            ]
            sP: cute.struct.Align[
                cute.struct.MemRange[self.q_dtype, sP_size],
                self.buffer_align_bytes,
            ]
            sOc: cute.struct.Align[
                cute.struct.MemRange[
                    self.o_dtype, cute.cosize(self.sOc_layout) if const_expr(self.fm_persistent) else 0
                ],
                self.buffer_align_bytes,
            ]
            # m_barriers for pipelines
            mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.mbar_total]
            # Tmem holding buffer
            tmem_holding_buf: Int32
            # MMA warp scratch for the descriptor-start bounce (fm_persistent); 8B aligned.
            mma_desc_scratch: cute.struct.MemRange[Int32, 2]
            # Smem tensors
            # store row max and row sum
            sScale: cute.struct.MemRange[Float32, self.sScale_size]

            s_startend_row_indices_size = 0
            s_startend_row_indices_block_max_min_size = 0
            s_n_block_size = 0
            s_extra_flags_size = 0

            if const_expr(self.enable_flashmask):
                s_startend_row_indices_size = 4 * self.n_block_size * self.kv_stage
                s_startend_row_indices_block_max_min_size = 8 * self.generate_block_buffer_block_count * self.generate_block_stage
                s_n_block_size = self.generate_block_buffer_block_count * self.generate_block_stage
                s_extra_flags_size = 4

            s_startend_row_indices: cute.struct.MemRange[Int32, s_startend_row_indices_size]
            s_startend_row_indices_block_max_min: cute.struct.Align[
                cute.struct.MemRange[Int32, s_startend_row_indices_block_max_min_size],
                128,
            ]
            s_n_block: cute.struct.MemRange[Int32, s_n_block_size]
            s_extra_flags: cute.struct.MemRange[Int32, s_extra_flags_size]
            sched_tile: cute.struct.MemRange[Int32, self.sched_ring_stages if self.fm_persistent else 0]

        self.shared_storage = SharedStorage

        LOG2_E = math.log2(math.e)
        softmax_scale_log2 = softmax_scale * LOG2_E
        softmax_scale = None

        if const_expr(window_size_left is not None):
            window_size_left = Int32(window_size_left)
        if const_expr(window_size_right is not None):
            window_size_right = Int32(window_size_right)

        self.use_block_sparsity = cutlass.const_expr(blocksparse_tensors is not None)
        # single_q_stage (q_stage=1, head_dim=256) shorts out the stage-1 softmax/correction/PV
        # path and assumes no other code path also drains those barriers. The block-sparse
        # empty-tile helper handle_block_sparse_empty_tile_correction_sm100 has not been
        # audited for q_stage=1; reject the combination until that audit lands.
        if cutlass.const_expr(self.single_q_stage):
            assert not self.use_block_sparsity, (
                "single_q_stage (head_dim=256) is not supported together with block sparsity; "
                "the stage-1 barrier drain pattern has not been audited in "
                "handle_block_sparse_empty_tile_correction_sm100."
            )

        # S ping-pong (N direction) for d == dv == 256 (dense and flashmask): two S/P TMEM slots
        # (columns [0, 128) and [128, 256), the second one otherwise idle) hold the scores of
        # two CONSECUTIVE KV blocks of the same Q tile. The MMA warp issues QK(j+1) before
        # PV(j), so the tensor core computes QK(j+1) while the softmax warps work on S(j).
        # Both PVs accumulate into the single O at tmem_o_offset[0].
        self.use_s_ping_pong = const_expr(
            all((
                self.single_q_stage,
                not self.use_block_sparsity,
                not self.use_2cta_instrs or self.hd256_2cta,
                not self.folded_acc,
                not self.is_split_kv,
                not self.uneven_kv_smem,
                self.split_p_store,
                self.head_dim_padded == 256,
                self.head_dim_v_padded == 256,
            ))
        )
        # hd256 CTA pair: correction waits for O only before a rescale. The
        # 1-CTA ping-pong keeps its lock-step O_full wait, except in the flashmask persistent
        # kernel (the slot/phase bookkeeping is the same global block count).
        # 1-CTA flashmask persistent: the softmax warps' per-tile code no longer fits the
        # 32 KB per-partition L0 instruction cache together with the correction warp's (ncu:
        # ~85% of softmax no_inst stalls sit on the first instruction of a 128 B line;
        # softmax + correction was ~40 KB). So run every block of the tile through ONE
        # softmax_step instance: the causal / seqlen R2P pass and the flashmask pass are
        # runtime-guarded inside it, the first block takes the regular path, and the S load /
        # P store (and the MMA warp's QK / PV issue) pick the ping-pong slot by TMEM address
        # arithmetic instead of two copies.
        self.fm_one_softmax = const_expr(self.fm_persistent and self.use_s_ping_pong)
        # On top of fm_one_softmax: the correction warps' final O normalization runs as a
        # runtime loop over the sOc chunks instead of being unrolled over all of dv. It runs
        # once per tile, but its ~530 unrolled instructions sit in every SMSP's L0
        # instruction cache next to softmax.
        self.fm_corr_epi_loop = const_expr(self.fm_one_softmax and not self.folded_acc)
        # On top of fm_one_softmax: shrink the load warp's code (it shares SMSP 2 with softmax
        # warp 2 / correction warp 6). The first block's K + startend loads go through the main
        # loop instead of a peeled copy, and the startend copy for <= 2 bound columns loads
        # everything first and then stores, instead of a while loop that ptxas unrolls with
        # remainders.
        self.fm_load_one_loop = const_expr(self.fm_one_softmax and self.q_stage == 1)
        self.pp_lazy_o_wait = const_expr(
            self.use_s_ping_pong and (self.hd256_2cta or self.fm_persistent)
        )

        # d == dv == 256 flashmask (1-CTA single_q_stage): a query block with
        # valid_block_count == 0 has no MMA / correction work and its O tile used to rely on
        # the wrapper zero-filling the whole output. The epilogue warp writes those tiles'
        # zeros itself instead, so ops/fwd.py can allocate O uninitialized for this config.
        self.zero_empty_o_tiles = const_expr(
            all((
                self.enable_flashmask,
                self.single_q_stage,
                not self.use_2cta_instrs,
                not self.folded_acc,
                not self.is_split_kv,
                not self.pack_gqa,
                self.use_tma_O,
                self.head_dim_padded == 256,
                self.head_dim_v_padded == 256,
            ))
        )

        # Launch the kernel synchronously
        self.kernel(
            mQ,
            mK,
            mV,
            mO,
            mLSE,
            tma_atom_Q,
            tma_atom_K,
            tma_atom_V,
            tma_atom_O,
            softmax_scale_log2,
            softmax_scale,
            window_size_left,
            window_size_right,
            learnable_sink,
            blocksparse_tensors,
            sQ_layout,
            sK_layout,
            tP_layout,
            sV_layout,
            sO_layout,
            gmem_tiled_copy_O,
            tiled_mma_qk,
            tiled_mma_pv,
            tile_sched_params,
            num_splits,
            flashmask_info,
            mBlockLogit if const_expr(self.has_block_logit) else None,
            mBlockBos if const_expr(self.has_block_bos) else None,
            overlap_info,
            mSchedCounter,
            mO_gmem if const_expr(self.fm_persistent) else None,
            tma_atom_Oc,
            mOc,
            self.sOc_layout,
        ).launch(
            grid=grid_dim,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk,
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
        )

    #  GPU device kernel
    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,  # (s_q, d, h, b)
        mK: cute.Tensor,  # (s_k, d, h_k, b_k)
        mV: cute.Tensor,  # (d, s_k, h_k, b_k)
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: Optional[cute.CopyAtom],
        tma_atom_V: Optional[cute.CopyAtom],
        tma_atom_O: Optional[cute.CopyAtom],
        softmax_scale_log2: Float32,
        softmax_scale: Float32 | None,
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
        learnable_sink: Optional[cute.Tensor],
        blocksparse_tensors: Optional[BlockSparseTensors],
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        tP_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sO_layout: cute.ComposedLayout,
        gmem_tiled_copy_O: Optional[cute.TiledCopy],
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tile_sched_params: ParamsBase,
        num_splits: Int32,
        flashmask_info: Optional[FlashMaskInfo] = None,
        mBlockLogit: Optional[cute.Tensor] = None,
        mBlockBos: Optional[cute.Tensor] = None,
        overlap_info: Optional[OverlapInfo] = None,
        mSchedCounter: Optional[cute.Tensor] = None,
        mO_gmem: Optional[cute.Tensor] = None,
        tma_atom_Oc: Optional[cute.CopyAtom] = None,
        mOc: Optional[cute.Tensor] = None,
        sOc_layout: Optional[cute.ComposedLayout] = None,
    ):
        """The device kernel implementation of the Fused Multi-Head Attention.

        This kernel coordinates multiple specialized warps to perform different phases of the FMHA computation:
        1. Load warp: Loads Q, K, V data from global memory to shared memory using TMA
        2. MMA warp: Performs matrix multiplications (Q*K^T and P*V)
        3. Softmax warps: Compute softmax normalization on attention scores
        4. Correction warps: Apply adjustments to intermediate results
        5. Epilogue warp: Handles final output transformation and storage

        The kernel implements a complex pipeline with overlapping computation and memory operations,
        using tensor memory access (TMA) for efficient data loading, warp specialization for different
        computation phases, and optional attention masking.
        """

        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        # Prefetch tma descriptor
        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_Q)
            if const_expr(tma_atom_K is not None):
                cpasync.prefetch_descriptor(tma_atom_K)
            if const_expr(tma_atom_V is not None):
                cpasync.prefetch_descriptor(tma_atom_V)
            if const_expr(tma_atom_O is not None):
                cpasync.prefetch_descriptor(tma_atom_O)
            if const_expr(tma_atom_Oc is not None):
                cpasync.prefetch_descriptor(tma_atom_Oc)

        # Alloc
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        mbar_ptr = storage.mbar_ptr.data_ptr()
        # Use the first N warps to initialize barriers
        if warp_idx == self.mbar_init_warp[1]:
            # Init "full" barrier with number of producers, "empty" barrier with number of consumers
            for i in cutlass.range_constexpr(self.q_stage):
                cute.arch.mbarrier_init(
                    mbar_ptr + self.mbar_load_q_full_offset + i, 1
                )
                cute.arch.mbarrier_init(
                    mbar_ptr + self.mbar_load_q_empty_offset + i, len([self.mma_warp_id])
                )
        if warp_idx == self.mbar_init_warp[2]:
            for i in cutlass.range_constexpr(2):
                # `empty` is driven by the correction warps, `full` by the softmax warps.
                # Derive the counts from the warp lists: the 2-CTA layout only has 2 warps
                # per group (64 rows per CTA), not 4.
                cute.arch.mbarrier_init(
                    mbar_ptr + self.mbar_softmax_corr_empty_offset + i,
                    cute.arch.WARP_SIZE * len(self.correction_warp_ids),
                )
                cute.arch.mbarrier_init(
                    mbar_ptr + self.mbar_softmax_corr_full_offset + i,
                    cute.arch.WARP_SIZE * len(self.softmax0_warp_ids),
                )
        if warp_idx == self.mbar_init_warp[3]:
            if const_expr(self.s0_s1_barrier):
                for i in cutlass.range_constexpr(8):
                    cute.arch.mbarrier_init(
                        mbar_ptr + self.mbar_s0_s1_sequence_offset + i, cute.arch.WARP_SIZE
                    )
        if warp_idx == self.mbar_init_warp[4]:
            for i in cutlass.range_constexpr(self.q_stage):
                cute.arch.mbarrier_init(
                    mbar_ptr + self.mbar_corr_epi_full_offset + i,
                    cute.arch.WARP_SIZE * len(self.correction_warp_ids),
                )
                cute.arch.mbarrier_init(
                    mbar_ptr + self.mbar_corr_epi_empty_offset + i,
                    cute.arch.WARP_SIZE * len(self.epilogue_warp_ids),
                )
            if const_expr(self.fm_persistent):
                # O chunk ring: full <- every correction thread, empty <- the epilogue warp.
                for i in cutlass.range_constexpr(self.o_chunk_stages):
                    cute.arch.mbarrier_init(
                        mbar_ptr + self.mbar_o_chunk_offset + i,
                        cute.arch.WARP_SIZE * len(self.correction_warp_ids),
                    )
                    cute.arch.mbarrier_init(
                        mbar_ptr + self.mbar_o_chunk_offset + self.o_chunk_stages + i,
                        cute.arch.WARP_SIZE * len(self.epilogue_warp_ids),
                    )
        if warp_idx == self.mbar_init_warp[5]:
            for i in cutlass.range_constexpr(2):
                # P_full_O_rescaled / P_full_2 / tmem_dealloc are waited on by the MMA warp,
                # which only exists in the leader CTA. Both CTAs of the pair arrive on the
                # leader's copy (see mma_barrier_arrive), so the expected count scales with
                # cta_group_size. The peer CTA's own copy is initialized identically but is
                # never waited on.
                cute.arch.mbarrier_init(
                    mbar_ptr + self.mbar_P_full_O_rescaled_offset + i,
                    cute.arch.WARP_SIZE
                    * (len(self.softmax0_warp_ids) + len(self.correction_warp_ids))
                    * self.cta_group_size,
                )
                cute.arch.mbarrier_init(
                    mbar_ptr + self.mbar_S_full_offset + i, len([self.mma_warp_id])
                )
                cute.arch.mbarrier_init(
                    mbar_ptr + self.mbar_O_full_offset + i, len([self.mma_warp_id])
                )
        if warp_idx == self.mbar_init_warp[6]:
            for i in cutlass.range_constexpr(2):
                cute.arch.mbarrier_init(
                    mbar_ptr + self.mbar_P_full_2_offset + i,
                    cute.arch.WARP_SIZE * len(self.softmax0_warp_ids) * self.cta_group_size,
                )
        if warp_idx == self.mbar_init_warp[7]:
            cute.arch.mbarrier_init(
                mbar_ptr + self.mbar_tmem_dealloc_offset,
                cute.arch.WARP_SIZE
                * len(
                    (
                        *self.softmax0_warp_ids,
                        *self.softmax1_warp_ids,
                        *self.correction_warp_ids,
                    )
                ),
            )
        # not really sure why only one warp to init barrier here, why not one thread? why PipelineX.create use the whole cta? ptx doc says mbarrier can only init once
        if warp_idx == self.mbar_init_warp[8]:
            for i in cutlass.range_constexpr(2):
                for j in cutlass.range_constexpr(self.kv_stage):
                    cute.arch.mbarrier_init(
                        mbar_ptr + self.mbar_load_startend_row_indices_full_offset + i * self.kv_stage + j,
                        cute.arch.WARP_SIZE * len(self.load_warp_ids)
                    )
                    cute.arch.mbarrier_init(
                        mbar_ptr + self.mbar_load_startend_row_indices_empty_offset + i * self.kv_stage + j,
                        cute.arch.WARP_SIZE * len(self.softmax0_warp_ids)
                    )
        if warp_idx == self.mbar_init_warp[9]:
            for i in cutlass.range_constexpr(self.generate_block_stage):
                cute.arch.mbarrier_init(
                    mbar_ptr + self.mbar_generate_block_full_offset + i,
                    cute.arch.WARP_SIZE * len(self.generate_block_warp_ids)
                )
                # Only the softmax warpgroups that actually run softmax_loop consume the
                # block list, so single_q_stage (num_s_stages == 1) counts softmax0 alone.
                if const_expr(self.num_s_stages == 2):
                    generate_block_empty_warps = (
                        self.load_warp_ids + self.softmax0_warp_ids + self.softmax1_warp_ids
                    )
                else:
                    generate_block_empty_warps = self.load_warp_ids + self.softmax0_warp_ids
                cute.arch.mbarrier_init(
                    mbar_ptr + self.mbar_generate_block_empty_offset + i,
                    cute.arch.WARP_SIZE * len(generate_block_empty_warps)
                )

        if const_expr(self.fm_persistent):
            if warp_idx == self.mbar_init_warp[10]:
                DynamicPersistentTileScheduler.init_barriers(
                    mbar_ptr + self.mbar_sched_offset,
                    self.sched_ring_stages,
                    self.sched_consumer_threads,
                )
        # Relying on pipeline_kv constructor to call mbarrier_init_fence and sync
        pipeline_kv = self.make_and_init_load_kv_pipeline(mbar_ptr + self.mbar_load_kv_full_offset)

        #  Generate smem tensor Q/K/V/O
        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        # (MMA, MMA_K, MMA_D, PIPE)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        # (MMA, MMA_K, MMA_D, PIPE)
        # Strip swizzle info to reuse smem
        sV = cute.make_tensor(cute.recast_ptr(sK.iterator, sV_layout.inner), sV_layout.outer)
        if const_expr(not self.overlap_sO_sQ):
            sO = storage.sO.get_tensor(sO_layout.outer, swizzle=sO_layout.inner)
        else:
            sO = cute.make_tensor(cute.recast_ptr(sQ.iterator, sO_layout.inner, self.o_dtype), sO_layout.outer)
        sOc, sOc_full_view = None, None
        if const_expr(self.fm_persistent):
            # (m, cols, stages) ring the epilogue warp TMA-stores from.
            sOc_ptr = cute.recast_ptr(storage.sOc.data_ptr(), sOc_layout.inner, self.o_dtype)
            sOc = cute.make_tensor(sOc_ptr, sOc_layout.outer)
            # The same ring seen as one (m, dv) tile: column n = j + cols * (s + stages * r) maps
            # to slot s; the repeat mode r has stride 0, so correction_epilogue's (m, dv)
            # partition walks the ring and only has to sync at chunk boundaries.
            oc = sOc_layout.outer
            oc_repeat = self.head_dim_v_padded // (self.o_chunk_cols * self.o_chunk_stages)
            sOc_full_view = cute.make_tensor(
                sOc_ptr,
                cute.make_layout(
                    (oc.shape[0], (oc.shape[1], oc.shape[2], oc_repeat)),
                    stride=(oc.stride[0], (oc.stride[1], oc.stride[2], 0)),
                ),
            )

        sScale = storage.sScale.get_tensor(cute.make_layout(self.sScale_size))
        mma_desc_scratch_ptr = storage.mma_desc_scratch.data_ptr()

        # V-mode coordinate of this CTA inside the MMA's CTA pair. For cta_group=1 this is
        # always 0; for cta_group=2 it selects which half of the MMA tiler's M rows (and
        # therefore which half of the accumulator / of the Q rows) this CTA owns.
        if const_expr(self.cta_group_size == 1):
            mma_tile_coord_v = Int32(0)
        else:
            mma_tile_coord_v = self.cta_coord_v()
        is_leader_cta = mma_tile_coord_v == 0

        thr_mma_qk = tiled_mma_qk.get_slice(mma_tile_coord_v)
        thr_mma_pv = tiled_mma_pv.get_slice(mma_tile_coord_v)

        qk_acc_shape = thr_mma_qk.partition_shape_C(self.mma_tiler_qk[:2])
        tStS_fake = thr_mma_qk.make_fragment_C(qk_acc_shape)
        # This is a fake tensor, by right need to retrieve tmem_ptr. But we know that we always
        # request 512 columns of tmem, so we know that it starts at 0.
        tmem_ptr = cute.make_ptr(Float32, 0, mem_space=cute.AddressSpace.tmem, assumed_align=16)
        tStS = cute.make_tensor(tmem_ptr, tStS_fake.layout)

        pv_acc_shape = thr_mma_pv.partition_shape_C(self.mma_tiler_pv[:2])
        tOtO = thr_mma_pv.make_fragment_C(pv_acc_shape)

        tStSs = tuple(
            cute.make_tensor(tStS.iterator + self.tmem_s_offset[stage], tStS.layout)
            for stage in range(2)
        )
        tOtOs = tuple(
            cute.make_tensor(
                # A FRESH tmem pointer, not `tOtO.iterator + offset`: iterator arithmetic
                # loses the pointer's alignment (the memref comes out as align<1>), and the
                # epilogue's tmem load atom for a 64-row epi tile is
                # tmem_load<f32, 16 DP, 256 bit, x2>, which requires the tmem memref to be at
                # least 2-column (8B) aligned. Every tmem_o_offset is a multiple of 4 columns
                # (asserted in __init__), so 16B is a safe claim.
                cute.make_ptr(
                    Float32,
                    self.tmem_o_offset[stage],
                    mem_space=cute.AddressSpace.tmem,
                    assumed_align=16,
                ),
                tOtO.layout,
            )
            for stage in range(self.q_stage)
        )

        if const_expr(not self.folded_acc):
            tP = cute.make_tensor(tStS.iterator, tP_layout.outer)
            tOrP = thr_mma_pv.make_fragment_A(tP)[None, None, None, 0]

            tOrPs = [
                cute.make_tensor(
                    tOrP.iterator
                    + self.qk_acc_dtype.width // self.q_dtype.width * self.tmem_p_offset[stage],
                    tOrP.layout,
                )
                for stage in range(2)
            ]
            sP = None
        else:
            # P in SMEM (see p_source): one buffer, in the MMA's own A-operand layout. The
            # softmax warps write it, the MMA warp reads it as A; the existing
            # S_full / P_full_O_rescaled handshake already serializes the two.
            sP = storage.sP.get_tensor(tP_layout.outer, swizzle=tP_layout.inner)
            tOrP = thr_mma_pv.make_fragment_A(sP)[None, None, None, 0]
            tOrPs = [tOrP, tOrP]

        block_info = BlockInfo(
            # M granularity of a *work tile*, not of this CTA's slice: `m_block` here is the
            # work tile index produced by the tile scheduler, and with a 2-CTA UMMA one work
            # tile spans cta_group_size * cta_tiler[0] rows. Using cta_tiler[0] would halve
            # n_block_max under cta_group=2, which truncates the generate_block list (and can
            # empty it) while valid_block_count is still built at work-tile granularity ->
            # load/softmax skip the tile, mma/correction/epilogue wait forever.
            # For cta_group=1 work_tile_m == cta_tiler[0], so this is a no-op there.
            self.work_tile_m,
            self.cta_tiler[1],
            self.is_causal,
            self.is_local,
            self.is_split_kv,
            window_size_left,
            window_size_right,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
        )
        SeqlenInfoCls = partial(
            SeqlenInfoQK.create,
            seqlen_q_static=mQ.shape[0] if const_expr(not self.pack_gqa) else mQ.shape[0][1],
            seqlen_k_static=mK.shape[0],
        )
        AttentionMaskCls = partial(
            AttentionMask,
            self.m_block_size,
            self.n_block_size,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
        )
        if const_expr(self.fm_persistent):
            TileSchedulerCls = partial(
                DynamicPersistentTileScheduler.create,
                tile_sched_params,
                storage.sched_tile.get_tensor(cute.make_layout(self.sched_ring_stages)),
                mbar_ptr + self.mbar_sched_offset,
                mSchedCounter,
                self.sched_ring_stages,
                self.sched_consumer_threads,
            )
        else:
            TileSchedulerCls = partial(self.tile_scheduler_cls.create, tile_sched_params)

        # prepare input for generate_block
        if const_expr(self.enable_flashmask):
            s_startend_row_indices_block_max_min = storage.s_startend_row_indices_block_max_min.get_tensor(
                cute.make_layout(8 * self.generate_block_buffer_block_count * self.generate_block_stage),
            )
            s_n_block = storage.s_n_block.get_tensor(
                cute.make_layout(self.generate_block_buffer_block_count * self.generate_block_stage),
            )
            s_extra_flags = storage.s_extra_flags.get_tensor(
                cute.make_layout(4),
            )
            s_startend_row_indices = storage.s_startend_row_indices.get_tensor(
                cute.make_layout(4 * self.n_block_size * self.kv_stage),
            )
        else:
            s_startend_row_indices_block_max_min = None
            s_n_block = None
            s_extra_flags = None
            s_startend_row_indices = None

        # ///////////////////////////////////////////////////////////////////////////////
        #  EMPTY
        # ///////////////////////////////////////////////////////////////////////////////
        if const_expr(len(self.empty_warp_ids) > 0):
            if warp_idx == self.empty_warp_ids[0]:
                cute.arch.warpgroup_reg_dealloc(self.num_regs_empty)

        if const_expr(len(self.empty_warp_ids) > 1):
            if warp_idx == self.empty_warp_ids[1]:
                cute.arch.warpgroup_reg_dealloc(self.num_regs_empty)

        assert len(self.empty_warp_ids) <= 2

        # ///////////////////////////////////////////////////////////////////////////////
        #  GENERATE BLOCK
        # ///////////////////////////////////////////////////////////////////////////////
        if const_expr(self.enable_flashmask):
            if warp_idx >= self.generate_block_warp_ids[0] and warp_idx <= self.generate_block_warp_ids[-1]:
                # TODO: tune reg for generate block
                cute.arch.warpgroup_reg_dealloc(self.num_regs_other)
                self.generate_block(
                    s_startend_row_indices_block_max_min,
                    s_n_block,
                    s_extra_flags,
                    block_info,
                    num_splits,
                    SeqlenInfoCls,
                    TileSchedulerCls,
                    mQ.shape[2], # (s_q, d, h, b)
                    flashmask_info,
                    mbar_ptr,
                )
        else:
            if warp_idx == self.generate_block_warp_ids[0]:
                cute.arch.warpgroup_reg_dealloc(self.num_regs_empty)

        # ///////////////////////////////////////////////////////////////////////////////
        #  LOAD
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx >= self.load_warp_ids[0] and warp_idx <= self.load_warp_ids[-1]:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_other)
            self.load(
                thr_mma_qk,
                thr_mma_pv,
                mQ,
                mK,
                mV,
                sQ,
                sK,
                sV,
                tma_atom_Q,
                tma_atom_K,
                tma_atom_V,
                pipeline_kv,
                mbar_ptr,
                block_info,
                num_splits,
                SeqlenInfoCls,
                TileSchedulerCls,
                blocksparse_tensors,
                s_n_block,
                s_extra_flags,
                s_startend_row_indices,
                flashmask_info,
                overlap_info,
            )

        # ///////////////////////////////////////////////////////////////////////////////
        #  MMA
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.mma_warp_id:
            # if warp_idx == self.mma_warp_id or warp_idx == self.empty_warp_ids:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_other)
            # Alloc tmem buffer. `tcgen05.alloc/relinquish_alloc_permit/dealloc` with
            # `.cta_group::2` are CTA-pair *collectives*: the hardware allocator
            # (UTCATOMSWS.2CTA.FIND_AND_SET) only completes once BOTH CTAs of the pair have
            # issued their own instruction. Gating them on the leader makes the leader spin
            # in the allocator forever. So every CTA allocates / relinquishes / deallocates;
            # only the UMMA issue itself (self.mma) is leader-only. Matches
            # cutlass.utils.TmemAllocator, which never gates on the leader either.
            tmem_alloc_cols = Int32(self.tmem_alloc_cols)
            cute.arch.alloc_tmem(
                tmem_alloc_cols,
                storage.tmem_holding_buf,
                is_two_cta=self.use_2cta_instrs,
            )
            cute.arch.sync_warp()

            if is_leader_cta:
                self.mma(
                    tiled_mma_qk,
                    tiled_mma_pv,
                    sQ,
                    sK,
                    sV,
                    tStSs,
                    tOtOs,
                    tOrPs,
                    pipeline_kv,
                    mbar_ptr,
                    block_info,
                    num_splits,
                    SeqlenInfoCls,
                    TileSchedulerCls,
                    blocksparse_tensors,
                    mQ.shape[2], # (s_q, d, h, b)
                    flashmask_info,
                    mma_desc_scratch_ptr=mma_desc_scratch_ptr,
                )

            # dealloc tmem buffer. Every CTA of the pair must issue relinquish/dealloc (see
            # above), and each one gates on its OWN tmem_dealloc barrier: its local softmax /
            # correction warps are the only readers of its own tmem, and `dealloc.cta_group::2`
            # itself joins the pair, so the leader cannot free the peer's columns early.
            cute.arch.relinquish_tmem_alloc_permit(is_two_cta=self.use_2cta_instrs)
            cute.arch.mbarrier_wait(mbar_ptr + self.mbar_tmem_dealloc_offset, 0)
            tmem_alloc_cols = Int32(self.tmem_alloc_cols)
            #  Retrieving tmem ptr and make acc
            tmem_ptr = cute.arch.retrieve_tmem_ptr(
                Float32,
                alignment=16,
                ptr_to_buffer_holding_addr=storage.tmem_holding_buf,
            )
            cute.arch.dealloc_tmem(
                tmem_ptr, tmem_alloc_cols, is_two_cta=self.use_2cta_instrs
            )

        # ///////////////////////////////////////////////////////////////////////////////
        #  Epilogue
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx >= self.epilogue_warp_ids[0] and warp_idx <= self.epilogue_warp_ids[-1]:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_other)
            self.epilogue_s2g(
                mO,
                sO,
                gmem_tiled_copy_O,
                tma_atom_O,
                mbar_ptr,
                block_info,
                num_splits,
                SeqlenInfoCls,
                TileSchedulerCls,
                mQ.shape[2],
                flashmask_info,
                mO_gmem=mO_gmem,
                sOc=sOc,
                tma_atom_Oc=tma_atom_Oc,
                mOc=mOc,
            )
        # ///////////////////////////////////////////////////////////////////////////////
        #  Softmax
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx < self.correction_warp_ids[0]:
            softmax_loop = partial(
                self.softmax_loop,
                softmax_scale_log2=softmax_scale_log2,
                softmax_scale=softmax_scale,
                thr_mma_qk=thr_mma_qk,
                sScale=sScale,
                mLSE=mLSE,
                learnable_sink=learnable_sink,
                mbar_ptr=mbar_ptr,
                block_info=block_info,
                num_splits=num_splits,
                SeqlenInfoCls=SeqlenInfoCls,
                AttentionMaskCls=AttentionMaskCls,
                TileSchedulerCls=TileSchedulerCls,
                blocksparse_tensors=blocksparse_tensors,
                s_n_block=s_n_block,
                s_extra_flags=s_extra_flags,
                s_startend_row_indices=s_startend_row_indices,
                mBlockLogit=mBlockLogit,
                mBlockBos=mBlockBos,
                sP=sP,
            )

            if const_expr(self.num_s_stages == 1):
                # single_q_stage: a single Q tile and a single S/P buffer, and softmax1's warps
                # are not part of the CTA at all (see the warp id table in __init__).
                cute.arch.warpgroup_reg_alloc(self.num_regs_softmax)
                softmax_loop(
                    stage=0,
                    tStSi=cute.make_tensor(
                        tStS.iterator + self.tmem_s_offset[0], tStS.layout
                    ),
                )
                self.tmem_dealloc_arrive(mbar_ptr + self.mbar_tmem_dealloc_offset)
            elif const_expr(not self.s0_s1_barrier):
                # increase register after decreasing
                cute.arch.warpgroup_reg_alloc(self.num_regs_softmax)
                stage = Int32(0 if warp_idx < self.softmax1_warp_ids[0] else 1)
                softmax_loop(
                    stage=stage,
                    tStSi=cute.make_tensor(
                        tStS.iterator
                        + (self.tmem_s_offset[0] if stage == 0 else self.tmem_s_offset[1]),
                        tStS.layout,
                    ),
                )
                self.tmem_dealloc_arrive(mbar_ptr + self.mbar_tmem_dealloc_offset)
            else:
                cute.arch.warpgroup_reg_alloc(self.num_regs_softmax)
                # If there's s0_s1_barrier, it's faster to have 2 WGs having different code
                if warp_idx < self.softmax1_warp_ids[0]:
                    tStSi = cute.make_tensor(tStS.iterator + self.tmem_s_offset[0], tStS.layout)
                    softmax_loop(stage=0, tStSi=tStSi)
                    self.tmem_dealloc_arrive(mbar_ptr + self.mbar_tmem_dealloc_offset)
                if warp_idx < self.correction_warp_ids[0] and warp_idx >= self.softmax1_warp_ids[0]:
                    tStSi = cute.make_tensor(tStS.iterator + self.tmem_s_offset[1], tStS.layout)
                    softmax_loop(stage=1, tStSi=tStSi)
                    self.tmem_dealloc_arrive(mbar_ptr + self.mbar_tmem_dealloc_offset)

        # ///////////////////////////////////////////////////////////////////////////////
        #  Correction
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx >= self.correction_warp_ids[0] and warp_idx < self.mma_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_correction)
            self.correction_loop(
                thr_mma_qk,
                thr_mma_pv,
                tStS,
                tOtOs,
                sScale,
                mO,
                mLSE,
                sO,
                learnable_sink,
                gmem_tiled_copy_O,
                tma_atom_O,
                mbar_ptr,
                softmax_scale_log2,
                block_info,
                num_splits,
                SeqlenInfoCls,
                TileSchedulerCls,
                blocksparse_tensors,
                flashmask_info,
                mQ.shape[2], # (s_q, d, h, b)
                sOc_full_view=sOc_full_view,
            )
            self.tmem_dealloc_arrive(mbar_ptr + self.mbar_tmem_dealloc_offset)

        if const_expr(self.use_2cta_instrs):
            # Cluster tail. With cta_group=2 the two CTAs drive each other's mbarriers:
            # the peer's softmax / correction warps do REMOTE arrives into the leader's
            # smem (`mma_barrier_arrive` -> mapa), and the leader's MMA warp commits with a
            # cluster multicast mask into the peer's smem. Several of those are the *last*
            # thing a warp does for a work tile and have no waiter left (e.g. the
            # correction warp's final P_full_O_rescaled arrive, which the mma warp has
            # already stopped waiting on). Without a cluster-wide barrier here, a CTA can
            # retire while its partner still has such an arrive / commit in flight; the
            # transaction then targets a dead CTA's shared memory, which surfaces as a
            # NONDETERMINISTIC cudaErrorLaunchFailure (719) that gets more likely the more
            # CTA pairs are resident (it showed up first on batch=2840 shapes).
            # cluster_arrive/cluster_wait are CTA-aligned, so they must stay OUTSIDE every
            # `warp_idx` branch: all threads of both CTAs reach this point.
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()

        return

    @cute.jit
    def prefix_sum_kernel(self, value_per_thread: Int32) -> Int32:
        """
        modified from cutlass.utils.grouped_gemm_tile_scheduler_helper
        Perform prefix sum within a full warp.

        :param value_per_thread: The value for this thread to contribute to the prefix sum
        :type value_per_thread: Int32
        :return: The prefix sum result for this thread
        :rtype: Int32
        """
        clamp_value = 0
        idx = 1
        sum_per_thread = value_per_thread
        num_generate_block_threads = cute.arch.WARP_SIZE * len(self.generate_block_warp_ids)
        tidx = cute.arch.thread_idx()[0] % num_generate_block_threads
        lane_idx = tidx & 31
        while const_expr(idx < cute.arch.WARP_SIZE):
            value = cute.arch.shuffle_sync_up(
                sum_per_thread, idx, mask_and_clamp=clamp_value
            )
            if lane_idx >= idx:
                sum_per_thread += value
            idx = idx << 1
        return sum_per_thread

    @cute.jit
    def generate_block(
        self,
        s_startend_row_indices_block_max_min: cute.Tensor,
        s_n_block: cute.Tensor,
        s_extra_flags: cute.Tensor,
        block_info: BlockInfo,
        num_splits: Int32,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        num_heads: Int32,
        flashmask_info: FlashMaskInfo,
        mbar_ptr: cute.Tensor,
    ):
        if const_expr(self.fm_persistent):
            # The generate_block warp runs ahead of every other role, so it draws the tickets.
            tile_scheduler = TileSchedulerCls(is_producer=True)
        else:
            tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        generate_block_producer_state = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Producer, self.generate_block_stage
        )
        # TODO: check is_valid_tile in fm

        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx

            seqlen = SeqlenInfoCls(batch_idx)
            assert not self.use_block_sparsity # TODO: support block sparsity in fm
            n_block_min, n_block_max = block_info.get_n_block_min_max(
                seqlen, m_block, split_idx, num_splits
            )

            if n_block_min < n_block_max:
                # for padding 32 and padding 4: the num_chunk (pad_32) >= num_chunk (pad_4) is always true
                # TODO: support 128 padding
                num_blocks = Int32((seqlen.seqlen_k + self.n_block_size - 1) // self.n_block_size)

                num_chunks = (num_blocks + self.generate_block_buffer_usable_block_count - 1) // self.generate_block_buffer_usable_block_count
                # reverse_chunk_idx, start from right to left: [5, 4, 3, 2, 1, 0], and fwd kernel scans from right to left
                chunk_valid = True
                # TODO: support cppl_stage
                for reverse_chunk_idx in cutlass.range(num_chunks):
                    if chunk_valid:
                        cute.arch.mbarrier_wait(mbar_ptr + self.mbar_generate_block_empty_offset + generate_block_producer_state.index, generate_block_producer_state.phase)
                    self.load_startend_row_indices_block_max_min(
                        bidb=batch_idx,
                        bidh=head_idx,
                        m_block=m_block,
                        reverse_chunk_idx=reverse_chunk_idx,
                        num_blocks=num_blocks,
                        num_heads=num_heads,
                        flashmask_info=flashmask_info,
                        s_startend_row_indices_block_max_min=cute.make_tensor(s_startend_row_indices_block_max_min.iterator + 8 * self.generate_block_buffer_block_count * generate_block_producer_state.index, cute.make_layout(8 * self.generate_block_buffer_block_count)),
                    )
                    chunk_valid = self.update_block_buffer(
                        m_block=m_block,
                        reverse_chunk_idx=reverse_chunk_idx,
                        num_blocks=num_blocks,
                        end_flag=Int32(self.generate_block_finish) if reverse_chunk_idx == num_chunks - 1 else Int32(self.generate_block_incomplete),
                        n_block_min=n_block_min,
                        n_block_max=n_block_max,
                        seqlen_q=seqlen.seqlen_q,
                        s_startend_row_indices_block_max_min=cute.make_tensor(s_startend_row_indices_block_max_min.iterator + 8 * self.generate_block_buffer_block_count * generate_block_producer_state.index, cute.make_layout(8 * self.generate_block_buffer_block_count)),
                        s_n_block=cute.make_tensor(s_n_block.iterator + self.generate_block_buffer_block_count * generate_block_producer_state.index, cute.make_layout(self.generate_block_buffer_block_count)),
                        s_extra_flags=cute.make_tensor(s_extra_flags.iterator + generate_block_producer_state.index, cute.make_layout(1)),
                    )
                    if chunk_valid:
                        cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_generate_block_full_offset + generate_block_producer_state.index)
                        generate_block_producer_state.advance()

            tile_scheduler.prefetch_next_work()
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()
            # End of persistent scheduler loop

    @cute.jit
    def load_startend_row_indices_block_max_min(
        self,
        bidb: Int32,
        bidh: Int32,
        m_block: Int32,
        reverse_chunk_idx: Int32,
        num_blocks: Int32,
        num_heads: Int32,
        # TODO: deal with padding of gmem
        flashmask_info: FlashMaskInfo,
        s_startend_row_indices_block_max_min: cute.Tensor,
    ):
        h_flashmask = flashmask_info.startend_row_indices.shape[1]
        h_h_flashmask_ratio = num_heads // h_flashmask
        bidh_fm = bidh // h_h_flashmask_ratio
        num_generate_block_threads = cute.arch.WARP_SIZE * len(self.generate_block_warp_ids)
        tidx = cute.arch.thread_idx()[0] % num_generate_block_threads
        # TODO: use cp.async to use less reg
        ################################################################
        # 1. num_blocks > buffer_usable_block_count and not divisible: #
        #   num_blocks = 8, buffer_usable_block_count = 5              #
        #   x means not write                                          #
        #   second load      |   first load                            #
        #   x  x  0  1  2    |   3  4  5  6  7                         #
        #   t4 t3 t2 t1 t0   |   t4 t3 t2 t1 t0                        #
        #                                                              #
        # 2. num_blocks > buffer_usable_block_count and divisible:     #
        #   num_blocks = 10, buffer_usable_block_count = 5             #
        #   second load      |   first load                            #
        #   0  1  2  3  4    |   5  6  7  8  9                         #
        #   t4 t3 t2 t1 t0   |   t3 t2 t2 t1 t0                        #
        #                                                              #
        # 3. num_blocks == buffer_usable_block_count                   #
        #   num_blocks = 5, buffer_usable_block_count = 5              #
        #   first load                                                 #
        #   0  1  2  3  4                                              #
        #   t4 t3 t2 t1 t0                                             #
        #                                                              #
        # 4. num_blocks < buffer_usable_block_count                    #
        #   num_blocks = 3, buffer_usable_block_count = 5              #
        #   first load                                                 #
        #   x  x  0  1  2                                              #
        #   t4 t3 t2 t1 t0                                             #
        #                                                              #
        ################################################################
        s_idx = self.generate_block_buffer_usable_block_count - 1 -tidx
        g_idx = num_blocks - 1 - reverse_chunk_idx * self.generate_block_buffer_usable_block_count - tidx
        if const_expr(self.has_ut_start):
            while g_idx >= 0 and s_idx >= 0:
                s_startend_row_indices_block_max_min[self.generate_block_buffer_block_count * 0 + s_idx] = flashmask_info.LTS_nblock_max[bidb, bidh_fm, g_idx]
                s_startend_row_indices_block_max_min[self.generate_block_buffer_block_count * 1 + s_idx] = flashmask_info.LTS_nblock_min[bidb, bidh_fm, g_idx]
                s_startend_row_indices_block_max_min[self.generate_block_buffer_block_count * 2 + s_idx] = flashmask_info.LTE_nblock_max[bidb, bidh_fm, g_idx]
                s_startend_row_indices_block_max_min[self.generate_block_buffer_block_count * 3 + s_idx] = flashmask_info.LTE_nblock_min[bidb, bidh_fm, g_idx]

                s_startend_row_indices_block_max_min[self.generate_block_buffer_block_count * 4 + s_idx] = flashmask_info.UTS_nblock_max[bidb, bidh_fm, g_idx]
                s_startend_row_indices_block_max_min[self.generate_block_buffer_block_count * 5 + s_idx] = flashmask_info.UTS_nblock_min[bidb, bidh_fm, g_idx]
                s_startend_row_indices_block_max_min[self.generate_block_buffer_block_count * 6 + s_idx] = flashmask_info.UTE_nblock_max[bidb, bidh_fm, g_idx]
                s_startend_row_indices_block_max_min[self.generate_block_buffer_block_count * 7 + s_idx] = flashmask_info.UTE_nblock_min[bidb, bidh_fm, g_idx]

                s_idx -= num_generate_block_threads
                g_idx -= num_generate_block_threads
        elif const_expr(self.has_lt_end):
            while g_idx >= 0 and s_idx >= 0:
                s_startend_row_indices_block_max_min[self.generate_block_buffer_block_count * 0 + s_idx] = flashmask_info.LTS_nblock_max[bidb, bidh_fm, g_idx]
                s_startend_row_indices_block_max_min[self.generate_block_buffer_block_count * 1 + s_idx] = flashmask_info.LTS_nblock_min[bidb, bidh_fm, g_idx]
                s_startend_row_indices_block_max_min[self.generate_block_buffer_block_count * 2 + s_idx] = flashmask_info.LTE_nblock_max[bidb, bidh_fm, g_idx]
                s_startend_row_indices_block_max_min[self.generate_block_buffer_block_count * 3 + s_idx] = flashmask_info.LTE_nblock_min[bidb, bidh_fm, g_idx]
                s_idx -= num_generate_block_threads
                g_idx -= num_generate_block_threads
        elif const_expr(self.has_ut_end):
            while g_idx >= 0 and s_idx >= 0:
                s_startend_row_indices_block_max_min[self.generate_block_buffer_block_count * 0 + s_idx] = flashmask_info.LTS_nblock_max[bidb, bidh_fm, g_idx]
                s_startend_row_indices_block_max_min[self.generate_block_buffer_block_count * 1 + s_idx] = flashmask_info.LTS_nblock_min[bidb, bidh_fm, g_idx]
                s_startend_row_indices_block_max_min[self.generate_block_buffer_block_count * 6 + s_idx] = flashmask_info.UTE_nblock_max[bidb, bidh_fm, g_idx]
                s_startend_row_indices_block_max_min[self.generate_block_buffer_block_count * 7 + s_idx] = flashmask_info.UTE_nblock_min[bidb, bidh_fm, g_idx]
                s_idx -= num_generate_block_threads
                g_idx -= num_generate_block_threads
        else:
            while g_idx >= 0 and s_idx >= 0:
                s_startend_row_indices_block_max_min[self.generate_block_buffer_block_count * 0 + s_idx] = flashmask_info.LTS_nblock_max[bidb, bidh_fm, g_idx]
                s_startend_row_indices_block_max_min[self.generate_block_buffer_block_count * 1 + s_idx] = flashmask_info.LTS_nblock_min[bidb, bidh_fm, g_idx]
                s_idx -= num_generate_block_threads
                g_idx -= num_generate_block_threads

        cute.arch.sync_warp()

    @cute.jit
    def update_block_buffer(
        self,
        m_block: Int32,
        reverse_chunk_idx: Int32,
        num_blocks: Int32,
        end_flag: Int32,
        n_block_min: Int32,
        n_block_max: Int32,
        seqlen_q: Int32,
        s_startend_row_indices_block_max_min: cute.Tensor,
        s_n_block: cute.Tensor,
        s_extra_flags: cute.Tensor,
    ) -> bool:
        num_generate_block_threads = len(self.generate_block_warp_ids) * cute.arch.WARP_SIZE
        tidx = cute.arch.thread_idx()[0] % num_generate_block_threads
        lt_start_max = Int32.max
        lt_start_min = Int32.max
        lt_end_max = Int32.max
        lt_end_min = Int32.max
        ut_start_max = Int32.min
        ut_start_min = Int32.min
        ut_end_max = Int32.min
        ut_end_min = Int32.min

        s_lt_start_max = cute.make_tensor(s_startend_row_indices_block_max_min.iterator, cute.make_layout(self.generate_block_buffer_block_count))
        s_lt_start_min = cute.make_tensor(s_startend_row_indices_block_max_min.iterator + self.generate_block_buffer_block_count, cute.make_layout(self.generate_block_buffer_block_count))

        s_lt_end_max = cute.make_tensor(s_startend_row_indices_block_max_min.iterator + 2 * self.generate_block_buffer_block_count, cute.make_layout(self.generate_block_buffer_block_count))
        s_lt_end_min = cute.make_tensor(s_startend_row_indices_block_max_min.iterator + 3 * self.generate_block_buffer_block_count, cute.make_layout(self.generate_block_buffer_block_count))

        s_ut_start_max = cute.make_tensor(s_startend_row_indices_block_max_min.iterator + 4 * self.generate_block_buffer_block_count, cute.make_layout(self.generate_block_buffer_block_count))
        s_ut_start_min = cute.make_tensor(s_startend_row_indices_block_max_min.iterator + 5 * self.generate_block_buffer_block_count, cute.make_layout(self.generate_block_buffer_block_count))

        s_ut_end_max = cute.make_tensor(s_startend_row_indices_block_max_min.iterator + 6 * self.generate_block_buffer_block_count, cute.make_layout(self.generate_block_buffer_block_count))
        s_ut_end_min = cute.make_tensor(s_startend_row_indices_block_max_min.iterator + 7 * self.generate_block_buffer_block_count, cute.make_layout(self.generate_block_buffer_block_count))

        valid_n_block_num = Int32(0)
        # Row window of the whole work tile (a CTA pair covers work_tile_m rows).
        m_block_s = Int32(m_block * self.work_tile_m)
        m_block_e = cutlass.min(m_block_s + self.work_tile_m, seqlen_q)

        # The trip count MUST be warp-uniform: the body ends in a warp-wide prefix sum and
        # a `shuffle_sync_op(..., 0xffffffff)`, and both require all 32 lanes to
        # participate. The previous `while s_idx >= 0 and n_block >= loop_end` was a
        # PER-LANE condition (s_idx / n_block are offset by tidx), so whenever num_blocks
        # was not a multiple of the warp width the lanes left the loop on different
        # iterations and whoever was left hung forever on the shuffle. That showed up as a
        # kernel-wide deadlock with the generate_block warp split across two PCs (lane 0 vs
        # lanes 1..31) for every seqlen_k that is not a multiple of
        # 32 * n_block_size -- e.g. seqlen_k = 32 or 300 at n_block = 64, where
        # num_blocks % 32 is 1 / 5; seqlen_k = 8192 (num_blocks = 128) was uniform and is
        # why the dense benchmark never hit it. Even without the hang the last, divergent
        # iteration corrupts the prefix sum, so the block list itself was unreliable there.
        #
        # Iterate a uniform number of times instead and keep the per-lane range checks as
        # data guards inside the body: an out-of-range lane keeps `fully_masked` True and
        # `prefix_sum` 0, so it adds nothing to the scan and stores nothing.
        chunk_first_n_block = Int32(
            num_blocks - 1 - reverse_chunk_idx * self.generate_block_buffer_usable_block_count
        )
        # Blocks this chunk covers, capped by the buffer size and by block 0.
        chunk_blocks = cutlass.max(
            cutlass.min(
                Int32(self.generate_block_buffer_usable_block_count),
                chunk_first_n_block + 1,
            ),
            Int32(0),
        )
        num_iters = (chunk_blocks + num_generate_block_threads - 1) // num_generate_block_threads

        for it in cutlass.range(num_iters, unroll=1):
            s_idx = (
                self.generate_block_buffer_usable_block_count
                - 1
                - tidx
                - it * num_generate_block_threads
            )
            n_block = chunk_first_n_block - tidx - it * num_generate_block_threads

            prefix_sum = Int32(0)
            fully_masked = bool(True)
            partially_masked = bool(False)
            if s_idx >= 0 and n_block >= n_block_min and n_block < n_block_max:
                lt_start_max = s_lt_start_max[s_idx]
                lt_start_min = s_lt_start_min[s_idx]
                if const_expr(self.has_ut_start):
                    lt_end_max = s_lt_end_max[s_idx]
                    lt_end_min = s_lt_end_min[s_idx]
                    ut_start_max = s_ut_start_max[s_idx]
                    ut_start_min = s_ut_start_min[s_idx]
                    ut_end_max = s_ut_end_max[s_idx]
                    ut_end_min = s_ut_end_min[s_idx]

                    fully_masked = (m_block_s >= lt_start_max and m_block_e <= lt_end_min) or (m_block_s >= ut_start_max and m_block_e <= ut_end_min)
                    partially_masked = (m_block_s < lt_end_max and m_block_e > lt_start_min) or (m_block_s < ut_end_max and m_block_e > ut_start_min)
                elif const_expr(self.has_lt_end):
                    lt_end_max = s_lt_end_max[s_idx]
                    lt_end_min = s_lt_end_min[s_idx]
                    fully_masked = m_block_s >= lt_start_max and m_block_e <= lt_end_min
                    partially_masked = m_block_s < lt_end_max and m_block_e > lt_start_min
                elif const_expr(self.has_ut_end):
                    ut_end_max = s_ut_end_max[s_idx]
                    ut_end_min = s_ut_end_min[s_idx]
                    fully_masked = (m_block_s >= lt_start_max) or (m_block_e <= ut_end_min)
                    partially_masked = (m_block_e > lt_start_min) or (m_block_s < ut_end_max)
                else:
                    fully_masked = m_block_s >= lt_start_max
                    partially_masked = m_block_e > lt_start_min

                prefix_sum = Int32(0) if fully_masked else Int32(1)
            warp_id = tidx >> 5
            lane_id = tidx & 31
            # warp-wide prefix-sum
            prefix_sum = self.prefix_sum_kernel(prefix_sum)

            if not fully_masked:
                s_n_block[valid_n_block_num + prefix_sum - 1] = n_block if partially_masked else (-n_block - 1)

            prefix_sum = cute.arch.shuffle_sync_op(
                prefix_sum, 31, 0xffffffff)

            valid_n_block_num += prefix_sum

        # TODO: maybe we can remove extra_flags
        if valid_n_block_num < self.generate_block_buffer_usable_block_count:
            s_n_block[valid_n_block_num] = end_flag
        else:
            # TODO: find a way to remove [0]
            s_extra_flags[0] = end_flag

        return valid_n_block_num != 0 or end_flag != Int32(self.generate_block_incomplete)

    @cute.jit
    def load_startend_row_indices(
        self,
        bidb,
        bidh,
        n_block,
        num_heads, # num_query_heads
        load_startend_row_indices_producer_state: cutlass.pipeline.PipelineState,
        s_startend_row_indices: cute.Tensor,
        flashmask_info: FlashMaskInfo,
        mbar_ptr: cute.Pointer,
    ) -> Tuple[cutlass.pipeline.PipelineState]:

        num_load_threads = len(self.load_warp_ids) * cute.arch.WARP_SIZE

        # TODO: could be faster if we skip redundant load for the same n_block
        for softmax_wg in cutlass.range_constexpr(self.num_s_stages):
            cute.arch.mbarrier_wait(mbar_ptr + self.mbar_load_startend_row_indices_empty_offset + softmax_wg * self.kv_stage + load_startend_row_indices_producer_state.index, load_startend_row_indices_producer_state.phase)

            s_startend_row_indices_cur_stage = cute.make_tensor(s_startend_row_indices.iterator + load_startend_row_indices_producer_state.index * 4 * self.n_block_size, cute.make_layout(4 * self.n_block_size))
            num_load_threads = cute.arch.WARP_SIZE * len(self.load_warp_ids)
            _, fm_heads, seqlen_k, _ = flashmask_info.startend_row_indices.shape
            h_h_flashmask_ratio = num_heads // fm_heads
            fm_head_idx = bidh // h_h_flashmask_ratio

            nb_mul_kBN = n_block * self.n_block_size
            loop_ub = min(self.n_block_size, seqlen_k - nb_mul_kBN)
            idx = cute.arch.thread_idx()[0] % num_load_threads
            if const_expr(
                self.fm_load_one_loop
                and not self.has_ut_start
                and self.n_block_size % num_load_threads == 0
            ):
                # Columns kept in smem slot order [LTS, LTE, UTS, UTE]; (column, slot) pairs.
                if const_expr(self.has_ut_start):
                    cols = ((0, 0), (1, 1), (2, 2), (3, 3))
                elif const_expr(self.has_lt_end):
                    cols = ((0, 0), (1, 1))
                elif const_expr(self.has_ut_end):
                    cols = ((0, 0), (1, 3))
                else:
                    cols = ((0, 0),)
                n_iter = self.n_block_size // num_load_threads
                vals = cute.make_rmem_tensor(cute.make_layout(n_iter * len(cols)), Int32)
                for it in cutlass.range_constexpr(n_iter):
                    if idx + it * num_load_threads < loop_ub:
                        for c in cutlass.range_constexpr(len(cols)):
                            vals[it * len(cols) + c] = flashmask_info.startend_row_indices[
                                bidb, fm_head_idx, nb_mul_kBN + idx + it * num_load_threads, cols[c][0]
                            ]
                for it in cutlass.range_constexpr(n_iter):
                    if idx + it * num_load_threads < loop_ub:
                        for c in cutlass.range_constexpr(len(cols)):
                            s_startend_row_indices_cur_stage[
                                self.n_block_size * cols[c][1] + idx + it * num_load_threads
                            ] = vals[it * len(cols) + c]
            elif const_expr(self.has_ut_start):
                while idx < loop_ub:
                    # lts
                    s_startend_row_indices_cur_stage[idx] = flashmask_info.startend_row_indices[bidb, fm_head_idx, nb_mul_kBN + idx, 0]
                    # lte
                    s_startend_row_indices_cur_stage[self.n_block_size + idx] = flashmask_info.startend_row_indices[bidb, fm_head_idx, nb_mul_kBN + idx, 1]
                    # uts
                    s_startend_row_indices_cur_stage[self.n_block_size * 2 + idx] = flashmask_info.startend_row_indices[bidb, fm_head_idx, nb_mul_kBN + idx, 2]
                    # ute
                    s_startend_row_indices_cur_stage[self.n_block_size * 3 + idx] = flashmask_info.startend_row_indices[bidb, fm_head_idx, nb_mul_kBN + idx, 3]
                    idx += num_load_threads
            elif const_expr(self.has_lt_end):
                while idx < loop_ub:
                    # lts
                    s_startend_row_indices_cur_stage[idx] = flashmask_info.startend_row_indices[bidb, fm_head_idx, nb_mul_kBN + idx, 0]
                    # lte
                    s_startend_row_indices_cur_stage[self.n_block_size + idx] = flashmask_info.startend_row_indices[bidb, fm_head_idx, nb_mul_kBN + idx, 1]
                    idx += num_load_threads
            elif const_expr(self.has_ut_end):
                while idx < loop_ub:
                    # lts
                    s_startend_row_indices_cur_stage[idx] = flashmask_info.startend_row_indices[bidb, fm_head_idx, nb_mul_kBN + idx, 0]
                    # ute
                    s_startend_row_indices_cur_stage[self.n_block_size * 3 + idx] = flashmask_info.startend_row_indices[bidb, fm_head_idx, nb_mul_kBN + idx, 1]
                    idx += num_load_threads
            else:
                while idx < loop_ub:
                    # lts
                    s_startend_row_indices_cur_stage[idx] = flashmask_info.startend_row_indices[bidb, fm_head_idx, nb_mul_kBN + idx, 0]
                    idx += num_load_threads
            cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_load_startend_row_indices_full_offset + softmax_wg * self.kv_stage + load_startend_row_indices_producer_state.index)

        load_startend_row_indices_producer_state.advance()

        return load_startend_row_indices_producer_state

    @cute.jit
    def load_startend_row_indices_producer_tail(
        self,
        load_startend_row_indices_producer_state: cutlass.pipeline.PipelineState,
        mbar_ptr: cute.Pointer,
    ):
        for i in range(self.kv_stage - 1):
            load_startend_row_indices_producer_state.advance()

        for softmax_wg in cutlass.range_constexpr(self.num_s_stages):
            cute.arch.mbarrier_wait(mbar_ptr + self.mbar_load_startend_row_indices_empty_offset + softmax_wg * self.kv_stage + load_startend_row_indices_producer_state.index, load_startend_row_indices_producer_state.phase)

    @cute.jit
    def n_block_getter(
        self,
        s_n_block: cute.Tensor,
        s_extra_flags: cute.Tensor,
        n_block_idx: Int32,
    ) -> cutlass.Int32:
        n_block = 0
        if n_block_idx < self.generate_block_buffer_usable_block_count:
            encoded = Int32(s_n_block[n_block_idx])
            mask = Int32(-1 * Int32(encoded <= Int32(0x80000001)))
            converted = Int32(encoded ^ (encoded >> 31))
            n_block = Int32((converted & ~mask) | (encoded & mask))
        else:
            n_block = s_extra_flags[0]
        return n_block

    @cute.jit
    def mask_n_block_getter(
        self,
        s_n_block: cute.Tensor,
        n_block_idx: Int32,
    ) -> cutlass.Int32:
        mask_n_block = 0
        if n_block_idx < self.generate_block_buffer_usable_block_count:
            mask_n_block = s_n_block[n_block_idx]
        return mask_n_block

    @cute.jit
    def load(
        self,
        thr_mma_qk: cute.core.ThrMma,
        thr_mma_pv: cute.core.ThrMma,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: Optional[cute.CopyAtom],
        tma_atom_V: Optional[cute.CopyAtom],
        pipeline_kv: cutlass.pipeline.PipelineAsync,
        mbar_ptr: cute.Pointer,
        block_info: BlockInfo,
        num_splits: Int32,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        blocksparse_tensors: Optional[BlockSparseTensors],
        s_n_block: Optional[cute.Tensor],
        s_extra_flags: Optional[cute.Tensor],
        s_startend_row_indices: Optional[cute.Tensor],
        flashmask_info: Optional[FlashMaskInfo],
        overlap_info: Optional[OverlapInfo],
    ):
        num_load_threads = len(self.load_warp_ids) * cute.arch.WARP_SIZE
        tidx = cute.arch.thread_idx()[0] % num_load_threads
        q_producer_phase = Int32(1)
        kv_producer_state = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Producer, self.kv_stage
        )
        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        if const_expr(self.enable_flashmask):
            generate_block_consumer_state = cutlass.pipeline.make_pipeline_state(
                cutlass.pipeline.PipelineUserType.Consumer, self.generate_block_stage
            )
            load_startend_row_indices_producer_state = cutlass.pipeline.make_pipeline_state(
                cutlass.pipeline.PipelineUserType.Producer, self.kv_stage
            )
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx

            seqlen = SeqlenInfoCls(batch_idx)
            mQ_cur = seqlen.offset_batch_Q(mQ, batch_idx, dim=3)[None, None, head_idx]
            gQ = cute.local_tile(mQ_cur, cute.select(self.mma_tiler_qk, mode=[0, 2]), (None, 0))

            head_idx_kv = (
                head_idx // self.qhead_per_kvhead if const_expr(not self.pack_gqa) else head_idx
            )
            mK_cur, mV_cur = [t[None, None, head_idx_kv, batch_idx] for t in (mK, mV)]
            gK = cute.local_tile(mK_cur, cute.select(self.mma_tiler_qk, mode=[1, 2]), (None, 0))
            gV = cute.local_tile(mV_cur, cute.select(self.mma_tiler_pv, mode=[1, 2]), (0, None))
            tSgQ = thr_mma_qk.partition_A(gQ)
            tSgK = thr_mma_qk.partition_B(gK)
            tOgV = thr_mma_pv.partition_B(gV)
            load_Q_fn, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_Q, 0, cute.make_layout(1), tSgQ, sQ
            )

            tKsK, tKgK = cpasync.tma_partition(
                tma_atom_K,
                0,  # no multicast
                cute.make_layout(1),
                cute.group_modes(sK, 0, 3),
                cute.group_modes(tSgK, 0, 3),
            )
            tVsV, tVgV = cpasync.tma_partition(
                tma_atom_V,
                0,  # no multicast
                cute.make_layout(1),
                cute.group_modes(sV, 0, 3),
                cute.group_modes(tOgV, 0, 3),
            )

            load_Q = partial(
                self.load_Q,
                load_Q_fn,
                mbar_ptr + self.mbar_load_q_full_offset,
                mbar_ptr + self.mbar_load_q_empty_offset,
                phase=q_producer_phase,
            )
            # We have to use mbarrier directly in the load for KV instead of replying on
            # pipeline_kv, because we could have different number of TMA bytes for K and V
            load_K = partial(
                self.load_KV,
                tma_atom_K,
                tKgK,
                tKsK,
                sK,
                mbar_ptr + self.mbar_load_kv_full_offset,
                mbar_ptr + self.mbar_load_kv_empty_offset,
                K_or_V="K",
            )
            load_V = partial(
                self.load_KV,
                tma_atom_V,
                tVgV,
                tVsV,
                sV,
                mbar_ptr + self.mbar_load_kv_full_offset,
                mbar_ptr + self.mbar_load_kv_empty_offset,
                K_or_V="V",
            )

            # FM-4 overlap gate: bound via partial (NOT a nested closure) so it has no
            # __closure__ and passes the DSL closure_check inside the dynamic load loop.
            if const_expr(self.enable_overlap):
                # The gate's warp sync only covers one load warp.
                assert len(self.load_warp_ids) == 1
                gate_batch_idx = batch_idx
                if const_expr(self.overlap_bhsd_layout):
                    gate_batch_idx = (
                        batch_idx * cute.size(mK.shape[2]) + head_idx_kv
                    )
                _gate = partial(
                    _overlap_gate,
                    tidx=tidx,
                    s_total=seqlen.seqlen_k,
                    batch_idx=gate_batch_idx,
                    write_ptr=overlap_info.write_ptr.iterator,
                    spin_timeout_ns=overlap_info.spin_timeout_ns,
                    n_block_size=self.n_block_size,
                    local_kv_seqlen=overlap_info.local_kv_seqlen,
                )

            if const_expr(self.enable_flashmask):
                n_block_min, n_block_max = block_info.get_n_block_min_max(
                    seqlen, m_block, split_idx, num_splits
                )
                if n_block_min < n_block_max:
                    cute.arch.mbarrier_wait(mbar_ptr + self.mbar_generate_block_full_offset + generate_block_consumer_state.index, generate_block_consumer_state.phase)
                    n_block_idx = 0
                    s_n_block_cur_stage = cute.make_tensor(s_n_block.iterator + generate_block_consumer_state.index * self.generate_block_buffer_block_count, cute.make_layout(self.generate_block_buffer_block_count))

                    s_extra_flags_cur_stage = cute.make_tensor(s_extra_flags.iterator + generate_block_consumer_state.index, cute.make_layout(1))
                    n_block_first = self.n_block_getter(s_n_block_cur_stage, s_extra_flags_cur_stage, n_block_idx)
                    n_block_idx += 1

                    # generate_block make sure n_block_first won't be self.generate_block_incomplete
                    if n_block_first < n_block_min and n_block_first != Int32(self.generate_block_incomplete):
                        cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_generate_block_empty_offset + generate_block_consumer_state.index)
                        generate_block_consumer_state.advance()
                        if const_expr(not self.fm_persistent):
                            # Persistent: the tile advance at the end of the loop body is the
                            # only one (a second advance here would skip a ticket).
                            tile_scheduler.prefetch_next_work()
                            tile_scheduler.advance_to_next_work()
                            work_tile = tile_scheduler.get_current_work()
                    elif const_expr(not self.is_split_kv) or n_block_min < n_block_max:
                        load_Q(block=self.q_stage * m_block + 0, stage=0)  # Q0
                        if const_expr(self.fm_load_one_loop):
                            # One loop issues K(j), V(j-1), startend(j) for every block j, the first
                            # block included (it only skips the V); see fm_load_one_loop.
                            q_producer_phase ^= 1
                            n_block = n_block_first
                            n_block_prev = n_block_first
                            is_first_block = cutlass.Boolean(True)
                            while n_block >= n_block_min or n_block == Int32(self.generate_block_incomplete):
                                while n_block >= n_block_min:
                                    if const_expr(self.enable_overlap):
                                        _gate(n_block)
                                    load_K(block=n_block, producer_state=kv_producer_state)  # Kj
                                    kv_producer_state.advance()
                                    if not is_first_block:
                                        load_V(block=n_block_prev, producer_state=kv_producer_state)  # V(j-1)
                                        kv_producer_state.advance()
                                    is_first_block = cutlass.Boolean(False)
                                    n_block_prev = n_block
                                    load_startend_row_indices_producer_state = self.load_startend_row_indices(
                                        batch_idx,
                                        head_idx,
                                        n_block,
                                        mQ.shape[2], # (s_q, d, h, b)
                                        load_startend_row_indices_producer_state,
                                        s_startend_row_indices,
                                        flashmask_info,
                                        mbar_ptr,
                                    )
                                    n_block = self.n_block_getter(s_n_block_cur_stage, s_extra_flags_cur_stage, n_block_idx)
                                    n_block_idx += 1

                                if n_block == Int32(self.generate_block_incomplete):
                                    cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_generate_block_empty_offset + generate_block_consumer_state.index)
                                    generate_block_consumer_state.advance()

                                    cute.arch.mbarrier_wait(mbar_ptr + self.mbar_generate_block_full_offset + generate_block_consumer_state.index, generate_block_consumer_state.phase)

                                    s_n_block_cur_stage = cute.make_tensor(
                                                              s_n_block.iterator + generate_block_consumer_state.index * self.generate_block_buffer_block_count,
                                                              cute.make_layout(self.generate_block_buffer_block_count),
                                                          )
                                    s_extra_flags_cur_stage = cute.make_tensor(
                                                                  s_extra_flags.iterator + generate_block_consumer_state.index,
                                                                  cute.make_layout(1),
                                                              )
                                    n_block_idx = 0
                                    n_block = self.n_block_getter(s_n_block_cur_stage, s_extra_flags_cur_stage, n_block_idx)
                                    n_block_idx += 1
                        else:
                            if const_expr(self.enable_overlap):
                                _gate(n_block_first)
                            load_K(block=n_block_first, producer_state=kv_producer_state)  # K0
                            kv_producer_state.advance()
                            if const_expr(self.q_stage == 2):
                                load_Q(block=self.q_stage * m_block + 1, stage=1)  # Q1
                            q_producer_phase ^= 1
                            # S ping-pong: V of a block is issued after the K of the NEXT block
                            # (order K0, [K(j+1), V(j)], V(n-1), same as the MMA consumes them),
                            # so the block whose V is still pending is carried in n_block_prev.
                            n_block_prev = n_block_first
                            if const_expr(not self.use_s_ping_pong):
                                load_V(block=n_block_first, producer_state=kv_producer_state)  # V0
                                kv_producer_state.advance()
                            # TODO: two possible optimization: 1.reuse kv_producer_state 2.skip load when not partially masked

                            load_startend_row_indices_producer_state = self.load_startend_row_indices(
                                batch_idx,
                                head_idx,
                                n_block_first,
                                mQ.shape[2], # (s_q, d, h, b)
                                load_startend_row_indices_producer_state,
                                s_startend_row_indices,
                                flashmask_info,
                                mbar_ptr,
                            )

                            n_block = self.n_block_getter(s_n_block_cur_stage, s_extra_flags_cur_stage, n_block_idx)
                            n_block_idx += 1
                            while n_block >= n_block_min or n_block == Int32(self.generate_block_incomplete):
                                while n_block >= n_block_min:

                                    if const_expr(self.enable_overlap):
                                        _gate(n_block)
                                    load_K(block=n_block, producer_state=kv_producer_state)  # Ki
                                    kv_producer_state.advance()
                                    if const_expr(self.use_s_ping_pong):
                                        load_V(block=n_block_prev, producer_state=kv_producer_state)  # V(i-1)
                                        kv_producer_state.advance()
                                        n_block_prev = n_block
                                    else:
                                        load_V(block=n_block, producer_state=kv_producer_state)  # Vi
                                        kv_producer_state.advance()

                                    load_startend_row_indices_producer_state = self.load_startend_row_indices(
                                        batch_idx,
                                        head_idx,
                                        n_block,
                                        mQ.shape[2], # (s_q, d, h, b)
                                        load_startend_row_indices_producer_state,
                                        s_startend_row_indices,
                                        flashmask_info,
                                        mbar_ptr,
                                    )

                                    n_block = self.n_block_getter(s_n_block_cur_stage, s_extra_flags_cur_stage, n_block_idx)
                                    n_block_idx += 1

                                if n_block == Int32(self.generate_block_incomplete):
                                    cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_generate_block_empty_offset + generate_block_consumer_state.index)
                                    generate_block_consumer_state.advance()

                                    cute.arch.mbarrier_wait(mbar_ptr + self.mbar_generate_block_full_offset + generate_block_consumer_state.index, generate_block_consumer_state.phase)

                                    s_n_block_cur_stage = cute.make_tensor(
                                                              s_n_block.iterator + generate_block_consumer_state.index * self.generate_block_buffer_block_count,
                                                              cute.make_layout(self.generate_block_buffer_block_count),
                                                          )
                                    s_extra_flags_cur_stage = cute.make_tensor(
                                                                  s_extra_flags.iterator + generate_block_consumer_state.index,
                                                                  cute.make_layout(1),
                                                              )
                                    n_block_idx = 0
                                    n_block = self.n_block_getter(s_n_block_cur_stage, s_extra_flags_cur_stage, n_block_idx)
                                    n_block_idx += 1

                        if const_expr(self.use_s_ping_pong):
                            load_V(block=n_block_prev, producer_state=kv_producer_state)  # V(n-1)
                            kv_producer_state.advance()
                        cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_generate_block_empty_offset + generate_block_consumer_state.index)
                        generate_block_consumer_state.advance()

            elif const_expr(not self.use_block_sparsity):
                n_block_min, n_block_max = block_info.get_n_block_min_max(
                    seqlen, m_block, split_idx, num_splits
                )
                # add n_block_first here to pass the cutedsl compile,
                # though it's unclear why it's not supported to remove n_block_first
                n_block_first = 0
                if const_expr(not self.is_split_kv) or n_block_min < n_block_max:
                    load_Q(block=self.q_stage * m_block + 0, stage=0)  # Q0
                    n_block_first = n_block_max - 1 if n_block_max > 0 else 0
                    if const_expr(self.enable_overlap):
                        _gate(n_block_max - 1)
                    load_K(block=n_block_max - 1, producer_state=kv_producer_state)  # K0
                    kv_producer_state.advance()
                    if const_expr(self.q_stage == 2):
                        load_Q(block=self.q_stage * m_block + 1, stage=1)  # Q1
                    q_producer_phase ^= 1
                    if const_expr(self.use_s_ping_pong):
                        # Same order as the MMA warp consumes them: K0, [K(j+1), V(j)], V(n-1).
                        for i in cutlass.range(n_block_max - 1 - n_block_min, unroll=1):
                            n_block = n_block_max - 2 - i
                            if const_expr(self.enable_overlap):
                                _gate(n_block)
                            load_K(block=n_block, producer_state=kv_producer_state)  # K(j+1)
                            kv_producer_state.advance()
                            load_V(block=n_block + 1, producer_state=kv_producer_state)  # V(j)
                            kv_producer_state.advance()
                        load_V(block=n_block_min, producer_state=kv_producer_state)  # V(n-1)
                        kv_producer_state.advance()
                    else:
                        load_V(block=n_block_max - 1, producer_state=kv_producer_state)  # V0
                        kv_producer_state.advance()
                        for i in cutlass.range(n_block_max - 1 - n_block_min, unroll=1):
                            n_block = n_block_max - 2 - i
                            if const_expr(self.enable_overlap):
                                _gate(n_block)
                            load_K(block=n_block, producer_state=kv_producer_state)  # Ki
                            kv_producer_state.advance()
                            load_V(block=n_block, producer_state=kv_producer_state)  # Vi
                            kv_producer_state.advance()

            else:
                kv_producer_state, q_producer_phase = produce_block_sparse_loads_sm100(
                    blocksparse_tensors,
                    batch_idx,
                    head_idx,
                    m_block,
                    kv_producer_state,
                    load_Q,
                    load_K,
                    load_V,
                    pipeline_kv,
                    self.q_stage,
                    q_producer_phase,
                )


            tile_scheduler.prefetch_next_work()
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()
            # End of persistent scheduler loop

    @cute.jit
    def mma(
        self,
        tiled_mma_qk: cute.core.ThrMma,
        tiled_mma_pv: cute.core.ThrMma,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        tStSs: Tuple[cute.Tensor, cute.Tensor],
        tOtOs: tuple[cute.Tensor],
        tOrPs: Tuple[cute.Tensor, cute.Tensor],
        pipeline_kv: cutlass.pipeline.PipelineAsync,
        mbar_ptr: cute.Pointer,
        block_info: BlockInfo,
        num_splits: Int32,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        blocksparse_tensors: Optional[BlockSparseTensors],
        num_heads: Int32,
        flashmask_info: FlashMaskInfo,
        mma_desc_scratch_ptr: Optional[cute.Pointer] = None,
    ):
        tSrQ = tiled_mma_qk.make_fragment_A(sQ)
        tSrK = tiled_mma_qk.make_fragment_B(sK)
        tOrV = tiled_mma_pv.make_fragment_B(sV)
        # One Q buffer per softmax stage (num_s_stages == q_stage).
        tSrQs = tuple(tSrQ[None, None, None, stage] for stage in range(self.num_s_stages))

        qk_mma_op, pv_mma_op = tiled_mma_qk.op, tiled_mma_pv.op

        if const_expr(self.use_2cta_instrs):
            # 2-CTA: go through cute.gemm so the LIBRARY builds the tcgen05 instruction and
            # operand descriptors. `gemm_ptx_partial` hand-rolls them (idesc from
            # `mma_op_to_idesc`, plus its own `[tmem_a + off]` / smem-desc-low-bits
            # arithmetic) and is only exercised with cta_group=1 here; the backward kernel's
            # 2-CTA gemms that DO use it pass an explicit `tA_addr` (see flash_bwd_sm100.py's
            # mma_pdo_fn) which the forward's P gemm does not, and `gemm_ptx_partial` itself
            # notes that `tCrA.iterator.toint()` returns 0 for a TS gemm. The library path
            # emits the same instructions; its only cost is that the mid-instruction-stream
            # P handoff (split_p_store / mbar_P_full_2) becomes a plain wait before the gemm,
            # which the 2-CTA config does not use anyway (split_p_store needs n_block >= 128).
            gemm_Si = [
                partial(self.gemm_lib, tiled_mma_qk, tStSs[stage], tSrQs[stage], zero_init=True)
                for stage in range(self.num_s_stages)
            ]
            gemm_Pi = [
                partial(self.gemm_lib, tiled_mma_pv, tOtOs[stage], tOrPs[stage])
                for stage in range(self.num_s_stages)
            ]
        else:
            gemm_Si = [
                partial(
                    sm100_utils.gemm_ptx_partial,
                    qk_mma_op,
                    self.tmem_s_offset[stage],
                    tSrQs[stage],
                    sA=sQ[None, None, None, stage],
                    zero_init=True,
                    cta_group=self.cta_group_size,
                )
                for stage in range(self.num_s_stages)
            ]
            gemm_Pi = [
                partial(
                    sm100_utils.gemm_ptx_partial,
                    pv_mma_op,
                    self.tmem_o_offset[stage],
                    tOrPs[stage],
                    sA=None,
                    cta_group=self.cta_group_size,
                )
                for stage in range(self.num_s_stages)
            ]

        mma_desc_scratch = None
        if const_expr(self.fm_persistent):
            # Persistent: ptxas would hoist the per-k 64-bit smem descriptors out of the tile
            # loop and spill them (32-register MMA warp); bounce their start addresses
            # through 8 bytes of smem instead (see gemm_ptx_partial).
            mma_desc_scratch = mma_desc_scratch_ptr.toint()
        if const_expr(self.use_s_ping_pong):
            # One QK / PV gemm per S/P slot. Both read the single Q tile and both PVs
            # accumulate into the single O.
            gemm_S_pp = [
                partial(
                    sm100_utils.gemm_ptx_partial,
                    qk_mma_op,
                    self.tmem_s_offset[slot],
                    tSrQs[0],
                    sA=sQ[None, None, None, 0],
                    zero_init=True,
                    cta_group=self.cta_group_size,
                    desc_scratch_smem_addr=mma_desc_scratch,
                )
                for slot in range(2)
            ]
            gemm_P_pp = [
                partial(
                    sm100_utils.gemm_ptx_partial,
                    pv_mma_op,
                    self.tmem_o_offset[0],
                    tOrPs[slot],
                    sA=None,
                    cta_group=self.cta_group_size,
                    desc_scratch_smem_addr=mma_desc_scratch,
                )
                for slot in range(2)
            ]
            if const_expr(self.fm_one_softmax):
                # One QK and one PV issue sequence for both slots; the slot only changes
                # the TMEM accumulator / A-operand address and the barrier index (computed at
                # runtime), instead of two unrolled copies of every UTCHMMA sequence.
                gemm_S_pp = [
                    partial(
                        sm100_utils.gemm_ptx_partial,
                        qk_mma_op,
                        tCrA=tSrQs[0],
                        sA=sQ[None, None, None, 0],
                        zero_init=True,
                        cta_group=self.cta_group_size,
                        desc_scratch_smem_addr=mma_desc_scratch,
                    )
                ]
                gemm_P_pp = [
                    partial(
                        sm100_utils.gemm_ptx_partial,
                        pv_mma_op,
                        acc_tmem_addr=self.tmem_o_offset[0],
                        tCrA=tOrPs[0],
                        sA=None,
                        cta_group=self.cta_group_size,
                        desc_scratch_smem_addr=mma_desc_scratch,
                    ),
                    # TMEM addresses of the two P slots, as gemm_ptx_partial derives them.
                    tOrPs[0][None, None, 0].iterator.toint(),
                    tOrPs[1][None, None, 0].iterator.toint(),
                ]
            # Global block counter across work tiles: block `pp_count` uses S/P slot
            # pp_count & 1, and that slot's barriers are on phase (pp_count >> 1) & 1.
            pp_count = Int32(0)

        mma_q_consumer_phase = Int32(0)
        mma_kv_consumer_state = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.kv_stage
        )
        P_full_O_rescaled_phase = Int32(0)

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        if const_expr(self.enable_flashmask):
            fm_num_heads = flashmask_info.startend_row_indices.shape[1]
            h_h_flashmask_ratio = num_heads // fm_num_heads

        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)

            block_iter_count = Int32(0)
            process_tile = False

            if const_expr(self.enable_flashmask):
                block_iter_count = flashmask_info.valid_block_count[batch_idx, head_idx // h_h_flashmask_ratio, m_block]
                process_tile = block_iter_count > Int32(0)
            elif const_expr(self.use_block_sparsity):
                block_iter_count = get_total_block_count(blocksparse_tensors, batch_idx, head_idx, m_block)
                process_tile = block_iter_count > Int32(0)
            else:
                n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, m_block, split_idx, num_splits)
                block_iter_count = n_block_max - n_block_min
                if const_expr(not self.is_split_kv):
                    process_tile = True
                else:
                    process_tile = n_block_min < n_block_max

            if process_tile:
                if const_expr(self.use_s_ping_pong):
                    # Software pipeline: QK(0); [QK(j+1), PV(j)] for j in 0..n-2; PV(n-1).
                    # Load order matches: Q, K0, [K(j+1), V(j)], V(n-1).
                    # QK(j+1) overwrites the slot of block j-1. That is safe without a softmax
                    # handshake: PV(j-1) read P(j-1) from it, was issued earlier from this same
                    # thread (tcgen05 MMAs execute in issue order), and was only issued once the
                    # softmax had written all of P(j-1) (P_full_O_rescaled / P_full_2).
                    cute.arch.mbarrier_wait(
                        mbar_ptr + self.mbar_load_q_full_offset + 0, mma_q_consumer_phase
                    )
                    pipeline_kv.consumer_wait(mma_kv_consumer_state)
                    self.mma_pp_qk(
                        gemm_S_pp,
                        mbar_ptr,
                        tSrK[None, None, None, mma_kv_consumer_state.index],
                        sK[None, None, None, mma_kv_consumer_state.index],
                        pp_count,
                    )
                    mma_q_consumer_phase ^= 1
                    pipeline_kv.consumer_release(mma_kv_consumer_state)
                    mma_kv_consumer_state.advance()
                    O_should_accumulate = False
                    for j in cutlass.range(block_iter_count - 1, unroll=1):
                        # QK(j+1) -> slot (pp_count + j + 1) & 1
                        pipeline_kv.consumer_wait(mma_kv_consumer_state)
                        self.mma_pp_qk(
                            gemm_S_pp,
                            mbar_ptr,
                            tSrK[None, None, None, mma_kv_consumer_state.index],
                            sK[None, None, None, mma_kv_consumer_state.index],
                            pp_count + j + 1,
                        )
                        pipeline_kv.consumer_release(mma_kv_consumer_state)
                        mma_kv_consumer_state.advance()
                        # PV(j) from slot (pp_count + j) & 1
                        pipeline_kv.consumer_wait(mma_kv_consumer_state)
                        self.mma_pp_pv(
                            gemm_P_pp,
                            mbar_ptr,
                            tOrV[None, None, None, mma_kv_consumer_state.index],
                            sV[None, None, None, mma_kv_consumer_state.index],
                            O_should_accumulate,
                            pp_count + j,
                        )
                        pipeline_kv.consumer_release(mma_kv_consumer_state)
                        mma_kv_consumer_state.advance()
                        O_should_accumulate = True
                    # release Q (the last QK has been issued)
                    with cute.arch.elect_one():
                        self.mma_barrier_commit(mbar_ptr + self.mbar_load_q_empty_offset + 0)
                    # PV(n-1)
                    pipeline_kv.consumer_wait(mma_kv_consumer_state)
                    self.mma_pp_pv(
                        gemm_P_pp,
                        mbar_ptr,
                        tOrV[None, None, None, mma_kv_consumer_state.index],
                        sV[None, None, None, mma_kv_consumer_state.index],
                        O_should_accumulate,
                        pp_count + block_iter_count - 1,
                    )
                    pipeline_kv.consumer_release(mma_kv_consumer_state)
                    mma_kv_consumer_state.advance()
                    pp_count += block_iter_count
                else:
                    for stage in cutlass.range_constexpr(self.num_s_stages):
                        # GEMM_QK00 (Q0 * K0 -> S0) or GEMM_QK01 (Q1 * K0 -> S1)
                        # single_q_stage has a single stage: one Q tile, one S buffer, one QK gemm.
                        # 1. wait for Q0 / Q1
                        cute.arch.mbarrier_wait(
                            mbar_ptr + self.mbar_load_q_full_offset + stage, mma_q_consumer_phase
                        )
                        # 2. wait for K0
                        if const_expr(stage == 0):
                            pipeline_kv.consumer_wait(mma_kv_consumer_state)
                        tSrKi = tSrK[None, None, None, mma_kv_consumer_state.index]
                        # 3. gemm
                        sK_cur = sK[None, None, None, mma_kv_consumer_state.index]
                        if const_expr(self.uneven_kv_smem):
                            sK_cur = self.offset_kv_smem(
                                sK_cur, mma_kv_consumer_state.index, mma_kv_consumer_state.phase
                            )
                        gemm_Si[stage](tCrB=tSrKi, sB=sK_cur)
                        # 4. release S0 / S1
                        with cute.arch.elect_one():
                            self.mma_barrier_commit(mbar_ptr + self.mbar_S_full_offset + stage)
                    mma_q_consumer_phase ^= 1
                    # 5. release K0
                    pipeline_kv.consumer_release(mma_kv_consumer_state)
                    mma_kv_consumer_state.advance()
                    # End of GEMM (Q1 * K0 -> S1)
                    # Note: Q0 & Q1 are still needed in the seqlen_kv loop
                    # so we need to release them after the seqlen_kv loop

                    # O hasn't been accumulated yet, its first MMA calculation doesn't need to accumulate
                    block_loop_count = block_iter_count - 1
                    O_should_accumulate = False
                    for i in cutlass.range(block_loop_count, unroll=1):

                        # GEMM_PV00 (P0 * V0 -> O0_partial), O0 needs to be accumulated in the seqlen_kv loop
                        # 1. wait for V0
                        pipeline_kv.consumer_wait(mma_kv_consumer_state)

                        mma_kv_release_state = mma_kv_consumer_state.clone()
                        Vi_index, Vi_phase = mma_kv_consumer_state.index, mma_kv_consumer_state.phase
                        tOrVi = tOrV[None, None, None, Vi_index]
                        for stage in cutlass.range_constexpr(self.num_s_stages):
                            # 2. acquire corrected O0/O1_partial and P0 / P1
                            cute.arch.mbarrier_wait(
                                mbar_ptr + self.mbar_P_full_O_rescaled_offset + stage,
                                P_full_O_rescaled_phase,
                            )
                            # 3. gemm
                            sV_cur = sV[None, None, None, Vi_index]
                            if const_expr(self.uneven_kv_smem):
                                sV_cur = self.offset_kv_smem(sV_cur, Vi_index, Vi_phase)
                            if const_expr(self.split_p_store):
                                gemm_Pi[stage](
                                    tCrB=tOrVi,
                                    sB=sV_cur,
                                    zero_init=not O_should_accumulate,
                                    mbar_ptr=mbar_ptr + self.mbar_P_full_2_offset + stage,
                                    mbar_phase=P_full_O_rescaled_phase,
                                )
                            else:
                                # P was stored in one shot: run the full PV contraction,
                                # then drain P_full_2 to keep the arrive counts balanced.
                                gemm_Pi[stage](
                                    tCrB=tOrVi,
                                    sB=sV_cur,
                                    zero_init=not O_should_accumulate,
                                )
                                cute.arch.mbarrier_wait(
                                    mbar_ptr + self.mbar_P_full_2_offset + stage,
                                    P_full_O_rescaled_phase,
                                )
                            # 5. release V(i-1)
                            if const_expr(stage == self.num_s_stages - 1):
                                pipeline_kv.consumer_release(mma_kv_release_state)
                                mma_kv_release_state.advance()
                            # End of GEMM_PV00 (P0 * V0 -> O0_partial)

                            # GEMM_QK0i (Q0 * Ki -> S0)
                            # 1. wait for Ki
                            if const_expr(stage == 0):
                                mma_kv_consumer_state.advance()
                                pipeline_kv.consumer_wait(mma_kv_consumer_state)
                            Ki_index, Ki_phase = mma_kv_consumer_state.index, mma_kv_consumer_state.phase
                            # 2. gemm
                            # Don't need to wait for the softmax warp to have finished reading the previous
                            # Si, since this gemm is scheduled after the PV gemm, which guaranteed that Si
                            # has been read and Pi has been written.
                            # tiled_mma_qk = sm100_utils.gemm(tiled_mma_qk, tStSs[stage], tSrQs[stage], tSrK[None, None, None, Ki_index], zero_init=True)
                            sK_cur = sK[None, None, None, Ki_index]
                            if const_expr(self.uneven_kv_smem):
                                sK_cur = self.offset_kv_smem(sK_cur, Ki_index, Ki_phase)
                            gemm_Si[stage](tCrB=tSrK[None, None, None, Ki_index], sB=sK_cur)
                            # 3. release S0
                            with cute.arch.elect_one():
                                self.mma_barrier_commit(mbar_ptr + self.mbar_S_full_offset + stage)
                            # End of GEMM_QK0i (Q0 * Ki -> S0)

                        # 4. release Ki
                        pipeline_kv.consumer_release(mma_kv_consumer_state)
                        mma_kv_consumer_state.advance()
                        P_full_O_rescaled_phase ^= 1
                        O_should_accumulate = True
                    # End of seqlen_kv loop

                    # release Q0 & Q1
                    with cute.arch.elect_one():
                        for stage in cutlass.range_constexpr(self.q_stage):
                            self.mma_barrier_commit(mbar_ptr + self.mbar_load_q_empty_offset + stage)

                    # GEMM_PV00 (P0 * V0 -> O0_partial), O0 needs to be accumulated in the seqlen_kv loop
                    # 1. wait for V0
                    pipeline_kv.consumer_wait(mma_kv_consumer_state)

                    Vi_index, Vi_phase = mma_kv_consumer_state.index, mma_kv_consumer_state.phase
                    tOrVi = tOrV[None, None, None, Vi_index]
                    for stage in cutlass.range_constexpr(self.num_s_stages):
                        # 2. acquire corrected Oi_partial and Pi

                        cute.arch.mbarrier_wait(
                            mbar_ptr + self.mbar_P_full_O_rescaled_offset + stage, P_full_O_rescaled_phase
                        )

                        # 3. gemm
                        sV_cur = sV[None, None, None, Vi_index]
                        if const_expr(self.uneven_kv_smem):
                            sV_cur = self.offset_kv_smem(sV_cur, Vi_index, Vi_phase)
                        if const_expr(self.split_p_store):
                            gemm_Pi[stage](
                                tCrB=tOrVi,
                                sB=sV_cur,
                                zero_init=not O_should_accumulate,
                                mbar_ptr=mbar_ptr + self.mbar_P_full_2_offset + stage,
                                mbar_phase=P_full_O_rescaled_phase,
                            )
                        else:
                            # P was stored in one shot: run the full PV contraction,
                            # then drain P_full_2 to keep the arrive counts balanced.
                            gemm_Pi[stage](
                                tCrB=tOrVi,
                                sB=sV_cur,
                                zero_init=not O_should_accumulate,
                            )
                            cute.arch.mbarrier_wait(
                                mbar_ptr + self.mbar_P_full_2_offset + stage,
                                P_full_O_rescaled_phase,
                            )
                        # 4. release accumulated O0_partial
                        # We do need O_full here since for the last tile, by the time the softmax warp
                        # has signaled to the correction warps, the softmax warp has just finished compute
                        # the row sum of the current tile. It does not guarantee that the 1st tile
                        # of the next work tile has been computed yet.
                        with cute.arch.elect_one():
                            self.mma_barrier_commit(mbar_ptr + self.mbar_O_full_offset + stage)
                        # End of GEMM_PV00 (P0 * V0 -> O0_partial)

                    P_full_O_rescaled_phase ^= 1
                    # 5. release Vi_end
                    pipeline_kv.consumer_release(mma_kv_consumer_state)
                    mma_kv_consumer_state.advance()
                    # End of GEMM_PV1(i_end) (P1 * Vi_end -> O1)

            # Advance to next tile
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

        # End of persistent scheduler loop

    @cute.jit
    def mma_pp_qk(
        self,
        gemm_S_pp: list,
        mbar_ptr: cute.Pointer,
        tSrKi: cute.Tensor,
        sK_cur: cute.Tensor,
        block_count: Int32,
    ):
        """S ping-pong: issue QK for global block `block_count` into S slot block_count & 1."""
        if const_expr(self.fm_one_softmax):
            slot = block_count & 1
            gemm_S_pp[0](
                acc_tmem_addr=self.tmem_s_offset[0]
                + slot * (self.tmem_s_offset[1] - self.tmem_s_offset[0]),
                tCrB=tSrKi,
                sB=sK_cur,
            )
            with cute.arch.elect_one():
                self.mma_barrier_commit(mbar_ptr + self.mbar_S_full_offset + slot)
        elif (block_count & 1) == 0:
            gemm_S_pp[0](tCrB=tSrKi, sB=sK_cur)
            with cute.arch.elect_one():
                self.mma_barrier_commit(mbar_ptr + self.mbar_S_full_offset + 0)
        else:
            gemm_S_pp[1](tCrB=tSrKi, sB=sK_cur)
            with cute.arch.elect_one():
                self.mma_barrier_commit(mbar_ptr + self.mbar_S_full_offset + 1)

    @cute.jit
    def mma_pp_pv(
        self,
        gemm_P_pp: list,
        mbar_ptr: cute.Pointer,
        tOrVi: cute.Tensor,
        sV_cur: cute.Tensor,
        O_should_accumulate,
        block_count: Int32,
    ):
        """S ping-pong: issue PV for global block `block_count` from P slot block_count & 1."""
        pv_phase = (block_count >> 1) & 1
        if const_expr(self.fm_one_softmax):
            slot = block_count & 1
            cute.arch.mbarrier_wait(mbar_ptr + self.mbar_P_full_O_rescaled_offset + slot, pv_phase)
            gemm_P_pp[0](
                tA_addr=gemm_P_pp[1] + slot * (gemm_P_pp[2] - gemm_P_pp[1]),
                tCrB=tOrVi,
                sB=sV_cur,
                zero_init=not O_should_accumulate,
                mbar_ptr=mbar_ptr + self.mbar_P_full_2_offset + slot,
                mbar_phase=pv_phase,
            )
        elif (block_count & 1) == 0:
            # P(block) fully stored up to the split point and O rescaled for this block
            cute.arch.mbarrier_wait(mbar_ptr + self.mbar_P_full_O_rescaled_offset + 0, pv_phase)
            gemm_P_pp[0](
                tCrB=tOrVi,
                sB=sV_cur,
                zero_init=not O_should_accumulate,
                mbar_ptr=mbar_ptr + self.mbar_P_full_2_offset + 0,
                mbar_phase=pv_phase,
            )
        else:
            cute.arch.mbarrier_wait(mbar_ptr + self.mbar_P_full_O_rescaled_offset + 1, pv_phase)
            gemm_P_pp[1](
                tCrB=tOrVi,
                sB=sV_cur,
                zero_init=not O_should_accumulate,
                mbar_ptr=mbar_ptr + self.mbar_P_full_2_offset + 1,
                mbar_phase=pv_phase,
            )
        # QK(block+1) was issued before this PV, so "S(block+1) ready" no longer implies
        # that O already holds this block: commit O_full after EVERY PV, the correction
        # warps wait on it before each O rescale / the final O read.
        if const_expr(self.pp_lazy_o_wait):
            # One O_full slot per S/P slot (block & 1), phase (block >> 1) & 1, so the
            # correction warps only wait when they actually rescale O.
            if const_expr(self.fm_one_softmax):
                with cute.arch.elect_one():
                    self.mma_barrier_commit(mbar_ptr + self.mbar_O_full_offset + (block_count & 1))
            elif (block_count & 1) == 0:
                with cute.arch.elect_one():
                    self.mma_barrier_commit(mbar_ptr + self.mbar_O_full_offset + 0)
            else:
                with cute.arch.elect_one():
                    self.mma_barrier_commit(mbar_ptr + self.mbar_O_full_offset + 1)
        else:
            with cute.arch.elect_one():
                self.mma_barrier_commit(mbar_ptr + self.mbar_O_full_offset + 0)

    # for both softmax0 and softmax1 warp group
    @cute.jit
    def softmax_loop(
        self,
        stage: int | Int32,
        softmax_scale_log2: Float32,
        softmax_scale: Float32,
        thr_mma_qk: cute.core.ThrMma,
        tStSi: cute.Tensor,
        sScale: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        learnable_sink: Optional[cute.Tensor],
        mbar_ptr: cute.Pointer,
        block_info: BlockInfo,
        num_splits: Int32,
        SeqlenInfoCls: Callable,
        AttentionMaskCls: Callable,
        TileSchedulerCls: Callable,
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        s_n_block: Optional[cute.Tensor] = None,
        s_extra_flags: Optional[cute.Tensor] = None,
        s_startend_row_indices: Optional[cute.Tensor] = None,
        mBlockLogit: Optional[cute.Tensor] = None,
        mBlockBos: Optional[cute.Tensor] = None,
        sP: Optional[cute.Tensor] = None,
    ):
        """Compute softmax on attention scores from QK matrix multiplication.

        This method handles the softmax computation for either the first or second half of the
        attention matrix, depending on the 'stage' parameter. It calculates row-wise maximum
        and sum values needed for stable softmax computation, applies optional masking, and
        transforms raw attention scores into probability distributions.

        The implementation uses specialized memory access patterns and efficient math operations
        for computing exp(x) using exp2 functions. It also coordinates pipeline
        synchronization between MMA, correction, and sequence processing stages.
        """
        tidx = cute.arch.thread_idx()[0] % (
            cute.arch.WARP_SIZE
            # * (len(self.softmax0_warp_ids) if stage == 0 else len(self.softmax1_warp_ids)
            * (len(self.softmax0_warp_ids))
        )

        tScS = thr_mma_qk.partition_C(cute.make_identity_tensor(self.mma_tiler_qk[:2]))

        tilePlikeFP32 = self.mma_tiler_qk[1] // 32 * self.v_dtype.width
        if const_expr(not self.folded_acc):
            tStP_layout = cute.composition(
                tStSi.layout, cute.make_layout((self.m_block_size, tilePlikeFP32))
            )
            tStP = cute.make_tensor(tStSi.iterator + self.tmem_s_to_p_offset, tStP_layout)
        else:
            # P does not live in TMEM on the folded path (see p_source): the A operand's TMEM
            # layout wants a whole row in one lane, but the folded softmax splits each row
            # across two lanes and tmem stores are lane-local. sP replaces it.
            tStP = None

        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)),
            Float32,
        )
        thr_tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, tStSi).get_slice(tidx)
        tStS_t2r = thr_tmem_load.partition_S(tStSi)
        if const_expr(self.use_s_ping_pong):
            # Second S/P slot (tmem_s_offset[1]); same layout, shifted by 128 columns.
            tStSi_1 = cute.make_tensor(
                tStSi.iterator + (self.tmem_s_offset[1] - self.tmem_s_offset[0]), tStSi.layout
            )
            tStS_t2r_1 = thr_tmem_load.partition_S(tStSi_1)
        else:
            tStSi_1 = None
            tStS_t2r_1 = None

        # The row_max/row_sum TMEM vec buffer. Every `cute.copy` through it is commented out
        # (the correction warps read the statistics from sScale instead), and with the folded
        # accumulator an (m, 1) tile degenerates to m lanes -- half the softmax threads -- so
        # only build it where it is harmless.
        if const_expr(not self.folded_acc):
            tStScale = cute.composition(tStSi, cute.make_layout((self.m_block_size, 1)))
            tScScale = cute.composition(tScS, cute.make_layout((self.m_block_size, 1)))
            tmem_store_scale_atom = cute.make_copy_atom(
                tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(1)),
                Float32,
            )
            thr_tmem_store_scale = tcgen05.make_tmem_copy(
                tmem_store_scale_atom, tStScale
            ).get_slice(tidx)
            tStScale_r2t = thr_tmem_store_scale.partition_D(tStScale)
        else:
            thr_tmem_store_scale = None
            tStScale_r2t = None
        tmem_store_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(16)),
            Float32,
        )
        if const_expr(not self.folded_acc):
            thr_tmem_store = tcgen05.make_tmem_copy(tmem_store_atom, tStP).get_slice(tidx)
            tStP_r2t = thr_tmem_store.partition_D(tStP)
            if const_expr(self.use_s_ping_pong):
                tStP_1 = cute.make_tensor(tStSi_1.iterator + self.tmem_s_to_p_offset, tStP_layout)
                tStP_r2t_1 = thr_tmem_store.partition_D(tStP_1)
            else:
                tStP_r2t_1 = None
        else:
            # P goes to SMEM as the PV MMA's A operand. sP's layout comes from
            # make_smem_layout_a: swizzled and core-matrix tiled,
            # ((MMA_M, MMA_K), m_tiles, k_tiles, stage). The row-major view below is only
            # valid because the SWIZZLE LIVES ON THE ITERATOR (get_tensor(..., swizzle=)) and,
            # at n_block_size * sizeof(q_dtype) == 128B, the tiled layout's PRE-SWIZZLE offset
            # for (row, col) is exactly row * n + col. Verified on GPU: addressing sP through
            # its natural ((row % M0, col % K0), row // M0, col // K0) coordinate instead
            # produced bit-identical output.
            assert self.n_block_size * self.q_dtype.width // 8 == 128, (
                "folded P-in-SMEM assumes one P row is exactly one 128B swizzle period; "
                "address sP by its natural MMA A coordinate instead"
            )
            thr_tmem_store = None
            sP_logical = cute.make_tensor(
                sP.iterator,
                # Same MODE STRUCTURE as tStSi (((m, n_folded), 1, 1)), because the tiled copy
                # built over tStSi carries a rank-3 tiler and partition_D requires
                # rank(input) >= rank(tiler). Mode 0 is the row-major (m, n) element space.
                cute.make_layout(
                    ((self.m_block_size, self.n_block_size), 1, 1),
                    stride=((self.n_block_size, 1), 0, 0),
                ),
            )
            tStP_r2t = thr_tmem_load.partition_D(sP_logical)

        # (row within this CTA's m_block_size accumulator rows, which half of the row this
        # thread holds). Non-folded: (tidx, 0). See acc_row_half.
        acc_row, acc_half = self.acc_row_half(thr_tmem_load.partition_D(tScS), tidx)

        # With S ping-pong this carries the global block counter instead of a phase bit
        # (see softmax_step): block c uses S/P slot c & 1 on barrier phase (c >> 1) & 1.
        mma_si_consumer_phase = Int32(0)
        si_corr_producer_phase = Int32(1)
        s0_s1_sequence_phase = Int32(1 if stage == 0 else 0)

        # self.warp_scheduler_barrier_init()

        warp_idx_in_wg = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
        mbar_s0_s1_sequence_offset = self.mbar_s0_s1_sequence_offset + warp_idx_in_wg

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()

        if const_expr(self.enable_flashmask):
            generate_block_consumer_state = cutlass.pipeline.make_pipeline_state(
                cutlass.pipeline.PipelineUserType.Consumer, self.generate_block_stage
            )
            load_startend_row_indices_consumer_state = cutlass.pipeline.make_pipeline_state(
                cutlass.pipeline.PipelineUserType.Consumer, self.kv_stage
            )

        while work_tile.is_valid_tile:

            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, m_block, split_idx, num_splits)

            mask = AttentionMaskCls(seqlen.seqlen_q, seqlen.seqlen_k)
            # `stage` is the Q tile index within this CTA's q_stage tiles (single_q_stage has a
            # single tile, so stage is always 0 there).
            m_block_for_mask = self.m_tile_pair_base(m_block, stage)
            if const_expr(self.enable_flashmask):
                shared_mask_kwargs = dict(
                    m_block=m_block_for_mask,
                    thr_mma=thr_mma_qk,
                    thr_tmem_load=thr_tmem_load,
                    mask_causal=self.is_causal,
                    enable_flashmask=self.enable_flashmask,
                    mask_local=self.is_local,
                    batch_idx=batch_idx,
                    head_idx=head_idx,
                    s_startend_row_indices=s_startend_row_indices,
                    has_lt_end=self.has_lt_end,
                    has_ut_start=self.has_ut_start,
                    has_ut_end=self.has_ut_end,
                    mbar_ptr=mbar_ptr,
                    mbar_load_startend_row_indices_empty_offset=self.mbar_load_startend_row_indices_empty_offset,
                    mbar_load_startend_row_indices_full_offset=self.mbar_load_startend_row_indices_full_offset,
                    kv_stage=self.kv_stage,
                    stage=stage,
                    generate_block_buffer_usable_block_count=self.generate_block_buffer_usable_block_count,
                )
            else:
                shared_mask_kwargs = dict(
                    m_block=m_block_for_mask,
                    thr_mma=thr_mma_qk,
                    thr_tmem_load=thr_tmem_load,
                    mask_causal=self.is_causal,
                    enable_flashmask=self.enable_flashmask,
                    mask_local=self.is_local,
                    batch_idx=batch_idx,
                    head_idx=head_idx,
                )
            mask_fn = partial(
                mask.apply_mask_sm100,
                # mask_r2p masks by register index (it assumes tScS_t2r[i][1] == i), which is
                # false for the folded accumulator: thread t + m_block_size holds columns
                # n_block/2.. so its register index is short by n_block/2 and the out-of-range
                # keys stay unmasked.
                use_r2p=not self.folded_acc,
                **shared_mask_kwargs,
            )
            if const_expr(self.use_block_sparsity):
                #  Full blocks don't need extra masking
                mask_fn_none = partial(
                    mask.apply_mask_sm100,
                    use_r2p=not self.folded_acc,
                    **shared_mask_kwargs,
                )
            else:
                mask_fn_none = None

            softmax = SoftmaxSm100.create(
                softmax_scale_log2,
                rescale_threshold=8.0 if const_expr(self.q_dtype.width == 16) else 0.0,
                softmax_scale=softmax_scale,
            )
            softmax.reset()

            if const_expr(self.use_block_sparsity):
                tile_block_count = get_total_block_count(blocksparse_tensors, batch_idx, head_idx, m_block)
                has_work = tile_block_count > Int32(0)
            else:
                tile_block_count = n_block_max - n_block_min
                has_work = const_expr(not self.is_split_kv) or tile_block_count > Int32(0)

            softmax_step = partial(
                self.softmax_step,
                softmax=softmax,
                mbar_ptr=mbar_ptr,
                mbar_s0_s1_sequence_offset=mbar_s0_s1_sequence_offset,
                thr_mma_qk=thr_mma_qk,
                thr_tmem_load=thr_tmem_load,
                thr_tmem_store=thr_tmem_store,
                thr_tmem_store_scale=thr_tmem_store_scale,
                tStS_t2r=tStS_t2r,
                tStScale_r2t=tStScale_r2t,
                tStP_r2t=tStP_r2t,
                tStS_t2r_1=tStS_t2r_1,
                tStP_r2t_1=tStP_r2t_1,
                sScale=sScale,
                stage=stage,
                batch_idx=batch_idx,
                head_idx=head_idx,
                m_block=m_block_for_mask,
                seqlen=seqlen,
                mBlockLogit=mBlockLogit,
                mBlockBos=mBlockBos,
            )

            if has_work:
                # Softmax acts as the producer: wait until correction signals the stage is empty
                cute.arch.mbarrier_wait(
                    mbar_ptr + self.mbar_softmax_corr_empty_offset + stage, si_corr_producer_phase
                )
                si_corr_producer_phase ^= 1

            if const_expr(self.fm_persistent):
                if n_block_min >= n_block_max:
                    # No KV block at all: no correction handshake either.
                    si_corr_producer_phase ^= 1

            if const_expr(self.enable_flashmask):
                if n_block_min < n_block_max:
                    cute.arch.mbarrier_wait(mbar_ptr + self.mbar_generate_block_full_offset + generate_block_consumer_state.index, generate_block_consumer_state.phase)
                    n_block_idx = 0
                    s_n_block_cur_stage = cute.make_tensor(s_n_block.iterator + generate_block_consumer_state.index * self.generate_block_buffer_block_count, cute.make_layout(self.generate_block_buffer_block_count))
                    s_extra_flags_cur_stage = cute.make_tensor(s_extra_flags.iterator + generate_block_consumer_state.index, cute.make_layout(1))
                    n_block_first = self.n_block_getter(s_n_block_cur_stage, s_extra_flags_cur_stage, n_block_idx)

                    # generate_block make sure n_block_first won't be self.generate_block_incomplete
                    if n_block_first < n_block_min and n_block_first != Int32(self.generate_block_incomplete):
                        cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_generate_block_empty_offset + generate_block_consumer_state.index)
                        generate_block_consumer_state.advance()
                        if const_expr(self.fm_persistent):
                            # Empty tile: correction does no handshake for it (valid_block_count
                            # == 0), so undo the producer phase flip taken in `if has_work`; the
                            # end-of-loop advance moves to the next tile.
                            si_corr_producer_phase ^= 1
                        else:
                            # Advance to next tile
                            tile_scheduler.advance_to_next_work()
                            work_tile = tile_scheduler.get_current_work()
                    elif const_expr(not self.is_split_kv) or tile_block_count > Int32(0):

                        if const_expr(self.fm_one_softmax):
                            # One loop and one softmax_step instance for every block of the tile (see
                            # fm_one_softmax). causal_active gates the causal / seqlen_k R2P pass to the
                            # diagonal blocks and the last block; the flashmask pass inside mask_fn already
                            # skips full blocks at runtime. The first block goes through the regular (not
                            # is_first) path: softmax.reset() set row_max = -inf and row_sum = 0, so
                            # acc_scale = 0 and row_sum starts from this block's sum; correction ignores the
                            # first block's signal, so the sScale value it writes is unused.
                            if const_expr(self.is_causal):
                                n_block_min_causal_local_mask = block_info.get_n_block_min_causal_local_mask(
                                    seqlen, m_block, n_block_min
                                )
                            else:
                                n_block_min_causal_local_mask = n_block_max
                            n_block_last = n_block_max - 1
                            n_block = n_block_first
                            while n_block >= n_block_min or n_block == Int32(self.generate_block_incomplete):
                                while n_block >= n_block_min:
                                    mma_si_consumer_phase, si_corr_producer_phase, s0_s1_sequence_phase, load_startend_row_indices_consumer_state = softmax_step(
                                        mma_si_consumer_phase, si_corr_producer_phase, s0_s1_sequence_phase, n_block,
                                        mask_fn=partial(
                                            mask_fn,
                                            mask_seqlen=True,
                                            load_startend_row_indices_consumer_state=load_startend_row_indices_consumer_state,
                                            n_block_idx=n_block_idx,
                                            encode_n_block=self.mask_n_block_getter(s_n_block_cur_stage, n_block_idx),
                                            causal_active=n_block >= n_block_min_causal_local_mask or n_block == n_block_last,
                                        ),
                                    )
                                    n_block_idx += 1
                                    n_block = self.n_block_getter(s_n_block_cur_stage, s_extra_flags_cur_stage, n_block_idx)

                                if n_block == Int32(self.generate_block_incomplete):
                                    cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_generate_block_empty_offset + generate_block_consumer_state.index)
                                    generate_block_consumer_state.advance()
                                    cute.arch.mbarrier_wait(mbar_ptr + self.mbar_generate_block_full_offset + generate_block_consumer_state.index, generate_block_consumer_state.phase)
                                    s_n_block_cur_stage = cute.make_tensor(
                                                              s_n_block.iterator + generate_block_consumer_state.index * self.generate_block_buffer_block_count,
                                                              cute.make_layout(self.generate_block_buffer_block_count),
                                                          )
                                    s_extra_flags_cur_stage = cute.make_tensor(
                                                                  s_extra_flags.iterator + generate_block_consumer_state.index,
                                                                  cute.make_layout(1),
                                                              )
                                    n_block_idx = 0
                                    n_block = self.n_block_getter(s_n_block_cur_stage, s_extra_flags_cur_stage, n_block_idx)
                        else:
                            mma_si_consumer_phase, si_corr_producer_phase, s0_s1_sequence_phase, load_startend_row_indices_consumer_state = softmax_step(
                                mma_si_consumer_phase,
                                si_corr_producer_phase,
                                s0_s1_sequence_phase,
                                n_block_first,
                                is_first=True,
                                mask_fn=partial(
                                    mask_fn,
                                    mask_seqlen=True,
                                    load_startend_row_indices_consumer_state=load_startend_row_indices_consumer_state,
                                    n_block_idx=n_block_idx,
                                    encode_n_block=self.mask_n_block_getter(s_n_block_cur_stage, n_block_idx)
                                ),
                            )

                            n_block_max -= 1
                            # Next couple of iterations with causal masking
                            n_block = n_block_first
                            if const_expr(self.is_causal or self.is_local):
                                n_block_min_causal_local_mask = block_info.get_n_block_min_causal_local_mask(
                                    seqlen, m_block, n_block_min
                                )
                                # advance n_block_idx right before a n_block_getter call, so n_block_idx is reserved for mask_fn (to check whether a block is fully masked)
                                n_block_idx += 1
                                n_block = self.n_block_getter(s_n_block_cur_stage, s_extra_flags_cur_stage, n_block_idx)
                                while n_block >= n_block_max - 1 - (n_block_max - n_block_min_causal_local_mask - 1) or n_block == Int32(self.generate_block_incomplete):
                                    while n_block >= n_block_max - 1 - (n_block_max - n_block_min_causal_local_mask - 1):
                                        mma_si_consumer_phase, si_corr_producer_phase, s0_s1_sequence_phase, load_startend_row_indices_consumer_state = (
                                            softmax_step(
                                                mma_si_consumer_phase,
                                                si_corr_producer_phase,
                                                s0_s1_sequence_phase,
                                                n_block,
                                                mask_fn=partial(
                                                    mask_fn,
                                                    mask_seqlen=False,
                                                    load_startend_row_indices_consumer_state=load_startend_row_indices_consumer_state,
                                                    n_block_idx=n_block_idx,
                                                    encode_n_block=self.mask_n_block_getter(s_n_block_cur_stage, n_block_idx)
                                                ),
                                            )
                                        )
                                        n_block_idx += 1
                                        n_block = self.n_block_getter(s_n_block_cur_stage, s_extra_flags_cur_stage, n_block_idx)

                                    if n_block == Int32(self.generate_block_incomplete):
                                        cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_generate_block_empty_offset + generate_block_consumer_state.index)
                                        generate_block_consumer_state.advance()
                                        cute.arch.mbarrier_wait(mbar_ptr + self.mbar_generate_block_full_offset + generate_block_consumer_state.index, generate_block_consumer_state.phase)
                                        s_n_block_cur_stage = cute.make_tensor(
                                                                  s_n_block.iterator + generate_block_consumer_state.index * self.generate_block_buffer_block_count,
                                                                  cute.make_layout(self.generate_block_buffer_block_count),
                                                              )
                                        s_extra_flags_cur_stage = cute.make_tensor(
                                                                      s_extra_flags.iterator + generate_block_consumer_state.index,
                                                                      cute.make_layout(1),
                                                                  )
                                        n_block_idx = 0
                                        n_block = self.n_block_getter(s_n_block_cur_stage, s_extra_flags_cur_stage, n_block_idx)

                                n_block_max = cutlass.min(n_block_max, n_block_min_causal_local_mask)
                            else:
                                n_block_idx += 1
                                n_block = self.n_block_getter(s_n_block_cur_stage, s_extra_flags_cur_stage, n_block_idx)

                            # The remaining iterations have no masking
                            n_block_min_before_local_mask = block_info.get_n_block_min_before_local_mask(
                                seqlen, m_block, n_block_min
                            )

                            while n_block >= n_block_max - (n_block_max - n_block_min_before_local_mask - 1) - 1 or n_block == Int32(self.generate_block_incomplete):
                                while n_block >= n_block_max - (n_block_max - n_block_min_before_local_mask - 1) - 1:
                                    mma_si_consumer_phase, si_corr_producer_phase, s0_s1_sequence_phase, load_startend_row_indices_consumer_state = softmax_step(
                                        mma_si_consumer_phase, si_corr_producer_phase, s0_s1_sequence_phase, n_block,
                                        # These blocks lie entirely inside the causal / local band, so only
                                        # the flashmask part of mask_fn applies (the non-flashmask path below
                                        # passes no mask_fn here). Keeping mask_causal cost an R2P mask plus
                                        # one SEL per S element on every block (ncu: ~150 of 731 softmax
                                        # instructions per block per warp, SW hd256).
                                        mask_fn=partial(
                                            mask_fn,
                                            mask_seqlen=False,
                                            mask_causal=False,
                                            mask_local=False,
                                            load_startend_row_indices_consumer_state=load_startend_row_indices_consumer_state,
                                            n_block_idx=n_block_idx,
                                            encode_n_block=self.mask_n_block_getter(s_n_block_cur_stage, n_block_idx)
                                        ),
                                    )

                                    n_block_idx += 1
                                    n_block = self.n_block_getter(s_n_block_cur_stage, s_extra_flags_cur_stage, n_block_idx)

                                if n_block == Int32(self.generate_block_incomplete):
                                    cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_generate_block_empty_offset + generate_block_consumer_state.index)
                                    generate_block_consumer_state.advance()
                                    cute.arch.mbarrier_wait(mbar_ptr + self.mbar_generate_block_full_offset + generate_block_consumer_state.index, generate_block_consumer_state.phase)
                                    s_n_block_cur_stage = cute.make_tensor(
                                                              s_n_block.iterator + generate_block_consumer_state.index * self.generate_block_buffer_block_count,
                                                              cute.make_layout(self.generate_block_buffer_block_count),
                                                          )
                                    s_extra_flags_cur_stage = cute.make_tensor(
                                                                  s_extra_flags.iterator + generate_block_consumer_state.index,
                                                                  cute.make_layout(1),
                                                              )
                                    n_block_idx = 0
                                    n_block = self.n_block_getter(s_n_block_cur_stage, s_extra_flags_cur_stage, n_block_idx)

                            # Separate iterations with local masking on the left
                            if const_expr(self.is_local and block_info.window_size_left is not None):
                                n_block_max = cutlass.min(n_block_max, n_block_min_before_local_mask)
                                while n_block >= n_block_max - 1 - (n_block_max - n_block_min - 1) or n_block == Int32(self.generate_block_incomplete):
                                    while n_block >= n_block_max - 1 - (n_block_max - n_block_min - 1):
                                        mma_si_consumer_phase, si_corr_producer_phase, s0_s1_sequence_phase, load_startend_row_indices_consumer_state = (
                                            softmax_step(
                                                mma_si_consumer_phase,
                                                si_corr_producer_phase,
                                                s0_s1_sequence_phase,
                                                n_block,
                                                mask_fn=partial(
                                                    mask_fn,
                                                    mask_seqlen=False,
                                                    load_startend_row_indices_consumer_state=load_startend_row_indices_consumer_state,
                                                    n_block_idx=n_block_idx,
                                                    encode_n_block=self.mask_n_block_getter(s_n_block_cur_stage, n_block_idx)
                                                ),
                                            )
                                        )
                                        n_block_idx += 1
                                        n_block = self.n_block_getter(s_n_block_cur_stage, s_extra_flags_cur_stage, n_block_idx)
                                        # Now that we no longer already have the 1st iteration, need mask_seqlen=True here

                                    if n_block == Int32(self.generate_block_incomplete):
                                        cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_generate_block_empty_offset + generate_block_consumer_state.index)
                                        generate_block_consumer_state.advance()
                                        cute.arch.mbarrier_wait(mbar_ptr + self.mbar_generate_block_full_offset + generate_block_consumer_state.index, generate_block_consumer_state.phase)
                                        s_n_block_cur_stage = cute.make_tensor(
                                                                  s_n_block.iterator + generate_block_consumer_state.index * self.generate_block_buffer_block_count,
                                                                  cute.make_layout(self.generate_block_buffer_block_count),
                                                              )
                                        s_extra_flags_cur_stage = cute.make_tensor(
                                                                      s_extra_flags.iterator + generate_block_consumer_state.index,
                                                                      cute.make_layout(1),
                                                                  )
                                        n_block_idx = 0
                                        n_block = self.n_block_getter(s_n_block_cur_stage, s_extra_flags_cur_stage, n_block_idx)

                        # TODO: softmax.reset() will deal with tile that all blocks of which are skipped.
                        # Dense path always writes scale / signals
                        # Folded accumulator: row_sum is a per-half partial, so combine it with
                        # the thread holding the other half of this row first. Both threads of
                        # the pair then store the same value to the same slot.
                        row_sum_total = self.pair_exchange(
                            sScale, acc_row, acc_half, softmax.row_sum[0], is_max=False
                        )
                        sScale[acc_row + stage * self.m_block_size] = row_sum_total
                        if const_expr(mLSE is not None or learnable_sink is not None):
                            sScale[
                                acc_row + stage * self.m_block_size + self.m_block_size * 2
                            ] = softmax.row_max[0]
                        cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_softmax_corr_full_offset + stage)

                        cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_generate_block_empty_offset + generate_block_consumer_state.index)
                        generate_block_consumer_state.advance()
            # Block sparse or dense iteration
            elif const_expr(self.use_block_sparsity):
                (
                    mma_si_consumer_phase,
                    si_corr_producer_phase,
                    s0_s1_sequence_phase,
                    empty_tile,
                ) = softmax_block_sparse_sm100(
                    blocksparse_tensors,
                    batch_idx,
                    head_idx,
                    m_block,
                    softmax_step,
                    mask_fn,
                    mask_fn_none,
                    mma_si_consumer_phase,
                    si_corr_producer_phase,
                    s0_s1_sequence_phase,
                    mbar_ptr,
                    self.mbar_softmax_corr_full_offset,
                    self.mbar_softmax_corr_empty_offset,
                    self.mbar_P_full_O_rescaled_offset,
                    self.mbar_P_full_2_offset,
                    self.q_stage,
                    Int32(stage),
                )
                if not empty_tile:
                    row_sum_total = self.pair_exchange(
                        sScale, acc_row, acc_half, softmax.row_sum[0], is_max=False
                    )
                    sScale[acc_row + stage * self.m_block_size] = row_sum_total
                    if const_expr(mLSE is not None or learnable_sink is not None):
                        sScale[
                            acc_row + stage * self.m_block_size + self.m_block_size * 2
                        ] = softmax.row_max[0]
                    cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_softmax_corr_full_offset + stage)
            else:
                if const_expr(not self.is_split_kv) or tile_block_count > Int32(0):
                    mma_si_consumer_phase, si_corr_producer_phase, s0_s1_sequence_phase, _ = softmax_step(
                        mma_si_consumer_phase,
                        si_corr_producer_phase,
                        s0_s1_sequence_phase,
                        n_block_max - 1,
                        is_first=True,
                        mask_fn=partial(mask_fn, mask_seqlen=True),
                    )
                    n_block_max -= 1
                    # Next couple of iterations with causal masking
                    if const_expr(self.is_causal or self.is_local):
                        n_block_min_causal_local_mask = block_info.get_n_block_min_causal_local_mask(
                            seqlen, m_block, n_block_min
                        )
                        for n_tile in cutlass.range(n_block_max - n_block_min_causal_local_mask, unroll=1):
                            n_block = n_block_max - 1 - n_tile
                            mma_si_consumer_phase, si_corr_producer_phase, s0_s1_sequence_phase, _ = (
                                softmax_step(
                                    mma_si_consumer_phase,
                                    si_corr_producer_phase,
                                    s0_s1_sequence_phase,
                                    n_block,
                                    mask_fn=partial(mask_fn, mask_seqlen=False),
                                )
                            )
                        n_block_max = cutlass.min(n_block_max, n_block_min_causal_local_mask)
                    # The remaining iterations have no masking
                    n_block_min_before_local_mask = block_info.get_n_block_min_before_local_mask(
                        seqlen, m_block, n_block_min
                    )
                    for n_tile in cutlass.range(n_block_max - n_block_min_before_local_mask, unroll=1):
                        n_block = n_block_max - n_tile - 1
                        mma_si_consumer_phase, si_corr_producer_phase, s0_s1_sequence_phase, _ = softmax_step(
                            mma_si_consumer_phase, si_corr_producer_phase, s0_s1_sequence_phase, n_block,
                        )
                    # Separate iterations with local masking on the left
                    if const_expr(self.is_local and block_info.window_size_left is not None):
                        n_block_max = cutlass.min(n_block_max, n_block_min_before_local_mask)
                        for n_tile in cutlass.range(0, n_block_max - n_block_min, unroll=1):
                            n_block = n_block_max - 1 - n_tile
                            mma_si_consumer_phase, si_corr_producer_phase, s0_s1_sequence_phase, _ = (
                                softmax_step(
                                    mma_si_consumer_phase,
                                    si_corr_producer_phase,
                                    s0_s1_sequence_phase,
                                    n_block,
                                    mask_fn=partial(mask_fn, mask_seqlen=False),
                                )
                            )
                            # Now that we no longer already have the 1st iteration, need mask_seqlen=True here

                    # Dense path always writes scale / signals
                    row_sum_total = self.pair_exchange(
                        sScale, acc_row, acc_half, softmax.row_sum[0], is_max=False
                    )
                    sScale[acc_row + stage * self.m_block_size] = row_sum_total
                    if const_expr(mLSE is not None or learnable_sink is not None):
                        sScale[
                            acc_row + stage * self.m_block_size + self.m_block_size * 2
                        ] = softmax.row_max[0]
                    cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_softmax_corr_full_offset + stage)

            # Advance to next tile
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()
        # End of persistent scheduler loop

    @cute.jit
    def softmax_step(
        self,
        mma_si_consumer_phase: Int32,
        si_corr_producer_phase: Int32,
        s0_s1_sequence_phase: Int32,
        n_block: Int32,
        softmax: SoftmaxSm100,
        mbar_ptr: cute.Pointer,
        mbar_s0_s1_sequence_offset: Int32,
        thr_mma_qk: cute.core.ThrMma,
        thr_tmem_load: cute.CopyAtom,
        thr_tmem_store: Optional[cute.CopyAtom],
        # Dead (their cute.copy calls are commented out) and None with the folded
        # accumulator, where an (m, 1) tmem tile is not well formed. Optional so the DSL does
        # not try to build an IR type for None.
        thr_tmem_store_scale: Optional[cute.CopyAtom],
        tStS_t2r: cute.Tensor,
        tStScale_r2t: Optional[cute.Tensor],
        tStP_r2t: cute.Tensor,
        sScale: cute.Tensor,
        stage: int | Int32,
        batch_idx: Int32,
        head_idx: Int32,
        m_block: Int32,
        seqlen,
        mask_fn: Optional[Callable] = None,
        is_first: bool = False,
        mBlockLogit: Optional[cute.Tensor] = None,
        mBlockBos: Optional[cute.Tensor] = None,
        tStS_t2r_1: Optional[cute.Tensor] = None,
        tStP_r2t_1: Optional[cute.Tensor] = None,
    ) -> Tuple[cute.Int32, cute.Int32, cute.Int32, Optional[cutlass.pipeline.PipelineState]]:
        """Perform a single step of the softmax computation on a block of attention scores.

        This method processes one block of the attention matrix, computing numerically stable
        softmax by first finding the row maximum, subtracting it from all elements, applying
        exponential function, and then normalizing by the sum of exponentials. It also handles
        optional masking of attention scores.

        The method involves several key operations:
        1. Loading attention scores from tensor memory
        2. Applying optional masking based on position
        3. Computing row-wise maximum values for numerical stability
        4. Transforming scores using exp2(x*scale - max*scale)
        5. Computing row sums for normalization
        6. Coordinating pipeline synchronization between different processing stages
        """

        tilePlikeFP32 = self.mma_tiler_qk[1] // Float32.width * self.v_dtype.width
        tScS = thr_mma_qk.partition_C(cute.make_identity_tensor(self.mma_tiler_qk[:2]))
        if const_expr(not self.folded_acc):
            tScScale = cute.composition(tScS, cute.make_layout((self.m_block_size, 1)))
            tScP = cute.composition(tScS, cute.make_layout((self.m_block_size, tilePlikeFP32)))
        else:
            # Folded accumulator: `composition` cannot express the lane-half split (see
            # softmax_loop's tStP_layout), and these two views are only ever used for their
            # per-thread SHAPE. Take it from the store's own tmem-side partition, which a
            # tiled copy guarantees to match the register side element for element.
            tScScale = None
            tScP = None

        # Wait for Si

        tSrS_t2r = cute.make_fragment(thr_tmem_load.partition_D(tScS).shape, self.qk_acc_dtype)
        if const_expr(self.use_s_ping_pong):
            # mma_si_consumer_phase is the global block counter here (see softmax_loop).
            pp_slot = mma_si_consumer_phase & 1
            cute.arch.mbarrier_wait(
                mbar_ptr + self.mbar_S_full_offset + pp_slot, (mma_si_consumer_phase >> 1) & 1
            )
            if const_expr(self.fm_one_softmax):
                # One code path for both S slots: offset the TMEM address by the slot instead
                # of branching between two copies of the load (and of the P store).
                pp_col = pp_slot * (self.tmem_s_offset[1] - self.tmem_s_offset[0])
                cute.copy(
                    thr_tmem_load,
                    cute.make_tensor(tStS_t2r.iterator + pp_col, tStS_t2r.layout),
                    tSrS_t2r,
                )
            elif pp_slot == 0:
                cute.copy(thr_tmem_load, tStS_t2r, tSrS_t2r)
            else:
                cute.copy(thr_tmem_load, tStS_t2r_1, tSrS_t2r)
        else:
            cute.arch.mbarrier_wait(mbar_ptr + self.mbar_S_full_offset + stage, mma_si_consumer_phase)
            cute.copy(thr_tmem_load, tStS_t2r, tSrS_t2r)

        if const_expr(mask_fn is not None):
            load_startend_row_indices_consumer_state = mask_fn(tSrS_t2r, n_block=n_block)
        else:
            load_startend_row_indices_consumer_state = None

        # --- Fused block-max score (HySparse) ---------------------------------
        # tSrS_t2r currently holds the post-mask q.k logit for this n-tile: mask_fn
        # has run, but the exp has not yet touched it. We take the per-block max here
        # and store the SCALED attention logit `softmax_scale * q.k` -- i.e. the exact
        # value that feeds softmax. Storing the scaled value keeps block_logit on a
        # single, head-independent scale so a downstream `block_logit - LSE` yields
        # log(max attention weight in the block), which IS comparable across heads.
        #
        # NB on where the scale lives: the scale is folded into scale_log2
        # (= softmax_scale * LOG2_E) and applied only later in exp2, so tSrS_t2r here
        # is still UNSCALED. To emit the scaled logit we therefore multiply by
        # softmax_scale (== scale_log2 * ln2) below.
        #
        # Masked-out columns are -Float32.inf and never win a block max. On sm100
        # each softmax thread owns ONE complete query row (all n_block_size columns)
        # -- SoftmaxSm100._compute_row_max is a pure within-fragment fmax_reduce with
        # NO cross-lane shuffle -- so we reduce this thread's fragment straight into
        # per-block maxes by bucketing columns into `blocks_per_ntile` sub-blocks of
        # `block_size`, with no warp reduce. The result is written to gmem mirroring
        # the LSE write (thr_idx -> row within the m-tile). This adds no MMA / smem /
        # cross-warp traffic, so the score rides the fast FA4 path.
        if const_expr(self.has_block_logit):
            tScS_t2r_blk = thr_tmem_load.partition_D(tScS)
            blk_tidx = thr_tmem_load.thr_idx
            row = m_block * self.m_block_size + blk_tidx
            # DOCUMENT-RELATIVE bucketing (pack-equivalence, Bug 2 fix).
            # The downstream HySparse pipeline interprets block ids relative to
            # each document's start (bos): block j covers key columns
            # [bos + j*block_size, bos + (j+1)*block_size). To make packed
            # (bos>0) selection bit-identical to running the document alone
            # (bos=0), we bucket each column by its DOCUMENT-relative block
            #   rel = floor((abs_col - bos) / block_size)
            # rather than the absolute packed-sequence block abs_col//block_size.
            # bos is per query row; masked-out (cross-document / future) columns
            # are already -Float32.inf here so they never win an fmax. When
            # has_block_bos is False we fall back to bos=0, i.e. the original
            # absolute (single-document) bucketing.
            bos = Int32(0)
            if const_expr(self.has_block_bos):
                # Padding threads (row >= seqlen_q) never write below, but must
                # still index in-bounds for the load; clamp the row.
                safe_row = cutlass.min(row, seqlen.seqlen_q - 1)
                bos = Int32(mBlockBos[batch_idx, safe_row])
            # floor-div by the power-of-two block_size via arithmetic shift, which
            # rounds toward -inf for the (abs_col - bos) < 0 (pre-document) columns.
            base_blk = (n_block * self.n_block_size - bos) >> self.block_size_log2
            blk_max = cute.make_fragment(self.n_block_slots, Float32)
            for b in cutlass.range_constexpr(self.n_block_slots):
                blk_max[b] = -Float32.inf
            n_elems = const_expr(cute.size(tScS_t2r_blk.shape))
            if const_expr(self.has_block_bos):
                # DOCUMENT-RELATIVE bucketing (bos may be > 0). col_i is a
                # COMPILE-TIME column coordinate within the n-tile, so qi (which
                # absolute block_size sub-block it is in) and mi (its offset inside
                # that sub-block) are constexpr; only the relative carry uses the
                # runtime bos. With r = (n_block*N - bos) mod block_size in
                # [0, block_size), the element's relative slot is
                #   qi + carry,   carry = (r + mi) >> block_size_log2  in {0, 1}
                # i.e. it lands in one of only TWO ADJACENT slots, both indexed by
                # a constexpr (qi, qi+1). The carry is 1 IFF r + mi >= block_size,
                # i.e. mi >= (block_size - r). With the runtime threshold
                # t = block_size - r computed ONCE per n-tile, each element needs
                # only a single runtime compare (mi >= t; mi is constexpr) driving
                # a complementary-predicate if/else into constexpr slots qi (low) /
                # qi+1 (high). Exactly one branch fires, so a masked-out (-inf)
                # column still cannot win either slot's fmax.
                t = self.block_size - (
                    (n_block * self.n_block_size - bos) & (self.block_size - 1)
                )
                for i in cutlass.range_constexpr(n_elems):
                    col_i = const_expr(tScS_t2r_blk[i][1])
                    qi = const_expr(col_i >> self.block_size_log2)
                    mi = const_expr(col_i & (self.block_size - 1))
                    if mi >= t:
                        blk_max[qi + 1] = cute.arch.fmax(
                            blk_max[qi + 1], tSrS_t2r[i]
                        )
                    else:
                        blk_max[qi] = cute.arch.fmax(
                            blk_max[qi], tSrS_t2r[i]
                        )
            else:
                # ABSOLUTE bucketing (bos == 0): the relative slot degenerates to
                # the constexpr absolute sub-block qi = col_i // block_size, which
                # folds at compile time to a single static fmax per element (the
                # fast path; kept byte-identical to the pre-relative kernel).
                for i in cutlass.range_constexpr(n_elems):
                    col_i = const_expr(tScS_t2r_blk[i][1])
                    qi = const_expr(col_i >> self.block_size_log2)
                    blk_max[qi] = cute.arch.fmax(blk_max[qi], tSrS_t2r[i])
            # Unify units: emit the SCALED attention logit. tSrS_t2r still carries
            # the raw q.k here, so multiply by the raw softmax_scale, recovered from
            # scale_log2 (= softmax_scale * log2(e)) as scale_log2 * ln(2).
            # -inf * (finite > 0) stays -inf, so masked blocks are untouched.
            blk_logit_scale = softmax.scale_log2 * math.log(2.0)
            for b in cutlass.range_constexpr(self.n_block_slots):
                blk_max[b] = blk_max[b] * blk_logit_scale
            # `m_block` here is already the per-stage query m-tile index
            # (m_block_for_mask = q_stage*m_block_raw + stage, bound by the
            # softmax_step partial), so it maps 1:1 to the query rows this stage
            # owns, and every stage owns a distinct m-tile.
            if blk_tidx < seqlen.seqlen_q - m_block * self.m_block_size:
                mBL_cur = mBlockLogit[None, head_idx, batch_idx, None]
                # mBlockLogit has exactly ceil(seqlen_k / block_size) columns.
                num_cols = (seqlen.seqlen_k + self.block_size - 1) >> self.block_size_log2
                for b in cutlass.range_constexpr(self.n_block_slots):
                    rb = base_blk + b
                    # A relative block can straddle two adjacent n-tiles (the
                    # bos shift is unaligned), so each row's thread combines
                    # this n-tile's partial into any prior write with fmax.
                    # This is a same-thread, same-address RMW across the n_block
                    # loop (one thread owns one query row for all n-tiles), so
                    # no cross-thread race. Guard columns outside [0, num_cols)
                    # (pre-document rb<0 or padding rb>=num_cols; both -inf).
                    if rb >= 0 and rb < num_cols:
                        mBL_cur[row, rb] = cute.arch.fmax(mBL_cur[row, rb], blk_max[b])
        # ----------------------------------------------------------------------

        # Folded accumulator: this thread's fragment only covers HALF of its row, so the raw
        # fmax_reduce is a half-row max. Both halves of a row must be scaled by the same max
        # (otherwise the two halves of P are on different scales and their PV contributions no
        # longer add up), so combine with the partner thread before turning the max into
        # acc_scale. After the exchange both threads hold identical row_max / acc_scale.
        acc_row, acc_half = self.acc_row_half(
            thr_tmem_load.partition_D(tScS), thr_tmem_load.thr_idx
        )
        if const_expr(self.folded_acc):
            row_max_local = softmax.compute_row_max_local(tSrS_t2r.load(), is_first)
            row_max_pair = self.pair_exchange(
                sScale, acc_row, acc_half, row_max_local, is_max=True
            )
            row_max, acc_scale = softmax.update_row_max_from_local(row_max_pair, is_first)
        else:
            row_max, acc_scale = softmax.update_row_max(tSrS_t2r.load(), is_first)

        if const_expr(not is_first):
            # tSrScale_r2t = cute.make_fragment(thr_tmem_store_scale.partition_S(tScScale).shape, Float32)
            # tSrScale_r2t[0] = acc_scale
            # cute.copy(thr_tmem_store_scale, tSrScale_r2t, tStScale_r2t)
            # cute.arch.fence_view_async_tmem_store()
            sScale[acc_row + stage * self.m_block_size] = acc_scale
        # Notify correction wg that row_max is ready

        cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_softmax_corr_full_offset + stage)

        softmax.scale_subtract_rowmax(tSrS_t2r, row_max)
        # Sequence barrier wait
        if const_expr(self.s0_s1_barrier):
            cute.arch.mbarrier_wait(
                mbar_ptr + mbar_s0_s1_sequence_offset + stage * 4, s0_s1_sequence_phase
            )

        if const_expr(not self.folded_acc):
            tSrP_r2t_f32 = cute.make_fragment(thr_tmem_store.partition_S(tScP).shape, Float32)
            tSrP_r2t = cute.make_tensor(
                cute.recast_ptr(tSrP_r2t_f32.iterator, dtype=self.q_dtype),
                tSrS_t2r.layout,
            )
        else:
            # P goes straight to SMEM, so there is no f32-packed TMEM staging view: allocate
            # the bf16 fragment directly, one element per S element this thread holds.
            tSrP_r2t_f32 = None
            tSrP_r2t = cute.make_fragment(tSrS_t2r.layout, self.q_dtype)
        # softmax.scale_apply_exp2_convert(tSrS_t2r, row_max, tSrP_r2t)
        # Preserve old `e2e=mask_fn is None and head_dim_padded<=128` semantics:
        # when that condition was False, old code went pure-hardware exp2 ⇒ ex2_emu_freq=0.
        ex2_emu_freq = (
            self.ex2_emu_freq if (mask_fn is None and self.head_dim_padded <= 128) else 0
        )
        softmax.apply_exp2_convert(
            tSrS_t2r,
            tSrP_r2t,
            ex2_emu_freq=ex2_emu_freq,
            ex2_emu_start_frg=self.ex2_emu_start_frg,
        )
        # Sequence barrier arrive
        if const_expr(self.s0_s1_barrier):
            cute.arch.mbarrier_arrive(mbar_ptr + mbar_s0_s1_sequence_offset + (1 - stage) * 4)
        # cute.copy(thr_tmem_store, tSrP_r2t_f32, tStP_r2t)
        if const_expr(self.folded_acc):
            # Register -> SMEM. The fragment and its sP partition describe the same (row, col)
            # slots (see the layout note in softmax_loop), so autovec_copy moves them directly;
            # the visibility fence is the shared one (the MMA reads sP as its A operand, not
            # TMEM).
            cute.autovec_copy(tSrP_r2t, tStP_r2t)
            cute.arch.fence_view_async_shared()
            self.mma_barrier_arrive(mbar_ptr + self.mbar_P_full_O_rescaled_offset + stage)
            self.mma_barrier_arrive(mbar_ptr + self.mbar_P_full_2_offset + stage)
        elif const_expr(self.use_s_ping_pong):
            # Same 3/4 + 1/4 split store as below, into this block's P slot.
            n_p_chunks = const_expr(cute.size(tStP_r2t.shape[2]))
            if const_expr(self.fm_one_softmax):
                tStP_cur = cute.make_tensor(
                    tStP_r2t.iterator + pp_slot * (self.tmem_s_offset[1] - self.tmem_s_offset[0]),
                    tStP_r2t.layout,
                )
                for i in cutlass.range_constexpr(n_p_chunks // 4 * 3):
                    cute.copy(thr_tmem_store, tSrP_r2t_f32[None, None, i], tStP_cur[None, None, i])
                cute.arch.fence_view_async_tmem_store()
                self.mma_barrier_arrive(mbar_ptr + self.mbar_P_full_O_rescaled_offset + pp_slot)
                for i in cutlass.range_constexpr(n_p_chunks // 4 * 3, n_p_chunks):
                    cute.copy(thr_tmem_store, tSrP_r2t_f32[None, None, i], tStP_cur[None, None, i])
                cute.arch.fence_view_async_tmem_store()
                self.mma_barrier_arrive(mbar_ptr + self.mbar_P_full_2_offset + pp_slot)
            else:
                if pp_slot == 0:
                    for i in cutlass.range_constexpr(n_p_chunks // 4 * 3):
                        cute.copy(thr_tmem_store, tSrP_r2t_f32[None, None, i], tStP_r2t[None, None, i])
                else:
                    for i in cutlass.range_constexpr(n_p_chunks // 4 * 3):
                        cute.copy(thr_tmem_store, tSrP_r2t_f32[None, None, i], tStP_r2t_1[None, None, i])
                cute.arch.fence_view_async_tmem_store()
                self.mma_barrier_arrive(mbar_ptr + self.mbar_P_full_O_rescaled_offset + pp_slot)
                if pp_slot == 0:
                    for i in cutlass.range_constexpr(n_p_chunks // 4 * 3, n_p_chunks):
                        cute.copy(thr_tmem_store, tSrP_r2t_f32[None, None, i], tStP_r2t[None, None, i])
                else:
                    for i in cutlass.range_constexpr(n_p_chunks // 4 * 3, n_p_chunks):
                        cute.copy(thr_tmem_store, tSrP_r2t_f32[None, None, i], tStP_r2t_1[None, None, i])
                cute.arch.fence_view_async_tmem_store()
                self.mma_barrier_arrive(mbar_ptr + self.mbar_P_full_2_offset + pp_slot)
        elif const_expr(self.split_p_store):
            for i in cutlass.range_constexpr(cute.size(tStP_r2t.shape[2]) // 4 * 3):
                cute.copy(thr_tmem_store, tSrP_r2t_f32[None, None, i], tStP_r2t[None, None, i])
            cute.arch.fence_view_async_tmem_store()
            # Notify mma warp that P is ready
            self.mma_barrier_arrive(mbar_ptr + self.mbar_P_full_O_rescaled_offset + stage)

            for i in cutlass.range_constexpr(
                cute.size(tStP_r2t.shape[2]) // 4 * 3, cute.size(tStP_r2t.shape[2])
            ):
                cute.copy(thr_tmem_store, tSrP_r2t_f32[None, None, i], tStP_r2t[None, None, i])
            cute.arch.fence_view_async_tmem_store()
            # Notify mma warp that the 2nd half of P is ready
            self.mma_barrier_arrive(mbar_ptr + self.mbar_P_full_2_offset + stage)
        else:
            # Too few chunks to split (see self.split_p_store): store all of P, then
            # arrive both barriers. The mma warp's P_full_2 wait then never stalls and
            # always reads fully-written P.
            for i in cutlass.range_constexpr(cute.size(tStP_r2t.shape[2])):
                cute.copy(thr_tmem_store, tSrP_r2t_f32[None, None, i], tStP_r2t[None, None, i])
            cute.arch.fence_view_async_tmem_store()
            self.mma_barrier_arrive(mbar_ptr + self.mbar_P_full_O_rescaled_offset + stage)
            self.mma_barrier_arrive(mbar_ptr + self.mbar_P_full_2_offset + stage)

        cute.arch.mbarrier_wait(
            mbar_ptr + self.mbar_softmax_corr_empty_offset + stage, si_corr_producer_phase
        )

        softmax.update_row_sum(tSrS_t2r.load(), acc_scale, is_first)
        # acc_scale = cute.arch.exp2(acc_scale_)
        # cutedsl does not support early exit
        if const_expr(self.use_s_ping_pong):
            next_mma_si = mma_si_consumer_phase + 1  # next global block
        else:
            next_mma_si = mma_si_consumer_phase ^ 1
        return (next_mma_si, si_corr_producer_phase ^ 1, s0_s1_sequence_phase ^ 1, load_startend_row_indices_consumer_state)

    @cute.jit
    def correction_loop(
        self,
        thr_mma_qk: cute.core.ThrMma,
        thr_mma_pv: cute.core.ThrMma,
        tStS: cute.Tensor,
        tOtOs: tuple[cute.Tensor],
        sScale: cute.Tensor,
        mO: cute.Tensor,
        mLSE: cute.Tensor,
        sO: cute.Tensor,
        learnable_sink: Optional[cute.Tensor],
        gmem_tiled_copy_O: cute.TiledCopy,
        tma_atom_O: cute.CopyAtom,
        mbar_ptr: cute.Pointer,
        softmax_scale_log2: Float32,
        block_info: BlockInfo,
        num_splits: Int32,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        flashmask_info: Optional[FlashMaskInfo] = None,
        num_heads: Optional[Int32] = None,
        sOc_full_view: Optional[cute.Tensor] = None,
    ):
        tidx = cute.arch.thread_idx()[0] % (cute.arch.WARP_SIZE * len(self.correction_warp_ids))

        tScS = thr_mma_qk.partition_C(cute.make_identity_tensor(self.mma_tiler_qk[:2]))
        # V-mode coord of this CTA in the MMA pair; selects which m_block_size rows of the
        # work tile this CTA's accumulator half maps to.
        corr_cta_coord_v = thr_mma_pv.thr_idx
        # TMEM vec (row_max/row_sum) views. Dead weight: every `cute.copy` through them is
        # commented out below -- the statistics travel through sScale -- and with the folded
        # accumulator an (m, 1) tile covers only m of the 128 TMEM lanes, so `get_slice(tidx)`
        # would be out of range for half the correction threads. Only build them when they are
        # well formed.
        if const_expr(not self.folded_acc):
            tStScale_layout = cute.composition(
                tStS.layout, cute.make_layout((self.m_block_size, 1))
            )
            tStScales = tuple(
                cute.make_tensor(tStS.iterator + self.tmem_vec_offset[stage], tStScale_layout)
                for stage in range(2)
            )
            tScScale = cute.composition(tScS, cute.make_layout((self.m_block_size, 1)))
            tmem_load_v_atom = cute.make_copy_atom(
                tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(1)),
                self.qk_acc_dtype,
            )
            thr_tmem_load_vec = tcgen05.make_tmem_copy(
                tmem_load_v_atom, tStScales[0]
            ).get_slice(tidx)
            tStScales_t2r = [
                thr_tmem_load_vec.partition_S(tStScales[stage])
                for stage in range(self.num_s_stages)
            ]
            tSrScale_t2r_shape = thr_tmem_load_vec.partition_D(tScScale).shape
        else:
            tStScales_t2r = None
            tSrScale_t2r_shape = None

        # Row this thread's O fragment belongs to. Non-folded: one accumulator row per thread,
        # so tidx. Folded: the O accumulator's 64 rows are spread over 128 TMEM lanes with the
        # dv halves in the lane halves (see folded_o_phys_view), so lane == tidx holds row
        # tidx % m_block_size -- threads t and t + m_block_size share a row's scale, each
        # handling its own dv half.
        if const_expr(not self.folded_acc):
            corr_row = tidx
        else:
            corr_row = tidx % self.m_block_size

        # First iter: no correction is required
        for stage in cutlass.range_constexpr(self.num_s_stages):
            self.mma_barrier_arrive(mbar_ptr + self.mbar_P_full_O_rescaled_offset + stage)

        softmax_corr_consumer_phase = Int32(0)
        o_corr_consumer_phase = Int32(0)
        corr_epi_producer_phase = Int32(1)
        if const_expr(self.use_s_ping_pong):
            # Global block count of the tiles already processed: block j of the current tile
            # uses S/P slot (pp_base + j) & 1 (same numbering as the MMA and softmax warps).
            pp_base = Int32(0)

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        if const_expr(self.enable_flashmask):
            fm_num_heads = flashmask_info.startend_row_indices.shape[1]
            h_h_flashmask_ratio = num_heads // fm_num_heads
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, m_block, split_idx, num_splits)

            if const_expr(self.is_split_kv):
                mO_cur = seqlen.offset_batch_Q(mO, batch_idx, dim=3)[None, None, head_idx, split_idx]
            else:
                mO_cur = seqlen.offset_batch_Q(mO, batch_idx, dim=3)[None, None, head_idx]
            gO = cute.local_tile(mO_cur, (self.m_block_size, self.head_dim_v_padded), (None, 0))

            # Default LSE to -inf for invalid split_idx tiles
            stats = [(0.0, -Float32.inf if const_expr(mLSE is not None or learnable_sink is not None) else None, True)] * self.q_stage

            if const_expr(self.enable_flashmask):
                total_block_count = flashmask_info.valid_block_count[batch_idx, head_idx // h_h_flashmask_ratio, m_block]
                has_work = total_block_count > Int32(0)
            elif const_expr(self.use_block_sparsity):
                total_block_count = get_total_block_count(blocksparse_tensors, batch_idx, head_idx, m_block)
                has_work = total_block_count > Int32(0)
            else:
                total_block_count = n_block_max - n_block_min
                has_work = const_expr(not self.is_split_kv) or total_block_count > Int32(0)

            if has_work:

                # Ignore first signal from softmax as no correction is required
                cute.arch.mbarrier_wait(
                    mbar_ptr + self.mbar_softmax_corr_full_offset + 0, softmax_corr_consumer_phase
                )

                cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_softmax_corr_empty_offset + 0)

                if const_expr(self.num_s_stages == 2):
                    cute.arch.mbarrier_wait(
                        mbar_ptr + self.mbar_softmax_corr_full_offset + 1,
                        softmax_corr_consumer_phase,
                    )

                softmax_corr_consumer_phase ^= 1

                if const_expr(not self.folded_acc):
                    tSrScale_t2r = cute.make_fragment(tSrScale_t2r_shape, Float32)
                for i in cutlass.range(total_block_count - 1, unroll=1):
                    for stage in cutlass.range_constexpr(self.num_s_stages):
                        # wait for S0 / S1

                        cute.arch.mbarrier_wait(
                            mbar_ptr + self.mbar_softmax_corr_full_offset + stage,
                            softmax_corr_consumer_phase,
                        )

                        # cute.copy(tiled_tmem_load_vec, tStScales_t2r[stage], tSrScale_t2r)
                        # cute.arch.fence_view_async_tmem_load()
                        # scale = tSrScale_t2r[0]
                        scale = sScale[corr_row + stage * self.m_block_size]
                        should_rescale = cute.arch.vote_ballot_sync(scale < 1.0) != 0
                        # should_rescale = True
                        # Don't need O_full anymore, since by the time softmax has signaled the correction
                        # warps, S_i must have been done, so O_i-1 must have been done as well.
                        # cute.arch.mbarrier_wait(mbar_ptr + self.mbar_O_full_offset + stage, o_corr_consumer_phase)
                        if const_expr(self.pp_lazy_o_wait):
                            # Wait for PV(i) (global block pp_base + i)
                            # only when O is rescaled. The MMA cannot be two PVs ahead on a
                            # slot, since PV(i+1) needs the P_full_O_rescaled arrive below.
                            if should_rescale:
                                prev_block = pp_base + i
                                cute.arch.mbarrier_wait(
                                    mbar_ptr + self.mbar_O_full_offset + (prev_block & 1),
                                    (prev_block >> 1) & 1,
                                )
                        elif const_expr(self.use_s_ping_pong):
                            # ...except with S ping-pong: QK(i+1) is issued before PV(i), so wait
                            # for PV(i) (this loop step handles block i + 1) before touching O.
                            # Waited even when no rescale is needed, to keep O_full's phase in
                            # lock-step with the MMA warp's per-PV commit.
                            cute.arch.mbarrier_wait(
                                mbar_ptr + self.mbar_O_full_offset + stage, o_corr_consumer_phase
                            )
                            o_corr_consumer_phase ^= 1
                        if should_rescale:
                            self.correction_rescale(thr_mma_pv, tOtOs[stage], tidx, scale)
                        if const_expr(self.use_s_ping_pong):
                            self.mma_barrier_arrive(
                                mbar_ptr + self.mbar_P_full_O_rescaled_offset + ((pp_base + i + 1) & 1)
                            )
                        else:
                            self.mma_barrier_arrive(mbar_ptr + self.mbar_P_full_O_rescaled_offset + stage)

                        # Hand the stage back to its softmax warpgroup. With two stages the
                        # signal is crossed (stage 0's correction frees stage 1) to keep the
                        # two warpgroups alternating; with a single stage it goes back to
                        # stage 0 itself.
                        cute.arch.mbarrier_arrive(
                            mbar_ptr
                            + self.mbar_softmax_corr_empty_offset
                            + (self.num_s_stages - 1 - stage)
                        )
                    softmax_corr_consumer_phase ^= 1
                    # o_corr_consumer_phase ^= 1
                if const_expr(self.num_s_stages == 2):
                    cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_softmax_corr_empty_offset + 1)
                # End of seqlen_corr_loop_steps

                # Even in the case of self.overlap_sO_sQ, we can write to stage 0 of sO without
                # additional sync because the MMA in the top half must have been done.
                # Similarly we can write to stage 1 of sO without additional sync.
                learnable_sink_val = [None] * self.q_stage
                if const_expr(learnable_sink is not None):
                    if const_expr(not self.pack_gqa):
                        sink_val = Float32(learnable_sink[head_idx])
                        learnable_sink_val = [sink_val] * self.q_stage
                    else:  # Each thread might have a different sink value due to different q_head
                        for stage in cutlass.range_constexpr(self.q_stage):
                            q_head_idx = (
                                self.m_tile_index(m_block, stage, corr_cta_coord_v)
                                * self.m_block_size + tidx
                            ) % self.qhead_per_kvhead + head_idx * self.qhead_per_kvhead
                            learnable_sink_val[stage] = Float32(learnable_sink[q_head_idx])
                for stage in cutlass.range_constexpr(self.q_stage):
                    cute.arch.mbarrier_wait(
                        mbar_ptr + self.mbar_softmax_corr_full_offset + stage,
                        softmax_corr_consumer_phase,
                    )
                    # cute.copy(tiled_tmem_load_vec, tStScales_t2r[stage], tSrScale_t2r)
                    # cute.arch.fence_view_async_tmem_load()
                    # scale = tSrScale_t2r[0]
                    row_sum = sScale[corr_row + stage * self.m_block_size]
                    if const_expr(mLSE is not None or learnable_sink is not None):
                        row_max = sScale[corr_row + stage * self.m_block_size + self.m_block_size * 2]
                    else:
                        row_max = None
                    cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_softmax_corr_empty_offset + stage)
                    if const_expr(learnable_sink is not None):
                        LOG2_E = math.log2(math.e)
                        sink_val = learnable_sink_val[stage]
                        if const_expr(not self.is_split_kv) or split_idx == 0:
                            if row_max == -Float32.inf:
                                # It's possible to have an empty row with splitKV.
                                row_max = sink_val * (LOG2_E / softmax_scale_log2)
                                row_sum = Float32(1.0)
                            else:
                                row_sum += utils.exp2f(
                                    sink_val * LOG2_E - row_max * softmax_scale_log2
                                )
                    acc_O_mn_row_is_zero_or_nan = row_sum == 0.0 or row_sum != row_sum
                    stats[stage] = (row_sum, row_max, acc_O_mn_row_is_zero_or_nan)
                    scale = cute.arch.rcp_approx(row_sum if not acc_O_mn_row_is_zero_or_nan else 1.0)
                    if const_expr(self.pp_lazy_o_wait):
                        last_block = pp_base + total_block_count - 1
                        cute.arch.mbarrier_wait(
                            mbar_ptr + self.mbar_O_full_offset + (last_block & 1),
                            (last_block >> 1) & 1,
                        )
                    else:
                        cute.arch.mbarrier_wait(
                            mbar_ptr + self.mbar_O_full_offset + stage, o_corr_consumer_phase
                        )
                    if const_expr(self.fm_persistent):
                        # O leaves through the sOc ring; the epilogue warp TMA-stores each
                        # chunk. Returns once the last chunk is in smem, i.e. TMEM O is free.
                        self.correction_epilogue(
                            thr_mma_pv,
                            tOtOs[stage],
                            tidx,
                            stage,
                            m_block,
                            seqlen.seqlen_q,
                            scale,
                            sOc_full_view,
                            o_chunk_mbar=mbar_ptr + self.mbar_o_chunk_offset,
                        )
                    else:
                        cute.arch.mbarrier_wait(
                            mbar_ptr + self.mbar_corr_epi_empty_offset + stage, corr_epi_producer_phase
                        )
                        self.correction_epilogue(
                            thr_mma_pv,
                            tOtOs[stage],
                            tidx,
                            stage,
                            m_block,
                            seqlen.seqlen_q,
                            scale,
                            sO[None, None, stage],
                            mO_cur,
                            gO,
                            gmem_tiled_copy_O,
                            corr_cta_coord_v,
                        )
                        cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_corr_epi_full_offset + stage)
                    # Signal for the next work tile that O buffers in tmem are already read, so
                    # mma warp can write to them
                    if const_expr(self.use_s_ping_pong):
                        # the next tile's first PV uses slot (pp_base + total_block_count) & 1
                        self.mma_barrier_arrive(
                            mbar_ptr
                            + self.mbar_P_full_O_rescaled_offset
                            + ((pp_base + total_block_count) & 1)
                        )
                    else:
                        self.mma_barrier_arrive(mbar_ptr + self.mbar_P_full_O_rescaled_offset + stage)

                if const_expr(self.use_s_ping_pong):
                    pp_base += total_block_count
                o_corr_consumer_phase ^= 1
                softmax_corr_consumer_phase ^= 1
                corr_epi_producer_phase ^= 1
            else:
                # WARNING: we need some code before the const_expr, see https://github.com/NVIDIA/cutlass/issues/2781
                gmem_tiled_copy_O_for_empty_tile = None
                if const_expr(self.use_block_sparsity):
                    (
                        softmax_corr_consumer_phase,
                        o_corr_consumer_phase,
                        corr_epi_producer_phase,
                    ) = handle_block_sparse_empty_tile_correction_sm100(
                        tidx,
                        self.q_stage,
                        self.m_block_size,
                        self.qhead_per_kvhead,
                        self.pack_gqa,
                        self.is_split_kv,
                        learnable_sink,
                        mLSE,
                        seqlen,
                        m_block,
                        head_idx,
                        batch_idx,
                        split_idx,
                        sScale,
                        stats,
                        self.correction_epilogue,
                        thr_mma_pv,
                        tOtOs,
                        sO,
                        mbar_ptr,
                        self.mbar_softmax_corr_full_offset,
                        self.mbar_softmax_corr_empty_offset,
                        self.mbar_P_full_O_rescaled_offset,
                        self.mbar_P_full_2_offset,
                        self.mbar_corr_epi_full_offset,
                        self.mbar_corr_epi_empty_offset,
                        softmax_corr_consumer_phase,
                        o_corr_consumer_phase,
                        corr_epi_producer_phase,
                        softmax_scale_log2,
                        mO_cur,
                        gO,
                        gmem_tiled_copy_O_for_empty_tile,
                    )

            if const_expr(mLSE is not None):
                if const_expr(self.is_split_kv):
                    mLSE_cur = mLSE[None, head_idx, batch_idx, split_idx]
                else:
                    mLSE_cur = mLSE[None, head_idx, batch_idx]
                for stage in cutlass.range_constexpr(self.q_stage):
                    gLSE = cute.local_tile(
                        mLSE_cur,
                        (self.m_block_size,),
                        (self.m_tile_index(m_block, stage, corr_cta_coord_v),),
                    )
                    row_sum, row_max, acc_O_mn_row_is_zero_or_nan = stats[stage]
                    LN2 = math.log(2.0)
                    lse = (
                        (row_max * softmax_scale_log2 + utils.log2f(row_sum)) * LN2
                        if not acc_O_mn_row_is_zero_or_nan
                        else -Float32.inf
                    )
                    seqlen_q = (
                        seqlen.seqlen_q
                        if const_expr(not self.pack_gqa)
                        else seqlen.seqlen_q * self.qhead_per_kvhead
                    )
                    if corr_row < seqlen_q - self.m_tile_index(
                        m_block, stage, corr_cta_coord_v
                    ) * self.m_block_size:
                        # This actually just works with PackGQA too
                        # Folded accumulator: the two threads sharing this row write the same
                        # lse to the same address.
                        gLSE[corr_row] = lse

            # Advance to next tile
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()
        # End of persistent scheduler loop

    @cute.jit
    def correction_rescale(
        self,
        thr_mma: cute.core.ThrMma,
        tOtO: cute.Tensor,
        tidx: Int32,
        scale: Float32,
    ):
        """Rescale intermediate attention results based on softmax normalization factor.

        This method performs a crucial correction step in the attention computation pipeline.
        When processing attention in blocks, the softmax normalization factors may change
        as new blocks are processed. This method rescales previously computed partial
        output values to account for updated normalization factors.

        The implementation uses efficient tensor memory operations to:
        1. Load existing partial attention output from tensor memory
        2. Apply the scaling factor to all elements
        3. Store the rescaled results back to tensor memory
        """
        # V=0 slice: tOtO is the per-CTA accumulator, so the coordinate tensor that shapes
        # the tmem copies must be the per-CTA (unshifted) one. See correction_epilogue.
        tOcO = thr_mma.get_slice(0).partition_C(cute.make_identity_tensor(self.mma_tiler_pv[:2]))
        corr_tile_size = 16  # tuneable parameter
        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(corr_tile_size)),
            self.pv_acc_dtype,
        )
        tmem_store_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(corr_tile_size)),
            self.pv_acc_dtype,
        )
        # logical_divide (like correction_epilogue), NOT composition + iterator arithmetic.
        # With the folded accumulator (m_block_size == 64) both tensors are first re-indexed
        # by their PHYSICAL (lane, column) shape (see folded_o_phys_view): the fold puts the
        # upper dv half of every N-tile in lanes 64..127, so a copy built over an
        # m_block_size-tall tile only spans 64 lanes and make_tmem_copy duplicates warps 2/3
        # onto warps 0/1's coordinates while the hardware keeps them on lanes 64..127. On the
        # physical view mode 0 is 128 lanes, so the copy is a real one-lane-per-thread copy
        # and chunk `i` walks tmem_cols(head_dim_v) / corr_tile_size physical column chunks.
        if const_expr(self.folded_acc):
            tOtO = self.folded_o_phys_view(tOtO)
            tOcO = self.folded_o_phys_view(tOcO)
        chunk_layout = cute.make_layout((self.folded_acc_lanes(), corr_tile_size))
        tOtO_i = cute.logical_divide(tOtO, chunk_layout)
        tOcO_i = cute.logical_divide(tOcO, chunk_layout)
        thr_tmem_load = tcgen05.make_tmem_copy(
            tmem_load_atom, tOtO_i[(None, None), 0]
        ).get_slice(tidx)
        thr_tmem_store = tcgen05.make_tmem_copy(
            tmem_store_atom, tOtO_i[(None, None), 0]
        ).get_slice(tidx)
        tOtO_t2r = thr_tmem_load.partition_S(tOtO_i[(None, None), None])
        tOtO_r2t = thr_tmem_store.partition_D(tOtO_i[(None, None), None])
        tOcO_t2r = thr_tmem_load.partition_D(tOcO_i[(None, None), None])

        frg_count = self.tmem_cols(self.head_dim_v_padded) // corr_tile_size
        for i in cutlass.range_constexpr(frg_count):
            tOrO_frg = cute.make_fragment(tOcO_t2r[None, 0, 0, i].shape, self.pv_acc_dtype)
            cute.copy(thr_tmem_load, tOtO_t2r[None, 0, 0, i], tOrO_frg)
            # range_constexpr, not range(..., unroll_full=True): the latter emits a real
            # loop with a dynamic induction variable, and a dynamically indexed store into
            # tOrO_frg keeps its alloca out of registers -- the O chunk then lives in local
            # memory, so every rescale round-trips dv_padded floats per thread through
            # L1TEX. correction_epilogue below already uses range_constexpr for the same
            # reason.
            for j in cutlass.range_constexpr(0, cute.size(tOrO_frg), 2):
                tOrO_frg[j], tOrO_frg[j + 1] = utils.mul_packed_f32x2(
                    (tOrO_frg[j], tOrO_frg[j + 1]),
                    (scale, scale),
                )
            cute.copy(thr_tmem_store, tOrO_frg, tOtO_r2t[None, 0, 0, i])
        cute.arch.fence_view_async_tmem_store()

    @cute.jit
    def correction_epilogue(
        self,
        thr_mma: cute.core.ThrMma,
        tOtO: cute.Tensor,
        tidx: Int32,
        stage: Int32,
        m_block: Int32,
        seqlen_q: Int32,
        scale: Float32,
        sO: cute.Tensor,
        mO_cur: Optional[cute.Tensor] = None,
        gO: Optional[cute.Tensor] = None,
        gmem_tiled_copy_O: Optional[cute.TiledCopy] = None,
        mma_tile_coord_v: Int32 = 0,
        o_chunk_mbar: Optional[cute.Pointer] = None,
    ):
        """Apply final scaling and transformation to attention output before writing to global memory.

        This correction_epilogue function handles the final processing step for attention output values.
        It applies a scaling factor to the accumulated attention results and prepares the
        data for efficient transfer back to global memory.

        The method performs:
        1. Loading of accumulated attention results from tensor memory
        2. Application of the final output scaling factor
        3. Type conversion if necessary (typically from higher precision accumulator to output precision)
        4. Reorganization of data for optimal memory access patterns
        5. Preparation for efficient TMA store operations

        :param thr_mma: Thread MMA operation for the computation
        :type thr_mma: cute.core.ThrMma
        :param tOtO: Tensor containing accumulated attention output
        :type tOtO: cute.Tensor
        :param scale: Final scaling factor to apply to the output
        :type scale: Float32
        :param sO: Shared memory tensor for the final output
        :type sO: cute.Tensor
        """

        corr_tile_size = 32 * 8 // self.o_dtype.width
        # `sO` and `tOtO` are PER CTA (m_block_size rows), but `thr_mma` is sliced at this
        # CTA's V coord. `partition_C` applies that coord to the M mode, so with
        # cta_group=2 the peer CTA would be handed a view offset by another m_block_size
        # rows: it writes O *past the end of its own sO buffer* and the epilogue then
        # stores whatever stale smem is left in rows 0..m_block_size-1 (observed as
        # garbage output rows 128..255 while the leader's rows 0..127 were exact).
        # The accumulator -> smem map is identical in both CTAs, so partition with V=0.
        # (`make_fragment_C` / `partition_shape_C` ignore the slice coord, which is why
        # tOtO is already per-CTA; only partition_A/B/C consume it.)
        thr_mma_c = thr_mma.get_slice(0)
        tOsO = thr_mma_c.partition_C(sO)
        tOcO = thr_mma_c.partition_C(cute.make_identity_tensor(self.mma_tiler_pv[:2]))

        # Folded accumulator: re-index all three views by the accumulator's PHYSICAL
        # (lane, column) shape so the tmem copy below is a real one-lane-per-thread copy and
        # the sO / identity views carry the logical dv column that belongs to each lane. See
        # folded_o_phys_view for why an m_block_size-tall tile silently duplicates warps 2/3.
        if const_expr(self.folded_acc):
            tOtO = self.folded_o_phys_view(tOtO)
            tOsO = self.folded_o_phys_view(tOsO)
            tOcO = self.folded_o_phys_view(tOcO)

        chunk_layout = cute.make_layout((self.folded_acc_lanes(), corr_tile_size))
        tOtO_i = cute.logical_divide(tOtO, chunk_layout)
        tOcO_i = cute.logical_divide(tOcO, chunk_layout)
        tOsO_i = cute.logical_divide(tOsO, chunk_layout)

        if const_expr(not self.folded_acc):
            # The epilogue subtile is PER CTA: get_tmem_load_op derives
            # num_dp = epi_tile_m / tmem_warp_shape_m and requires it to be 16 or 32, with
            # tmem_warp_shape_m == 4 for every cta_tile M except the 2-CTA m=64 case. So the
            # 2-CTA config (per-CTA M = 128, pair M = 256) wants epi_tile[0] = 128 -> num_dp 32,
            # exactly like 1-CTA. Passing the pair-wide 256 gives num_dp 64 and raises
            # "Cta tile and 2sm config does not generate correct num dp."
            epi_subtile = (self.epi_tile[0], corr_tile_size)
            tmem_copy_atom = sm100_utils_basic.get_tmem_load_op(
                self.mma_tiler_pv,
                self.o_layout,
                self.o_dtype,
                self.pv_acc_dtype,
                epi_subtile,
                use_2cta_instrs=self.use_2cta_instrs,
            )
        else:
            # Folded accumulator, addressed through folded_o_phys_view: 128 lanes x
            # tmem_cols(head_dim_v) columns. get_tmem_load_op's epi-tile ops (16DP / 256bit /
            # x2 for a 64-row epi tile) assume one row per lane and walk rows with stride 2
            # (measured: tidx 0/1/2 -> row 0 cols 0/2/4, tidx 8 -> row 2), which leaves every
            # odd row of sO unwritten. The plain Ld32x32b atom gives one lane per thread and
            # `corr_tile_size` contiguous columns each -- the physical view's own mapping (and
            # a contiguous sO run, so autovec_copy below vectorizes it).
            tmem_copy_atom = cute.make_copy_atom(
                tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(corr_tile_size)),
                self.pv_acc_dtype,
            )
        tiled_tmem_load = tcgen05.make_tmem_copy(tmem_copy_atom, tOtO_i[(None, None), 0]).get_slice(
            tidx
        )
        thr_tmem_load = tiled_tmem_load.get_slice(tidx)
        if const_expr(not self.folded_acc):
            smem_copy_atom = sm100_utils_basic.get_smem_store_op(
                self.o_layout, self.o_dtype, self.pv_acc_dtype, tiled_tmem_load
            )
            tiled_smem_store = cute.make_tiled_copy_D(smem_copy_atom, tiled_tmem_load)
        else:
            # No smem store ATOM at all on the folded path. STSM (stmatrix.m8n8.x4) needs each
            # thread to own 8 contiguous bf16 (16B) of sO, but the folded epi tile hands each
            # thread 2-element runs (dst memref (((2,2,2),1)):(((1,512,8),0))), so the atom the
            # library picks is rejected ("dst ptr alignment (32 bits) does not meet requirement
            # (128 bits)"), and building a CopyUniversalOp tiled copy for it segfaults the DSL
            # while constructing the tiled copy. The register fragment and its sO partition
            # have identical shapes, so `autovec_copy` moves them directly and lets the
            # compiler pick the vector width.
            tiled_smem_store = None

        tOtO_t2r = thr_tmem_load.partition_S(tOtO_i[(None, None), None])
        tOsO_s2r = thr_tmem_load.partition_D(tOsO_i[(None, None), None])
        tOcO_t2r = thr_tmem_load.partition_D(tOcO_i[(None, None), None])
        epi_loop = const_expr(self.fm_corr_epi_loop and o_chunk_mbar is not None)
        if const_expr(epi_loop):
            # One runtime iteration per sOc chunk; same per-tile work as the unrolled
            # loop below, but the code is one chunk long (see fm_corr_epi_loop).
            tiles_per_chunk = const_expr(self.o_chunk_cols // corr_tile_size)
            n_o_chunks = const_expr(self.tmem_cols(self.head_dim_v_padded) // self.o_chunk_cols)
            for c in cutlass.range(n_o_chunks, unroll=1):
                o_chunk_slot = c % self.o_chunk_stages
                cute.arch.mbarrier_wait(
                    o_chunk_mbar + self.o_chunk_stages + o_chunk_slot,
                    ((c // self.o_chunk_stages) & 1) ^ 1,
                )
                for k in cutlass.range_constexpr(tiles_per_chunk):
                    i = c * tiles_per_chunk + k
                    tOrO_frg = cute.make_fragment(tOcO_t2r[None, 0, 0, 0].shape, self.pv_acc_dtype)
                    cute.copy(tiled_tmem_load, tOtO_t2r[None, 0, 0, i], tOrO_frg)
                    for j in cutlass.range_constexpr(0, cute.size(tOrO_frg), 2):
                        tOrO_frg[j], tOrO_frg[j + 1] = utils.mul_packed_f32x2(
                            (tOrO_frg[j], tOrO_frg[j + 1]),
                            (scale, scale),
                        )
                    tOrO_frg_cvt = cute.make_fragment(tOrO_frg.shape, self.o_dtype)
                    tOrO_frg_cvt.store(tOrO_frg.load().to(self.o_dtype))
                    cute.copy(tiled_smem_store, tOrO_frg_cvt, tOsO_s2r[None, 0, 0, i])
                # Chunk complete in smem: make it visible to the TMA store, hand it over.
                cute.arch.fence_view_async_shared()
                cute.arch.mbarrier_arrive(o_chunk_mbar + o_chunk_slot)
        # Unrolled form (zero iterations when the fm_corr_epi_loop loop above ran).
        for i in cutlass.range_constexpr(
            0 if epi_loop else self.tmem_cols(self.head_dim_v_padded) // corr_tile_size
        ):
            tOtO_t2r_i = tOtO_t2r[None, 0, 0, i]
            tOsO_r2s_i = tOsO_s2r[None, 0, 0, i]
            if const_expr(o_chunk_mbar is not None):
                # sO is the sOc ring seen as (m, dv): wait for the slot of chunk c to be free
                # (the epilogue warp's TMA store finished reading its previous chunk).
                o_chunk_col = i * corr_tile_size
                o_chunk_idx = o_chunk_col // self.o_chunk_cols
                o_chunk_slot = o_chunk_idx % self.o_chunk_stages
                if const_expr(o_chunk_col % self.o_chunk_cols == 0):
                    cute.arch.mbarrier_wait(
                        o_chunk_mbar + self.o_chunk_stages + o_chunk_slot,
                        ((o_chunk_idx // self.o_chunk_stages) & 1) ^ 1,
                    )
            tOrO_frg = cute.make_fragment(tOcO_t2r[None, 0, 0, i].shape, self.pv_acc_dtype)
            cute.copy(tiled_tmem_load, tOtO_t2r_i, tOrO_frg)
            for j in cutlass.range_constexpr(0, cute.size(tOrO_frg), 2):
                tOrO_frg[j], tOrO_frg[j + 1] = utils.mul_packed_f32x2(
                    (tOrO_frg[j], tOrO_frg[j + 1]),
                    (scale, scale),
                )
            tOrO_frg_cvt = cute.make_fragment(tOrO_frg.shape, self.o_dtype)
            tOrO_frg_cvt.store(tOrO_frg.load().to(self.o_dtype))
            if const_expr(not self.folded_acc):
                cute.copy(tiled_smem_store, tOrO_frg_cvt, tOsO_r2s_i)
            else:
                cute.autovec_copy(tOrO_frg_cvt, tOsO_r2s_i)
            if const_expr(o_chunk_mbar is not None):
                if const_expr((o_chunk_col + corr_tile_size) % self.o_chunk_cols == 0):
                    # Chunk complete in smem: make it visible to the TMA store, hand it over.
                    cute.arch.fence_view_async_shared()
                    cute.arch.mbarrier_arrive(o_chunk_mbar + o_chunk_slot)
        # fence view async shared
        cute.arch.fence_view_async_shared()

    @cute.jit
    def epilogue_s2g(
        self,
        mO: cute.Tensor,
        sO: cute.Tensor,
        gmem_tiled_copy_O: cute.TiledCopy,
        tma_atom_O: Optional[cute.CopyAtom],
        mbar_ptr: cute.Pointer,
        block_info: BlockInfo,
        num_splits: int,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        num_heads: Int32,
        flashmask_info: FlashMaskInfo,
        mO_gmem: Optional[cute.Tensor] = None,
        sOc: Optional[cute.Tensor] = None,
        tma_atom_Oc: Optional[cute.CopyAtom] = None,
        mOc: Optional[cute.Tensor] = None,
    ):
        epi_consumer_phase = Int32(0)
        # This CTA's half of the 2-CTA UMMA's M range (0 for cta_group=1).
        epi_cta_coord_v = self.cta_coord_v()
        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        if const_expr(self.enable_flashmask):
            fm_num_heads = flashmask_info.startend_row_indices.shape[1]
            h_h_flashmask_ratio = num_heads // fm_num_heads
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, m_block, split_idx, num_splits)

            if const_expr(self.enable_flashmask):
                valid_block_count = flashmask_info.valid_block_count[batch_idx, head_idx // h_h_flashmask_ratio, m_block]
            if (const_expr(not self.is_split_kv) or n_block_min < n_block_max) and (const_expr(not self.enable_flashmask) or valid_block_count) > 0:
                if const_expr(self.is_split_kv):
                    mO_cur = seqlen.offset_batch_Q(mO, batch_idx, dim=3)[None, None, head_idx, split_idx]
                else:
                    mO_cur = seqlen.offset_batch_Q(mO, batch_idx, dim=3)[None, None, head_idx]
                gO = cute.local_tile(mO_cur, (self.m_block_size, self.head_dim_v_padded), (None, 0))
                if const_expr(self.fm_persistent):
                    # Drain the sOc ring: chunk c sits in slot c % stages once the correction
                    # warps arrive on its full barrier; free the slot once the TMA has read it.
                    mOc_cur = seqlen.offset_batch_Q(mOc, batch_idx, dim=3)[None, None, head_idx]
                    gOc = cute.local_tile(
                        mOc_cur,
                        (self.m_block_size, self.o_chunk_cols),
                        (self.m_tile_index(m_block, 0, epi_cta_coord_v), None),
                    )
                    store_Oc, _, _ = copy_utils.tma_get_copy_fn(
                        tma_atom_Oc, 0, cute.make_layout(1), sOc, gOc
                    )
                    o_chunk_mbar = mbar_ptr + self.mbar_o_chunk_offset
                    n_chunks = self.head_dim_v_padded // self.o_chunk_cols
                    n_slots = self.o_chunk_stages
                    # Keep up to n_slots stores in flight: after issuing chunk c, wait until
                    # chunk c - (n_slots - 1) has been read and free its slot (the one the
                    # correction warps need next); drain the rest after the last chunk.
                    for c in cutlass.range_constexpr(n_chunks):
                        cute.arch.mbarrier_wait(o_chunk_mbar + c % n_slots, (c // n_slots) & 1)
                        store_Oc(src_idx=c % n_slots, dst_idx=c)
                        cute.arch.cp_async_bulk_commit_group()
                        if const_expr(c >= n_slots - 1):
                            cute.arch.cp_async_bulk_wait_group(n_slots - 1, read=True)
                            cute.arch.mbarrier_arrive(
                                o_chunk_mbar + n_slots + (c - (n_slots - 1)) % n_slots
                            )
                    for d in cutlass.range_constexpr(n_slots - 1):
                        cute.arch.cp_async_bulk_wait_group(n_slots - 2 - d, read=True)
                        cute.arch.mbarrier_arrive(
                            o_chunk_mbar + n_slots + (n_chunks - (n_slots - 1) + d) % n_slots
                        )
                elif const_expr(self.use_tma_O):
                    store_O, _, _ = copy_utils.tma_get_copy_fn(
                        tma_atom_O, 0, cute.make_layout(1), sO, gO
                    )
                    for stage in cutlass.range_constexpr(self.q_stage):
                        # wait from corr, issue tma store on smem
                        # 1. wait for O0 / O1 final
                        cute.arch.mbarrier_wait(
                            mbar_ptr + self.mbar_corr_epi_full_offset + stage, epi_consumer_phase
                        )
                        # 2. copy O0 / O1 to gmem
                        store_O(
                            src_idx=stage,
                            dst_idx=self.m_tile_index(m_block, stage, epi_cta_coord_v),
                        )
                        cute.arch.cp_async_bulk_commit_group()
                    for stage in cutlass.range_constexpr(self.q_stage):
                        # Ensure O0 / O1 buffer is ready to be released
                        cute.arch.cp_async_bulk_wait_group(1 - stage, read=True)
                        cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_corr_epi_empty_offset + stage)
                else:
                    tidx = cute.arch.thread_idx()[0] % (
                        cute.arch.WARP_SIZE * len(self.epilogue_warp_ids)
                    )
                    gmem_thr_copy_O = gmem_tiled_copy_O.get_slice(tidx)
                    tOsO = gmem_thr_copy_O.partition_S(sO)
                    cO = cute.make_identity_tensor((self.m_block_size, self.head_dim_v_padded))
                    tOgO = gmem_thr_copy_O.partition_D(gO)
                    tOcO = gmem_thr_copy_O.partition_S(cO)
                    t0OcO = gmem_tiled_copy_O.get_slice(0).partition_S(cO)
                    tOpO = utils.predicate_k(tOcO, limit=mO.shape[1])
                    # TODO: the packgqa case isn't correct rn (sometimes IMA), disabling it
                    assert not self.pack_gqa
                    pack_gqa = PackGQA(
                        self.m_block_size,
                        self.head_dim_v_padded,
                        self.check_hdim_v_oob,
                        self.qhead_per_kvhead,
                    )
                    for stage in cutlass.range_constexpr(self.q_stage):
                        # wait from corr, issue tma store on smem
                        # 1. wait for O0 / O1 final
                        cute.arch.mbarrier_wait(
                            mbar_ptr + self.mbar_corr_epi_full_offset + stage, epi_consumer_phase
                        )
                        # 2. copy O0 / O1 to gmem
                        # load acc O from smem to rmem for wider vectorization
                        tOrO = cute.make_fragment_like(tOsO[None, None, None, 0], self.o_dtype)
                        cute.autovec_copy(tOsO[None, None, None, stage], tOrO)
                        # copy acc O from rmem to gmem
                        if const_expr(not self.pack_gqa):
                            for rest_m in cutlass.range_constexpr(cute.size(tOrO.shape[1])):
                                if (
                                    t0OcO[0, rest_m, 0][0]
                                    < seqlen.seqlen_q
                                    - self.m_tile_index(m_block, stage, epi_cta_coord_v)
                                    * self.m_block_size
                                    - tOcO[0][0]
                                ):
                                    cute.copy(
                                        gmem_tiled_copy_O,
                                        tOrO[None, rest_m, None],
                                        tOgO[
                                            None,
                                            rest_m,
                                            None,
                                            self.m_tile_index(
                                                m_block, stage, epi_cta_coord_v
                                            ),
                                        ],
                                        pred=tOpO[None, rest_m, None]
                                        if const_expr(self.check_hdim_v_oob)
                                        else None,
                                    )
                        else:
                            pack_gqa.store_O(
                                mO_cur,
                                tOrO,
                                gmem_tiled_copy_O,
                                tidx,
                                self.m_tile_index(m_block, stage, epi_cta_coord_v),
                                seqlen.seqlen_q,
                            )
                        cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_corr_epi_empty_offset + stage)

                epi_consumer_phase ^= 1
            else:
                if const_expr(self.fm_persistent):
                    # Fully masked query block, persistent path: sO aliases sQ, which may already
                    # hold the next tile's Q, so write the zero tile with plain stores instead.
                    mO_direct_cur = seqlen.offset_batch_Q(mO_gmem, batch_idx, dim=3)[None, None, head_idx]
                    gO_zero = cute.local_tile(
                        mO_direct_cur,
                        (self.m_block_size, self.head_dim_v_padded),
                        (self.m_tile_index(m_block, 0, epi_cta_coord_v), 0),
                    )
                    zero_copy = copy_utils.tiled_copy_2d(
                        self.o_dtype,
                        threads_per_row=cute.arch.WARP_SIZE,
                        num_threads=cute.arch.WARP_SIZE,
                        num_copy_elems=128 // self.o_dtype.width,
                    )
                    zero_thr = zero_copy.get_slice(cute.arch.lane_idx())
                    tZgZ = zero_thr.partition_D(gO_zero)
                    tZcZ = zero_thr.partition_D(
                        cute.make_identity_tensor((self.m_block_size, self.head_dim_v_padded))
                    )
                    tZrZ = cute.make_fragment_like(tZgZ[None, 0, 0])
                    tZrZ.fill(0.0)
                    zero_row_limit = (
                        seqlen.seqlen_q - self.m_tile_index(m_block, 0, epi_cta_coord_v) * self.m_block_size
                    )
                    for zm in cutlass.range(cute.size(tZgZ.shape[1]), unroll=4):
                        for zn in cutlass.range_constexpr(cute.size(tZgZ.shape[2])):
                            if tZcZ[0, zm, zn][0] < zero_row_limit:
                                cute.copy(zero_copy, tZrZ, tZgZ[None, zm, zn])
                elif const_expr(self.zero_empty_o_tiles):
                    # Fully masked query block: no MMA / correction work, and (non-persistent,
                    # one tile per CTA) neither Q nor O ever touched sO / sQ. Zero sO in place
                    # and TMA-store it like a regular O tile; TMA clips rows past seqlen_q.
                    # (mO here is the TMA coordinate tensor, so it cannot be written directly.)
                    mO_cur = seqlen.offset_batch_Q(mO, batch_idx, dim=3)[None, None, head_idx]
                    gO = cute.local_tile(mO_cur, (self.m_block_size, self.head_dim_v_padded), (None, 0))
                    store_O, _, _ = copy_utils.tma_get_copy_fn(
                        tma_atom_O, 0, cute.make_layout(1), sO, gO
                    )
                    num_epi_threads = cute.arch.WARP_SIZE * len(self.epilogue_warp_ids)
                    epi_tidx = cute.arch.thread_idx()[0] % num_epi_threads
                    # The whole stage-0 sO buffer, as raw 32-bit words (all zeros, so the
                    # swizzle of the real layout does not matter).
                    n_words = const_expr(self.m_block_size * self.head_dim_v_padded * self.o_dtype.width // 32)
                    sO_words = cute.make_tensor(
                        cute.make_ptr(Int32, sO.iterator.toint(), cute.AddressSpace.smem, assumed_align=16),
                        cute.make_layout(n_words),
                    )
                    for i in cutlass.range(n_words // num_epi_threads, unroll=8):
                        sO_words[i * num_epi_threads + epi_tidx] = Int32(0)
                    cute.arch.fence_view_async_shared()
                    cute.arch.sync_warp()
                    # Issued exactly like the regular O store above.
                    store_O(src_idx=0, dst_idx=self.m_tile_index(m_block, 0, epi_cta_coord_v))
                    cute.arch.cp_async_bulk_commit_group()
                    cute.arch.cp_async_bulk_wait_group(0, read=True)

            # Advance to next tile
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    def load_Q(
        self,
        load_Q_fn: Callable,
        mbar_full_ptr: cute.Pointer,
        mbar_empty_ptr: cute.Pointer,
        block: Int32,
        stage: int,
        phase: Int32,
    ):
        cute.arch.mbarrier_wait(mbar_empty_ptr + stage, phase)
        self.tma_expect_tx(mbar_full_ptr + stage, self.tma_copy_bytes["Q"])
        load_Q_fn(src_idx=block, dst_idx=stage, tma_bar_ptr=mbar_full_ptr + stage)

    @cute.jit
    def load_KV(
        self,
        tma_atom: Optional[cute.CopyAtom],
        tXgX: Optional[cute.Tensor],
        tXsX: Optional[cute.Tensor],
        sX: cute.Tensor,
        mbar_full_ptr: cute.Pointer,
        mbar_empty_ptr: cute.Pointer,
        block: Int32,
        producer_state: cutlass.pipeline.PipelineState,
        K_or_V: Literal["K", "V"],
        # Retained (always None) so the block-sparse SM100 load path, which calls this
        # with page_idx=None, keeps working.
        page_idx: Optional[Int32] = None,
    ):
        assert K_or_V in ("K", "V")
        assert page_idx is None
        stage, phase = producer_state.index, producer_state.phase
        cute.arch.mbarrier_wait(mbar_empty_ptr + stage, phase)
        if const_expr(K_or_V == "K" and self.uneven_kv_smem):
            # Before this round, the smem location was occupied by V, which is smaller than
            # K. So we need to wait for the stage after that (stage 1) to be empty as well.
            if stage == 0:
                cute.arch.mbarrier_wait(mbar_empty_ptr + 1, phase)

        assert (
            tXgX is not None and
            tXsX is not None and
            tma_atom is not None
        )
        self.tma_expect_tx(mbar_full_ptr + stage, self.tma_copy_bytes[K_or_V])
        tXsX_cur = tXsX[None, stage]
        if const_expr(self.uneven_kv_smem):
            # Since this is the producer_state, the phase starts at 1, so we have to invert it
            tXsX_cur = self.offset_kv_smem(tXsX_cur, stage, phase ^ 1)
        # Currently we assume that page_size == n_block_size so we index into tXgX with block = 0
        tXgX_cur = tXgX[None, block]
        cute.copy(tma_atom, tXgX_cur, tXsX_cur, tma_bar_ptr=mbar_full_ptr + stage)

    @cute.jit
    def offset_kv_smem(self, sX: cute.Tensor, stage: Int32, phase: Int32):
        if const_expr(self.uneven_kv_smem):
            # smem layout is [smem_large, smem_small, smem_large], and the current stride is
            # (smem_large + smem_small) // 2. So for stage == 1, move right by offset if
            # phase == 0, or left by offset if phase == 1.
            offset = 0 if stage != 1 else self.uneven_kv_smem_offset * (1 - 2 * phase)
            return cute.make_tensor(sX.iterator + offset, sX.layout)
        else:
            return sX

    def make_and_init_load_kv_pipeline(self, load_kv_mbar_ptr):
        load_kv_consumer_group = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread, len([self.mma_warp_id])
        )
        load_kv_producer_group = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread, len(self.load_warp_ids)
        )
        return cutlass.pipeline.PipelineTmaUmma.create(
            barrier_storage=load_kv_mbar_ptr,
            num_stages=self.kv_stage,
            producer_group=load_kv_producer_group,
            consumer_group=load_kv_consumer_group,
            tx_count=self.tma_copy_bytes["K"],
            # With cta_group=2 the pipeline needs the cluster layout so that (a) its
            # init does a cluster-wide sync and (b) consumer_release (issued by the
            # leader CTA's MMA warp) commits with a multicast mask that also frees the
            # peer CTA's K/V stage.
            cta_layout_vmnk=(
                # Rebuilt here (not reused from __call__) because the host-traced
                # cluster_layout_vmnk value would cross the kernel region boundary.
                cute.make_layout(((self.cta_group_size,), 1, 1, 1))
                if self.cta_group_size > 1
                else None
            ),
        )

    # @cute.jit
    # def warp_scheduler_barrier_init(self):
    #     warp_group_idx = utils.canonical_warp_group_idx(sync=False)
    #     if warp_group_idx == 0:
    #         cute.arch.barrier_arrive(
    #             barrier_id=int(NamedBarrierFwd.WarpSchedulerWG1), number_of_threads=2 * 128,
    #         )

    # def warp_scheduler_barrier_sync(self):
    #     cute.arch.barrier(
    #         barrier_id=int(NamedBarrierFwd.WarpSchedulerWG1) + utils.canonical_warp_group_idx(sync=False),
    #         number_of_threads=2 * 128
    #     )

    # def warp_scheduler_barrier_arrive(self):
    #     cur_wg = utils.canonical_warp_group_idx(sync=False)
    #     next_wg = 1 - cur_wg
    #     cute.arch.barrier_arrive(
    #         barrier_id=int(NamedBarrierFwd.WarpSchedulerWG1) + next_wg, number_of_threads=2 * 128,
    #     )
