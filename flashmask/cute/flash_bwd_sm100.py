# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025, Ted Zadouri, Markus Hoehnerbach, Jay Shah, Tri Dao.
# Copyright (c) 2026 The FlashMask Authors. Modifications for FlashMask.

import math
from typing import Callable, Optional
from functools import partial

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr
from cutlass.utils import LayoutEnum
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils_basic
from cutlass.pipeline import PipelineAsync, PipelineConsumer

from . import utils
from . import layout_utils
from .cute_dsl_utils import assume_tensor_aligned
from . import copy_utils
from .barrier import wait_flag_eq
from . import pipeline
from . import blackwell_helpers as sm100_utils
from .blackwell_helpers import gemm_w_idx, gemm_ptx_w_idx  # noqa
from .blackwell_helpers import SM100_SMEM_CAPACITY_BYTES
from .mask import AttentionMask
from .seqlen_info import SeqlenInfoQK
from .block_info import BlockInfo
from .tile_scheduler import (
    TileSchedulerArguments,
    SingleTileScheduler,
    SingleTileLPTBwdScheduler,  # noqa
    StaticPersistentClusterTileScheduler,
    ParamsBase,
)

from . import barrier
from .named_barrier import NamedBarrierBwdSm100
from .flashmask_utils import FlashMaskInfo


@cute.jit
def _overlap_gate_bwd(
    n_block: Int32,
    tidx: Int32,
    seqlen_k: Int32,
    batch_idx: Int32,
    work_done: cute.Pointer,
    spin_timeout_ns: cutlass.Int64,
    local_kv_seqlen: Int32,
    comm_rpb: cutlass.Constexpr[int],
    cta_group_size: cutlass.Constexpr[int],
    tile_n: cutlass.Constexpr[int],
):
    """Wait for every split-AG work item holding a row of this cluster-wide KV tile.

    Works restart at each local_kv_seqlen-row chunk, whose last work may be short, so a tile can
    span several. Only ``tidx == 0`` spins, but every lane of the load warp must call this so
    each lane, including the TMA issuer, runs the proxy fence.
    """
    if tidx == 0:
        work_per_chunk = (local_kv_seqlen + comm_rpb - 1) // comm_rpb
        first_row = n_block // cta_group_size * cta_group_size * tile_n
        last_row = cutlass.min(first_row + cta_group_size * tile_n, seqlen_k) - 1
        base = batch_idx * (seqlen_k // local_kv_seqlen) * work_per_chunk + 1
        first = first_row // local_kv_seqlen * work_per_chunk + first_row % local_kv_seqlen // comm_rpb
        last = last_row // local_kv_seqlen * work_per_chunk + last_row % local_kv_seqlen // comm_rpb
        # dynamic non-canonical length can produce tiles which cannot be transfered by only one comm work. 
        for work in cutlass.range(first, last + 1, unroll=1):
            wait_flag_eq(work_done, Int32(base + work), Int32(1), spin_timeout_ns)
    # Share lane 0's acquire with the whole warp.
    cute.arch.sync_warp()
    # Order the acquire before this lane's TMA (async-proxy) reads.
    cute.arch.fence_proxy("async.global")


class FlashAttentionBackwardSm100:
    arch = 100

    def __init__(
        self,
        head_dim: int,
        head_dim_v: Optional[int] = None,
        is_causal: bool = False,
        is_local: bool = False,
        qhead_per_kvhead: cutlass.Constexpr[int] = 1,
        tile_m: int = 128,
        tile_n: int = 128,
        is_persistent: bool = False,
        deterministic: bool = False,
        cluster_size: int = 1,
        use_2cta_instrs: bool = False,
    ):
        # padding head_dim to a multiple of 64 to match head_dim_rounded in interface
        hdim_multiple_of = 64
        self.tile_hdim = int(math.ceil(head_dim / hdim_multiple_of) * hdim_multiple_of)
        head_dim_v = head_dim_v if head_dim_v is not None else head_dim
        self.same_hdim_kv = head_dim == head_dim_v
        self.tile_hdimv = int(math.ceil(head_dim_v / hdim_multiple_of) * hdim_multiple_of)
        self.check_hdim_oob = head_dim != self.tile_hdim
        self.check_hdim_v_oob = head_dim_v != self.tile_hdimv

        self.tile_m = tile_m
        self.tile_n = tile_n

        self.use_2cta_instrs = bool(use_2cta_instrs and cluster_size == 2)
        self.cta_group_size = 2 if self.use_2cta_instrs else 1

        if use_2cta_instrs:
            assert (
                (self.tile_hdim <= 128 and self.tile_hdimv <= 128)
                or (self.tile_hdim == 192 and self.tile_hdimv == 128)
                or (self.tile_hdim == 256 and self.tile_hdimv == 256)
            ), (
                f"2CTA backward does not support d={self.tile_hdim}, "
                f"dv={self.tile_hdimv}; legal: both <= 128, (192, 128), (256, 256)"
            )
        else:
            assert self.tile_hdim <= 128 and self.tile_hdimv <= 128, (
                f"1CTA backward needs d <= 128 and dv <= 128, "
                f"got d={self.tile_hdim}, dv={self.tile_hdimv}"
            )

        # CTA tiler.
        hdim_for_mma = self.tile_hdim
        hdimv_for_mma = self.tile_hdimv
        self.cta_tiler = (tile_n, tile_m, hdim_for_mma)
        # S = K @ Q.T
        self.mma_tiler_kq = (self.cta_group_size * tile_n, tile_m, hdim_for_mma)
        # dP = V @ dO.T
        self.mma_tiler_vdo = (self.cta_group_size * tile_n, tile_m, hdimv_for_mma)
        # dV = P.T @ dO
        self.mma_tiler_pdo = (self.cta_group_size * tile_n, hdimv_for_mma, tile_m)
        # dK = dS.T @ Q
        self.mma_tiler_dsq = (self.cta_group_size * tile_n, hdim_for_mma, tile_m)
        # dQ = dS @ K
        # 2-CTA: reduction dim is cluster-wide (tile_n * cta_group_size).
        self.mma_tiler_dsk = (tile_m, hdim_for_mma, tile_n * self.cta_group_size)

        self.acc_dtype = Float32
        self.startend_row_indices_dtype = Int32

        assert cluster_size in (1, 2), "Only cluster_size=1 or 2 is supported"
        self.cluster_shape_mn = (cluster_size, 1)
        # Persistent is a win only for the sliding-window mask; other masks see no gain,
        # so callers leave it off. Restricted to the 2-CTA hdim>128 path.
        self.is_persistent = is_persistent
        self.is_causal = is_causal
        self.is_local = False
        self.qhead_per_kvhead = qhead_per_kvhead
        self.pack_gqa = False
        self.dKV_postprocess = self.qhead_per_kvhead > 1
        self.deterministic = deterministic

        # Speed optimizations, does not affect correctness
        self.shuffle_LSE = False
        self.shuffle_dPsum = False

        self.reduce_warp_ids = (0, 1, 2, 3)
        self.compute_warp_ids = (4, 5, 6, 7, 8, 9, 10, 11)
        self.mma_warp_id = 12
        self.load_warp_id = 13
        self.relay_warp_id = 14
        self.empty_warp_id = 15

        # 16 warps -> 512 threads
        self.threads_per_cta = cute.arch.WARP_SIZE * len(
            (
                *self.reduce_warp_ids,
                *self.compute_warp_ids,
                self.mma_warp_id,
                self.load_warp_id,
                self.relay_warp_id,
                self.empty_warp_id,
            )
        )

        # NamedBarrier
        self.compute_sync_barrier = cutlass.pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierBwdSm100.Compute),
            num_threads=len(self.compute_warp_ids) * cute.arch.WARP_SIZE,
        )

        self.reduce_sync_barrier = cutlass.pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierBwdSm100.dQaccReduce),
            num_threads=len(self.reduce_warp_ids) * cute.arch.WARP_SIZE,
        )

        # TMEM setup
        SM100_TMEM_CAPACITY_COLUMNS = 512
        self.tmem_alloc_cols = SM100_TMEM_CAPACITY_COLUMNS

        # A 2-CTA MMA splits its M across the CTA pair, so a CTA can end up owning
        # fewer than the 128 TMEM lanes. Such an accumulator is FOLDED: logical
        # columns j and j + N/2 share one column, living in lanes [0, rows) and
        # [rows, 128) respectively, which shrinks the column footprint by the same
        # factor (see tmem_cols, which the d192 dQ offset below relies on).
        # Folding has two consequences beyond the column count:
        #   - the accumulator cannot back an MMA A operand (that layout wants a
        #     whole row in one lane, and tmem stores are lane-local), so P / dS
        #     move to SMEM -- see mma_P_from_smem
        #   - copy_utils.make_tmem_copy's fixed 128-datapath tiler mis-slices it;
        #     the compute warp must go through tcgen05.make_tmem_copy instead
        self.folded_kv_acc = self.tile_n < 128  # S / dP / dV / dK
        # P (A of the dV MMA) and dS (A of the dK MMA) cannot live in a folded
        # S / dP accumulator.
        self.mma_P_from_smem = self.folded_kv_acc
        self.mma_dS_from_smem = self.mma_P_from_smem
        self.mma_A_source = (
            tcgen05.OperandSource.SMEM
            if self.mma_P_from_smem
            else tcgen05.OperandSource.TMEM
        )
        # sdSt (A of the dK MMA, M=n K=m) and sdS (A of the dQ MMA, M=m K=n) share
        # one allocation everywhere else, because at most one of them is ever live.
        # Both are m-contiguous in 128B blocks, so per CTA they tile the same two
        # blocks -- but they want different content in one of them:
        #   sdSt block b = (own n slice, m half b)          -- both blocks local
        #   sdS  block b = (n slice b,   own m half)        -- block peer is remote
        # The b == cta_rank block agrees; the other one does not. So when the dK MMA
        # also reads dS from SMEM (the folded path) the two views need their own
        # buffers.
        self.separate_sdS_buffers = self.mma_dS_from_smem and self.use_2cta_instrs

        if self.use_2cta_instrs:
            # Every offset below is "previous accumulator's first column + that
            # accumulator's own width", and the width always comes from
            # tmem_cols(its mma tiler) so that folding is accounted for.
            if self.tile_hdim == 256 and self.tile_hdimv == 256:
                # 2-CTA d=256/dv=256. Folding (tile_n=64 -> 64 lanes per CTA) halves
                # every accumulator, which is what makes dK resident -- so no separate
                # dK reduce / dKV_postprocess is needed for this config:
                #   dV [0,128) | dK [128,256) | S/P [256,320) | dP/dS [320,384) | dQ [384,512)
                # Exactly 512 columns with no overlap, so unlike the d192 layout below
                # no accumulator has to be time-shared.
                assert self.tile_m == 128
                assert self.tile_n == 64
                self.tmem_dV_offset = 0
                self.tmem_dK_offset = self.tmem_dV_offset + self.tmem_cols(self.mma_tiler_pdo)
                self.tmem_S_offset = self.tmem_dK_offset + self.tmem_cols(self.mma_tiler_dsq)
                self.tmem_P_offset = self.tmem_S_offset  # overlap with S
                self.tmem_dP_offset = self.tmem_S_offset + self.tmem_cols(self.mma_tiler_kq)
                self.tmem_dS_offset = self.tmem_dP_offset  # overlap with dP
                self.tmem_dQ_offset = self.tmem_dP_offset + self.tmem_cols(self.mma_tiler_vdo)
            elif self.tile_hdim == 192 and self.tile_hdimv == 128:
                assert self.tile_m == 128
                assert self.tile_n == 128
                # Only dV and dK are resident; S/P, dP/dS and dQ time-share the tail
                # 192 columns (the ranges below deliberately overlap and the MMA warp
                # gates each reuse on a pipeline empty-barrier).
                self.tmem_dV_offset = 0
                self.tmem_dK_offset = self.tmem_dV_offset + self.tmem_cols(self.mma_tiler_pdo)
                self.tmem_S_offset = self.tmem_dK_offset + self.tmem_cols(self.mma_tiler_dsq)
                self.tmem_P_offset = self.tmem_S_offset  # overlap with S
                self.tmem_dP_offset = SM100_TMEM_CAPACITY_COLUMNS - self.tmem_cols(self.mma_tiler_vdo)
                self.tmem_dS_offset = self.tmem_dP_offset  # overlaps with dP
                self.tmem_dQ_offset = SM100_TMEM_CAPACITY_COLUMNS - self.tmem_cols(self.mma_tiler_dsk)
            else:
                self.tmem_S_offset = 0
                self.tmem_P_offset = 0  # overlap with S
                self.tmem_dV_offset = self.tmem_S_offset + self.tmem_cols(self.mma_tiler_kq)
                self.tmem_dP_offset = self.tmem_dV_offset + self.tmem_cols(self.mma_tiler_pdo)
                self.tmem_dK_offset = self.tmem_dP_offset + self.tmem_cols(self.mma_tiler_vdo)
                self.tmem_dS_offset = self.tmem_dP_offset  # overlap with dP
                # dQ is folded (tile_hdim / 2 wide) and time-shares the tail of S/P.
                self.tmem_dQ_offset = self.tmem_S_offset + (self.tile_hdim // 2)
        else:
            # 1-CTA generic layout, following the 2-CTA convention. Keeps dK resident
            # in TMEM and does NOT time-share dK/dQ with S/P.
            self.tmem_S_offset = 0
            self.tmem_P_offset = 0  # overlap with S
            self.tmem_dV_offset = self.tmem_S_offset + self.tmem_cols(self.mma_tiler_kq)
            self.tmem_dP_offset = self.tmem_dV_offset + self.tmem_cols(self.mma_tiler_pdo)
            self.tmem_dQ_offset = self.tmem_dP_offset  # time-shares with dP/dS
            self.tmem_dK_offset = self.tmem_dP_offset + self.tmem_cols(self.mma_tiler_vdo)
            self.tmem_dS_offset = self.tmem_dP_offset  # overlap with dP

        # The layouts above deliberately overlap, so "fits in the 512-column window" is
        # the only invariant all of them share -- and the one whose violation shows up
        # as silently corrupted output rather than an error. dV is the one accumulator
        # that can be staged as two halves; tmem_cols sizes one.
        for acc_name, acc_offset, acc_tiler, acc_halves in (
            ("dV", self.tmem_dV_offset, self.mma_tiler_pdo, 1),
            ("dK", self.tmem_dK_offset, self.mma_tiler_dsq, 1),
            ("S/P", self.tmem_S_offset, self.mma_tiler_kq, 1),
            ("dP/dS", self.tmem_dP_offset, self.mma_tiler_vdo, 1),
            ("dQ", self.tmem_dQ_offset, self.mma_tiler_dsk, 1),
        ):
            acc_cols = acc_halves * self.tmem_cols(acc_tiler)
            assert acc_offset + acc_cols <= SM100_TMEM_CAPACITY_COLUMNS, (
                f"{acc_name} accumulator spans columns "
                f"[{acc_offset}, {acc_offset + acc_cols}), past the "
                f"{SM100_TMEM_CAPACITY_COLUMNS}-column TMEM window "
                f"(d={self.tile_hdim}, dv={self.tile_hdimv}, tile_m={self.tile_m}, "
                f"tile_n={self.tile_n}, 2cta={self.use_2cta_instrs})"
            )

        # The 2-CTA path with hdim > 128: dV and dK are TMEM-resident, Q/dO have no
        # separate transposed smem copies, and the MMA warp runs the flat loop.
        # d192/dv128 and d256/dv256 both land here.
        self.use_2cta_bigd = self.use_2cta_instrs and self.tile_hdim > 128
        # tile_boundary_barrier's participant count assumes the 2-CTA warp layout (the
        # relay warp only runs the work-tile loop when use_2cta_instrs), so it is only
        # built once that is guaranteed.
        assert not self.is_persistent or self.use_2cta_bigd, (
            "persistent bwd is only implemented for the 2-CTA hdim>128 path"
        )
        self.tile_boundary_sync = self.is_persistent
        if self.tile_boundary_sync:
            # Persistent only. Two things are only safe because a CTA currently owns
            # exactly one work tile:
            #   - sdV aliases sV and sdK aliases sK (see the sdV/sdK tensors in
            #     kernel()), and sFM_max_min / sStartEndRowIndices are single-buffered,
            #     so the load warp must not start tile N+1 before the epilogue / reduce
            #     of tile N is done.
            #   - the MMA warp only waits for pipeline_dKV's empty AFTER the m loop, so
            #     the next tile's zero-init dV / dK MMA would overwrite TMEM while the
            #     previous tile's epilogue is still reading it.
            # A rendezvous of all five roles at the tail of the work-tile body fixes
            # both. All roles run the same number of tiles because they share the tile
            # scheduler, and no role has a dependency that is only satisfied after
            # another role's barrier, so this cannot deadlock. Warp 15 (empty) exits
            # early and does not take part.
            self.tile_boundary_barrier = cutlass.pipeline.NamedBarrier(
                barrier_id=int(NamedBarrierBwdSm100.TileBoundary),
                num_threads=(
                    len(self.compute_warp_ids)
                    + len(self.reduce_warp_ids)
                    + len((self.mma_warp_id, self.load_warp_id, self.relay_warp_id))
                )
                * cute.arch.WARP_SIZE,
            )
        # Whether the bigd path's dQ accumulator shares columns with S/P. When it
        # does, the compute warp has to release the S_P pipeline early (before P is
        # written) and the MMA warp has to wait for dQ to drain before issuing S.
        # The d256/dv256 layout is disjoint, so it keeps the cheaper late release.
        # Scoped to bigd on purpose: dQ also overlaps S/P in the 2-CTA small-hdim
        # and 1-CTA layouts, but those signal "S is out of TMEM" through pipeline_dS
        # (see compute_step) rather than through this early release.
        self.bigd_dQ_overlaps_S = self.use_2cta_bigd and (
            self.tmem_dQ_offset < self.tmem_S_offset + self.tmem_cols(self.mma_tiler_kq)
            and self.tmem_S_offset < self.tmem_dQ_offset + self.tmem_cols(self.mma_tiler_dsk)
        )
        # Whether the late dQ-empty wait (see the bigd MMA loop) is used. It only
        # applies when dQ's TMEM columns are disjoint from S/P, i.e. d256/dv256.
        self.late_dq_empty_wait = not self.bigd_dQ_overlaps_S
        # Whether dP shares TMEM columns with S/P on the bigd path (d192: dP overlaps the
        # tail of S). If not (d256), the dP MMA does not have to wait for S to be read
        # out; the S_P empty wait only has to precede the NEXT S MMA, so it moves from
        # step 2 to step 1 of the following iteration and dP issues right behind S.
        self.bigd_dP_overlaps_S = self.use_2cta_bigd and (
            self.tmem_dP_offset < self.tmem_S_offset + self.tmem_cols(self.mma_tiler_kq)
            and self.tmem_S_offset < self.tmem_dP_offset + self.tmem_cols(self.mma_tiler_vdo)
        )
        # Release S_P right after S is read out of TMEM (before mask / exp / P store),
        # so the dP MMA, which waits on S_P empty, overlaps the softmax. Safe when P does
        # not go back over S in TMEM: d192 (dQ aliases S, already released early) and
        # the folded d256 layout (P lives in SMEM, S/P and dP/dS columns are disjoint).
        self.early_S_P_release = self.use_2cta_bigd and (
            self.bigd_dQ_overlaps_S or self.mma_P_from_smem
        )
        # Folded bigd (d256): the dS half the peer needs for its dQ MMA already sits in
        # sdSt (the dK A operand) in the peer's dQ A-operand layout (checked in
        # _check_dS_exchange_layouts), so it is sent from there. No sdS_xchg staging in
        # sdQaccum, hence no wait on the dQ reduce draining sdQaccum before storing dS.
        self.dS_xchg_from_sdSt = self.use_2cta_bigd and self.folded_kv_acc
        # Folded bigd (d256): split the folded S / dP fragment between the two compute
        # warpgroups (each takes one 32-column chunk of the 64 a lane holds) instead of
        # both warpgroups redoing the whole T2R / mask / exp / dS / SMEM store. Needs
        # dS_xchg_from_sdSt: the per-thread sdS_xchg store it replaces is gone there.
        self.split_folded_compute_wg = self.dS_xchg_from_sdSt

        if (not is_causal and not is_local) or deterministic:
            self.num_regs_reduce = 136 if self.use_2cta_instrs else 152
            self.num_regs_compute = 136
            self.num_regs_load = 104 if self.use_2cta_instrs else 96 - 8
            self.num_regs_mma = 104 if self.use_2cta_instrs else self.num_regs_load
        else:
            self.num_regs_reduce = 136 if self.use_2cta_instrs else 136
            self.num_regs_compute = 136 if self.use_2cta_instrs else 144
            self.num_regs_load = 104 if self.use_2cta_instrs else 96 - 8
            self.num_regs_mma = 104 if self.use_2cta_instrs else self.num_regs_load
        self.num_regs_empty = 24

        if const_expr(self.use_2cta_bigd):
            self.num_regs_reduce = 128 + 8
            self.num_regs_compute = 128 + 8
            self.num_regs_load = 128 - 24
            self.num_regs_mma = self.num_regs_load

        assert (
            self.num_regs_reduce
            + self.num_regs_compute * 2
            + max(self.num_regs_load, self.num_regs_mma)
            <= 512
        )

        self.buffer_align_bytes = 1024

    def tmem_cols(self, mma_tiler) -> int:
        """Per-CTA TMEM columns of the accumulator for `mma_tiler`.

        The tiler's M is cluster-wide; the column rule (including folding) lives in
        blackwell_helpers.tmem_cols.
        """
        return sm100_utils.tmem_cols(mma_tiler[1], mma_tiler[0], self.cta_group_size)

    def smem_A_mn_view(self, sA: cute.Tensor, dtype, swizzle) -> cute.Tensor:
        """(n, m) view of an MMA A operand in SMEM, as the compute warp writes it.

        make_smem_layout_a stores the operand as one 128B-period block per mma_k
        elements, i.e. offset(n, m) = n * mma_k + m % mma_k
        + (m // mma_k) * tile_n * mma_k. The two blocks line up with the folded
        accumulator's two lane halves, so this view needs no transpose. Rank 3 to
        satisfy the tiled copy's tiler rank.

        `swizzle` is the operand layout's .inner and is NOT optional: sA.iterator
        drops the swizzle, so writing through the bare iterator lands the values at
        unswizzled addresses while the MMA descriptor reads them swizzled.
        """
        mma_k = 128 // (dtype.width // 8)
        assert self.tile_m % mma_k == 0
        return cute.make_tensor(
            cute.recast_ptr(sA.iterator, swizzle, dtype=dtype),
            cute.make_layout(
                ((self.tile_n, (mma_k, self.tile_m // mma_k)), 1, 1),
                stride=((mma_k, (1, self.tile_n * mma_k)), 0, 0),
            ),
        )

    def smem_dS_dq_block_view(self, base_iter, swizzle) -> cute.Tensor:
        """(n, m) view of ONE mma_k-block of the dQ MMA's A operand (dS, M=m K=n).

        Same byte formula as smem_A_mn_view -- offset(n, m) = n * mma_k + m % mma_k
        -- because both operands are m-contiguous in 128B blocks. The difference is
        which (n slice, m half) pair each block holds: sdSt's two blocks are the two
        m halves of our own n slice, sdS's are the two n slices of our own m half
        (see separate_sdS_buffers). Coordinates stay the folded accumulator's, so m
        still runs over all tile_m rows and its high bit gets stride 0: a thread must
        only store through the block that matches its own lane half.
        """
        mma_k = 128 // (self.ds_dtype.width // 8)
        assert self.tile_n <= mma_k, "one n slice must fit in a single 128B block"
        assert self.tile_m == mma_k * self.cta_group_size
        return cute.make_tensor(
            cute.recast_ptr(base_iter, swizzle),
            cute.make_layout(
                ((self.tile_n, (mma_k, self.cta_group_size)), 1, 1),
                stride=((mma_k, (1, 0)), 0, 0),
            ),
        )

    def _check_dS_exchange_layouts(self):
        """dS_xchg_from_sdSt sends sdSt block peer_rank (our n slice, the peer's m half)
        as the peer's sdS block cta_rank (n slice cta_rank, its own m half), a raw
        DSMEM copy of tile_n * mma_k elements. Check that contract on the layouts:
        same swizzle, both halves one contiguous block at that offset, same element
        order."""
        mma_k = 128 // (self.ds_dtype.width // 8)
        src = self.sdSt_layout.outer  # ((kv, q16), 1, (q16_chunks, q_half))
        dst = cute.logical_divide(
            self.sdS_layout.outer, (None, None, cute.size(self.sdS_layout.outer.shape[2]) // 2)
        )  # ((q, kv16), 1, (kv16_chunks, kv_block))
        assert str(self.sdSt_layout.inner) == str(self.sdS_layout.inner)
        assert cute.crd2idx((0, 0, (0, 1)), src) == self.tile_n * mma_k
        assert cute.crd2idx((0, 0, (0, 1)), dst) == self.tile_n * mma_k
        for kv in (0, 1, 17, self.tile_n - 1):
            for q in (0, 1, 15, 16, self.tile_m // 2 - 1):
                assert cute.crd2idx(((kv, q % 16), 0, (q // 16, 1)), src) == cute.crd2idx(
                    ((q, kv % 16), 0, (kv // 16, 1)), dst
                ), "dS exchange: sdSt half is not in the peer's dQ A-operand layout"

    def _setup_attributes(self):
        self.Q_stage = 1 if self.use_2cta_instrs else 2
        self.dO_stage = 1
        self.K_smem_stages = 1
        self.single_stage = 1
        # LSE_stage = Q_stage and dPsum_stage = dO_stage
        self.sdKVaccum_stage = 2
        # number of tma reduce adds per dQacc mma
        # todo: try 32/1 or 48/2 for 2cta d=192 dv=128
        if self.use_2cta_instrs:
            self.dQ_reduce_ncol_t2r = 32
            if self.tile_hdim == 192:
                # 4-vec flashmask needs 2 extra Int32 columns of sStartEndRowIndices
                # (tile_n * 8 B) and this config is already exactly at the SMEM cap, so
                # give up the second dQaccum buffer. 32/1 keeps sdQaccum >= sdS_xchg,
                # which overlays it (see the assert in _setup_smem_layout); 24/1 would
                # be too small.
                if not self.is_causal and not self.has_ut_start:
                    self.dQ_reduce_ncol = 24
                    self.sdQaccum_stage = 2
                else:
                    self.dQ_reduce_ncol = 32
                    self.sdQaccum_stage = 1
            elif self.tile_hdim == 256:
                # ncol only chooses how the reduce warps chunk ONE register fragment
                # into bulk reduce-adds, so it does not change the flat gmem order of
                # dq_accum that flash_bwd_postprocess.py reads. Same 16 KiB of staging
                # as one 32-column stage, but as two 16-column stages so the r2s fill
                # of stage s+1 overlaps stage s's cpasync_reduce_bulk_add_f32.
                # ncol (16) != ncol_t2r (32) also turns off
                # split_dq_t2r: the whole dQ fragment is read out once and its TMEM is
                # released right away instead of after the last bulk reduce-add.
                self.dQ_reduce_ncol = 16
                self.sdQaccum_stage = 2
            else:
                self.dQ_reduce_ncol = 16 if self.deterministic else 8
                self.sdQaccum_stage = 2 if self.deterministic else 4
        else:
            self.dQ_reduce_ncol = 32
            self.dQ_reduce_ncol_t2r = self.dQ_reduce_ncol
            # Same SMEM squeeze as the 2cta d=192 case above: the 1cta d<=128 config is
            # exactly at the cap, so 4-vec flashmask drops to a single dQaccum buffer.
            self.sdQaccum_stage = 1 if self.has_ut_start else 64 // self.dQ_reduce_ncol

        # ncu on d256/dv256 shows the kernel is register-spill bound (large local
        # ld/st traffic, most warp stalls on an L1TEX scoreboard, tensor pipe idle),
        # not MMA or bandwidth bound. The source is dQacc_reduce_step's single T2R of
        # the whole dQ accumulator: that fragment is hdim // cta_group_size floats per
        # thread (128 at d256) against num_regs_reduce = 136. Loading one gmem stage at
        # a time keeps only dQ_reduce_ncol_t2r floats live, at the cost of holding dQ's
        # TMEM until every stage is out. Needs the t2r and gmem chunkings to agree so
        # that one t2r stage maps 1:1 onto one bulk reduce-add.
        self.split_dq_t2r = (
            self.use_2cta_bigd
            and self.tile_hdim >= 256
            and self.dQ_reduce_ncol == self.dQ_reduce_ncol_t2r
        )

        hdim_for_reduce = self.tile_hdim

        assert (hdim_for_reduce // self.cta_group_size) % self.dQ_reduce_ncol == 0
        self.dQaccum_reduce_stage = hdim_for_reduce // self.dQ_reduce_ncol
        self.dQaccum_reduce_stage_t2r = hdim_for_reduce // self.dQ_reduce_ncol_t2r
        self.cluster_reduce_dQ = False and cute.size(self.cluster_shape_mn) > 1
        # number of tma reduce adds for dKacc and dVacc epilogue
        self.dK_reduce_ncol = math.gcd(32, hdim_for_reduce // 2)
        # CTA group for MMA operations
        self.cta_group = tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE

    def _get_tiled_mma(self):
        # S = K @ Q.T
        tiled_mma_S = sm100_utils_basic.make_trivial_tiled_mma(
            self.q_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler_kq[:2],
        )
        # dP = V @ dO.T
        tiled_mma_dP = sm100_utils_basic.make_trivial_tiled_mma(
            self.do_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler_vdo[:2],
        )
        # dV += P @ dO --> (K, MN) major
        tiled_mma_dV = sm100_utils_basic.make_trivial_tiled_mma(
            self.do_dtype,
            tcgen05.OperandMajorMode.K,  # P_major_mode
            tcgen05.OperandMajorMode.MN,  # dO_major_mode
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler_pdo[:2],
            a_source=self.mma_A_source,
        )
        # dK += dS.T @ Q
        mma_dK_a_src = (
            tcgen05.OperandSource.SMEM
            if self.mma_dS_from_smem
            else tcgen05.OperandSource.TMEM
        )
        tiled_mma_dK = sm100_utils_basic.make_trivial_tiled_mma(
            self.do_dtype,
            tcgen05.OperandMajorMode.K,  # dS_major_mode
            tcgen05.OperandMajorMode.MN,  # Q_major_mode
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler_dsq[:2],
            a_source=mma_dK_a_src,
        )
        # dQ = dS @ K
        tiled_mma_dQ = sm100_utils_basic.make_trivial_tiled_mma(
            self.k_dtype,
            tcgen05.OperandMajorMode.MN,  # dS_major_mode
            tcgen05.OperandMajorMode.MN,  # Kt_major_mode
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler_dsk[:2],
        )
        return tiled_mma_S, tiled_mma_dP, tiled_mma_dK, tiled_mma_dV, tiled_mma_dQ

    def _setup_smem_layout(self):
        # S.T = K @ Q.T
        sK_layout = sm100_utils_basic.make_smem_layout_a(
            self.tiled_mma_S,
            self.mma_tiler_kq,
            self.k_dtype,
            self.K_smem_stages,
        )
        self.sK_layout = cute.slice_(sK_layout, (None, None, None, 0))
        self.sQ_layout = sm100_utils_basic.make_smem_layout_b(
            self.tiled_mma_S,
            self.mma_tiler_kq,
            self.q_dtype,
            self.Q_stage,
        )
        # dP.T = V @ dO.T
        sV_layout = sm100_utils_basic.make_smem_layout_a(
            self.tiled_mma_dP,
            self.mma_tiler_vdo,
            self.v_dtype,
            self.K_smem_stages,
        )
        self.sV_layout = cute.slice_(sV_layout, (None, None, None, 0))
        self.sdOt_layout = sm100_utils_basic.make_smem_layout_b(
            self.tiled_mma_dP,
            self.mma_tiler_vdo,
            self.do_dtype,
            self.dO_stage,
        )
        # dV += P.T @ dO
        tP_layout = sm100_utils_basic.make_smem_layout_a(
            self.tiled_mma_dV,
            self.mma_tiler_pdo,
            self.do_dtype,
            1,
        )
        self.tP_layout = cute.slice_(tP_layout, (None, None, None, 0))
        self.sdO_layout = sm100_utils_basic.make_smem_layout_b(
            self.tiled_mma_dV,
            self.mma_tiler_pdo,
            self.do_dtype,
            self.dO_stage,
        )
        # dK += dS.T @ Q
        sdSt_layout = sm100_utils_basic.make_smem_layout_a(
            self.tiled_mma_dK,
            self.mma_tiler_dsq,
            self.ds_dtype,
            1,
        )
        self.sdSt_layout = cute.slice_(sdSt_layout, (None, None, None, 0))
        tdS_layout = sm100_utils_basic.make_smem_layout_a(
            self.tiled_mma_dK,
            self.mma_tiler_dsq,
            self.ds_dtype,
            1,
        )
        self.tdS_layout = cute.slice_(tdS_layout, (None, None, None, 0))
        self.sQt_layout = sm100_utils_basic.make_smem_layout_b(
            self.tiled_mma_dK,
            self.mma_tiler_dsq,
            self.q_dtype,
            self.Q_stage,
        )
        # dQ = dS @ K
        sdS_layout = sm100_utils_basic.make_smem_layout_a(
            self.tiled_mma_dQ,
            self.mma_tiler_dsk,
            self.ds_dtype,
            1,
        )
        self.sdS_layout = cute.slice_(sdS_layout, (None, None, None, 0))
        if self.dS_xchg_from_sdSt:
            self._check_dS_exchange_layouts()
        sKt_layout = sm100_utils_basic.make_smem_layout_b(
            self.tiled_mma_dQ,
            self.mma_tiler_dsk,
            self.k_dtype,
            self.K_smem_stages,
        )
        self.sKt_layout = cute.slice_(sKt_layout, (None, None, None, 0))
        self.sdS_xchg_layout = cute.make_layout(shape=(self.tile_n, self.tile_m // 2))
        self.sdQaccum_layout = cute.make_layout(
            (self.tile_m * self.dQ_reduce_ncol, self.sdQaccum_stage)
        )
        if self.use_2cta_bigd and not self.dS_xchg_from_sdSt:
            # The bigd path has no sdS_xchg allocation of its own, it borrows
            # sdQaccum (gated on dQaccum_empty), so shrinking sdQaccum must not
            # shrink it below the exchange buffer.
            #
            # Giving sdS_xchg its own buffer (funded by a smaller dQ_reduce_ncol) so the
            # compute warp stops waiting on the reduce warps' dQaccum_empty measured
            # slower, as did the smaller ncol on its own. So the
            # compute -> dS exchange -> dQ MMA -> dQ reduce ring is NOT the critical
            # path here and this overlay is not worth removing. Do not retry without
            # profile evidence.
            assert cute.cosize(self.sdS_xchg_layout) * (
                self.ds_dtype.width // 8
            ) <= cute.cosize(self.sdQaccum_layout) * (self.dqaccum_dtype.width // 8), (
                "sdS_xchg does not fit in sdQaccum"
            )
        self.sLSE_layout = cute.make_layout(
            shape=(self.tile_m, self.Q_stage),
            stride=(1, cute.round_up(self.tile_m, 64)),
        )
        self.sdPsum_layout = cute.make_layout(
            shape=(self.tile_m, self.dO_stage),
            stride=(1, cute.round_up(self.tile_m, 64)),
        )
        hdim_epi = self.tile_hdim
        hdimv_epi = self.tile_hdimv
        # Folded dKV accumulator: a CTA owns only tile_n < 128 TMEM lanes, so the
        # accumulator's N (= hdim) is split and [hdim/2, hdim) lives in lanes
        # 64..127. One warpgroup's 128 threads therefore cover 128 lanes = two
        # disjoint hdim strips (delta = hdim/2) of tile_n rows each, instead of
        # one 128-row strip.
        #   * TMA store path: GMEM wants the two strips at different hdim
        #     offsets, so each strip needs its own SMEM buffer and its own store
        #     (epi_smem_strips = 2). Halving the columns per buffer keeps the
        #     staging bytes and the number of stores identical to the unfolded case.
        #   * fp32 reduce path: the accum "panel" convention is exactly
        #     "128 threads x ncol values, fold inside the panel" -- the same one
        #     dQ already uses (its accumulator is folded too) and the one
        #     flash_bwd_postprocess.py's 2-CTA branch decodes with row_groups=2.
        #     So the two strips share ONE 128*ncol buffer and ONE bulk reduce;
        #     only the panel size doubles.
        self.epi_num_strips = 2 if self.folded_kv_acc else 1
        self.epi_smem_strips = self.epi_num_strips if not self.dKV_postprocess else 1
        # One SMEM strip is epi_tile[0] = tile_n rows tall and is fed by
        # 128 / epi_smem_strips threads: when the strips are staged separately the
        # other half of the warpgroup owns the OTHER hdim strip and writes its own
        # buffer, so it must not be tiled into this one.
        self.epi_threads_r2s = 128 // self.epi_smem_strips
        epi_col_max = 128 // (self.dk_dtype.width // 8) // self.epi_smem_strips  # 64 or 32
        self.sdK_epi_tile = (
            self.tile_n,
            math.gcd(epi_col_max, hdim_epi // 2 // self.epi_smem_strips),  # 64 or 32
        )  # subtiles mma_tiler_dsq[:2] = mma_tiler_pdo[:2]
        self.sdV_epi_tile = (
            self.tile_n,
            math.gcd(epi_col_max, hdimv_epi // 2 // self.epi_smem_strips),  # 64 or 32
        )  # subtiles mma_tiler_pdo[:2]
        # headdim_64 gets 1 stage
        self.num_epi_stages = max(
            1, (hdim_epi // 2 // self.epi_smem_strips) // self.sdK_epi_tile[1]
        )
        self.num_epi_stages_v = max(
            1, (hdimv_epi // 2 // self.epi_smem_strips) // self.sdV_epi_tile[1]
        )
        self.sdK_flat_epi_tile = self.tile_n * (hdim_epi // 2) // self.num_epi_stages
        self.sdV_flat_epi_tile = self.tile_n * (hdimv_epi // 2) // self.num_epi_stages_v

        # fp32 reduce staging: one panel = 128 threads x dK_reduce_ncol values,
        # i.e. epi_num_strips * tile_n rows (both hdim strips live in one panel).
        self.dKV_reduce_panel = self.epi_num_strips * self.tile_n * self.dK_reduce_ncol

        if const_expr(not self.dKV_postprocess):
            self.sdK_layout = sm100_utils_basic.make_smem_layout_epi(
                self.dk_dtype,
                LayoutEnum.ROW_MAJOR,
                self.sdK_epi_tile,
                2 * self.epi_smem_strips,  # num compute wgs x staged hdim strips per wg
            )
            self.sdV_layout = sm100_utils_basic.make_smem_layout_epi(
                self.dv_dtype,
                LayoutEnum.ROW_MAJOR,
                self.sdV_epi_tile,
                2 * self.epi_smem_strips,  # num compute wgs x staged hdim strips per wg
            )
        else:
            self.sdK_layout = cute.make_layout((self.dKV_reduce_panel, 2))
            # self.dK_reduce_ncol same for dV
            self.sdV_layout = cute.make_layout((self.dKV_reduce_panel, 2))

        # 4 columns (LTS/LTE/UTS/UTE) only when startend_row_indices.shape[-1] == 4,
        # otherwise 2 columns (LTS + either LTE or UTE) as before.
        fm_num_vecs = 4 if const_expr(self.has_ut_start) else 2
        self.sStartEndRowIndices_layout = cute.make_layout(
            shape=(self.tile_n, fm_num_vecs),
            stride=(1, self.tile_n),
        )

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mdO: cute.Tensor,
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        mdQaccum: cute.Tensor,
        mdK: cute.Tensor,
        mdV: cute.Tensor,
        softmax_scale: Float32,
        window_size_left: Int32 | int | None = None,
        window_size_right: Int32 | int | None = None,
        mdQ_semaphore: Optional[cute.Tensor] = None,
        mdK_semaphore: Optional[cute.Tensor] = None,
        mdV_semaphore: Optional[cute.Tensor] = None,
        blocksparse_tensors=None,
        flashmask_info: Optional[FlashMaskInfo] = None,
        overlap_k_addr: Optional[cutlass.Int64] = None,
        overlap_v_addr: Optional[cutlass.Int64] = None,
        overlap_work_done_addr: Optional[cutlass.Int64] = None,
        overlap_spin_timeout_ns: Optional[cutlass.Int64] = None,
        overlap_segment_idx: Optional[cutlass.Int32] = None,
        overlap_local_kv_size: Optional[cutlass.Int32] = None,
        overlap_dk_addr: Optional[cutlass.Int64] = None,
        overlap_dv_addr: Optional[cutlass.Int64] = None,
        overlap_b: Optional[cutlass.Int32] = None,
        overlap_s: Optional[cutlass.Int32] = None,
        overlap_h: Optional[cutlass.Int32] = None,
        overlap_d: Optional[cutlass.Int32] = None,
        overlap_comm_rpb: cutlass.Constexpr = None,
        overlap_bhsd_layout: cutlass.Constexpr = False,
        # Always keep stream as the last parameter (EnvStream: obtained implicitly via TVM FFI).
        stream: cuda.CUstream = None,
    ):
        # FM-4 split-AG overlap: the segment K/V live in the NCCL-window SRBuffer
        # (no Paddle tensor / dlpack capsule), so they arrive as a raw addr plus
        # the segment (B, S_segment, H, D) dims as RUNTIME Int32 scalars. Rebuild the
        # views HERE in this jit body's MLIR Context (make_*_from_addr requires it),
        # with Int32 dims giving the dynamic (?,?,?,?):(?,?,?,1) layout that the
        # dlpack path produces -- static dims read the wrong bytes.
        # This mirrors the forward SRBuffer view construction. Readiness is gated
        # per communication work item in the load warp below.
        self.overlap_bhsd_layout = const_expr(overlap_bhsd_layout)
        assert (overlap_work_done_addr is None) == (overlap_spin_timeout_ns is None), (
            "the work_done gate needs its spin budget (barrier.spin_timeout_ns_from_env)"
        )
        if const_expr(overlap_k_addr is not None):
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
        if const_expr(overlap_dk_addr is not None):
            mdK = utils.make_contiguous_bshd_from_addr(
                overlap_dk_addr, overlap_b, overlap_s, overlap_h, overlap_d,
                mQ.element_type, align=16,
            )
            mdV = utils.make_contiguous_bshd_from_addr(
                overlap_dv_addr, overlap_b, overlap_s, overlap_h, overlap_d,
                mQ.element_type, align=16,
            )
        # 1-CTA skips the dKV epilogue for fully masked key blocks, so with an MHA direct
        # store (no dKV postprocess) those rows have to be written as zeros here: neither
        # the overlap NCCL send slot (reduce-scatter sums all rows) nor the host dk / dv
        # (allocated with empty_like) are pre-zeroed. 2-CTA always runs the epilogue (a
        # fully masked tile keeps one masked iteration, see fm_skip_info).
        self.zero_skipped_dKV = const_expr(
            not self.use_2cta_instrs
            and not self.dKV_postprocess
        )
        self.q_dtype = mQ.element_type
        self.k_dtype = mK.element_type
        self.v_dtype = mV.element_type
        self.do_dtype = mdO.element_type
        self.lse_dtype = mLSE.element_type
        self.dpsum_dtype = mdPsum.element_type
        self.dqaccum_dtype = mdQaccum.element_type
        self.dk_dtype = mdK.element_type
        self.dv_dtype = mdV.element_type
        self.ds_dtype = self.q_dtype

        self.enable_flashmask = cutlass.const_expr(flashmask_info is not None)
        self.has_lt_end = const_expr(flashmask_info is not None and flashmask_info.LTE_nblock_max is not None)
        self.has_ut_start = const_expr(flashmask_info is not None and flashmask_info.UTS_nblock_max is not None)
        self.has_ut_end = const_expr(flashmask_info is not None and flashmask_info.UTE_nblock_max is not None)

        if const_expr(self.dKV_postprocess):
            assert self.dk_dtype.width == 32, "Must accumulate dK in float precision for GQA"
            assert self.dv_dtype.width == 32, "Must accumulate dV in float precision for GQA"

        mdQaccum, mdK, mdV = [assume_tensor_aligned(t) for t in (mdQaccum, mdK, mdV)]

        # (b, s, n, h) --> (s, h, n, b)
        QO_layout_transpose = [1, 3, 2, 0]
        mQ, mdO = [layout_utils.select(t, mode=QO_layout_transpose) for t in (mQ, mdO)]

        KV_layout_transpose = [1, 3, 2, 0]
        mK, mV = [layout_utils.select(t, mode=KV_layout_transpose) for t in (mK, mV)]

        # (b, n, s) --> (s, n, b)
        LSE_dPsum_dQaccum_transpose = [2, 1, 0]
        mLSE, mdPsum, mdQaccum = [
            layout_utils.select(t, mode=LSE_dPsum_dQaccum_transpose) for t in (mLSE, mdPsum, mdQaccum)
        ]
        if const_expr(not self.dKV_postprocess):
            layout_dKV_transpose = KV_layout_transpose
        else:
            layout_dKV_transpose = LSE_dPsum_dQaccum_transpose
        mdK, mdV = [layout_utils.select(t, mode=layout_dKV_transpose) for t in (mdK, mdV)]
        # (s, h, n, b) --> (h, s, n, b)
        dO_transpose = [1, 0, 2, 3]
        mdO = layout_utils.select(mdO, mode=dO_transpose)

        # Transposes for 2-CTA K/Q paths (Q follows Q seqlens, K follows K seqlens)
        transpose_sh_q = dO_transpose
        transpose_sh_k = [1, 0, 2, 3]

        semaphore_transpose = [2, 3, 1, 0]  # (b, n, block, stage) -> (block, stage, n, b)
        if const_expr(self.deterministic):
            assert mdQ_semaphore is not None
            mdQ_semaphore = layout_utils.select(mdQ_semaphore, mode=semaphore_transpose)

        if const_expr(self.deterministic and (self.qhead_per_kvhead > 1 or self.dKV_postprocess)):
            assert mdK_semaphore is not None
            assert mdV_semaphore is not None
            mdK_semaphore, mdV_semaphore = [
                layout_utils.select(t, mode=semaphore_transpose) for t in (mdK_semaphore, mdV_semaphore)
            ]
        else:
            mdK_semaphore = None
            mdV_semaphore = None

        self._setup_attributes()
        (
            self.tiled_mma_S,
            self.tiled_mma_dP,
            self.tiled_mma_dK,
            self.tiled_mma_dV,
            self.tiled_mma_dQ,
        ) = self._get_tiled_mma()
        self._setup_smem_layout()

        self.use_tma_store = True

        self.cluster_shape_mnk = (*self.cluster_shape_mn, 1)
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (self.tiled_mma_S.thr_id.shape,),
        )
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.is_q_do_mcast = self.num_mcast_ctas_b > 1

        if const_expr(not self.dKV_postprocess):
            self.mdK_layout_enum = LayoutEnum.from_tensor(mdK)
            self.mdV_layout_enum = LayoutEnum.from_tensor(mdV)
            dK_major_mode = self.mdK_layout_enum.mma_major_mode()
            dV_major_mode = self.mdV_layout_enum.mma_major_mode()
            if const_expr(dK_major_mode != tcgen05.OperandMajorMode.K):
                raise RuntimeError("The layout of mdK is wrong")
            if const_expr(dV_major_mode != tcgen05.OperandMajorMode.K):
                raise RuntimeError("The layout of mdV is wrong")

        if const_expr(self.use_tma_store and not self.dKV_postprocess):
            tma_copy_op_dKV = cpasync.CopyBulkTensorTileS2GOp()
            tma_atom_dK, mdK_tma_tensor = cpasync.make_tiled_tma_atom(
                tma_copy_op_dKV,
                mdK,
                cute.select(self.sdK_layout, mode=[0, 1]),
                self.sdK_epi_tile,
                1,  # no mcast
            )
            tma_atom_dV, mdV_tma_tensor = cpasync.make_tiled_tma_atom(
                tma_copy_op_dKV,
                mdV,
                cute.select(self.sdV_layout, mode=[0, 1]),
                self.sdV_epi_tile,
                1,  # no mcast
            )
        else:
            mdV_tma_tensor = mdV
            mdK_tma_tensor = mdK
            tma_atom_dV = None
            tma_atom_dK = None

        if const_expr(not self.dKV_postprocess):
            thr_layout_r2s_dKV = cute.make_ordered_layout(
                (self.epi_threads_r2s, 1), order=(1, 0)
            )  # 128 or 64 threads (see epi_num_strips)
            val_layout_r2s_dKV = cute.make_ordered_layout(
                (1, 128 // self.dk_dtype.width), order=(1, 0)
            )  # 4 or 8 vals for 16 byte store
            copy_atom_r2s_dKV = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                self.dk_dtype,
                num_bits_per_copy=128,
            )
            tiled_copy_r2s_dKV = cute.make_tiled_copy_tv(
                copy_atom_r2s_dKV, thr_layout_r2s_dKV, val_layout_r2s_dKV
            )
        else:
            tiled_copy_r2s_dKV = copy_utils.tiled_copy_1d(
                Float32, 128, num_copy_elems=128 // Float32.width
            )

        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(self.cta_group)
        tma_load_op_multicast = cpasync.CopyBulkTensorTileG2SMulticastOp(self.cta_group)

        # S.T = K @ Q.T
        tma_atom_K, tma_tensor_K = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            mK,
            cute.select(self.sK_layout, mode=[0, 1, 2]),
            self.mma_tiler_kq,
            self.tiled_mma_S,
            self.cluster_layout_vmnk.shape,
        )
        Q_tma_op = sm100_utils_basic.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mnk, self.tiled_mma_S.thr_id
        )
        tma_atom_Q, tma_tensor_Q = cute.nvgpu.make_tiled_tma_atom_B(
            # tma_load_op if const_expr(self.cluster_shape_mnk[0] == 1) else tma_load_op_multicast,
            Q_tma_op,
            mQ,
            cute.select(self.sQ_layout, mode=[0, 1, 2]),
            self.mma_tiler_kq,
            self.tiled_mma_S,
            self.cluster_layout_vmnk.shape,
        )
        # dP.T = V @ dO.T
        tma_atom_V, tma_tensor_V = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            mV,
            cute.select(self.sV_layout, mode=[0, 1, 2]),
            self.mma_tiler_vdo,
            self.tiled_mma_dP,
            self.cluster_layout_vmnk.shape,
        )
        dO_tma_op = sm100_utils_basic.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mnk, self.tiled_mma_dV.thr_id
        )
        tma_atom_dO, tma_tensor_dO = cute.nvgpu.make_tiled_tma_atom_B(
            # tma_load_op if const_expr(self.cluster_shape_mnk[0] == 1) else tma_load_op_multicast,
            dO_tma_op,
            mdO,
            cute.select(self.sdO_layout, mode=[0, 1, 2]),
            self.mma_tiler_pdo,
            self.tiled_mma_dV,
            self.cluster_layout_vmnk.shape,
        )

        # ------------------------------------------------------------
        # 2-CTA
        # ------------------------------------------------------------
        tma_atom_dOt = tma_tensor_dOt = None
        if const_expr(self.use_2cta_instrs):
            tma_atom_dOt, tma_tensor_dOt = cute.nvgpu.make_tiled_tma_atom_B(
                dO_tma_op,
                layout_utils.select(mdO, mode=transpose_sh_q),
                cute.select(self.sdOt_layout, mode=[0, 1, 2]),
                self.mma_tiler_vdo,
                self.tiled_mma_dP,
                self.cluster_layout_vmnk.shape,
            )
        tma_atom_Qt = tma_tensor_Qt = None
        if const_expr(self.use_2cta_instrs):
            tma_atom_Qt, tma_tensor_Qt = cute.nvgpu.make_tiled_tma_atom_B(
                Q_tma_op,
                layout_utils.select(mQ, mode=transpose_sh_q),
                cute.select(self.sQt_layout, mode=[0, 1, 2]),
                self.mma_tiler_dsq,
                self.tiled_mma_dK,
                self.cluster_layout_vmnk.shape,
            )
        tma_atom_Kt = tma_tensor_Kt = None
        if const_expr(self.use_2cta_instrs):
            Kt_tma_op = sm100_utils_basic.cluster_shape_to_tma_atom_B(
                self.cluster_shape_mnk, self.tiled_mma_dQ.thr_id
            )
            tma_atom_Kt, tma_tensor_Kt = cute.nvgpu.make_tiled_tma_atom_B(
                Kt_tma_op,
                layout_utils.select(mK, mode=transpose_sh_k),
                cute.select(self.sKt_layout, mode=[0, 1, 2]),
                self.mma_tiler_dsk,
                self.tiled_mma_dQ,
                self.cluster_layout_vmnk.shape,
            )

        self.tma_copy_bytes = {
            name: self.cta_group_size
            * cute.size_in_bytes(mX.element_type, cute.select(layout, mode=[0, 1, 2]))
            for name, mX, layout in [
                ("Q", mQ, self.sQ_layout),
                ("K", mK, self.sK_layout),
                ("V", mV, self.sV_layout),
                ("dO", mdO, self.sdO_layout),
            ]
        }
        self.tma_copy_bytes["LSE"] = self.tile_m * Float32.width // 8
        self.tma_copy_bytes["dPsum"] = self.tile_m * Float32.width // 8
        self.tma_copy_bytes["dQ"] = self.tile_m * self.dQ_reduce_ncol * Float32.width // 8
        self.tma_copy_bytes["dKacc"] = self.tile_n * self.dK_reduce_ncol * Float32.width // 8
        # The dKV epilogue's own reduce panel: with a folded accumulator one panel
        # spans both hdim strips (2 * tile_n rows worth of lanes), so it is twice
        # tma_copy_bytes["dKacc"].
        self.dKV_reduce_bytes = self.dKV_reduce_panel * Float32.width // 8
        self.tma_copy_bytes["dS"] = cute.size_in_bytes(self.ds_dtype, self.sdS_layout)
        self.tma_copy_bytes["sdS_xchg"] = self.tma_copy_bytes["dS"] // 2  # Half of dS for exchange

        # TileScheduler = SingleTileScheduler
        if const_expr(self.deterministic):
            TileScheduler = SingleTileLPTBwdScheduler
        elif const_expr(self.is_persistent):
            # Pair-based (cluster-aware) persistent scheduling. The plain
            # StaticPersistentTileScheduler cannot be used here: it would split a CTA pair
            # across (head, batch) whenever num_block is odd.
            TileScheduler = StaticPersistentClusterTileScheduler
        else:
            TileScheduler = SingleTileScheduler
        # spt is disabled for 2-CTA temporarily
        self.spt = (
            self.is_causal and self.deterministic
        )
        tile_sched_args = TileSchedulerArguments(
            cute.ceil_div(cute.size(mK.shape[0]), self.cta_tiler[0]),
            cute.size(mQ.shape[2]),  # num_heads = num_query_heads
            cute.size(mK.shape[3]),
            1,  # num_splits
            cute.size(mQ.shape[0]),  # pass seqlen_q for seqlen_k
            mQ.shape[1],
            mV.shape[1],
            total_q=cute.size(mQ.shape[0]),
            tile_shape_mn=self.cta_tiler[:2],
            cluster_shape_mn=self.cluster_shape_mnk[:2],
            qhead_per_kvhead_packgqa=1,
            element_size=self.k_dtype.width // 8,
            is_persistent=self.is_persistent,
            lpt=self.spt,
        )

        tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
        self.tile_scheduler_cls = TileScheduler
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)

        # Compute allocation sizes for shared buffers that are reused
        # sQ is reused for sdK, sdO is reused for sdV
        sQ_alloc_bytes = max(
            cute.size_in_bytes(self.q_dtype, self.sQ_layout),
            cute.size_in_bytes(self.dk_dtype, self.sdK_layout),
        )
        sdO_alloc_bytes = max(
            cute.size_in_bytes(self.dv_dtype, self.sdV_layout),
            cute.size_in_bytes(self.do_dtype, self.sdO_layout),
        )
        # Sanity check that layouts fit in allocation
        sdV_bytes = cute.size_in_bytes(self.dv_dtype, self.sdV_layout)
        sdK_bytes = cute.size_in_bytes(self.dk_dtype, self.sdK_layout)
        assert sdV_bytes <= sdO_alloc_bytes, "sdV doesn't fit in sdO storage allocation"
        assert sdK_bytes <= sQ_alloc_bytes, "sdK doesn't fit in sQ storage allocation"
        # 2-CTA: sdV reuses sV, sdK reuses sK
        sV_bytes = cute.size_in_bytes(self.v_dtype, self.sV_layout)
        sK_bytes = cute.size_in_bytes(self.k_dtype, self.sK_layout)
        if const_expr(self.use_2cta_instrs):
            assert sdV_bytes <= sV_bytes, "sdV doesn't fit in sV storage allocation (2-CTA)"
            assert sdK_bytes <= sK_bytes, "sdK doesn't fit in sK storage allocation (2-CTA)"

        if const_expr(self.use_2cta_instrs):
            sQt_size = cute.cosize(self.sQt_layout) if const_expr(not self.use_2cta_bigd) else 0
            sdOt_size = cute.cosize(self.sdOt_layout) if const_expr(not self.use_2cta_bigd) else 0
            sdS_xchg_size = cute.cosize(self.sdS_xchg_layout) if const_expr(not self.use_2cta_bigd) else 0
            # Folded S/dP cannot back the dV MMA's A operand, so P lives in SMEM in
            # exactly the layout that MMA wants (see mma_P_from_smem). dS already has
            # sdS for the same reason on the dK side.
            sP_size = cute.cosize(self.tP_layout) if const_expr(self.mma_P_from_smem) else 0
            # Separate buffer for the dQ MMA's A view (see separate_sdS_buffers).
            sdS_dq_size = (
                cute.cosize(self.sdS_layout) if const_expr(self.separate_sdS_buffers) else 0
            )

            @cute.struct
            class SharedStorage:
                Q_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.Q_stage]
                dO_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.dO_stage]
                LSE_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.Q_stage]
                dPsum_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.dO_stage]
                S_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.single_stage]
                dP_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.single_stage]
                dS_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.single_stage]
                dKV_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.sdKVaccum_stage]
                dQ_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
                dQ_cluster_full_mbar_ptr: cute.struct.MemRange[
                    cutlass.Int64, self.dQaccum_reduce_stage // 2
                ]
                dQ_cluster_empty_mbar_ptr: cute.struct.MemRange[
                    cutlass.Int64, self.dQaccum_reduce_stage // 2
                ]
                tmem_holding_buf: Int32
                tmem_dealloc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 1]
                flashmask_loaded_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 1]
                sFM_max_min_ptr: cute.struct.MemRange[cutlass.Int32, 8]

                # 2-CTA
                Qt_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.Q_stage]
                Kt_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.single_stage]
                dS_cluster_empty_mbar_ptr: cutlass.Int64
                dS_cluster_full_mbar_ptr: cutlass.Int64
                dS_cluster_leader_mbar_ptr: cutlass.Int64
                tmem_cluster_mbar_ptr: cutlass.Int64
                dQaccum_empty_mbar_ptr: cutlass.Int64

                sQ: cute.struct.Align[
                    cute.struct.MemRange[self.q_dtype, cute.cosize(self.sQ_layout)],
                    self.buffer_align_bytes,
                ]
                sK: cute.struct.Align[
                    cute.struct.MemRange[self.k_dtype, cute.cosize(self.sK_layout)],
                    self.buffer_align_bytes,
                ]
                sV: cute.struct.Align[
                    cute.struct.MemRange[self.v_dtype, cute.cosize(self.sV_layout)],
                    self.buffer_align_bytes,
                ]
                sdO: cute.struct.Align[
                    cute.struct.MemRange[self.do_dtype, cute.cosize(self.sdO_layout)],
                    self.buffer_align_bytes,
                ]
                sQt: cute.struct.Align[
                    cute.struct.MemRange[self.q_dtype, sQt_size],
                    self.buffer_align_bytes,
                ]
                sdOt: cute.struct.Align[
                    cute.struct.MemRange[self.do_dtype, sdOt_size],
                    self.buffer_align_bytes,
                ]
                sdS_xchg: cute.struct.Align[
                    cute.struct.MemRange[self.ds_dtype, sdS_xchg_size],
                    self.buffer_align_bytes,
                ]
                sKt: cute.struct.Align[
                    cute.struct.MemRange[self.k_dtype, cute.cosize(self.sKt_layout)],
                    self.buffer_align_bytes,
                ]
                sdS: cute.struct.Align[
                    cute.struct.MemRange[self.ds_dtype, cute.cosize(self.sdSt_layout)],
                    self.buffer_align_bytes,
                ]
                sdS_dq: cute.struct.Align[
                    cute.struct.MemRange[self.ds_dtype, sdS_dq_size],
                    self.buffer_align_bytes,
                ]
                sP: cute.struct.Align[
                    cute.struct.MemRange[self.do_dtype, sP_size],
                    self.buffer_align_bytes,
                ]
                sLSE: cute.struct.Align[
                    cute.struct.MemRange[self.lse_dtype, cute.cosize(self.sLSE_layout)],
                    128,
                ]
                sdPsum: cute.struct.Align[
                    cute.struct.MemRange[self.dpsum_dtype, cute.cosize(self.sdPsum_layout)],
                    128,
                ]
                sdQaccum: cute.struct.Align[
                    cute.struct.MemRange[self.dqaccum_dtype, cute.cosize(self.sdQaccum_layout)],
                    self.buffer_align_bytes if sdS_xchg_size == 0 else 128,
                ]
                sStartEndRowIndices: cute.struct.Align[
                    cute.struct.MemRange[self.startend_row_indices_dtype, cute.cosize(self.sStartEndRowIndices_layout)],
                    64,
                ]
        else:

            @cute.struct
            class SharedStorage:
                Q_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.Q_stage]
                dO_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.dO_stage]
                LSE_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.Q_stage]
                dPsum_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.dO_stage]
                S_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.single_stage]
                dP_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.single_stage]
                dS_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.single_stage]
                dKV_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.sdKVaccum_stage]
                dQ_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
                dQ_cluster_full_mbar_ptr: cute.struct.MemRange[
                    cutlass.Int64, self.dQaccum_reduce_stage // 2
                ]
                dQ_cluster_empty_mbar_ptr: cute.struct.MemRange[
                    cutlass.Int64, self.dQaccum_reduce_stage // 2
                ]
                tmem_holding_buf: Int32
                tmem_dealloc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 1]
                flashmask_loaded_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 1]
                sFM_max_min_ptr: cute.struct.MemRange[cutlass.Int32, 8]

                sdPsum: cute.struct.Align[
                    cute.struct.MemRange[self.dpsum_dtype, cute.cosize(self.sdPsum_layout)],
                    128,
                ]

                # Smem tensors
                # sQ is reused for sdK which in the non-MHA case needs float32
                sQ: cute.struct.Align[
                    cute.struct.MemRange[cute.Uint8, sQ_alloc_bytes],
                    self.buffer_align_bytes,
                ]
                sK: cute.struct.Align[
                    cute.struct.MemRange[self.k_dtype, cute.cosize(self.sK_layout)],
                    self.buffer_align_bytes,
                ]
                sV: cute.struct.Align[
                    cute.struct.MemRange[self.v_dtype, cute.cosize(self.sV_layout)],
                    self.buffer_align_bytes,
                ]
                # sdO is reused for sdV which in the non-MHA case needs float32
                sdO: cute.struct.Align[
                    cute.struct.MemRange[cute.Uint8, sdO_alloc_bytes],
                    self.buffer_align_bytes,
                ]
                sdQaccum: cute.struct.Align[
                    cute.struct.MemRange[self.dqaccum_dtype, cute.cosize(self.sdQaccum_layout)],
                    self.buffer_align_bytes,
                ]
                sdS: cute.struct.Align[
                    cute.struct.MemRange[self.ds_dtype, cute.cosize(self.sdSt_layout)],
                    128,
                ]
                sLSE: cute.struct.Align[
                    cute.struct.MemRange[self.lse_dtype, cute.cosize(self.sLSE_layout)],
                    128,
                ]
                sStartEndRowIndices: cute.struct.Align[
                    cute.struct.MemRange[self.startend_row_indices_dtype, cute.cosize(self.sStartEndRowIndices_layout)],
                    64,
                ]

        self.shared_storage = SharedStorage
        # Overshooting the SM100 dynamic SMEM cap only shows up as a
        # CUDA_ERROR_INVALID_VALUE at launch, so check the real struct here.
        smem_bytes = SharedStorage.size_in_bytes()
        assert smem_bytes <= SM100_SMEM_CAPACITY_BYTES, (
            f"shared storage is {smem_bytes} B, over the {SM100_SMEM_CAPACITY_BYTES} B "
            f"SM100 cap (d={self.tile_hdim}, dv={self.tile_hdimv}, "
            f"tile_m={self.tile_m}, tile_n={self.tile_n}, "
            f"2cta={self.use_2cta_instrs}, folded={self.folded_kv_acc})"
        )

        LOG2_E = math.log2(math.e)
        softmax_scale_log2 = softmax_scale * LOG2_E

        self.kernel(
            tma_tensor_Q,
            tma_tensor_Qt,
            tma_tensor_K,
            tma_tensor_Kt,
            tma_tensor_V,
            mLSE,
            mdPsum,
            tma_tensor_dO,
            tma_tensor_dOt,
            mdV,
            mdK,
            mdQaccum,
            mdV_tma_tensor,
            mdK_tma_tensor,
            mdQ_semaphore,
            mdK_semaphore,
            mdV_semaphore,
            tma_atom_Q,
            tma_atom_Qt,
            tma_atom_K,
            tma_atom_Kt,
            tma_atom_V,
            tma_atom_dO,
            tma_atom_dOt,
            tma_atom_dV,
            tma_atom_dK,
            flashmask_info,
            overlap_work_done_addr,
            overlap_spin_timeout_ns,
            overlap_segment_idx,
            overlap_local_kv_size,
            overlap_comm_rpb,
            self.sQ_layout,
            self.sQt_layout,
            self.sK_layout,
            self.sKt_layout,
            self.sV_layout,
            self.sLSE_layout,
            self.sdPsum_layout,
            self.sdO_layout,
            self.sdOt_layout,
            self.sdSt_layout,
            self.sdS_layout,
            self.sdS_xchg_layout,
            self.sdQaccum_layout,
            self.sdK_layout,
            self.sdV_layout,
            self.tP_layout,
            self.tdS_layout,
            self.sStartEndRowIndices_layout,
            self.tiled_mma_S,
            self.tiled_mma_dP,
            self.tiled_mma_dV,
            self.tiled_mma_dK,
            self.tiled_mma_dQ,
            tiled_copy_r2s_dKV,
            softmax_scale,
            softmax_scale_log2,
            tile_sched_params,
        ).launch(
            grid=grid_dim,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk if cute.size(self.cluster_shape_mnk) > 1 else None,
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mQt: Optional[cute.Tensor],
        mK: cute.Tensor,
        mKt: Optional[cute.Tensor],
        mV: cute.Tensor,
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        mdO: cute.Tensor,
        mdOt: Optional[cute.Tensor],
        mdV: cute.Tensor,
        mdK: cute.Tensor,
        mdQaccum: cute.Tensor,
        mdV_tma_tensor: Optional[cute.Tensor],
        mdK_tma_tensor: Optional[cute.Tensor],
        mdQ_semaphore: Optional[cute.Tensor],
        mdK_semaphore: Optional[cute.Tensor],
        mdV_semaphore: Optional[cute.Tensor],
        tma_atom_Q: cute.CopyAtom,
        tma_atom_Qt: Optional[cute.CopyAtom],
        tma_atom_K: cute.CopyAtom,
        tma_atom_Kt: Optional[cute.CopyAtom],
        tma_atom_V: cute.CopyAtom,
        tma_atom_dO: cute.CopyAtom,
        tma_atom_dOt: Optional[cute.CopyAtom],
        tma_atom_dV: Optional[cute.CopyAtom],
        tma_atom_dK: Optional[cute.CopyAtom],
        flashmask_info: Optional[FlashMaskInfo],
        overlap_work_done_addr: Optional[cutlass.Int64],
        overlap_spin_timeout_ns: Optional[cutlass.Int64],
        overlap_segment_idx: Optional[cutlass.Int32],
        overlap_local_kv_size: Optional[cutlass.Int32],
        overlap_comm_rpb: cutlass.Constexpr,
        sQ_layout: cute.ComposedLayout,
        sQt_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sKt_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sLSE_layout: cute.Layout,
        sdPsum_layout: cute.Layout,
        sdO_layout: cute.ComposedLayout,
        sdOt_layout: cute.ComposedLayout,
        sdSt_layout: cute.ComposedLayout,
        sdS_layout: cute.ComposedLayout,
        sdS_xchg_layout: cute.Layout,
        sdQaccum_layout: cute.Layout,
        sdK_layout: cute.ComposedLayout | cute.Layout,
        sdV_layout: cute.ComposedLayout | cute.Layout,
        tP_layout: cute.ComposedLayout,
        tdS_layout: cute.ComposedLayout,
        sStartEndRowIndices_layout: cute.Layout,
        tiled_mma_S: cute.TiledMma,
        tiled_mma_dP: cute.TiledMma,
        tiled_mma_dV: cute.TiledMma,
        tiled_mma_dK: cute.TiledMma,
        tiled_mma_dQ: cute.TiledMma,
        tiled_copy_r2s_dKV: cute.TiledCopy,
        softmax_scale: cutlass.Float32,
        softmax_scale_log2: cutlass.Float32,
        tile_sched_params: ParamsBase,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        bidx, _, _ = cute.arch.block_idx()
        mma_tile_coord_v = bidx % self.cta_group_size
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())

        # Prefetch tma descriptor
        if warp_idx == self.load_warp_id:
            with cute.arch.elect_one():
                cpasync.prefetch_descriptor(tma_atom_Q)
                cpasync.prefetch_descriptor(tma_atom_K)
                cpasync.prefetch_descriptor(tma_atom_V)
                cpasync.prefetch_descriptor(tma_atom_dO)
                if const_expr(tma_atom_dV is not None):
                    cpasync.prefetch_descriptor(tma_atom_dV)
                if const_expr(tma_atom_dK is not None):
                    cpasync.prefetch_descriptor(tma_atom_dK)
                if const_expr(tma_atom_Qt is not None):
                    cpasync.prefetch_descriptor(tma_atom_Qt)
                if const_expr(tma_atom_Kt is not None):
                    cpasync.prefetch_descriptor(tma_atom_Kt)
                if const_expr(tma_atom_dOt is not None):
                    cpasync.prefetch_descriptor(tma_atom_dOt)

        cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (tiled_mma_S.thr_id.shape,),
        )

        # Alloc
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        tmem_dealloc_mbar_ptr = storage.tmem_dealloc_mbar_ptr.data_ptr()
        flashmask_loaded_mbar_ptr = storage.flashmask_loaded_mbar_ptr.data_ptr()
        dQ_cluster_full_mbar_ptr = storage.dQ_cluster_full_mbar_ptr.data_ptr()
        dQ_cluster_empty_mbar_ptr = storage.dQ_cluster_empty_mbar_ptr.data_ptr()

        if const_expr(self.use_2cta_instrs):
            dS_cluster_full_mbar_ptr = storage.dS_cluster_full_mbar_ptr
            dS_cluster_empty_mbar_ptr = storage.dS_cluster_empty_mbar_ptr
            dS_cluster_leader_mbar_ptr = storage.dS_cluster_leader_mbar_ptr
            tmem_cluster_mbar_ptr = storage.tmem_cluster_mbar_ptr
            dQaccum_empty_mbar_ptr = storage.dQaccum_empty_mbar_ptr
        else:
            dS_cluster_full_mbar_ptr = None
            dS_cluster_empty_mbar_ptr = None
            dS_cluster_leader_mbar_ptr = None
            tmem_cluster_mbar_ptr = None
            dQaccum_empty_mbar_ptr = None

        # Barrier initialization
        if warp_idx == 1:
            cute.arch.mbarrier_init(
                tmem_dealloc_mbar_ptr,
                cute.arch.WARP_SIZE
                * (len(self.compute_warp_ids) + len(self.reduce_warp_ids)),
            )
            cute.arch.mbarrier_init(
                flashmask_loaded_mbar_ptr, cute.arch.WARP_SIZE
            )
        if const_expr(self.use_2cta_instrs):
            if warp_idx == 1:
                cute.arch.mbarrier_init(
                    tmem_cluster_mbar_ptr, cute.arch.WARP_SIZE * len([self.mma_warp_id])
                )
            if const_expr(self.use_2cta_bigd):
                if warp_idx == 2:
                    cute.arch.mbarrier_init(
                        dQaccum_empty_mbar_ptr,
                        len(self.reduce_warp_ids),
                    )
            if warp_idx == 4:
                cute.arch.mbarrier_init(dS_cluster_full_mbar_ptr, 1)
                cute.arch.mbarrier_init(dS_cluster_empty_mbar_ptr, 1)
                cute.arch.mbarrier_init(dS_cluster_leader_mbar_ptr, 2)

        if const_expr(self.cluster_reduce_dQ):
            if warp_idx == 4:
                for i in range(self.dQaccum_reduce_stage // 2):
                    cute.arch.mbarrier_init(dQ_cluster_full_mbar_ptr + i, 1)
                    cute.arch.mbarrier_init(dQ_cluster_empty_mbar_ptr + i, 1)

        # UMMA producers and AsyncThread consumers
        pipeline_producer_group_MMA_AsyncThread = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread, len([self.mma_warp_id])
        )
        # Only 1 thread per warp will signal
        pipeline_consumer_group_MMA_AsyncThread = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread, len(self.compute_warp_ids) * self.cta_group_size
        )
        pipeline_S_P = cutlass.pipeline.PipelineUmmaAsync.create(
            num_stages=1,
            producer_group=pipeline_producer_group_MMA_AsyncThread,
            consumer_group=pipeline_consumer_group_MMA_AsyncThread,
            barrier_storage=storage.S_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
        )
        pipeline_dP = cutlass.pipeline.PipelineUmmaAsync.create(
            num_stages=1,
            producer_group=pipeline_producer_group_MMA_AsyncThread,
            consumer_group=pipeline_consumer_group_MMA_AsyncThread,
            barrier_storage=storage.dP_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
        )
        pipeline_dKV = cutlass.pipeline.PipelineUmmaAsync.create(
            num_stages=2,
            producer_group=pipeline_producer_group_MMA_AsyncThread,
            consumer_group=pipeline_consumer_group_MMA_AsyncThread,
            barrier_storage=storage.dKV_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
        )
        pipeline_consumer_group_MMA_AsyncThread_dQ = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread,
            len(self.reduce_warp_ids) * self.cta_group_size,
        )  # Compute
        pipeline_dQ = cutlass.pipeline.PipelineUmmaAsync.create(
            num_stages=1,
            producer_group=pipeline_producer_group_MMA_AsyncThread,
            consumer_group=pipeline_consumer_group_MMA_AsyncThread_dQ,
            barrier_storage=storage.dQ_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
        )

        # AsyncThread producers and UMMA consumers
        # Only 1 thread per warp will signal
        pipeline_PdS_producer_group = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread,
            len(self.compute_warp_ids) * self.cta_group_size,
        )  # Compute
        pipeline_PdS_consumer_group = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread, len([self.mma_warp_id])
        )  # MMA
        pipeline_dS = cutlass.pipeline.PipelineAsyncUmma.create(
            num_stages=1,
            producer_group=pipeline_PdS_producer_group,
            consumer_group=pipeline_PdS_consumer_group,
            barrier_storage=storage.dS_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
        )

        # TMA producer and UMMA consumers
        pipeline_producer_group = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread, len([self.load_warp_id])
        )
        # The arrive count is the number of mcast size
        pipeline_consumer_group = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread, len([self.mma_warp_id]) * self.num_mcast_ctas_b
        )
        pipeline_consumer_group_compute = cutlass.pipeline.CooperativeGroup(
            # cutlass.pipeline.Agent.Thread, len(self.compute_warp_ids) * self.num_mcast_ctas_b
            cutlass.pipeline.Agent.Thread,
            len(self.compute_warp_ids) * 1,
        )
        pipeline_LSE = cutlass.pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.LSE_mbar_ptr.data_ptr(),
            num_stages=self.Q_stage,
            producer_group=pipeline_producer_group,
            consumer_group=pipeline_consumer_group_compute,
            tx_count=self.tma_copy_bytes["LSE"],
            # cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        pipeline_dPsum = cutlass.pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.dPsum_mbar_ptr.data_ptr(),
            num_stages=self.dO_stage,
            producer_group=pipeline_producer_group,
            consumer_group=pipeline_consumer_group_compute,
            tx_count=self.tma_copy_bytes["dPsum"],
            # cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        pipeline_Q = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.Q_mbar_ptr.data_ptr(),
            num_stages=self.Q_stage,
            producer_group=pipeline_producer_group,
            consumer_group=pipeline_consumer_group,
            tx_count=self.tma_copy_bytes["Q"],
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        if const_expr(self.use_2cta_instrs):
            if const_expr(self.use_2cta_bigd):
                pipeline_Qt = pipeline_Q
            else:
                pipeline_Qt = pipeline.PipelineTmaUmma.create(
                    barrier_storage=storage.Qt_mbar_ptr.data_ptr(),
                    num_stages=self.Q_stage,
                    producer_group=pipeline_producer_group,
                    consumer_group=pipeline_consumer_group,
                    tx_count=self.tma_copy_bytes["Q"],
                    cta_layout_vmnk=cluster_layout_vmnk,
                    defer_sync=True,
                )
            pipeline_Kt = pipeline.PipelineTmaUmma.create(
                barrier_storage=storage.Kt_mbar_ptr.data_ptr(),
                num_stages=self.single_stage,
                producer_group=pipeline_producer_group,
                consumer_group=pipeline_consumer_group,
                tx_count=self.tma_copy_bytes["K"],
                cta_layout_vmnk=cluster_layout_vmnk,
                defer_sync=True,
            )
        else:
            pipeline_Qt = pipeline_Kt = pipeline_Q

        pipeline_dO = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.dO_mbar_ptr.data_ptr(),
            num_stages=self.dO_stage,
            producer_group=pipeline_producer_group,
            consumer_group=pipeline_consumer_group,
            tx_count=self.tma_copy_bytes["dO"],
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=False,
        )

        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner, dtype=self.q_dtype)
        if const_expr(self.use_2cta_instrs and not self.use_2cta_bigd):
            sQt = storage.sQt.get_tensor(
                sQt_layout.outer, swizzle=sQt_layout.inner, dtype=self.q_dtype
            )
        else:
            sQt = cute.make_tensor(
                cute.recast_ptr(sQ.iterator, sQt_layout.inner, dtype=self.q_dtype), sQt_layout.outer
            )
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        if const_expr(self.use_2cta_instrs):
            sKt = storage.sKt.get_tensor(sKt_layout.outer, swizzle=sKt_layout.inner)
        else:
            sKt = cute.make_tensor(cute.recast_ptr(sK.iterator, sKt_layout.inner), sKt_layout.outer)
        sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
        sdSt = storage.sdS.get_tensor(sdSt_layout.outer, swizzle=sdSt_layout.inner)
        if const_expr(self.separate_sdS_buffers):
            sdS = storage.sdS_dq.get_tensor(sdS_layout.outer, swizzle=sdS_layout.inner)
        else:
            sdS = cute.make_tensor(cute.recast_ptr(sdSt.iterator, sdS_layout.inner), sdS_layout.outer)

        if const_expr(self.use_2cta_instrs):
            if const_expr(not self.use_2cta_bigd):
                sdS_xchg = storage.sdS_xchg.get_tensor(sdS_xchg_layout)
            else:
                sdS_xchg = storage.sdQaccum.get_tensor(sdS_xchg_layout, dtype=self.ds_dtype)
        else:
            sdS_xchg = None

        sdO = storage.sdO.get_tensor(sdO_layout.outer, swizzle=sdO_layout.inner, dtype=self.do_dtype)
        if const_expr(self.use_2cta_instrs and not self.use_2cta_bigd):
            sdOt = storage.sdOt.get_tensor(
                sdOt_layout.outer, swizzle=sdOt_layout.inner, dtype=self.do_dtype
            )
        else:
            sdOt = cute.make_tensor(
                cute.recast_ptr(sdO.iterator, sdOt_layout.inner, dtype=self.do_dtype),
                sdOt_layout.outer,
            )

        sLSE = storage.sLSE.get_tensor(sLSE_layout)
        sdPsum = storage.sdPsum.get_tensor(sdPsum_layout)
        if const_expr(self.use_2cta_instrs):
            if const_expr(not self.dKV_postprocess):
                sdV = storage.sV.get_tensor(
                    sdV_layout.outer, swizzle=sdV_layout.inner, dtype=self.dv_dtype
                )
                sdK = storage.sK.get_tensor(
                    sdK_layout.outer, swizzle=sdK_layout.inner, dtype=self.dk_dtype
                )
            else:
                sdV = storage.sV.get_tensor(sdV_layout, dtype=self.dv_dtype)
                sdK = storage.sK.get_tensor(sdK_layout, dtype=self.dk_dtype)
        elif const_expr(not self.dKV_postprocess):
            sdV = storage.sdO.get_tensor(
                sdV_layout.outer, swizzle=sdV_layout.inner, dtype=self.dv_dtype
            )
            sdK = storage.sQ.get_tensor(
                sdK_layout.outer, swizzle=sdK_layout.inner, dtype=self.dk_dtype
            )
        else:
            sdV = storage.sdO.get_tensor(sdV_layout, dtype=self.dv_dtype)
            sdK = storage.sQ.get_tensor(sdK_layout, dtype=self.dk_dtype)

        # Buffer sizing is guaranteed by max(...) in SharedStorage declarations
        # for both sQ (reused as sdK) and sdO (reused as sdV)
        sdQaccum = storage.sdQaccum.get_tensor(sdQaccum_layout)
        sStartEndRowIndices = storage.sStartEndRowIndices.get_tensor(sStartEndRowIndices_layout)
        sFM_max_min = cute.make_tensor(storage.sFM_max_min_ptr.data_ptr(), cute.make_layout((cutlass.Int32(8)), stride=(cutlass.Int32(1))))

        # TMEM
        # This is a fake tensor, by right need to retrieve tmem_ptr. But we know that we always
        # request 512 columns of tmem, so we know that it starts at 0.
        tmem_ptr = cute.make_ptr(Float32, 0, mem_space=cute.AddressSpace.tmem, assumed_align=16)
        # S
        thr_mma_S = tiled_mma_S.get_slice(mma_tile_coord_v)
        Sacc_shape = thr_mma_S.partition_shape_C(self.mma_tiler_kq[:2])  # (M, N)
        tStS = thr_mma_S.make_fragment_C(Sacc_shape)
        # (MMA, MMA_M, MMA_N)
        tStS = cute.make_tensor(tmem_ptr + self.tmem_S_offset, tStS.layout)
        # dP
        thr_mma_dP = tiled_mma_dP.get_slice(mma_tile_coord_v)
        dPacc_shape = thr_mma_dP.partition_shape_C(self.mma_tiler_vdo[:2])
        tdPtdP = thr_mma_dP.make_fragment_C(dPacc_shape)
        tdPtdP = cute.make_tensor(tmem_ptr + self.tmem_dP_offset, tdPtdP.layout)
        # dV
        thr_mma_dV = tiled_mma_dV.get_slice(mma_tile_coord_v)
        dvacc_shape = thr_mma_dV.partition_shape_C(self.mma_tiler_pdo[:2])
        tdVtdV = thr_mma_dV.make_fragment_C(dvacc_shape)
        tdVtdV = cute.make_tensor(tmem_ptr + self.tmem_dV_offset, tdVtdV.layout)

        if const_expr(self.mma_P_from_smem):
            # Folded S accumulator: P is written to SMEM by the compute warp and read
            # from there by the dV MMA (see mma_P_from_smem).
            tP = storage.sP.get_tensor(
                tP_layout.outer, swizzle=tP_layout.inner, dtype=self.do_dtype
            )
        else:
            tP = cute.make_tensor(
                cute.recast_ptr(tmem_ptr + self.tmem_P_offset, dtype=self.do_dtype),
                tP_layout.outer,
            )
        # dK
        thr_mma_dK = tiled_mma_dK.get_slice(mma_tile_coord_v)
        dkacc_shape = thr_mma_dK.partition_shape_C(self.mma_tiler_dsq[:2])
        tdKtdK = thr_mma_dK.make_fragment_C(dkacc_shape)
        tdKtdK = cute.make_tensor(tmem_ptr + self.tmem_dK_offset, tdKtdK.layout)
        tdS = cute.make_tensor(
            cute.recast_ptr(tmem_ptr + self.tmem_dS_offset, dtype=self.ds_dtype), tdS_layout.outer
        )
        # dQ
        thr_mma_dQ = tiled_mma_dQ.get_slice(mma_tile_coord_v)
        dQacc_shape = thr_mma_dQ.partition_shape_C(self.mma_tiler_dsk[:2])
        tdQtdQ = thr_mma_dQ.make_fragment_C(dQacc_shape)
        tdQtdQ = cute.make_tensor(tmem_ptr + self.tmem_dQ_offset, tdQtdQ.layout)

        block_info = BlockInfo(
            self.tile_m,
            # self.tile_n,
            self.tile_n * self.cluster_shape_mnk[0],  # careful, this case is not very well-tested
            self.is_causal,
            self.is_local,
            False,  # is_split_kv
            None,
            None,
            qhead_per_kvhead_packgqa=1,
        )
        SeqlenInfoCls = partial(
            SeqlenInfoQK.create,
            seqlen_q_static=mQ.shape[0],
            seqlen_k_static=mK.shape[0],
        )
        TileSchedulerCls = partial(self.tile_scheduler_cls.create, tile_sched_params)

        # TODO: support local
        AttentionMaskCls = partial(
            AttentionMask,
            self.tile_m,
            self.tile_n * self.cta_group_size,
            swap_AB=True,
        )

        #  EMPTY
        # (15)
        if warp_idx == self.empty_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_empty)

        #  EPI / RELAY
        # (14)
        if warp_idx == self.relay_warp_id:
            if const_expr(self.use_2cta_instrs):
                cute.arch.setmaxregister_decrease(self.num_regs_mma)
                self.relay(
                    dS_cluster_full_mbar_ptr,
                    dS_cluster_empty_mbar_ptr,
                    dS_cluster_leader_mbar_ptr,
                    cluster_layout_vmnk,
                    block_info,
                    SeqlenInfoCls,
                    TileSchedulerCls,
                    flashmask_info,
                    sFM_max_min,
                    flashmask_loaded_mbar_ptr,
                )
            else:
                cute.arch.setmaxregister_decrease(self.num_regs_empty)

        #  LOAD
        # (13)
        if warp_idx == self.load_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_load)
            self.load(
                thr_mma_S,
                thr_mma_dP,
                thr_mma_dV,
                thr_mma_dK,
                thr_mma_dQ,
                mQ,
                mK,
                mKt,
                mV,
                mdO,
                mQt,
                mdOt,
                mLSE,
                mdPsum,
                sQ,
                sK,
                sKt,
                sV,
                sdO,
                sQt,
                sdOt,
                sLSE,
                sdPsum,
                tma_atom_Q,
                tma_atom_K,
                tma_atom_Kt,
                tma_atom_V,
                tma_atom_dO,
                tma_atom_Qt,
                tma_atom_dOt,
                pipeline_Q,
                pipeline_Qt,
                pipeline_Kt,
                pipeline_dO,
                pipeline_LSE,
                pipeline_dPsum,
                cluster_layout_vmnk,
                block_info,
                SeqlenInfoCls,
                TileSchedulerCls,
                flashmask_info,
                overlap_work_done_addr,
                overlap_spin_timeout_ns,
                overlap_segment_idx,
                overlap_local_kv_size,
                overlap_comm_rpb,
                sStartEndRowIndices,
                sFM_max_min,
                flashmask_loaded_mbar_ptr,
            )

        #  MMA
        # (12)
        if warp_idx == self.mma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_mma)

            # Alloc tmem buffer
            tmem_alloc_cols = Int32(self.tmem_alloc_cols)
            cute.arch.alloc_tmem(
                tmem_alloc_cols, storage.tmem_holding_buf, is_two_cta=self.use_2cta_instrs
            )
            cute.arch.sync_warp()

            self.mma(
                tiled_mma_S,
                tiled_mma_dP,
                tiled_mma_dV,
                tiled_mma_dK,
                tiled_mma_dQ,
                sQ,
                sQt,
                sK,
                sV,
                sdO,
                sdOt,
                sdSt,
                sdS,
                sKt,
                tP,
                tdS,
                tStS,
                tdPtdP,
                tdVtdV,
                tdKtdK,
                tdQtdQ,
                dS_cluster_full_mbar_ptr,
                dS_cluster_empty_mbar_ptr,
                dS_cluster_leader_mbar_ptr,
                dQaccum_empty_mbar_ptr,
                pipeline_Q,
                pipeline_Q.make_consumer(),
                pipeline_Qt,
                pipeline_Kt,
                pipeline_dO,
                pipeline_S_P,
                pipeline_dS,
                pipeline_dKV,
                pipeline_dP,
                pipeline_dQ,
                block_info,
                SeqlenInfoCls,
                TileSchedulerCls,
                flashmask_info,
                sFM_max_min,
                flashmask_loaded_mbar_ptr,
                is_leader_cta,
            )
            cute.arch.relinquish_tmem_alloc_permit(is_two_cta=self.use_2cta_instrs)
            tmem_ptr = cute.arch.retrieve_tmem_ptr(
                Float32, alignment=16, ptr_to_buffer_holding_addr=storage.tmem_holding_buf
            )
            cute.arch.mbarrier_wait(tmem_dealloc_mbar_ptr, 0)

            if const_expr(self.use_2cta_instrs):
                cute.arch.mbarrier_arrive(tmem_cluster_mbar_ptr, cta_rank_in_cluster ^ 1)
                cute.arch.mbarrier_wait(tmem_cluster_mbar_ptr, 0)

            tmem_alloc_cols = Int32(self.tmem_alloc_cols)
            cute.arch.dealloc_tmem(tmem_ptr, tmem_alloc_cols, is_two_cta=self.use_2cta_instrs)

        # Compute
        # (4, 5, 6, 7, 8, 9, 10, 11) --> 8 warps
        if warp_idx >= self.compute_warp_ids[0] and warp_idx <= self.compute_warp_ids[-1]:
            cute.arch.setmaxregister_increase(self.num_regs_compute)  # 8 warps
            self.compute_loop(
                thr_mma_S,
                thr_mma_dP,
                thr_mma_dV,
                thr_mma_dK,
                tStS,
                tdPtdP,
                tdVtdV,
                tdKtdK,
                sLSE,
                sdPsum,
                mdV,
                mdK,
                sdS,
                sdSt,
                tP,
                sdS_xchg,
                pipeline_LSE,
                pipeline_dPsum,
                pipeline_S_P,
                pipeline_dS,
                pipeline_dKV,
                pipeline_dP,
                dS_cluster_empty_mbar_ptr,
                dS_cluster_full_mbar_ptr,
                dQaccum_empty_mbar_ptr,
                softmax_scale,
                softmax_scale_log2,
                block_info,
                SeqlenInfoCls,
                AttentionMaskCls,
                TileSchedulerCls,
                sdV,
                sdK,
                mdV_tma_tensor,
                mdK_tma_tensor,
                tma_atom_dV,
                tma_atom_dK,
                tiled_copy_r2s_dKV,
                mdK_semaphore,
                mdV_semaphore,
                flashmask_info,
                sStartEndRowIndices,
                sFM_max_min,
                flashmask_loaded_mbar_ptr,
                is_leader_cta,
                sdS_layout.inner if const_expr(self.folded_kv_acc) else None,
                sdSt_layout.inner if const_expr(self.folded_kv_acc) else None,
                tP_layout.inner if const_expr(self.folded_kv_acc) else None,
            )
            cute.arch.mbarrier_arrive(tmem_dealloc_mbar_ptr)

        # Reduce
        # (0, 1, 2, 3) - dQ
        if warp_idx >= self.reduce_warp_ids[0] and warp_idx <= self.reduce_warp_ids[-1]:
            cute.arch.setmaxregister_increase(self.num_regs_reduce)
            self.dQacc_reduce(
                mdQaccum,
                sdQaccum,
                thr_mma_dQ,
                tdQtdQ,
                pipeline_dQ,
                dQaccum_empty_mbar_ptr,
                block_info,
                SeqlenInfoCls,
                TileSchedulerCls,
                mdQ_semaphore,
                None,
                flashmask_info,
                sFM_max_min,
                flashmask_loaded_mbar_ptr,
                is_leader_cta,
                mdK_semaphore,
                None,
            )
            # Reduce warp must also arrive on tmem_dealloc_mbar, otherwise the
            # MMA warp can dealloc TMEM while the reduce warp is still reading
            # dQ via T2R (races under GPU preemption).
            cute.arch.mbarrier_arrive(tmem_dealloc_mbar_ptr)

        return

    @cute.jit
    def relay(
        self,
        dS_cluster_full_mbar_ptr: cute.Pointer,
        dS_cluster_empty_mbar_ptr: cute.Pointer,
        dS_cluster_leader_mbar_ptr: cute.Pointer,
        cluster_layout_vmnk: cute.Layout,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        flashmask_info: Optional[FlashMaskInfo],
        sFM_max_min: Optional[cute.Tensor],
        flashmask_loaded_mbar_ptr: Optional[cute.Pointer],
    ):
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        dS_cluster_phase = Int32(0)
        if const_expr(self.enable_flashmask):
            flashmask_phase = Int32(0)

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            n_block, head_idx, batch_idx, _ = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            m_block_min, m_block_max = block_info.get_m_block_min_max(
                seqlen, n_block // self.cluster_shape_mnk[0]
            )
            head_idx_kv = head_idx // self.qhead_per_kvhead

            process_tile = m_block_min < m_block_max

            if process_tile:
                num_iters = m_block_max - m_block_min
                if const_expr(self.enable_flashmask):
                    # One relay per dS exchange, so the fully masked blocks the compute
                    # warp skips must be skipped here as well.
                    cute.arch.mbarrier_wait(flashmask_loaded_mbar_ptr, flashmask_phase)
                    num_iters = self.fm_skip_info(
                        flashmask_info, sFM_max_min, m_block_min, m_block_max
                    )[6]
                for _ in cutlass.range(num_iters, unroll=1):
                    # Wait for dS_xchg from peer CTA
                    cute.arch.mbarrier_wait(dS_cluster_full_mbar_ptr, phase=dS_cluster_phase)

                    # Arrive on MMA leader warp
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive(dS_cluster_leader_mbar_ptr, Int32(0))

                    dS_cluster_phase ^= 1

            if const_expr(self.enable_flashmask):
                flashmask_phase ^= 1

            if const_expr(self.tile_boundary_sync):
                # See tile_boundary_barrier.
                self.tile_boundary_barrier.arrive_and_wait()
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    @cute.jit
    def load(
        self,
        thr_mma_S: cute.ThrMma,
        thr_mma_dP: cute.ThrMma,
        thr_mma_dV: cute.ThrMma,
        thr_mma_dK: cute.ThrMma,
        thr_mma_dQ: cute.ThrMma,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mKt: Optional[cute.Tensor],
        mV: cute.Tensor,
        mdO: cute.Tensor,
        mQt: Optional[cute.Tensor],
        mdOt: Optional[cute.Tensor],
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sKt: cute.Tensor,
        sV: cute.Tensor,
        sdO: cute.Tensor,
        sQt: cute.Tensor,
        sdOt: cute.Tensor,
        sLSE: cute.Tensor,
        sdPsum: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_Kt: Optional[cute.CopyAtom],
        tma_atom_V: cute.CopyAtom,
        tma_atom_dO: cute.CopyAtom,
        tma_atom_Qt: Optional[cute.CopyAtom],
        tma_atom_dOt: Optional[cute.CopyAtom],
        pipeline_Q: PipelineAsync,
        pipeline_Qt: PipelineAsync,
        pipeline_Kt: PipelineAsync,
        pipeline_dO: PipelineAsync,
        pipeline_LSE: PipelineAsync,
        pipeline_dPsum: PipelineAsync,
        cluster_layout_vmnk: cute.Layout,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        flashmask_info: FlashMaskInfo,
        overlap_work_done_addr: Optional[cutlass.Int64],
        overlap_spin_timeout_ns: Optional[cutlass.Int64],
        overlap_segment_idx: Optional[cutlass.Int32],
        overlap_local_kv_size: Optional[cutlass.Int32],
        overlap_comm_rpb: cutlass.Constexpr,
        sStartEndRowIndices: cute.Tensor,
        sFM_max_min: cute.Tensor,
        flashmask_loaded_mbar_ptr: cute.Pointer,
        should_load_Q: bool = True,
        should_load_dO: bool = True,
    ):
        num_load_threads = cute.arch.WARP_SIZE
        tidx = cute.arch.thread_idx()[0] % num_load_threads
        if const_expr(overlap_work_done_addr is not None):
            work_done = cute.make_ptr(
                cutlass.Int32,
                overlap_work_done_addr,
                cute.AddressSpace.gmem,
                assumed_align=4,
            )

        producer_state_Q_LSE = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Producer, self.Q_stage
        )
        producer_state_Qt = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Producer, self.Q_stage
        )
        producer_state_Kt = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Producer, self.single_stage
        )
        producer_state_dO_dPsum = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Producer, self.dO_stage
        )
        # States used in the hdim192 path
        producer_state_Q_Qt = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Producer, self.Q_stage
        )
        producer_state_O_Ot = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Producer, self.dO_stage
        )
        producer_state_LSE = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Producer, self.Q_stage
        )
        producer_state_dPsum = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Producer, self.dO_stage
        )

        # Compute multicast mask for Q & dO buffer full
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)
        q_do_mcast_mask = None
        if const_expr(self.is_q_do_mcast):
            q_do_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
            )

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            n_block, head_idx, batch_idx, _ = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            head_idx_kv = head_idx // self.qhead_per_kvhead
            if const_expr(overlap_work_done_addr is not None):
                gate_batch_idx = batch_idx
                if const_expr(self.overlap_bhsd_layout):
                    gate_batch_idx = (
                        batch_idx * cute.size(mK.shape[2]) + head_idx_kv
                    )
                _overlap_gate_bwd(
                    n_block,
                    tidx,
                    seqlen.seqlen_k,
                    gate_batch_idx,
                    work_done,
                    overlap_spin_timeout_ns,
                    overlap_local_kv_size,
                    overlap_comm_rpb,
                    self.cta_group_size,
                    self.tile_n,
                )
            m_block_min, m_block_max = block_info.get_m_block_min_max(
                seqlen, n_block // self.cluster_shape_mnk[0]
            )
            n_block_cta_group = n_block // self.cta_group_size

            mQ_cur = mQ[None, None, head_idx, batch_idx]
            mK_cur = mK[None, None, head_idx_kv, batch_idx]
            mV_cur = mV[None, None, head_idx_kv, batch_idx]
            mdO_cur = mdO[None, None, head_idx, batch_idx]
            mLSE_cur = mLSE[None, head_idx, batch_idx]
            mPsum_cur = mdPsum[None, head_idx, batch_idx]

            if const_expr(self.use_2cta_instrs):
                mQt_cur = mQt[None, None, head_idx, batch_idx]
                mdOt_cur = mdOt[None, None, head_idx, batch_idx]
                mKt_cur = mKt[None, None, head_idx_kv, batch_idx]

            gK = cute.local_tile(mK_cur, cute.select(self.mma_tiler_kq, mode=[0, 2]), (n_block_cta_group, 0))
            tSgK = thr_mma_S.partition_A(gK)
            gV = cute.local_tile(mV_cur, cute.select(self.mma_tiler_vdo, mode=[0, 2]), (n_block_cta_group, 0))
            tdPgV = thr_mma_dP.partition_A(gV)
            gQ = cute.local_tile(mQ_cur, cute.select(self.mma_tiler_kq, mode=[1, 2]), (None, 0))
            tSgQ = thr_mma_S.partition_B(gQ)
            gLSE = cute.local_tile(mLSE_cur, (self.tile_m,), (None,))
            gdPsum = cute.local_tile(mPsum_cur, (self.tile_m,), (None,))
            gdO = cute.local_tile(mdO_cur, cute.select(self.mma_tiler_pdo, mode=[1, 2]), (0, None))
            tdPgdO = thr_mma_dV.partition_B(gdO)

            load_dOt = load_Qt = load_Kt = None

            a_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
            sK_stage0 = sK
            load_K, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_K,
                block_in_cluster_coord_vmnk[2],
                a_cta_layout,
                tSgK,
                sK_stage0,
                single_stage=True,
            )
            load_V, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_V,
                0,
                cute.make_layout(1),
                tdPgV,
                sV,
                single_stage=True,
            )
            b_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
            load_Q, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_Q,
                cta_coord=block_in_cluster_coord_vmnk[1],
                cta_layout=b_cta_layout,
                src_tensor=tSgQ,
                dst_tensor=sQ,
                mcast_mask=q_do_mcast_mask,
            )
            load_Q = copy_utils.tma_producer_copy_fn(load_Q, pipeline_Q)

            # (2) dP = V @ dO.T
            if const_expr(tma_atom_dOt is not None):
                gdOt = cute.local_tile(
                    mdOt_cur, cute.select(self.mma_tiler_vdo, mode=[1, 2]), (None, 0)
                )
                tdPgdOt = thr_mma_dP.partition_B(gdOt)
                load_dOt, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_dOt,
                    cta_coord=block_in_cluster_coord_vmnk[1],
                    cta_layout=b_cta_layout,
                    src_tensor=tdPgdOt,
                    dst_tensor=sdOt,
                    mcast_mask=q_do_mcast_mask,
                )
                load_dOt = copy_utils.tma_producer_copy_fn(load_dOt, pipeline_dO)

            # (3) dV += P.T @ dO
            load_dO, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_dO,
                cta_coord=block_in_cluster_coord_vmnk[1],
                cta_layout=b_cta_layout,
                src_tensor=tdPgdO,
                dst_tensor=sdO,
                mcast_mask=q_do_mcast_mask,
            )
            load_dO = copy_utils.tma_producer_copy_fn(load_dO, pipeline_dO)

            # (4) dK += dS.T @ Q (2-CTA: needs separate Qt load)
            if const_expr(tma_atom_Qt is not None):
                gQt = cute.local_tile(
                    mQt_cur, cute.select(self.mma_tiler_dsq, mode=[1, 2]), (0, None)
                )
                tdKgQt = thr_mma_dK.partition_B(gQt)
                load_Qt, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_Qt,
                    cta_coord=block_in_cluster_coord_vmnk[1],
                    cta_layout=b_cta_layout,
                    src_tensor=tdKgQt,
                    dst_tensor=sQt,
                    mcast_mask=q_do_mcast_mask,
                )
                load_Qt = copy_utils.tma_producer_copy_fn(load_Qt, pipeline_Qt)

            # (5) dQ = dS @ K
            if const_expr(self.use_2cta_instrs):
                gKt = cute.local_tile(
                    mKt_cur, cute.select(self.mma_tiler_dsk, mode=[1, 2]), (0, n_block_cta_group)
                )
                tdQgK = thr_mma_dQ.partition_B(gKt)
                load_Kt, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_Kt,
                    block_in_cluster_coord_vmnk[1],
                    b_cta_layout,
                    tdQgK,
                    sKt,
                    single_stage=True,
                )
            copy_atom_stats = cute.make_copy_atom(cpasync.CopyBulkG2SOp(), Float32)
            copy_stats = partial(cute.copy, copy_atom_stats)
            # copy_atom_stats = cute.make_copy_atom(cpasync.CopyBulkG2SMulticastOp(), Float32)
            # sLSE = cute.logical_divide(sLSE, (64,))[(None, block_in_cluster_coord_vmnk[1]), None]
            # gLSE = cute.logical_divide(gLSE, (64,))[(None, block_in_cluster_coord_vmnk[1]), None]
            # sdPsum = cute.logical_divide(sdPsum, (64,))[(None, block_in_cluster_coord_vmnk[1]), None]
            # gdPsum = cute.logical_divide(gdPsum, (64,))[(None, block_in_cluster_coord_vmnk[1]), None]
            # copy_stats = partial(cute.copy, copy_atom_stats, mcast_mask=q_do_mcast_mask)


            load_step_kwargs = dict(
                gLSE=gLSE,
                sLSE=sLSE,
                gdPsum=gdPsum,
                sdPsum=sdPsum,
                pipeline_Q=pipeline_Q,
                pipeline_LSE=pipeline_LSE,
                pipeline_dO=pipeline_dO,
                pipeline_dPsum=pipeline_dPsum,
                load_Q=load_Q,
                load_dO=load_dO,
                copy_stats=copy_stats,
                should_load_Q=should_load_Q,
                should_load_dO=should_load_dO,
                pipeline_Qt=pipeline_Qt,
                pipeline_Kt=pipeline_Kt,
                load_Qt=load_Qt,
                load_dOt=load_dOt,
            )
            load_step = partial(self.load_step, **load_step_kwargs)

            if const_expr(self.use_2cta_instrs):
                # The m blocks this CTA loads. With flashmask the fully masked ones drop
                # out; the bounds in sFM_max_min are the CTA pair's combined bounds (see
                # load_fm), so both CTAs of the cluster walk the same blocks and the
                # shared pipelines, the cta_group::2 MMAs and the multicast loads stay in
                # lockstep.
                num_m_iters = m_block_max - m_block_min
                fm_skip = None
                if const_expr(self.enable_flashmask):
                    self.load_fm(
                        flashmask_info,
                        sStartEndRowIndices,
                        sFM_max_min,
                        seqlen,
                        mQ.shape[2],
                        n_block,
                        head_idx,
                        batch_idx,
                        overlap_segment_idx,
                    )
                    cute.arch.mbarrier_arrive(flashmask_loaded_mbar_ptr)
                    fm_skip = self.fm_skip_info(
                        flashmask_info, sFM_max_min, m_block_min, m_block_max
                    )
                    num_m_iters = fm_skip[6]

                # fm_skip stays None without flashmask, which fm_m_block reads as "no
                # block is skipped". A closure would be easier to read, but the DSL
                # rejects closures that capture variables inside dynamic control flow.
                m_block_first = self.fm_m_block(fm_skip, m_block_min, Int32(0))


                if const_expr(self.use_2cta_bigd):
                    #### Prologue ####
                    assert should_load_Q and should_load_dO
                    # K & Q (for S)
                    pipeline_Q.producer_acquire(
                        producer_state_Q_Qt,
                        extra_tx_count=self.tma_copy_bytes["K"],
                    )
                    load_K(tma_bar_ptr=pipeline_Q.producer_get_barrier(producer_state_Q_Qt))
                    load_Q(m_block_first, producer_state=producer_state_Q_Qt)
                    pipeline_Q.producer_commit(producer_state_Q_Qt)
                    producer_state_Q_Qt.advance()
                    # LSE
                    pipeline_LSE.producer_acquire(producer_state_LSE)
                    with cute.arch.elect_one():
                        copy_stats(
                            gLSE[None, m_block_first],
                            sLSE[None, producer_state_LSE.index],
                            mbar_ptr=pipeline_LSE.producer_get_barrier(producer_state_LSE),
                        )
                    producer_state_LSE.advance()

                    # dOt + V, for dP.T = V @ dO.T
                    pipeline_dO.producer_acquire(
                        producer_state_O_Ot,
                        extra_tx_count=self.tma_copy_bytes["V"],
                    )
                    load_V(tma_bar_ptr=pipeline_dO.producer_get_barrier(producer_state_O_Ot))
                    load_dOt(m_block_first, producer_state=producer_state_O_Ot)
                    pipeline_dO.producer_commit(producer_state_O_Ot)
                    producer_state_O_Ot.advance()
                    # dPsum
                    pipeline_dPsum.producer_acquire(producer_state_dPsum)
                    with cute.arch.elect_one():
                        copy_stats(
                            gdPsum[None, m_block_first],
                            sdPsum[None, producer_state_dPsum.index],
                            mbar_ptr=pipeline_dPsum.producer_get_barrier(producer_state_dPsum),
                        )
                    producer_state_dPsum.advance()

                    # Qt, for dK = dS.T @ Q
                    pipeline_Qt.producer_acquire(
                        producer_state_Q_Qt,
                        extra_tx_count=self.tma_copy_bytes["K"],
                    )
                    load_Qt(m_block_first, producer_state=producer_state_Q_Qt)
                    load_Kt(tma_bar_ptr=pipeline_Qt.producer_get_barrier(producer_state_Q_Qt))
                    pipeline_Qt.producer_commit(producer_state_Q_Qt)
                    producer_state_Q_Qt.advance()

                    # dO, for dV = P.T @ dO
                    pipeline_dO.producer_acquire(producer_state_O_Ot)
                    load_dO(m_block_first, producer_state=producer_state_O_Ot)
                    pipeline_dO.producer_commit(producer_state_O_Ot)
                    producer_state_O_Ot.advance()

                    #### Mainloop ####
                    # 2CTA hdim192: [lse | Q | dOt | dPsum | Qt | dO]
                    for it in cutlass.range(1, num_m_iters, unroll=1):
                        m_block = self.fm_m_block(fm_skip, m_block_min, it)
                        # LSE
                        pipeline_LSE.producer_acquire(producer_state_LSE)
                        with cute.arch.elect_one():
                            copy_stats(
                                gLSE[None, m_block],
                                sLSE[None, producer_state_LSE.index],
                                mbar_ptr=pipeline_LSE.producer_get_barrier(producer_state_LSE),
                            )
                        producer_state_LSE.advance()

                        # Q
                        pipeline_Q.producer_acquire(producer_state_Q_Qt)
                        load_Q(m_block, producer_state=producer_state_Q_Qt)
                        pipeline_Q.producer_commit(producer_state_Q_Qt)
                        producer_state_Q_Qt.advance()

                        # dPsum
                        pipeline_dPsum.producer_acquire(producer_state_dPsum)
                        with cute.arch.elect_one():
                            copy_stats(
                                gdPsum[None, m_block],
                                sdPsum[None, producer_state_dPsum.index],
                                mbar_ptr=pipeline_dPsum.producer_get_barrier(
                                    producer_state_dPsum
                                ),
                            )
                        producer_state_dPsum.advance()

                        # dOt, for dP.T = V @ dO.T
                        pipeline_dO.producer_acquire(producer_state_O_Ot)
                        load_dOt(m_block, producer_state=producer_state_O_Ot)
                        pipeline_dO.producer_commit(producer_state_O_Ot)
                        producer_state_O_Ot.advance()

                        # Qt, for dK = dS.T @ Q
                        pipeline_Qt.producer_acquire(producer_state_Q_Qt)
                        load_Qt(m_block, producer_state=producer_state_Q_Qt)
                        pipeline_Qt.producer_commit(producer_state_Q_Qt)
                        producer_state_Q_Qt.advance()

                        # dO, for dV = P.T @ dO
                        pipeline_dO.producer_acquire(producer_state_O_Ot)
                        load_dO(m_block, producer_state=producer_state_O_Ot)
                        pipeline_dO.producer_commit(producer_state_O_Ot)
                        producer_state_O_Ot.advance()

                    #### Tail ####
                    if const_expr(not self.is_persistent):
                        # producer_tail is producer_acquire on the last stage, and for a TMA
                        # pipeline producer_acquire ALSO does the full barrier's
                        # mbarrier.arrive.expect_tx (see PipelineTmaAsync.producer_acquire).
                        # No TMA follows it, so the tail leaves the full barrier armed with
                        # a transaction count that never completes. Harmless when the CTA is
                        # about to exit; with the persistent scheduler the next tile's
                        # producer_acquire arms the same barrier a second time, which traps
                        # in the load warp (compute-sanitizer: "Unknown Error", whole warp,
                        # one PC) and leaves the compute warp waiting on LSE/dPsum forever.
                        # The tile boundary rendezvous already gives the ordering the tail
                        # was providing here.
                        pipeline_Q.producer_tail(producer_state_Q_Qt)
                        pipeline_LSE.producer_tail(producer_state_LSE)
                        pipeline_dO.producer_tail(producer_state_O_Ot)
                        pipeline_dPsum.producer_tail(producer_state_dPsum)
                else:
                    #### Prologue (2CTA hdim128) ####
                    # K & Q (for S)
                    pipeline_Q.producer_acquire(
                        producer_state_Q_LSE, extra_tx_count=self.tma_copy_bytes["K"]
                    )
                    load_K(tma_bar_ptr=pipeline_Q.producer_get_barrier(producer_state_Q_LSE))
                    load_Q(m_block_first, producer_state=producer_state_Q_LSE)
                    pipeline_Q.producer_commit(producer_state_Q_LSE)

                    # LSE
                    pipeline_LSE.producer_acquire(producer_state_Q_LSE)
                    with cute.arch.elect_one():
                        copy_stats(
                            gLSE[None, m_block_first],
                            sLSE[None, producer_state_Q_LSE.index],
                            mbar_ptr=pipeline_LSE.producer_get_barrier(producer_state_Q_LSE),
                        )
                    producer_state_Q_LSE.advance()

                    # V + dO + dOt (for dP and dV)
                    pipeline_dO.producer_acquire(
                        producer_state_dO_dPsum,
                        extra_tx_count=self.tma_copy_bytes["V"] + self.tma_copy_bytes["dO"]
                        if const_expr(tma_atom_dOt is not None)
                        else self.tma_copy_bytes["V"],
                    )
                    load_V(
                        tma_bar_ptr=pipeline_dO.producer_get_barrier(producer_state_dO_dPsum)
                    )
                    load_dO(m_block_first, producer_state=producer_state_dO_dPsum)
                    if const_expr(tma_atom_dOt is not None):
                        load_dOt(m_block_first, producer_state=producer_state_dO_dPsum)
                    pipeline_dO.producer_commit(producer_state_dO_dPsum)

                    # dPsum
                    pipeline_dPsum.producer_acquire(producer_state_dO_dPsum)
                    with cute.arch.elect_one():
                        copy_stats(
                            gdPsum[None, m_block_first],
                            sdPsum[None, producer_state_dO_dPsum.index],
                            mbar_ptr=pipeline_dPsum.producer_get_barrier(
                                producer_state_dO_dPsum
                            ),
                        )
                    producer_state_dO_dPsum.advance()

                    # Kt (loaded once, between prologue and main loop)
                    pipeline_Kt.producer_acquire(producer_state_Kt)
                    load_Kt(tma_bar_ptr=pipeline_Kt.producer_get_barrier(producer_state_Kt))
                    pipeline_Kt.producer_commit(producer_state_Kt)
                    producer_state_Kt.advance()

                    #### Mainloop (2CTA hdim128) ####
                    for it in cutlass.range(1, num_m_iters, unroll=1):
                        m_block = self.fm_m_block(fm_skip, m_block_min, it)
                        if const_expr(tma_atom_Qt is not None):
                            # Qt lags one iteration behind, so it wants the previous
                            # *processed* block, not m_block - 1.
                            pipeline_Qt.producer_acquire(producer_state_Qt)
                            load_Qt(
                                self.fm_m_block(fm_skip, m_block_min, it - 1),
                                producer_state=producer_state_Qt,
                            )
                            pipeline_Qt.producer_commit(producer_state_Qt)
                            producer_state_Qt.advance()

                        # Q (for S)
                        pipeline_Q.producer_acquire(producer_state_Q_LSE)
                        load_Q(m_block, producer_state=producer_state_Q_LSE)
                        pipeline_Q.producer_commit(producer_state_Q_LSE)

                        # LSE
                        pipeline_LSE.producer_acquire(producer_state_Q_LSE)
                        with cute.arch.elect_one():
                            copy_stats(
                                gLSE[None, m_block],
                                sLSE[None, producer_state_Q_LSE.index],
                                mbar_ptr=pipeline_LSE.producer_get_barrier(
                                    producer_state_Q_LSE
                                ),
                            )
                        producer_state_Q_LSE.advance()

                        # dO + dOt
                        pipeline_dO.producer_acquire(
                            producer_state_dO_dPsum,
                            extra_tx_count=self.tma_copy_bytes["dO"]
                            if const_expr(tma_atom_dOt is not None)
                            else 0,
                        )
                        load_dO(m_block, producer_state=producer_state_dO_dPsum)
                        if const_expr(tma_atom_dOt is not None):
                            load_dOt(m_block, producer_state=producer_state_dO_dPsum)
                        pipeline_dO.producer_commit(producer_state_dO_dPsum)

                        # dPsum
                        pipeline_dPsum.producer_acquire(producer_state_dO_dPsum)
                        with cute.arch.elect_one():
                            copy_stats(
                                gdPsum[None, m_block],
                                sdPsum[None, producer_state_dO_dPsum.index],
                                mbar_ptr=pipeline_dPsum.producer_get_barrier(
                                    producer_state_dO_dPsum
                                ),
                            )
                        producer_state_dO_dPsum.advance()

                    #### Tail (2CTA hdim128) ####
                    if const_expr(tma_atom_Qt is not None):
                        pipeline_Qt.producer_acquire(producer_state_Qt)
                        load_Qt(
                            self.fm_m_block(fm_skip, m_block_min, num_m_iters - 1),
                            producer_state=producer_state_Qt,
                        )
                        pipeline_Qt.producer_commit(producer_state_Qt)
                        producer_state_Qt.advance()

                    pipeline_Q.producer_tail(producer_state_Q_LSE.clone())
                    pipeline_LSE.producer_tail(producer_state_Q_LSE)
                    if const_expr(tma_atom_Qt is not None):
                        pipeline_Qt.producer_tail(producer_state_Qt)
                    pipeline_dO.producer_tail(producer_state_dO_dPsum.clone())
                    pipeline_dPsum.producer_tail(producer_state_dO_dPsum)

            elif const_expr(self.enable_flashmask):
                self.load_fm(
                    flashmask_info,
                    sStartEndRowIndices,
                    sFM_max_min,
                    seqlen,
                    mQ.shape[2],
                    n_block,
                    head_idx,
                    batch_idx,
                    overlap_segment_idx,
                )
                cute.arch.mbarrier_arrive(flashmask_loaded_mbar_ptr)

                zero_block = False
                prefetch_m_block = m_block_min
                prefetch_lte = False
                if const_expr(not self.is_causal):
                    has_uts = const_expr(flashmask_info.UTS_nblock_max is not None)
                    if not has_uts or prefetch_m_block > sFM_max_min[4]:
                        prefetch_m_block = sFM_max_min[7]
                if prefetch_m_block > sFM_max_min[0]:
                    has_lte = const_expr(flashmask_info.LTE_nblock_max is not None)
                    if has_lte:
                        prefetch_m_block = max(m_block_min, sFM_max_min[3])
                        prefetch_lte = True
                    else:
                        # masked whole n_block
                        prefetch_m_block = m_block_max
                if prefetch_m_block >= m_block_max:
                    zero_block = True

                # First iteration: load K together w Q & LSE, then V together w dO & dPsum
                if not zero_block and should_load_Q:
                    # K & Q
                    pipeline_Q.producer_acquire(
                        producer_state_Q_LSE, extra_tx_count=self.tma_copy_bytes["K"]
                    )
                    load_K(tma_bar_ptr=pipeline_Q.producer_get_barrier(producer_state_Q_LSE))
                    load_Q(prefetch_m_block, producer_state=producer_state_Q_LSE)
                    pipeline_Q.producer_commit(producer_state_Q_LSE)

                    if const_expr(self.use_2cta_instrs):
                        pipeline_Kt.producer_acquire(producer_state_Kt)
                        load_Kt(tma_bar_ptr=pipeline_Kt.producer_get_barrier(producer_state_Kt))
                        pipeline_Kt.producer_commit(producer_state_Kt)
                        producer_state_Kt.advance()

                    # LSE
                    pipeline_LSE.producer_acquire(producer_state_Q_LSE)
                    with cute.arch.elect_one():
                        copy_stats(
                            gLSE[None, prefetch_m_block],
                            sLSE[None, producer_state_Q_LSE.index],
                            mbar_ptr=pipeline_LSE.producer_get_barrier(producer_state_Q_LSE),
                        )
                    producer_state_Q_LSE.advance()
                if not zero_block and should_load_dO:
                    # V & dO
                    pipeline_dO.producer_acquire(
                        producer_state_dO_dPsum,
                        extra_tx_count=self.tma_copy_bytes["V"] + self.tma_copy_bytes["dO"]
                        if const_expr(tma_atom_dOt is not None)
                        else self.tma_copy_bytes["V"],
                    )
                    load_V(tma_bar_ptr=pipeline_dO.producer_get_barrier(producer_state_dO_dPsum))
                    load_dO(prefetch_m_block, producer_state=producer_state_dO_dPsum)
                    if const_expr(tma_atom_dOt is not None):
                        load_dOt(prefetch_m_block, producer_state=producer_state_dO_dPsum)
                    pipeline_dO.producer_commit(producer_state_dO_dPsum)
                    # dPsum
                    pipeline_dPsum.producer_acquire(producer_state_dO_dPsum)
                    with cute.arch.elect_one():
                        copy_stats(
                            gdPsum[None, prefetch_m_block],
                            sdPsum[None, producer_state_dO_dPsum.index],
                            mbar_ptr=pipeline_dPsum.producer_get_barrier(producer_state_dO_dPsum),
                        )
                    producer_state_dO_dPsum.advance()

                if const_expr(self.use_2cta_instrs) or not zero_block:
                    loop_start = m_block_min
                    loop_end = m_block_max
                    if const_expr(not self.is_causal):
                        has_uts = const_expr(flashmask_info.UTS_nblock_max is not None)
                        if has_uts and prefetch_m_block <= sFM_max_min[4]:
                            loop_end = sFM_max_min[4] + 1
                            # 0 ~ UTS
                            for m_block in cutlass.range(loop_start + 1, loop_end, unroll=1):
                                producer_state_Q_LSE, producer_state_dO_dPsum, producer_state_Qt, producer_state_Kt = load_step(
                                    m_block,
                                    producer_state_Q_LSE=producer_state_Q_LSE,
                                    producer_state_dO_dPsum=producer_state_dO_dPsum,
                                    producer_state_Qt=producer_state_Qt,
                                    producer_state_Kt=producer_state_Kt,
                                    m_block_prev=m_block - 1,
                                )
                            # Subtract 1 beforehand to use loop_start + 1 uniformly in the for loop.
                            # Clamp against UTS_max: with 4 vectors the two bands can overlap
                            # (UTE_min <= UTS_max), and the blocks up to UTS_max were already
                            # loaded above. mma and compute clamp the same way.
                            loop_start = max(sFM_max_min[4], sFM_max_min[7] - 1)
                        else:
                            loop_start = sFM_max_min[7]

                    # UTE ~ LTS
                    #loop_end = m_block_max if m_block_max < sFM_max_min[0] + 1 else sFM_max_min[0] + 1
                    loop_end = min(m_block_max, sFM_max_min[0] + 1)
                    for m_block in cutlass.range(loop_start + 1, loop_end, unroll=1):
                        producer_state_Q_LSE, producer_state_dO_dPsum, producer_state_Qt, producer_state_Kt = load_step(
                            m_block,
                            producer_state_Q_LSE=producer_state_Q_LSE,
                            producer_state_dO_dPsum=producer_state_dO_dPsum,
                            producer_state_Qt=producer_state_Qt,
                            producer_state_Kt=producer_state_Kt,
                            m_block_prev=m_block - 1,
                        )

                    # LTE ~ seqlen_q
                    has_lte = const_expr(flashmask_info.LTE_nblock_max is not None)
                    if has_lte:
                        loop_start = max(sFM_max_min[0], sFM_max_min[3])
                        #if prefetch_m_block == sFM_max_min[3]:
                        if not prefetch_lte and sFM_max_min[3] > sFM_max_min[0]:
                            # Subtract 1 beforehand to use loop_start + 1 uniformly in the for loop.
                            loop_start = sFM_max_min[3] - 1
                        loop_start = max(m_block_min, loop_start)

                        loop_end = m_block_max
                        for m_block in cutlass.range(loop_start + 1, loop_end, unroll=1):
                            producer_state_Q_LSE, producer_state_dO_dPsum, producer_state_Qt, producer_state_Kt = load_step(
                                m_block,
                                producer_state_Q_LSE=producer_state_Q_LSE,
                                producer_state_dO_dPsum=producer_state_dO_dPsum,
                                producer_state_Qt=producer_state_Qt,
                                producer_state_Kt=producer_state_Kt,
                                m_block_prev=m_block - 1,
                            )

                    if not zero_block and should_load_Q:
                        if const_expr(tma_atom_Qt is not None):
                            pipeline_Qt.producer_acquire(producer_state_Qt)
                            load_Qt(m_block_max - 1, producer_state=producer_state_Qt)
                            pipeline_Qt.producer_commit(producer_state_Qt)
                            producer_state_Qt.advance()

                        pipeline_Q.producer_tail(
                            producer_state_Q_LSE.clone()
                        )  # will hang if we don't clone
                        pipeline_LSE.producer_tail(producer_state_Q_LSE)
                        if const_expr(tma_atom_Qt is not None):
                            pipeline_Qt.producer_tail(producer_state_Qt.clone())
                    if not zero_block and should_load_dO:
                        pipeline_dO.producer_tail(producer_state_dO_dPsum.clone())
                        pipeline_dPsum.producer_tail(producer_state_dO_dPsum)

            else:
                # First iteration: load K together w Q & LSE, then V together w dO & dPsum
                if const_expr(should_load_Q):
                    # K & Q
                    pipeline_Q.producer_acquire(
                        producer_state_Q_LSE, extra_tx_count=self.tma_copy_bytes["K"]
                    )
                    load_K(tma_bar_ptr=pipeline_Q.producer_get_barrier(producer_state_Q_LSE))
                    load_Q(m_block_min, producer_state=producer_state_Q_LSE)
                    pipeline_Q.producer_commit(producer_state_Q_LSE)

                    # LSE
                    pipeline_LSE.producer_acquire(producer_state_Q_LSE)
                    with cute.arch.elect_one():
                        copy_stats(
                            gLSE[None, m_block_min],
                            sLSE[None, producer_state_Q_LSE.index],
                            mbar_ptr=pipeline_LSE.producer_get_barrier(producer_state_Q_LSE),
                        )
                    producer_state_Q_LSE.advance()
                if const_expr(should_load_dO):
                    # V & dO
                    pipeline_dO.producer_acquire(
                        producer_state_dO_dPsum,
                        extra_tx_count=self.tma_copy_bytes["V"] + self.tma_copy_bytes["dO"]
                        if const_expr(tma_atom_dOt is not None)
                        else self.tma_copy_bytes["V"],
                    )
                    load_V(tma_bar_ptr=pipeline_dO.producer_get_barrier(producer_state_dO_dPsum))
                    load_dO(m_block_min, producer_state=producer_state_dO_dPsum)
                    if const_expr(tma_atom_dOt is not None):
                        load_dOt(m_block_min, producer_state=producer_state_dO_dPsum)
                    pipeline_dO.producer_commit(producer_state_dO_dPsum)
                    # dPsum
                    pipeline_dPsum.producer_acquire(producer_state_dO_dPsum)
                    with cute.arch.elect_one():
                        copy_stats(
                            gdPsum[None, m_block_min],
                            sdPsum[None, producer_state_dO_dPsum.index],
                            mbar_ptr=pipeline_dPsum.producer_get_barrier(producer_state_dO_dPsum),
                        )
                    producer_state_dO_dPsum.advance()

                if const_expr(self.use_2cta_instrs and not self.use_2cta_bigd):
                    pipeline_Kt.producer_acquire(producer_state_Kt)
                    load_Kt(tma_bar_ptr=pipeline_Kt.producer_get_barrier(producer_state_Kt))
                    pipeline_Kt.producer_commit(producer_state_Kt)
                    producer_state_Kt.advance()

                for m_block in cutlass.range(m_block_min + 1, m_block_max, unroll=1):
                    producer_state_Q_LSE, producer_state_dO_dPsum, producer_state_Qt, producer_state_Kt = load_step(
                        m_block,
                        producer_state_Q_LSE=producer_state_Q_LSE,
                        producer_state_dO_dPsum=producer_state_dO_dPsum,
                        producer_state_Qt=producer_state_Qt,
                        producer_state_Kt=producer_state_Kt,
                        m_block_prev=m_block - 1,
                    )

                if const_expr(should_load_Q):
                    if const_expr(tma_atom_Qt is not None):
                        pipeline_Qt.producer_acquire(producer_state_Qt)
                        load_Qt(m_block_max - 1, producer_state=producer_state_Qt)
                        pipeline_Qt.producer_commit(producer_state_Qt)
                        producer_state_Qt.advance()

                    pipeline_Q.producer_tail(
                        producer_state_Q_LSE.clone()
                    )  # will hang if we don't clone
                    pipeline_LSE.producer_tail(producer_state_Q_LSE)
                    if const_expr(tma_atom_Qt is not None):
                        pipeline_Qt.producer_tail(producer_state_Qt)
                if const_expr(should_load_dO):
                    pipeline_dO.producer_tail(producer_state_dO_dPsum.clone())
                    pipeline_dPsum.producer_tail(producer_state_dO_dPsum)

            if const_expr(self.tile_boundary_sync):
                # Wait for the epilogue / reduce of this tile before touching
                # sK / sV / sFM again. See tile_boundary_barrier.
                self.tile_boundary_barrier.arrive_and_wait()

            tile_scheduler.prefetch_next_work()
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    @cute.jit
    def load_step(
        self,
        m_block: cute.Int32,
        gLSE: cute.Tensor,
        sLSE: cute.Tensor,
        gdPsum: cute.Tensor,
        sdPsum: cute.Tensor,
        pipeline_Q: PipelineAsync,
        pipeline_LSE: PipelineAsync,
        pipeline_dO: PipelineAsync,
        pipeline_dPsum: PipelineAsync,
        producer_state_Q_LSE: cutlass.pipeline.PipelineState,
        producer_state_dO_dPsum: cutlass.pipeline.PipelineState,
        load_Q: Callable,
        load_dO: Callable,
        copy_stats: Callable,
        should_load_Q: bool = True,
        should_load_dO: bool = True,
        pipeline_Qt: PipelineAsync = None,
        pipeline_Kt: PipelineAsync = None,
        producer_state_Qt: cutlass.pipeline.PipelineState = None,
        producer_state_Kt: cutlass.pipeline.PipelineState = None,
        load_Qt: Callable = None,
        load_dOt: Callable = None,
        m_block_prev: cute.Int32 = None,
    ):
        if const_expr(should_load_Q):
            if const_expr(load_Qt is not None):
                pipeline_Qt.producer_acquire(producer_state_Qt)
                load_Qt(m_block_prev, producer_state=producer_state_Qt)
                pipeline_Qt.producer_commit(producer_state_Qt)
                producer_state_Qt.advance()

            # Q
            pipeline_Q.producer_acquire(producer_state_Q_LSE)
            load_Q(m_block, producer_state=producer_state_Q_LSE)
            pipeline_Q.producer_commit(producer_state_Q_LSE)
            # LSE
            pipeline_LSE.producer_acquire(producer_state_Q_LSE)
            with cute.arch.elect_one():
                copy_stats(
                    gLSE[None, m_block],
                    sLSE[None, producer_state_Q_LSE.index],
                    mbar_ptr=pipeline_LSE.producer_get_barrier(producer_state_Q_LSE),
                )
            producer_state_Q_LSE.advance()

        if const_expr(should_load_dO):
            # dO
            pipeline_dO.producer_acquire(
                producer_state_dO_dPsum,
                extra_tx_count=self.tma_copy_bytes["dO"]
                if const_expr(load_dOt is not None)
                else 0,
            )
            load_dO(m_block, producer_state=producer_state_dO_dPsum)
            if const_expr(load_dOt is not None):
                load_dOt(m_block, producer_state=producer_state_dO_dPsum)
            pipeline_dO.producer_commit(producer_state_dO_dPsum)
            # dPsum
            pipeline_dPsum.producer_acquire(producer_state_dO_dPsum)
            with cute.arch.elect_one():
                copy_stats(
                    gdPsum[None, m_block],
                    sdPsum[None, producer_state_dO_dPsum.index],
                    mbar_ptr=pipeline_dPsum.producer_get_barrier(producer_state_dO_dPsum),
                )
            producer_state_dO_dPsum.advance()

        return producer_state_Q_LSE, producer_state_dO_dPsum, producer_state_Qt, producer_state_Kt

    @cute.jit
    def load_fm(
        self,
        flashmask_info: FlashMaskInfo,
        sStartEndRowIndices: cute.Tensor,
        sFM_max_min: cute.Tensor,
        seqlen_info: SeqlenInfoQK,
        num_heads: Int32,
        n_block: Int32,
        head_idx: Int32,
        batch_idx: Int32,
        overlap_segment_idx: Optional[cutlass.Int32],
    ):
        # (13) warp_idx == self.load_warp_id
        #num_load_threads = len([self.load_warp_id]) * cute.arch.WARP_SIZE
        num_load_threads = cute.arch.WARP_SIZE
        tidx = cute.arch.thread_idx()[0] % num_load_threads
        nblock_seqlen = ((seqlen_info.seqlen_k + self.tile_n - 1) // self.tile_n + 3) // 4 * 4
        ntimes_copy = (self.tile_n + num_load_threads - 1) // num_load_threads
        bsz, fm_heads, full_seqlen_k, num_vec = flashmask_info.startend_row_indices.shape
        fm_batch_idx = batch_idx if bsz > 1 else 0
        fm_head_idx = head_idx // (num_heads // fm_heads)
        bh_offset = fm_batch_idx * fm_heads + fm_head_idx
        if const_expr(overlap_segment_idx is not None):
            # Block metadata is laid out per segment (see ops/bwd.py), segments of seqlen_k rows.
            full_nblock_seqlen = full_seqlen_k // seqlen_info.seqlen_k * nblock_seqlen
            segment_row_offset = overlap_segment_idx * seqlen_info.seqlen_k
            bh_offset_block = (
                bh_offset * full_nblock_seqlen
                + overlap_segment_idx * nblock_seqlen
            )
        else:
            segment_row_offset = Int32(0)
            bh_offset_block = bh_offset * nblock_seqlen

        # The n blocks whose bounds decide which m blocks can be skipped. At cta_group=1
        # that is just this CTA's key block (nb1 == nb0, so the folding below is a no-op).
        # At cta_group=2 the two CTAs of a cluster own adjacent key blocks but share every
        # collective operation (the gemms are cta_group::2, the loads are multicast, some
        # mbarrier arrives carry the cluster mask), so they must walk the SAME m blocks.
        # Folding the pair's bounds together -- *_max with max, *_min with min, i.e.
        # intersecting the two fully-masked ranges -- makes them agree by construction:
        # same inputs, same expressions, no cross-CTA reduction. The per-element mask
        # stays the source of truth, so skipping less than one CTA alone could is only a
        # performance loss, never a wrong answer.
        nb0 = n_block
        nb1 = n_block
        if const_expr(self.cta_group_size > 1):
            n_block_last = (
                seqlen_info.seqlen_k + self.tile_n - 1
            ) // self.tile_n - 1
            nb0 = (n_block // self.cta_group_size) * self.cta_group_size
            nb1 = min(nb0 + 1, n_block_last)
        fm_layout = cute.make_layout(
            (cutlass.Int32(nblock_seqlen)), stride=(cutlass.Int32(1))
        )

        if tidx == 0:
            # LTS is always valid, otherwise this is not a valid flashmask computation instance
            LTS_max = cute.make_tensor(
                flashmask_info.LTS_nblock_max.iterator + bh_offset_block, fm_layout
            )
            LTS_min = cute.make_tensor(
                flashmask_info.LTS_nblock_min.iterator + bh_offset_block, fm_layout
            )
            sFM_max_min[0] = max(LTS_max[nb0] - 1, LTS_max[nb1] - 1) // self.tile_m
            sFM_max_min[1] = min(LTS_min[nb0], LTS_min[nb1]) // self.tile_m
            if const_expr(flashmask_info.LTE_nblock_max is not None):
                LTE_max = cute.make_tensor(
                    flashmask_info.LTE_nblock_max.iterator + bh_offset_block, fm_layout
                )
                LTE_min = cute.make_tensor(
                    flashmask_info.LTE_nblock_min.iterator + bh_offset_block, fm_layout
                )
                sFM_max_min[2] = max(LTE_max[nb0] - 1, LTE_max[nb1] - 1) // self.tile_m
                sFM_max_min[3] = min(LTE_min[nb0], LTE_min[nb1]) // self.tile_m
            if const_expr(flashmask_info.UTS_nblock_max is not None):
                UTS_max = cute.make_tensor(
                    flashmask_info.UTS_nblock_max.iterator + bh_offset_block, fm_layout
                )
                UTS_min = cute.make_tensor(
                    flashmask_info.UTS_nblock_min.iterator + bh_offset_block, fm_layout
                )
                sFM_max_min[4] = max(UTS_max[nb0] - 1, UTS_max[nb1] - 1) // self.tile_m
                sFM_max_min[5] = min(UTS_min[nb0], UTS_min[nb1]) // self.tile_m
            if const_expr(flashmask_info.UTE_nblock_max is not None):
                UTE_max = cute.make_tensor(
                    flashmask_info.UTE_nblock_max.iterator + bh_offset_block, fm_layout
                )
                UTE_min = cute.make_tensor(
                    flashmask_info.UTE_nblock_min.iterator + bh_offset_block, fm_layout
                )
                sFM_max_min[6] = max(UTE_max[nb0] - 1, UTE_max[nb1] - 1) // self.tile_m
                sFM_max_min[7] = min(UTE_min[nb0], UTE_min[nb1]) // self.tile_m

        for i in cutlass.range_constexpr(ntimes_copy):
            copy_offset = i * num_load_threads + tidx
            if const_expr(self.has_ut_start):
                # 4-vec: the mask is the union of [slot0, slot1) and [slot2, slot3).
                # For an out-of-range key column the lower band must cover every row so
                # that the column ends up fully masked, and the upper band must be empty.
                sStartEndRowIndices[copy_offset, 0] = 0
                sStartEndRowIndices[copy_offset, 1] = 2147483647
                sStartEndRowIndices[copy_offset, 2] = 0
                sStartEndRowIndices[copy_offset, 3] = 0
            else:
                sStartEndRowIndices[copy_offset, 0] = 2147483647
                sStartEndRowIndices[copy_offset, 1] = 2147483647
            local_k_row = n_block * self.tile_n + copy_offset
            if (copy_offset < self.tile_n and local_k_row < seqlen_info.seqlen_k):
                global_k_row = segment_row_offset + local_k_row
                LTS = flashmask_info.startend_row_indices[fm_batch_idx, fm_head_idx, None, 0]
                sStartEndRowIndices[copy_offset, 0] = LTS[global_k_row]
                if const_expr(self.has_ut_start):
                    # non-causal 4-vec: [LTS, LTE) and [UTS, UTE) from source cols 0..3
                    LTE = flashmask_info.startend_row_indices[fm_batch_idx, fm_head_idx, None, 1]
                    UTS = flashmask_info.startend_row_indices[fm_batch_idx, fm_head_idx, None, 2]
                    UTE = flashmask_info.startend_row_indices[fm_batch_idx, fm_head_idx, None, 3]
                    sStartEndRowIndices[copy_offset, 1] = LTE[global_k_row]
                    sStartEndRowIndices[copy_offset, 2] = UTS[global_k_row]
                    sStartEndRowIndices[copy_offset, 3] = UTE[global_k_row]
                elif const_expr(self.has_lt_end):
                    # causal 2-vec: [LTS, LTE)
                    LTE = flashmask_info.startend_row_indices[fm_batch_idx, fm_head_idx, None, 1]
                    sStartEndRowIndices[copy_offset, 1] = LTE[global_k_row]
                elif const_expr(self.has_ut_end):
                    # non-causal 2-vec: source col 1 is UTE, the mask is [0, UTE) | [LTS, inf)
                    UTE = flashmask_info.startend_row_indices[fm_batch_idx, fm_head_idx, None, 1]
                    sStartEndRowIndices[copy_offset, 1] = UTE[global_k_row]
        cute.arch.sync_warp()

    @cute.jit
    def fm_skip_info(
        self,
        flashmask_info: FlashMaskInfo,
        sFM_max_min: cute.Tensor,
        m_block_min: Int32,
        m_block_max: Int32,
    ):
        """The fully masked m blocks of this key block, as two ordered exclusion bands.

        An m block whose every element is masked contributes nothing (P == 0 => dV, and
        dS = P * (dP - dPsum) == 0 => dK, dQ), so it can be dropped from the m loop.
        flashmask's bounds collapse to at most two such half-open bands: the lower tail
        gives [LTS_max + 1, LTE_min) and, when the mask is not causal, the upper tail
        gives [UTS_max + 1, UTE_min). Clipping them to [m_block_min, m_block_max),
        ordering them by their start and merging them when they touch leaves the blocks
        that must be processed as three contiguous segments, which is what lets every
        warp -- load, mma, compute, dQ reduce -- walk the same blocks from one formula.

        Args:
            flashmask_info: Flashmask tensors and compile-time bound availability.
            sFM_max_min: Reduced lower and upper flashmask bounds for the KV tile.
            m_block_min: Inclusive first m-block considered by the scheduler.
            m_block_max: Exclusive last m-block considered by the scheduler.

        Returns:
            ``(a_lo, a_hi, b_lo, b_hi, n1, n2, num_iters)``, where the first four
            values are the ordered exclusion bands and ``n1`` / ``n2`` are the
            lengths of the first two retained segments. Use ``fm_m_block`` and
            ``fm_is_full_mask`` to consume the result.
        """
        l_lo = sFM_max_min[0] + 1
        if const_expr(flashmask_info.LTE_nblock_max is not None):
            l_hi = sFM_max_min[3]
        else:
            l_hi = m_block_max
        # A causal mask has no upper tail, so that band stays empty and the ordering
        # below pushes it to the far side. As in the segment walks, a non-causal
        # flashmask is assumed to carry UTE bounds.
        if const_expr(not self.is_causal):
            u_lo = Int32(0)
            if const_expr(flashmask_info.UTS_nblock_max is not None):
                u_lo = sFM_max_min[4] + 1
            u_hi = sFM_max_min[7]
        else:
            u_lo, u_hi = m_block_max, m_block_max

        # Clip both bands to [m_block_min, m_block_max) and keep them non-empty-safe
        # (hi >= lo), so the segment arithmetic below cannot go negative.
        l_lo = min(max(l_lo, m_block_min), m_block_max)
        l_hi = min(max(l_hi, l_lo), m_block_max)
        u_lo = min(max(u_lo, m_block_min), m_block_max)
        u_hi = min(max(u_hi, u_lo), m_block_max)
        u_first = u_lo <= l_lo
        a_lo = u_lo if u_first else l_lo
        a_hi = u_hi if u_first else l_hi
        b_lo = l_lo if u_first else u_lo
        b_hi = l_hi if u_first else u_hi
        # Merge: b starts at or after a ends, so the two segments below can neither
        # overlap nor leave a gap.
        b_lo = max(b_lo, a_hi)
        b_hi = max(b_hi, b_lo)

        n1 = a_lo - m_block_min
        n2 = b_lo - a_hi
        num_iters = n1 + n2 + (m_block_max - b_hi)
        if const_expr(self.use_2cta_instrs):
            # A key block that is masked everywhere would leave the m loop empty, but the
            # 2CTA epilogue writes dK / dV unconditionally and the accumulators are only
            # initialized by the first MMA of the loop. Keep the first block instead: it
            # is fully masked, so its contribution is zero, and it costs one iteration
            # only in this degenerate case.
            if num_iters == 0 and m_block_max > m_block_min:
                a_lo = a_lo + 1
                n1 = 1
                num_iters = 1
        return a_lo, a_hi, b_lo, b_hi, n1, n2, num_iters

    @cute.jit
    def fm_m_block(self, fm_skip_info, m_block_min: Int32, it: Int32):
        """The m block of iteration `it`, walking around the two exclusion bands.

        `fm_skip_info` is None when flashmask is off, i.e. nothing is skipped.
        """
        if const_expr(fm_skip_info is None):
            return m_block_min + it
        a_lo, a_hi, b_lo, b_hi, n1, n2, num_iters = fm_skip_info
        m_block = b_hi + (it - n1 - n2)
        m_block = a_hi + (it - n1) if it < n1 + n2 else m_block
        m_block = m_block_min + it if it < n1 else m_block
        return m_block

    @cute.jit
    def fm_needs_mask(
        self,
        flashmask_info: FlashMaskInfo,
        sFM_max_min: cute.Tensor,
        m_block: Int32,
    ):
        """Whether the per-element flashmask has to be applied to `m_block`.

        The three ranges the 1CTA segment walk processes with partially_masked=False
        are exactly the ones where no element of the block is masked: below UTS_min,
        between UTE_max and LTS_min, and above LTE_max. This returns the complement
        of those ranges, so the 2CTA flat loop can skip the mask on the same blocks.

        Safe under the 2CTA bound folding: load_fm reduces the pair's *_min with min
        and *_max with max, so a block that this test calls unmasked is unmasked for
        BOTH key blocks, hence for this CTA. It also never calls a FULLY masked block
        unmasked -- the fully masked bands are [LTS_max + 1, LTE_min) and
        [UTS_max + 1, UTE_min), which are disjoint from all three ranges above. That
        matters because fm_skip_info deliberately keeps one fully masked block alive
        when a KV tile is masked everywhere, and relies on the mask to zero it.

        With 4 vectors each test looks at ONE band only: below UTS_min it ignores the
        lower band, above LTE_max it ignores the upper one. That is sound because the
        two bands are per-column triangle-ordered (UTS <= UTE <= col < LTS <= LTE),
        which is what "upper / lower tail" means in flashmask -- a mask whose upper
        band reached below some other column's LTS would need the union predicate the
        host side uses for the fwd (fm_partial in flashmask_utils.prepare_block_maxmin).
        """
        needs_mask = cutlass.Boolean(True)
        if const_expr(not self.is_causal):
            # UTE_max ~ LTS_min
            if const_expr(flashmask_info.UTE_nblock_max is not None):
                if m_block < sFM_max_min[1]:
                    if m_block > sFM_max_min[6]:
                        needs_mask = cutlass.Boolean(False)
            if const_expr(flashmask_info.UTS_nblock_max is not None):
                # 0 ~ UTS_min
                if m_block < sFM_max_min[5]:
                    needs_mask = cutlass.Boolean(False)
        else:
            # Causal has no upper tail, so this range reaches down to m_block_min.
            if m_block < sFM_max_min[1]:
                needs_mask = cutlass.Boolean(False)
        if const_expr(flashmask_info.LTE_nblock_max is not None):
            # LTE_max ~ seqlen_q
            if m_block > sFM_max_min[2]:
                needs_mask = cutlass.Boolean(False)
        return needs_mask

    @cute.jit
    def fm_is_full_mask(self, fm_skip_info, m_block: Int32):
        """Whether `m_block` falls inside one of the two exclusion bands."""
        a_lo, a_hi, b_lo, b_hi, n1, n2, num_iters = fm_skip_info
        return (m_block >= a_lo and m_block < a_hi) or (
            m_block >= b_lo and m_block < b_hi
        )

    @cute.jit
    def mma(
        self,
        tiled_mma_S: cute.TiledMma,
        tiled_mma_dP: cute.TiledMma,
        tiled_mma_dV: cute.TiledMma,
        tiled_mma_dK: cute.TiledMma,
        tiled_mma_dQ: cute.TiledMma,
        sQ: cute.Tensor,
        sQt: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        sdO: cute.Tensor,
        sdOt: cute.Tensor,
        sdSt: cute.Tensor,
        sdS: cute.Tensor,
        sKt: cute.Tensor,
        tP: cute.Tensor,
        tdS: cute.Tensor,
        tStS: cute.Tensor,
        tdPtdP: cute.Tensor,
        tdVtdV: cute.Tensor,
        tdKtdK: cute.Tensor,
        tdQtdQ: cute.Tensor,
        dS_cluster_full_mbar_ptr: cute.Pointer,
        dS_cluster_empty_mbar_ptr: cute.Pointer,
        dS_cluster_leader_mbar_ptr: Optional[cute.Pointer],
        dQaccum_empty_mbar_ptr: Optional[cute.Pointer],
        pipeline_Q: PipelineAsync,
        pipeline_Q_consumer: PipelineConsumer,
        pipeline_Qt: PipelineAsync,
        pipeline_Kt: PipelineAsync,
        pipeline_dO: PipelineAsync,
        pipeline_S_P: PipelineAsync,
        pipeline_dS: PipelineAsync,
        pipeline_dKV: PipelineAsync,
        pipeline_dP: PipelineAsync,
        pipeline_dQ: PipelineAsync,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        flashmask_info: FlashMaskInfo,
        sFM_max_min: cute.Tensor,
        flashmask_loaded_mbar_ptr: cute.Pointer,
        is_leader_cta: cutlass.Boolean,
    ):
        # [2025-10-21] For reasons I don't understand, putting these partitioning in the main
        # kernel (before warp specialization) is a lot slower tha putting them here.
        # Partition smem / tmem tensors
        # S = K @ Q.T
        num_load_threads = cute.arch.WARP_SIZE
        tidx = cute.arch.thread_idx()[0] % num_load_threads
        tSrK = tiled_mma_S.make_fragment_A(sK)
        tSrQ = tiled_mma_S.make_fragment_B(sQ)
        # dP = V @ dO.T
        tdPrV = tiled_mma_dP.make_fragment_A(sV)
        tdPrdOt = tiled_mma_dP.make_fragment_B(sdOt)
        # dK = dS.T @ Q
        if const_expr(self.mma_dS_from_smem):
            tdKrdS = tiled_mma_dK.make_fragment_A(sdSt)  # From SMEM
        else:
            tdKrdS = tiled_mma_dK.make_fragment_A(tdS)  # From TMEM
        tdKrQ = tiled_mma_dK.make_fragment_B(sQt)
        # dQ = dS @ K
        tdQrdS = tiled_mma_dQ.make_fragment_A(sdS)
        tdQrK = tiled_mma_dQ.make_fragment_B(sKt)
        # dV = P @ dO.T
        tdVrdO = tiled_mma_dV.make_fragment_B(sdO)
        tdVrP = tiled_mma_dV.make_fragment_A(tP)

        # mma_qk_fn = partial(gemm_w_idx, tiled_mma_S, tStS, tSrK, tSrQ, zero_init=True)
        mma_qk_fn = partial(
            gemm_ptx_w_idx, tiled_mma_S, tStS, tSrK, tSrQ, sA=sK, sB=sQ, zero_init=True,
            cta_group=self.cta_group_size,
        )
        # mma_dov_fn = partial(gemm_w_idx, tiled_mma_dP, tdPtdP, tdPrV, tdPrdOt, zero_init=True)
        mma_dov_fn = partial(
            gemm_ptx_w_idx,
            tiled_mma_dP,
            tdPtdP,
            tdPrV,
            tdPrdOt,
            sA=sV,
            sB=sdOt,
            zero_init=True,
            cta_group=self.cta_group_size,
        )
        # mma_pdo_fn = partial(gemm_w_idx, tiled_mma_dV, tdVtdV, tdVrP, tdVrdO)
        mma_pdo_fn = partial(
            gemm_ptx_w_idx,
            tiled_mma_dV,
            tdVtdV,
            tdVrP,
            tdVrdO,
            # On the folded path P lives in SMEM (see mma_P_from_smem), so the MMA
            # needs its smem descriptor; tA_addr is A's TMEM address and only
            # applies when A comes from TMEM.
            sA=tP if const_expr(self.mma_P_from_smem) else None,
            sB=sdO,
            tA_addr=None if const_expr(self.mma_P_from_smem) else self.tmem_P_offset,
            cta_group=self.cta_group_size,
        )

        num_unroll_groups = 2 if const_expr(self.use_2cta_instrs) else 1
        mma_dsk_fn = partial(gemm_w_idx, tiled_mma_dQ, tdQtdQ, tdQrdS, tdQrK, zero_init=True, num_unroll_groups=num_unroll_groups)
        # mma_dsk_fn = partial(
        #     gemm_ptx_w_idx, tiled_mma_dQ, tdQtdQ, tdQrdS, tdQrK, sA=sdS, sB=sKt, zero_init=True
        # )
        # Need to explicitly pass in tA_addr for correctness
        mma_dsq_fn = partial(
            gemm_ptx_w_idx,
            tiled_mma_dK,
            tdKtdK,
            tdKrdS,
            tdKrQ,
            sA=sdSt if const_expr(self.mma_dS_from_smem) else None,
            sB=sQt,
            tA_addr=None if const_expr(self.mma_dS_from_smem) else self.tmem_dS_offset,
            cta_group=self.cta_group_size,
        )

        consumer_state_Q = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.Q_stage
        )
        consumer_state_Qt = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.Q_stage
        )
        consumer_state_Kt = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.single_stage
        )
        consumer_state_dO = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.dO_stage
        )
        producer_phase_acc = Int32(1)  # For S & P, dP, dQ
        producer_phase_dQ = Int32(1)  # 2-CTA: separate phase for dQ pipeline
        dS_cluster_phase = Int32(0)
        consumer_state_dS = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, 1
        )
        # producer_state_dKV = cutlass.pipeline.make_pipeline_state(
        #     cutlass.pipeline.PipelineUserType.Producer, 2
        # )
        producer_phase_dKV = Int32(1)
        cta_group = pipeline_S_P.cta_group

        if const_expr(self.enable_flashmask):
            flashmask_phase = Int32(0)

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            n_block, head_idx, batch_idx, _ = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            m_block_min, m_block_max = block_info.get_m_block_min_max(
                seqlen, n_block // self.cluster_shape_mnk[0]
            )

            num_blocks = m_block_max - m_block_min
            if const_expr(self.enable_flashmask):
                cute.arch.mbarrier_wait(flashmask_loaded_mbar_ptr, flashmask_phase)

                if not const_expr(self.use_2cta_instrs):
                    # 1CTA: compute num_blocks by subtracting full-mask blocks
                    num_blocks = 0
                    loop_start = m_block_min
                    loop_end = m_block_max
                    if const_expr(not self.is_causal):
                        has_uts = const_expr(flashmask_info.UTS_nblock_max is not None)
                        if has_uts:
                            loop_end = min(m_block_max, sFM_max_min[4] + 1)
                            #  ~ UTS
                            num_blocks = num_blocks + max(0, (loop_end - loop_start))
                            loop_start = loop_end
                        loop_start = max(loop_start, sFM_max_min[7])

                    # UTE ~ LTS
                    loop_end = min(m_block_max, sFM_max_min[0] + 1)
                    num_blocks = num_blocks + max(0, (loop_end - loop_start))

                    # LTE ~ seqlen_q
                    has_lte = const_expr(flashmask_info.LTE_nblock_max is not None)
                    if has_lte:
                        loop_start = max(sFM_max_min[0] + 1, sFM_max_min[3])
                        if sFM_max_min[3] == sFM_max_min[0]:
                            loop_start = sFM_max_min[3] + 1
                        loop_start = max(m_block_min, loop_start)
                        loop_end = m_block_max
                        num_blocks = num_blocks + (loop_end - loop_start)
                else:
                    # 2CTA: the same block count the load / compute / reduce warps derive
                    # from the pair's combined bounds.
                    num_blocks = self.fm_skip_info(
                        flashmask_info, sFM_max_min, m_block_min, m_block_max
                    )[6]

            if const_expr(self.use_2cta_bigd):
                # 2CTA big-hdim: flat loop with double pipeline_Q/dO consumption
                # Only leader CTA executes MMA loop; non-leader CTA's MMA warps are idle
                if is_leader_cta:
                    accumulate_dK = False
                    accumulate_dV = False

                    main_loop_iters = num_blocks

                    for _ in cutlass.range(main_loop_iters, unroll=1):
                        # 1) S.T = K @ Q.T
                        pipeline_Q.consumer_wait(consumer_state_Q)
                        if const_expr(not self.bigd_dP_overlaps_S):
                            # The previous S must be read out of TMEM before this S
                            # overwrites it (the first wait passes at the initial phase).
                            pipeline_S_P.sync_object_empty.wait(0, producer_phase_acc)
                        if const_expr(not self.late_dq_empty_wait):
                            # d192/dv128: dQ time-shares S/P's TMEM columns, so the
                            # previous iteration's dQ must be read out of TMEM before
                            # S may be written. d256/dv256's layout is disjoint
                            # (dV | dK | S/P | dP/dS | dQ), so it takes the late wait
                            # at step 5 instead. That matters because the release this
                            # waits on (dQacc_reduce_step, after its TMEM->RMEM copy)
                            # sits behind the previous bulk reduce-add drain whenever
                            # sdQaccum_stage == 1, so keeping it here would put the dQ
                            # gmem traffic on the critical path of MMAs 1-4.
                            pipeline_dQ.sync_object_empty.wait(0, producer_phase_acc)
                        mma_qk_fn(B_idx=consumer_state_Q.index)
                        pipeline_S_P.sync_object_full.arrive(
                            0, pipeline_S_P.producer_mask, cta_group
                        )
                        pipeline_Q.consumer_release(consumer_state_Q)
                        consumer_state_Q.advance()

                        producer_phase_acc ^= 1

                        # 2) dP.T = V @ dO.T
                        pipeline_dO.consumer_wait(consumer_state_dO)
                        if const_expr(self.bigd_dP_overlaps_S):
                            pipeline_S_P.sync_object_empty.wait(
                                0, producer_phase_acc
                            )  # dP tmem overlaps with S
                        mma_dov_fn(B_idx=consumer_state_dO.index)
                        pipeline_dP.sync_object_full.arrive(0, pipeline_dP.producer_mask, cta_group)
                        pipeline_dO.consumer_release(consumer_state_dO)
                        consumer_state_dO.advance()

                        # 3) dK = dS.T @ Q
                        pipeline_Q.consumer_wait(consumer_state_Q)
                        pipeline_dP.sync_object_empty.wait(0, producer_phase_acc)  # dP -> dS
                        mma_dsq_fn(B_idx=consumer_state_Q.index, zero_init=not accumulate_dK)
                        pipeline_Q.consumer_release(consumer_state_Q)
                        consumer_state_Q.advance()
                        accumulate_dK = True

                        # 4) dV = P.T @ dO
                        pipeline_dO.consumer_wait(consumer_state_dO)
                        mma_pdo_fn(B_idx=consumer_state_dO.index, zero_init=not accumulate_dV)
                        pipeline_dO.consumer_release(consumer_state_dO)
                        consumer_state_dO.advance()
                        accumulate_dV = True

                        # 5) dQ = dS @ K
                        pipeline_dS.consumer_wait(consumer_state_dS)
                        cute.arch.mbarrier_wait(dS_cluster_leader_mbar_ptr, phase=dS_cluster_phase)
                        if const_expr(self.late_dq_empty_wait):
                            # Late dQ-empty wait, moved down from step 1 (see there).
                            # mma_dsk_fn zero-initializes the accumulator, so the only
                            # ordering needed is that dQacc_reduce_step already copied
                            # the previous dQ out of TMEM. producer_phase_dQ is flipped
                            # once per iteration below, giving the same parity sequence
                            # the step-1 wait had (both start at 1).
                            pipeline_dQ.sync_object_empty.wait(0, producer_phase_dQ)
                        mma_dsk_fn()
                        pipeline_dQ.sync_object_full.arrive(0, pipeline_dQ.producer_mask, cta_group)
                        pipeline_dS.consumer_release(consumer_state_dS)
                        consumer_state_dS.advance()
                        dS_cluster_phase ^= 1
                        if const_expr(self.late_dq_empty_wait):
                            producer_phase_dQ ^= 1

                    # signal to the epilogue that dV is ready
                    pipeline_dKV.sync_object_empty.wait(0, producer_phase_dKV)
                    pipeline_dKV.sync_object_full.arrive(0, pipeline_dKV.producer_mask, cta_group)
                    # signal to the epilogue that dK is ready
                    pipeline_dKV.sync_object_empty.wait(1, producer_phase_dKV)
                    pipeline_dKV.sync_object_full.arrive(1, pipeline_dKV.producer_mask, cta_group)
                    producer_phase_dKV ^= 1

            elif const_expr(self.use_2cta_instrs):
                if is_leader_cta and num_blocks > 0:
                    accumulate_dK = False
                    # -----------------------------------------------------------
                    ###### Prologue (2CTA hdim128)
                    # -----------------------------------------------------------
                    # 1. S  = Q0 @ K.T
                    # 2. dP = V @ dOt.T
                    # 3. dV = P @ dO

                    # 1) S = K @ Q
                    pipeline_Q.consumer_wait(consumer_state_Q)
                    pipeline_S_P.sync_object_empty.wait(0, producer_phase_acc)
                    mma_qk_fn(B_idx=consumer_state_Q.index)
                    pipeline_S_P.sync_object_full.arrive(0, pipeline_S_P.producer_mask, cta_group)
                    pipeline_Q.consumer_release(consumer_state_Q)
                    consumer_state_Q.advance()

                    # 2) dP = V @ dOt.T
                    pipeline_dO.consumer_wait(consumer_state_dO)
                    pipeline_dP.sync_object_empty.wait(0, producer_phase_acc)
                    mma_dov_fn(B_idx=consumer_state_dO.index)
                    pipeline_dP.sync_object_full.arrive(0, pipeline_dP.producer_mask, cta_group)

                    # 3) dV = P.T @ dO
                    producer_phase_acc ^= 1
                    pipeline_S_P.sync_object_empty.wait(0, producer_phase_acc)
                    mma_pdo_fn(B_idx=consumer_state_dO.index, zero_init=True)
                    pipeline_dO.consumer_release(consumer_state_dO)
                    consumer_state_dO.advance()

                    pipeline_Kt.consumer_wait(consumer_state_Kt)
                    # -----------------------------------------------------------
                    ###### MAIN LOOP (2CTA hdim128)
                    # -----------------------------------------------------------
                    # 1. S.T  = K    @ Q.T
                    # 2. dK   = dS.T @ Q
                    # 3. dP.T = V    @ dO.T
                    # 4. dQ   = dS   @ K
                    # 5. dV   = P.T  @ dO

                    main_loop_iters = num_blocks - 1

                    for _ in cutlass.range(main_loop_iters, unroll=1):
                        # (1) S.T = K @ Q.T (next)
                        pipeline_Q.consumer_wait(consumer_state_Q)
                        pipeline_dQ.sync_object_empty.wait(0, producer_phase_dQ)
                        mma_qk_fn(B_idx=consumer_state_Q.index)
                        pipeline_S_P.sync_object_full.arrive(
                            0, pipeline_S_P.producer_mask, cta_group
                        )
                        pipeline_Q.consumer_release(consumer_state_Q)
                        consumer_state_Q.advance()

                        # (2) dK += dS.T @ Q (cur)
                        pipeline_Qt.consumer_wait(consumer_state_Qt)
                        pipeline_dP.sync_object_empty.wait(0, producer_phase_acc)  # dP -> dS
                        mma_dsq_fn(B_idx=consumer_state_Qt.index, zero_init=not accumulate_dK)
                        accumulate_dK = True
                        pipeline_Qt.consumer_release(consumer_state_Qt)
                        consumer_state_Qt.advance()

                        # (3) dP.T = V @ dO.T (next)
                        pipeline_dO.consumer_wait(consumer_state_dO)
                        mma_dov_fn(B_idx=consumer_state_dO.index)
                        pipeline_dP.sync_object_full.arrive(0, pipeline_dP.producer_mask, cta_group)

                        # (4) dQ = dS @ K (cur)
                        pipeline_dS.consumer_wait(consumer_state_dS)
                        cute.arch.mbarrier_wait(dS_cluster_leader_mbar_ptr, phase=dS_cluster_phase)
                        mma_dsk_fn()
                        pipeline_dQ.sync_object_full.arrive(0, pipeline_dQ.producer_mask, cta_group)
                        pipeline_dS.consumer_release(consumer_state_dS)
                        consumer_state_dS.advance()
                        dS_cluster_phase ^= 1
                        producer_phase_dQ ^= 1

                        # (5) dV += P.T @ dO (next)
                        producer_phase_acc ^= 1
                        pipeline_S_P.sync_object_empty.wait(0, producer_phase_acc)  # S -> P
                        mma_pdo_fn(B_idx=consumer_state_dO.index, zero_init=False)
                        pipeline_dO.consumer_release(consumer_state_dO)
                        consumer_state_dO.advance()

                    pipeline_S_P.sync_object_full.arrive(0, pipeline_S_P.producer_mask, cta_group)

                    # signal to the epilogue that dV is ready
                    pipeline_dKV.sync_object_empty.wait(0, producer_phase_dKV)
                    pipeline_dKV.sync_object_full.arrive(0, pipeline_dKV.producer_mask, cta_group)
                    pipeline_dKV.sync_object_empty.wait(1, producer_phase_dKV)

                    # -----------------------------------------------------------
                    # Tail: Remaining dK and dQ
                    # -----------------------------------------------------------
                    # dK += dS.T @ Q
                    pipeline_Qt.consumer_wait(consumer_state_Qt)
                    pipeline_dP.sync_object_empty.wait(0, producer_phase_acc)  # dP -> dS
                    mma_dsq_fn(B_idx=consumer_state_Qt.index, zero_init=not accumulate_dK)
                    pipeline_Qt.consumer_release(consumer_state_Qt)
                    consumer_state_Qt.advance()
                    # signal to the epilogue that dK is ready
                    pipeline_dKV.sync_object_full.arrive(1, pipeline_dKV.producer_mask, cta_group)
                    producer_phase_dKV ^= 1

                    # dQ = dS @ K
                    pipeline_dS.consumer_wait(consumer_state_dS)
                    cute.arch.mbarrier_wait(dS_cluster_leader_mbar_ptr, phase=dS_cluster_phase)
                    pipeline_dQ.sync_object_empty.wait(0, producer_phase_dQ)
                    mma_dsk_fn()
                    pipeline_dQ.sync_object_full.arrive(0, pipeline_dQ.producer_mask, cta_group)
                    pipeline_dS.consumer_release(consumer_state_dS)
                    pipeline_Kt.consumer_release(consumer_state_Kt)
                    consumer_state_dS.advance()
                    consumer_state_Kt.advance()
                    dS_cluster_phase ^= 1
                    producer_phase_dQ ^= 1

                    producer_phase_acc ^= 1

            elif is_leader_cta and num_blocks > 0:
                accumulate_dK = False
                # -----------------------------------------------------------
                ###### Prologue
                # -----------------------------------------------------------
                # 1. S  = Q0 @ K.T
                # 2. dP = V @ dO.T
                # 3. dV = P @ dO

                # 1) S  = Q0 @ K.T
                m_block_cur = cute.Int32(0)
                handle_Q = pipeline_Q_consumer.wait_and_advance()
                pipeline_S_P.sync_object_empty.wait(0, producer_phase_acc)
                mma_qk_fn(B_idx=handle_Q.index)
                # Don't release Q yet
                pipeline_S_P.sync_object_full.arrive(0, pipeline_S_P.producer_mask, cta_group)

                # 2) dP = V @ dO.T
                pipeline_dO.consumer_wait(consumer_state_dO)
                pipeline_dP.sync_object_empty.wait(0, producer_phase_acc)
                # dQ uses the same tmem as dP
                pipeline_dQ.sync_object_empty.wait(0, producer_phase_acc)
                mma_dov_fn(B_idx=consumer_state_dO.index)
                # Don't release dO yet
                pipeline_dP.sync_object_full.arrive(0, pipeline_dP.producer_mask, cta_group)

                producer_phase_acc ^= 1
                # 3) dV = P.T @ dO
                # wait for P to be ready, which uses the same tmem as S
                pipeline_S_P.sync_object_empty.wait(0, producer_phase_acc)
                mma_pdo_fn(B_idx=consumer_state_dO.index, zero_init=True)
                pipeline_dO.consumer_release(consumer_state_dO)
                consumer_state_dO.advance()
                # -----------------------------------------------------------
                ###### MAIN LOOP
                # -----------------------------------------------------------
                # 1. S  = K    @ Q.T
                # 2. dQ = dS   @ K
                # 3. dK = dS.T @ Q
                # 4. dP = V    @ dO.T
                # 5. dV = P.T  @ dO
                num_blocks = num_blocks - 1

                for m_block in cutlass.range(0, num_blocks, unroll=1):
                    m_block_cur = m_block_cur + 1

                    # 1) S = K @ Q_i
                    handle_Q_next = pipeline_Q_consumer.wait_and_advance()
                    # Don't need to wait for S, as P must have been ready ealier, i.e., S is ready
                    mma_qk_fn(B_idx=handle_Q_next.index)
                    pipeline_S_P.sync_object_full.arrive(0, pipeline_S_P.producer_mask, cta_group)

                    # 2-3) dK = dS.T @ Q, then dQ = dS @ K
                    pipeline_dS.consumer_wait(consumer_state_dS)

                    mma_dsq_fn(B_idx=handle_Q.index, zero_init=not accumulate_dK)
                    accumulate_dK = True
                    handle_Q.release()
                    mma_dsk_fn()
                    pipeline_dQ.sync_object_full.arrive(0, pipeline_dQ.producer_mask, cta_group)

                    # dP uses the same tmem as dQ
                    # However, if dS is ready, then dP must have been ready,
                    # so we don't need this wait before mma_dsk_fn()
                    # pipeline_dP.sync_object_empty.wait(0, producer_phase_acc)

                    pipeline_dS.consumer_release(consumer_state_dS)
                    consumer_state_dS.advance()

                    # 4) dP = V @ dO.T
                    pipeline_dO.consumer_wait(consumer_state_dO)
                    # dQ uses the same tmem as dP
                    pipeline_dQ.sync_object_empty.wait(0, producer_phase_acc)
                    mma_dov_fn(B_idx=consumer_state_dO.index)
                    pipeline_dP.sync_object_full.arrive(0, pipeline_dP.producer_mask, cta_group)

                    producer_phase_acc ^= 1
                    # 5) dV += P @ dO
                    # wait for P to be ready, which uses the same tmem as S
                    pipeline_S_P.sync_object_empty.wait(0, producer_phase_acc)
                    mma_pdo_fn(B_idx=consumer_state_dO.index, zero_init=False)
                    pipeline_dO.consumer_release(consumer_state_dO)
                    consumer_state_dO.advance()

                    handle_Q = handle_Q_next


                pipeline_S_P.sync_object_full.arrive(0, pipeline_S_P.producer_mask, cta_group)

                # signal to the epilogue that dV is ready
                # pipeline_dKV.producer_acquire(producer_state_dKV)
                pipeline_dKV.sync_object_empty.wait(0, producer_phase_dKV)
                # pipeline_dKV.producer_commit(producer_state_dKV)
                pipeline_dKV.sync_object_full.arrive(0, pipeline_dKV.producer_mask, cta_group)
                # producer_state_dKV.advance()
                # pipeline_dKV.producer_acquire(producer_state_dKV)
                pipeline_dKV.sync_object_empty.wait(1, producer_phase_dKV)

                # -----------------------------------------------------------
                ###### Remaining 2
                # -----------------------------------------------------------
                # 1) dK += dS.T @ Q
                pipeline_dS.consumer_wait(consumer_state_dS)
                mma_dsq_fn(B_idx=handle_Q.index, zero_init=not accumulate_dK)
                # signal to the epilogue that dK is ready
                # pipeline_dKV.producer_commit(producer_state_dKV)
                pipeline_dKV.sync_object_full.arrive(1, pipeline_dKV.producer_mask, cta_group)
                # producer_state_dKV.advance()
                producer_phase_dKV ^= 1

                # 2) dQ = dS @ K
                # dS is done, so dP must have been ready, we don't need to wait
                mma_dsk_fn()
                pipeline_dQ.sync_object_full.arrive(0, pipeline_dQ.producer_mask, cta_group)
                # Wait until dQ is done before releasing Q, since K and Q0 uses the same mbarrier
                handle_Q.release()
                pipeline_dS.consumer_release(consumer_state_dS)
                consumer_state_dS.advance()

                producer_phase_acc ^= 1

            if const_expr(self.enable_flashmask):
                flashmask_phase ^= 1

            if const_expr(self.tile_boundary_sync):
                # Do not start the next tile's zero-init dV / dK MMA before the epilogue
                # has read this tile's TMEM accumulators. See tile_boundary_barrier.
                self.tile_boundary_barrier.arrive_and_wait()
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()


        # Currently it hangs if we have this S_P.producer_tail, will need to understand why
        # pipeline_S_P.producer_tail(producer_state_S_P)
        # pipeline_dP.producer_tail(producer_state_dP)
        # pipeline_dKV.producer_tail(producer_state_dKV)
        # pipeline_dQ.producer_tail(producer_state_dQ)

    @cute.jit
    def split_wg(
        self,
        t: cute.Tensor,
        wg_idx: cutlass.Int32,
        num_wg: cutlass.Constexpr[int],
    ):
        reduced_shape = cute.product_each(t.shape)
        rank = len(reduced_shape)
        if const_expr(reduced_shape[1] > 1):
            assert rank >= 2, "Need rank >= 2 for t in split_wg"
            t = cute.logical_divide(t, (reduced_shape[0], reduced_shape[1] // num_wg))
            coord = (None, (None, wg_idx)) + (None,) * (rank - 2)
        else:
            assert rank >= 3, "Need rank >= 3 for t in split_wg"
            if const_expr(rank == 3):
                t = cute.logical_divide(
                    t, (reduced_shape[0], reduced_shape[1], reduced_shape[2] // num_wg)
                )
                coord = (
                    None,
                    None,
                    (None, wg_idx),
                ) + (None,) * (rank - 3)
            else:
                t = cute.logical_divide(
                    t,
                    (
                        reduced_shape[0],
                        reduced_shape[1],
                        reduced_shape[2],
                        reduced_shape[3] // num_wg,
                    ),
                )
                coord = (
                    None,
                    None,
                    None,
                    (None, wg_idx),
                ) + (None,) * (rank - 4)
        return t[coord]

    @cute.jit
    def compute_loop(
        self,
        thr_mma_S: cute.ThrMma,
        thr_mma_dP: cute.ThrMma,
        thr_mma_dV: cute.ThrMma,
        thr_mma_dK: cute.ThrMma,
        tStS: cute.Tensor,
        tdPtdP: cute.Tensor,
        tdVtdV: cute.Tensor,
        tdKtdK: cute.Tensor,
        sLSE: cute.Tensor,
        sdPsum: cute.Tensor,
        mdV: cute.Tensor,
        mdK: cute.Tensor,
        sdS: cute.Tensor,
        sdSt: cute.Tensor,
        tP: cute.Tensor,
        sdS_xchg: cute.Tensor,
        pipeline_LSE: PipelineAsync,
        pipeline_dPsum: PipelineAsync,
        pipeline_S_P: PipelineAsync,
        pipeline_dS: PipelineAsync,
        pipeline_dKV: PipelineAsync,
        pipeline_dP: PipelineAsync,
        dS_cluster_empty_mbar_ptr: cute.Pointer,
        dS_cluster_full_mbar_ptr: cute.Pointer,
        dQaccum_empty_mbar_ptr: Optional[cute.Pointer],
        softmax_scale: cutlass.Float32,
        softmax_scale_log2: cutlass.Float32,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        AttentionMaskCls: Callable,
        TileSchedulerCls: Callable,
        sdV: Optional[cute.Tensor],
        sdK: Optional[cute.Tensor],
        mdV_tma_tensor: Optional[cute.Tensor],
        mdK_tma_tensor: Optional[cute.Tensor],
        tma_atom_dV: Optional[cute.CopyAtom],
        tma_atom_dK: Optional[cute.CopyAtom],
        tiled_copy_r2s_dKV: Optional[cute.TiledCopy],
        mdK_semaphore: Optional[cute.Tensor],
        mdV_semaphore: Optional[cute.Tensor],
        flashmask_info: FlashMaskInfo,
        sStartEndRowIndices: cute.Tensor,
        sFM_max_min: cute.Tensor,
        flashmask_loaded_mbar_ptr: cute.Pointer,
        is_leader_cta: cutlass.Boolean,
        sdS_swizzle=None,
        sdSt_swizzle=None,
        tP_swizzle=None,
    ):
        sLSE_2D = cute.make_tensor(
            sLSE.iterator,
            cute.make_layout(
                (self.tile_m, self.tile_n, self.Q_stage),
                stride=(1, 0, cute.round_up(self.tile_m, 64)),
            ),
        )
        sdPsum_2D = cute.make_tensor(
            sdPsum.iterator,
            cute.make_layout(
                (self.tile_m, self.tile_n, self.dO_stage),
                stride=(1, 0, cute.round_up(self.tile_m, 64)),
            ),
        )
        # if const_expr(self.SdP_swapAB):
        if const_expr(True):
            sLSE_2D = layout_utils.transpose_view(sLSE_2D)
            sdPsum_2D = layout_utils.transpose_view(sdPsum_2D)

        # tix: [128...384]  8 warps
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())  # 4-11
        tidx = cute.arch.thread_idx()[0] % (cute.arch.WARP_SIZE * len(self.compute_warp_ids))
        # tidx = cute.arch.thread_idx()[0] - (cute.arch.WARP_SIZE * self.compute_warp_ids[0])
        dp_idx = tidx % 128
        num_wg = len(self.compute_warp_ids) // 4  # 2
        wg_idx = tidx // 128
        # wg_idx:
        # 0: [256...384]
        # 1: [128...256]

        tileP_f32_like = self.cta_tiler[1] // 32 * self.v_dtype.width  # (128, 64)
        # tStS has shape ((128, 128), 1, 1), tStP has shape ((128, 64), 1, 1)
        # tP overlap with tS
        # ((128,128),1,1):((65536,1),0,0)
        # (128,64):(1,128)
        tStP = cute.composition(tStS, (cute.make_layout((self.tile_n, tileP_f32_like)), 1, 1))
        # ((128,128),1,1):((65536,1),0,0) o (128,64):(1,128) => ((128,64),1,1):((65536,1),0,0)
        tStP = cute.make_tensor(tStS.iterator, tStP.layout)  # Otherwise the tmem address is wrong
        tScS = thr_mma_S.partition_C(cute.make_identity_tensor(self.mma_tiler_kq[:2]))
        tScP = cute.composition(tScS, (cute.make_layout((self.tile_n, tileP_f32_like)), 1, 1))
        # tdS overlap with tdP
        tdPtdS = cute.composition(tdPtdP, (cute.make_layout((self.tile_n, tileP_f32_like)), 1, 1))
        tdPcdP = thr_mma_dP.partition_C(cute.make_identity_tensor(self.mma_tiler_vdo[:2]))
        tdPcdS = cute.composition(tdPcdP, (cute.make_layout((self.tile_n, tileP_f32_like)), 1, 1))


        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), Float32
        )
        tmem_store_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(16)), Float32
        )

        if const_expr(self.folded_kv_acc):
            # copy_utils.make_tmem_copy's fixed 128-datapath tiler mis-slices a
            # folded accumulator -- measured on SM100, it walks every OTHER column
            # (coords (0,0), (0,2), (0,4), ... instead of (0,0), (0,1), (0,2), ...).
            # Use the folded-aware tiler tcgen05 derives from the tensor instead.
            #
            # This copy covers 128 threads: each lane gets its 64 columns as two
            # 32-column chunks. With split_folded_compute_wg (d256) warpgroup g takes
            # chunk g, so S / dP are loaded, exp'd and stored once instead of by
            # both warpgroups. A
            # thread's lane half is still its m half, which the dS exchange relies on.
            # Without the split both groups recompute the same values and store them
            # to the same addresses (idempotent).
            tiled_copy_t2r = tcgen05.make_tmem_copy(tmem_load_atom, tStS)
            thr_copy_t2r = tiled_copy_t2r.get_slice(dp_idx)
            tStS_t2r = thr_copy_t2r.partition_S(tStS)
            tdPtdP_t2r = thr_copy_t2r.partition_S(tdPtdP)
            tScS_t2r = thr_copy_t2r.partition_D(tScS)
            t0ScS_t2r = tiled_copy_t2r.get_slice(0).partition_D(tScS)
            tSsLSE = thr_copy_t2r.partition_D(thr_mma_S.partition_C(sLSE_2D))
            tSsdPsum = thr_copy_t2r.partition_D(thr_mma_dP.partition_C(sdPsum_2D))
            if const_expr(self.split_folded_compute_wg):
                tStS_t2r = self.split_wg(tStS_t2r, wg_idx, num_wg)
                tdPtdP_t2r = self.split_wg(tdPtdP_t2r, wg_idx, num_wg)
                tScS_t2r = self.split_wg(tScS_t2r, wg_idx, num_wg)
                t0ScS_t2r = self.split_wg(t0ScS_t2r, 0, num_wg)
                tSsLSE = self.split_wg(tSsLSE, wg_idx, num_wg)
                tSsdPsum = self.split_wg(tSsdPsum, wg_idx, num_wg)
            # P never reaches TMEM on this path (see mma_P_from_smem).
            thr_copy_r2t = None
            tScP_r2t, tStP_r2t = None, None
            tdPcdS_r2t, tdPtdS_r2t = None, None
        else:
            # tmem -> rmem
            thr_copy_t2r = copy_utils.make_tmem_copy(tmem_load_atom, num_wg).get_slice(tidx)
            tStS_t2r = thr_copy_t2r.partition_S(tStS)  # (((32, 32), 1), 2, 1, 1)
            tdPtdP_t2r = thr_copy_t2r.partition_S(tdPtdP)
            tScS_t2r = thr_copy_t2r.partition_D(tScS)  # ((32, 1), 2, 1, 1)
            t0ScS_t2r = thr_copy_t2r.get_slice(0).partition_D(tScS)  # ((32, 1), 2, 1, 1)
            # ((32, 1), 2, 1, 1, STAGE)
            tSsLSE = thr_copy_t2r.partition_D(thr_mma_S.partition_C(sLSE_2D))
            tSsdPsum = thr_copy_t2r.partition_D(thr_mma_dP.partition_C(sdPsum_2D))
            # rmem -> tmem
            thr_copy_r2t = copy_utils.make_tmem_copy(tmem_store_atom, num_wg).get_slice(tidx)
            tScP_r2t = thr_copy_r2t.partition_S(tScP)
            tStP_r2t = thr_copy_r2t.partition_D(tStP)
            tdPcdS_r2t = thr_copy_r2t.partition_S(tdPcdS)
            tdPtdS_r2t = thr_copy_r2t.partition_D(tdPtdS)
        # rmem -> smem
        # This part is a bit iffy, we might be making a lot of assumptions here
        copy_atom_r2s = sm100_utils_basic.get_smem_store_op(
            LayoutEnum.ROW_MAJOR, self.ds_dtype, Float32, thr_copy_t2r
        )
        if const_expr(self.folded_kv_acc):
            thr_copy_r2s = cute.make_tiled_copy_D(
                copy_atom_r2s, tiled_copy_t2r
            ).get_slice(dp_idx)
            # Both dS (A of the dK MMA) and P (A of the dV MMA) are written through
            # the MMA's own A-operand layout, addressed by its natural coordinate.
            # Drop the trailing singleton modes so a [None, stage] slice works, the
            # same shape the non-folded sdS_epi partition has.
            tRS_sdS = thr_copy_r2s.partition_D(
                self.smem_A_mn_view(sdSt, self.ds_dtype, sdSt_swizzle)
            )
            tRS_sP = thr_copy_r2s.partition_D(
                self.smem_A_mn_view(tP, self.do_dtype, tP_swizzle)
            )
            if const_expr(self.split_folded_compute_wg):
                tRS_sdS = self.split_wg(tRS_sdS, wg_idx, num_wg)
                tRS_sP = self.split_wg(tRS_sP, wg_idx, num_wg)
            tRS_sdS = tRS_sdS[None, None, 0, 0]
            tRS_sP = tRS_sP[None, None, 0, 0]
            # The epi view only exists on the non-folded path; compute_step takes
            # it as None.
            sdS_epi_layout = None
        else:
            thr_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, thr_copy_t2r).get_slice(tidx)
            # We assume the swizzle (i.e. layout.inner) stays the same
            sdS_epi_layout = sm100_utils_basic.make_smem_layout_epi(
                self.ds_dtype, LayoutEnum.ROW_MAJOR, (self.tile_n, self.tile_m), 1
            )
            sdS_layout = cute.slice_(sdS_epi_layout.outer, (None, None, 0))  # ((8,16), (64,2))
            # Need to group into 1 mode to be compatible w thr_copy_r2s
            sdS_layout = cute.make_layout((sdS_layout.shape,), stride=(sdS_layout.stride,))
            sdS_epi = cute.make_tensor(sdS.iterator, sdS_layout)
            tRS_sdS = thr_copy_r2s.partition_D(sdS_epi)
            tRS_sP = None

        tRS_sdS_xchg = None
        tRS_sdS_dq = None
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        dS_cluster_empty_phase = Int32(1)
        if const_expr(self.folded_kv_acc):
            # Folded: a thread's m half is its lane half, not a stage index, so the
            # dS store for the dQ MMA picks a destination instead of a slice:
            # our own m half goes to our sdS block cta_rank, the peer's m half goes
            # to sdS_xchg and from there to the peer's sdS block cta_rank.
            dS_block_elems = const_expr(
                self.tile_n * (128 // (self.ds_dtype.width // 8))
            )
            tRS_sdS_dq = thr_copy_r2s.partition_D(
                self.smem_dS_dq_block_view(
                    sdS.iterator + cta_rank_in_cluster * dS_block_elems,
                    sdS_swizzle,
                )
            )
            if const_expr(self.split_folded_compute_wg):
                tRS_sdS_dq = self.split_wg(tRS_sdS_dq, wg_idx, num_wg)
            tRS_sdS_dq = tRS_sdS_dq[None, None, 0, 0]
            if const_expr(self.dS_xchg_from_sdSt):
                # The peer's m half of our dS already sits in sdSt block peer_rank, in
                # the peer's dQ A-operand layout: send it from there, nothing to stage.
                # The layout is built here: self.sdS_xchg_layout is a host-side value
                # and cannot be used inside the kernel region.
                sdS_xchg = cute.make_tensor(
                    sdSt.iterator + (cta_rank_in_cluster ^ 1) * dS_block_elems,
                    cute.make_layout((self.tile_n, self.tile_m // 2)),
                )
            else:
                tRS_sdS_xchg = thr_copy_r2s.partition_D(
                    self.smem_dS_dq_block_view(sdS_xchg.iterator, sdS_swizzle)
                )[None, None, 0, 0]
        elif const_expr(self.use_2cta_instrs):
            sdS_xchg_epi = cute.make_tensor(
                cute.recast_ptr(sdS_xchg.iterator, sdS_epi_layout.inner), sdS_layout
            )
            tRS_sdS_xchg = thr_copy_r2s.partition_D(sdS_xchg_epi)

        # 2-CTA: CTA 0 exchanges stage 1 (bottom half), CTA 1 exchanges stage 0 (top half)
        exchange_stage = cta_rank_in_cluster ^ 1 if const_expr(self.use_2cta_instrs) else Int32(0)

        consumer_state_S_P_dP = pipeline.make_pipeline_state(  # Our impl has shortcut for stage==1
            cutlass.pipeline.PipelineUserType.Consumer, 1
        )
        # consumer_phase_S_P_dP = Int32(0)
        producer_state_dS = pipeline.make_pipeline_state(  # Our impl has shortcut for stage==1
            cutlass.pipeline.PipelineUserType.Producer, 1
        )
        consumer_state_dKV = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, 2
        )
        consumer_state_LSE = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.Q_stage
        )
        # consumer_state_dPsum = cutlass.pipeline.make_pipeline_state(
        consumer_state_dPsum = pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.dO_stage
        )
        if const_expr(self.enable_flashmask):
            flashmask_phase = Int32(0)

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            n_block, head_idx, batch_idx, _ = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            m_block_min, m_block_max = block_info.get_m_block_min_max(
                seqlen, n_block // self.cluster_shape_mnk[0]
            )
            n_block_for_cluster = n_block // self.cta_group_size
            mask = AttentionMaskCls(seqlen.seqlen_q, seqlen.seqlen_k)
            # TODO: condition mask_seqlen
            mask_fn = partial(
                mask.apply_mask_sm100_transposed,
                tScS_t2r=tScS_t2r,
                t0ScS_t2r=t0ScS_t2r,
                n_block=n_block_for_cluster,
                mask_seqlen=True,
                mask_causal=self.is_causal,
                mask_local=self.is_local,
                sStartEndRowIndices=sStartEndRowIndices,
                per_cta_tile_n=self.tile_n,
                has_ut_start=self.has_ut_start,
            )

            # prefetch_LSE = not self.is_causal
            prefetch_LSE = False

            compute_step = partial(
                self.compute_step,
                thr_copy_t2r=thr_copy_t2r,
                thr_copy_r2t=thr_copy_r2t,
                tScS_t2r=tScS_t2r,
                tStS_t2r=tStS_t2r,
                tScP_r2t=tScP_r2t,
                tStP_r2t=tStP_r2t,
                tSsLSE=tSsLSE,
                tRS_sdS=tRS_sdS,
                tRS_sP=tRS_sP,
                tdPtdS_r2t=tdPtdS_r2t,
                tdPtdP_t2r=tdPtdP_t2r,
                tSsdPsum=tSsdPsum,
                prefetch_LSE=prefetch_LSE,
                pipeline_LSE=pipeline_LSE,
                pipeline_S_P=pipeline_S_P,
                pipeline_dPsum=pipeline_dPsum,
                pipeline_dP=pipeline_dP,
                pipeline_dS=pipeline_dS,
                softmax_scale_log2=softmax_scale_log2,
                mask_fn=mask_fn,
                sdS_xchg=sdS_xchg,
                tRS_sdS_xchg=tRS_sdS_xchg,
                tRS_sdS_dq=tRS_sdS_dq,
                exchange_stage=exchange_stage,
                dS_cluster_empty_mbar_ptr=dS_cluster_empty_mbar_ptr,
                dS_cluster_full_mbar_ptr=dS_cluster_full_mbar_ptr,
                dQaccum_empty_mbar_ptr=dQaccum_empty_mbar_ptr,
                sdS_epi_layout=sdS_epi_layout,
                cta_rank_in_cluster=cta_rank_in_cluster,
                tidx=tidx,
                sdS=sdS,
            )

            if const_expr(True):  # Both CTAs must execute compute body for pipeline sync in 2-CTA mode
                zero_block = m_block_max <= m_block_min
                if const_expr(self.enable_flashmask):
                    cute.arch.mbarrier_wait(flashmask_loaded_mbar_ptr, flashmask_phase)
                    if const_expr(self.use_2cta_instrs):
                        # 2CTA: one flat loop over the blocks that are not fully masked.
                        # sFM_max_min holds the CTA pair's combined bounds (load_fm), so
                        # both CTAs walk the same blocks -- required, since they share the
                        # pipelines and the cta_group::2 MMAs. The iteration count and the
                        # m_block sequence therefore MUST stay exactly what fm_skip_info /
                        # fm_m_block produce (load, mma and relay derive theirs from the
                        # same formula). Only the per-block work varies: fm_needs_mask
                        # picks out the blocks that no key row of the pair masks, and those
                        # skip the element mask entirely.
                        fm_skip = self.fm_skip_info(
                            flashmask_info, sFM_max_min, m_block_min, m_block_max
                        )
                        for it in cutlass.range(fm_skip[6], unroll=1):
                            m_block = self.fm_m_block(fm_skip, m_block_min, it)
                            zero_block = False
                            needs_mask = self.fm_needs_mask(
                                flashmask_info, sFM_max_min, m_block
                            )
                            consumer_state_LSE, consumer_state_S_P_dP, consumer_state_dPsum, producer_state_dS, dS_cluster_empty_phase = compute_step(
                                m_block=m_block,
                                consumer_state_LSE=consumer_state_LSE,
                                consumer_state_S_P_dP=consumer_state_S_P_dP,
                                consumer_state_dPsum=consumer_state_dPsum,
                                producer_state_dS=producer_state_dS,
                                dS_cluster_empty_phase=dS_cluster_empty_phase,
                                partially_masked=needs_mask,
                                iter_idx=it,
                            )
                    else:
                        # 1CTA: iterate flashmask segments, skipping full-mask blocks
                        loop_start = m_block_min
                        loop_end = m_block_max
                        zero_block = True
                        # 0: 0 ~ UTS_min, no mask
                        # 1: UTS_min ~ UTS_max, partially mask
                        # 2: UTE_min ~ UTE_max, partially mask
                        # 3: UTE_max ~ LTS_min, no mask
                        # 4: LTS_min ~ LTS_max, partially mask
                        # 5: LTE_min ~ LTE_max, partially mask
                        # 6: LTE_max ~ max_seq_k, no mask
                        if const_expr(not self.is_causal):
                            has_uts = const_expr(flashmask_info.UTS_nblock_max is not None)
                            if has_uts:
                                # 0 ~ UTS
                                loop_end = sFM_max_min[5] # UTS_min
                                for m_block in cutlass.range(loop_start, loop_end, unroll=1):
                                    zero_block = False
                                    consumer_state_LSE, consumer_state_S_P_dP, consumer_state_dPsum, producer_state_dS, dS_cluster_empty_phase = compute_step(
                                        m_block=m_block,
                                        consumer_state_LSE=consumer_state_LSE,
                                        consumer_state_S_P_dP=consumer_state_S_P_dP,
                                        consumer_state_dPsum=consumer_state_dPsum,
                                        producer_state_dS=producer_state_dS,
                                        dS_cluster_empty_phase=dS_cluster_empty_phase,
                                        partially_masked=False,
                                    )

                                loop_start = sFM_max_min[5] # UTS_min
                                loop_end = sFM_max_min[4] + 1 # UTS_max
                                for m_block in cutlass.range(loop_start, loop_end, unroll=1):
                                    zero_block = False
                                    consumer_state_LSE, consumer_state_S_P_dP, consumer_state_dPsum, producer_state_dS, dS_cluster_empty_phase = compute_step(
                                        m_block=m_block,
                                        consumer_state_LSE=consumer_state_LSE,
                                        consumer_state_S_P_dP=consumer_state_S_P_dP,
                                        consumer_state_dPsum=consumer_state_dPsum,
                                        producer_state_dS=producer_state_dS,
                                        dS_cluster_empty_phase=dS_cluster_empty_phase,
                                        partially_masked=True,
                                    )
                                # Advance past UTS_max: with 4 vectors the upper band's
                                # [UTE_min, UTE_max] range can overlap [UTS_min, UTS_max],
                                # and those blocks are already done. Same clamp as in the
                                # mma block count and the load walk.
                                loop_start = loop_end

                            loop_start = max(loop_start, sFM_max_min[7]) # UTE_min
                            loop_end = min(sFM_max_min[6] + 1, m_block_max) # UTE_max
                            for m_block in cutlass.range(loop_start, loop_end, unroll=1):
                                zero_block = False
                                consumer_state_LSE, consumer_state_S_P_dP, consumer_state_dPsum, producer_state_dS, dS_cluster_empty_phase = compute_step(
                                    m_block=m_block,
                                    consumer_state_LSE=consumer_state_LSE,
                                    consumer_state_S_P_dP=consumer_state_S_P_dP,
                                    consumer_state_dPsum=consumer_state_dPsum,
                                    producer_state_dS=producer_state_dS,
                                    dS_cluster_empty_phase=dS_cluster_empty_phase,
                                    partially_masked=True,
                                )
                            loop_start = max(loop_start, loop_end)

                        # UTE ~ LTS
                        loop_end = min(m_block_max, sFM_max_min[1])
                        for m_block in cutlass.range(loop_start, loop_end, unroll=1):
                            zero_block = False
                            consumer_state_LSE, consumer_state_S_P_dP, consumer_state_dPsum, producer_state_dS, dS_cluster_empty_phase = compute_step(
                                m_block=m_block,
                                consumer_state_LSE=consumer_state_LSE,
                                consumer_state_S_P_dP=consumer_state_S_P_dP,
                                consumer_state_dPsum=consumer_state_dPsum,
                                producer_state_dS=producer_state_dS,
                                dS_cluster_empty_phase=dS_cluster_empty_phase,
                                partially_masked=False,
                            )

                        loop_start = max(loop_start, loop_end)
                        loop_end = min(m_block_max, sFM_max_min[0] + 1)
                        for m_block in cutlass.range(loop_start, loop_end, unroll=1):
                            zero_block = False
                            consumer_state_LSE, consumer_state_S_P_dP, consumer_state_dPsum, producer_state_dS, dS_cluster_empty_phase = compute_step(
                                m_block=m_block,
                                consumer_state_LSE=consumer_state_LSE,
                                consumer_state_S_P_dP=consumer_state_S_P_dP,
                                consumer_state_dPsum=consumer_state_dPsum,
                                producer_state_dS=producer_state_dS,
                                dS_cluster_empty_phase=dS_cluster_empty_phase,
                                partially_masked=True,
                            )

                        # LTE ~ seqlen_q
                        has_lte = const_expr(flashmask_info.LTE_nblock_max is not None)
                        if has_lte:
                            loop_start = max(sFM_max_min[0] + 1, sFM_max_min[3])
                            if sFM_max_min[3] == sFM_max_min[0]:
                                loop_start = sFM_max_min[3] + 1
                            loop_start = max(loop_start, m_block_min)
                            loop_end = min(m_block_max, sFM_max_min[2] + 1)
                            #loop_end = m_block_max
                            for m_block in cutlass.range(loop_start, loop_end, unroll=1):
                                zero_block = False
                                consumer_state_LSE, consumer_state_S_P_dP, consumer_state_dPsum, producer_state_dS, dS_cluster_empty_phase = compute_step(
                                    m_block=m_block,
                                    consumer_state_LSE=consumer_state_LSE,
                                    consumer_state_S_P_dP=consumer_state_S_P_dP,
                                    consumer_state_dPsum=consumer_state_dPsum,
                                    producer_state_dS=producer_state_dS,
                                    dS_cluster_empty_phase=dS_cluster_empty_phase,
                                    partially_masked=True,
                                )

                            loop_start = max(loop_start, loop_end)
                            loop_end = m_block_max
                            for m_block in cutlass.range(loop_start, loop_end, unroll=1):
                                zero_block = False
                                consumer_state_LSE, consumer_state_S_P_dP, consumer_state_dPsum, producer_state_dS, dS_cluster_empty_phase = compute_step(
                                    m_block=m_block,
                                    consumer_state_LSE=consumer_state_LSE,
                                    consumer_state_S_P_dP=consumer_state_S_P_dP,
                                    consumer_state_dPsum=consumer_state_dPsum,
                                    producer_state_dS=producer_state_dS,
                                    dS_cluster_empty_phase=dS_cluster_empty_phase,
                                    partially_masked=False,
                                )
                else:
                    # Mainloop
                    compute_iter_idx = cute.Int32(0)
                    for m_block in cutlass.range(m_block_min, m_block_max, unroll=1):
                        consumer_state_LSE, consumer_state_S_P_dP, consumer_state_dPsum, producer_state_dS, dS_cluster_empty_phase = compute_step(
                            m_block=m_block,
                            consumer_state_LSE=consumer_state_LSE,
                            consumer_state_S_P_dP=consumer_state_S_P_dP,
                            consumer_state_dPsum=consumer_state_dPsum,
                            producer_state_dS=producer_state_dS,
                            dS_cluster_empty_phase=dS_cluster_empty_phase,
                            iter_idx=compute_iter_idx,
                        )
                        compute_iter_idx = compute_iter_idx + 1

                # Final signal for dS smem store completion (deferred for 2CTA hdim128)
                if const_expr(self.use_2cta_instrs and self.tile_hdim == 128):
                    with cute.arch.elect_one():
                        pipeline_dS.producer_commit(producer_state_dS)
                    producer_state_dS.advance()

                if const_expr(self.use_2cta_instrs) or not zero_block:
                    if const_expr(not self.use_tma_store):
                        consumer_state_dKV = self.epilogue_dKV(
                            dp_idx,
                            warp_idx,
                            batch_idx,
                            head_idx,
                            n_block,
                            thr_mma_dV,
                            thr_mma_dK,
                            tdVtdV,
                            tdKtdK,
                            mdV,
                            mdK,
                            pipeline_dKV,
                            consumer_state_dKV,
                            softmax_scale,
                        )
                    else:
                        # With epi_num_strips == 2 the upper 64 threads serve the
                        # other hdim strip (its own SMEM buffer), so they reuse
                        # rows 0..63 of the tile: slice modulo the strip's threads.
                        thr_copy_r2s_dKV = tiled_copy_r2s_dKV.get_slice(
                            dp_idx % self.epi_threads_r2s
                        )

                    #### STORE dV
                    consumer_state_dKV = self.epilogue_dK_or_dV_tma(
                        dp_idx,
                        batch_idx,
                        head_idx,
                        n_block,
                        thr_mma_dV,
                        tdVtdV,
                        mdV_tma_tensor,
                        sdV,
                        tma_atom_dV,
                        thr_copy_r2s_dKV,
                        pipeline_dKV,
                        consumer_state_dKV,
                        None,  # Don't scale
                        int(NamedBarrierBwdSm100.EpilogueWG1),  # barrier_id
                        mdV_semaphore,
                        "V",
                    )
                    #### STORE dK
                    consumer_state_dKV = self.epilogue_dK_or_dV_tma(
                        dp_idx,
                        batch_idx,
                        head_idx,
                        n_block,
                        thr_mma_dK,
                        tdKtdK,
                        mdK_tma_tensor,
                        sdK,
                        tma_atom_dK,
                        thr_copy_r2s_dKV,
                        pipeline_dKV,
                        consumer_state_dKV,
                        softmax_scale if const_expr(not self.dKV_postprocess) else None,
                        int(NamedBarrierBwdSm100.EpilogueWG1),  # barrier_id
                        mdK_semaphore,
                        "K",
                    )
                if const_expr(self.zero_skipped_dKV):
                    if zero_block:
                        self.store_zero_dK_or_dV(
                            tidx, batch_idx, head_idx, n_block, seqlen.seqlen_k, mdV, "V"
                        )
                        self.store_zero_dK_or_dV(
                            tidx, batch_idx, head_idx, n_block, seqlen.seqlen_k, mdK, "K"
                        )
                if const_expr(
                    self.deterministic
                    and self.dKV_postprocess
                    and self.enable_flashmask
                    and not self.use_2cta_instrs
                ):
                    # A per-q-head mask may skip this tile for one q-head only; it must
                    # still take its turn in the dK/dV accumulate order.
                    if zero_block:
                        self.pass_dKV_semaphore(
                            dp_idx, wg_idx, batch_idx, head_idx, n_block, mdV_semaphore
                        )
                        self.pass_dKV_semaphore(
                            dp_idx, wg_idx, batch_idx, head_idx, n_block, mdK_semaphore
                        )
                if const_expr(self.enable_flashmask):
                    flashmask_phase ^= 1

            if const_expr(self.tile_boundary_sync):
                # dKV epilogue is done: sdK / sdV (== sK / sV) are free again.
                self.tile_boundary_barrier.arrive_and_wait()
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    @cute.jit
    def compute_step(
        self,
        m_block: cute.Int32,
        thr_copy_t2r: cute.TiledCopy,
        thr_copy_r2t: cute.TiledCopy,
        tScS_t2r: cute.Tensor,
        tStS_t2r: cute.Tensor,
        tScP_r2t: cute.Tensor,
        tStP_r2t: cute.Tensor,
        tSsLSE: cute.Tensor,
        tRS_sdS: cute.Tensor,
        tdPtdS_r2t: cute.Tensor,
        tdPtdP_t2r: cute.Tensor,
        tSsdPsum: cute.Tensor,
        prefetch_LSE: bool,
        pipeline_LSE: PipelineAsync,
        pipeline_S_P: PipelineAsync,
        pipeline_dPsum: PipelineAsync,
        pipeline_dP: PipelineAsync,
        pipeline_dS: PipelineAsync,
        softmax_scale_log2: cutlass.Float32,
        consumer_state_LSE: cutlass.pipeline.PipelineState,
        consumer_state_S_P_dP: cutlass.pipeline.PipelineState,
        consumer_state_dPsum: cutlass.pipeline.PipelineState,
        producer_state_dS: cutlass.pipeline.PipelineState,
        mask_fn: Callable,
        # bool for the 1CTA segment walk (compile-time branch in the mask), or a
        # dynamic cutlass.Boolean for the 2CTA flat loop (runtime branch around the
        # mask's unrolled element loop). Do NOT annotate: a `bool` annotation would
        # make the DSL treat the 2CTA value as constexpr.
        partially_masked = False,
        sdS_xchg: cute.Tensor = None,
        tRS_sdS_xchg = None,
        tRS_sdS_dq = None,
        exchange_stage: cute.Int32 = Int32(0),
        dS_cluster_empty_mbar_ptr: cute.Pointer = None,
        dS_cluster_full_mbar_ptr: cute.Pointer = None,
        dQaccum_empty_mbar_ptr: Optional[cute.Pointer] = None,
        sdS_epi_layout = None,
        dS_cluster_empty_phase: cute.Int32 = Int32(1),
        cta_rank_in_cluster: cute.Int32 = Int32(0),
        tidx: cute.Int32 = Int32(0),
        tRS_sP: cute.Tensor = None,
        sdS: cute.Tensor = None,
        iter_idx: cute.Int32 = Int32(0),
    ):
        # Prefetch 1 stage of LSE
        pipeline_LSE.consumer_wait(consumer_state_LSE)
        tSrLSE_s2r = cute.make_rmem_tensor(tScS_t2r[None, 0, 0, 0].shape, Float32)
        if const_expr(prefetch_LSE and not self.shuffle_LSE):
            cute.autovec_copy(tSsLSE[None, 0, 0, 0, consumer_state_LSE.index], tSrLSE_s2r)

        pipeline_S_P.consumer_wait(consumer_state_S_P_dP)
        #### TMEM->RMEM (Load S from TMEM)
        tSrS_t2r = cute.make_rmem_tensor(tScS_t2r.shape, Float32)
        cute.copy(thr_copy_t2r, tStS_t2r, tSrS_t2r)

        if const_expr(self.early_S_P_release):
            # S is in registers now. d192: dQ shares TMEM columns with S/P, so S_P has
            # to be released before P is written over it. d256: P goes to SMEM and the
            # dP/dS columns are disjoint from S, so releasing here lets the dP MMA run
            # under the mask / exp / P store below instead of after it.
            cute.arch.fence_view_async_tmem_load()
            with cute.arch.elect_one():
                pipeline_S_P.consumer_release(consumer_state_S_P_dP)
        if const_expr(self.use_2cta_instrs and not self.use_2cta_bigd):
            # 2-CTA small hdim: pipeline_dS doubles as the "S is out of TMEM" signal to
            # the MMA warp, because dQ time-shares S/P's columns there. The commit for
            # iteration i-1 is therefore deferred to here, after this iteration's S t2r.
            if iter_idx > 0:
                cute.arch.fence_view_async_tmem_load()
                with cute.arch.elect_one():
                    pipeline_dS.producer_commit(producer_state_dS)
                producer_state_dS.advance()

        #### APPLY MASK
        mask_fn(tSrS_t2r, m_block=m_block, partially_masked=partially_masked)

        num_stages = cute.size(tScS_t2r, mode=[1])

        # ---------------------------------------------
        #### P = exp(S - LSE)
        # ---------------------------------------------
        lane_idx = cute.arch.lane_idx()
        if const_expr(self.folded_kv_acc):
            # Folded accumulator: the CTA owns tile_n TMEM lanes and the copy covers
            # 2 * tile_n threads, the upper half being the fold, so a thread's m half
            # is exactly which half of the datapath index it sits in.
            m_half_idx = (tidx % 128) // self.tile_n
        if const_expr(not self.mma_P_from_smem):
            tSrP_r2t_f32 = cute.make_rmem_tensor(tScP_r2t.shape, Float32)  # 64
            tSrP_r2t = cute.recast_tensor(tSrP_r2t_f32, self.q_dtype)
        for stage in cutlass.range_constexpr(num_stages):
            tSrS_cur = tSrS_t2r[None, stage, 0, 0]
            tSsLSE_cur = tSsLSE[None, stage, 0, 0, consumer_state_LSE.index]
            if const_expr(not self.shuffle_LSE):
                if const_expr(stage > 0 or not prefetch_LSE):
                    cute.autovec_copy(tSsLSE_cur, tSrLSE_s2r)
                tSrLSE = tSrLSE_s2r
            else:
                tSrLSE = tSsLSE_cur[lane_idx]
            for v in cutlass.range_constexpr(cute.size(tSrS_t2r, mode=[0]) // 2):
                if const_expr(not self.shuffle_LSE):
                    lse_pair = (tSrLSE[2 * v], tSrLSE[2 * v + 1])
                else:
                    lse_pair = (
                        utils.shuffle_sync(tSrLSE, offset=2 * v),
                        utils.shuffle_sync(tSrLSE, offset=2 * v + 1),
                    )
                tSrS_cur[2 * v], tSrS_cur[2 * v + 1] = utils.fma_packed_f32x2(
                    ((tSrS_cur[2 * v], tSrS_cur[2 * v + 1])),
                    (softmax_scale_log2, softmax_scale_log2),
                    (-lse_pair[0], -lse_pair[1]),
                )
                tSrS_cur[2 * v] = cute.math.exp2(tSrS_cur[2 * v], fastmath=True)
                tSrS_cur[2 * v + 1] = cute.math.exp2(tSrS_cur[2 * v + 1], fastmath=True)
            if const_expr(self.mma_P_from_smem):
                # P goes straight to SMEM as the dV MMA's A operand: a folded S
                # accumulator cannot hold it (see mma_P_from_smem).
                tSrP_cvt = cute.make_fragment_like(tSrS_cur, self.q_dtype)
                utils.cvt_f16(tSrS_cur, tSrP_cvt)
                cute.autovec_copy(tSrP_cvt, tRS_sP[None, stage])
            else:
                utils.cvt_f16(tSrS_cur, tSrP_r2t[None, stage, 0, 0])
                if const_expr(stage == 0):
                    cute.arch.fence_view_async_tmem_load()
                    # Without this barrier, we could have 1 warp writing to P in tmem while
                    # another warp is still reading S from tmem.
                    self.compute_sync_barrier.arrive_and_wait()
                cute.copy(
                    thr_copy_r2t,
                    tSrP_r2t_f32[None, stage, None, None],
                    tStP_r2t[None, stage, None, None],
                )

        if const_expr(self.mma_P_from_smem):
            # P was stored to SMEM, not TMEM: make those stores visible to the
            # async (MMA) proxy instead.
            cute.arch.fence_view_async_shared()
        else:
            cute.arch.fence_view_async_tmem_store()
        if const_expr(self.mma_P_from_smem and self.early_S_P_release):
            # Nothing cross-warp left to order here: S_P is already released, and P's
            # readiness reaches the dV MMA through each warp's own pipeline_dP arrival
            # (after its P / dS stores and fences). The LSE mbarrier takes one arrival
            # per warp, so a warp sync is enough.
            cute.arch.sync_warp()
        else:
            self.compute_sync_barrier.arrive_and_wait()

        if const_expr(not self.early_S_P_release):
            # Late S_P release: P is written back over S in TMEM.
            with cute.arch.elect_one():
                pipeline_S_P.consumer_release(consumer_state_S_P_dP)
        pipeline_LSE.consumer_release(consumer_state_LSE)
        consumer_state_LSE.advance()

        # ---------------------------------------------
        # dS.T = P.T * (dP.T - D)
        # ---------------------------------------------
        pipeline_dPsum.consumer_wait(consumer_state_dPsum)

        pipeline_dP.consumer_wait(consumer_state_S_P_dP)
        # pipeline_dP.sync_object_full.wait(0, consumer_phase_S_P_dP)
        ### Now delayed to after loop
        # consumer_state_S_P_dP.advance()
        # if const_expr(self.use_2cta_instrs):
        #     cute.arch.mbarrier_wait(dS_cluster_empty_mbar_ptr, phase=dS_cluster_empty_phase)
        #     dS_cluster_empty_phase ^= 1

        ##### dS.T = P.T * (dP.T - Psum)
        for stage in cutlass.range_constexpr(num_stages):
            tdPrdP_t2r = cute.make_rmem_tensor(tScS_t2r[None, 0, None, None].shape, Float32)
            cute.copy(thr_copy_t2r, tdPtdP_t2r[None, stage, None, None], tdPrdP_t2r)
            cute.arch.fence_view_async_tmem_load()
            if const_expr(not self.mma_dS_from_smem):
                # dS is written back over dP in TMEM, with the same cross-warp lane
                # sharing as P over S: every warp must have loaded dP first. With dS in
                # SMEM (d256) nothing overwrites dP, so no barrier is needed.
                self.compute_sync_barrier.arrive_and_wait()
            tdPrdP_cur = tdPrdP_t2r[None, 0, 0]
            tSrS_cur = tSrS_t2r[None, stage, 0, 0]
            tSsdPsum_cur = tSsdPsum[None, stage, 0, 0, consumer_state_dPsum.index]
            if const_expr(not self.shuffle_dPsum):
                tSrdPsum = cute.make_fragment_like(tSsdPsum_cur, Float32)
                cute.autovec_copy(tSsdPsum_cur, tSrdPsum)
            else:
                tSrdPsum = tSsdPsum_cur[lane_idx]
            for v in cutlass.range_constexpr(cute.size(tdPrdP_t2r, mode=[0]) // 2):
                if const_expr(not self.shuffle_dPsum):
                    dPsum_pair = (tSrdPsum[2 * v], tSrdPsum[2 * v + 1])
                else:
                    dPsum_pair = (
                        utils.shuffle_sync(tSrdPsum, offset=2 * v),
                        utils.shuffle_sync(tSrdPsum, offset=2 * v + 1),
                    )
                tdPrdP_cur[2 * v], tdPrdP_cur[2 * v + 1] = utils.sub_packed_f32x2(
                    (tdPrdP_cur[2 * v], tdPrdP_cur[2 * v + 1]), dPsum_pair
                )
                tdPrdP_cur[2 * v], tdPrdP_cur[2 * v + 1] = utils.mul_packed_f32x2(
                    (tSrS_cur[2 * v], tSrS_cur[2 * v + 1]),
                    (tdPrdP_cur[2 * v], tdPrdP_cur[2 * v + 1]),
                )
            tdPrdS_cvt = cute.make_fragment_like(tdPrdP_cur, self.ds_dtype)
            utils.cvt_f16(tdPrdP_cur, tdPrdS_cvt)
            if const_expr(stage == 0):
                pipeline_dS.producer_acquire(producer_state_dS)
                if const_expr(self.use_2cta_instrs and not self.folded_kv_acc):
                    tdPrdS_xchg = cute.make_fragment_like(tdPrdS_cvt, self.ds_dtype)
                if const_expr(
                    self.folded_kv_acc and self.use_2cta_bigd and not self.dS_xchg_from_sdSt
                ):
                    # sdS_xchg aliases sdQaccum on the bigd path and the folded store
                    # below goes straight into it, so the drain has to be done first.
                    # Removing the overlay to get rid of this wait was measured and is
                    # slower -- see the sdS_xchg note in __init__.
                    cute.arch.mbarrier_wait(
                        dQaccum_empty_mbar_ptr, phase=producer_state_dS.phase
                    )

            # RMEM->TMEM: only when the dK MMA reads dS from TMEM
            if const_expr(not self.mma_dS_from_smem):
                tdPrdS_r2t_f32 = cute.recast_tensor(tdPrdS_cvt, Float32)
                cute.copy(thr_copy_r2t, tdPrdS_r2t_f32, tdPtdS_r2t[None, stage, 0, 0])

            if const_expr(self.folded_kv_acc):
                # The dK MMA's A view wants every m row of our own n slice, so this
                # store is unconditional. The dQ MMA's A view wants our own m half
                # over both n slices: a thread's m half is its lane half, so half the
                # threads store locally and half store into the exchange buffer (or,
                # with dS_xchg_from_sdSt, nowhere: the peer's half is sent from sdSt).
                cute.autovec_copy(tdPrdS_cvt, tRS_sdS[None, stage])
                if const_expr(self.dS_xchg_from_sdSt):
                    if m_half_idx == cta_rank_in_cluster:
                        cute.autovec_copy(tdPrdS_cvt, tRS_sdS_dq[None, stage])
                else:
                    if m_half_idx == cta_rank_in_cluster:
                        cute.autovec_copy(tdPrdS_cvt, tRS_sdS_dq[None, stage])
                    else:
                        cute.autovec_copy(tdPrdS_cvt, tRS_sdS_xchg[None, stage])
            # RMEM->SMEM: For 2-CTA, keep exchange stage in registers, write non-exchange to sdS
            elif const_expr(self.use_2cta_instrs):
                if exchange_stage == stage:
                    cute.autovec_copy(tdPrdS_cvt, tdPrdS_xchg)
                else:
                    cute.autovec_copy(tdPrdS_cvt, tRS_sdS[None, stage])
            else:
                cute.autovec_copy(tdPrdS_cvt, tRS_sdS[None, stage])

        if const_expr(self.mma_dS_from_smem):
            # dS^T for the dK MMA is in SMEM: make this warp's stores visible to the
            # async proxy before the per-warp pipeline_dP arrival the MMA warp waits on
            # (there is no dS TMEM store to fence on this path).
            cute.arch.fence_view_async_shared()
        else:
            cute.arch.fence_view_async_tmem_store()

        if const_expr(self.use_2cta_instrs):
            # use pipeline_dP to signal the dS store (TMEM, or SMEM on the folded path)
            with cute.arch.elect_one():
                pipeline_dP.consumer_release(consumer_state_S_P_dP)
        consumer_state_S_P_dP.advance()

        # After the loop: copy exchange registers to sdS_xchg buffer. The folded path
        # stored straight into sdS_xchg inside the loop, so it has nothing to do here.
        if const_expr(self.use_2cta_instrs and not self.folded_kv_acc):
            if const_expr(self.use_2cta_bigd):
                cute.arch.mbarrier_wait(
                    dQaccum_empty_mbar_ptr, phase=producer_state_dS.phase
                )
            cute.autovec_copy(tdPrdS_xchg, tRS_sdS_xchg[None, 0])

        if const_expr(not (self.mma_dS_from_smem and self.folded_kv_acc)):
            # Folded (d256): every dS store happened in the loop above and is already
            # fenced by the mma_dS_from_smem fence before the pipeline_dP arrival, and
            # the block just above is compiled out, so a second fence adds nothing.
            cute.arch.fence_view_async_shared()
        self.compute_sync_barrier.arrive_and_wait()
        pipeline_dPsum.consumer_release(consumer_state_dPsum)
        consumer_state_dPsum.advance()
        # when 2cta hdim 128, pipeline_dS also signals S tmem load completion so is deferred
        if const_expr(not (self.use_2cta_instrs and self.tile_hdim == 128)):
            with cute.arch.elect_one():
                pipeline_dS.producer_commit(producer_state_dS)
            producer_state_dS.advance()

        # 2-CTA: DSMEM copy from sdS_xchg to peer's sdS buffer
        if const_expr(self.use_2cta_instrs):
            stage_copy_bytes = const_expr(self.tma_copy_bytes["dS"] // 2)
            stage_copy_elems = const_expr(stage_copy_bytes // (self.ds_dtype.width // 8))
            if tidx == 0:
                peer_cta_rank_in_cluster = cta_rank_in_cluster ^ 1
                smem_src_ptr = sdS_xchg.iterator
                # Destination is peer's sdS at our CTA's offset (exchange_stage position)
                smem_dst_ptr = sdS.iterator + cta_rank_in_cluster * stage_copy_elems
                cute.arch.mbarrier_arrive_and_expect_tx(
                    dS_cluster_full_mbar_ptr,
                    stage_copy_bytes,
                    peer_cta_rank_in_cluster=peer_cta_rank_in_cluster,
                )
                copy_utils.cpasync_bulk_s2cluster(
                    smem_src_ptr,
                    smem_dst_ptr,
                    dS_cluster_full_mbar_ptr,
                    stage_copy_bytes,
                    peer_cta_rank_in_cluster=peer_cta_rank_in_cluster,
                )



        return consumer_state_LSE, consumer_state_S_P_dP, consumer_state_dPsum, producer_state_dS, dS_cluster_empty_phase

    @cute.jit
    def dQacc_reduce(
        self,
        mdQaccum: cute.Tensor,
        sdQaccum: cute.Tensor,
        thr_mma_dQ: cute.ThrMma,
        tdQtdQ: cute.Tensor,
        pipeline_dQ: PipelineAsync,
        dQaccum_empty_mbar_ptr: Optional[cute.Pointer],
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        mdQ_semaphore: Optional[cute.Tensor],
        mdK: Optional[cute.Tensor],
        flashmask_info: FlashMaskInfo,
        sFM_max_min: cute.Tensor,
        flashmask_loaded_mbar_ptr: cute.Pointer,
        is_leader_cta: cutlass.Boolean,
        mdK_semaphore: Optional[cute.Tensor] = None,
        mdV: Optional[cute.Tensor] = None,
    ):
        num_reduce_threads = cute.arch.WARP_SIZE * len(self.reduce_warp_ids)
        tidx = cute.arch.thread_idx()[0] % num_reduce_threads
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx() % len(self.reduce_warp_ids))
        is_tma_warp = warp_idx == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        # TMEM -> RMEM
        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(self.dQ_reduce_ncol_t2r)), Float32
        )
        thr_copy_t2r = tcgen05.make_tmem_copy(tmem_load_atom, tdQtdQ).get_slice(tidx)
        tdQtdQ_t2r = thr_copy_t2r.partition_S(tdQtdQ)
        tdQcdQ = thr_mma_dQ.partition_C(cute.make_identity_tensor(self.mma_tiler_dsk[:2]))
        tdQcdQ_t2r = thr_copy_t2r.partition_D(tdQcdQ)
        tdQrdQ_t2r_shape = tdQcdQ_t2r.shape
        expected_reduce_stages_t2r = self.dQaccum_reduce_stage_t2r // self.cta_group_size
        assert cute.size(tdQrdQ_t2r_shape, mode=[1]) == expected_reduce_stages_t2r, (
            "dQaccum t2r reduce stage mismatch"
        )
        expected_reduce_stages = self.dQaccum_reduce_stage // self.cta_group_size
        # 2-CTA: CTA 0 -> (M/2, D) (stage 0, 1) & CTA 1 -> (M/2, D) (stage 2, 3)
        stage_offset = (
            expected_reduce_stages * cta_rank_in_cluster if const_expr(self.use_2cta_instrs) else Int32(0)
        )

        thr_copy_dQaccum_r2s = copy_utils.tiled_copy_1d(
            self.dqaccum_dtype, num_reduce_threads, num_copy_elems=128 // self.dqaccum_dtype.width
        ).get_slice(tidx)
        tdQsdQ = thr_copy_dQaccum_r2s.partition_D(sdQaccum)

        read_flag = const_expr(not self.deterministic)
        if const_expr(self.enable_flashmask):
            flashmask_phase = Int32(0)

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        dQ_consumer_state = pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, 1
        )
        dQ_tma_store_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.sdQaccum_stage
        )
        while work_tile.is_valid_tile:
            n_block, head_idx, batch_idx, _ = work_tile.tile_idx
            n_block_cta_group = n_block // self.cta_group_size  # for 2cta
            seqlen = SeqlenInfoCls(batch_idx)
            m_block_min, m_block_max = block_info.get_m_block_min_max(
                seqlen, n_block_cta_group
            )

            mdQaccum_cur = mdQaccum[None, head_idx, batch_idx]
            gdQaccum_ = cute.local_tile(mdQaccum_cur, (self.tile_m * self.tile_hdim,), (None,))
            # (M * K / STAGE, STAGE, _)
            gdQaccum = cute.flat_divide(
                gdQaccum_, (self.tile_m * self.tile_hdim // self.dQaccum_reduce_stage,)
            )


            if const_expr(self.deterministic):
                mdQ_semaphore_cur = mdQ_semaphore[None, None, head_idx, batch_idx]
            else:
                mdQ_semaphore_cur = None

            # 2CTA d=192 uses dQaccum_empty_mbar_ptr to gate reduce/MMA on dQ tmem.
            delay_semaphore_release = not self.use_2cta_bigd
            n_block_global_max = cute.ceil_div(seqlen.seqlen_k, self.tile_n)


            dQacc_reduce_step = partial(
                self.dQacc_reduce_step,
                m_block_min=m_block_min,
                n_block=n_block,
                n_block_cta_group=n_block_cta_group,
                n_block_global_max=n_block_global_max,
                tidx=tidx,
                tdQrdQ_t2r_shape=tdQrdQ_t2r_shape,
                tdQcdQ_t2r=tdQcdQ_t2r,
                tdQtdQ_t2r=tdQtdQ_t2r,
                tdQsdQ=tdQsdQ,
                sdQaccum=sdQaccum,
                gdQaccum=gdQaccum,
                thr_copy_dQaccum_r2s=thr_copy_dQaccum_r2s,
                thr_copy_t2r=thr_copy_t2r,
                pipeline_dQ=pipeline_dQ,
                dQ_consumer_state=dQ_consumer_state,
                dQ_tma_store_producer_state=dQ_tma_store_producer_state,
                seqlen=seqlen,
                delay_semaphore_release=delay_semaphore_release,
                read_flag=read_flag,
                is_tma_warp=is_tma_warp,
                mdQ_semaphore_cur=mdQ_semaphore_cur,
                stage_offset=stage_offset,
                dQaccum_empty_mbar_ptr=dQaccum_empty_mbar_ptr,
                block_info=block_info,
            )

            if const_expr(self.enable_flashmask):
                cute.arch.mbarrier_wait(flashmask_loaded_mbar_ptr, flashmask_phase)
                # Walk every block but reduce only the ones the mma warp produced. The
                # skipped blocks still have to run the deterministic semaphore handshake
                # below, which is why this loop is written as a predicate instead of the
                # segment walk the load / mma / compute warps use. In 2CTA mode the bounds
                # are the CTA pair's combined bounds (load_fm), so both CTAs agree on the
                # skipped set.
                fm_skip = self.fm_skip_info(
                    flashmask_info, sFM_max_min, m_block_min, m_block_max
                )
                for m_block in cutlass.range(m_block_min, m_block_max, unroll=1):
                    full_mask = self.fm_is_full_mask(fm_skip, m_block)

                    if not full_mask:
                        dQ_consumer_state, dQ_tma_store_producer_state = dQacc_reduce_step(
                            m_block=m_block,
                            dQ_consumer_state=dQ_consumer_state,
                            dQ_tma_store_producer_state=dQ_tma_store_producer_state,
                        )

                    if const_expr(self.deterministic):
                        if full_mask:

                            if const_expr(self.spt):
                                _, n_block_max_for_m_block = block_info.get_n_block_min_max(
                                    seqlen, m_block
                                )
                                lock_value = n_block_max_for_m_block - 1 - n_block_cta_group
                            else:
                                lock_value = n_block_cta_group
                            barrier.wait_eq(
                                mdQ_semaphore_cur[(m_block, None)].iterator, tidx, cta_rank_in_cluster, lock_value
                            )

                            if const_expr(delay_semaphore_release):
                                if m_block > m_block_min:
                                    barrier.arrive_inc(
                                        mdQ_semaphore_cur[(m_block - 1, None)].iterator, tidx, cta_rank_in_cluster, 1
                                    )
                            else:
                                barrier.arrive_inc(
                                    mdQ_semaphore_cur[m_block, None].iterator,
                                    tidx,
                                    cta_rank_in_cluster,
                                    1,
                                )


            else:
                for m_block in cutlass.range(m_block_min, m_block_max, unroll=1):
                    dQ_consumer_state, dQ_tma_store_producer_state = dQacc_reduce_step(
                        m_block=m_block,
                        dQ_consumer_state=dQ_consumer_state,
                        dQ_tma_store_producer_state=dQ_tma_store_producer_state,
                    )

            if is_tma_warp:
                cute.arch.cp_async_bulk_wait_group(0, read=read_flag)
            self.reduce_sync_barrier.arrive_and_wait()
            # final semaphore release
            if const_expr(self.deterministic and delay_semaphore_release):
                barrier.arrive_inc(mdQ_semaphore_cur[(m_block_max - 1, None)].iterator, tidx, cta_rank_in_cluster, 1)

            if const_expr(self.enable_flashmask):
                flashmask_phase ^= 1

            if const_expr(self.tile_boundary_sync):
                # Done reading sFM / writing sdQaccum for this tile.
                self.tile_boundary_barrier.arrive_and_wait()
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    @cute.jit
    def dQacc_reduce_step(
        self,
        m_block: cute.Int32,
        m_block_min: cute.Int32,
        n_block: cute.Int32,
        n_block_cta_group: cute.Int32,
        n_block_global_max: cute.Int32,
        tidx: cute.Int32,
        tdQrdQ_t2r_shape: cute.Shape,
        tdQcdQ_t2r: cute.Tensor,
        tdQtdQ_t2r: cute.Tensor,
        tdQsdQ: cute.Tensor,
        sdQaccum: cute.Tensor,
        gdQaccum: cute.Tensor,
        thr_copy_dQaccum_r2s: cute.TiledCopy,
        thr_copy_t2r: cute.TiledCopy,
        pipeline_dQ: PipelineAsync,
        dQ_consumer_state: cutlass.pipeline.PipelineState,
        dQ_tma_store_producer_state: cutlass.pipeline.PipelineState,
        seqlen: SeqlenInfoQK,
        delay_semaphore_release: bool,
        read_flag: bool,
        is_tma_warp: bool,
        mdQ_semaphore_cur: Optional[cute.Tensor],
        stage_offset: cute.Int32,
        dQaccum_empty_mbar_ptr: Optional[cute.Pointer],
        block_info: BlockInfo,
    ):
        num_reduce_threads = cute.arch.WARP_SIZE * len(self.reduce_warp_ids)
        tidx = cute.arch.thread_idx()[0] % num_reduce_threads
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())

        hdim_for_reduce_shape = self.tile_hdim

        # Single reduce per M-block.
        num_reduces = 1

        for reduce_idx in cutlass.range_constexpr(num_reduces):
            pipeline_dQ.consumer_wait(dQ_consumer_state)

            gdAccum_cur = gdQaccum[None, None, m_block]
            cur_tma_bytes = self.tma_copy_bytes["dQ"]

            num_gmem_stages = const_expr(
                hdim_for_reduce_shape // self.cta_group_size // self.dQ_reduce_ncol
            )
            if const_expr(not self.split_dq_t2r):
                # TMEM -> RMEM, whole accumulator at once. See split_dq_t2r.
                tdQrdQ_t2r = cute.make_rmem_tensor(tdQrdQ_t2r_shape, Float32)
                cute.copy(thr_copy_t2r, tdQtdQ_t2r, tdQrdQ_t2r)
                cute.arch.fence_view_async_tmem_load()
                cute.arch.sync_warp()
                with cute.arch.elect_one():
                    pipeline_dQ.consumer_release(dQ_consumer_state)
                dQ_consumer_state.advance()
                tdQrdQ = cute.make_tensor(
                    tdQrdQ_t2r.iterator,
                    (self.dQ_reduce_ncol, num_gmem_stages),
                )

            for stage in cutlass.range_constexpr(num_gmem_stages):
                smem_idx = dQ_tma_store_producer_state.index
                tdQsdQ_r2s = tdQsdQ[None, None, smem_idx]
                if const_expr(self.split_dq_t2r):
                    # TMEM -> RMEM for just this stage, so only one stage's worth
                    # of the accumulator is live in registers at a time. The 4-mode
                    # slice and the shape-from-coordinate-tensor follow the same
                    # pattern the compute warp uses for dP (see tdPtdP_t2r).
                    tdQrdQ_t2r = cute.make_rmem_tensor(
                        tdQcdQ_t2r[None, 0, None, None].shape, Float32
                    )
                    cute.copy(
                        thr_copy_t2r,
                        tdQtdQ_t2r[None, stage, None, None],
                        tdQrdQ_t2r,
                    )
                    cute.arch.fence_view_async_tmem_load()
                    tdQrdQ_r2s = cute.make_tensor(
                        tdQrdQ_t2r.iterator, tdQsdQ_r2s.shape
                    )
                else:
                    tdQrdQ_r2s = cute.make_tensor(
                        tdQrdQ[None, stage].iterator, tdQsdQ_r2s.shape
                    )
                cute.copy(thr_copy_dQaccum_r2s, tdQrdQ_r2s, tdQsdQ_r2s)
                # Fence and barrier to make sure shared memory store is visible to TMA store
                cute.arch.fence_view_async_shared()
                # semaphore acquire (dQ single-reduce path)
                sem_active = const_expr(self.deterministic and stage == 0)
                if const_expr(sem_active):
                    if const_expr(self.spt):
                        _, n_block_max_for_m_block = block_info.get_n_block_min_max(
                            seqlen, m_block
                        )
                        lock_value = n_block_max_for_m_block - 1 - n_block_cta_group
                    else:
                        lock_value = n_block_cta_group
                    barrier.wait_eq(
                        mdQ_semaphore_cur[(m_block, None)].iterator,
                        tidx,
                        cta_rank_in_cluster,
                        lock_value,
                    )
                self.reduce_sync_barrier.arrive_and_wait()
                # Copy from shared memory to global memory
                if is_tma_warp:
                    with cute.arch.elect_one():
                        copy_utils.cpasync_reduce_bulk_add_f32(
                            sdQaccum[None, smem_idx].iterator,
                            gdAccum_cur[None, stage + stage_offset].iterator,
                            cur_tma_bytes // 1,
                        )
                    cute.arch.cp_async_bulk_commit_group()
                    cute.arch.cp_async_bulk_wait_group(
                        self.sdQaccum_stage - 1, read=read_flag
                    )
                self.reduce_sync_barrier.arrive_and_wait()
                dQ_tma_store_producer_state.advance()
                # semaphore release for prior m_block (only on dQ reduce)
                sem_release_active = const_expr(
                    self.deterministic
                    and stage == 0
                    and delay_semaphore_release
                )
                if const_expr(sem_release_active):
                    if m_block > m_block_min:
                        barrier.arrive_inc(
                            mdQ_semaphore_cur[(m_block - 1, None)].iterator,
                            tidx,
                            cta_rank_in_cluster,
                            1,
                        )

            if const_expr(self.split_dq_t2r):
                # dQ's TMEM is only free once every stage has been read out.
                cute.arch.sync_warp()
                with cute.arch.elect_one():
                    pipeline_dQ.consumer_release(dQ_consumer_state)
                dQ_consumer_state.advance()

        # 2CTA big-hdim: drain dQ-accum bulk reduces and arrive on dQaccum_empty
        # barrier so the compute warps can reuse sdQaccum as sdS_xchg. With
        # dS_xchg_from_sdSt nobody waits on it, so the in-flight reduce-adds stay
        # overlapped with the next block (unless deterministic needs them done).
        if const_expr(self.use_2cta_bigd and not self.dS_xchg_from_sdSt):
            if const_expr(self.sdQaccum_stage > 1):
                if is_tma_warp:
                    cute.arch.cp_async_bulk_wait_group(0, read=read_flag)
                self.reduce_sync_barrier.arrive_and_wait()
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive(dQaccum_empty_mbar_ptr)
        elif const_expr(
            self.use_2cta_bigd and self.deterministic and self.sdQaccum_stage > 1
        ):
            if is_tma_warp:
                cute.arch.cp_async_bulk_wait_group(0, read=read_flag)
            self.reduce_sync_barrier.arrive_and_wait()

        # semaphore release
        # NOTE: barrier.red_release emits a bare red.release.gpu, which orders only
        # the releasing thread's own prior writes -- it is NOT a membar. What makes
        # this release safe is that dQaccum is written by a single elected thread's
        # cpasync_reduce_bulk_add_f32 and the wait_group above runs with
        # read_flag = not deterministic, i.e. read=False, so the reduce has completed
        # before the counter moves. The big-headdim kernel drains dQaccum with
        # per-thread red.global.add instead, so it needs an explicit
        # cute.arch.fence_acq_rel_gpu() before its rendezvous.
        if const_expr(self.deterministic and not delay_semaphore_release):
            if const_expr(self.sdQaccum_stage > 1 and not self.use_2cta_bigd):
                if is_tma_warp:
                    cute.arch.cp_async_bulk_wait_group(0, read=read_flag)
                self.reduce_sync_barrier.arrive_and_wait()
            barrier.arrive_inc(
                mdQ_semaphore_cur[m_block, None].iterator,
                tidx,
                cta_rank_in_cluster,
                1,
            )

        return dQ_consumer_state, dQ_tma_store_producer_state

    @cute.jit
    def store_zero_dK_or_dV(
        self,
        tidx: Int32,
        batch_idx: Int32,
        head_idx: Int32,
        n_block: Int32,
        seqlen_k: Int32,
        mdKV: cute.Tensor,
        K_or_V: cutlass.Constexpr[str],
    ):
        """Zero this key block's dK or dV rows (MHA, 1-CTA), bounded by seqlen_k and head_dim."""
        assert K_or_V in ("K", "V")
        if const_expr(K_or_V == "K"):
            tile_hdim, check_hdim_oob = self.tile_hdim, self.check_hdim_oob
        else:
            tile_hdim, check_hdim_oob = self.tile_hdimv, self.check_hdim_v_oob
        dtype = mdKV.element_type
        num_copy_elems = 128 // dtype.width
        gmem_tiled_copy = copy_utils.tiled_copy_2d(
            dtype,
            tile_hdim // num_copy_elems,
            cute.arch.WARP_SIZE * len(self.compute_warp_ids),
            num_copy_elems,
        )
        gmem_thr_copy = gmem_tiled_copy.get_slice(tidx)
        gdKV = cute.local_tile(
            mdKV[None, None, head_idx, batch_idx], (self.tile_n, tile_hdim), (n_block, 0)
        )
        cdKV = cute.make_identity_tensor((self.tile_n, tile_hdim))
        tdKVgdKV = gmem_thr_copy.partition_D(gdKV)
        tdKVcdKV = gmem_thr_copy.partition_S(cdKV)
        t0dKVcdKV = gmem_tiled_copy.get_slice(0).partition_S(cdKV)
        tdKVpdKV = utils.predicate_k(tdKVcdKV, limit=mdKV.shape[1])
        tdKVrdKV = cute.make_rmem_tensor_like(tdKVgdKV)
        tdKVrdKV.fill(0.0)
        for rest_m in cutlass.range_constexpr(cute.size(tdKVrdKV.shape[1])):
            if t0dKVcdKV[0, rest_m, 0][0] < seqlen_k - n_block * self.tile_n - tdKVcdKV[0][0]:
                cute.copy(
                    gmem_tiled_copy,
                    tdKVrdKV[None, rest_m, None],
                    tdKVgdKV[None, rest_m, None],
                    pred=tdKVpdKV[None, rest_m, None] if const_expr(check_hdim_oob) else None,
                )

    @cute.jit
    def pass_dKV_semaphore(
        self,
        tidx: Int32,
        wg_idx: Int32,
        batch_idx: Int32,
        head_idx: Int32,
        n_block: Int32,
        mdKV_semaphore: cute.Tensor,
    ):
        """Pass this q-head's dKV semaphore turn without accumulating.

        The acquire -> release on one thread keeps earlier q-heads' accumulates ordered first.
        """
        mdKV_semaphore_cur = mdKV_semaphore[
            n_block, None, head_idx // self.qhead_per_kvhead, batch_idx
        ]
        barrier.wait_eq(
            mdKV_semaphore_cur.iterator, tidx, wg_idx, head_idx % self.qhead_per_kvhead
        )
        barrier.arrive_inc(mdKV_semaphore_cur.iterator, tidx, wg_idx, 1)

    @cute.jit
    def epilogue_dKV(
        self,
        tidx: Int32,
        warp_idx: Int32,
        batch_idx: Int32,
        head_idx: Int32,
        n_block: Int32,
        thr_mma_dV: cute.ThrMma,
        thr_mma_dK: cute.ThrMma,
        tdVtdV: cute.Tensor,
        tdKtdK: cute.Tensor,
        mdV: cute.Tensor,
        mdK: cute.Tensor,
        pipeline_dKV: PipelineAsync,
        consumer_state_dKV: cutlass.pipeline.PipelineState,
        softmax_scale: Float32,
    ):
        wg_idx = (
            cute.arch.thread_idx()[0] % (cute.arch.WARP_SIZE * len(self.compute_warp_ids))
        ) // 128
        num_wg = cute.arch.WARP_SIZE * len(self.compute_warp_ids) // 128

        assert self.qhead_per_kvhead == 1, "This epilogue path is only for MHA"
        mdV_cur = mdV[None, None, head_idx, batch_idx]
        mdK_cur = mdK[None, None, head_idx, batch_idx]

        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(16)), Float32
        )

        # dV
        pipeline_dKV.consumer_wait(consumer_state_dKV)

        tiled_tmem_ld_dV = tcgen05.make_tmem_copy(tmem_load_atom, tdVtdV)
        thr_tmem_ld_dV = tiled_tmem_ld_dV.get_slice(tidx)

        tdVtdV_t2r_p = thr_tmem_ld_dV.partition_S(tdVtdV)
        tdVtdV_t2r = self.split_wg(tdVtdV_t2r_p, wg_idx, num_wg)

        cdV = cute.make_identity_tensor((self.mma_tiler_pdo[0], self.mma_tiler_pdo[1]))
        tdVcdV = thr_mma_dV.partition_C(cdV)
        tdVcdV_tensor = cute.make_tensor(tdVcdV.iterator, tdVcdV.layout)

        tdVcdV_t2r_p = thr_tmem_ld_dV.partition_D(tdVcdV_tensor)
        tdVcdV_t2r = self.split_wg(tdVcdV_t2r_p, wg_idx, num_wg)
        tdVrdV_t2r = cute.make_rmem_tensor(tdVcdV_t2r.shape, Float32)

        cute.copy(thr_tmem_ld_dV, tdVtdV_t2r, tdVrdV_t2r)
        cute.arch.fence_view_async_tmem_load()

        universal_copy_bits = 128
        atom_universal_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.dv_dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        tiled_gmem_store_dV = cute.make_tiled_copy(
            atom_universal_copy,
            layout_tv=tiled_tmem_ld_dV.layout_dst_tv_tiled,
            tiler_mn=tiled_tmem_ld_dV.tiler_mn,
        )

        tdVrdV_r2s = cute.make_rmem_tensor(tdVrdV_t2r.shape, self.dv_dtype)
        for i in cutlass.range_constexpr(cute.size(tdVrdV_t2r, mode=[1])):
            dV_vec = tdVrdV_t2r[(None, i, 0, 0)].load()
            tdVrdV_r2s[(None, i, 0, 0)].store(dV_vec.to(self.dv_dtype))

        gdV = cute.local_tile(mdV_cur, (self.mma_tiler_pdo[0], self.tile_hdimv), (None, 0))
        gdV_tile = gdV[None, None, n_block // self.cta_group_size]

        tdVgdV = thr_mma_dV.partition_C(gdV_tile)
        tdVgdV_r2g_p = thr_tmem_ld_dV.partition_D(tdVgdV)
        tdVgdV_r2g = self.split_wg(tdVgdV_r2g_p, wg_idx, num_wg)

        cute.copy(tiled_gmem_store_dV, tdVrdV_r2s, tdVgdV_r2g)

        cute.arch.sync_warp()
        with cute.arch.elect_one():
            pipeline_dKV.consumer_release(consumer_state_dKV)
        consumer_state_dKV.advance()

        # dK
        pipeline_dKV.consumer_wait(consumer_state_dKV)

        tiled_tmem_ld_dK = tcgen05.make_tmem_copy(tmem_load_atom, tdKtdK)
        thr_tmem_ld_dK = tiled_tmem_ld_dK.get_slice(tidx)

        tdKtdK_t2r_p = thr_tmem_ld_dK.partition_S(tdKtdK)
        tdKtdK_t2r = self.split_wg(tdKtdK_t2r_p, wg_idx, num_wg)

        cdK = cute.make_identity_tensor((self.mma_tiler_dsq[0], self.mma_tiler_dsq[1]))
        tdKcdK = thr_mma_dK.partition_C(cdK)
        tdKcdK_tensor = cute.make_tensor(tdKcdK.iterator, tdKcdK.layout)

        tdKcdK_t2r_p = thr_tmem_ld_dK.partition_D(tdKcdK_tensor)
        tdKcdK_t2r = self.split_wg(tdKcdK_t2r_p, wg_idx, num_wg)
        tdKrdK_t2r = cute.make_rmem_tensor(tdKcdK_t2r.shape, Float32)

        cute.copy(tiled_tmem_ld_dK, tdKtdK_t2r, tdKrdK_t2r)
        cute.arch.fence_view_async_tmem_load()

        universal_copy_bits = 128
        atom_universal_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.dk_dtype,
            num_bits_per_copy=universal_copy_bits,
        )

        tiled_gmem_store_dK = cute.make_tiled_copy(
            atom_universal_copy,
            layout_tv=tiled_tmem_ld_dK.layout_dst_tv_tiled,
            tiler_mn=tiled_tmem_ld_dK.tiler_mn,
        )

        tdKrdK_r2s = cute.make_rmem_tensor(tdKrdK_t2r.shape, self.dk_dtype)

        for i in cutlass.range_constexpr(cute.size(tdKrdK_t2r, mode=[1])):
            dK_vec = tdKrdK_t2r[(None, i, 0, 0)].load() * softmax_scale
            tdKrdK_r2s[(None, i, 0, 0)].store(dK_vec.to(self.dk_dtype))

        gdK = cute.local_tile(mdK_cur, (self.mma_tiler_dsq[0], self.tile_hdim), (None, 0))
        gdK_tile = gdK[None, None, n_block // self.cta_group_size]

        tdKgdK = thr_mma_dK.partition_C(gdK_tile)
        tdKgdK_r2g_p = thr_tmem_ld_dK.partition_D(tdKgdK)
        tdKgdK_r2g = self.split_wg(tdKgdK_r2g_p, wg_idx, num_wg)

        cute.copy(tiled_gmem_store_dK, tdKrdK_r2s, tdKgdK_r2g)

        cute.arch.sync_warp()
        with cute.arch.elect_one():
            pipeline_dKV.consumer_release(consumer_state_dKV)
        consumer_state_dKV.advance()
        return consumer_state_dKV

    @cute.jit
    def epilogue_dK_or_dV_tma(
        self,
        tidx: Int32,
        batch_idx: Int32,
        head_idx: Int32,
        n_block: Int32,
        thr_mma: cute.ThrMma,
        tdKVtdKV: cute.Tensor,
        mdKV: cute.Tensor,
        sdKV: cute.Tensor,
        tma_atom_dKV: cute.CopyAtom,
        thr_copy_r2s_dKV: cute.TiledCopy,
        pipeline_dKV: PipelineAsync,
        consumer_state_dKV: cutlass.pipeline.PipelineState,
        scale: Optional[Float32],
        barrier_id: Int32,
        mdKV_semaphore: Optional[cute.Tensor],
        K_or_V: cutlass.Constexpr[str],
    ) -> cutlass.pipeline.PipelineState:
        assert K_or_V in ("K", "V")
        if const_expr(K_or_V == "K"):
            tile_hdim = self.tile_hdim
        else:
            tile_hdim = self.tile_hdimv
        dtype = self.dk_dtype if const_expr(K_or_V == "K") else self.dv_dtype
        epi_tile = self.sdK_epi_tile if const_expr(K_or_V == "K") else self.sdV_epi_tile
        flat_epi_tile = (
            self.sdK_flat_epi_tile if const_expr(K_or_V == "K") else self.sdV_flat_epi_tile
        )
        num_compute_threads = cute.arch.WARP_SIZE * len(self.compute_warp_ids)
        wg_idx = (cute.arch.thread_idx()[0] % num_compute_threads) // 128
        num_wg = num_compute_threads // 128
        leader_warp = (cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4) == 0

        cta_group_tile_n = const_expr(self.tile_n * self.cta_group_size)

        if const_expr(not self.dKV_postprocess):
            num_strips = self.epi_smem_strips
            # Buffer index = wg * num_strips + strip. With a folded accumulator
            # threads 0..63 hold hdim strip 0 and threads 64..127 strip 1
            # (delta = tile_hdim / 2, i.e. the TMEM lane halves).
            strip_idx = tidx // self.epi_threads_r2s
            sdKV_r2s = sdKV[None, None, wg_idx * num_strips + strip_idx]
            sdKV_strips = [
                sdKV[None, None, wg_idx * num_strips + s]
                # plain range: the AST preprocessor does not rewrite comprehensions,
                # so range_constexpr would reach its runtime stub and raise
                for s in range(num_strips)
            ]
        else:
            num_strips = 1
            sdKV_r2s = sdKV[None, wg_idx]  # (tile_n * 32) for fp32
            sdKV_strips = [sdKV_r2s]

        # (8, tile_n / 128, 64 / 8) = (8, 1, 8) or (4, tile_n * 32 / (128 * 4)) = (4, 8)
        tdKVsdKV_r2s = thr_copy_r2s_dKV.partition_D(sdKV_r2s)

        head_idx_kv = head_idx // self.qhead_per_kvhead
        if const_expr(not self.dKV_postprocess):
            mdKV_cur = mdKV[None, None, head_idx_kv, batch_idx]  # (seqlen, hdim)
            gdKV_p = cute.local_tile(
                mdKV_cur, (self.tile_n, tile_hdim), (n_block, 0)
            )  # (tile_n, hdim) - per CTA
            if const_expr(num_strips == 1):
                gdKV = self.split_wg(gdKV_p, wg_idx, num_wg)  # (tile_n, hdim / 2)
            else:
                # A folded warpgroup's two strips are hdim/2 apart, so the wg does
                # NOT own a contiguous hdim half: tile the whole hdim and index
                # the (wg, epi_stage, strip) triple explicitly below.
                gdKV = gdKV_p
            gdKV_epi = cute.local_tile(
                gdKV, epi_tile, (0, None)
            )  # (tile_n, 64, epi_stage = (hdim / 2) / 64)

        else:
            mdKV_cur = mdKV[None, head_idx_kv, batch_idx]  # (seqlen * hdim)

            gdKV_p = cute.local_tile(
                mdKV_cur, (self.tile_n * tile_hdim,), (n_block,)
            )  # (tile_n * hdim)
            # Panel = 128 threads x dK_reduce_ncol values. With a folded
            # accumulator that is (2 * tile_n) rows worth of lanes covering two
            # hdim strips, so the panel is twice as long and there are half as
            # many; panel index stays "wg-major, then physical column chunk",
            # which is the contiguous hdim order the postprocess decodes.
            gdKV_epi = cute.flat_divide(
                gdKV_p, (self.dKV_reduce_panel,)
            )  # (panel, num_panels)

        deterministic_KV = self.deterministic and self.dKV_postprocess
        if const_expr(deterministic_KV):
            mdKV_semaphore_cur = mdKV_semaphore[n_block, None, head_idx_kv, batch_idx]

        if const_expr(not self.dKV_postprocess):
            tdKVsdKV_list = []
            for s in cutlass.range_constexpr(num_strips):
                tdKVsdKV_s, tdKVgdKV = cpasync.tma_partition(
                    tma_atom_dKV,
                    0,  # no multicast
                    cute.make_layout(1),
                    cute.group_modes(sdKV_strips[s], 0, 2),
                    cute.group_modes(gdKV_epi, 0, 2),
                )  # (TMA) and (TMA, EPI_STAGE)
                assert len(tdKVsdKV_s.shape) == 1, "Wrong rank for SMEM fragment tdKVsdKV"
                assert len(tdKVgdKV.shape) == 2, "Wrong rank for GMEM fragment tdKVgdKV"
                tdKVsdKV_list.append(tdKVsdKV_s)
            num_gmem_tiles = cute.size(tdKVgdKV.shape[1])
            num_epi_stages = num_gmem_tiles // (num_strips * (num_wg if num_strips > 1 else 1))
            if const_expr(K_or_V == "K"):
                assert num_epi_stages == self.num_epi_stages, f"Epi stage calculation is wrong (K). num_epi_stages:{num_epi_stages} != self.num_epi_stages: {self.num_epi_stages}"
            else:
                assert num_epi_stages == self.num_epi_stages_v, f"Epi stage calculation is wrong (V). num_epi_stages:{num_epi_stages} != self.num_epi_stages_v: {self.num_epi_stages_v}"
        else:
            num_epi_stages = (
                tile_hdim // num_wg // self.epi_num_strips // self.dK_reduce_ncol
            )

        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(self.dK_reduce_ncol)), Float32
        )


        read_flag = const_expr(not deterministic_KV)

        pipeline_dKV.consumer_wait(consumer_state_dKV)

        # semaphore acquire (one per Q-head ordering)
        if const_expr(deterministic_KV):
            barrier.wait_eq(
                mdKV_semaphore_cur.iterator, tidx, wg_idx, head_idx % self.qhead_per_kvhead
            )
            cute.arch.barrier(barrier_id=barrier_id + wg_idx, number_of_threads=128)

        for epi_stage in cutlass.range_constexpr(num_epi_stages):
            # TMEM -> RMEM -- setup
            thr_copy_t2r = tcgen05.make_tmem_copy(tmem_load_atom, tdKVtdKV).get_slice(tidx)
            tdKVtdKV_t2r_p = thr_copy_t2r.partition_S(tdKVtdKV)
            tdKVtdKV_t2r = self.split_wg(tdKVtdKV_t2r_p, wg_idx, num_wg)[None, None, 0, 0]
            if const_expr(num_epi_stages > 1):
                tdKVtdKV_t2r = tdKVtdKV_t2r[None, epi_stage]

            cdKV = cute.make_identity_tensor((cta_group_tile_n, tile_hdim))
            tdKVcdKV = thr_mma.partition_C(cdKV)
            tdKVcdKV_t2r_p = thr_copy_t2r.partition_D(tdKVcdKV)
            tdKVcdKV_t2r = self.split_wg(tdKVcdKV_t2r_p, wg_idx, num_wg)[None, None, 0, 0]
            if const_expr(num_epi_stages > 1):
                tdKVcdKV_t2r = tdKVcdKV_t2r[None, epi_stage]

            tdKVrdKV_t2r = cute.make_rmem_tensor(tdKVcdKV_t2r.shape, Float32)

            assert cute.size(tdKVrdKV_t2r) == cute.size(tdKVtdKV_t2r) // cute.arch.WARP_SIZE, (
                "RMEM<->TMEM fragment size mismatch"
            )
            assert cute.size(tdKVrdKV_t2r) == cute.size(tdKVsdKV_r2s), (
                f"RMEM<->SMEM fragment size mismatch: {cute.size(tdKVrdKV_t2r)} != "
                f"{cute.size(tdKVsdKV_r2s)} (epi_num_strips={self.epi_num_strips}, "
                f"epi_tile={epi_tile})"
            )

            # TMEM -> RMEM -- copy and fence
            cute.copy(thr_copy_t2r, tdKVtdKV_t2r, tdKVrdKV_t2r)
            cute.arch.fence_view_async_tmem_load()

            # RMEM -- scale and convert
            if const_expr(scale is not None):
                for i in cutlass.range(cute.size(tdKVrdKV_t2r.shape) // 2, unroll_full=True):
                    tdKVrdKV_t2r[2 * i], tdKVrdKV_t2r[2 * i + 1] = utils.mul_packed_f32x2(
                        (tdKVrdKV_t2r[2 * i], tdKVrdKV_t2r[2 * i + 1]), (scale, scale)
                    )
            tdKVrdKV = cute.make_rmem_tensor(tdKVrdKV_t2r.shape, dtype)  # (32 columns)
            tdKVrdKV.store(tdKVrdKV_t2r.load().to(dtype))

            # RMEM -> SMEM -- copy, fence and barrier
            tdKVrdKV_r2s = cute.make_tensor(tdKVrdKV.iterator, tdKVsdKV_r2s.shape)
            cute.copy(thr_copy_r2s_dKV, tdKVrdKV_r2s, tdKVsdKV_r2s)
            cute.arch.fence_view_async_shared()
            cute.arch.barrier(barrier_id=barrier_id + wg_idx, number_of_threads=128)

            # SMEM -> GMEM
            if leader_warp:
                if const_expr(not self.dKV_postprocess):
                    for s in cutlass.range_constexpr(num_strips):
                        if const_expr(num_strips == 1):
                            gmem_tile_idx = epi_stage
                        else:
                            # d offset = wg * (stages * ncol) + epi_stage * ncol
                            #            + strip * (tile_hdim / 2)
                            gmem_tile_idx = (
                                wg_idx * num_epi_stages
                                + epi_stage
                                + s * num_wg * num_epi_stages
                            )
                        cute.copy(
                            tma_atom_dKV, tdKVsdKV_list[s], tdKVgdKV[None, gmem_tile_idx]
                        )
                else:
                    with cute.arch.elect_one():
                        copy_utils.cpasync_reduce_bulk_add_f32(
                            sdKV_r2s.iterator,
                            gdKV_epi[None, wg_idx * num_epi_stages + epi_stage].iterator,
                            self.dKV_reduce_bytes,
                        )
                # The last epi_stage normally skips the drain because the CTA is about to
                # exit and nothing reuses the SMEM. Persistent has to drain: sdK / sdV
                # alias sK / sV, which the next work tile's load warp overwrites.
                if const_expr(
                    epi_stage < num_epi_stages - 1
                    or self.is_persistent
                ):
                    cute.arch.cp_async_bulk_commit_group()
                    cute.arch.cp_async_bulk_wait_group(0, read=read_flag)
                cute.arch.barrier_arrive(
                    barrier_id=barrier_id + wg_idx, number_of_threads=128 + cute.arch.WARP_SIZE
                )

            # Barrier since all warps need to wait for SMEM to be freed
            cute.arch.fence_view_async_shared()
            cute.arch.barrier(
                barrier_id=barrier_id + wg_idx, number_of_threads=128 + cute.arch.WARP_SIZE
            )

        # semaphore release: a single dK/dV store increments the semaphore exactly
        # once per Q-head.
        if const_expr(deterministic_KV):
            if leader_warp:
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0, read=read_flag)
            cute.arch.barrier(barrier_id=barrier_id + wg_idx, number_of_threads=128)
            barrier.arrive_inc(mdKV_semaphore_cur.iterator, tidx, wg_idx, 1)

        cute.arch.sync_warp()
        with cute.arch.elect_one():
            pipeline_dKV.consumer_release(consumer_state_dKV)
        consumer_state_dKV.advance()
        return consumer_state_dKV
