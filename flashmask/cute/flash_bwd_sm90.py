# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# Copyright (c) 2026 The FlashMask Authors. Modifications for FlashMask.

import math
from typing import Callable, Optional, Type
from functools import partial

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.utils.hopper_helpers as sm90_utils_basic
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass.cute import FastDivmodDivisor
from cutlass import Float32, Int32, Boolean, const_expr
from cutlass.utils import LayoutEnum

from . import copy_utils
from . import layout_utils
from . import hopper_helpers as sm90_utils
from .hopper_helpers import gemm_zero_init, gemm_w_idx

from .cute_dsl_utils import assume_tensor_aligned
from . import utils
from .mask import AttentionMask
from .seqlen_info import SeqlenInfoQK
from .block_info import BlockInfo
from . import pipeline
from .tile_scheduler import (
    ParamsBase,
    TileSchedulerArguments,
    SingleTileScheduler,
    SingleTileLPTBwdScheduler,
)
from . import barrier
from .named_barrier import NamedBarrierBwd
from .block_sparsity import BlockSparseTensors
from .flashmask_utils import FlashMaskInfo
from .block_sparse_utils import (
    get_total_q_block_count_bwd,
    produce_block_sparse_q_loads_bwd_sm90,
    consume_block_sparse_mma_bwd_sm90,
    dQaccum_store_block_sparse_bwd_sm90,
    _load_q_do_block_sm90,
)


class FlashAttentionBackwardSm90:
    arch = 90

    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        head_dim: int,
        head_dim_v: Optional[int] = None,
        qhead_per_kvhead: int = 1,
        is_causal: bool = False,
        is_local: bool = False,
        deterministic: bool = False,
        tile_m: int = 64,
        tile_n: int = 128,
        Q_stage: int = 2,
        dO_stage: int = 2,
        PdS_stage: int = 2,
        SdP_swapAB: bool = False,
        dKV_swapAB: bool = False,
        dQ_swapAB: bool = False,
        AtomLayoutMSdP: int = 1,
        AtomLayoutNdKV: int = 2,
        AtomLayoutMdQ: int = 1,
        num_threads: int = 384,
        V_in_regs: bool = False,
        subtile_factor: cutlass.Constexpr[int] = 1,
        dQ_single_wg: bool = False,
    ):
        self.dtype = dtype
        # Pad head_dim to a multiple of 32 (matching the SM100 bwd, the fp32
        # dQaccum allocation in interface.py, the preprocess zero-fill, and the
        # postprocess read). Using 16 here made the dQaccum per-m_block stride
        # (tile_m * tile_hdim) disagree with those (which round to 32), so for
        # head_dim that is a multiple of 16 but not 32 (e.g. 80) m_block>=1 landed
        # at the wrong dQaccum offset -> wrong dQ. 32 keeps every stage consistent;
        # the extra padding columns are zero-filled by TMA on the K/Q/dO/V loads.
        hdim_multiple_of = 32
        self.tile_hdim = int(math.ceil(head_dim / hdim_multiple_of) * hdim_multiple_of)
        head_dim_v = head_dim_v if head_dim_v is not None else head_dim
        self.same_hdim_kv = head_dim == head_dim_v
        self.tile_hdimv = int(math.ceil(head_dim_v / hdim_multiple_of) * hdim_multiple_of)
        # Can save registers (and hence be faster) if we don't have to check hdim predication
        self.check_hdim_oob = head_dim != self.tile_hdim
        self.check_hdim_v_oob = head_dim_v != self.tile_hdimv
        self.qhead_per_kvhead = qhead_per_kvhead
        self.is_causal = is_causal
        self.is_local = is_local
        self.deterministic = deterministic
        self.tile_m = tile_m
        self.tile_n = tile_n
        self.num_threads = num_threads
        self.Q_stage = Q_stage
        self.dO_stage = dO_stage
        self.PdS_stage = PdS_stage
        assert self.dO_stage in [1, self.Q_stage]
        assert self.PdS_stage in [1, self.Q_stage]
        self.SdP_swapAB = SdP_swapAB
        self.dKV_swapAB = dKV_swapAB
        self.dQ_swapAB = dQ_swapAB
        self.AtomLayoutMSdP = AtomLayoutMSdP
        self.AtomLayoutNdKV = AtomLayoutNdKV
        self.AtomLayoutMdQ = AtomLayoutMdQ
        self.num_wg_mma = (self.num_threads // 128) - 1
        self.mma_dkv_is_rs = (
            AtomLayoutMSdP == 1
            and AtomLayoutNdKV == self.num_wg_mma
            and SdP_swapAB
            and not dKV_swapAB
        )
        self.V_in_regs = V_in_regs
        if qhead_per_kvhead > 1:
            assert self.num_wg_mma == 2, "GQA backward assumes 2 warp groups"
        # These are tuned for speed
        # Do we keep the LSE and dPsum in each thread, or split them across 8 threads that share
        # them and then shuffle to get the value whenever we need? This can reduce register
        # pressure when SdP_swapAB, where each thread needs to keep statistics for (kBlockM / 4)
        # rows. If !SdP_swapAB, each thread only needs to keep statistics for 2 rows.
        self.shuffle_LSE = self.SdP_swapAB and self.tile_hdim <= 64
        self.shuffle_dPsum = self.SdP_swapAB and self.tile_hdim <= 64

        self.buffer_align_bytes = 1024

        self.subtile_factor = subtile_factor
        self.vec_size: cutlass.Constexpr = 4
        self.qk_acc_dtype = Float32
        # dQ_single_wg: WG0 computes the full dQ GEMM, WG1 skips it.
        # Only valid for 2 MMA warp groups.
        # Credit: Ben Spector
        if dQ_single_wg:
            assert self.num_wg_mma == 2, "dQ_single_wg only supports 2 warp groups"
        self.num_wg_dQ = 1 if dQ_single_wg else self.num_wg_mma

        # dQaccum accumulation strategy (mirrors C++ FA3 hopper bwd):
        #   - head_dim < 256: stage dQ in a smem buffer, then TMA bulk-reduce-add
        #     it to gmem from the dedicated store warp (dQacc_use_TMA=True).
        #   - head_dim == 256: the 64KB smem staging buffer would force
        #     num_stages_Q=1 (no Q/dO prefetch). Instead atomicAdd the dQ
        #     accumulator registers straight to gmem, freeing the smem for a
        #     2-stage Q pipeline.
        # Deterministic mode requires an ordered accumulation (semaphore-gated
        # smem+TMA path); atomicAdd order is non-deterministic, so force TMA.
        self.dQacc_use_TMA = const_expr(self.tile_hdim < 256 or deterministic)

        # Slice the dQ/dK/dV MMAs into two M-halves and interleave them so the
        # dQ atomicAdd of the first half overlaps the second half's MMA (mirrors
        # C++ FA3 Slice_dQKV_Mma for head_dim=256). Only valid on the atomicAdd dQ
        # path with dQ_swapAB, AtomLayoutMdQ==1 and 2 MMA warp groups (same
        # predicate as cpp). mma_dkv_is_rs must be False (guaranteed here since
        # SdP_swapAB is False for the d256 config), matching cpp's static_assert.
        self.slice_dQKV_mma = const_expr(
            self.tile_hdim == 256
            and not self.dQacc_use_TMA
            and self.dQ_swapAB
            and self.AtomLayoutMdQ == 1
            and self.num_wg_mma == 2
            and not self.mma_dkv_is_rs
        )

    def _check_type(
        self,
        mQ_type: Type[cutlass.Numeric],
        mK_type: Type[cutlass.Numeric],
        mV_type: Type[cutlass.Numeric],
        mdO_type: Type[cutlass.Numeric],
        mLSE_type: Type[cutlass.Numeric],
        mdPsum_type: Type[cutlass.Numeric],
        mdQaccum_type: Type[cutlass.Numeric],
        mdK_type: Type[cutlass.Numeric],
        mdV_type: Type[cutlass.Numeric],
    ):
        # Get the data type and check if it is fp16 or bf16
        if const_expr(not (mQ_type == mK_type == mV_type == mdO_type)):
            raise TypeError("All tensors must have the same data type")
        if const_expr(mQ_type not in [cutlass.Float16, cutlass.BFloat16]):
            raise TypeError("Only Float16 or BFloat16 is supported")
        if const_expr(mLSE_type not in [Float32]):
            raise TypeError("LSE tensor must be Float32")
        if const_expr(mdPsum_type not in [Float32]):
            raise TypeError("dPsum tensor must be Float32")
        if const_expr(mdQaccum_type not in [Float32]):
            raise TypeError("dQaccum tensor must be Float32")
        if const_expr(self.qhead_per_kvhead == 1):
            if const_expr(not (mdK_type == mdV_type == mQ_type)):
                raise TypeError("mdK and mdV tensors must have the same data type as mQ")
        else:
            if const_expr(not (mdK_type == mdV_type == Float32)):
                raise TypeError("mdKaccum and mdVaccum tensors must have the data type Float32")
        assert mQ_type == self.dtype

    def _setup_attributes(self):
        # We need to accommodate both Q and Q^T (and dO and dO^T) in shared memory.
        # Q & dO are used in the SdP Mma and Q^T and dO^T are used in the dKV Mma.
        # The M dimension (tile_m) doesn't matter for the layout, only the K dimension
        wg_d_dKV = self.num_wg_mma // self.AtomLayoutNdKV
        self.sQ_layout, self.sdO_layout = [
            # Need to set major_mode_size (mms) to accommodate Q and Q.T
            sm90_utils.make_smem_layout(self.dtype, LayoutEnum.ROW_MAJOR, shape, stage, mms)
            for shape, stage, mms in [
                ((self.tile_m, self.tile_hdim), self.Q_stage, self.tile_hdim // wg_d_dKV),
                ((self.tile_m, self.tile_hdimv), self.dO_stage, self.tile_hdim // wg_d_dKV),
            ]
        ]
        wg_d_dQ = self.num_wg_dQ // self.AtomLayoutMdQ
        # Accomodate both K and K.T
        self.sK_layout = sm90_utils.make_smem_layout(
            self.dtype,
            LayoutEnum.ROW_MAJOR,
            (self.tile_n, self.tile_hdim),
            stage=None,
            major_mode_size=self.tile_hdim // wg_d_dQ,
        )
        # There's only V, no V.T, so layout is normal
        self.sV_layout = sm90_utils.make_smem_layout(
            self.dtype, LayoutEnum.ROW_MAJOR, (self.tile_n, self.tile_hdimv), None
        )
        # Accomodate both S and S.T
        wg_n_SdP = self.num_wg_mma // self.AtomLayoutMSdP
        wg_n_dKV = self.AtomLayoutNdKV
        self.sPdS_layout = sm90_utils.make_smem_layout(
            self.dtype,
            LayoutEnum.ROW_MAJOR,
            (self.tile_m, self.tile_n),
            stage=self.PdS_stage,
            major_mode_size=math.gcd(self.tile_n // wg_n_SdP, self.tile_n // wg_n_dKV),
        )
        self.sdQaccum_layout = cute.make_layout(
            (self.tile_m * self.tile_hdim // self.num_wg_dQ, self.num_wg_dQ)
        )
        # dQaccum R->S
        self.r2s_tiled_copy_dQaccum = cute.make_tiled_copy_tv(
            cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), Float32, num_bits_per_copy=128),
            # thr_layout
            cute.make_layout((self.num_threads_per_warp_group, self.num_wg_dQ)),
            cute.make_layout(128 // Float32.width),  # val_layout
        )
        # dKVaccum for GQA epilogue - reuses sV+sK memory recast as f32
        # TODO: assert that sVaccum and sKaccum don't overflow smem

    def _get_tiled_mma(self):
        maybe_swap_mn = lambda shape, swap: (shape[1], shape[0], *shape[2:]) if swap else shape
        # S = Q @ K.T, dP = dO @ V.T
        atom_layout_SdP = (self.AtomLayoutMSdP, self.num_wg_mma // self.AtomLayoutMSdP, 1)
        tiler_mn_SdP = (self.tile_m // atom_layout_SdP[0], self.tile_n // atom_layout_SdP[1])
        tiled_mma_SdP = sm90_utils_basic.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.K,
            Float32,
            atom_layout_mnk=maybe_swap_mn(atom_layout_SdP, self.SdP_swapAB),
            tiler_mn=(64, tiler_mn_SdP[1] if not self.SdP_swapAB else tiler_mn_SdP[0]),
        )
        # dV = P.T @ dO, dK = dS.T @ Q
        atom_layout_dKV = (self.AtomLayoutNdKV, self.num_wg_mma // self.AtomLayoutNdKV, 1)
        tiler_mn_dK = (self.tile_n // atom_layout_dKV[0], self.tile_hdim // atom_layout_dKV[1])
        tiler_mn_dV = (self.tile_n // atom_layout_dKV[0], self.tile_hdimv // atom_layout_dKV[1])
        tiled_mma_dK, tiled_mma_dV = [
            sm90_utils_basic.make_trivial_tiled_mma(
                self.dtype,
                self.dtype,
                warpgroup.OperandMajorMode.MN
                if not self.mma_dkv_is_rs
                else warpgroup.OperandMajorMode.K,
                warpgroup.OperandMajorMode.MN,
                Float32,
                atom_layout_mnk=maybe_swap_mn(atom_layout_dKV, self.dKV_swapAB),
                tiler_mn=(64, tiler_mn_d[1] if not self.dKV_swapAB else tiler_mn_d[0]),
                a_source=warpgroup.OperandSource.RMEM
                if self.mma_dkv_is_rs
                else warpgroup.OperandSource.SMEM,
            )
            for tiler_mn_d in (tiler_mn_dK, tiler_mn_dV)
        ]
        # dQ = dS @ K
        assert self.num_wg_dQ % self.AtomLayoutMdQ == 0
        atom_layout_dQ = (self.AtomLayoutMdQ, self.num_wg_dQ // self.AtomLayoutMdQ, 1)
        tiler_mn_dQ = (self.tile_m // atom_layout_dQ[0], self.tile_hdim // atom_layout_dQ[1])
        tiled_mma_dQ = sm90_utils_basic.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            warpgroup.OperandMajorMode.K if not self.dQ_swapAB else warpgroup.OperandMajorMode.MN,
            warpgroup.OperandMajorMode.MN if not self.dQ_swapAB else warpgroup.OperandMajorMode.K,
            Float32,
            atom_layout_mnk=maybe_swap_mn(atom_layout_dQ, self.dQ_swapAB),
            tiler_mn=(64, tiler_mn_dQ[1] if not self.dQ_swapAB else tiler_mn_dQ[0]),
        )
        return tiled_mma_SdP, tiled_mma_dK, tiled_mma_dV, tiled_mma_dQ

    def _get_shared_storage_cls(self):
        sQ_struct, sK_struct, sV_struct, sdO_struct = [
            cute.struct.Align[cute.struct.MemRange[t, cute.cosize(layout)], self.buffer_align_bytes]
            for (layout, t) in [
                (self.sQ_layout, self.dtype),
                (self.sK_layout, self.dtype),
                (self.sV_layout, self.dtype),
                (self.sdO_layout, self.dtype),
            ]
        ]
        # dQaccum smem staging is only used by the TMA bulk-reduce path. In the
        # atomicAdd path (head_dim=256) it's dropped entirely to free the ~64KB
        # for the 2-stage Q pipeline.
        sdqaccum_cosize = cute.cosize(self.sdQaccum_layout) if const_expr(self.dQacc_use_TMA) else 0
        sdQaccum_struct = cute.struct.Align[
            cute.struct.MemRange[Float32, sdqaccum_cosize], self.buffer_align_bytes
        ]
        # For GQA, the epilogue recasts sV's bytes (extending into the
        # immediately-following sK field - see SharedStorageQKV field order
        # below: sV comes right before sK, both 1024B-aligned) to fp32, to
        # stage dK-accum/dV-accum (see epilogue_dKV). K/V bf16 data and the
        # accum staging are never live at the same time, so they can share
        # bytes. When tile_hdim == tile_hdimv (e.g. the hdim=256 config),
        # cosize(sV_layout) + cosize(sK_layout) (in dtype elements) already
        # happens to equal the fp32 accum element count on the nose. When
        # tile_hdim != tile_hdimv (e.g. 192/128), that equality breaks
        # (dK-accum needs more bytes than sV+sK combined provide today), so
        # we pad sK here to make up the difference - this only grows the
        # allocation by the (small) shortfall, not by a whole new buffer.
        if const_expr(self.qhead_per_kvhead > 1):
            dK_accum_elems = self.tile_n * self.tile_hdim * 2  # fp32 = 2x dtype-sized elements
            dV_accum_elems = self.tile_n * self.tile_hdimv * 2
            accum_elems_needed = max(dK_accum_elems, dV_accum_elems)
            extra_elems_needed = accum_elems_needed - cute.cosize(self.sV_layout)
            cosize_sK = max(cute.cosize(self.sK_layout), extra_elems_needed)
            sK_struct = cute.struct.Align[
                cute.struct.MemRange[self.dtype, cosize_sK], self.buffer_align_bytes
            ]

        cosize_sdS = cute.cosize(self.sPdS_layout)
        cosize_sP = cute.cosize(self.sPdS_layout) if const_expr(not self.mma_dkv_is_rs) else 0
        sLSE_struct = cute.struct.Align[
            cute.struct.MemRange[Float32, cute.round_up(self.tile_m, 64) * self.Q_stage], 128
        ]
        sdPsum_struct = cute.struct.Align[
            cute.struct.MemRange[Float32, cute.round_up(self.tile_m, 64) * self.dO_stage], 128
        ]
        # flashmask: single staging buffer for the 4 startend_row_indices vectors
        # (LTS/LTE/UTS/UTE) of the current n_block; layout [vec*tile_n + col].
        rowidx_smem_count = (4 * self.tile_n) if const_expr(self.enable_flashmask) else 0
        sRowidx_struct = cute.struct.Align[
            cute.struct.MemRange[Int32, rowidx_smem_count], 128
        ]

        @cute.struct
        class SharedStorageQKV:
            mbar_ptr_Q: cute.struct.MemRange[cutlass.Int64, self.Q_stage * 2]
            mbar_ptr_dO: cute.struct.MemRange[cutlass.Int64, self.dO_stage * 2]
            sLSE: sLSE_struct
            sdPsum: sdPsum_struct
            sRowidx: sRowidx_struct
            sQ: sQ_struct
            sV: sV_struct
            sK: sK_struct
            sdO: sdO_struct
            sP: cute.struct.Align[cute.struct.MemRange[self.dtype, cosize_sP], 1024]
            sdS: cute.struct.Align[cute.struct.MemRange[self.dtype, cosize_sdS], 1024]
            sdQaccum: sdQaccum_struct

        return SharedStorageQKV

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
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        flashmask_info: Optional[FlashMaskInfo] = None,
        # Always keep stream as the last parameter (EnvStream: obtained implicitly via TVM FFI).
        stream: cuda.CUstream = None,
    ):
        # For GQA (qhead_per_kvhead > 1), multiple Q heads accumulate into the same dK/dV,
        # so we need the float32 accum path + postprocess.
        self._check_type(
            *(
                t.element_type if t is not None else None
                for t in (mQ, mK, mV, mdO, mLSE, mdPsum, mdQaccum, mdK, mdV)
            )
        )

        mQ, mK, mV, mdO, mLSE, mdPsum, mdQaccum, mdK, mdV = [
            assume_tensor_aligned(t) for t in (mQ, mK, mV, mdO, mLSE, mdPsum, mdQaccum, mdK, mdV)
        ]

        # Inputs are (b, s, n, h). We convert to a seqlen-major view with head-dim second.
        def _qkv_transpose(t):
            return layout_utils.select(t, [1, 3, 2, 0] if cute.rank(t.shape) == 4 else [0, 2, 1])

        mQ, mK, mV, mdO = [_qkv_transpose(t) for t in (mQ, mK, mV, mdO)]
        if const_expr(self.qhead_per_kvhead == 1):
            mdK, mdV = [_qkv_transpose(t) for t in (mdK, mdV)]
        else:
            # Accum tensors are (b, n, s*h).
            accum_transpose = [2, 1, 0] if cute.rank(mdK.shape) == 3 else [1, 0]
            mdK, mdV = [layout_utils.select(t, accum_transpose) for t in (mdK, mdV)]
        # Stats are (b, n, s).
        LSE_dPsum_dQaccum_transpose = [2, 1, 0] if cute.rank(mLSE.shape) == 3 else [1, 0]
        mLSE, mdPsum, mdQaccum = [
            layout_utils.select(t, LSE_dPsum_dQaccum_transpose) for t in (mLSE, mdPsum, mdQaccum)
        ]

        tiled_mma_SdP, tiled_mma_dK, tiled_mma_dV, tiled_mma_dQ = self._get_tiled_mma()
        # (batch, num_head, num_m_blocks, cluster_size) -> (num_m_blocks, cluster_size, num_head, batch)
        if const_expr(self.deterministic):
            assert mdQ_semaphore is not None
            mdQ_semaphore = layout_utils.select(mdQ_semaphore, mode=[2, 3, 1, 0])
        if const_expr(self.deterministic and self.qhead_per_kvhead > 1):
            assert mdK_semaphore is not None
            assert mdV_semaphore is not None
            mdK_semaphore, mdV_semaphore = [
                layout_utils.select(t, mode=[2, 3, 1, 0]) for t in (mdK_semaphore, mdV_semaphore)
            ]
        else:
            mdK_semaphore = None
            mdV_semaphore = None

        self.num_mma_threads = tiled_mma_SdP.size
        assert self.num_mma_threads + 128 == self.num_threads

        self.num_threads_per_warp_group = 128
        self.num_producer_threads = 32

        REG_LIMIT = 504 if self.num_wg_mma == 2 else 512
        if const_expr(self.num_wg_mma == 2):
            if const_expr(self.num_wg_dQ == 1):
                self.num_mma_regs_wg0 = 256
                self.num_mma_regs_wg1 = 224
            else:
                self.num_mma_regs_wg0 = 240
                self.num_mma_regs_wg1 = 240
            self.num_mma_regs = self.num_mma_regs_wg0  # for backward compat
            self.num_producer_regs = 24
            assert (
                self.num_mma_regs_wg0 + self.num_mma_regs_wg1 + self.num_producer_regs <= REG_LIMIT
            )
        else:  # 3 warp groups
            self.num_mma_regs_wg0 = 160
            self.num_mma_regs_wg1 = 160
            self.num_mma_regs = 160
            self.num_producer_regs = 32
            assert self.num_mma_regs_wg0 * self.num_wg_mma + self.num_producer_regs <= REG_LIMIT

        # flashmask (startend_row_indices) support. For bwd the per-KV-column
        # indices are fixed for a whole n_block, so they are staged into smem once
        # per n_block and reused across the inner m_block loop.
        self.enable_flashmask = const_expr(flashmask_info is not None)
        self.has_lt_end = const_expr(
            flashmask_info is not None and flashmask_info.LTE_nblock_max is not None
        )
        self.has_ut_start = const_expr(
            flashmask_info is not None and flashmask_info.UTS_nblock_max is not None
        )
        self.has_ut_end = const_expr(
            flashmask_info is not None and flashmask_info.UTE_nblock_max is not None
        )

        self._setup_attributes()
        SharedStorage = self._get_shared_storage_cls()

        self.tma_copy_bytes = {
            name: cute.size_in_bytes(mX.element_type, cute.select(layout, mode=[0, 1]))
            for name, mX, layout in [
                ("Q", mQ, self.sQ_layout),
                ("K", mK, self.sK_layout),
                ("V", mV, self.sV_layout),
                ("dO", mdO, self.sdO_layout),
            ]
        }
        self.tma_copy_bytes["LSE"] = self.tile_m * Float32.width // 8
        self.tma_copy_bytes["dPsum"] = self.tile_m * Float32.width // 8
        self.tma_copy_bytes["dQ"] = (
            self.tile_m * self.tile_hdim * Float32.width // 8 // self.num_wg_dQ
        )
        self.tma_copy_bytes["dKacc"] = self.tile_n * self.tile_hdim * Float32.width // 8
        self.tma_copy_bytes["dVacc"] = self.tile_n * self.tile_hdimv * Float32.width // 8

        tma_atom_Q, tma_tensor_Q = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mQ,
            cute.select(self.sQ_layout, mode=[0, 1]),
            (self.tile_m, self.tile_hdim),
        )
        tma_atom_K, tma_tensor_K = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mK,
            cute.select(self.sK_layout, mode=[0, 1]),
            (self.tile_n, self.tile_hdim),
        )
        tma_atom_V, tma_tensor_V = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mV,
            cute.select(self.sV_layout, mode=[0, 1]),
            (self.tile_n, self.tile_hdimv),
        )
        tma_atom_dO, tma_tensor_dO = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mdO,
            cute.select(self.sdO_layout, mode=[0, 1]),
            (self.tile_m, self.tile_hdimv),
        )
        if const_expr(self.qhead_per_kvhead == 1):
            tma_atom_dK, tma_tensor_dK = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileS2GOp(),
                mdK,
                cute.select(self.sK_layout, mode=[0, 1]),
                (self.tile_n, self.tile_hdim),
            )
            tma_atom_dV, tma_tensor_dV = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileS2GOp(),
                mdV,
                cute.select(self.sV_layout, mode=[0, 1]),
                (self.tile_n, self.tile_hdimv),
            )
        else:
            tma_atom_dK = tma_atom_dV = tma_tensor_dK = tma_tensor_dV = None

        if const_expr(self.deterministic):
            TileScheduler = SingleTileLPTBwdScheduler
        else:
            TileScheduler = SingleTileScheduler
        self.spt = (self.is_causal or self.is_local) and self.deterministic
        tile_sched_args = TileSchedulerArguments(
            cute.ceil_div(cute.size(mK.shape[0]), self.tile_n),
            cute.size(mQ.shape[2]),
            cute.size(mK.shape[3]),  # num_batch
            1,  # num_splits
            cute.size(mQ.shape[0]),  # pass seqlen_q or total_q for seqlen_k
            mQ.shape[1],  # headdim
            mV.shape[1],  # headdim_v
            total_q=cute.size(mK.shape[0]) * cute.size(mK.shape[3]),
            tile_shape_mn=(self.tile_n, self.tile_m),  # Swapping the role of Q & K
            qhead_per_kvhead_packgqa=1,
            element_size=self.dtype.width // 8,
            is_persistent=False,
            lpt=self.spt,
            # head_swizzle=self.deterministic,
        )

        tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)

        LOG2_E = math.log2(math.e)
        softmax_scale_log2 = softmax_scale * LOG2_E

        fastdiv_mods = None

        qhead_per_kvhead_divmod = None
        if const_expr(self.qhead_per_kvhead > 1):
            qhead_per_kvhead_divmod = FastDivmodDivisor(self.qhead_per_kvhead)

        self.use_block_sparsity = cutlass.const_expr(blocksparse_tensors is not None)
        if const_expr(self.use_block_sparsity):
            assert self.dQacc_use_TMA, (
                "block-sparse backward requires the TMA dQaccum staging path "
                "(the atomicAdd dQ path is only used for head_dim=256)"
            )

        if const_expr(window_size_left is not None):
            window_size_left = Int32(window_size_left)
        if const_expr(window_size_right is not None):
            window_size_right = Int32(window_size_right)

        self.kernel(
            tma_tensor_Q,
            tma_tensor_K,
            tma_tensor_V,
            tma_tensor_dO,
            tma_tensor_dK if const_expr(self.qhead_per_kvhead == 1) else mdK,
            tma_tensor_dV if const_expr(self.qhead_per_kvhead == 1) else mdV,
            tma_atom_Q,
            tma_atom_K,
            tma_atom_V,
            tma_atom_dO,
            tma_atom_dK,
            tma_atom_dV,
            mLSE,
            mdPsum,
            mdQaccum,
            self.sQ_layout,
            self.sK_layout,
            self.sV_layout,
            self.sPdS_layout,
            self.sdO_layout,
            self.sdQaccum_layout,
            self.r2s_tiled_copy_dQaccum,
            tiled_mma_SdP,
            tiled_mma_dK,
            tiled_mma_dV,
            tiled_mma_dQ,
            softmax_scale_log2,
            softmax_scale,
            tile_sched_params,
            TileScheduler,
            SharedStorage,
            fastdiv_mods,
            blocksparse_tensors,
            qhead_per_kvhead_divmod,
            mdQ_semaphore,
            mdK_semaphore,
            mdV_semaphore,
            window_size_left,
            window_size_right,
            flashmask_info,
        ).launch(
            grid=grid_dim,
            block=[self.num_threads, 1, 1],
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mdO: cute.Tensor,
        mdK: cute.Tensor,
        mdV: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        tma_atom_dO: cute.CopyAtom,
        tma_atom_dK: cute.CopyAtom,
        tma_atom_dV: cute.CopyAtom,
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        mdQaccum: cute.Tensor,
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sPdS_layout: cute.ComposedLayout,
        sdO_layout: cute.ComposedLayout,
        sdQaccum_layout: cute.Layout,
        r2s_tiled_copy_dQaccum: cute.TiledCopy,
        tiled_mma_SdP: cute.TiledMma,
        tiled_mma_dK: cute.TiledMma,
        tiled_mma_dV: cute.TiledMma,
        tiled_mma_dQ: cute.TiledMma,
        softmax_scale_log2,
        softmax_scale,
        tile_sched_params: ParamsBase,
        TileScheduler: cutlass.Constexpr[Callable],
        SharedStorage: cutlass.Constexpr[Callable],
        fastdiv_mods=(None, None),
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        qhead_per_kvhead_divmod: Optional[FastDivmodDivisor] = None,
        mdQ_semaphore: Optional[cute.Tensor] = None,
        mdK_semaphore: Optional[cute.Tensor] = None,
        mdV_semaphore: Optional[cute.Tensor] = None,
        window_size_left: Optional[Int32] = None,
        window_size_right: Optional[Int32] = None,
        flashmask_info: Optional[FlashMaskInfo] = None,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        # prefetch TMA descriptors
        if warp_idx == 0:
            for atom in [tma_atom_Q, tma_atom_K, tma_atom_V, tma_atom_dO, tma_atom_dK, tma_atom_dV]:
                if const_expr(atom is not None):
                    cpasync.prefetch_descriptor(atom)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        pipeline_producer_group = cutlass.pipeline.CooperativeGroup(cutlass.pipeline.Agent.Thread)
        pipeline_consumer_group = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread, self.num_mma_threads // cute.arch.WARP_SIZE
        )
        pipeline_Q = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.mbar_ptr_Q.data_ptr(),
            num_stages=self.Q_stage,
            producer_group=pipeline_producer_group,
            consumer_group=pipeline_consumer_group,
            tx_count=self.tma_copy_bytes["Q"] + self.tma_copy_bytes["LSE"],
            defer_sync=True,
        )
        pipeline_dO = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.mbar_ptr_dO.data_ptr(),
            num_stages=self.dO_stage,
            producer_group=pipeline_producer_group,
            consumer_group=pipeline_consumer_group,
            tx_count=self.tma_copy_bytes["dO"] + self.tma_copy_bytes["dPsum"],
            defer_sync=False,
        )

        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sdO = storage.sdO.get_tensor(sdO_layout.outer, swizzle=sdO_layout.inner)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
        sP = None
        if const_expr(not self.mma_dkv_is_rs):
            sP = storage.sP.get_tensor(sPdS_layout.outer, swizzle=sPdS_layout.inner)
        sdS = storage.sdS.get_tensor(sPdS_layout.outer, swizzle=sPdS_layout.inner)
        sLSE = storage.sLSE.get_tensor(
            cute.make_layout(
                (self.tile_m, self.Q_stage),
                stride=(1, cute.round_up(self.tile_m, 64)),
            )
        )
        sdPsum = storage.sdPsum.get_tensor(
            cute.make_layout(
                (self.tile_m, self.dO_stage),
                stride=(1, cute.round_up(self.tile_m, 64)),
            )
        )
        sdQaccum = None
        if const_expr(self.dQacc_use_TMA):
            sdQaccum = storage.sdQaccum.get_tensor(sdQaccum_layout)

        s_rowidx = None
        if const_expr(self.enable_flashmask):
            s_rowidx = storage.sRowidx.get_tensor(cute.make_layout(4 * self.tile_n))

        block_info = BlockInfo(
            self.tile_m,
            self.tile_n,
            self.is_causal,
            self.is_local,
            False,  # is_split_kv
            window_size_left,
            window_size_right,
            qhead_per_kvhead_packgqa=1,
        )
        SeqlenInfoCls = partial(
            SeqlenInfoQK.create,
            seqlen_q_static=mQ.shape[0],
            seqlen_k_static=mK.shape[0],
            tile_m=self.tile_m,
            tile_n=self.tile_n,
        )
        # The mask's swap_AB MUST match the SdP MMA layout (acc_S is produced with
        # swap_AB=self.SdP_swapAB). Hardcoding True only worked for the head_dim<=128
        # configs (SdP_swapAB=True); for SdP_swapAB=False configs (e.g. head_dim=256,
        # 192) it transposed the causal/seqlen mask axes -> mask applied to the wrong
        # positions -> hugely wrong dQ/dK/dV.
        AttentionMaskCls = partial(
            AttentionMask,
            self.tile_m,
            self.tile_n,
            swap_AB=self.SdP_swapAB,
        )
        TileSchedulerCls = partial(TileScheduler.create, tile_sched_params)

        if warp_idx < 4:
            cute.arch.setmaxregister_decrease(self.num_producer_regs)
            if warp_idx == 0:
                self.load(
                    mQ,
                    mK,
                    mV,
                    mdO,
                    mLSE,
                    mdPsum,
                    sQ,
                    sK,
                    sV,
                    sdO,
                    sLSE,
                    sdPsum,
                    tma_atom_Q,
                    tma_atom_K,
                    tma_atom_V,
                    tma_atom_dO,
                    pipeline_Q,
                    pipeline_dO,
                    block_info,
                    SeqlenInfoCls,
                    TileSchedulerCls,
                    blocksparse_tensors,
                    qhead_per_kvhead_divmod,
                    flashmask_info,
                    cute.size(mQ.shape[2]),
                )
            if warp_idx == 1:
                if const_expr(self.dQacc_use_TMA):
                    self.dQaccum_store(
                        mdQaccum,
                        sdQaccum,
                        block_info,
                        TileSchedulerCls,
                        SeqlenInfoCls,
                        blocksparse_tensors,
                        mdQ_semaphore,
                        flashmask_info,
                        cute.size(mQ.shape[2]),
                    )
        else:
            tidx, _, _ = cute.arch.thread_idx()
            tidx = tidx - 128
            mma_args = (
                tiled_mma_SdP,
                tiled_mma_dK,
                tiled_mma_dV,
                tiled_mma_dQ,
                mdK,
                mdV,
                mdK_semaphore,
                mdV_semaphore,
                mdQaccum,
                sQ,
                sK,
                sV,
                sdO,
                sP,
                sdS,
                sLSE,
                sdPsum,
                sdQaccum,
                pipeline_Q,
                pipeline_dO,
                tidx,
                tma_atom_dK,
                tma_atom_dV,
                r2s_tiled_copy_dQaccum,
                softmax_scale_log2,
                softmax_scale,
                block_info,
                SeqlenInfoCls,
                AttentionMaskCls,
                TileSchedulerCls,
                fastdiv_mods,
                blocksparse_tensors,
                qhead_per_kvhead_divmod,
                flashmask_info,
                s_rowidx,
                cute.size(mQ.shape[2]),
            )
            if const_expr(self.num_wg_dQ == self.num_wg_mma):
                # Both WGs compute dQ
                cute.arch.setmaxregister_increase(self.num_mma_regs_wg0)
                self.mma(*mma_args, is_dQ_wg=True)
            else:
                # WG0 computes dQ, WG1 skips it
                warp_idx_in_mma = cute.arch.make_warp_uniform(cute.arch.warp_idx()) - 4
                if warp_idx_in_mma < 4:
                    cute.arch.setmaxregister_increase(self.num_mma_regs_wg0)
                    self.mma(*mma_args, is_dQ_wg=True)
                else:
                    cute.arch.setmaxregister_increase(self.num_mma_regs_wg1)
                    self.mma(*mma_args, is_dQ_wg=False)

    @cute.jit
    def load(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mdO: cute.Tensor,
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        sdO: cute.Tensor,
        sLSE: cute.Tensor,
        sdPsum: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        tma_atom_dO: cute.CopyAtom,
        pipeline_Q: cutlass.pipeline.PipelineAsync,
        pipeline_dO: cutlass.pipeline.PipelineAsync,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        qhead_per_kvhead_divmod: Optional[FastDivmodDivisor] = None,
        flashmask_info: Optional[FlashMaskInfo] = None,
        num_heads: Int32 = None,
    ):
        warp_idx_in_wg = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4

        if warp_idx_in_wg == 0:
            producer_state_Q = cutlass.pipeline.make_pipeline_state(
                cutlass.pipeline.PipelineUserType.Producer, self.Q_stage
            )
            producer_state_dO = cutlass.pipeline.make_pipeline_state(
                cutlass.pipeline.PipelineUserType.Producer, self.dO_stage
            )
            tile_scheduler = TileSchedulerCls()
            work_tile = tile_scheduler.initial_work_tile_info()
            while work_tile.is_valid_tile:
                n_block, head_idx, batch_idx, _ = work_tile.tile_idx
                seqlen = SeqlenInfoCls(batch_idx)
                head_idx_kv = (
                    head_idx
                    if const_expr(self.qhead_per_kvhead == 1)
                    else head_idx // qhead_per_kvhead_divmod
                )
                mK_cur = seqlen.offset_batch_K(mK, batch_idx, dim=3)[None, None, head_idx_kv]
                mV_cur = seqlen.offset_batch_K(mV, batch_idx, dim=3)[None, None, head_idx_kv]
                gK = cute.local_tile(mK_cur, (self.tile_n, self.tile_hdim), (n_block, 0))
                gV = cute.local_tile(mV_cur, (self.tile_n, self.tile_hdimv), (n_block, 0))

                mQ_cur = seqlen.offset_batch_Q(mQ, batch_idx, dim=3)[None, None, head_idx]
                mLSE_cur = seqlen.offset_batch_Q(mLSE, batch_idx, dim=2, padded=True)[
                    None, head_idx
                ]
                mdO_cur = seqlen.offset_batch_Q(mdO, batch_idx, dim=3)[None, None, head_idx]
                mdPsum_cur = seqlen.offset_batch_Q(mdPsum, batch_idx, dim=2, padded=True)[
                    None, head_idx
                ]
                gQ = cute.local_tile(mQ_cur, (self.tile_m, self.tile_hdim), (None, 0))
                gdO = cute.local_tile(mdO_cur, (self.tile_m, self.tile_hdimv), (None, 0))
                gLSE = cute.local_tile(mLSE_cur, (self.tile_m,), (None,))
                gdPsum = cute.local_tile(mdPsum_cur, (self.tile_m,), (None,))

                load_K, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_K, 0, cute.make_layout(1), gK, sK, single_stage=True
                )
                load_V, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_V, 0, cute.make_layout(1), gV, sV, single_stage=True
                )
                load_Q, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_Q, 0, cute.make_layout(1), gQ, sQ
                )
                load_Q = copy_utils.tma_producer_copy_fn(load_Q, pipeline_Q)
                load_dO, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_dO, 0, cute.make_layout(1), gdO, sdO
                )
                load_dO = copy_utils.tma_producer_copy_fn(load_dO, pipeline_dO)
                load_LSE = copy_utils.cpasync_bulk_get_copy_fn(gLSE, sLSE)
                load_LSE = copy_utils.tma_producer_copy_fn(load_LSE, pipeline_Q)
                load_dPsum = copy_utils.cpasync_bulk_get_copy_fn(gdPsum, sdPsum)
                load_dPsum = copy_utils.tma_producer_copy_fn(load_dPsum, pipeline_dO)

                m_block_min, m_block_max = block_info.get_m_block_min_max(seqlen, n_block)

                if const_expr(self.enable_flashmask and not self.deterministic):
                    # O(1) segment jump (must match the MMA/dQ-store warps'
                    # segments exactly so the Q/dO TMA pipeline producer/consumer
                    # counts stay in lockstep).
                    flashmask_segments = self.flashmask_m_block_segments(
                        flashmask_info,
                        batch_idx,
                        head_idx,
                        n_block,
                        num_heads,
                        m_block_min,
                        m_block_max,
                    )
                    total_m_block_cnt = Int32(0)
                    for seg_lo, seg_hi in flashmask_segments:
                        total_m_block_cnt = total_m_block_cnt + (seg_hi - seg_lo)
                    process_tile = total_m_block_cnt > Int32(0)
                elif const_expr(not self.use_block_sparsity):
                    total_m_block_cnt = m_block_max - m_block_min
                    process_tile = (
                        const_expr(not self.is_local)
                        or m_block_min < m_block_max
                    )
                else:
                    total_m_block_cnt = get_total_q_block_count_bwd(
                        blocksparse_tensors,
                        batch_idx,
                        head_idx,
                        n_block,
                        subtile_factor=self.subtile_factor,
                        m_block_max=m_block_max,
                    )
                    process_tile = total_m_block_cnt > Int32(0)

                if process_tile:
                    if const_expr(not self.use_block_sparsity):
                        if const_expr(self.enable_flashmask and not self.deterministic):
                            # Segment-based load: iterate only the non-masked
                            # m_block segments (cpp-equivalent), loading K/V once
                            # on the first visited block (kv_loaded flag, same as
                            # the block-sparse producer). No per-block gmem skip.
                            # Wait for bwd preprocess to finish writing LSE/dPsum
                            # before the first LSE/dPsum load.
                            cute.arch.griddepcontrol_wait()
                            kv_loaded = False
                            for seg_lo, seg_hi in flashmask_segments:
                                for m_block in cutlass.range(seg_lo, seg_hi, unroll=1):
                                    producer_state_Q, producer_state_dO = _load_q_do_block_sm90(
                                        m_block,
                                        producer_state_Q,
                                        producer_state_dO,
                                        pipeline_Q,
                                        pipeline_dO,
                                        load_K,
                                        load_V,
                                        load_Q,
                                        load_dO,
                                        load_LSE,
                                        load_dPsum,
                                        self.tma_copy_bytes["K"],
                                        self.tma_copy_bytes["V"],
                                        self.Q_stage == self.dO_stage,
                                        load_kv=not kv_loaded,
                                    )
                                    kv_loaded = True
                        else:
                            first_m_block = m_block_min
                            pipeline_Q.producer_acquire(
                                producer_state_Q, extra_tx_count=self.tma_copy_bytes["K"]
                            )
                            load_K(tma_bar_ptr=pipeline_Q.producer_get_barrier(producer_state_Q))
                            load_Q(first_m_block, producer_state=producer_state_Q)
                            # Wait for bwd preprocess to finish writing LSE and dPsum
                            cute.arch.griddepcontrol_wait()
                            load_LSE(first_m_block, producer_state=producer_state_Q)
                            producer_state_dO_cur = (
                                producer_state_dO
                                if const_expr(self.Q_stage != self.dO_stage)
                                else producer_state_Q
                            )
                            pipeline_dO.producer_acquire(
                                producer_state_dO_cur, extra_tx_count=self.tma_copy_bytes["V"]
                            )
                            load_V(tma_bar_ptr=pipeline_dO.producer_get_barrier(producer_state_dO_cur))
                            load_dO(first_m_block, producer_state=producer_state_dO_cur)
                            load_dPsum(first_m_block, producer_state=producer_state_dO_cur)
                            producer_state_Q.advance()
                            producer_state_dO.advance()

                            for m_block in cutlass.range(m_block_min + 1, m_block_max, unroll=1):
                                pipeline_Q.producer_acquire(producer_state_Q)
                                load_Q(m_block, producer_state=producer_state_Q)
                                load_LSE(m_block, producer_state=producer_state_Q)
                                producer_state_dO_cur = (
                                    producer_state_dO
                                    if const_expr(self.Q_stage != self.dO_stage)
                                    else producer_state_Q
                                )
                                pipeline_dO.producer_acquire(producer_state_dO_cur)
                                load_dO(m_block, producer_state=producer_state_dO_cur)
                                load_dPsum(m_block, producer_state=producer_state_dO_cur)
                                producer_state_Q.advance()
                                producer_state_dO.advance()
                    else:
                        producer_state_Q, producer_state_dO = produce_block_sparse_q_loads_bwd_sm90(
                            blocksparse_tensors,
                            batch_idx,
                            head_idx,
                            n_block,
                            producer_state_Q,
                            producer_state_dO,
                            pipeline_Q,
                            pipeline_dO,
                            load_K,
                            load_V,
                            load_Q,
                            load_dO,
                            load_LSE,
                            load_dPsum,
                            self.tma_copy_bytes["K"],
                            self.tma_copy_bytes["V"],
                            Q_stage_eq_dO_stage=(self.Q_stage == self.dO_stage),
                            subtile_factor=self.subtile_factor,
                            m_block_max=m_block_max,
                        )

                tile_scheduler.prefetch_next_work()
                tile_scheduler.advance_to_next_work()
                work_tile = tile_scheduler.get_current_work()

    @cute.jit
    def mma(
        self,
        tiled_mma_SdP: cute.TiledMma,
        tiled_mma_dK: cute.TiledMma,
        tiled_mma_dV: cute.TiledMma,
        tiled_mma_dQ: cute.TiledMma,
        mdK: cute.Tensor,
        mdV: cute.Tensor,
        mdK_semaphore: Optional[cute.Tensor],
        mdV_semaphore: Optional[cute.Tensor],
        mdQaccum: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        sdO: cute.Tensor,
        sP: Optional[cute.Tensor],
        sdS: cute.Tensor,
        sLSE: cute.Tensor,
        sdPsum: cute.Tensor,
        sdQaccum: cute.Tensor,
        pipeline_Q: cutlass.pipeline.PipelineAsync,
        pipeline_dO: cutlass.pipeline.PipelineAsync,
        tidx: Int32,
        tma_atom_dK: cute.CopyAtom,
        tma_atom_dV: cute.CopyAtom,
        r2s_tiled_copy_dQaccum: cute.TiledCopy,
        softmax_scale_log2: Float32,
        softmax_scale: Float32,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        AttentionMaskCls: Callable,
        TileSchedulerCls: Callable,
        fastdiv_mods=(None, None),
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        qhead_per_kvhead_divmod: Optional[FastDivmodDivisor] = None,
        flashmask_info: Optional[FlashMaskInfo] = None,
        s_rowidx: Optional[cute.Tensor] = None,
        num_heads: Int32 = None,
        is_dQ_wg: cutlass.Constexpr[bool] = True,
    ):
        warp_group_idx = cute.arch.make_warp_uniform(tidx // self.num_threads_per_warp_group)
        warp_group_thread_layout = cute.make_layout(
            self.num_wg_mma, stride=self.num_threads_per_warp_group
        )
        thr_mma_SdP = tiled_mma_SdP.get_slice(tidx)
        wg_mma_SdP = tiled_mma_SdP.get_slice(warp_group_thread_layout(warp_group_idx))
        wg_mma_dK = tiled_mma_dK.get_slice(warp_group_thread_layout(warp_group_idx))
        wg_mma_dV = tiled_mma_dV.get_slice(warp_group_thread_layout(warp_group_idx))
        wg_mma_dQ = None
        if const_expr(is_dQ_wg):
            wg_idx_dQ = warp_group_idx if const_expr(self.num_wg_dQ > 1) else 0
            wg_mma_dQ = tiled_mma_dQ.get_slice(warp_group_thread_layout(wg_idx_dQ))
        # S = Q @ K.T
        shape_mnk_S = (self.tile_m, self.tile_n, self.tile_hdim)
        _, tSrQ, tSrK = sm90_utils.partition_fragment_ABC(
            wg_mma_SdP, shape_mnk_S, sQ, sK, swap_AB=self.SdP_swapAB
        )
        mma_qk_fn = partial(
            gemm_zero_init, tiled_mma_SdP, shape_mnk_S[:2], tSrQ, tSrK, swap_AB=self.SdP_swapAB
        )
        # dP = dO @ V.T
        shape_mnk_dP = (self.tile_m, self.tile_n, self.tile_hdimv)
        _, tdPrdO, tdPrV = sm90_utils.partition_fragment_ABC(
            wg_mma_SdP, shape_mnk_dP, sdO, sV, swap_AB=self.SdP_swapAB
        )
        mma_dov_fn = partial(
            gemm_zero_init, tiled_mma_SdP, shape_mnk_dP[:2], tdPrdO, tdPrV, swap_AB=self.SdP_swapAB
        )
        # dV += P.T @ dO
        sPt = layout_utils.transpose_view(sP) if sP is not None else None
        sdOt = layout_utils.transpose_view(sdO)
        shape_mnk_dV = (self.tile_n, self.tile_hdimv, self.tile_m)
        acc_dV, tdVrPt, tdVrdOt = sm90_utils.partition_fragment_ABC(
            wg_mma_dV, shape_mnk_dV, sPt, sdOt, swap_AB=self.dKV_swapAB
        )
        if const_expr(not self.mma_dkv_is_rs):
            mma_pdo_fn = partial(
                gemm_w_idx, tiled_mma_dV, acc_dV, tdVrPt, tdVrdOt, swap_AB=self.dKV_swapAB
            )
        else:
            mma_pdo_fn = partial(gemm_w_idx, tiled_mma_dV, acc_dV, tCrB=tdVrdOt)
        # dK += dS.T @ Q
        sdSt = layout_utils.transpose_view(sdS)
        sQt = layout_utils.transpose_view(sQ)
        shape_mnk_dK = (self.tile_n, self.tile_hdim, self.tile_m)
        acc_dK, tdKrdSt, tdKrQt = sm90_utils.partition_fragment_ABC(
            wg_mma_dK, shape_mnk_dK, sdSt, sQt, swap_AB=self.dKV_swapAB
        )
        if const_expr(not self.mma_dkv_is_rs):
            mma_dsq_fn = partial(
                gemm_w_idx, tiled_mma_dK, acc_dK, tdKrdSt, tdKrQt, swap_AB=self.dKV_swapAB
            )
        else:
            mma_dsq_fn = partial(gemm_w_idx, tiled_mma_dK, acc_dK, tCrB=tdKrQt)
        # dQ = dS @ K
        sKt = layout_utils.transpose_view(sK)
        shape_mnk_dQ = (self.tile_m, self.tile_hdim, self.tile_n)
        mma_dsk_fn = None
        mma_dsk_slice_fn = None
        dQ_acc_shape = None
        if const_expr(is_dQ_wg):
            _, tdQrdS, tdQrKt = sm90_utils.partition_fragment_ABC(
                wg_mma_dQ, shape_mnk_dQ, sdS, sKt, swap_AB=self.dQ_swapAB
            )
            mma_dsk_fn = partial(
                gemm_zero_init,
                tiled_mma_dQ,
                shape_mnk_dQ[:2],
                tdQrdS,
                tdQrKt,
                swap_AB=self.dQ_swapAB,
            )
            if const_expr(self.slice_dQKV_mma):
                # For the sliced schedule we allocate acc_dQ once per m_block and
                # drive it with two m_slice MMA calls, so bind a gemm_w_idx variant
                # (acc passed in) and precompute the (swapped) dQ C-accum shape.
                mma_dsk_slice_fn = partial(
                    gemm_w_idx, tiled_mma_dQ, tCrA=tdQrdS, tCrB=tdQrKt, swap_AB=self.dQ_swapAB
                )
                dQ_acc_shape = tiled_mma_dQ.partition_shape_C((self.tile_hdim, self.tile_m))

        # Smem copy atom tiling for P/dS R2S
        copy_P_r2s = None
        mms_PdS = self.tile_n // (self.num_wg_mma // self.AtomLayoutMSdP)
        if const_expr(sP is not None):
            sP_cpy = sP if const_expr(not self.SdP_swapAB) else sPt
            copy_P_r2s, _, _ = copy_utils.get_smem_store_C(
                tiled_mma_SdP,
                sP_cpy,
                tidx,
                transpose=self.SdP_swapAB,
                position_independent=True,
                major_mode_size=mms_PdS,
            )
        sdS_cpy = sdS if const_expr(not self.SdP_swapAB) else sdSt
        copy_dS_r2s, _, _ = copy_utils.get_smem_store_C(
            tiled_mma_SdP,
            sdS_cpy,
            tidx,
            transpose=self.SdP_swapAB,
            position_independent=True,
            major_mode_size=mms_PdS,
        )

        tLSEsLSE = layout_utils.mma_partition_C_vec(
            sLSE, thr_mma_SdP, expand_shape=self.tile_n, is_colvec=not self.SdP_swapAB
        )
        tLSEsdPsum = layout_utils.mma_partition_C_vec(
            sdPsum, thr_mma_SdP, expand_shape=self.tile_n, is_colvec=not self.SdP_swapAB
        )
        # When shuffle=True, rows are distributed across 8 quads (4 threads each) within a warp.
        # Each thread loads only ceil(num_rows/8) values;
        shfl_copy = copy_utils.tiled_copy_1d(sLSE.element_type, num_threads=8, num_copy_elems=2)
        if const_expr(self.shuffle_LSE):
            tLSEsLSE = shfl_copy.get_slice(cute.arch.lane_idx() // 4).partition_S(tLSEsLSE)
            # ((2, 1), 1, 2) -> (((2, 1), 1), 2)
            tLSEsLSE = cute.group_modes(tLSEsLSE, 0, 2)
        if const_expr(self.shuffle_dPsum):
            tLSEsdPsum = shfl_copy.get_slice(cute.arch.lane_idx() // 4).partition_S(tLSEsdPsum)
            tLSEsdPsum = cute.group_modes(tLSEsdPsum, 0, 2)

        tdQsdQaccum = None
        smem_thr_copy_dQaccum = None
        if const_expr(is_dQ_wg):
            smem_thr_copy_dQaccum = r2s_tiled_copy_dQaccum.get_slice(tidx)
            if const_expr(self.dQacc_use_TMA):
                tdQsdQaccum = smem_thr_copy_dQaccum.partition_D(sdQaccum)

        PdS_barrier = cutlass.pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierBwd.PdS), num_threads=self.num_mma_threads
        )
        flashmask_barrier = None
        if const_expr(self.enable_flashmask):
            flashmask_barrier = cutlass.pipeline.NamedBarrier(
                barrier_id=int(NamedBarrierBwd.FlashMask), num_threads=self.num_mma_threads
            )

        mma_one_m_block_all = partial(
            self.mma_one_m_block,
            warp_group_idx=warp_group_idx,
            mma_qk_fn=mma_qk_fn,
            mma_dov_fn=mma_dov_fn,
            mma_pdo_fn=mma_pdo_fn,
            mma_dsq_fn=mma_dsq_fn,
            mma_dsk_fn=mma_dsk_fn,
            copy_P_r2s=copy_P_r2s,
            copy_dS_r2s=copy_dS_r2s,
            pipeline_Q=pipeline_Q,
            pipeline_dO=pipeline_dO,
            tLSEsLSE=tLSEsLSE,
            tLSEsdPsum=tLSEsdPsum,
            tdQsdQaccum=tdQsdQaccum,
            softmax_scale_log2=softmax_scale_log2,
            PdS_barrier=PdS_barrier,
            mma_dsk_slice_fn=mma_dsk_slice_fn,
            dQ_acc_shape=dQ_acc_shape,
            # acc_dV=acc_dV,
            # acc_dK=acc_dK,
            is_dQ_wg=is_dQ_wg,
        )

        consumer_state_Q = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.Q_stage
        )
        consumer_state_dO = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.dO_stage
        )
        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            n_block, head_idx, batch_idx, _ = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            # atomicAdd dQ path: partition this (head, batch)'s gmem dQaccum with
            # the same tiled copy used for the smem staging, so element i of the
            # dQ register accumulator maps to gmem index i (mirrors C++ FA3).
            tdQgdQaccum = None
            if const_expr(is_dQ_wg and not self.dQacc_use_TMA):
                mdQaccum_cur = mdQaccum[None, head_idx, batch_idx]
                # (M*K, num_m_blocks) -> (M*K/WG, WG, num_m_blocks), matching the
                # (M*K/WG, WG) smem staging layout so partition_D lines up 1:1.
                gdQaccum_ = cute.local_tile(
                    mdQaccum_cur, (self.tile_m * self.tile_hdim,), (None,)
                )
                gdQaccum = cute.flat_divide(
                    gdQaccum_, (self.tile_m * self.tile_hdim // self.num_wg_dQ,)
                )
                tdQgdQaccum = smem_thr_copy_dQaccum.partition_D(gdQaccum)
            # mask = AttentionMaskCls(seqlen)
            mask = AttentionMaskCls(seqlen.seqlen_q, seqlen.seqlen_k)
            m_block_min, m_block_max = block_info.get_m_block_min_max(seqlen, n_block)

            if const_expr(self.enable_flashmask and not self.deterministic):
                # flashmask O(1) segment jump: the non-masked query blocks form
                # up to 3 contiguous segments (cpp-equivalent). Non-deterministic
                # only, so dQ is atomic-accumulated and skipping n/m pairs cannot
                # break the dQ semaphore protocol.
                flashmask_segments = self.flashmask_m_block_segments(
                    flashmask_info,
                    batch_idx,
                    head_idx,
                    n_block,
                    num_heads,
                    m_block_min,
                    m_block_max,
                )
                total_m_block_cnt = Int32(0)
                for seg_lo, seg_hi in flashmask_segments:
                    total_m_block_cnt = total_m_block_cnt + (seg_hi - seg_lo)
                process_tile = total_m_block_cnt > Int32(0)
            elif const_expr(not self.use_block_sparsity):
                process_tile = (
                    const_expr(not self.is_local)
                    or m_block_min < m_block_max
                )
            else:
                total_m_block_cnt = get_total_q_block_count_bwd(
                    blocksparse_tensors,
                    batch_idx,
                    head_idx,
                    n_block,
                    subtile_factor=self.subtile_factor,
                    m_block_max=m_block_max,
                )
                process_tile = total_m_block_cnt > Int32(0)

            if process_tile:
                if const_expr(not self.use_block_sparsity):
                    mask_fn = partial(
                        mask.apply_mask,
                        batch_idx=batch_idx,
                        head_idx=head_idx,
                        n_block=n_block,
                        thr_mma=thr_mma_SdP,
                        mask_seqlen=True,
                        mask_causal=self.is_causal,
                        mask_local=self.is_local,
                    )
                    dKV_accumulate = False
                    # flashmask: stage this n_block's startend_row_indices into smem
                    # once (fixed across the whole m loop), and build the apply fn.
                    flashmask_fn = None
                    if const_expr(self.enable_flashmask):
                        # Ensure any prior n_block's reads of s_rowidx are done.
                        flashmask_barrier.arrive_and_wait()
                        self.load_flashmask_n_block(
                            flashmask_info,
                            batch_idx,
                            head_idx,
                            n_block,
                            num_heads,
                            s_rowidx,
                            seqlen.seqlen_k,
                            tidx,
                        )
                        flashmask_barrier.arrive_and_wait()
                        flashmask_fn = partial(
                            mask.apply_flashmask_sm90,
                            thr_mma=thr_mma_SdP,
                            s_startend_row_indices=s_rowidx,
                            has_lt_end=self.has_lt_end,
                            has_ut_start=self.has_ut_start,
                            has_ut_end=self.has_ut_end,
                        )
                    if const_expr(self.enable_flashmask and not self.deterministic):
                        # Iterate only the non-masked segments; all three warps
                        # walk the identical segment list in the same order to
                        # stay in lockstep. The element mask is applied only on
                        # partially-masked blocks (full/partial split, cpp-equivalent):
                        # fully-visible interior blocks skip the per-element
                        # comparison, which is the dominant remaining cost on
                        # dense masks at long seqlen. Gating flashmask_fn does not
                        # change which blocks are visited, so lockstep is preserved.
                        (
                            p_lts_min,
                            p_lte_max,
                            p_uts_min,
                            p_ute_max,
                        ) = self.flashmask_m_block_partial_bounds(
                            flashmask_info,
                            batch_idx,
                            head_idx,
                            n_block,
                            num_heads,
                        )
                        for seg_lo, seg_hi in flashmask_segments:
                            for m_block in cutlass.range(seg_lo, seg_hi, unroll=1):
                                if const_expr(self.has_ut_start):
                                    partially_masked = (
                                        ((m_block < p_lte_max) & (m_block >= p_lts_min))
                                        | ((m_block < p_ute_max) & (m_block >= p_uts_min))
                                    )
                                elif const_expr(self.has_lt_end):
                                    partially_masked = (m_block < p_lte_max) & (
                                        m_block >= p_lts_min
                                    )
                                elif const_expr(self.has_ut_end):
                                    partially_masked = (m_block >= p_lts_min) | (
                                        m_block < p_ute_max
                                    )
                                else:
                                    partially_masked = m_block >= p_lts_min
                                consumer_state_Q, consumer_state_dO = mma_one_m_block_all(
                                    m_block,
                                    consumer_state_Q,
                                    consumer_state_dO,
                                    mask_fn=mask_fn,
                                    flashmask_fn=flashmask_fn,
                                    partially_masked=partially_masked,
                                    dKV_accumulate=dKV_accumulate,
                                    tdQgdQaccum=tdQgdQaccum,
                                )
                                dKV_accumulate = True
                    else:
                        for m_block in cutlass.range(m_block_min, m_block_max, unroll=1):
                            consumer_state_Q, consumer_state_dO = mma_one_m_block_all(
                                m_block,
                                consumer_state_Q,
                                consumer_state_dO,
                                mask_fn=mask_fn,
                                flashmask_fn=flashmask_fn,
                                dKV_accumulate=dKV_accumulate,
                                tdQgdQaccum=tdQgdQaccum,
                            )
                            dKV_accumulate = True
                else:
                    consumer_state_Q, consumer_state_dO = consume_block_sparse_mma_bwd_sm90(
                        blocksparse_tensors,
                        batch_idx,
                        head_idx,
                        n_block,
                        consumer_state_Q,
                        consumer_state_dO,
                        mma_one_m_block_all,
                        mask,
                        is_causal=self.is_causal,
                        is_local=self.is_local,
                        thr_mma_SdP=thr_mma_SdP,
                        subtile_factor=self.subtile_factor,
                        m_block_max=m_block_max,
                    )

                if const_expr(self.qhead_per_kvhead == 1):
                    acc_dK.store(acc_dK.load() * softmax_scale)
                self.epilogue_dKV(
                    acc_dV,
                    mdV,
                    sV,
                    acc_dK,
                    mdK,
                    sK,
                    seqlen,
                    tma_atom_dK,
                    tma_atom_dV,
                    tiled_mma_dK,
                    tiled_mma_dV,
                    tidx,
                    n_block,
                    head_idx,
                    batch_idx,
                    qhead_per_kvhead_divmod,
                    mdK_semaphore,
                    mdV_semaphore,
                )
            else:
                # KV tile with zero Q blocks produces no dK/dV; write zeros.
                if const_expr(
                    self.use_block_sparsity
                    or self.is_local
                    or (self.enable_flashmask and not self.deterministic)
                ):
                    acc_dK.fill(0.0)
                    acc_dV.fill(0.0)
                    self.epilogue_dKV(
                        acc_dV,
                        mdV,
                        sV,
                        acc_dK,
                        mdK,
                        sK,
                        seqlen,
                        tma_atom_dK,
                        tma_atom_dV,
                        tiled_mma_dK,
                        tiled_mma_dV,
                        tidx,
                        n_block,
                        head_idx,
                        batch_idx,
                        qhead_per_kvhead_divmod,
                        mdK_semaphore,
                        mdV_semaphore,
                    )

            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == 4:
            cute.arch.cp_async_bulk_wait_group(0, read=True)

    @cute.jit
    def load_flashmask_n_block(
        self,
        flashmask_info: FlashMaskInfo,
        batch_idx: Int32,
        head_idx: Int32,
        n_block: Int32,
        num_heads: Int32,
        s_rowidx: cute.Tensor,
        seqlen_k: Int32,
        tidx: Int32,
    ) -> None:
        """Stage the current n_block's startend_row_indices (LTS/LTE/UTS/UTE) into
        smem, once per n_block, cooperatively across all MMA/consumer threads.
        smem layout: 4 vectors of tile_n Int32 at [0, tile_n, 2*tile_n, 3*tile_n].
        Caller must barrier after this so all threads see the staged indices."""
        fm_heads = flashmask_info.startend_row_indices.shape[1]
        fm_head_idx = head_idx // (num_heads // fm_heads)
        nb_mul_kBN = n_block * self.tile_n
        loop_ub = cutlass.min(self.tile_n, seqlen_k - nb_mul_kBN)
        srow = flashmask_info.startend_row_indices
        idx = tidx
        if const_expr(self.has_ut_start):
            while idx < loop_ub:
                s_rowidx[idx] = srow[batch_idx, fm_head_idx, nb_mul_kBN + idx, 0]
                s_rowidx[self.tile_n + idx] = srow[batch_idx, fm_head_idx, nb_mul_kBN + idx, 1]
                s_rowidx[self.tile_n * 2 + idx] = srow[batch_idx, fm_head_idx, nb_mul_kBN + idx, 2]
                s_rowidx[self.tile_n * 3 + idx] = srow[batch_idx, fm_head_idx, nb_mul_kBN + idx, 3]
                idx += self.num_mma_threads
        elif const_expr(self.has_lt_end):
            while idx < loop_ub:
                s_rowidx[idx] = srow[batch_idx, fm_head_idx, nb_mul_kBN + idx, 0]
                s_rowidx[self.tile_n + idx] = srow[batch_idx, fm_head_idx, nb_mul_kBN + idx, 1]
                idx += self.num_mma_threads
        elif const_expr(self.has_ut_end):
            while idx < loop_ub:
                s_rowidx[idx] = srow[batch_idx, fm_head_idx, nb_mul_kBN + idx, 0]
                s_rowidx[self.tile_n * 3 + idx] = srow[batch_idx, fm_head_idx, nb_mul_kBN + idx, 1]
                idx += self.num_mma_threads
        else:
            while idx < loop_ub:
                s_rowidx[idx] = srow[batch_idx, fm_head_idx, nb_mul_kBN + idx, 0]
                idx += self.num_mma_threads

    def flashmask_m_block_segments(
        self,
        flashmask_info: FlashMaskInfo,
        batch_idx: Int32,
        head_idx: Int32,
        n_block: Int32,
        num_heads: Int32,
        m_block_min: Int32,
        m_block_max: Int32,
    ):
        """O(1) replacement for the per-block flashmask trim/skip.

        Reads the 1-4 precomputed ``*_nblock`` max/min scalars for this KV block
        once into registers and derives the (up to 3) contiguous m_block
        segments that are NOT fully masked, mirroring the cpp mainloop's fm_mem
        segment jumps (csrc/flashmask_v2/mainloop_bwd_sm90_tma_gmma_ws.hpp,
        ``load()`` / ``store_dq()``). This replaces the old both-end ``while``
        trim plus the per-m_block predicate, which for seqlen=128k issued
        O(m_block) serial gmem scalar reads per KV block (the dominant bwd
        overhead vs the cpp version).

        The complement of the returned segments is exactly the union of the two
        fully-masked bands (lt band ``[p0+1, p3)`` and ut band ``[p4+1, p7)``);
        boundaries use floor/ceil the same way the cpp reference does, so they
        err toward over-visiting (a fully-masked block still yields zero once
        the element mask is applied) and never under-visit.

        Returns a Python list (length fixed at trace time by the mask config)
        of ``(lo, hi)`` Int32 half-open ranges. The producer, MMA-consumer and
        dQ-store warps all call this with identical arguments and iterate the
        identical list in the same order, so the Q/dO pipeline and per-m_block
        dQ handshake stay in lockstep.
        """
        T = self.tile_m
        fm_heads = flashmask_info.startend_row_indices.shape[1]
        fm_head_idx = head_idx // (num_heads // fm_heads)
        lts_max = flashmask_info.LTS_nblock_max[batch_idx, fm_head_idx, n_block]
        # p0: largest m_block whose m_start (= m*tile_m) is still < lts_max, i.e.
        # the last visible block of the lt_start bottom band (cpp fm_mem[0]).
        p0 = (lts_max - 1) // T
        segments = []
        m_block = m_block_min
        # Upper-triangle start band (num_vecs == 4 only): segment [m_block, p4].
        if const_expr(self.has_ut_start):
            uts_max = flashmask_info.UTS_nblock_max[batch_idx, fm_head_idx, n_block]
            p4 = (uts_max - 1) // T
            a_lo = m_block
            a_hi = cutlass.min(m_block_max, cutlass.max(p4 + 1, a_lo))
            segments.append((a_lo, a_hi))
            m_block = a_hi
        # Skip the upper-triangle masked band by jumping to ute_min. Runs for all
        # non-causal configs, i.e. whenever ut_end exists (cpp: m_block =
        # max(m_block, fm_mem[7])).
        if const_expr(self.has_ut_end):
            ute_min = flashmask_info.UTE_nblock_min[batch_idx, fm_head_idx, n_block]
            p7 = ute_min // T
            m_block = cutlass.max(m_block, p7)
        # lt_start bottom band: segment [m_block, min(m_block_max - 1, p0)].
        b_lo = m_block
        b_hi = cutlass.max(b_lo, cutlass.min(m_block_max, p0 + 1))
        segments.append((b_lo, b_hi))
        m_block = b_hi
        # lt_end top band (LTE present): jump past the lt masked middle band to
        # lte_min, then run to m_block_max (cpp: m_block = max(m_block,
        # fm_mem[3])).
        if const_expr(self.has_lt_end):
            lte_min = flashmask_info.LTE_nblock_min[batch_idx, fm_head_idx, n_block]
            p3 = lte_min // T
            # Cap c_lo at m_block_max so the segment runs [c_lo, m_block_max) like
            # the lt/ut bands above; degenerates to an empty range when the jump
            # target p3 is past m_block_max.
            c_lo = cutlass.min(m_block_max, cutlass.max(m_block, p3))
            c_hi = m_block_max
            segments.append((c_lo, c_hi))
        return segments

    def flashmask_m_block_partial_bounds(
        self,
        flashmask_info: FlashMaskInfo,
        batch_idx: Int32,
        head_idx: Int32,
        n_block: Int32,
        num_heads: Int32,
    ):
        """Compute the m_block thresholds that decide, per visited query block,
        whether the flashmask element mask must be applied (partial block) or
        can be skipped (fully-visible block). Mirrors the cpp bwd_step
        `partially_masked` flag (mainloop_bwd_sm90_tma_gmma_ws.hpp) and the
        forward `split_full` classification in flashmask_utils.compute_flashmask_block_lists.

        Reads only the "inner" boundary scalars once into registers (LTS min,
        LTE max, UTS min, UTE max; 1-4 depending on config) and converts them
        to m_block space:
          `m_end   > Y` (needs mask, low side)  <=> m_block >= Y // tile_m
          `m_start < X` (needs mask, high side) <=> m_block < ceil_div(X, tile_m)
        The block-level max/min aggregate is a superset of the true per-column
        partial set, and the m_end -> (m+1)*tile_m rounding over-includes at the
        seqlen edge, so the predicate never skips a block that needs masking
        (over-applying the element mask is a no-op). Returns
        (p_lts_min, p_lte_max, p_uts_min, p_ute_max); the caller selects the
        relevant subset per config via const_expr.
        """
        T = self.tile_m
        fm_heads = flashmask_info.startend_row_indices.shape[1]
        fm_head_idx = head_idx // (num_heads // fm_heads)
        lts_min = flashmask_info.LTS_nblock_min[batch_idx, fm_head_idx, n_block]
        p_lts_min = lts_min // T
        p_lte_max = Int32(0)
        p_uts_min = Int32(0)
        p_ute_max = Int32(0)
        if const_expr(self.has_lt_end):
            lte_max = flashmask_info.LTE_nblock_max[batch_idx, fm_head_idx, n_block]
            p_lte_max = (lte_max + T - 1) // T
        if const_expr(self.has_ut_start):
            uts_min = flashmask_info.UTS_nblock_min[batch_idx, fm_head_idx, n_block]
            p_uts_min = uts_min // T
        if const_expr(self.has_ut_end):
            ute_max = flashmask_info.UTE_nblock_max[batch_idx, fm_head_idx, n_block]
            p_ute_max = (ute_max + T - 1) // T
        return p_lts_min, p_lte_max, p_uts_min, p_ute_max

    @staticmethod
    @cute.jit
    def _get_stat(tSrS: cute.Tensor, row: Int32, lane: Int32, shuffle: bool) -> Float32:
        """Retrieve the statistic for a given accumulator row.

        When shuffle=False, direct register indexing.
        When shuffle=True, warp shuffle from the thread group that holds the value.
        """
        if const_expr(not shuffle):
            return tSrS[row]
        # tSrS: (((2, 1), 1), 1)), distributed across 8 threads in the warp
        vecsize = cute.size(tSrS, mode=[0, 0])  # 2
        idx0, off, idx1 = cute.idx2crd(row, (vecsize, 8, cute.shape(tSrS, mode=[0, 1])))
        # register index: 0, 1, 0, 1, ..., 2, 3, 2, 3, ...
        return utils.shuffle_sync(tSrS[idx0 + idx1 * vecsize], offset=off * 4 + (lane % 4))

    @cute.jit
    def mma_one_m_block(
        self,
        m_block: Int32,
        consumer_state_Q: cutlass.pipeline.PipelineState | pipeline.PipelineStateSimple,
        consumer_state_dO: cutlass.pipeline.PipelineState | pipeline.PipelineStateSimple,
        warp_group_idx: Int32,
        mma_qk_fn: Callable,
        mma_dov_fn: Callable,
        mma_pdo_fn: Callable,
        mma_dsq_fn: Callable,
        mma_dsk_fn: Callable,
        copy_P_r2s: Optional[Callable],
        copy_dS_r2s: Callable,
        pipeline_Q: cutlass.pipeline.PipelineAsync,
        pipeline_dO: cutlass.pipeline.PipelineAsync,
        tLSEsLSE: cute.Tensor,
        tLSEsdPsum: cute.Tensor,
        tdQsdQaccum: Optional[cute.Tensor],
        softmax_scale_log2: Float32,
        PdS_barrier: cutlass.pipeline.NamedBarrier,
        is_dQ_wg: cutlass.Constexpr[bool] = True,
        mask_fn: Optional[Callable] = None,
        flashmask_fn: Optional[Callable] = None,
        dKV_accumulate: Boolean = True,
        partially_masked: Boolean = True,
        tdQgdQaccum: Optional[cute.Tensor] = None,
        mma_dsk_slice_fn: Optional[Callable] = None,
        dQ_acc_shape: cutlass.Constexpr = None,
    ):
        consumer_state_dO_cur = (
            consumer_state_Q if const_expr(self.Q_stage == self.dO_stage) else consumer_state_dO
        )
        smem_idx_Q = consumer_state_Q.index
        smem_idx_dO = consumer_state_dO_cur.index if const_expr(self.dO_stage > 1) else 0
        smem_idx_PdS = smem_idx_Q if const_expr(self.PdS_stage > 1) else 0
        # (1) [GEMM 1] S = Q @ K^T
        pipeline_Q.consumer_wait(consumer_state_Q, pipeline_Q.consumer_try_wait(consumer_state_Q))
        acc_S = mma_qk_fn(A_idx=smem_idx_Q, wg_wait=-1)
        # If shuffle_LSE, OOB reads are OK since sLSE is already padded
        tLSErLSE = copy_utils.load_s2r(tLSEsLSE[None, smem_idx_Q])
        # (2) [GEMM 2] dP = dO @ V.T
        pipeline_dO.consumer_wait(
            consumer_state_dO_cur, pipeline_dO.consumer_try_wait(consumer_state_dO_cur)
        )
        acc_dP = mma_dov_fn(A_idx=smem_idx_Q, wg_wait=1)

        # (3) [Pointwise 1] P = exp(S - LSE)
        if cutlass.const_expr(mask_fn is not None):
            mask_fn(acc_S, m_block=m_block)
        # flashmask (startend_row_indices): applied after causal/seqlen mask, on
        # this n_block's staged smem indices (fixed across the m loop). Only run
        # the per-element comparison on partially-masked blocks; fully-visible
        # blocks are a no-op (matches cpp bwd_step partially_masked gating).
        if cutlass.const_expr(flashmask_fn is not None):
            if partially_masked:
                flashmask_fn(acc_S, m_block=m_block)
        acc_S_mn = layout_utils.reshape_acc_to_mn(acc_S, transpose=self.SdP_swapAB)
        lane_idx = cute.arch.lane_idx()
        for r in cutlass.range_constexpr(cute.size(acc_S_mn, mode=[0])):
            lse_val = self._get_stat(tLSErLSE, r, lane_idx, shuffle=self.shuffle_LSE)
            for c in cutlass.range(cute.size(acc_S_mn, mode=[1]), unroll_full=True):
                acc_S_mn[r, c] = cute.math.exp2(
                    acc_S_mn[r, c] * softmax_scale_log2 - lse_val, fastmath=True
                )
        tLSErdPsum = copy_utils.load_s2r(tLSEsdPsum[None, smem_idx_dO])

        # Convert P from f32 -> f16
        tdVrP = utils.cvt_f16(layout_utils.reshape_acc_to_frgA(acc_S), self.dtype)
        # R2S for P
        if const_expr(not self.mma_dkv_is_rs):
            # sync to ensure P has already been used in the previous iteration before overwriting
            if const_expr(self.PdS_stage == 1):
                PdS_barrier.arrive_and_wait()
            copy_P_r2s(tdVrP, dst_idx=smem_idx_PdS)

        # (4) [Pointwise 2] dS = P*(dP-dPsum)
        warpgroup.wait_group(0)
        acc_dP_mn = layout_utils.reshape_acc_to_mn(acc_dP, transpose=self.SdP_swapAB)
        for r in cutlass.range_constexpr(cute.size(acc_dP_mn, mode=[0])):
            dpsum_val = self._get_stat(tLSErdPsum, r, lane_idx, shuffle=self.shuffle_dPsum)
            for c in cutlass.range(cute.size(acc_dP_mn, mode=[1]), unroll_full=True):
                acc_dP_mn[r, c] = acc_S_mn[r, c] * (acc_dP_mn[r, c] - dpsum_val)

        # Convert dS from f32 -> f16
        tdKrdS = utils.cvt_f16(layout_utils.reshape_acc_to_frgA(acc_dP), self.dtype)

        # If there's double buffering on dS, we don't need to sync here.
        # Otherwise we might have WG1 writing to dS before WG2 is done reading from it during MmadQ.
        # But because both WGs have to sync at the end of the loop and double buffering,
        # this race condition is not possible.
        # This sync is to ensure (1) P is written in case of !mma_dkv_is_rs and
        # (2) dS is already read by the Mma in the previous iteration in case of mma_dkv_is_rs.
        if const_expr(not self.mma_dkv_is_rs or (self.PdS_stage == 1 and self.mma_dkv_is_rs)):
            cute.arch.fence_view_async_shared()
            PdS_barrier.arrive_and_wait()

        # R2S for dS
        copy_dS_r2s(tdKrdS, dst_idx=smem_idx_PdS)

        if const_expr(self.slice_dQKV_mma):
            # Slice_dQKV_Mma (d256 atomicAdd path, mirrors cpp run_mha_bwd_hdim256):
            # split dV/dQ/dK into two M-halves and interleave so the dQ atomicAdd of
            # the first half overlaps the second half's MMA, and only half the dQ
            # accumulator is transiently consumed by the atomics at a time.
            # dV half0
            mma_pdo_fn(
                A_idx=smem_idx_PdS, B_idx=smem_idx_dO, zero_init=not dKV_accumulate,
                wg_wait=-1, m_slice=0,
            )
            cute.arch.fence_view_async_shared()
            PdS_barrier.arrive_and_wait()
            acc_dQ = cute.make_rmem_tensor(dQ_acc_shape, Float32)
            # dQ half0
            mma_dsk_slice_fn(acc_dQ, zero_init=True, A_idx=smem_idx_PdS, wg_wait=-1, m_slice=0)
            # dV half1 (wg_wait=1 -> dQ half0 complete)
            mma_pdo_fn(
                A_idx=smem_idx_PdS, B_idx=smem_idx_dO, zero_init=not dKV_accumulate,
                wg_wait=1, m_slice=1,
            )
            # atomicAdd dQ half0 (first n/2 flat elements == MMA_M slice 0)
            tdQg_m = tdQgdQaccum[None, None, None, m_block]
            n = cute.size(tdQg_m)
            half = n // 2
            assert n % 8 == 0, "dQ per-thread element count must be a multiple of 8 for sliced v4 atomics"
            tdQr = cute.make_tensor(acc_dQ.iterator, cute.make_layout(tdQg_m.shape))
            for i in cutlass.range_constexpr(0, half, 4):
                copy_utils.atomic_add_fp32x4(
                    tdQr[i], tdQr[i + 1], tdQr[i + 2], tdQr[i + 3],
                    utils.elem_pointer(tdQg_m, i),
                )
            # dK half0 (wg_wait=1)
            mma_dsq_fn(
                A_idx=smem_idx_PdS, B_idx=smem_idx_Q, zero_init=not dKV_accumulate,
                wg_wait=1, m_slice=0,
            )
            pipeline_dO.consumer_release(consumer_state_dO_cur)  # release dO
            # dQ half1 (wg_wait=0 -> dV half1 + dK half0 complete)
            mma_dsk_slice_fn(acc_dQ, zero_init=True, A_idx=smem_idx_PdS, wg_wait=0, m_slice=1)
            # atomicAdd dQ half1
            for i in cutlass.range_constexpr(half, n, 4):
                copy_utils.atomic_add_fp32x4(
                    tdQr[i], tdQr[i + 1], tdQr[i + 2], tdQr[i + 3],
                    utils.elem_pointer(tdQg_m, i),
                )
            # dK half1
            mma_dsq_fn(
                A_idx=smem_idx_PdS, B_idx=smem_idx_Q, zero_init=not dKV_accumulate,
                wg_wait=-1, m_slice=1,
            )
            warpgroup.wait_group(0)
            pipeline_Q.consumer_release(consumer_state_Q)
            consumer_state_Q.advance()
            consumer_state_dO.advance()
            return consumer_state_Q, consumer_state_dO

        # (5) [GEMM 3] dV += P.T @ dO
        if const_expr(not self.mma_dkv_is_rs):
            mma_pdo_fn(
                A_idx=smem_idx_PdS, B_idx=smem_idx_dO, zero_init=not dKV_accumulate, wg_wait=-1
            )
        else:
            mma_pdo_fn(tCrA=tdVrP, B_idx=smem_idx_dO, zero_init=not dKV_accumulate, wg_wait=-1)

        # smem fence to make sure sdS is written before it's read by WGMMA
        cute.arch.fence_view_async_shared()
        PdS_barrier.arrive_and_wait()

        if const_expr(is_dQ_wg):
            # (6) [GEMM 4] dQ = dS @ K
            acc_dQ = mma_dsk_fn(A_idx=smem_idx_PdS, wg_wait=1)
            pipeline_dO.consumer_release(consumer_state_dO_cur)  # release dO as dV mma is done

            # (7) [GEMM 5] dK += dS.T @ Q
            if const_expr(not self.mma_dkv_is_rs):
                mma_dsq_fn(
                    A_idx=smem_idx_PdS, B_idx=smem_idx_Q, zero_init=not dKV_accumulate, wg_wait=1
                )
            else:
                mma_dsq_fn(tCrA=tdKrdS, B_idx=smem_idx_Q, zero_init=not dKV_accumulate, wg_wait=1)

            if const_expr(self.dQacc_use_TMA):
                # dQ R2S: wait for dQaccum_store to free the smem buffer, then write dQ to smem
                # When dQ_single_wg, only WG0 enters here so warp_group_idx == 0
                cute.arch.barrier(
                    barrier_id=int(NamedBarrierBwd.dQEmptyWG0) + warp_group_idx,
                    number_of_threads=self.num_threads_per_warp_group + cute.arch.WARP_SIZE,
                )
                tdQrdQaccum_flat = cute.make_tensor(
                    acc_dQ.iterator, cute.make_layout(tdQsdQaccum.shape)
                )
                cute.autovec_copy(tdQrdQaccum_flat, tdQsdQaccum)
                cute.arch.fence_view_async_shared()
                cute.arch.barrier_arrive(
                    barrier_id=int(NamedBarrierBwd.dQFullWG0) + warp_group_idx,
                    number_of_threads=self.num_threads_per_warp_group + cute.arch.WARP_SIZE,
                )

                warpgroup.wait_group(0)
                pipeline_Q.consumer_release(consumer_state_Q)
            else:
                # atomicAdd path (head_dim=256): dK MMA above ran with wg_wait=1,
                # so the earlier dQ MMA is complete -> acc_dQ is ready. Reduce the
                # dQ registers straight into gmem with vectorized (v4.f32) atomics,
                # then wait for the dK MMA and release Q. Element i of the flat dQ
                # accumulator maps to gmem index i of tdQgdQaccum[..., m_block]
                # (same tiled copy that partitions the smem staging buffer).
                tdQg_m = tdQgdQaccum[None, None, None, m_block]
                n = cute.size(tdQg_m)
                assert n % 4 == 0, "dQ per-thread element count must be a multiple of 4 (v4.f32 atomics)"
                tdQr = cute.make_tensor(acc_dQ.iterator, cute.make_layout(tdQg_m.shape))
                for i in cutlass.range_constexpr(0, n, 4):
                    copy_utils.atomic_add_fp32x4(
                        tdQr[i],
                        tdQr[i + 1],
                        tdQr[i + 2],
                        tdQr[i + 3],
                        utils.elem_pointer(tdQg_m, i),
                    )
                warpgroup.wait_group(0)
                pipeline_Q.consumer_release(consumer_state_Q)
        else:
            # dQ_single_wg: WG1 skips dQ, only does dV wait + dK
            # (7) [GEMM 5] dK += dS.T @ Q
            if const_expr(not self.mma_dkv_is_rs):
                mma_dsq_fn(
                    A_idx=smem_idx_PdS, B_idx=smem_idx_Q, zero_init=not dKV_accumulate, wg_wait=1
                )
            else:
                mma_dsq_fn(tCrA=tdKrdS, B_idx=smem_idx_Q, zero_init=not dKV_accumulate, wg_wait=1)
            pipeline_dO.consumer_release(consumer_state_dO_cur)
            warpgroup.wait_group(0)
            pipeline_Q.consumer_release(consumer_state_Q)

        consumer_state_Q.advance()
        consumer_state_dO.advance()
        return consumer_state_Q, consumer_state_dO

    @cute.jit
    def epilogue_dKV(
        self,
        acc_dV: cute.Tensor,
        mdV: cute.Tensor,
        sV: cute.Tensor,
        acc_dK: cute.Tensor,
        mdK: cute.Tensor,
        sK: cute.Tensor,
        seqlen: SeqlenInfoQK,
        tma_atom_dK: cute.CopyAtom,
        tma_atom_dV: cute.CopyAtom,
        tiled_mma_dK: cute.TiledMma,
        tiled_mma_dV: cute.TiledMma,
        tidx: Int32,
        n_block: Int32,
        head_idx: Int32,
        batch_idx: Int32,
        qhead_per_kvhead_divmod: Optional[FastDivmodDivisor] = None,
        mdK_semaphore: Optional[cute.Tensor] = None,
        mdV_semaphore: Optional[cute.Tensor] = None,
    ):
        epi_barrier = cutlass.pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierBwd.Epilogue), num_threads=self.num_mma_threads
        )
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        if const_expr(self.qhead_per_kvhead == 1):
            mdK_cur = seqlen.offset_batch_K(mdK, batch_idx, dim=3)[
                None, None, head_idx
            ]
            mdV_cur = seqlen.offset_batch_K(mdV, batch_idx, dim=3)[
                None, None, head_idx
            ]
            gdK = cute.local_tile(mdK_cur, (self.tile_n, self.tile_hdim), (n_block, 0))
            gdV = cute.local_tile(mdV_cur, (self.tile_n, self.tile_hdimv), (n_block, 0))
            store_dK, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_dK, 0, cute.make_layout(1), sK, gdK, single_stage=True
            )
            store_dV, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_dV, 0, cute.make_layout(1), sV, gdV, single_stage=True
            )
            sdV = sV if const_expr(not self.dKV_swapAB) else layout_utils.transpose_view(sV)
            sdK = sK if const_expr(not self.dKV_swapAB) else layout_utils.transpose_view(sK)
            copy_dV_r2s, _, _ = copy_utils.get_smem_store_C(
                tiled_mma_dV,
                sdV,
                tidx,
                transpose=self.dKV_swapAB,
                position_independent=True,
            )
            copy_dK_r2s, _, _ = copy_utils.get_smem_store_C(
                tiled_mma_dK,
                sdK,
                tidx,
                transpose=self.dKV_swapAB,
                position_independent=True,
            )
            cute.arch.cp_async_bulk_wait_group(1, read=True)
            epi_barrier.arrive_and_wait()
            copy_dV_r2s(acc_dV, dst_idx=None)
            cute.arch.fence_view_async_shared()
            epi_barrier.arrive_and_wait()
            if warp_idx == 4:
                store_dV()
                cute.arch.cp_async_bulk_commit_group()
            cute.arch.cp_async_bulk_wait_group(1, read=True)
            epi_barrier.arrive_and_wait()
            copy_dK_r2s(acc_dK, dst_idx=None)
            cute.arch.fence_view_async_shared()
            epi_barrier.arrive_and_wait()
            if warp_idx == 4:
                store_dK()
                cute.arch.cp_async_bulk_commit_group()
        else:
            deterministic_KV = self.deterministic and self.qhead_per_kvhead > 1
            sdKaccum_shape0 = self.tile_n * self.tile_hdim // self.num_wg_mma
            sdVaccum_shape0 = self.tile_n * self.tile_hdimv // self.num_wg_mma
            sdKaccum_layout = cute.make_layout((sdKaccum_shape0, self.num_wg_mma))
            sdVaccum_layout = cute.make_layout((sdVaccum_shape0, self.num_wg_mma))
            head_idx_kv = head_idx // qhead_per_kvhead_divmod
            if const_expr(deterministic_KV):
                assert mdK_semaphore is not None
                assert mdV_semaphore is not None
                mdK_semaphore_cur = mdK_semaphore[n_block, None, head_idx_kv, batch_idx]
                mdV_semaphore_cur = mdV_semaphore[n_block, None, head_idx_kv, batch_idx]
                lock_value = head_idx % self.qhead_per_kvhead
            mdKaccum_cur = seqlen.offset_batch_K(
                mdK, batch_idx, dim=2, padded=True, multiple=self.tile_hdim
            )[None, head_idx_kv]
            mdVaccum_cur = seqlen.offset_batch_K(
                mdV, batch_idx, dim=2, padded=True, multiple=self.tile_hdimv
            )[None, head_idx_kv]
            gdKaccum_ = cute.local_tile(mdKaccum_cur, (self.tile_n * self.tile_hdim,), (n_block,))
            gdKaccum = cute.flat_divide(gdKaccum_, (sdKaccum_shape0,))
            gdVaccum_ = cute.local_tile(mdVaccum_cur, (self.tile_n * self.tile_hdimv,), (n_block,))
            gdVaccum = cute.flat_divide(gdVaccum_, (sdVaccum_shape0,))
            # dK-accum/dV-accum staging overlaps the (no-longer-needed) K/V
            # bf16 smem. sV comes immediately before sK in SharedStorageQKV
            # (both 1024B-aligned, and cosize(sV_layout) is itself a multiple
            # of 1024B so there's no gap between them), so the fp32 view
            # starting at sV's address spans contiguously into sK's bytes.
            # When tile_hdim == tile_hdimv this combined region exactly
            # matches the accum size; when it doesn't (e.g. 192/128),
            # _get_shared_storage_cls pads sK so the combined region is still
            # big enough for both dK-accum (tile_hdim-wide) and dV-accum
            # (tile_hdimv-wide) - these two overlap each other in turn.
            sVaccum_ptr = cute.recast_ptr(sV.iterator, dtype=Float32)
            sdKaccum = cute.make_tensor(sVaccum_ptr, sdKaccum_layout)
            sdVaccum = cute.make_tensor(sVaccum_ptr, sdVaccum_layout)
            tiled_copy_dKVaccum_r2s = cute.make_tiled_copy_tv(
                cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), Float32, num_bits_per_copy=128),
                cute.make_layout((self.num_threads_per_warp_group, self.num_wg_mma)),
                cute.make_layout(128 // Float32.width),
            )
            thr_copy_dKVaccum_r2s = tiled_copy_dKVaccum_r2s.get_slice(tidx)
            tdKsdKaccum = thr_copy_dKVaccum_r2s.partition_D(sdKaccum)
            tdVsdVaccum = thr_copy_dKVaccum_r2s.partition_D(sdVaccum)

            read_flag = const_expr(not deterministic_KV)
            cute.arch.cp_async_bulk_wait_group(0, read=read_flag)
            if const_expr(deterministic_KV):
                barrier.wait_eq(mdK_semaphore_cur.iterator, tidx, 0, lock_value)
            epi_barrier.arrive_and_wait()
            tdKrdKaccum_flat = cute.make_tensor(acc_dK.iterator, tdKsdKaccum.shape)
            cute.autovec_copy(tdKrdKaccum_flat, tdKsdKaccum)
            cute.arch.fence_view_async_shared()
            epi_barrier.arrive_and_wait()
            if warp_idx == 4:
                with cute.arch.elect_one():
                    for wg_idx in cutlass.range_constexpr(self.num_wg_mma):
                        copy_utils.cpasync_reduce_bulk_add_f32(
                            sdKaccum[None, wg_idx].iterator,
                            gdKaccum[None, wg_idx].iterator,
                            self.tma_copy_bytes["dKacc"] // self.num_wg_mma,
                        )
                cute.arch.cp_async_bulk_commit_group()

            cute.arch.cp_async_bulk_wait_group(0, read=read_flag)
            if const_expr(deterministic_KV):
                barrier.arrive_inc(mdK_semaphore_cur.iterator, tidx, 0, 1)
                barrier.wait_eq(mdV_semaphore_cur.iterator, tidx, 0, lock_value)
            epi_barrier.arrive_and_wait()
            tdVrdVaccum_flat = cute.make_tensor(acc_dV.iterator, tdVsdVaccum.shape)
            cute.autovec_copy(tdVrdVaccum_flat, tdVsdVaccum)
            cute.arch.fence_view_async_shared()
            epi_barrier.arrive_and_wait()
            if warp_idx == 4:
                with cute.arch.elect_one():
                    for wg_idx in cutlass.range_constexpr(self.num_wg_mma):
                        copy_utils.cpasync_reduce_bulk_add_f32(
                            sdVaccum[None, wg_idx].iterator,
                            gdVaccum[None, wg_idx].iterator,
                            self.tma_copy_bytes["dVacc"] // self.num_wg_mma,
                        )
                cute.arch.cp_async_bulk_commit_group()
            if const_expr(deterministic_KV):
                cute.arch.cp_async_bulk_wait_group(0, read=read_flag)
                barrier.arrive_inc(mdV_semaphore_cur.iterator, tidx, 0, 1)

    @cute.jit
    def dQaccum_store(
        self,
        mdQaccum: cute.Tensor,
        sdQaccum: cute.Tensor,
        block_info: BlockInfo,
        TileSchedulerCls: cutlass.Constexpr[Callable],
        SeqlenInfoCls: cutlass.Constexpr[Callable],
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        mdQ_semaphore: Optional[cute.Tensor] = None,
        flashmask_info: Optional[FlashMaskInfo] = None,
        num_heads: Int32 = None,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        # warp-local thread index (dQaccum_store runs on warp 1, global tidx 32-63)
        warp_local_tidx = tidx % cute.arch.WARP_SIZE
        read_flag = const_expr(not self.deterministic)

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            n_block, head_idx, batch_idx, _ = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            mdQaccum_cur = mdQaccum[None, head_idx, batch_idx]
            # ((M * K / num_wg_dQ, num_wg_dQ), num_m_blocks)
            gdQaccum = cute.local_tile(
                mdQaccum_cur,
                (
                    cute.make_layout(
                        (self.tile_m * self.tile_hdim // self.num_wg_dQ, self.num_wg_dQ)
                    ),
                ),
                (None,),
            )

            if const_expr(mdQ_semaphore is not None):
                # mdQ_semaphore is (num_m_blocks, cluster_size, num_head, batch) after transpose
                mdQ_semaphore_cur = mdQ_semaphore[None, None, head_idx, batch_idx]

            m_block_min, m_block_max = block_info.get_m_block_min_max(seqlen, n_block)
            if const_expr(self.enable_flashmask and not self.deterministic):
                # Must match the MMA warps' flashmask segments exactly so the
                # per-m_block dQ producer/consumer barrier handshake stays in sync.
                flashmask_segments = self.flashmask_m_block_segments(
                    flashmask_info,
                    batch_idx,
                    head_idx,
                    n_block,
                    num_heads,
                    m_block_min,
                    m_block_max,
                )
                total_block_cnt = Int32(0)
                for seg_lo, seg_hi in flashmask_segments:
                    total_block_cnt = total_block_cnt + (seg_hi - seg_lo)
                process_tile = total_block_cnt > Int32(0)
            elif const_expr(not self.use_block_sparsity):
                process_tile = (
                    const_expr(not self.is_local)
                    or m_block_min < m_block_max
                )
                loop_count = m_block_max - m_block_min
            else:
                total_block_cnt = get_total_q_block_count_bwd(
                    blocksparse_tensors,
                    batch_idx,
                    head_idx,
                    n_block,
                    subtile_factor=self.subtile_factor,
                    m_block_max=m_block_max,
                )
                process_tile = total_block_cnt > Int32(0)

            if process_tile:
                if const_expr(not self.use_block_sparsity):
                    if const_expr(self.enable_flashmask and not self.deterministic):
                        # Do the dQ handshake only for the non-masked segments.
                        # Must walk the identical segment list (same order) as the
                        # load/mma warps so the per-m_block dQEmpty/dQFull barrier
                        # handshake stays paired 1:1.
                        for seg_lo, seg_hi in flashmask_segments:
                            for m_block in cutlass.range(seg_lo, seg_hi, unroll=1):
                                m_block_safe = m_block
                                num_dQ_chunks = self.num_wg_dQ
                                for warp_group_idx in cutlass.range_constexpr(num_dQ_chunks):
                                    cute.arch.cp_async_bulk_wait_group(
                                        num_dQ_chunks - 1 - warp_group_idx, read=read_flag
                                    )
                                    cute.arch.barrier_arrive(
                                        barrier_id=int(NamedBarrierBwd.dQEmptyWG0)
                                        + warp_group_idx,
                                        number_of_threads=self.num_threads_per_warp_group
                                        + cute.arch.WARP_SIZE,
                                    )
                                for warp_group_idx in cutlass.range_constexpr(num_dQ_chunks):
                                    cute.arch.barrier(
                                        barrier_id=int(NamedBarrierBwd.dQFullWG0)
                                        + warp_group_idx,
                                        number_of_threads=self.num_threads_per_warp_group
                                        + cute.arch.WARP_SIZE,
                                    )
                                    with cute.arch.elect_one():
                                        copy_utils.cpasync_reduce_bulk_add_f32(
                                            sdQaccum[None, warp_group_idx].iterator,
                                            gdQaccum[(None, warp_group_idx), m_block_safe].iterator,
                                            self.tma_copy_bytes["dQ"],
                                        )
                                    cute.arch.cp_async_bulk_commit_group()
                    else:
                        for iter_idx in cutlass.range(loop_count, unroll=1):
                            m_block = m_block_min + iter_idx
                            m_block_safe = m_block

                            num_dQ_chunks = self.num_wg_dQ
                            for warp_group_idx in cutlass.range_constexpr(num_dQ_chunks):
                                if const_expr(not self.deterministic):
                                    # If deterministic, we already waited at the end of the prev iter
                                    cute.arch.cp_async_bulk_wait_group(
                                        num_dQ_chunks - 1 - warp_group_idx, read=read_flag
                                    )
                                cute.arch.barrier_arrive(
                                    barrier_id=int(NamedBarrierBwd.dQEmptyWG0) + warp_group_idx,
                                    number_of_threads=self.num_threads_per_warp_group
                                    + cute.arch.WARP_SIZE,
                                )

                            # Semaphore acquire: wait for prior n_blocks to finish writing this m_block
                            if const_expr(self.deterministic):
                                if const_expr(self.spt):
                                    _, n_block_max_for_m_block = block_info.get_n_block_min_max(
                                        seqlen, m_block_safe
                                    )
                                    lock_value = n_block_max_for_m_block - 1 - n_block
                                else:
                                    lock_value = n_block
                                barrier.wait_eq(
                                    mdQ_semaphore_cur[(m_block_safe, None)].iterator,
                                    warp_local_tidx,
                                    0,  # flag_offset
                                    lock_value,
                                )

                            for warp_group_idx in cutlass.range_constexpr(num_dQ_chunks):
                                cute.arch.barrier(
                                    barrier_id=int(NamedBarrierBwd.dQFullWG0) + warp_group_idx,
                                    number_of_threads=self.num_threads_per_warp_group
                                    + cute.arch.WARP_SIZE,
                                )
                                with cute.arch.elect_one():
                                    copy_utils.cpasync_reduce_bulk_add_f32(
                                        sdQaccum[None, warp_group_idx].iterator,
                                        gdQaccum[(None, warp_group_idx), m_block_safe].iterator,
                                        self.tma_copy_bytes["dQ"],
                                    )
                                cute.arch.cp_async_bulk_commit_group()

                            # Semaphore release: signal that this n_block is done with this m_block
                            if const_expr(self.deterministic):
                                cute.arch.cp_async_bulk_wait_group(0, read=read_flag)
                                barrier.arrive_inc(
                                    mdQ_semaphore_cur[(m_block_safe, None)].iterator,
                                    warp_local_tidx,
                                    0,  # flag_offset
                                    1,
                                )
                else:
                    assert not self.deterministic, (
                        "Deterministic not implemented for block-sparse backward"
                    )
                    dQaccum_store_block_sparse_bwd_sm90(
                        blocksparse_tensors,
                        batch_idx,
                        head_idx,
                        n_block,
                        sdQaccum,
                        gdQaccum,
                        subtile_factor=self.subtile_factor,
                        m_block_max=m_block_max,
                        num_dQ_warp_groups=self.num_wg_dQ,
                        num_threads_per_warp_group=self.num_threads_per_warp_group,
                        tma_copy_bytes_dQ=self.tma_copy_bytes["dQ"],
                    )

            # For local masking + deterministic (non-spt): signal remaining m_blocks
            # that this n_block won't visit, so they don't deadlock waiting.
            if const_expr(
                self.deterministic and not self.spt and block_info.window_size_left is not None
            ):
                m_block_global_max = cute.ceil_div(seqlen.seqlen_q, self.tile_m)
                for m_block in cutlass.range(m_block_max, m_block_global_max, unroll=1):
                    barrier.arrive_inc(
                        mdQ_semaphore_cur[(m_block, None)].iterator,
                        warp_local_tidx,
                        0,  # flag_offset
                        1,
                    )

            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

        if const_expr(not self.deterministic):
            cute.arch.cp_async_bulk_wait_group(0, read=True)
