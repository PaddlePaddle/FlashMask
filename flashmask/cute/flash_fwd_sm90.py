# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# Copyright (c) 2026 The FlashMask Authors. Modifications for FlashMask.

from types import SimpleNamespace
from typing import Callable, Optional
from functools import partial

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass.utils import LayoutEnum
import cutlass.utils.hopper_helpers as sm90_utils_basic
from cutlass import pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.base_dsl.arch import Arch

from . import copy_utils
from . import layout_utils
from . import hopper_helpers as sm90_utils

from .cute_dsl_utils import assume_tensor_aligned
from . import utils
from .mask import AttentionMask
from .softmax import Softmax
from .seqlen_info import SeqlenInfoQK
from .block_info import BlockInfo
from .block_sparsity import BlockSparseTensors
from .block_sparse_utils import (
    produce_block_sparse_loads,
    consume_block_sparse_loads,
)
from . import pipeline as pipeline_custom
from .pack_gqa import PackGQA, pack_gqa_layout, make_packgqa_tiled_tma_atom
from .named_barrier import NamedBarrierFwd
from .tile_scheduler import (
    TileSchedulerArguments,
    SingleTileScheduler,
    SingleTileLPTScheduler,
    ParamsBase,
)

from .flash_fwd import FlashAttentionForwardBase
from .flashmask_utils import FlashMaskInfo


class FlashAttentionForwardSm90(FlashAttentionForwardBase):
    arch = 90
    def __init__(
        self,
        *args,
        intra_wg_overlap: bool = True,
        mma_pv_is_rs: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.intra_wg_overlap = intra_wg_overlap
        self.mma_pv_is_rs = mma_pv_is_rs
        self.buffer_align_bytes = 1024
        self.cluster_shape_mn = (1, 1)
        assert self.arch.is_family_of(Arch.sm_90a), "Only SM 9.x is supported"

    def _get_smem_layout_atom(self):
        sQ_layout_atom = warpgroup.make_smem_layout_atom(
            sm90_utils_basic.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, self.dtype, self.tile_hdim),
            self.dtype,
        )
        sK_layout_atom = sQ_layout_atom
        sV_layout_atom = warpgroup.make_smem_layout_atom(
            sm90_utils_basic.get_smem_layout_atom(
                LayoutEnum.ROW_MAJOR, self.dtype, self.tile_hdimv
            ),
            self.dtype,
        )
        sO_layout_atom = sV_layout_atom
        if not self.mma_pv_is_rs:
            sP_layout_atom = warpgroup.make_smem_layout_atom(
                sm90_utils_basic.get_smem_layout_atom(
                    LayoutEnum.ROW_MAJOR, self.dtype, self.tile_n
                ),
                self.dtype,
            )
        else:
            sP_layout_atom = None
        return sQ_layout_atom, sK_layout_atom, sV_layout_atom, sO_layout_atom, sP_layout_atom

    def _get_tiled_mma(self):
        tiled_mma_qk = sm90_utils_basic.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.K,
            Float32,
            atom_layout_mnk=(self.tile_m // 64, 1, 1),
            tiler_mn=(64, self.tile_n),
        )
        tiled_mma_pv = sm90_utils_basic.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.MN,
            Float32,
            atom_layout_mnk=(self.tile_m // 64, 1, 1),  # Might need (1, 2, 1) for hdim 512
            tiler_mn=(64, self.tile_hdimv),
            a_source=warpgroup.OperandSource.RMEM
            if self.mma_pv_is_rs
            else warpgroup.OperandSource.SMEM,
        )
        return tiled_mma_qk, tiled_mma_pv

    def _get_shared_storage_cls(self):
        sQ_struct, sK_struct, sV_struct = [
            cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(layout)], self.buffer_align_bytes
            ]
            for layout in (self.sQ_layout, self.sK_layout, self.sV_layout)
        ]
        cosize_sQV = max(cute.cosize(self.sQ_layout), cute.cosize(self.sV_layout))
        sQV_struct = cute.struct.Align[cute.struct.MemRange[self.dtype, cosize_sQV], 1024]
        cosize_sP = cute.cosize(self.sP_layout) if const_expr(self.sP_layout is not None) else 0
        sP_struct = cute.struct.Align[cute.struct.MemRange[self.dtype, cosize_sP], 1024]
        # 1 stage * 2 for Q pipeline (full + empty), self.num_stages*2 for K, self.num_stages*2 for V,
        mbar_ptr_Q_struct = cute.struct.MemRange[cutlass.Int64, 1 * 2]
        mbar_ptr_K_struct = cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]
        mbar_ptr_V_struct = cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]
        # flashmask: full + empty mbarriers (one each per stage) and per-stage
        # staging buffer for the 4 startend_row_indices vectors (LTS/LTE/UTS/UTE).
        rowidx_mbar_count = (2 * self.num_stages) if const_expr(self.enable_flashmask) else 0
        rowidx_smem_count = (
            (4 * self.tile_n * self.num_stages) if const_expr(self.enable_flashmask) else 0
        )
        mbar_ptr_rowidx_struct = cute.struct.MemRange[cutlass.Int64, rowidx_mbar_count]
        s_rowidx_struct = cute.struct.MemRange[Int32, rowidx_smem_count]

        @cute.struct
        class SharedStorageQKV:
            mbar_ptr_Q: mbar_ptr_Q_struct
            mbar_ptr_K: mbar_ptr_K_struct
            mbar_ptr_V: mbar_ptr_V_struct
            mbar_ptr_rowidx: mbar_ptr_rowidx_struct
            s_rowidx: s_rowidx_struct
            sV: sV_struct
            sQ: sQ_struct
            sK: sK_struct
            sP: sP_struct

        @cute.struct
        class SharedStorageSharedQV:
            mbar_ptr_Q: mbar_ptr_Q_struct
            mbar_ptr_K: mbar_ptr_K_struct
            mbar_ptr_V: mbar_ptr_V_struct
            mbar_ptr_rowidx: mbar_ptr_rowidx_struct
            s_rowidx: s_rowidx_struct
            sQ: sQV_struct
            sK: sK_struct
            sP: sP_struct

        return SharedStorageQKV if const_expr(not self.Q_in_regs) else SharedStorageSharedQV

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
        stream: cuda.CUstream = None,
    ):
        """Configures and launches the flash attention kernel.

        mQ/mK/mV/mO has same data types(supports fp16 and bf16) and same layout:
        (batch_size, seqlen_q, num_head, head_dim):(_, _, _, 1)
        """
        self._check_type(
            *(
                t.element_type if t is not None else None
                for t in (mQ, mK, mV, mO, mLSE)
            )
        )

        mQ, mK, mV, mO = [assume_tensor_aligned(t) for t in (mQ, mK, mV, mO)]
        QO_layout_transpose = [1, 3, 2, 0]
        mQ, mO = [layout_utils.select(t, QO_layout_transpose) for t in (mQ, mO)]
        KV_layout_transpose = [1, 3, 2, 0]
        mK, mV = [layout_utils.select(t, KV_layout_transpose) for t in (mK, mV)]
        LSE_layout_transpose = [2, 1, 0]
        mLSE = (
            layout_utils.select(mLSE, LSE_layout_transpose)
            if const_expr(mLSE is not None)
            else None
        )

        tiled_mma_qk, tiled_mma_pv = self._get_tiled_mma()
        self.num_mma_threads = tiled_mma_qk.size
        self.num_threads_per_warp_group = 128
        self.num_wg_mma = self.num_mma_threads // self.num_threads_per_warp_group
        assert self.num_wg_mma in [1, 2, 3]
        self.num_threads = self.num_threads_per_warp_group * (self.num_wg_mma + 1)
        self.num_producer_threads = 32
        self.num_Q_load_threads = self.num_threads_per_warp_group  # If not TMA_Q
        self.num_epilogue_threads = self.num_mma_threads
        self.num_mma_regs, self.num_producer_regs = {1: (256, 56), 2: (240, 24), 3: (160, 32)}[
            self.num_wg_mma
        ]
        self.use_block_sparsity = cutlass.const_expr(blocksparse_tensors is not None)

        # flashmask (startend_row_indices) support. When enabled we stage the
        # per-n_block row indices into smem and apply an extra mask on S.
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

        self.use_scheduler_barrier = (
            (self.num_wg_mma >= 2 and self.tile_hdim <= 128)
            if const_expr(self.intra_wg_overlap)
            else (self.num_wg_mma == 2)
        )
        self.use_tma_Q = self.arch >= Arch.sm_90 and not (
            self.pack_gqa and self.tile_m % self.qhead_per_kvhead != 0
        )
        self.use_tma_O = self.use_tma_Q
        # Producer needs more registers when doing cp.async Q or KV loads
        if const_expr(self.num_wg_mma == 2 and not self.use_tma_Q):
            self.num_mma_regs, self.num_producer_regs = 224, 40
        self.rescale_O_before_gemm = self.tile_hdimv > 128 and self.intra_wg_overlap
        self._setup_attributes()
        # TODO: we prob don't need most of what's in _setup_attributes
        self.sQ_layout, self.sK_layout, self.sV_layout, self.sO_layout = [
            sm90_utils.make_smem_layout(mX.element_type, LayoutEnum.ROW_MAJOR, shape, stage)
            for mX, shape, stage in [
                (mQ, (self.tile_m, self.tile_hdim), None),
                (mK, (self.tile_n, self.tile_hdim), self.num_stages),
                (mV, (self.tile_n, self.tile_hdimv), self.num_stages),
                (mO, (self.tile_m, self.tile_hdimv), None),
            ]
        ]
        self.sP_layout = None
        if const_expr(not self.mma_pv_is_rs):
            self.sP_layout = sm90_utils.make_smem_layout(
                mV.element_type, LayoutEnum.ROW_MAJOR, (self.tile_m, self.tile_n)
            )

        SharedStorage = self._get_shared_storage_cls()

        mQ_og, mO_og = mQ, mO
        if const_expr(self.pack_gqa):
            nheads_kv = mK.shape[2]
            mQ = pack_gqa_layout(mQ, self.qhead_per_kvhead, nheads_kv, head_idx=2)
            mO = pack_gqa_layout(mO, self.qhead_per_kvhead, nheads_kv, head_idx=2)
            if const_expr(mLSE is not None):
                mLSE = pack_gqa_layout(mLSE, self.qhead_per_kvhead, nheads_kv, head_idx=1)

        # TMA
        gmem_tiled_copy_Q = cpasync.CopyBulkTensorTileG2SOp()
        gmem_tiled_copy_KV = cpasync.CopyBulkTensorTileG2SOp()  # Might multicast
        gmem_tiled_copy_O = cpasync.CopyBulkTensorTileS2GOp()
        self.tma_copy_bytes = {
            name: cute.size_in_bytes(mX.element_type, cute.select(layout, mode=[0, 1]))
            for name, mX, layout in [
                ("Q", mQ, self.sQ_layout),
                ("K", mK, self.sK_layout),
                ("V", mV, self.sV_layout),
            ]
        }
        make_tiled_tma_atom_fn = (
            partial(make_packgqa_tiled_tma_atom, qhead_per_kvhead=self.qhead_per_kvhead, head_idx=2)
            if const_expr(self.pack_gqa)
            else cpasync.make_tiled_tma_atom
        )
        tma_atom_Q, tma_tensor_Q = None, None
        if const_expr(self.use_tma_Q):
            tma_atom_Q, tma_tensor_Q = make_tiled_tma_atom_fn(
                gmem_tiled_copy_Q,
                mQ_og if const_expr(self.pack_gqa) else mQ,
                self.sQ_layout,
                (self.tile_m, self.tile_hdim),  # No mcast
            )
        tma_atom_K, tma_tensor_K = cpasync.make_tiled_tma_atom(
            gmem_tiled_copy_KV,
            mK,
            cute.select(self.sK_layout, mode=[0, 1]),
            (self.tile_n, self.tile_hdim),
            1,  # No mcast for now
        )
        tma_atom_V, tma_tensor_V = cpasync.make_tiled_tma_atom(
            gmem_tiled_copy_KV,
            mV,
            cute.select(self.sV_layout, mode=[0, 1]),
            (self.tile_n, self.tile_hdimv),
            1,  # No mcast for now
        )
        tma_atom_O, tma_tensor_O = None, None
        if const_expr(self.use_tma_O):
            mO_tma = mO_og if const_expr(self.pack_gqa) else mO
            tma_atom_O, tma_tensor_O = make_tiled_tma_atom_fn(
                gmem_tiled_copy_O,
                mO_tma,
                self.sO_layout,
                (self.tile_m, self.tile_hdimv),  # No mcast
            )
        TileScheduler = (
            SingleTileScheduler
            if const_expr(not self.is_causal or self.is_local)
            else SingleTileLPTScheduler
        )
        tile_sched_args = TileSchedulerArguments(
            cute.ceil_div(cute.size(mQ.shape[0]), self.tile_m),
            cute.size(mQ.shape[2]),
            cute.size(mQ.shape[3]),
            1,  # num_splits
            cute.size(mK.shape[0]),
            mQ.shape[1],
            mV.shape[1],
            total_q=cute.size(mQ.shape[0]) * cute.size(mQ.shape[3]),
            tile_shape_mn=(self.tile_m, self.tile_n),
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
            element_size=self.dtype.width // 8,
            is_persistent=False,
            lpt=self.is_causal or self.is_local,
        )
        tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)
        softmax_scale_log2, softmax_scale = utils.compute_softmax_scale_log2(
            softmax_scale
        )
        window_size_left = Int32(window_size_left) if window_size_left is not None else None
        window_size_right = Int32(window_size_right) if window_size_right is not None else None

        self.kernel(
            tma_tensor_Q if const_expr(self.use_tma_Q) else mQ,
            tma_tensor_K,
            tma_tensor_V,
            tma_tensor_O if const_expr(self.use_tma_O) else mO,
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
            self.sQ_layout,
            self.sK_layout,
            self.sV_layout,
            self.sO_layout,
            self.sP_layout,
            self.gmem_tiled_copy_Q,
            self.gmem_tiled_copy_K,
            self.gmem_tiled_copy_V,
            self.gmem_tiled_copy_O,
            tiled_mma_qk,
            tiled_mma_pv,
            tile_sched_params,
            TileScheduler,
            SharedStorage,
            flashmask_info,
        ).launch(
            grid=grid_dim,
            block=[self.num_threads, 1, 1],
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        tma_atom_Q: Optional[cute.CopyAtom],
        tma_atom_K: Optional[cute.CopyAtom],
        tma_atom_V: Optional[cute.CopyAtom],
        tma_atom_O: Optional[cute.CopyAtom],
        softmax_scale_log2: Float32,
        softmax_scale: Optional[Float32],
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
        learnable_sink: Optional[cute.Tensor],
        blocksparse_tensors: Optional[BlockSparseTensors],
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sO_layout: cute.ComposedLayout,
        sP_layout: cute.ComposedLayout | None,
        gmem_tiled_copy_Q: cute.TiledCopy,
        gmem_tiled_copy_K: cute.TiledCopy,
        gmem_tiled_copy_V: cute.TiledCopy,
        gmem_tiled_copy_O: cute.TiledCopy,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tile_sched_params: ParamsBase,
        TileScheduler: cutlass.Constexpr[Callable],
        SharedStorage: cutlass.Constexpr[Callable],
        flashmask_info: Optional[FlashMaskInfo] = None,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        # Prefetch tma descriptor
        if warp_idx == 0:
            for tma_atom in (tma_atom_Q, tma_atom_K, tma_atom_V, tma_atom_O):
                if const_expr(tma_atom is not None):
                    cpasync.prefetch_descriptor(tma_atom)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        # Mbarrier / pipeline init
        mbar_ptr_Q = storage.mbar_ptr_Q.data_ptr()

        # flashmask: init the startend_row_indices staging mbarriers before any
        # pipeline is created so their init is ordered by the pipeline init fence.
        # Layout: [full[0..num_stages), empty[0..num_stages)].
        mbar_ptr_rowidx = None
        if const_expr(self.enable_flashmask):
            mbar_ptr_rowidx = storage.mbar_ptr_rowidx.data_ptr()
            if warp_idx == 0:
                for j in cutlass.range_constexpr(self.num_stages):
                    # full: written by the KV load warp (32 threads)
                    cute.arch.mbarrier_init(mbar_ptr_rowidx + j, cute.arch.WARP_SIZE)
                    # empty: released by all MMA/consumer threads
                    cute.arch.mbarrier_init(
                        mbar_ptr_rowidx + self.num_stages + j, self.num_mma_threads
                    )

        ThreadCooperativeGroup = partial(pipeline.CooperativeGroup, pipeline.Agent.Thread)
        tma_warp = ThreadCooperativeGroup(1)
        load_threads = ThreadCooperativeGroup(self.num_threads_per_warp_group)
        mma_warps = ThreadCooperativeGroup(self.num_mma_threads // cute.arch.WARP_SIZE)
        if const_expr(self.use_tma_Q):
            pipeline_q = pipeline_custom.PipelineTmaAsync.create(
                barrier_storage=mbar_ptr_Q,
                num_stages=1,
                producer_group=tma_warp,
                consumer_group=mma_warps,
                tx_count=self.tma_copy_bytes["Q"],
                defer_sync=True,
            )
        else:
            pipeline_q = pipeline_custom.PipelineCpAsync.create(
                barrier_storage=mbar_ptr_Q,
                num_stages=1,
                producer_group=load_threads,
                consumer_group=mma_warps,
                defer_sync=True,
                elect_one_release=True,
                syncwarp_before_release=False,
            )

        pipeline_k = pipeline_custom.PipelineTmaAsync.create(
            barrier_storage=storage.mbar_ptr_K.data_ptr(),
            num_stages=self.num_stages,
            producer_group=tma_warp,
            consumer_group=mma_warps,
            tx_count=self.tma_copy_bytes["K"],
            defer_sync=True,
        )
        pipeline_v = pipeline_custom.PipelineTmaAsync.create(
            barrier_storage=storage.mbar_ptr_V.data_ptr(),
            num_stages=self.num_stages,
            producer_group=tma_warp,
            consumer_group=mma_warps,
            tx_count=self.tma_copy_bytes["V"],
            defer_sync=True,
        )

        # Cluster arrive after barrier init
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        # ///////////////////////////////////////////////////////////////////////////////
        # Get shared memory buffer
        # ///////////////////////////////////////////////////////////////////////////////
        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        if const_expr(not self.Q_in_regs):
            sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
        else:
            sV = storage.sQ.get_tensor(
                sV_layout.outer, swizzle=sV_layout.inner, dtype=mV.element_type
            )
        # Transpose view of V to tensor with layout (head_dim_v, tile_n) for tiled mma
        sVt = layout_utils.transpose_view(sV)
        sP = None
        if const_expr(sP_layout is not None):
            sP = storage.sP.get_tensor(sP_layout.outer, swizzle=sP_layout.inner)
        # reuse sQ's data iterator
        sO = storage.sQ.get_tensor(sO_layout.outer, swizzle=sO_layout.inner, dtype=self.dtype)

        # flashmask staging buffer (flat): 4 vectors x tile_n x num_stages Int32.
        s_rowidx = None
        if const_expr(self.enable_flashmask):
            s_rowidx = storage.s_rowidx.get_tensor(
                cute.make_layout(4 * self.tile_n * self.num_stages)
            )

        block_info = BlockInfo(
            self.tile_m,
            self.tile_n,
            self.is_causal,
            self.is_local,
            False,  # is_split_kv
            window_size_left,
            window_size_right,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
        )
        SeqlenInfoCls = partial(
            SeqlenInfoQK.create,
            seqlen_q_static=mQ.shape[0] if const_expr(not self.pack_gqa) else mQ.shape[0][1],
            seqlen_k_static=mK.shape[0],
            mCuTotalMBlocks=(
                blocksparse_tensors.cu_total_m_blocks if blocksparse_tensors is not None else None
            ),
            mCuBlockIdxOffsets=(
                blocksparse_tensors.cu_block_idx_offsets if blocksparse_tensors is not None else None
            ),
            # Don't need to pass in tile_mn because we won't access offset_padded
        )
        AttentionMaskCls = partial(
            AttentionMask,
            self.tile_m,
            self.tile_n,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
        )
        TileSchedulerCls = partial(TileScheduler.create, tile_sched_params)

        # Cluster wait before starting
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        if warp_idx < 4:  # Producer
            cute.arch.setmaxregister_decrease(self.num_producer_regs)
            self.load(
                mQ,
                mK,
                mV,
                sQ,
                sK,
                sV,
                tma_atom_Q,
                tma_atom_K,
                tma_atom_V,
                pipeline_k,
                pipeline_v,
                pipeline_q,
                gmem_tiled_copy_Q,
                blocksparse_tensors,
                block_info,
                SeqlenInfoCls,
                TileSchedulerCls,
                flashmask_info,
                s_rowidx,
                mbar_ptr_rowidx,
            )

        else:  # Consumer
            cute.arch.setmaxregister_increase(self.num_mma_regs)
            # ///////////////////////////////////////////////////////////////////////////////
            # Tile MMA compute thread partitions and allocate accumulators
            # ///////////////////////////////////////////////////////////////////////////////
            tidx, _, _ = cute.arch.thread_idx()
            tidx = tidx - 128
            self.mma(
                tiled_mma_qk,
                tiled_mma_pv,
                mO,
                mLSE,
                sQ,
                sK,
                sVt,
                sP,
                sO,
                learnable_sink,
                pipeline_k,
                pipeline_v,
                pipeline_q,
                gmem_tiled_copy_O,
                tma_atom_O,
                tidx,
                softmax_scale_log2,
                softmax_scale,
                block_info,
                SeqlenInfoCls,
                AttentionMaskCls,
                TileSchedulerCls,
                blocksparse_tensors,
                flashmask_info,
                s_rowidx,
                mbar_ptr_rowidx,
            )

    @cute.jit
    def load(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        tma_atom_Q: Optional[cute.CopyAtom],
        tma_atom_K: Optional[cute.CopyAtom],
        tma_atom_V: Optional[cute.CopyAtom],
        pipeline_k: pipeline.PipelineAsync,
        pipeline_v: pipeline.PipelineAsync,
        pipeline_q: pipeline.PipelineAsync,
        gmem_tiled_copy_Q: cute.TiledCopy,
        blocksparse_tensors: Optional[BlockSparseTensors],
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        flashmask_info: Optional[FlashMaskInfo] = None,
        s_rowidx: Optional[cute.Tensor] = None,
        mbar_ptr_rowidx: Optional[cute.Pointer] = None,
    ):
        warp_idx_in_wg = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
        tidx, _, _ = cute.arch.thread_idx()

        # TMA: only warp 0 loads. cp_async: all warps load.
        # When not use_tma_Q, all 128 producer threads participate in Q loading.
        is_load_warp = warp_idx_in_wg == 0 or const_expr(not self.use_tma_Q)
        # KV loading restricted to warp 0 for TMA.
        is_kv_load_warp = warp_idx_in_wg == 0

        if is_load_warp:
            q_producer_phase = Int32(1)
            kv_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_stages
            )
            num_heads = cute.size(mQ.shape[2])
            tile_scheduler = TileSchedulerCls()
            work_tile = tile_scheduler.initial_work_tile_info()
            while work_tile.is_valid_tile:
                # if work_tile.is_valid_tile:
                m_block, head_idx, batch_idx, _ = work_tile.tile_idx
                seqlen = SeqlenInfoCls(batch_idx)
                mQ_cur = seqlen.offset_batch_Q(mQ, batch_idx, dim=3)[None, None, head_idx]
                head_idx_kv = (
                    head_idx // self.qhead_per_kvhead if const_expr(not self.pack_gqa) else head_idx
                )

                load_Q = None
                if const_expr(self.use_tma_Q):
                    gQ = cute.local_tile(mQ_cur, (self.tile_m, self.tile_hdim), (m_block, 0))
                    load_Q, _, _ = copy_utils.tma_get_copy_fn(
                        tma_atom_Q, 0, cute.make_layout(1), gQ, sQ, single_stage=True,
                    )

                # === TMA path ===
                mK_cur = seqlen.offset_batch_K(mK, batch_idx, dim=3)[
                    None, None, head_idx_kv
                ]
                mV_cur = seqlen.offset_batch_K(mV, batch_idx, dim=3)[
                    None, None, head_idx_kv
                ]
                gK = cute.local_tile(mK_cur, (self.tile_n, self.tile_hdim), (None, 0))
                gV = cute.local_tile(mV_cur, (self.tile_n, self.tile_hdimv), (None, 0))
                # TODO: mcast
                tma_load_K_fn, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_K, 0, cute.make_layout(1), gK, sK,
                )
                tma_load_K_fn = copy_utils.tma_producer_copy_fn(tma_load_K_fn, pipeline_k)
                tma_load_V_fn, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_V, 0, cute.make_layout(1), gV, sV,
                )
                tma_load_V_fn = copy_utils.tma_producer_copy_fn(tma_load_V_fn, pipeline_v)

                load_K = partial(
                    self.load_KV,
                    tma_load_K_fn,
                    pipeline_kv=pipeline_k,
                )
                load_V = partial(
                    self.load_KV,
                    tma_load_V_fn,
                    pipeline_kv=pipeline_v,
                )

                pack_gqa = None
                if const_expr(not self.use_tma_Q):
                    pack_gqa = PackGQA(
                        self.tile_m, self.tile_hdim, self.check_hdim_oob, self.qhead_per_kvhead
                    )

                if const_expr(not self.use_block_sparsity):
                    n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, m_block)
                    # flashmask: skip fully-masked KV blocks by tightening the
                    # [n_block_min, n_block_max) range from both ends. Producer and
                    # consumer run the identical scan so they stay in lockstep.
                    if const_expr(self.enable_flashmask):
                        n_block_min, n_block_max = self.flashmask_n_block_min_max(
                            flashmask_info,
                            batch_idx,
                            head_idx,
                            m_block,
                            seqlen.seqlen_q,
                            num_heads,
                            n_block_min,
                            n_block_max,
                        )
                    # TMA handles n_block=-1 gracefully (fills zeros).
                    n_block = n_block_max - 1

                    # First iteration: load K on pipeline_k, Q on pipeline_q
                    if is_kv_load_warp:
                        pipeline_k.producer_acquire(kv_producer_state)
                        load_K(block=n_block, producer_state=kv_producer_state)
                        # flashmask: stage the first block's startend_row_indices at the
                        # same pipeline stage as its K (before kv_producer_state advances).
                        if const_expr(self.enable_flashmask):
                            self.load_startend_row_indices(
                                batch_idx,
                                head_idx,
                                n_block,
                                num_heads,
                                kv_producer_state,
                                s_rowidx,
                                flashmask_info,
                                mbar_ptr_rowidx,
                                seqlen.seqlen_k,
                                m_block,
                                seqlen.seqlen_q,
                            )
                    if const_expr(self.use_tma_Q):
                        if warp_idx_in_wg == 0:
                            pipeline_q.producer_acquire_w_index_phase(0, q_producer_phase)
                            load_Q(tma_bar_ptr=pipeline_q.sync_object_full.get_barrier(0))
                            q_producer_phase ^= 1
                    else:
                        pipeline_q.producer_acquire_w_index_phase(0, q_producer_phase)
                        pack_gqa.load_Q(
                            mQ_cur, sQ, gmem_tiled_copy_Q, tidx, m_block, seqlen.seqlen_q
                        )
                        cute.arch.cp_async_commit_group()
                        pipeline_q.producer_commit_w_index(0)
                        q_producer_phase ^= 1

                    if is_kv_load_warp:
                        if const_expr(not self.intra_wg_overlap):
                            pipeline_v.producer_acquire(kv_producer_state)
                            load_V(
                                block=n_block, producer_state=kv_producer_state
                            )
                            kv_producer_state.advance()

                            for i in cutlass.range(n_block_max - 1 - n_block_min, unroll=1):
                                n_block = n_block_max - 1 - i - 1
                                process = True
                                if const_expr(self.enable_flashmask):
                                    # Skip fully-masked KV blocks (incl. mid-range);
                                    # the consumer applies the identical predicate.
                                    process = not self._flashmask_n_block_skip(
                                        flashmask_info,
                                        batch_idx,
                                        head_idx,
                                        m_block,
                                        n_block,
                                        seqlen.seqlen_q,
                                        num_heads,
                                    )
                                if process:
                                    kv_producer_state = self._produce_kv_block(
                                        kv_producer_state,
                                        n_block,
                                        pipeline_k,
                                        pipeline_v,
                                        load_K,
                                        load_V,
                                        batch_idx,
                                        head_idx,
                                        num_heads,
                                        s_rowidx,
                                        flashmask_info,
                                        mbar_ptr_rowidx,
                                        seqlen.seqlen_k,
                                        m_block,
                                        seqlen.seqlen_q,
                                    )
                        else:
                            for i in cutlass.range(n_block_max - 1 - n_block_min, unroll=1):
                                n_block_prev = n_block_max - i - 1
                                n_block = n_block_prev - 1
                                kv_producer_state_prev = kv_producer_state.clone()
                                kv_producer_state.advance()
                                pipeline_k.producer_acquire(kv_producer_state)
                                load_K(
                                    block=n_block,
                                    producer_state=kv_producer_state,
                                )
                                # flashmask: stage this block's indices at the same
                                # pipeline stage as its K load.
                                if const_expr(self.enable_flashmask):
                                    self.load_startend_row_indices(
                                        batch_idx,
                                        head_idx,
                                        n_block,
                                        num_heads,
                                        kv_producer_state,
                                        s_rowidx,
                                        flashmask_info,
                                        mbar_ptr_rowidx,
                                        seqlen.seqlen_k,
                                        m_block,
                                        seqlen.seqlen_q,
                                    )
                                pipeline_v.producer_acquire(kv_producer_state_prev)
                                load_V(
                                    block=n_block_prev,
                                    producer_state=kv_producer_state_prev,
                                )
                            n_block = n_block_min
                            pipeline_v.producer_acquire(kv_producer_state)
                            load_V(
                                block=n_block, producer_state=kv_producer_state
                            )
                            kv_producer_state.advance()
                else:
                    # Block sparsity: use TMA closures directly
                    # Load Q on pipeline_q, separate from K/V pipeline
                    if const_expr(self.use_tma_Q):
                        if warp_idx_in_wg == 0:
                            pipeline_q.producer_acquire_w_index_phase(0, q_producer_phase)
                            load_Q(tma_bar_ptr=pipeline_q.sync_object_full.get_barrier(0))
                            q_producer_phase ^= 1
                    else:
                        pipeline_q.producer_acquire_w_index_phase(0, q_producer_phase)
                        pack_gqa.load_Q(
                            mQ_cur, sQ, gmem_tiled_copy_Q, tidx, m_block, seqlen.seqlen_q
                        )
                        cute.arch.cp_async_commit_group()
                        pipeline_q.producer_commit_w_index(0)
                        q_producer_phase ^= 1
                    if is_kv_load_warp:
                        if const_expr(self.enable_flashmask):
                            kv_producer_state = self._produce_flashmask_bs(
                                blocksparse_tensors,
                                batch_idx,
                                head_idx,
                                m_block,
                                kv_producer_state,
                                tma_load_K_fn,
                                tma_load_V_fn,
                                pipeline_k,
                                pipeline_v,
                                num_heads,
                                s_rowidx,
                                flashmask_info,
                                mbar_ptr_rowidx,
                                seqlen.seqlen_k,
                                seqlen.seqlen_q,
                            )
                        else:
                            kv_producer_state = produce_block_sparse_loads(
                                blocksparse_tensors,
                                batch_idx,
                                head_idx,
                                m_block,
                                kv_producer_state,
                                tma_load_K_fn,
                                tma_load_V_fn,
                                pipeline_k,
                                pipeline_v,
                                self.intra_wg_overlap,
                                self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
                                self.q_subtile_factor if self.q_subtile_factor is not None else 1,
                            )

                tile_scheduler.prefetch_next_work()
                tile_scheduler.advance_to_next_work()
                work_tile = tile_scheduler.get_current_work()
                # End of persistent scheduler loop

            # Producer tail is only useful for cluster to avoid early exit of blocks.
            # We only need producer_tail on V since that's the last that's loaded, we don't
            # need it for Q (no cluster) and K.
            if is_kv_load_warp:
                pipeline_v.producer_tail(kv_producer_state)

    @cute.jit
    def _produce_flashmask_bs(
        self,
        blocksparse_tensors: BlockSparseTensors,
        batch_idx: Int32,
        head_idx: Int32,
        m_block: Int32,
        kv_producer_state: pipeline.PipelineState,
        tma_load_K_fn: Callable,
        tma_load_V_fn: Callable,
        pipeline_k: pipeline.PipelineAsync,
        pipeline_v: pipeline.PipelineAsync,
        num_heads: Int32,
        s_rowidx: cute.Tensor,
        flashmask_info: FlashMaskInfo,
        mbar_ptr_rowidx: cute.Pointer,
        seqlen_k: Int32,
        seqlen_q: Int32,
    ):
        """flashmask block-sparse producer. Loads K/V for the surviving KV blocks
        in the same order the consumer (consume_block_sparse_loads) walks them:
        the partially-masked `mask` list first (processed high-n -> low-n), then the
        fully-visible `full` list. startend_row_indices are staged into smem only
        for the mask-list prefix (the full list needs no per-element mask), keyed to
        the KV pipeline stage so apply_flashmask_block reads them via the rowidx
        mbar. Because the partial blocks are a contiguous prefix, the rowidx pipeline
        stays in lockstep with the KV pipeline over that prefix, so kv_producer_state
        phase is a valid expected-phase for the rowidx mbar. Mirrors
        load_block_list + finish_overlap_v_load with the intra_wg_overlap staggering
        (self.load_startend_row_indices / self._bs_block_at are methods, not captured
        closures, so they are legal under this loop)."""
        (
            mask_block_cnt,
            mask_block_idx,
            full_block_cnt,
            full_block_idx,
            *_,
        ) = blocksparse_tensors
        mask_cnt = mask_block_cnt[batch_idx, head_idx, m_block]
        full_cnt = full_block_cnt[batch_idx, head_idx, m_block]
        mask_idx = mask_block_idx[batch_idx, head_idx, m_block, None]
        full_idx = full_block_idx[batch_idx, head_idx, m_block, None]
        total = mask_cnt + full_cnt
        if total > 0:
            if const_expr(self.intra_wg_overlap):
                n_block_prev = self._bs_block_at(mask_idx, full_idx, mask_cnt, full_cnt, Int32(0))
                pipeline_k.producer_acquire(kv_producer_state)
                tma_load_K_fn(src_idx=n_block_prev, producer_state=kv_producer_state)
                if mask_cnt > 0:
                    self.load_startend_row_indices(
                        batch_idx, head_idx, n_block_prev, num_heads, kv_producer_state,
                        s_rowidx, flashmask_info, mbar_ptr_rowidx, seqlen_k, m_block, seqlen_q,
                        is_partial=self._mask_partial_at(mask_idx, mask_cnt, Int32(0)),
                    )
                for p in cutlass.range(1, total, unroll=1):
                    n_block = self._bs_block_at(mask_idx, full_idx, mask_cnt, full_cnt, p)
                    n_block_prev = self._bs_block_at(
                        mask_idx, full_idx, mask_cnt, full_cnt, p - 1
                    )
                    kv_producer_state_prev = kv_producer_state.clone()
                    kv_producer_state.advance()
                    pipeline_k.producer_acquire(kv_producer_state)
                    tma_load_K_fn(src_idx=n_block, producer_state=kv_producer_state)
                    if p < mask_cnt:
                        self.load_startend_row_indices(
                            batch_idx, head_idx, n_block, num_heads, kv_producer_state,
                            s_rowidx, flashmask_info, mbar_ptr_rowidx, seqlen_k, m_block, seqlen_q,
                            is_partial=self._mask_partial_at(mask_idx, mask_cnt, p),
                        )
                    pipeline_v.producer_acquire(kv_producer_state_prev)
                    tma_load_V_fn(src_idx=n_block_prev, producer_state=kv_producer_state_prev)
                n_block_last = self._bs_block_at(
                    mask_idx, full_idx, mask_cnt, full_cnt, total - 1
                )
                pipeline_v.producer_acquire(kv_producer_state)
                tma_load_V_fn(src_idx=n_block_last, producer_state=kv_producer_state)
                kv_producer_state.advance()
            else:
                for p in cutlass.range(total, unroll=1):
                    n_block = self._bs_block_at(mask_idx, full_idx, mask_cnt, full_cnt, p)
                    pipeline_k.producer_acquire(kv_producer_state)
                    tma_load_K_fn(src_idx=n_block, producer_state=kv_producer_state)
                    if p < mask_cnt:
                        self.load_startend_row_indices(
                            batch_idx, head_idx, n_block, num_heads, kv_producer_state,
                            s_rowidx, flashmask_info, mbar_ptr_rowidx, seqlen_k, m_block, seqlen_q,
                            is_partial=self._mask_partial_at(mask_idx, mask_cnt, p),
                        )
                    pipeline_v.producer_acquire(kv_producer_state)
                    tma_load_V_fn(src_idx=n_block, producer_state=kv_producer_state)
                    kv_producer_state.advance()
        return kv_producer_state

    @cute.jit
    def _bs_block_at(
        self,
        mask_idx: cute.Tensor,
        full_idx: cute.Tensor,
        mask_cnt: Int32,
        full_cnt: Int32,
        p: Int32,
    ) -> Int32:
        """Return the KV block index at processing position `p` of the combined
        (mask ++ full) sequence: positions [0, mask_cnt) map to the mask list
        (reversed, high-n first), positions [mask_cnt, total) to the full list
        (reversed). Indices are clamped to >= 0 so the not-taken branch never reads
        a negative offset (its value is discarded)."""
        in_mask = p < mask_cnt
        mi = cutlass.max(mask_cnt - 1 - p, Int32(0))
        fi = cutlass.max(full_cnt - 1 - (p - mask_cnt), Int32(0))
        n = full_idx[fi]
        if in_mask:
            # mask-list entries carry the fm_partial flag in bit 30 (see
            # build_flashmask_block_lists_kernel); strip it for the KV block index.
            n = mask_idx[mi] & 0x3FFFFFFF
        return n

    @cute.jit
    def _mask_partial_at(
        self,
        mask_idx: cute.Tensor,
        mask_cnt: Int32,
        p: Int32,
    ) -> Int32:
        """Return the fm_partial flag (0/1) for mask-list processing position `p`
        (reversed, high-n first), decoded from bit 30 of the stored index. Lets the
        producer/consumer skip the rowidx copy / per-element apply for mask-list
        blocks that only need causal/seqlen masking (not flashmask) without the
        per-block gmem read that _flashmask_block_partial would do."""
        mi = cutlass.max(mask_cnt - 1 - p, Int32(0))
        return (mask_idx[mi] >> 30) & 1

    @cute.jit
    def _produce_kv_block(
        self,
        kv_producer_state: pipeline.PipelineState,
        n_block: Int32,
        pipeline_k: pipeline.PipelineAsync,
        pipeline_v: pipeline.PipelineAsync,
        load_K: Callable,
        load_V: Callable,
        batch_idx: Int32,
        head_idx: Int32,
        num_heads: Int32,
        s_rowidx: cute.Tensor,
        flashmask_info: FlashMaskInfo,
        mbar_ptr_rowidx: cute.Pointer,
        seqlen_k: Int32,
        m_block: Int32,
        seqlen_q: Int32,
    ):
        """Load K and V (and stage flashmask indices) for one KV block, then
        advance and return the pipeline state. Passed as an explicit-arg method
        (not a capturing closure) so it can be called under the dynamic flashmask
        skip branch; the caller reassigns kv_producer_state to keep the
        loop-carried state correct (mirrors load_block_list)."""
        pipeline_k.producer_acquire(kv_producer_state)
        load_K(block=n_block, producer_state=kv_producer_state)
        pipeline_v.producer_acquire(kv_producer_state)
        load_V(block=n_block, producer_state=kv_producer_state)
        if const_expr(self.enable_flashmask):
            self.load_startend_row_indices(
                batch_idx,
                head_idx,
                n_block,
                num_heads,
                kv_producer_state,
                s_rowidx,
                flashmask_info,
                mbar_ptr_rowidx,
                seqlen_k,
                m_block,
                seqlen_q,
            )
        kv_producer_state.advance()
        return kv_producer_state

    @cute.jit
    def load_KV(
        self,
        tma_load_fn: Optional[Callable],
        block: Int32,
        pipeline_kv: pipeline.PipelineAsync,
        producer_state: pipeline.PipelineState,
    ):
        tma_load_fn(src_idx=block, producer_state=producer_state)
        pipeline_kv.producer_commit(producer_state)

    @cute.jit
    def load_startend_row_indices(
        self,
        batch_idx: Int32,
        head_idx: Int32,
        n_block: Int32,
        num_heads: Int32,
        kv_producer_state: pipeline.PipelineState,
        s_rowidx: cute.Tensor,
        flashmask_info: FlashMaskInfo,
        mbar_ptr_rowidx: cute.Pointer,
        seqlen_k: Int32,
        m_block: Int32,
        seqlen_q: Int32,
        is_partial: Optional[Int32] = None,
    ) -> None:
        """Stage the per-n_block startend_row_indices vectors (LTS/LTE/UTS/UTE)
        into shared memory for the consumer's flashmask application. Runs on the
        KV load warp (32 threads). The staging buffer stage and pipeline phase are
        taken from the KV producer state (flashmask advances in lockstep with KV),
        so no separate pipeline state is needed. Must be called before the KV
        producer state is advanced for this block. smem layout per stage: 4
        vectors of tile_n Int32 at offsets [0, tile_n, 2*tile_n, 3*tile_n].

        The actual gmem->smem copy is skipped for fully-visible blocks (no element
        masked); the empty-wait / full-arrive stay unconditional so the staging
        pipeline remains 1:1 with the KV pipeline."""
        stage = kv_producer_state.index
        # Wait until the consumer has finished reading this stage's buffer.
        cute.arch.mbarrier_wait(
            mbar_ptr_rowidx + self.num_stages + stage, kv_producer_state.phase
        )
        fm_heads = flashmask_info.startend_row_indices.shape[1]
        h_h_flashmask_ratio = num_heads // fm_heads
        fm_head_idx = head_idx // h_h_flashmask_ratio
        # Prefer the precomputed fm_partial flag (decoded from the block-list index
        # by the caller); fall back to the per-block gmem predicate only when the
        # caller doesn't provide it (non-block-sparse path).
        if const_expr(is_partial is None):
            m_start = m_block * self.tile_m
            m_end = cutlass.min(m_start + self.tile_m, seqlen_q)
            partial = self._flashmask_block_partial(
                flashmask_info, batch_idx, fm_head_idx, n_block, m_start, m_end
            )
        else:
            partial = is_partial != 0
        if partial:
            base = stage * 4 * self.tile_n
            s_cur = cute.make_tensor(
                s_rowidx.iterator + base, cute.make_layout(4 * self.tile_n)
            )
            nb_mul_kBN = n_block * self.tile_n
            loop_ub = cutlass.min(self.tile_n, seqlen_k - nb_mul_kBN)
            idx = cute.arch.thread_idx()[0] % cute.arch.WARP_SIZE
            srow = flashmask_info.startend_row_indices
            if const_expr(self.has_ut_start):
                while idx < loop_ub:
                    s_cur[idx] = srow[batch_idx, fm_head_idx, nb_mul_kBN + idx, 0]
                    s_cur[self.tile_n + idx] = srow[batch_idx, fm_head_idx, nb_mul_kBN + idx, 1]
                    s_cur[self.tile_n * 2 + idx] = srow[batch_idx, fm_head_idx, nb_mul_kBN + idx, 2]
                    s_cur[self.tile_n * 3 + idx] = srow[batch_idx, fm_head_idx, nb_mul_kBN + idx, 3]
                    idx += cute.arch.WARP_SIZE
            elif const_expr(self.has_lt_end):
                while idx < loop_ub:
                    s_cur[idx] = srow[batch_idx, fm_head_idx, nb_mul_kBN + idx, 0]
                    s_cur[self.tile_n + idx] = srow[batch_idx, fm_head_idx, nb_mul_kBN + idx, 1]
                    idx += cute.arch.WARP_SIZE
            elif const_expr(self.has_ut_end):
                while idx < loop_ub:
                    s_cur[idx] = srow[batch_idx, fm_head_idx, nb_mul_kBN + idx, 0]
                    s_cur[self.tile_n * 3 + idx] = srow[batch_idx, fm_head_idx, nb_mul_kBN + idx, 1]
                    idx += cute.arch.WARP_SIZE
            else:
                while idx < loop_ub:
                    s_cur[idx] = srow[batch_idx, fm_head_idx, nb_mul_kBN + idx, 0]
                    idx += cute.arch.WARP_SIZE
        # Signal the consumer that this stage's indices are ready (or skipped).
        cute.arch.mbarrier_arrive(mbar_ptr_rowidx + stage)

    @cute.jit
    def mma(
        self,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sVt: cute.Tensor,
        sP: Optional[cute.Tensor],
        sO: cute.Tensor,
        learnable_sink: Optional[cute.Tensor],
        pipeline_k: pipeline.PipelineAsync,
        pipeline_v: pipeline.PipelineAsync,
        pipeline_q: pipeline.PipelineAsync,
        gmem_tiled_copy_O: cute.TiledCopy,
        tma_atom_O: Optional[cute.CopyAtom],
        tidx: Int32,
        softmax_scale_log2: Float32,
        softmax_scale: Optional[Float32],
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        AttentionMaskCls: Callable,
        TileSchedulerCls: Callable,
        blocksparse_tensors: Optional[BlockSparseTensors],
        flashmask_info: Optional[FlashMaskInfo] = None,
        s_rowidx: Optional[cute.Tensor] = None,
        mbar_ptr_rowidx: Optional[cute.Pointer] = None,
    ):
        warp_group_idx = cute.arch.make_warp_uniform(tidx // self.num_threads_per_warp_group)
        warp_group_thread_layout = cute.make_layout(
            self.num_wg_mma, stride=self.num_threads_per_warp_group
        )
        thr_mma_qk = tiled_mma_qk.get_slice(tidx)
        wg_mma_qk = tiled_mma_qk.get_slice(warp_group_thread_layout(warp_group_idx))
        wg_mma_pv = tiled_mma_pv.get_slice(warp_group_thread_layout(warp_group_idx))
        _, tSrQ, tSrK = sm90_utils.partition_fragment_ABC(
            wg_mma_qk, (self.tile_m, self.tile_n, self.tile_hdim), sQ, sK
        )
        mma_qk_fn = partial(
            sm90_utils.gemm_zero_init, tiled_mma_qk, (self.tile_m, self.tile_n), tSrQ, tSrK
        )
        acc_O, tOrP, tOrVt = sm90_utils.partition_fragment_ABC(
            wg_mma_pv, (self.tile_m, self.tile_hdimv, self.tile_n), sP, sVt
        )
        mma_pv_fn = partial(sm90_utils.gemm_w_idx, tiled_mma_pv, acc_O, tOrP, tOrVt)

        # ///////////////////////////////////////////////////////////////////////////////
        # Smem copy atom tiling
        # ///////////////////////////////////////////////////////////////////////////////
        smem_copy_atom_P = utils.get_smem_store_atom(
            self.arch.major * 10 + self.arch.minor, self.dtype
        )
        smem_thr_copy_P = cute.make_tiled_copy_C(smem_copy_atom_P, tiled_mma_qk).get_slice(tidx)
        tPsP = smem_thr_copy_P.partition_D(sP) if const_expr(sP is not None) else None
        smem_copy_params = SimpleNamespace(smem_thr_copy_P=smem_thr_copy_P, tPsP=tPsP)

        self.mma_init()

        q_consumer_phase = Int32(0)
        kv_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_stages
        )

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        softmax = Softmax.create(
            softmax_scale_log2,
            num_rows=acc_O.shape[0][0] * acc_O.shape[1],
            softmax_scale=softmax_scale,
        )

        # For RescaleOBeforeGemm: persistent scores_scale across iterations
        scores_scale = None
        if const_expr(self.rescale_O_before_gemm):
            scores_scale = cute.make_rmem_tensor_like(softmax.row_max, Float32)

        mma_one_n_block_all = partial(
            self.mma_one_n_block_intrawg_overlap
            if const_expr(self.intra_wg_overlap)
            else self.mma_one_n_block,
            mma_qk_fn=mma_qk_fn,
            pipeline_k=pipeline_k,
            pipeline_v=pipeline_v,
            acc_O=acc_O,
            tOrP=tOrP,
            smem_copy_params=smem_copy_params,
            check_inf=True,
            scores_scale=scores_scale,
        )

        process_first_half_block = partial(
            self.first_half_block_overlap,
            mma_qk_fn=mma_qk_fn,
            pipeline_k=pipeline_k,
            tOrP=tOrP,
            smem_copy_params=smem_copy_params,
            scores_scale=scores_scale,
            softmax=softmax,
            acc_O=acc_O,
        )
        process_last_half_block = partial(
            self.last_half_block_overlap,
            pipeline_v=pipeline_v,
            mma_pv_fn=mma_pv_fn,
            scores_scale=scores_scale,
            softmax=softmax,
            acc_O=acc_O,
        )
        while work_tile.is_valid_tile:
            # if work_tile.is_valid_tile:

            # shape: (atom_v_m * rest_m)
            m_block, head_idx, batch_idx, _ = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)

            mask = AttentionMaskCls(seqlen.seqlen_q, seqlen.seqlen_k)
            mask_fn = partial(
                mask.apply_mask,
                batch_idx=batch_idx,
                head_idx=head_idx,
                m_block=m_block,
                thr_mma=thr_mma_qk,
                mask_causal=self.is_causal,
                mask_local=self.is_local,
            )
            flashmask_fn = None
            if const_expr(self.enable_flashmask):
                flashmask_fn = partial(
                    self.apply_flashmask_block,
                    m_block=m_block,
                    thr_mma=thr_mma_qk,
                    mask=mask,
                    s_rowidx=s_rowidx,
                    mbar_ptr_rowidx=mbar_ptr_rowidx,
                    flashmask_info=flashmask_info,
                    batch_idx=batch_idx,
                    head_idx=head_idx,
                    num_heads=cute.size(mO.shape[2]),
                    seqlen_q=seqlen.seqlen_q,
                )
            mma_one_n_block = partial(
                mma_one_n_block_all,
                seqlen=seqlen,
                softmax=softmax,
                flashmask_fn=flashmask_fn,
            )
            # Fully-visible (full-list) blocks need no per-element flashmask apply
            # and have no rowidx staged by the producer, so use a flashmask-free
            # variant to keep the rowidx mbarrier balanced.
            mma_one_n_block_full = partial(
                mma_one_n_block_all,
                seqlen=seqlen,
                softmax=softmax,
                flashmask_fn=None,
            )
            n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, m_block)
            # flashmask: identical block-skipping scan as the producer, so the
            # consumer processes exactly the blocks that were loaded.
            if const_expr(self.enable_flashmask):
                n_block_min, n_block_max = self.flashmask_n_block_min_max(
                    flashmask_info,
                    batch_idx,
                    head_idx,
                    m_block,
                    seqlen.seqlen_q,
                    cute.size(mO.shape[2]),
                    n_block_min,
                    n_block_max,
                )
            pipeline_q.consumer_wait_w_index_phase(0, q_consumer_phase)
            # For performance reason, we separate out two kinds of iterations:
            # those that need masking on S, and those that don't.
            # We need masking on S for the very last block when K and V has length not multiple of tile_n.
            # We also need masking on S if it's causal, for the last several blocks.
            # softmax.reset()  # Don't need reset as we explicitly call softmax w is_first=True
            O_should_accumulate = False

            # ==========================================
            # MAINLOOP
            # ==========================================
            if const_expr(not self.use_block_sparsity):
                # ==========================================
                # No block-sparsity (original path)
                # ==========================================
                # First iteration with seqlen masking
                if const_expr(self.intra_wg_overlap):
                    kv_consumer_state = process_first_half_block(
                        n_block=n_block_max - 1,
                        seqlen=seqlen,
                        kv_consumer_state=kv_consumer_state,
                        mask_fn=mask_fn,
                        flashmask_fn=flashmask_fn,
                        is_first_block=True,
                    )
                else:
                    self.warp_scheduler_barrier_sync()
                    kv_consumer_state = mma_one_n_block(
                        kv_consumer_state,
                        n_block=n_block_max - 1,
                        seqlen=seqlen,
                        mma_pv_fn=partial(mma_pv_fn, zero_init=True),
                        is_first_n_block=True,
                        mask_fn=partial(mask_fn, mask_seqlen=True),
                    )
                    O_should_accumulate = True
                n_block_max -= 1
                # Next couple of iterations with causal masking
                if const_expr(self.is_causal or self.is_local):
                    n_block_min_causal_local_mask = block_info.get_n_block_min_causal_local_mask(
                        seqlen, m_block, n_block_min
                    )
                    for n_tile in cutlass.range(
                        n_block_max - n_block_min_causal_local_mask, unroll=1
                    ):
                        n_block = n_block_max - 1 - n_tile
                        process = True
                        if const_expr(self.enable_flashmask):
                            process = not self._flashmask_n_block_skip(
                                flashmask_info, batch_idx, head_idx, m_block,
                                n_block, seqlen.seqlen_q, cute.size(mO.shape[2]),
                            )
                        if process:
                            kv_consumer_state = mma_one_n_block(
                                kv_consumer_state,
                                n_block=n_block,
                                seqlen=seqlen,
                                mma_pv_fn=partial(mma_pv_fn, zero_init=not O_should_accumulate),
                                mask_fn=partial(mask_fn, mask_seqlen=False),
                            )
                            O_should_accumulate = True
                    n_block_max = cutlass.min(n_block_max, n_block_min_causal_local_mask)
                # The remaining iterations have no masking
                n_block_min_before_local_mask = block_info.get_n_block_min_before_local_mask(
                    seqlen, m_block, n_block_min
                )
                for n_tile in cutlass.range(n_block_max - n_block_min_before_local_mask, unroll=1):
                    n_block = n_block_max - 1 - n_tile
                    process = True
                    if const_expr(self.enable_flashmask):
                        process = not self._flashmask_n_block_skip(
                            flashmask_info, batch_idx, head_idx, m_block,
                            n_block, seqlen.seqlen_q, cute.size(mO.shape[2]),
                        )
                    if process:
                        # These blocks are strictly below the causal/local diagonal
                        # (all columns valid, no seqlen edge), so causal masking is a
                        # no-op here. Skip mask_fn entirely to avoid the redundant
                        # per-element compare (matches cpp's empty no_mask_fn).
                        no_mask_fn = None
                        kv_consumer_state = mma_one_n_block(
                            kv_consumer_state,
                            n_block=n_block,
                            seqlen=seqlen,
                            mma_pv_fn=partial(mma_pv_fn, zero_init=not O_should_accumulate),
                            mask_fn=no_mask_fn,
                        )
                        O_should_accumulate = True
                # Separate iterations with local masking on the left
                if const_expr(self.is_local and block_info.window_size_left is not None):
                    n_block_max = cutlass.min(n_block_max, n_block_min_before_local_mask)
                    for n_tile in cutlass.range(n_block_max - n_block_min, unroll=1):
                        n_block = n_block_max - 1 - n_tile
                        process = True
                        if const_expr(self.enable_flashmask):
                            process = not self._flashmask_n_block_skip(
                                flashmask_info, batch_idx, head_idx, m_block,
                                n_block, seqlen.seqlen_q, cute.size(mO.shape[2]),
                            )
                        if process:
                            kv_consumer_state = mma_one_n_block(
                                kv_consumer_state,
                                n_block=n_block,
                                seqlen=seqlen,
                                mma_pv_fn=partial(mma_pv_fn, zero_init=not O_should_accumulate),
                                mask_fn=partial(mask_fn, mask_seqlen=False),
                            )
                            O_should_accumulate = True
                # Release Q pipeline so the producer can load the next tile's Q
                pipeline_q.consumer_release_w_index(0)
                # Last "half" iteration
                if const_expr(self.intra_wg_overlap):
                    kv_consumer_state = process_last_half_block(
                        kv_consumer_state=kv_consumer_state,
                        zero_init=not O_should_accumulate,
                    )
                    O_should_accumulate = True
                else:
                    self.warp_scheduler_barrier_arrive()

            else:
                # ==========================================
                # Block sparsity
                # ==========================================
                kv_consumer_state, O_should_accumulate, processed_any = consume_block_sparse_loads(
                    blocksparse_tensors,
                    batch_idx,
                    head_idx,
                    m_block,
                    seqlen,
                    kv_consumer_state,
                    mma_pv_fn,
                    mma_one_n_block,
                    process_first_half_block,
                    process_last_half_block,
                    mask_fn,
                    O_should_accumulate,
                    self.intra_wg_overlap,
                    self.warp_scheduler_barrier_sync,
                    self.warp_scheduler_barrier_arrive,
                    self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
                    self.q_subtile_factor if self.q_subtile_factor is not None else 1,
                    flashmask_fn=flashmask_fn,
                    mma_one_n_block_full=mma_one_n_block_full,
                )

                # Release Q pipeline so the producer can load the next tile's Q
                pipeline_q.consumer_release_w_index(0)

                # Handle empty case (when no blocks to process)
                if not processed_any:
                    softmax.reset()
                    acc_O.fill(0.0)

            q_consumer_phase ^= 1

            sink_val = None
            if const_expr(learnable_sink is not None):
                if const_expr(not self.pack_gqa):
                    sink_val = Float32(learnable_sink[head_idx])
                else:  # Each thread might have a different sink value due to different q_head
                    sink_val = cute.make_rmem_tensor_like(softmax.row_max, Float32)
                    cS = cute.make_identity_tensor((self.tile_m, self.tile_n))
                    tScS_mn = layout_utils.reshape_acc_to_mn(thr_mma_qk.partition_C(cS))
                    for r in cutlass.range(cute.size(sink_val), unroll_full=True):
                        row = m_block * self.tile_m + tScS_mn[r][0]
                        q_head_idx = row % self.qhead_per_kvhead + head_idx * self.qhead_per_kvhead
                        sink_val[r] = Float32(learnable_sink[q_head_idx])

            # normalize acc_O by row_sum and calculate the lse
            row_scale = softmax.finalize(sink_val=sink_val)
            softmax.rescale_O(acc_O, row_scale)

            # ///////////////////////////////////////////////////////////////////////////////
            # Epilogue
            # ///////////////////////////////////////////////////////////////////////////////
            self.epilogue(
                acc_O,
                softmax.row_sum,
                mO,
                mLSE,
                sO,
                seqlen,
                gmem_tiled_copy_O,
                tma_atom_O,
                tiled_mma_pv,
                tidx,
                m_block,
                head_idx,
                batch_idx,
            )

            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    @cute.jit
    def first_half_block_overlap(
        self,
        n_block: Int32,
        mma_qk_fn: Callable,
        kv_consumer_state,
        pipeline_k,
        tOrP: cute.Tensor,
        smem_copy_params: SimpleNamespace,
        softmax: Softmax,
        seqlen: SeqlenInfoQK,
        scores_scale: Optional[cute.Tensor] = None,
        acc_O: Optional[cute.Tensor] = None,
        mask_fn: Callable = None,
        flashmask_fn: Optional[Callable] = None,
        flashmask_is_partial: Optional[Int32] = None,
        is_first_block: bool = False,
    ):
        """Processes the first half block when using intra-warpgroup-overlap"""

        pipeline_k.consumer_wait(kv_consumer_state, pipeline_k.consumer_try_wait(kv_consumer_state))
        acc_S = mma_qk_fn(B_idx=kv_consumer_state.index, wg_wait=0)
        pipeline_k.consumer_release(kv_consumer_state)

        # Apply mask; mask_seqlen always True for first block
        # Caveat: if full block further right than mask block, seqlen masking is redundant;
        # however, masking is being applied anyway, so essentially no perf hit
        mask_fn(acc_S, n_block=n_block, mask_seqlen=True)
        # flashmask: applied after causal/seqlen mask, on the first block's stage.
        if const_expr(flashmask_fn is not None):
            flashmask_fn(acc_S, kv_consumer_state, n_block, is_partial=flashmask_is_partial)

        row_scale = softmax.online_softmax(acc_S, is_first=is_first_block)

        tOrP_acc = layout_utils.reshape_acc_to_frgA(acc_S)
        tOrP_cur = (
            tOrP
            if const_expr(self.mma_pv_is_rs)
            else cute.make_rmem_tensor_like(tOrP_acc, self.dtype)
        )
        tOrP_cur.store(tOrP_acc.load().to(self.dtype))

        if const_expr(not self.mma_pv_is_rs):
            tPrP = smem_copy_params.smem_thr_copy_P.retile(tOrP_cur)
            cute.copy(smem_copy_params.smem_thr_copy_P, tPrP, smem_copy_params.tPsP)
            # Fence and barrier to make smem store visible to WGMMA
            cute.arch.fence_view_async_shared()
            cute.arch.sync_warp()

        # For RescaleOBeforeGemm: initialize acc_O
        if const_expr(self.rescale_O_before_gemm):
            acc_O.fill(0.0)
            scores_scale.store(row_scale.load())

        return kv_consumer_state

    @cute.jit
    def last_half_block_overlap(
        self,
        kv_consumer_state,
        pipeline_v,
        mma_pv_fn: Callable,
        zero_init: bool,
        scores_scale: Optional[cute.Tensor] = None,
        softmax: Optional[Softmax] = None,
        acc_O: Optional[cute.Tensor] = None,
    ):
        """Processes the final PV GEMM when using intra-warpgroup-overlap"""

        # For RescaleOBeforeGemm: rescale O before the final PV GEMM
        if const_expr(self.rescale_O_before_gemm):
            softmax.rescale_O(acc_O, scores_scale)

        pipeline_v.consumer_wait(kv_consumer_state, pipeline_v.consumer_try_wait(kv_consumer_state))
        mma_pv_fn(B_idx=kv_consumer_state.index, zero_init=zero_init, wg_wait=0)
        pipeline_v.consumer_release(kv_consumer_state)
        kv_consumer_state.advance()
        return kv_consumer_state

    @cute.jit
    def apply_flashmask_block(
        self,
        acc_S: cute.Tensor,
        smem_pipe_read: pipeline.PipelineState,
        n_block: Int32,
        m_block: Int32,
        thr_mma: cute.TiledMma,
        mask: AttentionMask,
        s_rowidx: cute.Tensor,
        mbar_ptr_rowidx: cute.Pointer,
        flashmask_info: FlashMaskInfo,
        batch_idx: Int32,
        head_idx: Int32,
        num_heads: Int32,
        seqlen_q: Int32,
        is_partial: Optional[Int32] = None,
    ):
        """Consumer-side flashmask application for one n_block. The staging buffer
        stage and pipeline phase come from the KV consumer state (flashmask is in
        lockstep with KV), so no separate pipeline state is needed. Must be called
        before the KV consumer state is advanced for this block.

        The per-element mask is only applied for blocks that actually overlap the
        masked region (partially masked); fully-visible blocks skip it. The
        full-wait / empty-arrive stay unconditional to keep the staging pipeline
        1:1 with KV. The producer computes the identical predicate and skips the
        gmem->smem copy for the same fully-visible blocks."""
        stage = smem_pipe_read.index
        cute.arch.mbarrier_wait(mbar_ptr_rowidx + stage, smem_pipe_read.phase)
        # Prefer the precomputed fm_partial flag (decoded from the block-list index
        # by the caller); fall back to the per-block gmem predicate only when the
        # caller doesn't provide it (non-block-sparse path).
        if const_expr(is_partial is None):
            fm_heads = flashmask_info.startend_row_indices.shape[1]
            fm_head_idx = head_idx // (num_heads // fm_heads)
            m_start = m_block * self.tile_m
            m_end = cutlass.min(m_start + self.tile_m, seqlen_q)
            partial = self._flashmask_block_partial(
                flashmask_info, batch_idx, fm_head_idx, n_block, m_start, m_end
            )
        else:
            partial = is_partial != 0
        if partial:
            stage_base = stage * 4 * self.tile_n
            s_cur = cute.make_tensor(
                s_rowidx.iterator + stage_base, cute.make_layout(4 * self.tile_n)
            )
            mask.apply_flashmask_sm90(
                acc_S,
                m_block=m_block,
                thr_mma=thr_mma,
                s_startend_row_indices=s_cur,
                has_lt_end=self.has_lt_end,
                has_ut_start=self.has_ut_start,
                has_ut_end=self.has_ut_end,
            )
        cute.arch.mbarrier_arrive(mbar_ptr_rowidx + self.num_stages + stage)

    @cute.jit
    def _flashmask_block_fully_masked(
        self,
        flashmask_info: FlashMaskInfo,
        batch_idx: Int32,
        fm_head_idx: Int32,
        n_block: Int32,
        m_start: Int32,
        m_end: Int32,
    ):
        """Return True if KV block `n_block` is fully masked out for query rows
        [m_start, m_end). Mirrors reduce_block_count_kernel's predicate."""
        lts_max = flashmask_info.LTS_nblock_max[batch_idx, fm_head_idx, n_block]
        if const_expr(self.has_ut_start):
            lte_min = flashmask_info.LTE_nblock_min[batch_idx, fm_head_idx, n_block]
            uts_max = flashmask_info.UTS_nblock_max[batch_idx, fm_head_idx, n_block]
            ute_min = flashmask_info.UTE_nblock_min[batch_idx, fm_head_idx, n_block]
            return ((m_start >= lts_max) & (m_end <= lte_min)) | (
                (m_start >= uts_max) & (m_end <= ute_min)
            )
        elif const_expr(self.has_lt_end):
            lte_min = flashmask_info.LTE_nblock_min[batch_idx, fm_head_idx, n_block]
            return (m_start >= lts_max) & (m_end <= lte_min)
        elif const_expr(self.has_ut_end):
            ute_min = flashmask_info.UTE_nblock_min[batch_idx, fm_head_idx, n_block]
            return (m_start >= lts_max) | (m_end <= ute_min)
        else:
            return m_start >= lts_max

    @cute.jit
    def _flashmask_block_partial(
        self,
        flashmask_info: FlashMaskInfo,
        batch_idx: Int32,
        fm_head_idx: Int32,
        n_block: Int32,
        m_start: Int32,
        m_end: Int32,
    ):
        """Return True if KV block `n_block` overlaps the flashmask masked region
        for query rows [m_start, m_end) (i.e. needs flashmask applied). Mirrors
        generate_n_block's `partially_masked` predicate. Note this is a superset of
        fully-masked, so gating flashmask application on it is correct; blocks for
        which this is False are fully visible and need no flashmask."""
        lts_min = flashmask_info.LTS_nblock_min[batch_idx, fm_head_idx, n_block]
        if const_expr(self.has_ut_start):
            lte_max = flashmask_info.LTE_nblock_max[batch_idx, fm_head_idx, n_block]
            uts_min = flashmask_info.UTS_nblock_min[batch_idx, fm_head_idx, n_block]
            ute_max = flashmask_info.UTE_nblock_max[batch_idx, fm_head_idx, n_block]
            return ((m_start < lte_max) & (m_end > lts_min)) | (
                (m_start < ute_max) & (m_end > uts_min)
            )
        elif const_expr(self.has_lt_end):
            lte_max = flashmask_info.LTE_nblock_max[batch_idx, fm_head_idx, n_block]
            return (m_start < lte_max) & (m_end > lts_min)
        elif const_expr(self.has_ut_end):
            ute_max = flashmask_info.UTE_nblock_max[batch_idx, fm_head_idx, n_block]
            return (m_end > lts_min) | (m_start < ute_max)
        else:
            return m_end > lts_min

    @cute.jit
    def flashmask_n_block_min_max(
        self,
        flashmask_info: FlashMaskInfo,
        batch_idx: Int32,
        head_idx: Int32,
        m_block: Int32,
        seqlen_q: Int32,
        num_heads: Int32,
        n_block_min: Int32,
        n_block_max: Int32,
    ):
        """Tighten [n_block_min, n_block_max) by skipping contiguous fully-masked
        blocks at both ends. Deterministic from the flashmask nblock max/min, so
        producer and consumer compute the same range and stay in lockstep."""
        fm_heads = flashmask_info.startend_row_indices.shape[1]
        h_h_flashmask_ratio = num_heads // fm_heads
        fm_head_idx = head_idx // h_h_flashmask_ratio
        m_start = m_block * self.tile_m
        m_end = cutlass.min(m_start + self.tile_m, seqlen_q)
        orig_nonempty = n_block_max > n_block_min
        lo = n_block_min
        hi = n_block_max
        # Trim fully-masked blocks from the high end.
        active = hi > lo
        while active:
            fully = self._flashmask_block_fully_masked(
                flashmask_info, batch_idx, fm_head_idx, hi - 1, m_start, m_end
            )
            hi = hi - 1 if fully else hi
            active = fully & (hi > lo)
        # Trim fully-masked blocks from the low end.
        active = lo < hi
        while active:
            fully = self._flashmask_block_fully_masked(
                flashmask_info, batch_idx, fm_head_idx, lo, m_start, m_end
            )
            lo = lo + 1 if fully else lo
            active = fully & (lo < hi)
        # If every block was masked but there was originally work, keep one block
        # so the mainloop structure (which assumes >= 1 block) still holds; a fully
        # masked block yields a zero row, matching the reference.
        all_masked = orig_nonempty & (lo >= hi)
        lo = n_block_min if all_masked else lo
        hi = (n_block_min + 1) if all_masked else hi
        return lo, hi

    @cute.jit
    def _flashmask_n_block_skip(
        self,
        flashmask_info: FlashMaskInfo,
        batch_idx: Int32,
        head_idx: Int32,
        m_block: Int32,
        n_block: Int32,
        seqlen_q: Int32,
        num_heads: Int32,
    ):
        """Return True if KV block `n_block` is fully masked for this query block
        `m_block` and should be skipped entirely (loads + compute). Enables
        arbitrary (mid-range) block skipping that the both-end trim
        (flashmask_n_block_min_max) cannot do for masks whose visible KV region
        is non-contiguous (e.g. share-question). Deterministic from gmem, so the
        producer and consumer skip the identical set of blocks and stay in
        lockstep."""
        fm_heads = flashmask_info.startend_row_indices.shape[1]
        fm_head_idx = head_idx // (num_heads // fm_heads)
        m_start = m_block * self.tile_m
        m_end = cutlass.min(m_start + self.tile_m, seqlen_q)
        return self._flashmask_block_fully_masked(
            flashmask_info, batch_idx, fm_head_idx, n_block, m_start, m_end
        )

    @cute.jit
    def mma_one_n_block(
        self,
        smem_pipe_read: pipeline.PipelineState | pipeline_custom.PipelineStateSimple,
        n_block: Int32,
        mma_qk_fn: Callable,
        mma_pv_fn: Callable,
        pipeline_k: pipeline.PipelineAsync,
        pipeline_v: pipeline.PipelineAsync,
        acc_O: cute.Tensor,
        tOrP: cute.Tensor,
        smem_copy_params: SimpleNamespace,
        softmax: Softmax,
        seqlen: SeqlenInfoQK,
        scores_scale: Optional[cute.Tensor] = None,  # not used
        mask_fn: Optional[Callable] = None,
        flashmask_fn: Optional[Callable] = None,
        flashmask_is_partial: Optional[Int32] = None,
        is_first_n_block: cutlass.Constexpr = False,
        check_inf: cutlass.Constexpr = True,
    ):
        pipeline_k.consumer_wait(smem_pipe_read, pipeline_k.consumer_try_wait(smem_pipe_read))
        # S = Q @ K.T
        acc_S = mma_qk_fn(B_idx=smem_pipe_read.index, wg_wait=-1)
        self.warp_scheduler_barrier_arrive()
        warpgroup.wait_group(0)
        pipeline_k.consumer_release(smem_pipe_read)

        # handle masking
        if const_expr(mask_fn is not None):
            mask_fn(acc_S=acc_S, n_block=n_block)
        # flashmask (startend_row_indices): applied on every block, after the
        # causal/seqlen mask, reading the producer-staged smem indices.
        if const_expr(flashmask_fn is not None):
            flashmask_fn(acc_S, smem_pipe_read, n_block, is_partial=flashmask_is_partial)

        row_scale = softmax.online_softmax(acc_S, is_first=is_first_n_block, check_inf=check_inf)
        tOrP_acc = layout_utils.reshape_acc_to_frgA(acc_S)
        tOrP_cur = (
            tOrP
            if const_expr(self.mma_pv_is_rs)
            else cute.make_rmem_tensor_like(tOrP_acc, self.dtype)
        )
        # tOrP.store(tOrP_acc.load().to(self.dtype))
        # the "to(self.dtype)" conversion fails to vectorize for block sizes other
        # than 128 x 128, i.e. it calls convert on 1 fp32 element at a time instead of
        # 2 elements. So we just call ptx directly.
        utils.cvt_f16(tOrP_acc, tOrP_cur)
        if const_expr(not self.mma_pv_is_rs):
            tPrP = smem_copy_params.smem_thr_copy_P.retile(tOrP_cur)
            cute.copy(smem_copy_params.smem_thr_copy_P, tPrP, smem_copy_params.tPsP)
        softmax.rescale_O(acc_O, row_scale)
        if const_expr(not self.mma_pv_is_rs):
            # Fence and barrier to make sure smem store is visible to WGMMA
            cute.arch.fence_view_async_shared()
            cute.arch.sync_warp()  # Only need syncwarp since each warp is using its own P values for MmaPV
        pipeline_v.consumer_wait(smem_pipe_read, pipeline_v.consumer_try_wait(smem_pipe_read))
        self.warp_scheduler_barrier_sync()
        # O += P @ V
        mma_pv_fn(B_idx=smem_pipe_read.index, wg_wait=0)
        pipeline_v.consumer_release(smem_pipe_read)
        smem_pipe_read.advance()
        return smem_pipe_read

    @cute.jit
    def mma_one_n_block_intrawg_overlap(
        self,
        smem_pipe_read: pipeline.PipelineState | pipeline_custom.PipelineStateSimple,
        n_block: Int32,
        mma_qk_fn: Callable,
        mma_pv_fn: Callable,
        pipeline_k: pipeline.PipelineAsync,
        pipeline_v: pipeline.PipelineAsync,
        acc_O: cute.Tensor,
        tOrP: cute.Tensor,
        smem_copy_params: SimpleNamespace,
        softmax: Softmax,
        seqlen: SeqlenInfoQK,
        scores_scale: Optional[cute.Tensor] = None,
        mask_fn: Optional[Callable] = None,
        flashmask_fn: Optional[Callable] = None,
        flashmask_is_partial: Optional[Int32] = None,
        check_inf: cutlass.Constexpr = True,
    ):
        smem_pipe_read_v = smem_pipe_read.clone()
        smem_pipe_read.advance()
        pipeline_k.consumer_wait(smem_pipe_read, pipeline_k.consumer_try_wait(smem_pipe_read))
        self.warp_scheduler_barrier_sync()
        # S = Q @ K.T
        acc_S = mma_qk_fn(B_idx=smem_pipe_read.index, wg_wait=-1)
        # RescaleOBeforeGemm: rescale O while QK GEMM is in flight, before PV GEMM
        if const_expr(self.rescale_O_before_gemm):
            softmax.rescale_O(acc_O, scores_scale)
        pipeline_v.consumer_wait(smem_pipe_read_v, pipeline_v.consumer_try_wait(smem_pipe_read_v))
        # O += P @ V
        mma_pv_fn(B_idx=smem_pipe_read_v.index, wg_wait=-1)
        self.warp_scheduler_barrier_arrive()
        warpgroup.wait_group(1)
        pipeline_k.consumer_release(smem_pipe_read)

        # handle masking
        if const_expr(mask_fn is not None):
            mask_fn(acc_S=acc_S, n_block=n_block)
        # flashmask: applied after causal/seqlen mask, on the current block's stage
        # (smem_pipe_read points at the current block's K/V stage here).
        if const_expr(flashmask_fn is not None):
            flashmask_fn(acc_S, smem_pipe_read, n_block, is_partial=flashmask_is_partial)

        row_scale = softmax.online_softmax(acc_S, check_inf=check_inf)
        warpgroup.wait_group(0)
        pipeline_v.consumer_release(smem_pipe_read_v)
        tOrP_acc = layout_utils.reshape_acc_to_frgA(acc_S)
        tOrP_cur = (
            tOrP
            if const_expr(self.mma_pv_is_rs)
            else cute.make_rmem_tensor_like(tOrP_acc, self.dtype)
        )
        # tOrP_cur.store(tOrP_acc.load().to(self.dtype))
        # the "to(self.dtype)" conversion fails to vectorize for block sizes other
        # than 128 x 128, i.e. it calls convert on 1 fp32 element at a time instead of
        # 2 elements. So we just call ptx directly.
        utils.cvt_f16(tOrP_acc, tOrP_cur)
        if const_expr(not self.mma_pv_is_rs):
            tPrP = smem_copy_params.smem_thr_copy_P.retile(tOrP_cur)
            cute.copy(smem_copy_params.smem_thr_copy_P, tPrP, smem_copy_params.tPsP)
        if const_expr(not self.rescale_O_before_gemm):
            softmax.rescale_O(acc_O, row_scale)
        if const_expr(self.rescale_O_before_gemm):
            scores_scale.store(row_scale.load())
        if const_expr(not self.mma_pv_is_rs):
            # Fence and barrier to make sure smem store is visible to WGMMA
            cute.arch.fence_view_async_shared()
            cute.arch.sync_warp()  # Only need syncwarp since each warp is using its own P values for MmaPV
        return smem_pipe_read

    @cute.jit
    def mma_init(self):
        warp_group_idx = utils.canonical_warp_group_idx(sync=False)
        if const_expr(self.use_scheduler_barrier):
            if warp_group_idx == 1:
                cute.arch.barrier_arrive(
                    barrier_id=int(NamedBarrierFwd.WarpSchedulerWG1),
                    number_of_threads=2 * self.num_threads_per_warp_group,
                )

    def warp_scheduler_barrier_sync(self):
        if const_expr(self.use_scheduler_barrier):
            cute.arch.barrier(
                barrier_id=int(NamedBarrierFwd.WarpSchedulerWG1)
                - 1
                + utils.canonical_warp_group_idx(sync=False),
                number_of_threads=2 * self.num_threads_per_warp_group,
            )

    def warp_scheduler_barrier_arrive(self):
        if const_expr(self.use_scheduler_barrier):
            assert self.num_wg_mma in [2, 3]
            cur_wg = utils.canonical_warp_group_idx(sync=False) - 1
            if const_expr(self.num_wg_mma == 2):
                next_wg = 1 - cur_wg
            else:
                t = cur_wg + 1
                next_wg = t % self.num_wg_mma
            cute.arch.barrier_arrive(
                barrier_id=int(NamedBarrierFwd.WarpSchedulerWG1) + next_wg,
                number_of_threads=2 * self.num_threads_per_warp_group,
            )
