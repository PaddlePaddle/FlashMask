# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 The FlashMask Authors. All rights reserved.

import math
import operator

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Float32

from . import utils


class FlashAttentionBackwardDsink:
    def __init__(self, num_threads: int = 128):
        assert (
            num_threads % cute.arch.WARP_SIZE == 0
        ), "num_threads must be a multiple of the warp size"
        self.num_threads = num_threads
        self.num_warps = num_threads // cute.arch.WARP_SIZE

    @cute.jit
    def __call__(
        self,
        mDpsum: cute.Tensor,
        mLseLog2: cute.Tensor,
        mSink: cute.Tensor,
        mDsink: cute.Tensor,
        stream: cuda.CUstream,
    ):
        if cutlass.const_expr(mDpsum.element_type not in [Float32]):
            raise TypeError("dpsum tensor must be Float32")
        if cutlass.const_expr(mLseLog2.element_type not in [Float32]):
            raise TypeError("lse_log2 tensor must be Float32")
        if cutlass.const_expr(mDsink.element_type not in [Float32]):
            raise TypeError("dsink output tensor must be Float32")
        if cutlass.const_expr(
            mSink.element_type not in [cutlass.Float16, cutlass.BFloat16, Float32]
        ):
            raise TypeError("sink tensor must be Float16, BFloat16 or Float32")

        num_head = mDpsum.shape[1]

        @cute.struct
        class SharedStorage:
            reduce_buf: cute.struct.Align[
                cute.struct.MemRange[Float32, self.num_warps], 128
            ]

        smem_size = SharedStorage.size_in_bytes()

        self.kernel(
            mDpsum,
            mLseLog2,
            mSink,
            mDsink,
            SharedStorage,
        ).launch(
            grid=(num_head, 1, 1),
            block=[self.num_threads, 1, 1],
            smem=smem_size,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mDpsum: cute.Tensor,
        mLseLog2: cute.Tensor,
        mSink: cute.Tensor,
        mDsink: cute.Tensor,
        SharedStorage: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        head_idx, _, _ = cute.arch.block_idx()

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        reduce_buf = storage.reduce_buf.get_tensor((self.num_warps,))

        batch = mDpsum.shape[0]
        seqlen_q_rounded = mDpsum.shape[2]
        total = batch * seqlen_q_rounded

        LOG2_E = math.log2(math.e)
        sink_log2 = Float32(mSink[head_idx]) * LOG2_E

        # Each thread accumulates a strided slice of the (batch, seqlen) elements.
        acc = Float32(0.0)
        num_iters = cute.ceil_div(total, self.num_threads)
        for it in cutlass.range(num_iters, unroll=1):
            idx = tidx + it * self.num_threads
            if idx < total:
                b = idx // seqlen_q_rounded
                s = idx % seqlen_q_rounded
                lse_log2 = mLseLog2[b, head_idx, s]
                dpsum = mDpsum[b, head_idx, s]
                acc += cute.arch.exp2(sink_log2 - lse_log2) * dpsum

        # Warp-level reduction, then a final reduction across warps through smem.
        acc = utils.warp_reduce(acc, operator.add)
        lane_id = tidx % cute.arch.WARP_SIZE
        warp_id = tidx // cute.arch.WARP_SIZE
        if lane_id == 0:
            reduce_buf[warp_id] = acc
        cute.arch.sync_threads()

        if tidx == 0:
            total_acc = Float32(0.0)
            for w in cutlass.range(self.num_warps, unroll_full=True):
                total_acc += reduce_buf[w]
            mDsink[head_idx] = -total_acc
