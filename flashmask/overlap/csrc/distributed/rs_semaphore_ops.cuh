// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 The FlashMask Authors. All rights reserved.

#pragma once
#include <cuda_runtime.h>
#include "debug_logger.cuh"
#include "nccl_gin_backend.cuh"
#include "hierarchical_rank_map.cuh"

namespace flashmask {
namespace sema {
namespace rs {

__global__ void FusedConsumerNotifyEmpty(
    int64_t* const __restrict__ semaphores,
    int remote_producer_end_rank,
    int nranks,
    int self_rank
) {
    // for example: rank 3 local consumer needs the data from [12, 15] (remote producer) for seg 1
    // remote_producer_end_rank will be 15 (computed by mod_nranks(3 - 4 * seg_idx) --> (-1 % 16) --> 15)
    if (threadIdx.x == 0) {
        semaphores[self_rank] = 0;
        DEBUG_PRINT("Consumer %d fused, resets self refcount, nranks: %d, end_rank: %d\n",
            self_rank, nranks, remote_producer_end_rank);
    }
    // the following fence makes sure semaphore setting is visible across all CP ranks 
    __threadfence();
    __syncthreads();
    int target_rank = remote_producer_end_rank - threadIdx.x;
    target_rank = target_rank >= 0 ? target_rank : target_rank + nranks;
    if (target_rank == self_rank) return;
    gin::remote_store(semaphores + self_rank, int64_t(1), target_rank);
    DEBUG_PRINT("Consumer %d notifies remote %d empty, end_rank: %d\n", self_rank, target_rank, remote_producer_end_rank);
}

// [local consumer (dk dv reducer and recv buffer)] sends out an empty
// notifcation for [remote producer (put kernel)] to fill the buffer
//
// semaphores[self_rank] is a per-stage refcount: it starts at 0, each producer
// adds +1, and the consumer waits for == K then CASes it back to 0.
void notify_consumer_empty(
    int64_t* const semaphores,
    int remote_producer_end_rank,
    int seg_size,
    int nranks,
    int self_rank,
    cudaStream_t comm_stream
) {
    FusedConsumerNotifyEmpty<<<1, seg_size, 0, comm_stream>>>(semaphores,
                    remote_producer_end_rank, nranks, self_rank);
}

static __global__ void SpinWaitAndReplaceKernel(int64_t* ptr, int64_t target) {
    if (threadIdx.x > 0) return;
    while (true) {
        int64_t val;
        asm volatile("ld.global.cg.s64 %0, [%1];" 
            : "=l"(val) 
            : "l"(ptr) 
            : "memory"
        );

        if (val == target) {
            if (atomicCAS(reinterpret_cast<unsigned long long int*>(ptr), 
                          static_cast<unsigned long long int>(target), 
                          0ULL) == static_cast<unsigned long long int>(target)) {
                break;
            }
        }
        __nanosleep(16); 
    }
}

/**
 * @brief Wait (on stream) until all K producers of this stage committed, then reset the refcount.
 * @param semaphores int64 semaphores in the registered NCCL window: size is total_n_pes
 * @param my_pe the id of semaphore to wait for
 * @param wait_value K, the number of remote producers of this stage
*/
void consumer_wait_full(
    int64_t* const __restrict__ semaphores,
    int my_pe,
    int wait_value,
    cudaStream_t comm_stream
) {
    WARN_PRINT("Consumer wait full ...\n");
    SpinWaitAndReplaceKernel<<<1, 1, 0, comm_stream>>>(
        semaphores + my_pe, wait_value
    );
    WARN_PRINT_SYNC(comm_stream, "Consumer wait full succeeded.\n");
}

__device__ void producer_wait_empty(
    const int64_t* const __restrict__ semaphores,
    const int target_pe
) {
    WARN_PRINT("Producer block %d waits remote %d empty ...\n", blockIdx.x, target_pe);
    gin::wait_until(const_cast<int64_t*>(semaphores) + target_pe, gin::Compare::NotEqual, 0);   // wait until not 0
    WARN_PRINT("Producer block %d waits remote %d empty succeeded.\n", blockIdx.x, target_pe);
}

// ---------------------------------------------------------------------------
// Hierarchical variants for RS-overlap
// ---------------------------------------------------------------------------

/**
 * Hierarchical FusedConsumerNotifyEmpty.
 *
 * In hierarchical mode, the producers for my_pe are NOT a contiguous rank range.
 * For segment [logical_pos_base, logical_pos_base+num_chunks), PE X sends chunk c
 * to my_pe iff X == hier_sender_at_pos(logical_pos_base+c, my_pe, ...).
 * This kernel resets semaphores[self_rank] (the per-stage refcount) to 0, then
 * notifies each sender that the buffer slot is empty and ready to receive.
 *
 * Launch: <<<1, num_chunks>>>
 */
__global__ void HierFusedConsumerNotifyEmpty(
    int64_t* const __restrict__ semaphores,
    int logical_pos_base,
    int num_chunks,
    int total_n_pes,
    int gpus_per_node,
    int self_rank
) {
    if (threadIdx.x == 0) {
        semaphores[self_rank] = 0;
        DEBUG_PRINT("HierConsumer %d resets self refcount\n", self_rank);
    }
    __threadfence();
    __syncthreads();

    const int logical_pos = logical_pos_base + threadIdx.x;
    const int sender = hier::hier_sender_at_pos(logical_pos, self_rank, total_n_pes, gpus_per_node);
    if (sender == self_rank) return;  // local chunk, no notification needed
    gin::remote_store(semaphores + self_rank, int64_t(1), sender);
    DEBUG_PRINT("HierConsumer %d notifies sender %d (logical_pos %d) empty\n", self_rank, sender, logical_pos);
}

/**
 * Host wrapper: launch HierFusedConsumerNotifyEmpty.
 */
void notify_consumer_empty_hier(
    int64_t* const semaphores,
    int segment_idx,
    int num_chunks,
    int total_n_pes,
    int gpus_per_node,
    int self_rank,
    cudaStream_t stream
) {
    const int logical_pos_base = segment_idx * num_chunks;
    HierFusedConsumerNotifyEmpty<<<1, num_chunks, 0, stream>>>(
        semaphores, logical_pos_base, num_chunks, total_n_pes, gpus_per_node, self_rank
    );
}

}   // namespace rs
}   // namespace sema
}   // namespace flashmask