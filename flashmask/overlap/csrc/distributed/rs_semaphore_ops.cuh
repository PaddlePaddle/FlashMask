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

// Raises the empty flag semaphores[self_rank] on each producer of this segment.
__global__ void FusedConsumerNotifyEmpty(
    int64_t* const __restrict__ semaphores,
    int remote_producer_end_rank,
    int nranks,
    int self_rank
) {
    // for example: rank 3 local consumer needs the data from [12, 15] (remote producer) for seg 1
    // remote_producer_end_rank will be 15 (computed by mod_nranks(3 - 4 * seg_idx) --> (-1 % 16) --> 15)
    int target_rank = remote_producer_end_rank - threadIdx.x;
    target_rank = target_rank >= 0 ? target_rank : target_rank + nranks;
    if (target_rank == self_rank) return;
    gin::remote_store(semaphores + self_rank, int64_t(1), target_rank);
    DEBUG_PRINT("Consumer %d notifies remote %d empty, end_rank: %d\n", self_rank, target_rank, remote_producer_end_rank);
}

// [local consumer (dk dv reducer and recv buffer)] sends out an empty
// notifcation for [remote producer (put kernel)] to fill the buffer
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

// rs_full counts one strong add per producer per BWD; rs_expected is its running total.
static __global__ void RsWaitFullKernel(
    const uint64_t* const __restrict__ rs_full_seg,
    uint64_t* const __restrict__ rs_expected_seg,
    uint64_t num_producers
) {
    const uint64_t want = *rs_expected_seg + num_producers;
    *rs_expected_seg = want;
    gin::signal_wait_exact(rs_full_seg, want, "rs_full");
}

/**
 * @brief Wait (on stream) until all K producers of this stage committed.
 *   Exactly one call per segment per BWD, matching the producers' one add each.
 * @param rs_full_seg SignalBuffer::rs_full(seg)
 * @param rs_expected_seg SignalBuffer::rs_expected(seg)
 * @param num_producers K, the number of remote producers of this stage
*/
void consumer_wait_full(
    const uint64_t* const __restrict__ rs_full_seg,
    uint64_t* const __restrict__ rs_expected_seg,
    int num_producers,
    cudaStream_t comm_stream
) {
    WARN_PRINT("Consumer wait full ...\n");
    RsWaitFullKernel<<<1, 1, 0, comm_stream>>>(rs_full_seg, rs_expected_seg, num_producers);
    WARN_PRINT_SYNC(comm_stream, "Consumer wait full succeeded.\n");
}

__device__ void producer_wait_empty(
    const int64_t* const __restrict__ semaphores,
    const int target_pe
) {
    WARN_PRINT("Producer block %d waits remote %d empty ...\n", blockIdx.x, target_pe);
    gin::wait_until(semaphores + target_pe, gin::Compare::NotEqual, 0, "rs_empty");
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
 * This kernel notifies each sender that the buffer slot is empty and ready to receive.
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