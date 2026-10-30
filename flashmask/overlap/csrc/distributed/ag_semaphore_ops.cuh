// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 The FlashMask Authors. All rights reserved.

#pragma once
#include <cuda_runtime.h>
#include "hierarchical_rank_map.cuh"
#include "nccl_gin_backend.cuh"

namespace flashmask {
namespace sema {
namespace ag {

// sema_intra holds `batch_ready_words(num_batch)` int64 words per rank; word w carries the
// per-batch readiness bits of batches [64w, 64w+64). This lifts the old single-word cap of
// 64 effective batches (BHSD: B*H). Word 0 doubles as the refcount slot on the rank that
// owns or relays the data, exactly as before -- a slot is never both at once.
__host__ __device__ __forceinline__ int batch_ready_words(int num_batch) {
    return num_batch <= 0 ? 1 : ((num_batch + 63) >> 6);
}

/**
 * @brief Bits set in word `word_idx` once all num_batch batches are ready.
 */
__host__ __device__ __forceinline__ int64_t batch_ready_bits(int num_batch, int word_idx) {
    const int bits = num_batch - (word_idx << 6);
    if (bits >= 64) return static_cast<int64_t>(~0ULL);      // avoid UB: shift by 64
    return bits <= 0 ? 0 : static_cast<int64_t>((1ULL << bits) - 1);
}

/**
 * @brief int64 semaphore slots to allocate. Non-hierarchical indexes by rank, so one word each.
 */
__host__ __forceinline__ int semaphore_count(
    bool use_hierarchical, int num_nodes, int total_n_pes, int num_batch) {
    return use_hierarchical ? num_nodes + total_n_pes * batch_ready_words(num_batch)
                            : total_n_pes;
}

/**
 * @brief (Device Function) Wait until semaphores[target_pe] > 0 (data ready).
 */
__device__ __forceinline__ void wait_full(
    const int64_t* const __restrict__ semaphores,
    const int target_pe
) {
    gin::wait_until(const_cast<int64_t*>(semaphores) + target_pe, gin::Compare::GreaterThan, 0);
}


/**
 * @brief (Device Function) Wait until semaphores' batch_idx readiness bit is set for target_pe.
 *   Used in hierarchical Phase 2 to wait for per-batch relay data readiness.
 */
__device__ __forceinline__ void wait_full_one_batch(
    const int64_t* const __restrict__ semaphores,
    const int batch_idx,
    const int target_pe,
    const int num_batch
) {
    const int64_t batch_bit = static_cast<int64_t>(1ULL << (batch_idx & 63));
    const int64_t* const word =
        semaphores + target_pe * batch_ready_words(num_batch) + (batch_idx >> 6);
    int64_t current_val;
    do {
        asm volatile("ld.volatile.global.s64 %0, [%1];"
            : "=l"(current_val) : "l"(word) : "memory");
    } while (!(current_val & batch_bit));
}

/**
 * @brief (Device Function) Publish "batch_idx of target_pe's data is readable" to the other
 *   ranks of this node. The leading fence pairs with the release fence the producing CTAs
 *   issue before their relaxed counter increment, so their SR buffer stores are ordered
 *   before this announcement.
 */
__device__ __forceinline__ void notify_batch_ready_to_same_node(
    int64_t* const __restrict__ semaphores,
    const int target_pe,
    const int batch_idx,
    const int num_batch,
    const int my_pe,
    const int gpus_per_node
) {
    int64_t* const word =
        semaphores + target_pe * batch_ready_words(num_batch) + (batch_idx >> 6);
    const int64_t ready_bit = static_cast<int64_t>(1ULL << (batch_idx & 63));
    const int my_pe_node = my_pe % gpus_per_node;
    const int my_node_id = my_pe / gpus_per_node;
    __threadfence();
    for (int slot = 1; slot < gpus_per_node; slot++) {
        const int base = (my_pe_node + slot) % gpus_per_node;
        gin::remote_or(word, ready_bit, base + my_node_id * gpus_per_node);
    }
}

/**
 * @brief (Device Function) Wait until all batch bits are set for target_pe.
 *   Used in HierNotifyEmpty to drain per-batch relay signals before clearing.
 */
__device__ __forceinline__ void wait_full_all_batch(
    const int64_t* const __restrict__ semaphores,
    const int num_batch,
    const int target_pe
) {
    const int words = batch_ready_words(num_batch);
    const int64_t* const base = semaphores + target_pe * words;
    for (int w = 0; w < words; w++) {
        const int64_t expected = batch_ready_bits(num_batch, w);
        int64_t current_val;
        do {
            asm volatile("ld.volatile.global.s64 %0, [%1];"
                : "=l"(current_val) : "l"(base + w) : "memory");
        } while (current_val != expected);
    }
}

// Note(heqianyue): single node AMO can use int (4B) as semaphore types, but when in multi-node
// Semaphore updates use 64-bit operations to match the registered window layout.
// So we need to use int64_t semaphores. If we know for sure that our CP distributed overlap
// utilizes only 1 node, change the dtype of SR buffer, remote_get kernels and current file.

// num thread: total_pes
__global__ void NotifySemaphoreEmptyKernel(
    int64_t* const __restrict__ semaphores,
    const int my_pe
) {
    if (threadIdx.x != my_pe) {
        // the other PE will not notify us before we reset
        wait_full(semaphores, threadIdx.x);
        semaphores[threadIdx.x] = 0;
        // Note(heqianyue): bitwise op is generally safer than add, if we are using only 1 node
        // we can opt for the following atomic_and approach
        // clear bit representing the current PE on the all other target PE
        const int64_t clear_mask = -(static_cast<int64_t>(1) << my_pe);
        gin::remote_add(semaphores + threadIdx.x, clear_mask, threadIdx.x);
    }
}

// notify some of the remote kernels: local rank has finished 
// using the data of yours. Used in RS-overlap splitted AG
__global__ void NotifySegmentSemaphoreEmptyKernel(
    int64_t* const __restrict__ semaphores,
    const int my_pe,
    const int start_rank,
    const int total_pes
) {
    const int target_rank = (start_rank + threadIdx.x) % total_pes;
    if (target_rank != my_pe) {
        // the other PE will not notify us before we reset
        wait_full(semaphores, target_rank);
        semaphores[target_rank] = 0;
        const int64_t clear_mask = -(static_cast<int64_t>(1) << my_pe);
        gin::remote_add(semaphores + target_rank, clear_mask, target_rank);
    }
}

__global__ void SetFullKernel(
    int64_t* const __restrict__ semaphores,
    int64_t value,
    int self_rank
) {
    if (threadIdx.x == 0) {
        semaphores[self_rank] = value;
    }
    __threadfence();
    __syncthreads();
    if (threadIdx.x == self_rank) return;
    // set the semaphores[self_rank] = 1 for all remote ranks
    gin::remote_store(semaphores + self_rank, int64_t(1), threadIdx.x);
}

/**
 * @brief CPU wait until the semaphores[my_pe] reached 0
 * @param semaphores int64 semaphores in the registered NCCL window: size is total_n_pes
 * @param my_pe the id of semaphore to wait for
 * @param stream waiting stream. This API is therefore async on stream (if non-blocking)
*/
void wait_self_empty(
    int64_t* const __restrict__ semaphores,
    int my_pe,
    cudaStream_t stream
) {
    gin::wait_until_on_stream(
        semaphores + my_pe,
        gin::Compare::Equal,
        0,
        stream
    );
}

/**
 * @brief Tell all other PEs that the local PE has finished using their data
    so that the semaphore value on the specific PE is decreased by 1. 

    The behavior is simple: set all semaphores[i] except i = my_pe, to 0, locally.
    So that the next remote_get kernel on comm_stream will know that there is no
    data available (before we do copy on aux_stream). Also, decrease all semaphores[i]
    (i != my_pe) by 1, so other PEs will know that their local data has one few
    dependent PE. If 0 is reached, they can start clean up. 

 * @param semaphores int64 semaphores in the registered NCCL window: size is total_n_pes
 * @param my_pe except for semaphores[my_pe], for all other local semaphores: set zero
    , and for remote semaphores: decrease (data ref_cnt) by 1
 * @param stream waiting stream. This API is therefore async on stream (if non-blocking)
*/
void notify_all_empty(
    int64_t* const __restrict__ semaphore,
    int my_pe,
    int total_pes,
    cudaStream_t stream
) {
    NotifySemaphoreEmptyKernel<<<1, total_pes, 0, stream>>>(semaphore, my_pe);
}

void notify_segment_empty(
    int64_t* const __restrict__ semaphore,
    int my_pe,
    int start_rank,
    int chunk_per_seg,
    int total_pes,
    cudaStream_t stream
) {
    NotifySegmentSemaphoreEmptyKernel<<<1, chunk_per_seg, 0, stream>>>(semaphore, my_pe, start_rank, total_pes);
}

/**
 * Hierarchical notify_full: sets refcounts and broadcasts to congruence partners + same-node ranks.
 *
 * Dual-refcount protocol (local writes by thread 0, refcount lives in word 0 of the slot):
 *   sema_intra[my_pe] = producer refcount = (num_nodes-1) + (gpus_per_node-1)
 *   sema_intra[partner] = relay refcount = (gpus_per_node-1), for each congruence partner
 *     (partner's data will be relayed through us; same-node ranks will read and decrement)
 *
 * Remote signals (threads 1..N):
 *   sema_inter[my_node_id] = 1 on each congruence partner  (cross-node data ready)
 *   sema_intra[my_pe] = every batch bit set, on each same-node rank  (local chunk ready)
 *
 * Precondition: wait_self_empty_hier has ensured all local sema_intra entries being
 *   written here are already 0 (previous round fully consumed).
 */
__global__ void HierSetFullKernel(
    int64_t* const __restrict__ sema_inter,   // [num_nodes] cross-node flags
    int64_t* const __restrict__ sema_intra,   // [total_n_pes] refcount + intra-node signals
    int64_t refcount,
    int self_rank,
    int my_pe_node,
    int my_node_id,
    int num_nodes,
    int gpus_per_node,
    int num_batch
) {
    const int ready_words = batch_ready_words(num_batch);
    if (threadIdx.x == 0) {
        // Producer refcount: decremented by congruence partners + same-node ranks
        sema_intra[self_rank * ready_words] = refcount;
        // Relay refcounts: each congruence partner's data will be relayed through us,
        // and (gpus_per_node-1) same-node ranks will read the relay then decrement.
        for (int i = 1; i < num_nodes; i++) {
            int partner = my_pe_node + ((my_node_id + i) % num_nodes) * gpus_per_node;
            sema_intra[partner * ready_words] = static_cast<int64_t>(gpus_per_node - 1);
        }
    }
    __threadfence();
    __syncthreads();

    int tid = threadIdx.x;
    if (tid == 0) return;

    const int congruence_count = num_nodes - 1;
    if (tid <= congruence_count) {
        // Congruence partner: same node-local index, different node
        int node_offset = tid;  // 1..num_nodes-1
        int target_rank = my_pe_node + ((my_node_id + node_offset) % num_nodes) * gpus_per_node;
        // Write sema_inter[my_node_id] = 1 on target (tells them "node my_node_id's data is ready")
        gin::remote_store(sema_inter + my_node_id, int64_t(1), target_rank);
    } else if (tid <= congruence_count + gpus_per_node - 1) {
        // Same-node rank: different node-local index, same node
        int slot = tid - congruence_count;  // 1..gpus_per_node-1
        int base = (my_pe_node + slot) % gpus_per_node;
        int target_rank = base + my_node_id * gpus_per_node;
        // Write all-batch-bits on target (all batches ready for my local KV)
        for (int w = 0; w < ready_words; w++) {
            gin::remote_store(
                sema_intra + self_rank * ready_words + w, batch_ready_bits(num_batch, w), target_rank);
        }
    }
}

void notify_full(
    int64_t* const __restrict__ semaphores,
    int my_pe,
    int total_pes,
    cudaStream_t stream
) {
    int64_t bit_val = (1LL << total_pes) - (1LL << my_pe) - 1;
    // make sure local store is visible to other ranks and notify other PE that data is ready (full)
    SetFullKernel<<<1, total_pes, 0, stream>>>(semaphores, bit_val, my_pe);
}

/**
 * Hierarchical notify_full: sets reference count = (num_nodes - 1) + (gpus_per_node - 1).
 * Broadcasts to:
 *   - sema_inter[my_node_id] = 1 on congruence partners (Phase 1 cross-node signal)
 *   - sema_intra[my_pe] = 1 on same-node ranks (Phase 2 local chunk signal)
 */
void notify_full_hier(
    int64_t* const __restrict__ sema_inter,
    int64_t* const __restrict__ sema_intra,
    int my_pe,
    int total_pes,
    int gpus_per_node,
    int num_nodes,
    int num_batch,
    cudaStream_t stream
) {
    const int my_pe_node = my_pe % gpus_per_node;
    const int my_node_id = my_pe / gpus_per_node;
    // Phase 1 consumers (num_nodes - 1) + Phase 2 consumers reading local chunk (gpus_per_node - 1)
    const int64_t refcount = hier::hier_local_refcount(total_pes, gpus_per_node);
    // Thread count: 1 (self) + congruence partners + same-node ranks
    const int num_notify = 1 + refcount;
    HierSetFullKernel<<<1, num_notify, 0, stream>>>(
        sema_inter, sema_intra, refcount, my_pe, my_pe_node, my_node_id, num_nodes, gpus_per_node, num_batch);
}

/**
 * Hierarchical post-AG cleanup kernel.
 * Runs after AG remote_get kernel completes on comm_stream.
 *
 * Dual-refcount protocol:
 *   - Producer refcount: sema_intra[X] on producer PE X = (num_nodes-1) + (gpus_per_node-1)
 *     Decremented by: congruent ranks (Phase 1 consumers) + same-node ranks (Phase 2 direct readers)
 *   - Relay refcount: sema_intra[X] on relay holder PE = (gpus_per_node-1)
 *     Set by AG kernel's congruence_notify after Phase 1 fetch.
 *     Decremented by: same-node non-congruent ranks that read the relay copy.
 *
 * Thread layout: threadIdx.x = 0..total_n_pes-1
 *
 * Three cases per target:
 *   1. Congruent (same slot, different node): Phase 1 target.
 *      Clear sema_inter, decrement producer's refcount on target PE.
 *   2. Non-congruent, same node: Phase 2 direct read from producer.
 *      Wait for signal, clear local sema_intra, decrement producer's refcount on target PE.
 *   3. Non-congruent, different node: Phase 2 relay read from same-node relay holder.
 *      Wait for signal, clear local sema_intra, decrement relay holder's relay refcount.
 */
__global__ void HierNotifyEmptyKernel(
    int64_t* const __restrict__ sema_inter,
    int64_t* const __restrict__ sema_intra,
    const int my_pe,
    const int total_n_pes,
    const int num_nodes,
    const int gpus_per_node,
    const int num_batch
) {
    const int target = threadIdx.x;
    const int my_slot = my_pe % gpus_per_node;
    const int my_node = my_pe / gpus_per_node;

    if (target == my_pe) return;

    const int target_slot = target % gpus_per_node;
    const int target_node = target / gpus_per_node;
    const int ready_words = batch_ready_words(num_batch);

    if (target_slot == my_slot) {
        // Case 1: Congruent (Phase 1 target, cross-node).
        sema_inter[target_node] = 0;
        gin::remote_add(sema_intra + target * ready_words, int64_t(-1), target);
    } else {
        wait_full_all_batch(sema_intra, num_batch, target);
        for (int w = 0; w < ready_words; w++) {
            sema_intra[target * ready_words + w] = 0;
        }
        // same node: no remote_pe relay. Othewise we should calculate the relay rank
        int remote_rank = target_node == my_node ? target : target_slot + my_node * gpus_per_node;
        gin::remote_add(sema_intra + target * ready_words, int64_t(-1), remote_rank);
    }
}

/**
 * Hierarchical post-AG cleanup for splitted (BWD RS-overlap) AG.
 * Same logic as HierNotifyEmptyKernel but only processes ranks in the current segment.
 *
 * Thread layout: threadIdx.x = 0..chunk_per_seg-1, maps to segment's logical positions.
 */
__global__ void HierNotifySegmentEmptyKernel(
    int64_t* const __restrict__ sema_inter,
    int64_t* const __restrict__ sema_intra,
    const int my_pe,
    const int segment_idx,
    const int num_chunks,
    const int total_n_pes,
    const int gpus_per_node,
    const int num_batch
) {
    const int my_slot = my_pe % gpus_per_node;

    // Map threadIdx.x to the global logical position in this segment
    const int local_chunk_id = threadIdx.x;
    const int global_logical_pos = segment_idx * num_chunks + local_chunk_id;
    if (global_logical_pos == 0) return;  // local chunk, nothing to do

    const int target = hier::hier_target_rank(global_logical_pos, my_pe, total_n_pes, gpus_per_node);
    if (target == my_pe) return;

    const int target_slot = target % gpus_per_node;
    const int target_node = target / gpus_per_node;

    const int my_node = my_pe / gpus_per_node;
    const int ready_words = batch_ready_words(num_batch);

    if (target_slot == my_slot) {
        // Case 1: Congruent (Phase 1 target, cross-node).
        sema_inter[target_node] = 0;
        // Do NOT zero sema_intra[target] — relay refcount managed by Phase 2 consumers.
        gin::remote_add(sema_intra + target * ready_words, int64_t(-1), target);
    } else {
        wait_full_all_batch(sema_intra, num_batch, target);
        for (int w = 0; w < ready_words; w++) {
            sema_intra[target * ready_words + w] = 0;
        }
        // same node: no remote_pe relay. Othewise we should calculate the relay rank
        int remote_rank = target_node == my_node ? target : target_slot + my_node * gpus_per_node;
        gin::remote_add(sema_intra + target * ready_words, int64_t(-1), remote_rank);
    }
}

void notify_all_empty_hier(
    int64_t* const __restrict__ sema_inter,
    int64_t* const __restrict__ sema_intra,
    int my_pe,
    int total_pes,
    int gpus_per_node,
    int num_nodes,
    int num_batch,
    cudaStream_t stream
) {
    HierNotifyEmptyKernel<<<1, total_pes, 0, stream>>>(
        sema_inter, sema_intra, my_pe, total_pes, num_nodes, gpus_per_node, num_batch);
}

void notify_segment_empty_hier(
    int64_t* const __restrict__ sema_inter,
    int64_t* const __restrict__ sema_intra,
    int my_pe,
    int segment_idx,
    int num_chunks,
    int total_pes,
    int gpus_per_node,
    int num_batch,
    cudaStream_t stream
) {
    HierNotifySegmentEmptyKernel<<<1, num_chunks, 0, stream>>>(
        sema_inter, sema_intra, my_pe, segment_idx, num_chunks, total_pes, gpus_per_node, num_batch);
}

__global__ void WaitSelfEmptyHierarchicalKernel(
    int64_t* const __restrict__ sema_intra,
    int my_pe_node,
    int my_node_id,
    int gpus_per_node,
    int num_batch
) {
    int partner = my_pe_node + ((my_node_id + threadIdx.x) % blockDim.x) * gpus_per_node;
    // Refcounts live in word 0 of each rank's slot.
    const int64_t* const refcount = sema_intra + partner * batch_ready_words(num_batch);
    int64_t current_val;
    do {
        asm volatile("ld.volatile.global.s64 %0, [%1];"
            : "=l"(current_val) : "l"(refcount) : "memory");
    } while (current_val);
}

/**
 * @brief CPU wait until both producer refcount and all relay refcounts reach 0.
 *   Waits for sema_intra[my_pe] == 0 (producer refcount) and
 *   sema_intra[each_congruence_partner] == 0 (relay refcounts).
 *   Must be called before HierSetFullKernel to guarantee no data overwrite.
 */
void wait_self_empty_hier(
    int64_t* const __restrict__ sema_intra,
    int my_pe,
    int gpus_per_node,
    int num_nodes,
    int num_batch,
    cudaStream_t stream
) {
    const int my_pe_node = my_pe % gpus_per_node;
    const int my_node_id = my_pe / gpus_per_node;
    WaitSelfEmptyHierarchicalKernel<<<1, num_nodes, 0, stream>>>(
        sema_intra, my_pe_node, my_node_id, gpus_per_node, num_batch
    );
}

}   // namespace ag
}   // namespace sema
}   // namespace flashmask