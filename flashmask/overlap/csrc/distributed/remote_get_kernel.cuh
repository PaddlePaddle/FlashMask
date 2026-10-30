// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 The FlashMask Authors. All rights reserved.

#pragma once
#include <cuda_runtime.h>
#include "sr_buffer.cuh"
#include "gin_copy_utils.cuh"
#include "ag_semaphore_ops.cuh"
#include "hierarchical_rank_map.cuh"
#include "debug_logger.cuh"

namespace flashmask {

// Tells the compute stream that this comm CTA is resident (see
// WaitAndResetStreamCoordKernel). Release: it also publishes everything stream-ordered
// before this kernel, which reaches the compute kernel only through this edge.
__device__ __forceinline__ void announce_comm_cta(int* stream_coordinator) {
    if (threadIdx.x == 0) {
        cuda::atomic_ref<int, cuda::thread_scope_device>(*stream_coordinator)
            .fetch_or(1 << blockIdx.x, cuda::memory_order_release);
    }
}

__global__ void InitBitmapForLocalSkip(
    int* const __restrict__ work_done,
    int work_to_skip,       // local chunk works per batch
    int work_per_seg,       // total works per batch (= num_chunks * work_per_chunk)
    int num_batch
) {
    const int total = num_batch * work_to_skip;
    for (int i = threadIdx.x; i < total; i += blockDim.x) {
        int b = i / work_to_skip;
        int w = i % work_to_skip;
        work_done[b * work_per_seg + w + 1] = 1;   // 1-indexed
    }
}

// Supports mask_head >= 1. When mask_head > 1, AND-reduces across all mask heads
// so that a chunk is skippable only when ALL heads agree it is fully masked.
// When mask_head == 1, the head loop runs once with negligible overhead.
// mask: (B, num_head, S_k, 2) int32 [lt_start, ut_end] pairs with row stride 2; each thread
// checks 4 rows with two aligned int4 loads.
template <int num_warps = 8, int row_per_warp = 32>
__global__ void __launch_bounds__(num_warps * 32, 64 / num_warps) BlockSparsityCheckSpecializedKernel(
    const int* const __restrict__ mask,
    int* const __restrict__ copy_chunk_mask,
    const int num_head,                         // mask head count (H_mask)
    const int64_t batch_stride,                 // int32 elements, multiple of 4
    const int64_t head_stride                   // int32 elements, multiple of 4
) {
    static_assert(num_warps == 16 || num_warps == 8 || num_warps == 4);
    static_assert(row_per_warp == 32 || row_per_warp == 64 || row_per_warp == 16);
    static_assert(row_per_warp != 16 || (row_per_warp == 16 && num_warps == 4));
    static constexpr int rows_per_cta = row_per_warp * num_warps;
    __shared__ int warps_masked[num_warps];
    __shared__ int temp_result[4];              // cross-head AND-reduction buffer
    const int load_index = blockIdx.x * num_warps * 32 + threadIdx.x;
    const int* const rows = mask + blockIdx.y * batch_stride + int64_t(load_index) * 8;
    for (int head_id = 0; head_id < num_head; head_id++) {
        const int4* const pairs = reinterpret_cast<const int4*>(rows + head_id * head_stride);
        const int4 p01 = pairs[0];
        const int4 p23 = pairs[1];
        int is_masked = (p01.x <= p01.y) && (p01.z <= p01.w) && (p23.x <= p23.y) && (p23.z <= p23.w);
        int current_warp_masked = __all_sync(0xffffffff, is_masked);
        if ((threadIdx.x % 32) == 0) {
            warps_masked[threadIdx.x / 32] = current_warp_masked;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            const int4* smem_int4 = reinterpret_cast<const int4*>(warps_masked);
            if constexpr (rows_per_cta == 1024) {
                int2 result;
                int4 src = *smem_int4;
                result.x = src.x & src.y & src.z & src.w;
                src = *(smem_int4 + 1);
                result.x &= src.x & src.y & src.z & src.w;
                src = *(smem_int4 + 2);
                result.y = src.x & src.y & src.z & src.w;
                src = *(smem_int4 + 3);
                result.y &= src.x & src.y & src.z & src.w;
                int2* const addr = reinterpret_cast<int2*>(temp_result);
                if (head_id == 0) { *addr = result; }
                else { int2 old = *addr; old.x &= result.x; old.y &= result.y; *addr = old; }
            }
            if constexpr (rows_per_cta == 512) {
                int4 src = *smem_int4;
                if constexpr (num_warps == 16) {
                    int4 result;
                    result.x = src.x & src.y & src.z & src.w;
                    src = *(smem_int4 + 1);
                    result.y = src.x & src.y & src.z & src.w;
                    src = *(smem_int4 + 2);
                    result.z = src.x & src.y & src.z & src.w;
                    src = *(smem_int4 + 3);
                    result.w = src.x & src.y & src.z & src.w;
                    int4* const addr = reinterpret_cast<int4*>(temp_result);
                    if (head_id == 0) { *addr = result; }
                    else { int4 old = *addr; old.x &= result.x; old.y &= result.y; old.z &= result.z; old.w &= result.w; *addr = old; }
                } else {
                    int2 result;
                    result.x = src.x & src.y & src.z & src.w;
                    src = *(smem_int4 + 1);
                    result.y = src.x & src.y & src.z & src.w;
                    int2* const addr = reinterpret_cast<int2*>(temp_result);
                    if (head_id == 0) { *addr = result; }
                    else { int2 old = *addr; old.x &= result.x; old.y &= result.y; *addr = old; }
                }
            }
            if constexpr (rows_per_cta == 256) {
                int4 result;
                int4 src = *smem_int4;
                result.x = src.x & src.y;
                result.y = src.z & src.w;
                src = *(smem_int4 + 1);
                result.z = src.x & src.y;
                result.w = src.z & src.w;
                int4* const addr = reinterpret_cast<int4*>(temp_result);
                if (head_id == 0) { *addr = result; }
                else { int4 old = *addr; old.x &= result.x; old.y &= result.y; old.z &= result.z; old.w &= result.w; *addr = old; }
            }
            if constexpr (rows_per_cta == 64) {
                int4 result = *smem_int4;
                int4* const addr = reinterpret_cast<int4*>(temp_result);
                if (head_id == 0) { *addr = result; }
                else { 
                    int4 old = *addr;
                    old.x &= result.x;
                    old.y &= result.y;
                    old.z &= result.z;
                    old.w &= result.w;
                    *addr = old; 
                }
            }
        }
        // barrier before next head iteration so warps_masked is not overwritten prematurely
        if (head_id < num_head - 1) __syncthreads();
    }
    // write the AND-reduced result to global memory
    if (threadIdx.x == 0) {
        const int block_offset = blockIdx.y * gridDim.x + blockIdx.x;
        if constexpr (rows_per_cta == 1024 || (rows_per_cta == 512 && num_warps != 16)) {
            *(reinterpret_cast<int2*>(copy_chunk_mask) + block_offset) = *(reinterpret_cast<int2*>(temp_result));
        } else {
            *(reinterpret_cast<int4*>(copy_chunk_mask) + block_offset) = *(reinterpret_cast<int4*>(temp_result));
        }
    }
}

/**
 * 'Sparse' means that we will check whether the 256/512-row chunk can be skipped or not.
 * @param copy_chunk_mask Pre-computed chunk mask. If all 256/512 rows in a chunk are masked, mask[chunk_id] = 1
 *  copy_chunk_mask is generated by `BlockSparsityCheckKernel`, 1D buffer (size: (S - S_chunk) / 256 or 512)
 * @param  block_cnt_semaphore Used in dynamic scheduling. Since each CTA is responsible for remote-getting
 *  one entire chunk (256/512 rows), if the chunk is masked then the computation power will be wasted. To avoid
 *  load-imbalance for this communication kernel (which is crucial to overlap performance), a chunk-cnt semaphore
 *  is used so every CTA use atomic op to get a chunk ID to process.
*/
template <typename T, int S, int S_chunk, int num_warps=8, int row_per_warp=32, bool use_hierarchical=false>
__global__ void __launch_bounds__(num_warps * 32, 64 / num_warps) SparseLargeKVChunkRemoteGetKernel(
    T* const __restrict__ k_sr,
    T* const __restrict__ v_sr,
    int* const __restrict__ wptr,
    int* const __restrict__ block_work_idx,
    int* const __restrict__ block_cnt_semaphore,      // for dynamic scheduling
    int* const __restrict__ stream_coordinator,
    const int* const __restrict__ copy_chunk_mask,
    const int my_pe,
    const int total_n_pes,
    const int num_batch,                // B (or B*H when simulating per-head RDMA)
    const int S_stride,                 // H * D (or D when simulating)
    int64_t* const __restrict__ semaphores = nullptr,
    int* const __restrict__ rank_empty_counters = nullptr,
    const int gpus_per_node = 1,
    const int sema_inter_size = 0,
    const int num_heads_per_batch = 1   // >1: num_batch = B*H, mask indexed by batch_id/num_heads_per_batch
) {
    // Degenerate case: S == S_chunk means nranks=1, nothing to remote-get.
    // This template instantiation is unreachable at runtime (overlap requires nranks > 1)
    // but is generated by SChunkDispatch × SeqlenDispatch combinatorial expansion.
    // Early return with proper wptr signaling to prevent hang if ever launched by mistake.
    if constexpr (S <= S_chunk) {
        announce_comm_cta(stream_coordinator);
        if (threadIdx.x == 0) atomicMax(wptr, INT_MAX);
        return;
    }

    announce_comm_cta(stream_coordinator);
    constexpr int row_per_block = num_warps * row_per_warp;     // 256 or 512 row per block (32 or 64 per warp)
    constexpr int seqlen_offset = S - S_chunk;
    constexpr int chunk_per_batch = (S - S_chunk) / row_per_block;
    constexpr int work_per_chunk = S_chunk / row_per_block;
    const int total_chunks = num_batch * chunk_per_batch;
    // the gathered KV (num_batch * S * S_stride) exceeds INT_MAX for large shapes,
    // so every offset built from it is 64-bit. Loop counters stay int.
    const int64_t row_stride = S_stride;
    const int64_t batch_stride = S * row_stride;
    __shared__ int cached_semaphores[64];
    __shared__ int next_work_id;
    // note that block_cnt_semaphore starts from 1. dyn-scheduling from the beginning.

    // Dual-array pointers (only meaningful when use_hierarchical)
    int64_t* const sema_inter = semaphores;
    int64_t* const sema_intra = semaphores + sema_inter_size;

    // Publishes the frontier: wptr = min over all CTAs' latest completed work id. Lane j
    // of every CTA reads slot j, so the grid has exactly 32 CTAs (static_assert at the
    // launch site). Each CTA releases its slot after the bar.sync that ends its copy;
    // the lanes' acquires plus bar.warp.sync carry every CTA's data to the publishing
    // lane, whose release fetch_max pairs with the compute gate's acquire. A stale slot
    // is only ever smaller, so a stale read keeps wptr conservative.
    auto update_wptr_and_work_id_sync = [&](int wid) {
        if (threadIdx.x < 32) {
            if (threadIdx.x == blockIdx.x) {
                int next_wid = atomicAdd(block_cnt_semaphore, 1);       // fetch and check the next work ID
                // if there is no more work to do: set the wid for the current block to be INT_MAX
                wid = next_wid <= total_chunks ? wid : INT_MAX;
                next_work_id = next_wid;
                cuda::atomic_ref<int, cuda::thread_scope_device>(block_work_idx[blockIdx.x])
                    .store(wid, cuda::memory_order_release);
            } else {
                wid = cuda::atomic_ref<int, cuda::thread_scope_device>(block_work_idx[threadIdx.x])
                    .load(cuda::memory_order_acquire);
            }
            wid = __reduce_min_sync(0xffffffff, wid);
            __syncwarp();
            if (threadIdx.x == blockIdx.x) {
                cuda::atomic_ref<int, cuda::thread_scope_device>(*wptr).fetch_max(
                    wid == INT_MAX ? INT_MAX : (wid * row_per_block),       // 256 or 512 * wid, or INT_MAX
                    cuda::memory_order_release);
            }
        }
        __syncthreads();
        return next_work_id;
    };

    // Phase 1 per-batch congruence_notify (hierarchical only).
    // When all works for a Phase 1 target in a given batch complete, notify same-node ranks
    // via atomic_or so Phase 2 cross-node consumers can proceed per-batch.
    // All cleanup (refcount add, local state clear) is deferred to a separate
    // post-AG kernel on comm_stream (HierNotifyEmptyKernel / NotifySemaphoreEmptyKernel).
    auto try_congruence_notify = [&](int target_rank_val, bool is_phase1_val, int batch_id) {
        if constexpr (use_hierarchical) {
            if (threadIdx.x == 0 && is_phase1_val) {
                const int counter_offset = target_rank_val + total_n_pes * batch_id;
                // acq_rel: releases this CTA's SR buffer writes; the winner acquires all of
                // them before notify_batch_ready_to_same_node releases to the peers.
                int prev = cuda::atomic_ref<int, cuda::thread_scope_device>(rank_empty_counters[counter_offset])
                    .fetch_add(1, cuda::memory_order_acq_rel);
                if (prev + 1 == work_per_chunk) {
                    // All works for this (batch, Phase 1 target) are done.
                    // Notify same-node ranks: relay data for this batch is ready.
                    sema::ag::notify_batch_ready_to_same_node(
                        sema_intra, target_rank_val, batch_id, num_batch, my_pe, gpus_per_node);
                }
            }
        }
    };

    if (threadIdx.x < total_n_pes) {
        cached_semaphores[threadIdx.x] = 0;
    }

    // // Uncomment this line to check what happens when communication is not stalling the computation
    // if (threadIdx.x == 0) {
    //     atomicMax(wptr, INT_MAX);
    // }

    for (int work_id = update_wptr_and_work_id_sync(0); work_id <= total_chunks;) {
        const int work_id_m1 = work_id - 1;
        const int batch_id = work_id_m1 / chunk_per_batch;
        const int seq_work_id = (work_id_m1 % chunk_per_batch) + 1;     // this is in range [1, chunk_per_batch]
        int mask_index = 0;
        int seqlen_id = 0, remote_pe = 0;
        int64_t src_addr = 0;
        bool is_phase1 = true;
        int target_rank = 0;  // For hierarchical: data owner rank (semaphore wait target)

        if constexpr (use_hierarchical) {
            // Hierarchical traversal: congruence group first, intra-node second
            const int logical_pos = (seq_work_id - 1) / work_per_chunk + 1;  // 1-based remote chunk: skip local
            // Reverse chunk-internal row order so that high-seqlen (rightmost) rows are
            // fetched first (low work_id), matching wptr's right-to-left contiguous frontier semantic.
            const int row_within_chunk = work_per_chunk - 1 - (seq_work_id - 1) % work_per_chunk;
            auto info = hier::hier_map_chunk(logical_pos, my_pe, total_n_pes, gpus_per_node);

            target_rank = info.target_rank;  // Data owner (semaphore wait target)
            remote_pe = info.src_pe;         // PE to fetch from (Phase 1: = target_rank; Phase 2: = same-node rank)
            is_phase1 = info.is_phase1;
            seqlen_id = hier::hier_seqlen_id(logical_pos, row_within_chunk, row_per_block, total_n_pes, S_chunk, false);

            // mask_index: seqlen_id to copy_chunk_mask entry (mask laid out in SR buffer order from seqlen 0)
            mask_index = (batch_id / num_heads_per_batch) * chunk_per_batch + seqlen_id / row_per_block;

            // Source address in remote_pe's SR buffer
            const int src_chunk_seqlen = hier::hier_src_chunk_offset(
                info, total_n_pes, S_chunk, total_n_pes, gpus_per_node, false);
            src_addr = batch_id * batch_stride + (src_chunk_seqlen + row_within_chunk * row_per_block) * row_stride;
        } else {
            // Original circular shift traversal, reversed
            seqlen_id = seqlen_offset - seq_work_id * row_per_block;
            const int cp_chunk_id = (S - 1 - seqlen_id) / S_chunk;
            remote_pe = my_pe - cp_chunk_id;
            remote_pe = remote_pe >= 0 ? remote_pe : remote_pe + total_n_pes;
            target_rank = remote_pe;  // Non-hierarchical: target_rank = remote_pe
            src_addr = batch_id * batch_stride + (seqlen_offset + (seqlen_id % S_chunk)) * row_stride;
            mask_index = chunk_per_batch * (batch_id / num_heads_per_batch + 1) - seq_work_id;
        }

        bool should_skip = false;
        if constexpr (use_hierarchical) {
            // Sparse skip: Phase 1 (cross-node) must NEVER skip because same-node
            // ranks may need the data via Phase 2. Only Phase 2 can be skipped.
            // TODO(heqianyue): we can actually use sparse comm if we reduce-and all the 
            // copy_chunk_mask across the local ranks (the union of all the required chunks)
            should_skip = (!is_phase1) && copy_chunk_mask[mask_index];
        } else {
            should_skip = copy_chunk_mask[mask_index];
        }
        if (should_skip) {
            try_congruence_notify(target_rank, is_phase1, batch_id);
            __syncthreads();
            work_id = update_wptr_and_work_id_sync(work_id);
            continue;
        }
        if constexpr (use_hierarchical) {
            if (is_phase1) {
                // Phase 1: wait_full only (wait_empty done by host wait_sr_buffer_empty)
                if (threadIdx.x == 0 && cached_semaphores[target_rank] == 0) {
                    const int target_node = target_rank / gpus_per_node;
                    sema::ag::wait_full(sema_inter, target_node);
                    cached_semaphores[target_rank] = 1;
                }
            } else {
                // Phase 2 (intra-node only): per-batch bit-check on sema_intra
                // Waits until the specific batch_id bit is set by congruence_notify
                if (threadIdx.x == 0) {
                    sema::ag::wait_full_one_batch(sema_intra, batch_id, target_rank, num_batch);
                }
            }
        } else {
            if (threadIdx.x == 0 && cached_semaphores[target_rank] == 0) {
                sema::ag::wait_full(semaphores, target_rank);
                cached_semaphores[target_rank] = 1;
            }
        }
        // copy upto 4 heads (S_stride = H * D), and row_per_block rows (seqlen axis) per CTA
        const int64_t dst_addr = batch_id * batch_stride + seqlen_id * row_stride;
        shmem::two_buffers_getmem_block(
            k_sr + dst_addr,
            shmem::offset_or_null(v_sr, dst_addr),
            k_sr + src_addr,
            shmem::offset_or_null(v_sr, src_addr),
            row_per_block * S_stride * sizeof(T), remote_pe
        );
        // buffer getmem_block will call syncthreads(), so next_work_id will not be updated
        // therefore, next_work_id's update is visible to all threads and won't be overwritten
        // before some threads reading it. Safe!
        try_congruence_notify(target_rank, is_phase1, batch_id);
        work_id = update_wptr_and_work_id_sync(work_id);
    }
}

// Backward remote get kernel: gathers one segment (num_chunks chunks) of the split AG.
// The bwd compute kernel gates each KV tile on the per-work work_done flag.
//
// @param num_chunks  total chunks per segment (always the real count, e.g. 4 for CP16)
// @param has_local_chunk  true iff segment 0 (the first chunk is local and should be skipped)
// @param use_hierarchical  when true, use hierarchical comm (Phase1: cross-node IB, Phase2: intra-node NVLink).
//                          In hierarchical mode the SR buffer is NOT reused across segments; each segment writes
//                          to a unique [segment_idx * num_chunks * S_chunk, (segment_idx+1) * ...) region so that
//                          later segments' Phase-2 consumers can still read Phase-1 data fetched in earlier segments.
template <typename T, int S_chunk, int num_warps=8, int row_per_warp=32, int num_chunks=4, bool has_local_chunk=false, bool use_hierarchical=false>
__global__ void __launch_bounds__(num_warps * 32, 64 / num_warps) SparseLargeKVChunkSplittedRemoteGetKernel(
    T* const __restrict__ k_sr,
    T* const __restrict__ v_sr,
    const T* const __restrict__ local_k,
    const T* const __restrict__ local_v,
    int* const __restrict__ block_work_idx,
    int* const __restrict__ block_cnt_semaphore,      // for dynamic scheduling
    int* const __restrict__ stream_coordinator,
    const int* const __restrict__ copy_chunk_mask,
    const int my_pe,
    const int start_rank,               // first call is my_pe + 1 (non-hierarchical only)
    const int segment_idx,
    const int total_n_pes,
    const int num_batch,                // B
    const int S_stride,                 // H * D
    const int num_segments = 4,
    int64_t* const __restrict__ semaphores = nullptr,
    int* const __restrict__ rank_empty_counters = nullptr,
    const int gpus_per_node = 1,        // only used when use_hierarchical=true
    const int sema_inter_size = 0,
    const int num_heads_per_batch = 1   // >1: num_batch = B*H, mask indexed by batch_id/num_heads_per_batch
) {
    announce_comm_cta(stream_coordinator);
    // segment has only a local chunk, nothing to remote-get.
    if constexpr (has_local_chunk && num_chunks == 1) {
        return;
    }
    constexpr bool has_local = has_local_chunk;
    constexpr int row_per_block = num_warps * row_per_warp;
    constexpr int S = S_chunk * num_chunks;                  // seqlen of this segment (num_chunks is already the real total)
    constexpr int work_per_seg = S / row_per_block;
    constexpr int work_per_chunk = S_chunk / row_per_block;
    // for each segment, get the number of work we can skip (due to being local). Note that
    // if work_to_skip is not 0, there will be some skippable works for **each batch**
    constexpr int work_to_skip = has_local ? (S_chunk / row_per_block) : 0;
    constexpr int chunk_offset = has_local ? 1 : 0;
    // though the actual skippable work is `num_batch * (work_per_seg - work_to_skip)`
    // for batch_idx > 0, those skippable local works cannot be skipped **directly**.
    const int total_works = num_batch * work_per_seg;
    // Per-stage batch stride = num_chunks * S_chunk * S_stride.
    // Both dst (k_sr) and src (local_k) live in the same SR buffer that is partitioned
    // into num_segments independent sub-tensors each of size B * num_chunks * S_chunk.
    // Stage separation is handled by the caller advancing k_sr/local_k per segment.
    // Non-hierarchical: same formula, overwrites the single segment region.
    const int64_t row_stride = S_stride;
    const int64_t batch_stride = S * row_stride;
    const int64_t local_batch_stride = S_chunk * row_stride;

    extern __shared__ int smem_chunk_mask[];
    __shared__ int cached_semaphores[64];
    __shared__ int next_work_id;

    // Dual-array pointers (only meaningful when use_hierarchical)
    int64_t* const sema_inter = semaphores;
    int64_t* const sema_intra = semaphores + sema_inter_size;

    // Load copy_chunk_mask into shared memory (block-stride loop for total_works > blockDim.x).
    // Barrier: first update_wptr_and_work_id_sync call has __syncthreads() before any read.
    for (int i = threadIdx.x; i < total_works; i += blockDim.x) {
        const int batch_id = i / work_per_seg;
        const int real_batch = batch_id / num_heads_per_batch;
        auto* src_ptr = copy_chunk_mask + (segment_idx + real_batch * num_segments) * work_per_seg
            + (i % work_per_seg);
        smem_chunk_mask[i] = *src_ptr;
    }

    if (threadIdx.x < total_n_pes) {
        cached_semaphores[threadIdx.x] = 0;
    }

    // bitmap pointers (compute kernel reads work_done[work_id] directly — per-work ready flag)
    int* const work_done = block_work_idx;

    // Per-work ready flag update (thread-0 only):
    //   Mark:  release store work_done[wid]=1. Thread 0 is ordered after every thread's copy
    //          by the bar.sync ending two_buffers_getmem_block, and after NIC data by its own
    //          completed wait(); pairs with the compute kernel's acquire load.
    //   Next:  atomicAdd to grab the next work item
    // No scan or wptr update needed — compute kernel checks work_done[] directly.
    // All local chunk works (every batch) are pre-filled by InitBitmapForLocalSkip,
    // so we skip the mark for any local chunk work: (wid-1) % work_per_seg < work_to_skip.
    auto update_wptr_and_work_id_sync = [&](int wid) {
        if (threadIdx.x == 0) {
            if constexpr (has_local) {
                if ((wid - 1) % work_per_seg >= work_to_skip) {
                    cuda::atomic_ref<int, cuda::thread_scope_device>(work_done[wid])
                        .store(1, cuda::memory_order_release);
                }
            } else {
                if (wid > 0) {
                    cuda::atomic_ref<int, cuda::thread_scope_device>(work_done[wid])
                        .store(1, cuda::memory_order_release);
                }
            }
            next_work_id = atomicAdd(block_cnt_semaphore, 1) + work_to_skip;
        }
        __syncthreads();
        return next_work_id;
    };

    // Phase 1 per-batch congruence_notify (hierarchical only).
    // Same logic as in the forward kernel — see comments there.
    auto try_congruence_notify = [&](int chunk_id, int target_rank_val, bool is_phase1_val, int batch_id) {
        if constexpr (use_hierarchical) {
            if (threadIdx.x == 0 && is_phase1_val) {
                const int counter_idx = (chunk_id - chunk_offset) + num_chunks * batch_id;
                // acq_rel: see the non-split kernel.
                int prev = cuda::atomic_ref<int, cuda::thread_scope_device>(rank_empty_counters[counter_idx])
                    .fetch_add(1, cuda::memory_order_acq_rel);
                if (prev + 1 == work_per_chunk) {
                    sema::ag::notify_batch_ready_to_same_node(
                        sema_intra, target_rank_val, batch_id, num_batch, my_pe, gpus_per_node);
                }
            }
        }
    };

    // Note(heqianyue): the simple way to skip the first chunk: offset the initial work_cnt
    // and the atomicAdd result (semaphore fetch result)
    for (int work_id = update_wptr_and_work_id_sync(work_to_skip); work_id <= total_works;) {
        const int work_id_m1 = work_id - 1;
        const int batch_id = work_id_m1 / work_per_seg;
        const int seq_work_id = work_id_m1 % work_per_seg;

        // Local chunk works: skip, no release needed
        if (seq_work_id < work_to_skip) {
            __syncthreads();
            work_id = update_wptr_and_work_id_sync(work_id);
            continue;
        }

        int chunk_id = seq_work_id / work_per_chunk;

        // Compute remote PE, src/dst addresses (done early so masked-skip can check Phase 1)
        int seqlen_id = 0, remote_pe = 0;
        int64_t src_addr = 0;
        bool is_phase1 = false;
        int target_rank = 0;

        if constexpr (use_hierarchical) {
            // Hierarchical: global_logical_pos maps to target_rank via congruence-group-first order
            const int global_logical_pos = segment_idx * num_chunks + chunk_id;
            const int row_within_chunk = seq_work_id % work_per_chunk;
            auto info = hier::hier_map_chunk(global_logical_pos, my_pe, total_n_pes, gpus_per_node);

            target_rank = info.target_rank;
            remote_pe   = info.src_pe;
            is_phase1   = info.is_phase1;

            // dst: segment-local offset (k_sr is already advanced to this segment's region by caller)
            // seq_work_id * row_per_block == chunk_id * S_chunk + row_within_chunk * row_per_block
            seqlen_id = seq_work_id * row_per_block;

            // src: from local_k (caller passes the stage-advanced SR buffer base for Phase 2 reads,
            //   OR the actual local KV base for Phase 1; both use the same batch_stride).
            //   Phase 1 (bwd): target_rank's local KV at position 0 in its SR buffer stage.
            //   Phase 2 (bwd): target_rank's data at position j in src_pe's SR buffer stage 0,
            //     where src_pe wrote Phase 1 data during segment 0 at seqlen j * S_chunk.
            const int src_chunk_seqlen = hier::hier_src_chunk_offset(
                info, total_n_pes, S_chunk, total_n_pes, gpus_per_node, /*bwd=*/true);
            src_addr = batch_id * batch_stride + (src_chunk_seqlen + row_within_chunk * row_per_block) * row_stride;
        } else {
            // Non-hierarchical: circular shift order
            seqlen_id = seq_work_id * row_per_block;
            remote_pe = start_rank + seqlen_id / S_chunk;
            remote_pe = remote_pe < total_n_pes ? remote_pe : remote_pe - total_n_pes;
            target_rank = remote_pe;
            src_addr = batch_id * local_batch_stride + (seqlen_id % S_chunk) * row_stride;
        }

        bool should_skip = false;
        if constexpr (use_hierarchical) {
            should_skip = (!is_phase1) && smem_chunk_mask[work_id_m1];
        } else {
            should_skip = smem_chunk_mask[work_id_m1];
        }

        if (should_skip) {
            try_congruence_notify(chunk_id, target_rank, is_phase1, batch_id);
            __syncthreads();
            work_id = update_wptr_and_work_id_sync(work_id);
            continue;
        }

        if constexpr (use_hierarchical) {
            if (is_phase1) {
                // Phase 1: wait_full only (wait_empty done by host wait_sr_buffer_empty)
                if (threadIdx.x == 0 && cached_semaphores[target_rank] == 0) {
                    const int target_node = target_rank / gpus_per_node;
                    sema::ag::wait_full(sema_inter, target_node);
                    cached_semaphores[target_rank] = 1;
                }
            } else {
                // Phase 2 (intra-node only): per-batch bit-check on sema_intra
                if (threadIdx.x == 0) {
                    sema::ag::wait_full_one_batch(sema_intra, batch_id, target_rank, num_batch);
                }
            }
        } else {
            if (threadIdx.x == 0 && cached_semaphores[target_rank] == 0) {
                sema::ag::wait_full(semaphores, target_rank);
                cached_semaphores[target_rank] = 1;
            }
        }

        // copy upto 4 heads (S_stride = H * D), and row_per_block rows (seqlen axis) per CTA
        const int64_t dst_addr = batch_id * batch_stride + seqlen_id * row_stride;
        // src is always local_k/local_v (the SR buffer base pointer on remote_pe, passed by caller):
        //   Hierarchical Phase 1: remote_pe == target_rank, reads its local KV at SR offset 0.
        //   Hierarchical Phase 2: remote_pe == src_pe (same-node), reads target's congruence slot.
        //   Non-hierarchical: remote_pe's original local KV buffer (same pointer name, different meaning).
        // In all cases local_k is the correct base; src_addr encodes the per-case offset.
        shmem::two_buffers_getmem_block(
            k_sr + dst_addr,
            shmem::offset_or_null(v_sr, dst_addr),
            local_k + src_addr,
            shmem::offset_or_null(local_v, src_addr),
            row_per_block * S_stride * sizeof(T), remote_pe
        );
        // buffer getmem_block will call syncthreads(), so next_work_id will not be updated.
        // therefore, next_work_id's update is visible to all threads and won't be overwritten
        // before some threads reading it. Safe!
        try_congruence_notify(chunk_id, target_rank, is_phase1, batch_id);
        work_id = update_wptr_and_work_id_sync(work_id);
    }
}

}   // namespace flashmask