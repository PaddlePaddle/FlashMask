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

// Max AG world size (one cached wait_full per rank in smem); init_singleton_instance rejects more.
inline constexpr int kMaxCommRanks = 64;

// Marks this comm CTA resident for the compute stream. Release: this edge is the only path
// by which earlier stream-ordered work reaches the compute kernel.
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

// copy_chunk_mask[b * num_works + w] = 1 iff every mask head masks every row of work w, on the
// AG/RS work grid: work w covers rows [j * rows_per_work, (j + 1) * rows_per_work) of chunk
// c = w / work_per_chunk (j = w % work_per_chunk), clipped to the chunk's S_chunk rows.
// mask: (B, num_head, S_k, 2) int32 [lt_start, ut_end]; each thread checks 4 rows via two int4
// loads, so S_chunk % 4 == 0.
template <int rows_per_work>
__global__ void __launch_bounds__(rows_per_work / 4) ChunkMaskKernel(
    const int* const __restrict__ mask,
    int* const __restrict__ copy_chunk_mask,
    const int num_head,                         // mask head count (H_mask)
    const int64_t batch_stride,                 // int32 elements, multiple of 4
    const int64_t head_stride,                  // int32 elements, multiple of 4
    const int S_chunk,
    const int num_works                         // works per batch
) {
    const int work_per_chunk = (S_chunk + rows_per_work - 1) / rows_per_work;
    const int chunk = blockIdx.x / work_per_chunk;
    const int work_row = blockIdx.x % work_per_chunk * rows_per_work;
    const int row = chunk * S_chunk + work_row + 4 * threadIdx.x;
    int masked = 1;
    if (work_row + 4 * threadIdx.x < S_chunk) {
        const int* const rows = mask + blockIdx.y * batch_stride + int64_t(row) * 2;
        for (int head_id = 0; head_id < num_head; head_id++) {
            const int4* const pairs = reinterpret_cast<const int4*>(rows + head_id * head_stride);
            const int4 p01 = pairs[0];
            const int4 p23 = pairs[1];
            masked &= (p01.x <= p01.y) && (p01.z <= p01.w) && (p23.x <= p23.y) && (p23.z <= p23.w);
        }
    }
    masked = __syncthreads_and(masked);
    if (threadIdx.x == 0) copy_chunk_mask[blockIdx.y * num_works + blockIdx.x] = masked;
}

/**
 * 'Sparse' means that we will check whether the 512-row work can be skipped or not.
 * @param copy_chunk_mask Pre-computed by `ChunkMaskKernel`: mask[work] = 1 if all rows of the work are masked.
 *  Each chunk is cut into ceil(S_chunk / 512) works from its first row, so its last work may be shorter.
 * @param  block_cnt_semaphore Used in dynamic scheduling. Since each CTA is responsible for remote-getting
 *  one entire chunk (256/512 rows), if the chunk is masked then the computation power will be wasted. To avoid
 *  load-imbalance for this communication kernel (which is crucial to overlap performance), a chunk-cnt semaphore
 *  is used so every CTA use atomic op to get a chunk ID to process.
*/
template <typename T, int num_warps=8, int row_per_warp=32, bool use_hierarchical=false>
__global__ void __launch_bounds__(num_warps * 32, shmem::comm_min_blocks_per_sm(num_warps)) SparseLargeKVChunkRemoteGetKernel(
    T* const __restrict__ k_sr,
    T* const __restrict__ v_sr,
    int* const __restrict__ wptr,
    int* const __restrict__ block_work_idx,
    int* const __restrict__ block_cnt_semaphore,      // for dynamic scheduling
    int* const __restrict__ exit_count,               // exited CTAs; zeroed per launch
    int* const __restrict__ stream_coordinator,
    const int* const __restrict__ copy_chunk_mask,
    const int my_pe,
    const int total_n_pes,
    const int num_batch,                // B (or B*H when simulating per-head RDMA)
    const int S_stride,                 // H * D (or D when simulating)
    const int S_chunk,                  // S_local
    int64_t* const __restrict__ semaphores = nullptr,
    int* const __restrict__ rank_empty_counters = nullptr,
    const int gpus_per_node = 1,
    const int sema_inter_size = 0,
    const int num_heads_per_batch = 1   // >1: num_batch = B*H, mask indexed by batch_id/num_heads_per_batch
) {
    announce_comm_cta(stream_coordinator);
    constexpr int row_per_block = num_warps * row_per_warp;     // 256 or 512 row per block (32 or 64 per warp)
    const int S = S_chunk * total_n_pes;
    const int seqlen_offset = S - S_chunk;
    const int work_per_chunk = (S_chunk + row_per_block - 1) / row_per_block;
    const int chunk_per_batch = (total_n_pes - 1) * work_per_chunk;
    // rows the first-fetched (last) work of a chunk is short of row_per_block
    const int work_pad = work_per_chunk * row_per_block - S_chunk;
    const int total_chunks = num_batch * chunk_per_batch;
    // the gathered KV (num_batch * S * S_stride) exceeds INT_MAX for large shapes,
    // so every offset built from it is 64-bit. Loop counters stay int.
    const int64_t row_stride = S_stride;
    const int64_t batch_stride = S * row_stride;
    __shared__ int cached_semaphores[kMaxCommRanks];
    __shared__ int next_work_id;
    // note that block_cnt_semaphore starts from 1. dyn-scheduling from the beginning.

    // Dual-array pointers (only meaningful when use_hierarchical)
    int64_t* const sema_inter = semaphores;
    int64_t* const sema_intra = semaphores + sema_inter_size;

    // wptr = rows gathered by the works up to the min of all CTAs' latest done work id (works of a
    // chunk run right to left, its short work first); lane j reads CTA j's slot, so the grid is 32 CTAs.
    // Slot release + lane acquires carry every CTA's data to the publisher; a stale slot only lowers wptr.
    // The min can miss the final INT_MAX (two exiting CTAs may read each other's stale slots), so
    // the last exiter, found by the acq_rel exit_count RMW, publishes INT_MAX.
    auto update_wptr_and_work_id_sync = [&](int wid) {
        if (threadIdx.x < 32) {
            bool exiting = false;                                       // owning lane only
            if (threadIdx.x == blockIdx.x) {
                int next_wid = atomicAdd(block_cnt_semaphore, 1);       // fetch and check the next work ID
                // if there is no more work to do: set the wid for the current block to be INT_MAX
                exiting = next_wid > total_chunks;
                wid = exiting ? INT_MAX : wid;
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
                cuda::atomic_ref<int, cuda::thread_scope_device> frontier(*wptr);
                frontier.fetch_max(wid == INT_MAX ? INT_MAX
                        : wid / work_per_chunk * S_chunk + max(wid % work_per_chunk * row_per_block - work_pad, 0),
                    cuda::memory_order_release);
                if (exiting && cuda::atomic_ref<int, cuda::thread_scope_device>(*exit_count)
                        .fetch_add(1, cuda::memory_order_acq_rel) == static_cast<int>(gridDim.x) - 1) {
                    frontier.fetch_max(INT_MAX, cuda::memory_order_release);
                }
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
                // acq_rel: release this CTA's SR writes; the winner acquires all before notifying peers.
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

    for (int work_id = update_wptr_and_work_id_sync(0); work_id <= total_chunks;) {
        const int work_id_m1 = work_id - 1;
        const int batch_id = work_id_m1 / chunk_per_batch;
        const int seq_work_id = (work_id_m1 % chunk_per_batch) + 1;     // this is in range [1, chunk_per_batch]
        int mask_index = 0;
        int seqlen_id = 0, remote_pe = 0, rows = row_per_block;
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
            seqlen_id = hier::hier_seqlen_id(logical_pos, row_within_chunk, row_per_block, total_n_pes, S_chunk);

            // copy_chunk_mask is in SR buffer order from seqlen 0
            mask_index = (batch_id / num_heads_per_batch) * chunk_per_batch + seqlen_id / row_per_block;

            // Source address in remote_pe's SR buffer
            const int src_chunk_seqlen = hier::hier_src_chunk_offset(
                info, total_n_pes, S_chunk, total_n_pes, gpus_per_node);
            src_addr = batch_id * batch_stride + (src_chunk_seqlen + row_within_chunk * row_per_block) * row_stride;
        } else {
            // Circular-shift traversal, right to left; work e of the batch is row row_in_chunk of SR chunk c
            const int e = chunk_per_batch - seq_work_id;
            const int c = e / work_per_chunk;
            const int row_in_chunk = e % work_per_chunk * row_per_block;
            seqlen_id = c * S_chunk + row_in_chunk;
            rows = min(row_per_block, S_chunk - row_in_chunk);
            remote_pe = my_pe - (total_n_pes - 1 - c);
            remote_pe = remote_pe >= 0 ? remote_pe : remote_pe + total_n_pes;
            target_rank = remote_pe;  // Non-hierarchical: target_rank = remote_pe
            src_addr = batch_id * batch_stride + (seqlen_offset + row_in_chunk) * row_stride;
            mask_index = chunk_per_batch * (batch_id / num_heads_per_batch) + e;
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
        // copy upto 4 heads (S_stride = H * D), and `rows` rows (seqlen axis) per CTA
        const int64_t dst_addr = batch_id * batch_stride + seqlen_id * row_stride;
        shmem::two_buffers_getmem_block(
            k_sr + dst_addr,
            shmem::offset_or_null(v_sr, dst_addr),
            k_sr + src_addr,
            shmem::offset_or_null(v_sr, src_addr),
            rows * S_stride * sizeof(T), remote_pe
        );
        // buffer getmem_block will call syncthreads(), so next_work_id will not be updated
        // therefore, next_work_id's update is visible to all threads and won't be overwritten
        // before some threads reading it. Safe!
        try_congruence_notify(target_rank, is_phase1, batch_id);
        work_id = update_wptr_and_work_id_sync(work_id);
    }
}

// Bwd remote get of one split-AG segment; the compute kernel gates each KV tile on work_done.
//
// @param num_chunks  total chunks per segment (always the real count, e.g. 4 for CP16)
// @param has_local_chunk  true iff segment 0 (the first chunk is local and should be skipped)
// @param use_hierarchical  when true, use hierarchical comm (Phase1: cross-node IB, Phase2: intra-node NVLink).
//                          In hierarchical mode the SR buffer is NOT reused across segments; each segment writes
//                          to a unique [segment_idx * num_chunks * S_chunk, (segment_idx+1) * ...) region so that
//                          later segments' Phase-2 consumers can still read Phase-1 data fetched in earlier segments.
template <typename T, int num_warps=8, int row_per_warp=32, int num_chunks=4, bool has_local_chunk=false, bool use_hierarchical=false>
__global__ void __launch_bounds__(num_warps * 32, shmem::comm_min_blocks_per_sm(num_warps)) SparseLargeKVChunkSplittedRemoteGetKernel(
    T* const __restrict__ k_sr,
    T* const __restrict__ v_sr,
    const T* const __restrict__ local_k,
    const T* const __restrict__ local_v,
    int* const __restrict__ block_work_idx,
    int* const __restrict__ block_cnt_semaphore,      // for dynamic scheduling
    int* const __restrict__ stream_coordinator,
    const int* const __restrict__ copy_chunk_mask,
    const int my_pe,
    const int start_rank,               // (my_pe + segment_idx * num_chunks) % total_n_pes
    const int segment_idx,
    const int total_n_pes,
    const int num_batch,                // B
    const int S_stride,                 // H * D
    const int S_chunk,                  // S_local
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
    // works of a chunk start at its first row; the last one may be short (see ChunkMaskKernel)
    const int work_per_chunk = (S_chunk + row_per_block - 1) / row_per_block;
    const int work_per_seg = num_chunks * work_per_chunk;
    // for each segment, get the number of work we can skip (due to being local). Note that
    // if work_to_skip is not 0, there will be some skippable works for **each batch**
    const int work_to_skip = has_local ? work_per_chunk : 0;
    constexpr int chunk_offset = has_local ? 1 : 0;
    // though the actual skippable work is `num_batch * (work_per_seg - work_to_skip)`
    // for batch_idx > 0, those skippable local works cannot be skipped **directly**.
    const int total_works = num_batch * work_per_seg;
    // Per-stage batch stride = num_chunks * S_chunk * S_stride.
    // Hierarchical: one SR region per segment; the caller advances only k_sr, local_k stays at the base.
    // Non-hierarchical: k_sr stays at the base; local_k is the base or the tail backup.
    const int64_t row_stride = S_stride;
    const int64_t batch_stride = int64_t(num_chunks) * S_chunk * row_stride;
    // Non-hierarchical remote local K/V: SR base for one segment, else the (B, S_chunk) tail backup.
    const int64_t local_batch_stride = num_segments == 1 ? batch_stride : S_chunk * row_stride;

    extern __shared__ int smem_chunk_mask[];
    __shared__ int cached_semaphores[kMaxCommRanks];
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
    //   Mark:  release store work_done[wid]=1, after all copies (bar.sync) and NIC data (wait());
    //          pairs with the compute kernel's acquire load.
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
    // Same as the forward kernel.
    auto try_congruence_notify = [&](int chunk_id, int target_rank_val, bool is_phase1_val, int batch_id) {
        if constexpr (use_hierarchical) {
            if (threadIdx.x == 0 && is_phase1_val) {
                const int counter_idx = (chunk_id - chunk_offset) + num_chunks * batch_id;
                // acq_rel: see the forward kernel.
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

        const int chunk_id = seq_work_id / work_per_chunk;
        const int row_in_chunk = seq_work_id % work_per_chunk * row_per_block;
        const int rows = min(row_per_block, S_chunk - row_in_chunk);
        // dst: segment-local row (hierarchical: k_sr is already advanced to this segment's region)
        const int seqlen_id = chunk_id * S_chunk + row_in_chunk;

        // Compute remote PE and src address (done early so masked-skip can check Phase 1)
        int remote_pe = 0;
        int64_t src_addr = 0;
        bool is_phase1 = false;
        int target_rank = 0;

        if constexpr (use_hierarchical) {
            // Hierarchical: global_logical_pos maps to target_rank via congruence-group-first order
            const int global_logical_pos = segment_idx * num_chunks + chunk_id;
            auto info = hier::hier_map_chunk(global_logical_pos, my_pe, total_n_pes, gpus_per_node);

            target_rank = info.target_rank;
            remote_pe   = info.src_pe;
            is_phase1   = info.is_phase1;

            // src: local_k is the SR base, laid out [segment][batch][chunk][row].
            //   Phase 1: target_rank's local KV (segment 0, chunk 0). Phase 2: src_pe's copy, stored
            //   at its fetch position p (segment p / num_chunks, chunk p % num_chunks).
            const hier::HierBwdChunk src = hier::hier_bwd_src_chunk(
                info, num_chunks, total_n_pes, gpus_per_node);
            src_addr = (src.segment * num_batch + batch_id) * batch_stride
                + (src.chunk * S_chunk + row_in_chunk) * row_stride;
        } else {
            // Non-hierarchical: circular shift order
            remote_pe = start_rank + chunk_id;
            remote_pe = remote_pe < total_n_pes ? remote_pe : remote_pe - total_n_pes;
            target_rank = remote_pe;
            src_addr = batch_id * local_batch_stride + row_in_chunk * row_stride;
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

        // copy upto 4 heads (S_stride = H * D), and `rows` rows (seqlen axis) per CTA
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
            rows * S_stride * sizeof(T), remote_pe
        );
        // buffer getmem_block will call syncthreads(), so next_work_id will not be updated.
        // therefore, next_work_id's update is visible to all threads and won't be overwritten
        // before some threads reading it. Safe!
        try_congruence_notify(chunk_id, target_rank, is_phase1, batch_id);
        work_id = update_wptr_and_work_id_sync(work_id);
    }
}

}   // namespace flashmask