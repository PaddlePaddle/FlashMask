// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 The FlashMask Authors. All rights reserved.

#pragma once
#include <cuda_runtime.h>
#include "sr_buffer.cuh"
#include "gin_copy_utils.cuh"
#include "rs_semaphore_ops.cuh"
#include "hierarchical_rank_map.cuh"
#include "debug_logger.cuh"

namespace flashmask {

/**
 * @brief Splitted remote put kernel for RS-overlap with P2P commit.
 *
 * @param num_chunk  total chunks per segment (always the real count, e.g. 4 for CP16)
 * @param has_local_chunk  true iff segment 0 (the first chunk is local and should be skipped)
 *
 * @param x_send & x_recv: SepSRBuffer separates send and recv into two buffers, since this is
 * actually an all-gather op implemented by an A2A op, the buffer cannot be reused the same way as the
 * all-gather overlap.
 *
 * P2P commit: the last CTA done with a rank signals that consumer at once (strong add on
 * SignalBuffer::rs_full); no quiet or group commit.
*/
template <typename T, int num_warps=8, int row_per_warp=32,
    int num_chunk=4, bool has_local_chunk=false, bool use_hierarchical=false>
__global__ void __launch_bounds__(num_warps * 32, shmem::comm_min_blocks_per_sm(num_warps)) SparseLargeKVChunkRemotePutKernel(
    const T* const __restrict__ k_send,                 // K src addr (local)
    const T* const __restrict__ v_send,                 // V src addr (local)
    T* const __restrict__ k_recv,                       // K dst addr (remote)
    T* const __restrict__ v_recv,                       // V dst addr (remote)
    int* const __restrict__ block_cnt_semaphore,        // for dynamic scheduling
    int* const __restrict__ rank_commit_counters,       // per-chunk completion counters [num_chunk]
    const int* const __restrict__ copy_chunk_mask,
    const int my_pe,
    const int start_rank,               // start rank is chunk 0 (non-hierarchical only)
    const int nranks,
    const int segment_idx,
    const int num_batch,                // B
    const int S_stride,                 // H * D
    const int S_chunk,                  // S_local
    int64_t* const __restrict__ semaphores,          // empty flags, written by the consumers
    uint64_t* const __restrict__ rs_full_seg,        // SignalBuffer::rs_full(segment_idx)
    const int num_segments = 4,
    const int gpus_per_node = 1         // hierarchical only
) {
    // segment has only a local chunk, nothing to remote-put. Return immediately.
    if constexpr (has_local_chunk && num_chunk == 1) {
        return;
    }
    static constexpr bool has_local = has_local_chunk;
    // skip the local chunk by using 1 as offset
    static constexpr int chunk_offset = has_local ? 1 : 0;
    static constexpr int remote_chunks = num_chunk - chunk_offset > 0 ? num_chunk - chunk_offset : 1;  // clamp to 1 to suppress div-by-zero warning; total_works=0 guards runtime
    static constexpr int row_per_block = num_warps * row_per_warp;     // 256 or 512 row per block (32 or 64 per warp)
    // works of a chunk start at its first row; the last one may be short (see ChunkMaskKernel)
    const int work_per_chunk = (S_chunk + row_per_block - 1) / row_per_block;
    const int work_per_seg = work_per_chunk * remote_chunks;

    const int total_works = num_batch * work_per_seg;
    const int works_per_rank = num_batch * work_per_chunk;  // total put operations per target rank
    // the segment (num_chunk * S_chunk * S_stride) exceeds INT_MAX for large shapes
    const int64_t row_stride = S_stride;
    const int64_t batch_stride = int64_t(S_chunk) * num_chunk * row_stride;   // num_chunk is the real total

    extern __shared__ int smem_chunk_mask[];
    __shared__ int cached_empty[num_chunk];
    __shared__ int next_work_id;

    // Block-stride loop: total_works grows with num_batch * S_chunk * num_chunk and can exceed
    // blockDim.x, same as the get kernel's staging loop.
    const int start_offset = chunk_offset * work_per_chunk;
    for (int i = threadIdx.x; i < total_works; i += blockDim.x) {
        const int batch_id = i / work_per_seg;
        const int seqlen_id = i % work_per_seg;
        auto* src_ptr = copy_chunk_mask + (segment_idx + batch_id * num_segments) * num_chunk * work_per_chunk + seqlen_id;
        smem_chunk_mask[start_offset + i] = *(src_ptr + start_offset);
    }
    if (threadIdx.x < num_chunk) {
        cached_empty[threadIdx.x] = 0;
    }
    __syncthreads();

    // Helper: compute the target (consumer) rank for a given seg_chunk_id.
    // Non-hierarchical: (start_rank + chunk_id) % nranks
    // Hierarchical: hier_target_rank(segment_idx * num_chunk + chunk_id, my_pe, ...)
    auto compute_target_rank = [&](int chunk_id) -> int {
        if constexpr (use_hierarchical) {
            return hier::hier_target_rank(
                segment_idx * num_chunk + chunk_id, my_pe, nranks, gpus_per_node
            );
        } else {
            return (start_rank + chunk_id) % nranks;
        }
    };

    // Thread blockIdx.x claims the CTA's next work id; __syncthreads shares it. No frontier, unlike the AG.
    auto update_work_id_sync = [&]() {
        if (threadIdx.x == blockIdx.x) {
            // Note: block_cnt_semaphore for remote_put starts from 0 (for remote get, starts from 1)
            int _work_id = atomicAdd(block_cnt_semaphore, 1);
            next_work_id = _work_id;
        }
        __syncthreads();
        return next_work_id;
    };

    // P2P commit: after completing a work, check if all works for that rank are done.
    // If so, signal that specific consumer rank.
    // target_rank is passed in to avoid recomputing it (already computed at loop top).
    auto try_commit_rank = [&](int seg_chunk_id, int target_rank) {
        if (threadIdx.x == 0) {
            // chunk_offset maps seg_chunk_id back to the counter index for remote chunks
            int counter_idx = seg_chunk_id - chunk_offset;
            // acq_rel: the winner must acquire every CTA's puts before its release add below.
            int prev = cuda::atomic_ref<int, cuda::thread_scope_device>(rank_commit_counters[counter_idx])
                .fetch_add(1, cuda::memory_order_acq_rel);
            if (prev + 1 == works_per_rank) {
                // wait before reset, in-case all chunks are skipped and we reset before remote put
                sema::rs::producer_wait_empty(semaphores, target_rank);
                // Last CTA to finish work for this rank: consume the empty flag.
                semaphores[target_rank] = 0;
                // Strong VA signal on the shared ordered context settles every
                // preceding put to target_rank (from ANY CTA, cross-node) before it lands.
                gin::remote_strong_add(rs_full_seg, 1, target_rank);
            }
        }
    };

    for (int work_id = update_work_id_sync(); work_id < total_works;) {
        const int chunk_work_id = work_id / remote_chunks;         // the work_id within the chunk
        // seg_chunk_id is also the offset to the start_rank
        const int seg_chunk_id = (work_id % remote_chunks) + chunk_offset;         // which chunk the work_id falls into
        const int target_rank = compute_target_rank(seg_chunk_id);

        // within segment seqlen work_id
        const int batch_id = chunk_work_id / work_per_chunk;
        const int seq_work_id = chunk_work_id % work_per_chunk;     // this is in range [0, work_per_chunk)

        // Note(heqianyue): this copy_chunk_mask is computed without sharding. Therefore we need to compute
        // the correct index, considering the stride of the full seqlen_k
        int mask_index = seq_work_id + work_per_chunk * (seg_chunk_id + remote_chunks * batch_id);

        if (smem_chunk_mask[mask_index]) {
            // Masked work: no put needed, but still counts toward rank completion
            try_commit_rank(seg_chunk_id, target_rank);
            __syncthreads();
            work_id = update_work_id_sync();
            continue;
        }

        if (threadIdx.x == 0) {
            // the current CTA might be scheduled to send things to different ranks
            // so each CTA should keep track of whether the target rank is empty
            if (cached_empty[seg_chunk_id] == 0) {
                sema::rs::producer_wait_empty(semaphores, target_rank);
                cached_empty[seg_chunk_id] = 1;
            }
        }

        // batch_offset + chunk_offset + work_offset, the src and dst in the buffer is the same
        const int row_in_chunk = seq_work_id * row_per_block;
        const int64_t addr = batch_id * batch_stride + (seg_chunk_id * S_chunk + row_in_chunk) * row_stride;

        shmem::two_buffers_putmem_block(
            k_recv + addr,
            shmem::offset_or_null(v_recv, addr),
            k_send + addr,
            shmem::offset_or_null(v_send, addr),
            min(row_per_block, S_chunk - row_in_chunk) * S_stride * sizeof(T), target_rank
        );
        // buffer putmem_block will call syncthreads(), so next_work_id will not be updated
        // therefore, next_work_id's update is visible to all threads and won't be overwritten
        // before some threads reading it. Safe!
        try_commit_rank(seg_chunk_id, target_rank);
        work_id = update_work_id_sync();
    }
    // PUT is nbi: drain before the kernel ends so the send buffer isn't reused
    // while the NIC still reads it. Single-node meshes have no GIN context at all
    // (host requests them only when world_size > num_lsa_ranks) and every put took
    // the LSA store path, so there is nothing to drain and nothing valid to drain on.
    if (threadIdx.x == 0 && gin::mesh_is_cross_node()) {
        gin::flush_ordered();
    }
}

}   // namespace flashmask
