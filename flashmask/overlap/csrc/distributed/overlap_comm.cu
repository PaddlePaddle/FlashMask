// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 The FlashMask Authors. All rights reserved.

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <algorithm>
#include "overlap_comm.cuh"
#include "remote_get_kernel.cuh"
#include "remote_put_kernel.cuh"
#include "dk_dv_reduce_bf16.cuh"
#include "cp_heuristic.cuh"
#include "layout_transpose.cuh"

namespace flashmask {

// allowed value: [16, 32, 64] (larger number is generally better for KV with larger num_head and higher CP)
static constexpr int RDMA_ROW_PER_WARP = 32;

// allowed value: 16 or 8 (16 warps are generally better for KV with larger num_head and higher CP)
static constexpr int num_warps = 16;    // making the grid larger is generally better
static constexpr int num_blocks = 32;   // comm CTAs per kernel; register budget: shmem::comm_regs_per_thread
static_assert(num_blocks == 32,
    "fwd frontier publish: lane j of every AG CTA reads block_work_idx[j], one slot per CTA");

// Dynamic smem of the split AG / RS put kernels: one int chunk-mask entry per
// num_warps * RDMA_ROW_PER_WARP-row work of a segment, staged at kernel start.
inline size_t split_ag_smem_bytes(int effective_batch, int nc, int s_chunk) {
    return effective_batch * sizeof(int) * nc * s_chunk / (num_warps * RDMA_ROW_PER_WARP);
}
inline size_t rs_put_smem_bytes(int b, int nc, int s_chunk) {
    return b * sizeof(int) * s_chunk * nc / (num_warps * RDMA_ROW_PER_WARP);
}

// A kernel whose static + dynamic smem exceeds the default 48 KiB limit needs the
// MaxDynamicSharedMemorySize opt-in, or its launch fails. The attribute is per kernel
// and only raised, so each instantiation pays the call once per larger shape.
inline void reserve_dynamic_smem(const void* kernel, size_t bytes, const char* name) {
    cudaFuncAttributes attr;
    cudaError_t err = cudaFuncGetAttributes(&attr, kernel);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("[FlashMask Overlap] cudaFuncGetAttributes(") + name
            + ") failed: " + cudaGetErrorString(err));
    }
    if (bytes <= static_cast<size_t>(attr.maxDynamicSharedSizeBytes)) return;
    int device = 0, optin = 0;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (attr.sharedSizeBytes + bytes > static_cast<size_t>(optin)) {
        throw std::invalid_argument(std::string("[FlashMask Overlap] ") + name + " needs "
            + std::to_string(bytes) + " B of dynamic shared memory (its chunk mask: one int per "
            + std::to_string(num_warps * RDMA_ROW_PER_WARP) + "-row work of a backward segment) plus "
            + std::to_string(attr.sharedSizeBytes) + " B static, above the device's per-block opt-in "
            "maximum of " + std::to_string(optin) + " B. Reduce the batch size, the KV heads "
            "(BHSD layout) or S_local.");
    }
    err = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(bytes));
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("[FlashMask Overlap] cudaFuncSetAttribute(") + name
            + ", " + std::to_string(bytes) + " B) failed: " + cudaGetErrorString(err));
    }
}

// A failed launch never announces its CTAs, so the compute stream's coordinator wait
// would spin until its watchdog. Report it at the launch instead.
inline void check_comm_launch(const char* name) {
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("[FlashMask Overlap] launch of ") + name
            + " failed: " + cudaGetErrorString(err));
    }
}

// ====== S_local Static Dispatch ======
// S_local (= S_chunk) must be a power of 2 and divisible by row_per_block (= num_warps * RDMA_ROW_PER_WARP = 512).
// Host-side dispatch selects the compile-time S_chunk constant for GPU kernels.
// Supported values: {4096, 8192, 16384, 32768, 65536, 131072}.
// Chunks per backward segment come from get_num_chunk_per_segment (1, 2, 4 or 8); the
// split AG/RS kernels handle num_chunks == 1 (segment 0 is then local-only).

#define SChunkCase(MACRO_FUNC, _S_chunk, ...) \
    case _S_chunk: { MACRO_FUNC(_S_chunk, ##__VA_ARGS__); break; }

#define SChunkDispatch(MACRO_FUNC, _S_local, ...)                                \
    switch (_S_local) {                                                          \
        SChunkCase(MACRO_FUNC, 4096, ##__VA_ARGS__)                              \
        SChunkCase(MACRO_FUNC, 8192, ##__VA_ARGS__)                              \
        SChunkCase(MACRO_FUNC, 16384, ##__VA_ARGS__)                             \
        SChunkCase(MACRO_FUNC, 32768, ##__VA_ARGS__)                             \
        SChunkCase(MACRO_FUNC, 65536, ##__VA_ARGS__)                             \
        SChunkCase(MACRO_FUNC, 131072, ##__VA_ARGS__)                            \
    default:                                                                     \
        throw std::invalid_argument(                                             \
            "[FlashMask Overlap] S_local must be one of {4096, 8192, 16384, 32768, 65536, 131072}, got: " \
            + std::to_string(_S_local));                                         \
    }

template <typename KVType>
OverlapCommunicator<KVType>::OverlapCommunicator(
    int b_kv,
    int s_kv,
    int h_kv,
    int d_kv,
    int rank,
    int nranks,
    const uint8_t* unique_id_ptr,
    bool kv_shared
): gin_context_(nullptr),
   dkv_buffer(nullptr),
   kv_buffer(nullptr),
   B(b_kv),
   S_local(s_kv),
   H(h_kv),
   D(d_kv),
   num_chunks(get_num_chunk_per_segment(s_kv, nranks, h_kv)),
   _kv_shared(kv_shared),
   _kv_alloc_v(!kv_shared),
   _sema_count(0),
   block_work_ids(nullptr),
   block_cnt_semaphore(nullptr),
   copy_chunk_mask(nullptr),
   put_streams(nullptr),
   bwd_done_events(nullptr),
   rs_block_cnt(nullptr),
   _num_put_streams(0)
{
    gin_context_ = gin::create_context(unique_id_ptr, rank, nranks);
    _my_pe = gin::rank(*gin_context_);
    _total_n_pes = gin::nranks(*gin_context_);
    WARN_PRINT("[FlashMask Overlap] NCCL GIN initialized. Rank: %d / %d\n", rank, nranks);

    // Hierarchical overlap topology discovery comes from the NCCL LSA team.
    _gpus_per_node = gin::num_lsa_ranks(*gin_context_);
    _num_nodes = _total_n_pes / _gpus_per_node;
    _use_hierarchical = hier::hier_is_effective(_total_n_pes, _gpus_per_node);
    // Set before any sizing: num_effective_batch() reads use_bhsd.
    _config = {b_kv, s_kv, h_kv, d_kv, nranks, OverlapConfig::bhsd_for(d_kv)};
    reserve_comm_kernel_smem(S_local, num_chunks, num_effective_batch(), B);

    int least_priority, greatest_priority;
    cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority);
    cudaStreamCreateWithPriority(&comm_stream, 
                                  cudaStreamNonBlocking,
                                  greatest_priority);
    cudaEventCreateWithFlags(&wptr_init, cudaEventDisableTiming);
    cudaEventCreateWithFlags(&sr_usable, cudaEventDisableTiming);
    cudaEventCreateWithFlags(&ag_done, cudaEventDisableTiming);
    cudaEventRecord(sr_usable, comm_stream);                    // set initial status for SR buffer
    cudaEventRecord(ag_done, comm_stream);                      // initial state: no AG pending
    _local_batch_stride = static_cast<size_t>(s_kv) * h_kv * d_kv;
    _total_numel = _local_batch_stride * b_kv * nranks;

    _sema_count = sema::ag::semaphore_count(
        _use_hierarchical, _num_nodes, _total_n_pes, num_effective_batch());
    _sema_capacity = _sema_count;
    _sema_inter_size = _use_hierarchical ? _num_nodes : 0;

    sig_buffer = std::make_unique<SignalBuffer>(*gin_context_, _total_n_pes);
    kv_buffer = std::make_unique<SRBuffer<KVType>>(
        *gin_context_, _total_numel, _sema_capacity, kv_components());
    size_block_work_ids();
    // RS reduce stream
    cudaStreamCreateWithPriority(&aux_c_stream, cudaStreamNonBlocking, std::min(greatest_priority + 1, least_priority));
    cudaEventCreateWithFlags(&reduce_done, cudaEventDisableTiming);
    cudaEventCreateWithFlags(&local_moved, cudaEventDisableTiming);
    // setup_dkv_buffer runs the first team barrier: every PE's windows are zero before any
    // PE primes the RS empty flags.
    setup_dkv_buffer();
    if (rank == 0) {
        printf("[FlashMask Overlap] hierarchical=%d, bhsd=%d, num_segments=%d, num_chunks=%d\n",
               int(_use_hierarchical), int(_config.use_bhsd), _num_put_streams, num_chunks);
    }
    _sr_buffer_capacity = _total_numel;
    // copy to the last position of the SR buffer
    WARN_PRINT("SR buffer valid: %d, B, S, H, D: %d, %d, %d, %d, nranks: %d\n", int(kv_buffer->is_valid()), B, S_local, H, D, nranks);
    WARN_PRINT("[FlashMask Overlap] constructor rank: %d, nranks: %d\n", rank, nranks);
}

template <typename KVType>
OverlapCommunicator<KVType>::~OverlapCommunicator() {
    CUDA_DEBUG_CHECK(cudaDeviceSynchronize());
    CUDA_DEBUG_CHECK(cudaFree(block_work_ids));
    CUDA_DEBUG_CHECK(cudaDeviceSynchronize());
    CUDA_DEBUG_CHECK(cudaEventDestroy(wptr_init));
    CUDA_DEBUG_CHECK(cudaEventDestroy(sr_usable));
    CUDA_DEBUG_CHECK(cudaEventDestroy(ag_done));
    CUDA_DEBUG_CHECK(cudaStreamDestroy(comm_stream));
    CUDA_DEBUG_CHECK(cudaEventDestroy(reduce_done));
    CUDA_DEBUG_CHECK(cudaEventDestroy(local_moved));
    CUDA_DEBUG_CHECK(cudaStreamDestroy(aux_c_stream));
    for (int i = 0; i < _num_put_streams; i++) {
        CUDA_DEBUG_CHECK(cudaStreamDestroy(put_streams[i]));
        CUDA_DEBUG_CHECK(cudaEventDestroy(bwd_done_events[i]));
    }
    delete[] put_streams;
    delete[] bwd_done_events;
    CUDA_DEBUG_CHECK(cudaFree(rs_block_cnt));
    dkv_buffer.reset();
    kv_buffer.reset();
    sig_buffer.reset();
    WARN_PRINT("[FlashMask Overlap] Finalizing NCCL GIN...\n");
    gin_context_.reset();
}

template <typename KVType>
void OverlapCommunicator<KVType>::wait_wptr_init() {
    cudaStreamWaitEvent(comm_stream, wptr_init);
}

// Every comm CTA release-ORs bit blockIdx.x at kernel start; spins until all num_blocks
// bits are set. The acquire makes comm_stream work ordered before the AG kernel (local
// chunk copy/transpose, work_done memset, InitBitmapForLocalSkip) visible to the
// compute kernel launched after this one; no event carries that edge.
__global__ void WaitAndResetStreamCoordKernel(
    int* const stream_coordinator,
    const unsigned full_mask
) {
    cuda::atomic_ref<int, cuda::thread_scope_device> coordinator(*stream_coordinator);
    gin::SpinWatchdog watchdog;
    for (int seen; static_cast<unsigned>(seen = coordinator.load(cuda::memory_order_acquire)) != full_mask;) {
        __nanosleep(10);
        watchdog.tick("stream_coordinator", stream_coordinator, full_mask, static_cast<unsigned>(seen));
    }
    *stream_coordinator = 0;
    __threadfence();
}

template <typename KVType>
void OverlapCommunicator<KVType>::wait_reset_stream_coordinator(cudaStream_t stream) {
    constexpr unsigned full_mask = ~0u >> (32 - num_blocks);
    WaitAndResetStreamCoordKernel<<<1, 1, 0, stream>>>(stream_coordinator, full_mask);
}

template <typename KVType>
void OverlapCommunicator<KVType>::wait_sr_buffer_empty(cudaStream_t stream) {
    // TBH, since there is enough time between two attention call,
    // for an unsafe impl, we actually don't need to wait.
    // This would save some time, but not entirely safe if
    // the attention's workload is too too small. Yet currently
    // we haven't trigger unsafe problems even once
    cudaEventRecord(sr_usable, stream);             // in-case there is previously unfinished comp stream ops
    cudaStreamWaitEvent(comm_stream, sr_usable);
    WARN_PRINT("Before wait_self_empty\n");
    // block if computation that uses the local chunk is not finished
    // so that we won't corrupt the local buffer being used
    if (_use_hierarchical) {
        sema::ag::wait_self_empty_hier(
            sig_buffer->ag_refcnt(),
            sig_buffer->ag_issued(),
            _my_pe,
            _gpus_per_node,
            _num_nodes,
            comm_stream
        );
    } else {
        sema::ag::wait_self_empty(
            sig_buffer->ag_refcnt(),
            sig_buffer->ag_issued(),
            _my_pe,
            comm_stream
        );
    }
    WARN_PRINT_SYNC(comm_stream, "After wait_self_empty\n");
}

template <typename KVType>
void OverlapCommunicator<KVType>::update_kv_buffer(
    const KVType* const new_k_data,
    const KVType* const new_v_data,
    const bool fwd
) {
    // remember to pair `update_kv_buffer` with `wait_sr_buffer_empty` (except from the constructor call)
    WARN_PRINT("Before cudaMemcpyAsync... is fwd: %d\n", int(fwd));
    // bwd copies the data to the start chunk of the SR, while fwd copies to the last chunk
    const size_t local_offset = fwd ? (_local_batch_stride * (_total_n_pes - 1)) : 0;
    // for the bwd split AG, the batch stride is num_chunks * S_local * S_stride
    size_t batch_stride = _local_batch_stride * _total_n_pes;
    if (!fwd) {
        batch_stride = _local_batch_stride * num_chunks;
        // Non-hierarchical BWD: save local KV at the SR tail so remote ranks can Phase-1 fetch it.
        // Hierarchical BWD: local KV lives at segment-0 offset-0 (SR base) and is never overwritten,
        // so no tail backup is needed.
        if (!_use_hierarchical) {
            if (_config.use_bhsd) {
                // BHSD: transpose local KV into the tail backup region as (B,H,S_local,D)
                layout::transpose_copy_to_sr(
                    reinterpret_cast<const __nv_bfloat16*>(new_k_data),
                    reinterpret_cast<__nv_bfloat16*>(local_k_data()),
                    B, S_local, H, D, S_local, 0, comm_stream);
                if (!_kv_shared) {
                    layout::transpose_copy_to_sr(
                        reinterpret_cast<const __nv_bfloat16*>(new_v_data),
                        reinterpret_cast<__nv_bfloat16*>(local_v_data()),
                        B, S_local, H, D, S_local, 0, comm_stream);
                }
            } else {
                const size_t copy_bytes = B * _local_batch_stride * sizeof(KVType);
                CUDA_DEBUG_CHECK(cudaMemcpyAsync(local_k_data(), new_k_data, copy_bytes,
                                        cudaMemcpyDeviceToDevice, comm_stream));
                if (!_kv_shared) {
                    CUDA_DEBUG_CHECK(cudaMemcpyAsync(local_v_data(), new_v_data, copy_bytes,
                                            cudaMemcpyDeviceToDevice, comm_stream));
                }
            }
        }
    }
    if (_config.use_bhsd) {
        // BHSD path: transpose-copy BSHD src → BHSD SR buffer at correct S offset
        const int S_dst = fwd ? (S_local * _total_n_pes) : (S_local * num_chunks);
        const int s_offset = fwd ? (S_dst - S_local) : 0;
        layout::transpose_copy_to_sr(
            reinterpret_cast<const __nv_bfloat16*>(new_k_data),
            reinterpret_cast<__nv_bfloat16*>(kv_buffer->k_data()),
            B, S_local, H, D, S_dst, s_offset, comm_stream);
        if (!_kv_shared) {
            layout::transpose_copy_to_sr(
                reinterpret_cast<const __nv_bfloat16*>(new_v_data),
                reinterpret_cast<__nv_bfloat16*>(kv_buffer->v_data()),
                B, S_local, H, D, S_dst, s_offset, comm_stream);
        }
    } else {
        // BSHD path: per-batch flat memcpy into SR buffer
        for (int bid = 0; bid < B; bid++) {
            CUDA_DEBUG_CHECK(cudaMemcpyAsync(kv_buffer->k_data() + local_offset + bid * batch_stride,
                                    new_k_data + bid * _local_batch_stride,
                                    _local_batch_stride * sizeof(KVType),
                                    cudaMemcpyDeviceToDevice,
                                    comm_stream));
            if (_kv_shared) continue;
            CUDA_DEBUG_CHECK(cudaMemcpyAsync(kv_buffer->v_data() + local_offset + bid * batch_stride,
                                    new_v_data + bid * _local_batch_stride,
                                    _local_batch_stride * sizeof(KVType),
                                    cudaMemcpyDeviceToDevice,
                                    comm_stream));
        }
    }
    // notify all other PEs that the local data is ready (1. book the expected consumer adds. 2. broadcast to other PEs)
    const int effective_batch = num_effective_batch();
    if (_use_hierarchical) {
        sema::ag::notify_full_hier(
            sema_inter(),
            sema_intra(),
            sig_buffer->ag_issued(),
            _my_pe, _total_n_pes,
            _gpus_per_node, _num_nodes,
            effective_batch, comm_stream
        );
    } else {
        sema::ag::notify_full(
            kv_buffer->semaphores(),
            sig_buffer->ag_issued(),
            _my_pe, _total_n_pes,
            comm_stream
        );
    }
    WARN_PRINT("After cudaMemcpyAsync and notify full\n");
}

// Guard: if constexpr (_S > S_chunk) prevents template instantiation for degenerate
// S==S_chunk (nranks=1) combinations generated by SChunkDispatch × SeqlenDispatch.
// These are unreachable at runtime (overlap requires nranks >= 2, so S_full >= 2*S_chunk).
// Eliminating them saves ~40% of kernel binary from combinatorial expansion.

#define LaunchSparseKernel(_S, _hier)                                                 \
    SparseLargeKVChunkRemoteGetKernel<KVType, _S, S_chunk,                            \
        num_warps, RDMA_ROW_PER_WARP, _hier>                                          \
        <<<num_blocks, num_warps * 32, 0, comm_stream>>>(                             \
                        kv_buffer->k_data(),                                          \
                        _kv_shared ? nullptr : kv_buffer->v_data(),                   \
                        fwd_write_ptr,                                                \
                        block_work_ids,                                               \
                        block_cnt_semaphore,                                          \
                        fwd_exit_count,                                               \
                        stream_coordinator,                                           \
                        copy_chunk_mask,                                              \
                        _my_pe,                                                       \
                        _total_n_pes,                                                 \
                        effective_batch,                                              \
                        effective_s_stride,                                           \
                        kv_buffer->semaphores(),                                      \
                        rank_empty_counters,                                          \
                        _gpus_per_node,                                               \
                        _sema_inter_size,                                             \
                        effective_num_heads                                           \
                    );                                                                \
    check_comm_launch("SparseLargeKVChunkRemoteGetKernel")

#define SparseLargeChunkKernel(_S)                                                  \
    if constexpr ((_S) > S_chunk) {                                                 \
        if (_use_hierarchical) {                                                    \
            LaunchSparseKernel(_S, true);                                           \
        } else {                                                                    \
            LaunchSparseKernel(_S, false);                                          \
        }                                                                           \
    }

#define SeqlenCase(MACRO_FUNC, _S)                          \
    case _S: {                                              \
        MACRO_FUNC(_S); break;                              \
    }

#define SeqlenDispatch(MACRO_FUNC, _S)                      \
    switch (_S) {                                           \
        SeqlenCase(MACRO_FUNC, 8192)                        \
        SeqlenCase(MACRO_FUNC, 16384)                       \
        SeqlenCase(MACRO_FUNC, 32768)                       \
        SeqlenCase(MACRO_FUNC, 65536)                       \
        SeqlenCase(MACRO_FUNC, 131072)                      \
        SeqlenCase(MACRO_FUNC, 262144)                      \
        SeqlenCase(MACRO_FUNC, 524288)                      \
        SeqlenCase(MACRO_FUNC, 1048576)                     \
        SeqlenCase(MACRO_FUNC, 2097152)                     \
    default:                                                \
        throw std::invalid_argument(                        \
            "[FlashMask Overlap] Full seqlen (S_local * nranks) must be one of " \
            "{8192..2097152}, got: " + std::to_string(_S)); \
    }

template <typename KVType>
void OverlapCommunicator<KVType>::compute_chunk_mask(
    const int* const mask,
    const int num_heads,
    const int64_t batch_stride,
    const int64_t head_stride,
    cudaStream_t stream,
    const bool fwd
) {
    if (mask == nullptr || num_heads <= 0) {
        throw std::invalid_argument("[FlashMask Overlap] compute_chunk_mask needs a (B, H_mask, S, 2) mask.");
    }
    // copy_chunk_mask is one shared buffer: the previous pass's AG kernels and RS put
    // kernels stage it at kernel start, so do not overwrite it before they finished.
    ensure_ag_done(stream);
    dkv_buffer->wait_all_send_slots(stream);
    constexpr int num_reduce_warp = RDMA_ROW_PER_WARP == 16 ? 4 : num_warps;
    // fwd checks every non-local SR row (the local chunk is last); bwd checks all rows.
    const int checked_rows = S_local * _total_n_pes - (fwd ? S_local : 0);
    const dim3 grid(checked_rows / (num_reduce_warp * 32 * 4), B, 1);
    WARN_PRINT("(%d) Before compute_chunk_mask (fwd: %d).\n", _my_pe, int(fwd));
    BlockSparsityCheckSpecializedKernel<num_reduce_warp, RDMA_ROW_PER_WARP>
        <<<grid, num_reduce_warp * 32, 0, stream>>>(mask, copy_chunk_mask, num_heads, batch_stride, head_stride);
    WARN_PRINT_SYNC(stream, "(%d) After compute_chunk_mask (fwd: %d).\n", _my_pe, int(fwd));
}

template <typename KVType>
int OverlapCommunicator<KVType>::chunk_per_seg() const {
    return num_chunks;
}

template <typename KVType>
void OverlapCommunicator<KVType>::run_overlap_ag_kernel() {
    // BHSD layout: treat data as (B*H, S, D) — each (b,h) pair is an independent batch with S_stride=D.
    const int effective_batch = num_effective_batch();
    const int effective_s_stride = _config.use_bhsd ? D : H * D;
    const int effective_num_heads = _config.use_bhsd ? H : 1;
    // set 0 every time we start the comm kernel --- meaning that we don't have any available attn-blocks
    cudaMemsetAsync(block_work_ids, 0, sizeof(int) * num_blocks, comm_stream);
    // Hierarchical: per-batch Phase 1 counters
    if (_use_hierarchical) {
        cudaMemsetAsync(rank_empty_counters, 0, sizeof(int) * effective_batch * _total_n_pes, comm_stream);
    }
    WARN_PRINT_SYNC(comm_stream, "Before remote get kernel: %d\n", _my_pe);

#define AgKernelBody(_S_chunk)                                                       \
    do {                                                                             \
        constexpr int S_chunk = _S_chunk;                                            \
        const int _S_full = S_local * _total_n_pes;                                  \
        SeqlenDispatch(SparseLargeChunkKernel, _S_full);                             \
    } while(0)

    SChunkDispatch(AgKernelBody, S_local);
#undef AgKernelBody

    // Post-AG cleanup: notify all remote producers that we finished consuming their data.
    // This replaces the old inline try_release_rank in the get kernel.
    if (_use_hierarchical) {
        sema::ag::notify_all_empty_hier(
            sema_inter(), sema_intra(), sig_buffer->ag_refcnt(),
            _my_pe, _total_n_pes,
            _gpus_per_node, _num_nodes,
            effective_batch,
            comm_stream
        );
    } else {
        sema::ag::notify_all_empty(
            kv_buffer->semaphores(),
            sig_buffer->ag_refcnt(),
            _my_pe, _total_n_pes,
            comm_stream
        );
    }
    cudaEventRecord(ag_done, comm_stream);
    WARN_PRINT_SYNC(comm_stream, "After remote_get kernel\n");
}
#undef SeqlenDispatch
#undef SeqlenCase
#undef SparseLargeChunkKernel
#undef LaunchSparseKernel

#define SparseLargeChunkSplittedKernel(num_chunk, _has_local, _use_hier, start_rank, seg_idx, num_segs, smem_bytes) \
    SparseLargeKVChunkSplittedRemoteGetKernel<KVType, S_chunk,                                  \
        num_warps, RDMA_ROW_PER_WARP, num_chunk, _has_local, _use_hier>                         \
        <<<num_blocks, num_warps * 32, smem_bytes, comm_stream>>>(                              \
                        _k_sr_ptr,                                                              \
                        _v_sr_ptr,                                                              \
                        _local_k_ptr,                                                           \
                        _local_v_ptr,                                                           \
                        block_work_ids,                                                         \
                        block_cnt_semaphore,                                                    \
                        stream_coordinator,                                                     \
                        copy_chunk_mask,                                                        \
                        _my_pe,                                                                 \
                        start_rank,                                                             \
                        seg_idx,                                                                \
                        _total_n_pes,                                                           \
                        effective_batch,                                                        \
                        effective_s_stride,                                                     \
                        num_segs,                                                               \
                        kv_buffer->semaphores(),                                                \
                        rank_empty_counters,                                                    \
                        _gpus_per_node,                                                         \
                        _sema_inter_size,                                                       \
                        effective_num_heads                                                     \
                    );                                                                          \
    check_comm_launch("SparseLargeKVChunkSplittedRemoteGetKernel")

#define NumChunkDispatchSplitted(MACRO_FUNC, num_chunk, _has_local, ...)  \
    switch (num_chunk) {                                                  \
        case 4: { MACRO_FUNC(4, _has_local, __VA_ARGS__); break; }        \
        case 2: { MACRO_FUNC(2, _has_local, __VA_ARGS__); break; }        \
        case 8: { MACRO_FUNC(8, _has_local, __VA_ARGS__); break; }        \
        case 1: { MACRO_FUNC(1, _has_local, __VA_ARGS__); break; }        \
    default:                                                              \
        throw std::invalid_argument(                                      \
            "[FlashMask Overlap] Num chunk per segment must be one of "   \
            "{1, 2, 4, 8}, got: " + std::to_string(num_chunk));           \
    }

template <typename KVType>
void OverlapCommunicator<KVType>::run_overlap_splitted_ag_kernel(int segment_idx) {
    WARN_PRINT("(%d) Before run_overlap_splitted_ag_kernel, segment: %d\n", _my_pe, segment_idx);
    const int start_pe = (_my_pe + segment_idx * num_chunks) % _total_n_pes;
    const int num_segs = _total_n_pes / num_chunks;
    // BHSD: effective batch = B*H, stride between S rows = D
    const int effective_batch = num_effective_batch();
    const int effective_s_stride = _config.use_bhsd ? D : H * D;
    const int effective_num_heads = _config.use_bhsd ? H : 1;

#define SplittedAgBody(_S_chunk)                                                                              \
    do {                                                                                                      \
        constexpr int S_chunk = _S_chunk;                                                                     \
        const size_t mask_smem_bytes = split_ag_smem_bytes(effective_batch, num_chunks, S_chunk);             \
        /* Hierarchical: per-(effective_batch) Phase 1 counters */                                            \
        /* Non-hierarchical: no counters needed (release by post-AG kernel) */                                \
        if (_use_hierarchical) {                                                                              \
            cudaMemsetAsync(rank_empty_counters, 0, sizeof(int) * effective_batch * num_chunks, comm_stream); \
        }                                                                                                     \
        cudaMemsetAsync(block_work_ids, 0, sizeof(int) * _bitmap_region_size, comm_stream);                   \
        /* Compute stage-advanced dst (k_sr) and base src (local_k) pointers. */                              \
        /* Hierarchical: SR buffer is num_segments sub-tensors; each stage uses its own region. */            \
        /* Non-hierarchical: circular-reuse; k_sr stays at base, local_k at tail backup. */                   \
        const size_t _stage_elems = (size_t)num_chunks * S_chunk * H * D;                                     \
        KVType* const _k_sr_ptr = _use_hierarchical                                                           \
            ? kv_buffer->k_data() + (size_t)segment_idx * B * _stage_elems                                    \
            : kv_buffer->k_data();                                                                            \
        KVType* const _v_sr_ptr = _kv_shared ? nullptr : (_use_hierarchical                                  \
            ? kv_buffer->v_data() + (size_t)segment_idx * B * _stage_elems                                    \
            : kv_buffer->v_data());                                                                           \
        const KVType* const _local_k_ptr = _use_hierarchical ? kv_buffer->k_data() : local_k_data();          \
        const KVType* const _local_v_ptr = _kv_shared ? nullptr : (_use_hierarchical                          \
            ? kv_buffer->v_data() : local_v_data());                                                          \
        if (segment_idx) {                                                                                    \
            if (_use_hierarchical) {                                                                          \
                NumChunkDispatchSplitted(SparseLargeChunkSplittedKernel, num_chunks, false, true,             \
                    start_pe, segment_idx, num_segs, mask_smem_bytes);                                        \
            } else {                                                                                          \
                NumChunkDispatchSplitted(SparseLargeChunkSplittedKernel, num_chunks, false, false,            \
                    start_pe, segment_idx, num_segs, mask_smem_bytes);                                        \
            }                                                                                                 \
        } else {                                                                                              \
            constexpr int work_to_skip = S_chunk / (num_warps * RDMA_ROW_PER_WARP);                           \
            const int work_per_seg = num_chunks * work_to_skip;                                             \
            const int total_fill = effective_batch * work_to_skip;                                          \
            InitBitmapForLocalSkip<<<1, std::min(std::max(total_fill, 1), 256), 0, comm_stream>>>(         \
                block_work_ids, work_to_skip, work_per_seg, effective_batch);                                            \
            if (_use_hierarchical) {                                                                          \
                NumChunkDispatchSplitted(SparseLargeChunkSplittedKernel, num_chunks, true, true,              \
                    start_pe, segment_idx, num_segs, mask_smem_bytes);                                        \
            } else {                                                                                          \
                NumChunkDispatchSplitted(SparseLargeChunkSplittedKernel, num_chunks, true, false,             \
                    start_pe, segment_idx, num_segs, mask_smem_bytes);                                        \
            }                                                                                                 \
        }                                                                                                     \
    } while(0)

    SChunkDispatch(SplittedAgBody, S_local);
#undef SplittedAgBody

    // Post-AG cleanup: notify producers we finished consuming their data in this segment.
    if (_use_hierarchical) {
        sema::ag::notify_segment_empty_hier(
            sema_inter(), sema_intra(), sig_buffer->ag_refcnt(),
            _my_pe, segment_idx, num_chunks,
            _total_n_pes, _gpus_per_node,
            effective_batch,
            comm_stream
        );
    } else {
        sema::ag::notify_segment_empty(
            kv_buffer->semaphores(),
            sig_buffer->ag_refcnt(),
            _my_pe, start_pe, num_chunks,
            _total_n_pes,
            comm_stream
        );
    }
    cudaEventRecord(ag_done, comm_stream);
    WARN_PRINT_SYNC(comm_stream, "(%d) After run_overlap_splitted_ag_kernel, segment: %d\n", _my_pe, segment_idx);
}
#undef SparseLargeChunkSplittedKernel

#define SegmentIdxPutKernelDispatch(_num_chunks, _has_local, _use_hier, seg_idx)     \
SparseLargeKVChunkRemotePutKernel<KVType, S_chunk, num_warps,                        \
            RDMA_ROW_PER_WARP, _num_chunks, _has_local, _use_hier>                   \
            <<<num_blocks, num_warps * 32, dynamic_smem, p_stream>>>(               \
        dkv_buffer->k_send(seg_idx),                                                \
        _kv_shared ? nullptr : dkv_buffer->v_send(seg_idx),                         \
        dkv_buffer->k_recv(seg_idx),                                                \
        _kv_shared ? nullptr : dkv_buffer->v_recv(seg_idx),                         \
        rs_cnt,                                                                     \
        commit_counters,                                                            \
        copy_chunk_mask,                                                            \
        _my_pe,                                                                     \
        remote_consumer_start_rank,                                                 \
        _total_n_pes,                                                               \
        seg_idx,                                                                    \
        B,                                                                          \
        H * D,                                                                      \
        dkv_buffer->semaphores(seg_idx),                                            \
        sig_buffer->rs_full(seg_idx),                                               \
        num_segments(),                                                             \
        _gpus_per_node                                                              \
    );                                                                              \
    check_comm_launch("SparseLargeKVChunkRemotePutKernel")

template <typename KVType>
void OverlapCommunicator<KVType>::run_overlap_rs_kernel(
    KVType* const dk_accum,
    KVType* const dv_accum,
    int segment_idx,
    cudaStream_t comp_stream
) {
    WARN_PRINT("(%d) Before run_overlap_rs_kernel, seg: %d\n", _my_pe, segment_idx);
    cudaStream_t p_stream = put_streams[segment_idx];
    int* const rs_cnt = rs_block_cnt + segment_idx;
    int* const commit_counters = rank_commit_counters + segment_idx * num_chunks;
    cudaEvent_t done_event = bwd_done_events[segment_idx];

    // step 1 (pre-process and prepare) comp_stream should notify p_stream, post-process is done.
    // also, reset the block_cnt_semaphore for remote_put dynamic scheduling
    cudaMemsetAsync(rs_cnt, 0, sizeof(int), p_stream);
    cudaMemsetAsync(commit_counters, 0, sizeof(int) * num_chunks, p_stream);
    cudaEventRecord(done_event, comp_stream);
    cudaStreamWaitEvent(p_stream, done_event);
    cudaStreamWaitEvent(aux_c_stream, done_event);

    const int remote_consumer_start_rank = (_my_pe + num_chunks * segment_idx) % _total_n_pes;

    // step 2. local producer (put) wait empty (in the kernel) and start remote_put
#define RsKernelBody(_S_chunk)                                                                              \
    do {                                                                                                    \
        constexpr int S_chunk = _S_chunk;                                                                   \
        const size_t dynamic_smem = rs_put_smem_bytes(B, num_chunks, S_chunk);                              \
        if (segment_idx) {                                                                                  \
            if (_use_hierarchical) {                                                                        \
                NumChunkDispatchSplitted(SegmentIdxPutKernelDispatch, num_chunks, false, true, segment_idx);\
            } else {                                                                                        \
                NumChunkDispatchSplitted(SegmentIdxPutKernelDispatch, num_chunks, false, false, segment_idx); \
            }                                                                                           \
        } else {                                                                                        \
            const int S_stride = H * D;                                                                 \
            const size_t batch_stride = static_cast<size_t>(num_chunks) * S_chunk * S_stride;           \
            KVType* const dk_dst = dkv_buffer->k_recv(0);                                               \
            const KVType* const dk_src = dkv_buffer->k_send(0);                                         \
            KVType* const dv_dst = _kv_shared ? nullptr : dkv_buffer->v_recv(0);                        \
            const KVType* const dv_src = _kv_shared ? nullptr : dkv_buffer->v_send(0);                  \
            for (int bid = 0; bid < B; bid ++) {                                                        \
                const size_t batch_offset = bid * batch_stride;                                         \
                cudaMemcpyAsync(dk_dst + batch_offset, dk_src + batch_offset,                           \
                    sizeof(KVType) * S_chunk * S_stride, cudaMemcpyDeviceToDevice, p_stream);           \
                if (_kv_shared) continue;                                                               \
                cudaMemcpyAsync(dv_dst + batch_offset, dv_src + batch_offset,                           \
                    sizeof(KVType) * S_chunk * S_stride, cudaMemcpyDeviceToDevice, p_stream);           \
            }                                                                                           \
            cudaEventRecord(local_moved, p_stream);                                                     \
            cudaStreamWaitEvent(aux_c_stream, local_moved);                                             \
            if (_use_hierarchical) {                                                                         \
                NumChunkDispatchSplitted(SegmentIdxPutKernelDispatch, num_chunks, true, true, segment_idx);  \
            } else {                                                                                         \
                NumChunkDispatchSplitted(SegmentIdxPutKernelDispatch, num_chunks, true, false, segment_idx); \
            }                                                                                           \
        }                                                                                               \
    } while(0)

    SChunkDispatch(RsKernelBody, S_local);
#undef RsKernelBody

    dkv_buffer->release_buffer(segment_idx, p_stream);
    // consumer wait full + reduce
    sema::rs::consumer_wait_full(
        sig_buffer->rs_full(segment_idx),
        sig_buffer->rs_expected(segment_idx),
        segment_idx == 0 ? (num_chunks - 1) : num_chunks, aux_c_stream
    );
    // ns >= 2 always (nc <= nranks / 2). recv(0), plus recv(1) when nc == 1, holds the
    // fp32 running sum (see ReducedKdVKernel) from the segment-1 reduce to the last one,
    // so those slots are recycled only after the last reduce; segment 0 reduces nothing.
    const int num_home_slots = num_chunks == 1 ? 2 : 1;
    if (segment_idx) {
        auto operand = [&](KVType* slot0, KVType* slot1, KVType* recv, KVType* out) {
            return ReduceOperand{slot0, num_chunks == 1 ? slot1 : slot0 + _local_batch_stride, recv, out};
        };
        const ReduceOperand dk = operand(
            dkv_buffer->k_recv(0), dkv_buffer->k_recv(1), dkv_buffer->k_recv(segment_idx), dk_accum);
        const ReduceOperand dv = _kv_shared ? ReduceOperand{} : operand(
            dkv_buffer->v_recv(0), dkv_buffer->v_recv(1), dkv_buffer->v_recv(segment_idx), dv_accum);
        launch_dk_dv_reduce(
            dk, dv,
            B, S_local, H, D, num_chunks,
            segment_idx, num_segments(),
            aux_c_stream
        );
    }
    // Notify only after the reduce consumed the slot: the producer_wait_empty latency
    // then hides behind the whole inter-BWD interval.
    if (segment_idx >= num_home_slots) recycle_recv_slot(segment_idx);
    if (segment_idx == num_segments() - 1) {
        for (int seg = 0; seg < num_home_slots; seg++) recycle_recv_slot(seg);
    }

    WARN_PRINT_SYNC(aux_c_stream, "(%d) After run_overlap_rs_kernel, segment: %d\n", _my_pe, segment_idx);
}

template <typename KVType>
void OverlapCommunicator<KVType>::recycle_recv_slot(int segment_idx) {
    dkv_buffer->zero_recv_buf(segment_idx, B * _local_batch_stride, !_kv_shared, aux_c_stream);
    notify_recv_empty(segment_idx);
}

template <typename KVType>
void OverlapCommunicator<KVType>::notify_recv_empty(int segment_idx) {
    if (_use_hierarchical) {
        sema::rs::notify_consumer_empty_hier(
            dkv_buffer->semaphores(segment_idx), segment_idx, num_chunks,
            _total_n_pes, _gpus_per_node, _my_pe, aux_c_stream);
    } else {
        const int remote_producer_end_rank = (_my_pe - num_chunks * segment_idx + _total_n_pes) % _total_n_pes;
        sema::rs::notify_consumer_empty(
            dkv_buffer->semaphores(segment_idx),
            remote_producer_end_rank, num_chunks, _total_n_pes, _my_pe, aux_c_stream);
    }
}

// Both segment variants (segment 0 with the local chunk, later segments) launch with the
// same dynamic smem, so both get the opt-in. Runs at construction and on reconfiguration,
// before the shape is committed, so an impossible shape fails here instead of at launch.
template <typename KVType>
void OverlapCommunicator<KVType>::reserve_comm_kernel_smem(
    int s_local, int nc, int effective_batch, int b
) const {
#define ReserveSegmentKernelSmem(num_chunk, _has_local, _use_hier)                                  \
    reserve_dynamic_smem(reinterpret_cast<const void*>(&SparseLargeKVChunkSplittedRemoteGetKernel<  \
            KVType, S_chunk, num_warps, RDMA_ROW_PER_WARP, num_chunk, _has_local, _use_hier>),      \
        ag_bytes, "SparseLargeKVChunkSplittedRemoteGetKernel");                                     \
    reserve_dynamic_smem(reinterpret_cast<const void*>(&SparseLargeKVChunkRemotePutKernel<          \
            KVType, S_chunk, num_warps, RDMA_ROW_PER_WARP, num_chunk, _has_local, _use_hier>),      \
        rs_bytes, "SparseLargeKVChunkRemotePutKernel")

#define ReserveSmemBody(_S_chunk)                                                                   \
    do {                                                                                            \
        constexpr int S_chunk = _S_chunk;                                                           \
        const size_t ag_bytes = split_ag_smem_bytes(effective_batch, nc, S_chunk);                  \
        const size_t rs_bytes = rs_put_smem_bytes(b, nc, S_chunk);                                  \
        if (_use_hierarchical) {                                                                    \
            NumChunkDispatchSplitted(ReserveSegmentKernelSmem, nc, true, true);                     \
            NumChunkDispatchSplitted(ReserveSegmentKernelSmem, nc, false, true);                    \
        } else {                                                                                    \
            NumChunkDispatchSplitted(ReserveSegmentKernelSmem, nc, true, false);                    \
            NumChunkDispatchSplitted(ReserveSegmentKernelSmem, nc, false, false);                   \
        }                                                                                           \
    } while (0)

    SChunkDispatch(ReserveSmemBody, s_local);
#undef ReserveSmemBody
#undef ReserveSegmentKernelSmem
}

#undef SegmentIdxPutKernelDispatch
#undef NumChunkDispatchSplitted

template <typename KVType>
int OverlapCommunicator<KVType>::num_segments() const {
    return _total_n_pes / num_chunks;
}

template <typename KVType>
int OverlapCommunicator<KVType>::get_comm_rpb() const {
    return num_warps * RDMA_ROW_PER_WARP;
}

// ====== Dynamic Reconfiguration ======

template <typename KVType>
void OverlapCommunicator<KVType>::size_block_work_ids() {
    constexpr int rows_per_work = RDMA_ROW_PER_WARP * num_warps;
    const int effective_batch = num_effective_batch();
    const int num_copy_chunks = B * S_local * _total_n_pes / rows_per_work;
    // FWD uses num_blocks slots (reduce-based wptr). BWD uses work_done[0..N] bitmap (per-work flag).
    // BHSD: bitmap indexed by effective_batch (B*H) × works_per_batch, needs more entries.
    const int bitmap_entries = effective_batch * S_local * _total_n_pes / rows_per_work;
    _bitmap_region_size = ((bitmap_entries + 1) + 3) & ~3;
    const int head_region_size = std::max(num_blocks, _bitmap_region_size);
    if (head_region_size <= _bwi_head_capacity && num_copy_chunks <= _bwi_copy_capacity
        && effective_batch <= _bwi_batch_capacity) {
        return;
    }

    // The AG, put and compute kernels of earlier passes may still use the old block.
    CUDA_DEBUG_CHECK(cudaDeviceSynchronize());
    if (block_work_ids) CUDA_DEBUG_CHECK(cudaFree(block_work_ids));
    _bwi_head_capacity = std::max(_bwi_head_capacity, head_region_size);
    _bwi_copy_capacity = std::max(_bwi_copy_capacity, num_copy_chunks);
    _bwi_batch_capacity = std::max(_bwi_batch_capacity, effective_batch);
    // block_cnt_semaphore[0..3]: AG scheduler counter, fwd write_ptr, stream coordinator, pad
    // rank_commit_counters: nranks (for RS P2P commit)
    // rank_empty_counters: effective_batch * nranks (for hierarchical per-batch Phase 1 AG counters)
    const size_t total_ints = static_cast<size_t>(_bwi_head_capacity) + _bwi_copy_capacity + 4
        + _total_n_pes + static_cast<size_t>(_bwi_batch_capacity) * _total_n_pes;
    CUDA_DEBUG_CHECK(cudaMalloc(&block_work_ids, sizeof(int) * total_ints));
    copy_chunk_mask = block_work_ids + _bwi_head_capacity;
    block_cnt_semaphore = copy_chunk_mask + _bwi_copy_capacity;
    fwd_write_ptr = block_cnt_semaphore + 1;
    stream_coordinator = block_cnt_semaphore + 2;
    fwd_exit_count = block_cnt_semaphore + 3;
    rank_commit_counters = block_cnt_semaphore + 4;
    rank_empty_counters = rank_commit_counters + _total_n_pes;
    // comp stream waits until comm CTAs have set every bit of this
    cudaMemsetAsync(stream_coordinator, 0, sizeof(int), comm_stream);
    // Zeroed before any compute-stream user (reset kernel, chunk-mask kernel) can run.
    CUDA_DEBUG_CHECK(cudaStreamSynchronize(comm_stream));
}

template <typename KVType>
void OverlapCommunicator<KVType>::setup_dkv_buffer() {
    const size_t single_k_numel = _local_batch_stride * B;
    if (!dkv_needs_recreate()) return;

    // Exact size: this runs on growth, or on an nc change, which re-lays the window out
    // whatever its capacity. The caller has quiesced.
    const int ns = _total_n_pes / num_chunks;
    if (dkv_buffer) dkv_buffer->release_for_realloc();
    dkv_buffer = std::make_unique<SepSRBuffer<KVType>>(
        *gin_context_,
        single_k_numel, _total_n_pes, num_chunks, ns, kv_components()
    );
    dkv_buffer->initialize_buffer(_my_pe);
    _dkv_single_k_numel_capacity = single_k_numel;
    _dkv_num_chunks = num_chunks;

    // Per-segment producer streams
    if (ns != _num_put_streams) {
        for (int i = 0; i < _num_put_streams; i++) {
            cudaStreamDestroy(put_streams[i]);
            cudaEventDestroy(bwd_done_events[i]);
        }
        delete[] put_streams;
        delete[] bwd_done_events;
        CUDA_DEBUG_CHECK(cudaFree(rs_block_cnt));
        _num_put_streams = ns;
        put_streams = new cudaStream_t[ns];
        bwd_done_events = new cudaEvent_t[ns];
        int least_priority, greatest_priority;
        cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority);
        for (int i = 0; i < ns; i++) {
            cudaStreamCreateWithPriority(&put_streams[i], cudaStreamNonBlocking,
                std::min(greatest_priority + 1, least_priority));
            cudaEventCreateWithFlags(&bwd_done_events[i], cudaEventDisableTiming);
        }
        CUDA_DEBUG_CHECK(cudaMalloc(&rs_block_cnt, sizeof(int) * ns));
    }
    // Every PE's recv slots and empty flags are zero before any PE primes them.
    kv_buffer->team_bar();
    CUDA_DEBUG_CHECK(cudaDeviceSynchronize());
    prime_rs_semaphores();

    WARN_PRINT("[FlashMask Overlap] (re)created dkv_buffer, chunks: %d, numel: %zu\n", num_chunks, single_k_numel);
}

template <typename KVType>
void OverlapCommunicator<KVType>::prime_rs_semaphores() {
    for (int seg = 0; seg < num_segments(); seg++) notify_recv_empty(seg);
}

template <typename KVType>
bool OverlapCommunicator<KVType>::reconfigure_if_needed(
    int new_b, int new_s_local, int new_h, int new_d,
    int nranks, bool kv_shared
) {
    // BHSD per config, so a D change can flip it back.
    OverlapConfig new_config = {new_b, new_s_local, new_h, new_d, nranks, OverlapConfig::bhsd_for(new_d)};

    // Shared K/V allocates no V/dV region, so the first non-shared pass has to grow the
    // buffers back. Sticky, hence at most one such reallocation per communicator.
    const bool grow_v = !kv_shared && !_kv_alloc_v;
    _kv_shared = kv_shared;

    if (new_config == _config && !grow_v) return false;  // No change needed

    if (grow_v) {
        _kv_alloc_v = true;
        _sr_buffer_capacity = 0;              // force the SRBuffer realloc below
        _dkv_single_k_numel_capacity = 0;     // force the SepSRBuffer recreate below
    }

    if (nranks != _total_n_pes) {
        throw std::logic_error(
            "[FlashMask Overlap] World-size changes require rebuilding the NCCL communicator.");
    }

    // S_local change: validate the new value is in the supported dispatch set
    if (new_s_local != S_local) {
        if (new_s_local != 4096 && new_s_local != 8192 && new_s_local != 16384
            && new_s_local != 32768 && new_s_local != 65536 && new_s_local != 131072) {
            throw std::invalid_argument(
                "[FlashMask Overlap] S_local must be one of {4096, 8192, 16384, 32768, 65536, 131072}, got: "
                + std::to_string(new_s_local)
            );
        }
    }
    reserve_comm_kernel_smem(new_s_local, get_num_chunk_per_segment(new_s_local, _total_n_pes, new_h),
        new_config.use_bhsd ? new_b * new_h : new_b, new_b);

    WARN_PRINT("[FlashMask Overlap] Reconfiguring: old=[%s], new=[%s]\n",
        _config.to_string().c_str(), new_config.to_string().c_str());

    // Update member variables
    B = new_b;
    S_local = new_s_local;
    H = new_h;
    D = new_d;
    _config.use_bhsd = new_config.use_bhsd;  // num_effective_batch() below must see the new layout
    num_chunks = get_num_chunk_per_segment(S_local, _total_n_pes, H);
    _local_batch_stride = new_config.cp_chunk_size();
    _total_numel = new_config.sr_buffer_numel();

    // Exact, grow-only sizing. The hierarchical semaphore stride follows effective_batch
    // (B*H in BHSD), so the semaphore layout can change while the capacities still fit.
    const int new_sema_count = sema::ag::semaphore_count(
        _use_hierarchical, _num_nodes, _total_n_pes, num_effective_batch());
    const bool sr_realloc = _total_numel > _sr_buffer_capacity || new_sema_count > _sema_capacity;
    const bool sema_relayout = new_sema_count != _sema_count;
    const bool dkv_recreate = dkv_needs_recreate();
    // Quiesce only when a window is freed or rezeroed, so no GIN write is still in flight
    // to it; otherwise in-flight flags land at unchanged offsets. All ranks decide alike.
    if (sr_realloc || sema_relayout || dkv_recreate) quiesce();

    if (sr_realloc) {
        const size_t new_capacity = std::max(_total_numel, _sr_buffer_capacity);
        _sema_capacity = std::max(new_sema_count, _sema_capacity);
        kv_buffer->release_for_realloc();
        kv_buffer = std::make_unique<SRBuffer<KVType>>(
            *gin_context_, new_capacity, _sema_capacity, kv_components());
        WARN_PRINT("[FlashMask Overlap] Reconfigure: reallocated SRBuffer, capacity: %zu -> %zu\n",
            _sr_buffer_capacity, new_capacity);
        _sr_buffer_capacity = new_capacity;
        kv_buffer->team_bar();
    } else if (sema_relayout) {
        // Every PE's flags are zero before any PE raises one in the new layout.
        CUDA_DEBUG_CHECK(cudaMemset(kv_buffer->semaphores(), 0, sizeof(int64_t) * _sema_capacity));
        kv_buffer->team_bar();
    }
    _sema_count = new_sema_count;

    size_block_work_ids();
    setup_dkv_buffer();

    _config = new_config;

    WARN_PRINT("[FlashMask Overlap] Reconfigured: B=%d, S_local=%d, H=%d, D=%d, num_chunks=%d\n",
        B, S_local, H, D, num_chunks);

    return true;
}

// explicit instantiation and singleton management

template class OverlapCommunicator<cutlass::bfloat16_t>;
static std::unique_ptr<flashmask::OverlapCommunicator<cutlass::bfloat16_t>> overlap_comm = nullptr;
// UID of the live communicator's generation; a different UID means a different group.
static ncclUniqueId overlap_comm_uid;
static uint64_t overlap_comm_generation = 0;

namespace comm {

OverlapCommunicator<cutlass::bfloat16_t>& init_singleton_instance(
    int b_kv,
    int s_kv,
    int h_kv,
    int d_kv,
    int rank,
    int nranks,
    const uint8_t* unique_id_ptr,
    bool kv_shared
) {
    // Overlap gathers from peers; the bwd split AG/RS buffers assume num_segments >= 2.
    // num_chunks=1 (CP2 or S_local >= 32768) is supported by the splitted kernels:
    // segment 0's local-only chunk triggers early return, and segment 1 processes the single remote chunk.
    if (nranks < 2) {
        throw std::invalid_argument("[FlashMask Overlap] overlap requires nranks > 1, got: "
            + std::to_string(nranks));
    }
    // The AG kernels cache per-rank wait_full results in a kMaxCommRanks-entry smem array.
    if (nranks > kMaxCommRanks) {
        throw std::invalid_argument("[FlashMask Overlap] overlap supports at most "
            + std::to_string(kMaxCommRanks) + " ranks, got: " + std::to_string(nranks));
    }

    // Validate S_local is in the supported dispatch set (applies to both first-time and reconfigure)
    if (s_kv != 4096 && s_kv != 8192 && s_kv != 16384
        && s_kv != 32768 && s_kv != 65536 && s_kv != 131072) {
        throw std::invalid_argument(
            "[FlashMask Overlap] S_local must be one of {4096, 8192, 16384, 32768, 65536, 131072}, got: "
            + std::to_string(s_kv)
        );
    }
    if (unique_id_ptr == nullptr) {
        throw std::invalid_argument("[FlashMask Overlap] NCCL unique ID must not be null.");
    }

    const bool same_generation = overlap_comm
        && std::memcmp(&overlap_comm_uid, unique_id_ptr, sizeof(ncclUniqueId)) == 0;
    if (same_generation) {
        // Shape changes reuse the existing communicator and rebuild only regions.
        const bool was_kv_shared = overlap_comm->kv_shared();
        if (overlap_comm->reconfigure_if_needed(b_kv, s_kv, h_kv, d_kv, nranks, kv_shared)
            || overlap_comm->kv_shared() != was_kv_shared) {
            ++overlap_comm_generation;
        }
        return *overlap_comm;
    }
    // New UID = new group generation (same size or not): rebuild. The teardown barrier
    // is collective on the old communicator, so all its ranks must switch at the same call.
    if (overlap_comm) {
        overlap_comm->quiesce();
        overlap_comm.reset();
    }
    overlap_comm = std::make_unique<OverlapCommunicator<cutlass::bfloat16_t>>(
        b_kv, s_kv, h_kv, d_kv, rank, nranks,
        unique_id_ptr, kv_shared
    );
    std::memcpy(&overlap_comm_uid, unique_id_ptr, sizeof(ncclUniqueId));
    ++overlap_comm_generation;
    return *overlap_comm;
}

OverlapCommunicator<cutlass::bfloat16_t>& singleton() {
    return *overlap_comm;
}

bool is_singleton_null() {
    return overlap_comm == nullptr;
}

uint64_t config_generation() {
    return overlap_comm_generation;
}

void destroy_singleton() {
    if (overlap_comm) {
        // Ensure all async work is complete before destroying
        overlap_comm->quiesce();
        overlap_comm.reset();   // triggers the communicator and window cleanup
        WARN_PRINT("[FlashMask Overlap] Singleton destroyed for topology refresh.\n");
    }
}

}   // namespace comm

}   // namespace flashmask

#undef SChunkCase
#undef SChunkDispatch
