// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 The FlashMask Authors. All rights reserved.

#pragma once
#include <memory>
#include <string>
#include "sr_buffer.cuh"
#include "sep_sr_buffer.cuh"
#include "hierarchical_rank_map.cuh"
#include "cutlass/bfloat16.h"

namespace flashmask {

// Configuration snapshot for detecting parameter changes between calls.
// All fields that affect buffer sizes or kernel behavior are tracked here.
struct OverlapConfig {
    int B = 0;
    int S_local = 0;       // dispatched at runtime: {4096, 8192, 16384, 32768, 65536, 131072}
    int H = 0;
    int H_mask = 0;
    int D = 0;
    int nranks = 0;        // communicator size; changes require a rebuild
    bool overlap_rs = false;
    bool use_bhsd = false;     // SR buffer uses (B,H,S,D); derived from D (<= 256) per config

    bool operator==(const OverlapConfig& other) const {
        return B == other.B && S_local == other.S_local && H == other.H
            && H_mask == other.H_mask && D == other.D && nranks == other.nranks
            && overlap_rs == other.overlap_rs && use_bhsd == other.use_bhsd;
    }
    bool operator!=(const OverlapConfig& other) const { return !(*this == other); }

    // Compute the SRBuffer numel for one of K or V (= B * S_local * H * D * nranks)
    size_t sr_buffer_numel() const {
        return static_cast<size_t>(S_local) * H * D * B * nranks;
    }

    // Compute the per-chunk size (= S_local * H * D)
    size_t cp_chunk_size() const {
        return static_cast<size_t>(S_local) * H * D;
    }

    std::string to_string() const {
        return "B=" + std::to_string(B) + ", S_local=" + std::to_string(S_local)
            + ", H=" + std::to_string(H) + ", H_mask=" + std::to_string(H_mask)
            + ", D=" + std::to_string(D) + ", nranks=" + std::to_string(nranks)
            + ", overlap_rs=" + std::to_string(int(overlap_rs))
            + ", use_bhsd=" + std::to_string(int(use_bhsd));
    }
};

/**
 * SM-level overlapping communicator
 *
 * An RAII object managing CP-groups / comm stream / buffer lifetimes automatically
 *
 * Call the constructor of this communicator, and call `run_overlap_ag_kernel` before
 * the main attention kernel, make sure the lifetime of the instance outlast the main kernel
 * Then you should be able to get async remote get, costing only 4 SMs.
 *
 * Dynamic reconfiguration: when B/H/D/mask_head/S_local change between calls,
 * `reconfigure_if_needed()` will detect the change and reallocate buffers as needed.
 * S_local is dispatched at compile time: supported values are {4096, 8192, 16384, 32768, 65536, 131072}.
 * RS-overlap is automatically enabled when H_k >= rs_overlap_min_h_k and nranks > 1.
*/
template <typename KVType>
class OverlapCommunicator {
private:
    // Declared first so it outlives every registered buffer.
    gin::ContextPtr gin_context_;

public:
    OverlapCommunicator(
        int b_kv,
        int s_kv,
        int h_kv,
        int d_kv,
        int rank,
        int nranks,
        const uint8_t* unique_id_ptr,
        int mask_head = 0,
        bool overlap_rs = false,
        bool kv_shared = false
    );

    ~OverlapCommunicator();

    /**
     * Check if reconfiguration is needed given new shape params, and perform it if so.
     * Returns true if reconfiguration happened. Communicator identity (UID) changes
     * are handled by init_singleton_instance, which rebuilds instead.
     * S_local must be one of {4096, 8192, 16384, 32768, 65536, 131072}.
    */
    bool reconfigure_if_needed(
        int new_b, int new_s_local, int new_h, int new_d,
        int nranks, int new_mask_head, bool new_overlap_rs, bool kv_shared
    );

    // Launch the forward (fwd=true) or non-split AG kernel on comm_stream; gathers
    // S_local * nranks rows. Forward progress is published through fwd_write_ptr.
    void run_overlap_ag_kernel(const bool fwd = true);

    // only used when use_rs_overlap and in the bwd
    void run_overlap_splitted_ag_kernel(int segment_idx);

    void run_overlap_rs_kernel(
        KVType* const dk_accum,
        KVType* const dv_accum,
        int segment_idx,
        cudaStream_t comp_stream    // compute_stream (stream of Tensor and bwd kernel)
    );

    void wait_wptr_init();
    void barrier() { gin::barrier(*gin_context_); }

    void* k_data() const { return kv_buffer->k_data(); }
    // Shared K/V leaves the V region ungathered, so consumers must read K instead.
    void* v_data() const { return _kv_shared ? kv_buffer->k_data() : kv_buffer->v_data(); }
    void* segment_k_data(int segment_idx) const {
        const size_t offset = _use_hierarchical
            ? static_cast<size_t>(segment_idx) * B * num_chunks * _local_batch_stride
            : 0;
        return kv_buffer->k_data() + offset;
    }
    void* segment_v_data(int segment_idx) const {
        const size_t offset = _use_hierarchical
            ? static_cast<size_t>(segment_idx) * B * num_chunks * _local_batch_stride
            : 0;
        return (_kv_shared ? kv_buffer->k_data() : kv_buffer->v_data()) + offset;
    }

    // we need to reroute the bwd dx_accum output buffer to dk_send and dv_send
    // so that the output of post-proc kernel can be directly sent
    // DO NOT call the following methods, if overlap_rs = false
    void* dk_send(int seg_idx) const { return dkv_buffer->k_send(seg_idx); }
    void* dv_send(int seg_idx) const { return _kv_shared ? nullptr : dkv_buffer->v_send(seg_idx); }

    // computation stream wait the comm_stream kernel to be scheduled with SMs
    void wait_reset_stream_coordinator(cudaStream_t stream);

    void update_kv_buffer(
        const KVType* const new_k_data,
        const KVType* const new_v_data,
        const bool fwd = true
    );

    // A send slot is read by its put kernel on p_stream and reused across passes, so
    // comp_stream must not run the dK/dV epilogue into it before release_buffer's event.
    // On the first pass the events are unrecorded, which makes cudaStreamWaitEvent a no-op.
    void wait_dkv_buffer(int segment_idx, cudaStream_t stream) const {
        dkv_buffer->wait_buffer(segment_idx, stream);
    }

    void wait_reduce_done(cudaStream_t stream) const {
        cudaEventRecord(reduce_done, aux_c_stream);
        cudaStreamWaitEvent(stream, reduce_done);
    }

    void compute_chunk_mask(
        const int* const lt_start_ptr,
        const int* const lt_end_ptr,
        const int* const ut_start_ptr,
        const int* const ut_end_ptr,
        cudaStream_t stream,
        const bool fwd = true
    );

    // call this function before calling `update_kv_buffer`
    // to make sure other PEs have finished reading the local KV data in our SR buffer
    // also, barriers the remote_get `comm_kernel`
    void wait_sr_buffer_empty(cudaStream_t stream);

    int* get_block_cnt_semaphore() const {
        return block_cnt_semaphore;
    }

    // Forward AG frontier the compute gate spins on; reset with block_cnt_semaphore.
    int* get_fwd_write_ptr() const {
        return fwd_write_ptr;
    }

    // Ensure previous AG kernel on comm_stream has completed before comp_stream
    // resets block_cnt_semaphore / fwd_write_ptr (reset_ag_counter) or overwrites
    // copy_chunk_mask (compute_chunk_mask), both of which that kernel still uses.
    void ensure_ag_done(cudaStream_t stream) {
        cudaStreamWaitEvent(stream, ag_done);
    }

    // Per-work ready flag accessors for BWD compute kernel.
    // work_done array = block_work_ids (bitmap region, 1-based indexing).
    int* get_work_done_ptr() const {
        return block_work_ids;
    }

    // row_per_block used by comm kernel = num_warps * RDMA_ROW_PER_WARP.
    // Returns 0 if per-work mode should not be used.
    int get_comm_rpb() const;

    int nranks() const {
        return _total_n_pes;
    }

    int rank() const {
        return _my_pe;
    }

    int s_local() const {
        return S_local;
    }

    int gpus_per_node() const {
        return _gpus_per_node;
    }
    int num_nodes() const {
        return _num_nodes;
    }
    bool use_hierarchical() const {
        return _use_hierarchical;
    }
    bool use_bhsd_layout() const {
        return _config.use_bhsd;
    }

    int num_segments() const;
    // for RS overlap, returns number of chunks per segment
    int chunk_per_seg() const;
    void prepare_dkv_buffer();

    // ── Cross-stream synchronization events ──────────────────────────────
    //
    //  Event          Direction                        Contract
    //  ─────────────  ───────────────────────────────  ──────────────────────────────────────
    //  wptr_init      comp_stream → comm_stream        write_ptr is initialized, AG kernel may start
    //  sr_usable      comp_stream → comm_stream        Attention consumed local KV chunk, SR buffer reusable
    //  ag_done        comm_stream → comp_stream        AG kernel finished; it no longer reads copy_chunk_mask nor writes block_cnt_semaphore / fwd_write_ptr
    //  reduce_done    aux_c_stream → comp_stream       dK/dV recv buffers released and ready          (RS-overlap only)
    //  local_moved    put_streams[0] → aux_c_stream    Segment-0 local dKV memcpy d2d completed       (RS-overlap only)
    //
    cudaEvent_t wptr_init;
    cudaEvent_t sr_usable;
    cudaEvent_t ag_done;
    cudaEvent_t reduce_done;    // RS-overlap only
    cudaEvent_t local_moved;    // RS-overlap only
    /**
     * If overlap_rs is set, dkv_buffer will be populated.
     * and since the fwd AG buffer is always bigger than bwd AG
     * (due to the fact that bwd AG is splitted), fwd kv_buffer
     * can be reused (carefully).
    */
    std::unique_ptr<SepSRBuffer<KVType>> dkv_buffer;
private:
    /**
     * Note(heqianyue): for B > 1, RS-overlap, we need a place to store the local KV chunk
     * so that each split AG remote_get call can correctly send the data to other ranks.
     * We choose to store one more copy of the local KV chunk data at the end of the SR buffer.
     * Note that this makes it two copies of the local KV chunks: the first copy is ordered
     * with a batch stride of num_chunks * S_local * H * D, so that for the first segement,
     * attention kernel can directly use SR buffer for bwd recompute. This copy of local KV chunks
     * will be overwritten by the upcoming segments, so we need the second copy (gauranteed:
     * will never be overwritten) for remote ranks to get from.
     *
     * return the last B * S_local * H * D elems in the respective SR buffer
     *
    */
    inline KVType* local_k_data() const {
        return kv_buffer->k_data() + _total_numel - B * _local_batch_stride;
    }
    inline KVType* local_v_data() const {
        return kv_buffer->v_data() + _total_numel - B * _local_batch_stride;
    }

    // Semaphore accessors for hierarchical dual-array protocol:
    //   sema_inter [0..num_nodes-1]: cross-node data-ready flags (0/1)
    //   sema_intra [total_n_pes ranks x batch_ready_words(B*H) words]: word 0 is the refcount,
    //     all words carry the per-batch intra-node consumption signals
    // Non-hierarchical: sema_inter_size=0, sema_intra() == semaphores()
    inline int64_t* sema_inter() const { return kv_buffer->semaphores(); }
    inline int64_t* sema_intra() const { return kv_buffer->semaphores() + _sema_inter_size; }

    // BHSD folds heads into the batch dimension: every (b,h) pair is an independent batch.
    inline int num_effective_batch() const { return _config.use_bhsd ? B * H : B; }

    // Helper to (re)allocate block_work_ids and derived pointers
    void reallocate_block_work_ids();
    // Helper to create or recreate the dkv_buffer for RS-overlap
    void setup_dkv_buffer(bool need_rs);
    // Prime per-stage RS semaphores so the first BWD's producer_wait_empty can proceed
    void prime_rs_semaphores();

private:
    std::unique_ptr<SRBuffer<KVType>> kv_buffer;
    cudaStream_t comm_stream;
    cudaStream_t aux_c_stream;      // RS-overlap: consumer (reduce) stream

    // Shape parameters (non-const to allow dynamic reconfiguration)
    int B;
    int S_local;            // dispatched at runtime: {4096, 8192, 16384, 32768, 65536, 131072}
    int H;
    int H_mask;             // mask head
    int D;
    int num_chunks;

    int _my_pe;
    int _total_n_pes;
    size_t _local_batch_stride;
    size_t _total_numel;

    // Hierarchical overlap topology
    int _gpus_per_node;     // Number of ranks in the NCCL LSA team
    int _num_nodes;         // Number of nodes (= _total_n_pes / _gpus_per_node)
    bool _use_hierarchical;      // CP spans more than one LSA domain
    bool _kv_shared = false;     // K and V alias one tensor: transport K only
    // Whether the buffers hold V/dV regions. Sticky: shared K/V allocates without them,
    // and the first non-shared pass grows them back once.
    bool _kv_alloc_v = true;
    int kv_components() const { return _kv_alloc_v ? 2 : 1; }
    int _sema_inter_size;   // num_nodes for hierarchical, 0 otherwise (offset into semaphore array)
    int _sema_count;        // allocated int64 semaphore slots

    // Configuration tracking for dynamic reconfiguration
    OverlapConfig _config;
    size_t _sr_buffer_capacity;             // allocated SRBuffer numel capacity
    size_t _dkv_single_k_numel_capacity;    // allocated SepSRBuffer single_k_numel capacity
    int _dkv_num_chunks;                    // SepSRBuffer's chunks_per_seg (layout-defining)

    // RS-overlap: fp32 scratch holding the cross-segment dK/dV sum (dK half | dV half),
    // so the only bf16 rounding is the last segment's store into dk_ptr / dv_ptr.
    // Allocated lazily (grow-only) in prepare_dkv_buffer, and only when num_segments > 1.
    float* dkv_f32_accum;
    size_t _dkv_f32_numel;                  // allocated float count of dkv_f32_accum (0 when unallocated)

    // dV half of the scratch; null when K/V are shared, which drops the dV half entirely.
    float* dkv_f32_dv() const {
        return _kv_shared ? nullptr : dkv_f32_accum + _dkv_f32_numel / 2;
    }

    int _num_copy_chunks;                   // block_work_ids array size tracking
    int _bitmap_region_size;                // bitmap region size (work_done + frontier)

    // ── Compound device allocation (single cudaMalloc) ───────────────────
    //
    //  Offset (int elements)                       Field
    //  ──────────────────────────────────────────   ──────────────────────────────
    //  [0 .. head_region)                          block_work_ids  (fwd: per-CTA work ids, bwd: work_done bitmap)
    //  [head_region .. +N)                         copy_chunk_mask  (N = _num_copy_chunks)
    //  [+N .. +N+4)                                block_cnt_semaphore, fwd_write_ptr, stream_coordinator, pad
    //  [+N+4 .. +N+4+nranks)                       rank_commit_counters
    //  [.. + effective_batch * nranks)             rank_empty_counters
    //
    int* block_work_ids;
    int* block_cnt_semaphore;
    int* copy_chunk_mask;
    int* fwd_write_ptr;
    int* stream_coordinator;        // make sure comm kernel is scheduled to GPU before computation kernel
    int* rank_commit_counters;      // per-rank put completion counters for P2P RS commit
    int* rank_empty_counters;       // per-rank get completion counters for P2P AG empty notify

    // RS per-segment producer resources
    cudaStream_t* put_streams;      // [num_segments]
    cudaEvent_t* bwd_done_events;   // [num_segments]
    int* rs_block_cnt;              // device memory [num_segments]
    int _num_put_streams;
};

namespace comm {

// OverlapCommunicator instance is managed via this singleton, therefore
// the instance is accessible to both fwd and bwd passes

// init or reconfigure instance (mutable ref), used in fwd.
// On first call: creates the singleton.
// On subsequent calls: rebuilds when the UID (group generation) changed, otherwise
// reconfigures buffers if the shape changed.
OverlapCommunicator<cutlass::bfloat16_t>& init_singleton_instance(
    int b_kv,
    int s_kv,
    int h_kv,
    int d_kv,
    int rank,
    int nranks,
    const uint8_t* unique_id_ptr,
    int mask_head = 1,
    bool kv_shared = false
);

// get instance (mutable ref), make sure the instance is initialized, used in both fwd and bwd
OverlapCommunicator<cutlass::bfloat16_t>& singleton();

// check whether the singleton unique_ptr is nullptr
bool is_singleton_null();

// Destroy the singleton for topology refresh.
// After calling this, the next init_singleton_instance() will re-create from scratch.
// IMPORTANT: communicator generation changes require a full destroy/rebuild when
// rank/nranks remain the same. This function is intended for refreshing
// transport-level resources (e.g., after node migration), not for changing topology.
void destroy_singleton();

}   // namespace comm

}   // namespace flashmask
