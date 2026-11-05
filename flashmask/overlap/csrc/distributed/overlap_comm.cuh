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
    int S_local = 0;       // runtime; see check_s_local
    int H = 0;
    int D = 0;
    int nranks = 0;        // communicator size; changes require a rebuild
    bool use_bhsd = false;     // SR buffer is (B,H,S,D); set from bhsd_for(D)

    // BHSD puts D in a TMA box dimension, which caps at 256 elements.
    static constexpr int kMaxBhsdHeadDim = 256;
    static bool bhsd_for(int d) { return d <= kMaxBhsdHeadDim; }

    bool operator==(const OverlapConfig& other) const {
        return B == other.B && S_local == other.S_local && H == other.H
            && D == other.D && nranks == other.nranks && use_bhsd == other.use_bhsd;
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
};

/**
 * SM-level overlapping communicator
 *
 * An RAII object managing CP-groups / comm stream / buffer lifetimes automatically
 *
 * Call the constructor of this communicator, and call `run_overlap_ag_kernel` before
 * the main attention kernel, make sure the lifetime of the instance outlast the main kernel.
 * The comm kernels run on a separate stream with a fixed grid of comm CTAs.
 *
 * Dynamic reconfiguration: when B/H/D/S_local change between calls,
 * `reconfigure_if_needed()` will detect the change and reallocate buffers as needed.
 * S_local is a runtime value (check_s_local); capacities and the chunks per segment follow its
 * power-of-two bucket (seqlen_bucket), so lengths inside a seen bucket only re-point the kernels.
 * Backward always runs the split AG plus the dK/dV RS overlap (nranks > 1).
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
        bool kv_shared = false
    );

    ~OverlapCommunicator();

    /**
     * Reconfigure if the shape changed; returns true if it did.
     * UID changes go through init_singleton_instance, which rebuilds.
     * S_local must pass check_s_local.
    */
    bool reconfigure_if_needed(
        int new_b, int new_s_local, int new_h, int new_d,
        int nranks, bool kv_shared
    );

    // Forward AG on comm_stream; progress is published through fwd_write_ptr.
    void run_overlap_ag_kernel();

    // Backward AG of one segment, on comm_stream.
    void run_overlap_splitted_ag_kernel(int segment_idx);

    void run_overlap_rs_kernel(
        KVType* const dk_accum,
        KVType* const dv_accum,
        int segment_idx,
        cudaStream_t comp_stream    // compute_stream (stream of Tensor and bwd kernel)
    );

    void wait_wptr_init();
    // Collective; call before freeing or rezeroing any window (see gin::quiesce).
    void quiesce() { sig_buffer->quiesce(); }

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
    void* dk_send(int seg_idx) const { return dkv_buffer->k_send(seg_idx); }
    void* dv_send(int seg_idx) const { return _kv_shared ? nullptr : dkv_buffer->v_send(seg_idx); }

    // computation stream wait the comm_stream kernel to be scheduled with SMs
    void wait_reset_stream_coordinator(cudaStream_t stream);

    void update_kv_buffer(
        const KVType* const new_k_data,
        const KVType* const new_v_data,
        const bool fwd = true
    );

    // Send slots (ring of min(ns, 2)) are reused across segments and passes: wait for
    // release_buffer's event before writing the dK/dV epilogue into one.
    void wait_dkv_buffer(int segment_idx, cudaStream_t stream) const {
        dkv_buffer->wait_buffer(segment_idx, stream);
    }

    void wait_reduce_done(cudaStream_t stream) const {
        cudaEventRecord(reduce_done, aux_c_stream);
        cudaStreamWaitEvent(stream, reduce_done);
    }

    // mask: int32 (B, num_heads, S_total, 2) [lt_start, ut_end] pairs, 16-byte aligned;
    // strides in int32 elements, multiples of 4.
    void compute_chunk_mask(
        const int* const mask,
        const int num_heads,
        const int64_t batch_stride,
        const int64_t head_stride,
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

    // Forward AG exit counter; its last CTA publishes INT_MAX.
    int* get_fwd_exit_count() const {
        return fwd_exit_count;
    }

    // Ensure previous AG kernel on comm_stream has completed before comp_stream
    // resets the AG counters or overwrites copy_chunk_mask, which that kernel still uses.
    void ensure_ag_done(cudaStream_t stream) {
        cudaStreamWaitEvent(stream, ag_done);
    }

    // Per-work ready flag accessors for BWD compute kernel.
    // work_done array = block_work_ids (bitmap region, 1-based indexing).
    int* get_work_done_ptr() const {
        return block_work_ids;
    }

    // Rows per comm work item (always 512; the last work of a chunk may be shorter).
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
    bool use_hierarchical() const {
        return _use_hierarchical;
    }
    bool use_bhsd_layout() const {
        return _config.use_bhsd;
    }
    bool kv_shared() const {
        return _kv_shared;
    }

    int num_segments() const;
    // for RS overlap, returns number of chunks per segment
    int chunk_per_seg() const;

    // Bytes of all registered symmetric windows.
    size_t symmetric_buffer_bytes() const { return gin::window_bytes(*gin_context_); }

    // ── Cross-stream synchronization events ──────────────────────────────
    //
    //  Event          Direction                        Contract
    //  ─────────────  ───────────────────────────────  ──────────────────────────────────────
    //  wptr_init      comp_stream → comm_stream        write_ptr is initialized, AG kernel may start
    //  sr_usable      comp_stream → comm_stream        Attention consumed local KV chunk, SR buffer reusable
    //  ag_done        comm_stream → comp_stream        AG kernel done with copy_chunk_mask and the AG counters
    //  reduce_done    aux_c_stream → comp_stream       dK/dV recv buffers released and ready
    //  local_moved    put_streams[0] → aux_c_stream    Segment-0 local dKV memcpy d2d completed
    //
    cudaEvent_t wptr_init;
    cudaEvent_t sr_usable;
    cudaEvent_t ag_done;
    cudaEvent_t reduce_done;
    cudaEvent_t local_moved;
    /**
     * dK/dV send/recv slots of the backward RS overlap; always allocated (nranks > 1).
     * The fwd kv_buffer is at least as big as one bwd AG segment, so it
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
    // Backward keeps local KV at the SR base (no tail backup) when hierarchical or
    // single-segment: no segment re-gathers over it. The split AG kernel matches this.
    bool bwd_local_kv_at_base() const { return _use_hierarchical || num_segments() == 1; }

    // Hierarchical semaphore accessors (LSA flags only; refcounts live in sig_buffer):
    //   sema_inter [0..num_nodes-1]: cross-node data-ready flags (0/1)
    //   sema_intra [total_n_pes x batch_ready_words(B*H)]: per-batch intra-node ready bits
    // Non-hierarchical: sema_inter_size=0, sema_intra() == semaphores()
    inline int64_t* sema_inter() const { return kv_buffer->semaphores(); }
    inline int64_t* sema_intra() const { return kv_buffer->semaphores() + _sema_inter_size; }

    // BHSD folds heads into the batch dimension: every (b,h) pair is an independent batch.
    inline int num_effective_batch() const { return _config.use_bhsd ? B * H : B; }

    // Grow-only block_work_ids sizing (per S bucket); offsets move only on growth, which device-syncs first.
    void size_block_work_ids();
    // (Re)create dkv_buffer and its per-segment streams for the current shape
    void setup_dkv_buffer();
    // dK/dV slots too small or laid out for another num_chunks
    bool dkv_needs_recreate() const {
        return B * _local_batch_stride > _dkv_single_k_numel_capacity || num_chunks != _dkv_num_chunks;
    }
    // Opt this shape's split AG / RS put kernels into their dynamic smem; throws if the
    // device can't. Call before the shape is committed.
    void reserve_comm_kernel_smem(int s_local, int nc, int effective_batch, int b) const;
    // Prime per-stage RS semaphores so the first BWD's producer_wait_empty can proceed
    void prime_rs_semaphores();
    // Zero recv(segment_idx) and tell its producers it is empty, on aux_c_stream.
    void recycle_recv_slot(int segment_idx);
    void notify_recv_empty(int segment_idx);

private:
    // Allocated once, before the data windows; world size is fixed.
    std::unique_ptr<SignalBuffer> sig_buffer;
    std::unique_ptr<SRBuffer<KVType>> kv_buffer;
    cudaStream_t comm_stream;
    cudaStream_t aux_c_stream;      // RS consumer (reduce) stream

    // Shape parameters (non-const to allow dynamic reconfiguration)
    int B;
    int S_local;
    int H;
    int D;
    int num_chunks;         // chunks per bwd segment, picked for _s_bucket
    int _s_bucket;          // largest seqlen_bucket(S_local) seen; sizes every buffer
    size_t sr_bucket_numel() const { return static_cast<size_t>(_s_bucket) * H * D * B * _total_n_pes; }

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
    int _sema_count;        // int64 semaphore slots of the current layout
    int _sema_capacity;     // allocated int64 semaphore slots (grow-only)

    // Configuration tracking for dynamic reconfiguration
    OverlapConfig _config;
    size_t _sr_buffer_capacity;             // allocated SRBuffer numel capacity
    size_t _dkv_single_k_numel_capacity = 0;    // allocated SepSRBuffer single_k_numel capacity
    int _dkv_num_chunks = 0;                    // SepSRBuffer's chunks_per_seg (layout-defining)

    int _bitmap_region_size;                // bitmap entries the current shape uses (work_done + frontier)
    int _bwi_head_capacity = 0;             // allocated head region (>= num_blocks and the bitmap)
    int _bwi_copy_capacity = 0;             // allocated copy_chunk_mask entries
    int _bwi_batch_capacity = 0;            // allocated effective batches of rank_empty_counters

    // ── Compound device allocation (single cudaMalloc) ───────────────────
    //
    //  Offset (int elements)                       Field
    //  ──────────────────────────────────────────   ──────────────────────────────
    //  [0 .. head_region)                          block_work_ids  (fwd: per-CTA work ids, bwd: work_done bitmap)
    //  [head_region .. +N)                         copy_chunk_mask  (N = _bwi_copy_capacity)
    //  [+N .. +N+4)                                block_cnt_semaphore, fwd_write_ptr, stream_coordinator, fwd_exit_count
    //  [+N+4 .. +N+4+nranks)                       rank_commit_counters
    //  [.. + effective_batch * nranks)             rank_empty_counters
    //
    int* block_work_ids;
    int* block_cnt_semaphore;
    int* copy_chunk_mask;
    int* fwd_write_ptr;
    int* stream_coordinator;        // make sure comm kernel is scheduled to GPU before computation kernel
    int* fwd_exit_count;            // forward AG CTAs past their last work
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
// Later calls: rebuild on a UID change, else reconfigure if the shape changed.
OverlapCommunicator<cutlass::bfloat16_t>& init_singleton_instance(
    int b_kv,
    int s_kv,
    int h_kv,
    int d_kv,
    int rank,
    int nranks,
    const uint8_t* unique_id_ptr,
    bool kv_shared = false
);

// The live instance; throws std::logic_error before init_singleton_instance.
// Lives until static destruction at process exit.
OverlapCommunicator<cutlass::bfloat16_t>& singleton();

// Bumped on every create, rebuild, reconfigure and kv_shared flip. Getter results
// (sizes, topology, window addresses) stay fixed while it is unchanged. Never 0 once created.
uint64_t config_generation();

}   // namespace comm

}   // namespace flashmask
