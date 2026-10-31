// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 The FlashMask Authors. All rights reserved.

#include "overlap_comm.cuh"
#include "cp_heuristic.cuh"
#include "cutlass/bfloat16.h"

#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include <nccl.h>

using bf16 = cutlass::bfloat16_t;

namespace {

// uint64 handle from Python -> cudaStream_t (0 == default stream).
inline cudaStream_t as_stream(uint64_t handle) {
    return reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(handle));
}

// FM-4 does not run prepare_flashmask_kernel, so reset the persistent AG scheduler
// counter, the forward frontier and the forward exit counter here before every gather.
__global__ void fm4_reset_ag_counter_kernel(int* const block_cnt_semaphore, int* const write_ptr,
                                            int* const exit_count) {
    if (threadIdx.x == 0) {
        *block_cnt_semaphore = 1;
        *write_ptr = 0;
        *exit_count = 0;
    }
}

// Reset the scheduler/frontier on `stream` and record wptr_init; the next AG waits on it. The
// caller has ordered `stream` behind the prior AG (ensure_ag_done), which still uses all three.
void reset_ag_counter(flashmask::OverlapCommunicator<bf16>& c, cudaStream_t stream) {
    fm4_reset_ag_counter_kernel<<<1, 32, 0, stream>>>(
        c.get_block_cnt_semaphore(), c.get_fwd_write_ptr(), c.get_fwd_exit_count());
    cudaEventRecord(c.wptr_init, stream);
}

// Common head of the fwd AG and the bwd segment-0 split AG. Issue order:
//   compute: [record sr_usable] [wait ag_done + send slots] chunk-mask, reset, [record wptr_init],
//            stream coordinator
//   comm:    [wait sr_usable] AgWaitConsumed, local K/V copy, SetFull, [wait wptr_init], AG
// sr_usable is recorded before the chunk-mask kernel, whose waits and writes (copy_chunk_mask)
// the comm-side SR writes do not depend on; the AG reads copy_chunk_mask behind wptr_init.
// The reset needs no ensure_ag_done of its own: compute_chunk_mask just made the compute
// stream wait on ag_done, and nothing records ag_done in between.
void start_ag(flashmask::OverlapCommunicator<bf16>& c,
              uint64_t mask_ptr, int num_heads, int64_t batch_stride, int64_t head_stride,
              uint64_t k_ptr, uint64_t v_ptr, cudaStream_t stream, bool fwd) {
    c.wait_sr_buffer_empty(stream);
    c.compute_chunk_mask(reinterpret_cast<const int*>(static_cast<uintptr_t>(mask_ptr)),
                         num_heads, batch_stride, head_stride, stream, fwd);
    c.update_kv_buffer(reinterpret_cast<const bf16*>(static_cast<uintptr_t>(k_ptr)),
                       reinterpret_cast<const bf16*>(static_cast<uintptr_t>(v_ptr)), fwd);
    reset_ag_counter(c, stream);
    c.wait_wptr_init();
    if (fwd) {
        c.run_overlap_ag_kernel();
    } else {
        c.run_overlap_splitted_ag_kernel(0);
    }
    c.wait_reset_stream_coordinator(stream);
}

}  // namespace

// The FM-3 backend reports setup failures (NCCL version mismatch, topology, GIN
// capability, CUDA OOM) by throwing. ctypes calls these entry points directly,
// and unwinding a C++ exception into CPython is undefined behaviour -- it lands
// in the host framework's pybind translator and dies as a bare SIGSEGV with the
// message lost. `noexcept` turns any escape into std::terminate, which prints
// what() before aborting.
extern "C" {

// Rank 0 fills `out` with the NCCL unique ID; the caller broadcasts it and
// passes the same bytes to fm4_overlap_init. Returns 1 on success.
int fm4_overlap_get_unique_id(uint8_t* out) noexcept {
    if (out == nullptr) return 0;
    ncclUniqueId unique_id;
    if (ncclGetUniqueId(&unique_id) != ncclSuccess) return 0;
    std::memcpy(out, &unique_id, sizeof(unique_id));
    return 1;
}

size_t fm4_overlap_unique_id_size() noexcept {
    return sizeof(ncclUniqueId);
}

// Chunks per backward segment the communicator picks for this K shape (the host
// heuristic alone; no communicator needed), so check_inputs can reject a shape the
// split AG / RS kernels are not instantiated for before any communication.
int fm4_overlap_num_chunks_per_segment(int b_kv, int s_kv, int h_kv, int d_kv,
                                       int nranks) noexcept {
    return flashmask::get_num_chunk_per_segment(b_kv, s_kv, h_kv, d_kv, nranks);
}

// Create, rebuild (new UID = new group) or reconfigure (new shape) the singleton.
// Only shape (b/s/h/d) + topology (rank/nranks/uid) are needed; the local-KV copy
// happens later in fm4_overlap_start_ag. Returns comm::config_generation() (> 0): every
// getter below returns the same value until the generation changes.
// kv_shared: K and V alias one tensor, so only K is transported.
uint64_t fm4_overlap_init(int b_kv, int s_kv, int h_kv, int d_kv,
                          int rank, int nranks,
                          const uint8_t* unique_id,
                          int kv_shared) noexcept {
    flashmask::comm::init_singleton_instance(
        b_kv, s_kv, h_kv, d_kv, rank, nranks, unique_id, kv_shared != 0);
    return flashmask::comm::config_generation();
}

// SRBuffer K/V base pointers as raw uint64 (wrapped into cute tensors in Python).
uint64_t fm4_overlap_k_data() noexcept {
    return reinterpret_cast<uint64_t>(flashmask::comm::singleton().k_data());
}
uint64_t fm4_overlap_v_data() noexcept {
    return reinterpret_cast<uint64_t>(flashmask::comm::singleton().v_data());
}
uint64_t fm4_overlap_work_done() noexcept {
    return reinterpret_cast<uint64_t>(
        flashmask::comm::singleton().get_work_done_ptr());
}
// Device int the forward compute gate spins on (row frontier of the forward AG).
uint64_t fm4_overlap_fwd_write_ptr() noexcept {
    return reinterpret_cast<uint64_t>(
        flashmask::comm::singleton().get_fwd_write_ptr());
}

// Local seqlen chunk (S_local); after AG, S_total = s_local * nranks.
int fm4_overlap_s_local() noexcept { return flashmask::comm::singleton().s_local(); }
int fm4_overlap_nranks() noexcept { return flashmask::comm::singleton().nranks(); }
// This process's rank in the communicator (= its CP group rank).
int fm4_overlap_rank() noexcept { return flashmask::comm::singleton().rank(); }
// Size of the NCCL LSA domain; hierarchical iff nranks > gpus_per_node.
int fm4_overlap_gpus_per_node() noexcept { return flashmask::comm::singleton().gpus_per_node(); }
int fm4_overlap_use_bhsd_layout() noexcept {
    return flashmask::comm::singleton().use_bhsd_layout() ? 1 : 0;
}
int fm4_overlap_use_hierarchical() noexcept {
    return flashmask::comm::singleton().use_hierarchical() ? 1 : 0;
}
int fm4_overlap_comm_rpb() noexcept { return flashmask::comm::singleton().get_comm_rpb(); }
int fm4_overlap_num_segments() noexcept { return flashmask::comm::singleton().num_segments(); }
int fm4_overlap_segment_seqlen() noexcept {
    auto& c = flashmask::comm::singleton();
    return c.s_local() * c.chunk_per_seg();
}
uint64_t fm4_overlap_segment_k_data(int segment_idx) noexcept {
    return reinterpret_cast<uint64_t>(
        flashmask::comm::singleton().segment_k_data(segment_idx));
}
uint64_t fm4_overlap_segment_v_data(int segment_idx) noexcept {
    return reinterpret_cast<uint64_t>(
        flashmask::comm::singleton().segment_v_data(segment_idx));
}
uint64_t fm4_overlap_dk_send(int segment_idx) noexcept {
    return reinterpret_cast<uint64_t>(
        flashmask::comm::singleton().dk_send(segment_idx));
}
uint64_t fm4_overlap_dv_send(int segment_idx) noexcept {
    return reinterpret_cast<uint64_t>(
        flashmask::comm::singleton().dv_send(segment_idx));
}

// Total bytes of the registered symmetric windows (for memory accounting).
uint64_t fm4_overlap_symmetric_buffer_bytes() noexcept {
    return flashmask::comm::singleton().symmetric_buffer_bytes();
}

// Start one forward or backward pass's gather (see start_ag): refresh the per-chunk
// sparsity mask (copy_chunk_mask) the AG and RS kernels read, stage the local K/V, launch
// the forward AG (fwd != 0; the forward gate spins on fm4_overlap_fwd_write_ptr) or the split
// AG of backward segment 0, then hold compute_stream until the AG kernel occupies its SMs.
// mask is device int32 (B, H_mask, S_total, 2) [lt_start, ut_end] pairs (non-causal), each
// pair contiguous and the base 16-byte aligned; batch_stride / head_stride are its strides in
// int32 elements, multiples of 4.
void fm4_overlap_start_ag(uint64_t mask_ptr, int num_heads,
                          int64_t batch_stride, int64_t head_stride,
                          uint64_t k_ptr, uint64_t v_ptr,
                          uint64_t compute_stream, int fwd) noexcept {
    start_ag(flashmask::comm::singleton(), mask_ptr, num_heads, batch_stride, head_stride,
             k_ptr, v_ptr, as_stream(compute_stream), fwd != 0);
}

void fm4_overlap_start_bwd_segment(int segment_idx, uint64_t compute_stream) noexcept {
    auto& c = flashmask::comm::singleton();
    cudaStream_t stream = as_stream(compute_stream);
    c.ensure_ag_done(stream);
    reset_ag_counter(c, stream);
    c.wait_wptr_init();
    c.run_overlap_splitted_ag_kernel(segment_idx);
    c.wait_reset_stream_coordinator(stream);
}

void fm4_overlap_wait_dkv_buffer(int segment_idx, uint64_t compute_stream) noexcept {
    flashmask::comm::singleton().wait_dkv_buffer(segment_idx, as_stream(compute_stream));
}

void fm4_overlap_run_rs(uint64_t dk_ptr, uint64_t dv_ptr, int segment_idx,
                        uint64_t compute_stream) noexcept {
    flashmask::comm::singleton().run_overlap_rs_kernel(
        reinterpret_cast<bf16*>(static_cast<uintptr_t>(dk_ptr)),
        reinterpret_cast<bf16*>(static_cast<uintptr_t>(dv_ptr)),
        segment_idx, as_stream(compute_stream));
}

void fm4_overlap_wait_reduce_done(uint64_t compute_stream) noexcept {
    flashmask::comm::singleton().wait_reduce_done(as_stream(compute_stream));
}

}  // extern "C"
