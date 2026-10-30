// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 The FlashMask Authors. All rights reserved.

#include "overlap_comm.cuh"
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
// counter and the forward frontier here before every gather.
__global__ void fm4_reset_ag_counter_kernel(int* const block_cnt_semaphore, int* const write_ptr) {
    if (threadIdx.x == 0) {
        *block_cnt_semaphore = 1;
        *write_ptr = 0;
    }
}

// Wait for the prior AG before resetting the scheduler/frontier; the next AG waits on wptr_init.
void reset_ag_counter(flashmask::OverlapCommunicator<bf16>& c, cudaStream_t stream) {
    c.ensure_ag_done(stream);
    fm4_reset_ag_counter_kernel<<<1, 32, 0, stream>>>(
        c.get_block_cnt_semaphore(), c.get_fwd_write_ptr());
    cudaEventRecord(c.wptr_init, stream);
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

// Create, rebuild (new UID = new group) or reconfigure (new shape) the singleton.
// Only shape (b/s/h/d) + topology (rank/nranks/uid) are needed; the local-KV copy
// happens later in fm4_overlap_update_kv. Returns 1.
// kv_shared: K and V alias one tensor, so only K is transported.
int fm4_overlap_init(int b_kv, int s_kv, int h_kv, int d_kv,
                     int rank, int nranks,
                     const uint8_t* unique_id,
                     int kv_shared) noexcept {
    flashmask::comm::init_singleton_instance(
        b_kv, s_kv, h_kv, d_kv, rank, nranks, unique_id, kv_shared != 0);
    return 1;
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

// Copy local K/V into the SRBuffer on the internal comm_stream (cudaMemcpyAsync).
void fm4_overlap_update_kv(uint64_t k_ptr, uint64_t v_ptr, int fwd) noexcept {
    flashmask::comm::singleton().update_kv_buffer(
        reinterpret_cast<const bf16*>(static_cast<uintptr_t>(k_ptr)),
        reinterpret_cast<const bf16*>(static_cast<uintptr_t>(v_ptr)), fwd != 0);
}

// Compute the per-chunk sparsity mask (copy_chunk_mask) so the AG kernel can skip
// fully-masked chunks. MUST precede fm4_overlap_run_ag (the AG and RS kernels read
// the mask; it is uninitialized otherwise). mask is device int32 (B, H_mask, S_total, 2)
// [lt_start, ut_end] pairs (non-causal), each pair contiguous and the base 16-byte
// aligned; batch_stride / head_stride are its strides in int32 elements, multiples of 4.
// Pass the compute stream so the write is ordered before the AG kernel picks it
// up via the wait_sr_buffer_empty handshake.
void fm4_overlap_compute_chunk_mask(uint64_t mask_ptr, int num_heads,
                                    int64_t batch_stride, int64_t head_stride,
                                    uint64_t stream, int fwd) noexcept {
    flashmask::comm::singleton().compute_chunk_mask(
        reinterpret_cast<const int*>(static_cast<uintptr_t>(mask_ptr)),
        num_heads, batch_stride, head_stride,
        as_stream(stream), fwd != 0);
}

// Launch the forward all-gather remote-get kernel on the internal comm_stream. The forward
// gate spins on fm4_overlap_fwd_write_ptr.
void fm4_overlap_run_ag() noexcept {
    flashmask::comm::singleton().run_overlap_ag_kernel();
}

// compute_stream notifies comm_stream that the local SRBuffer chunk may be reused.
void fm4_overlap_wait_sr_buffer_empty(uint64_t compute_stream) noexcept {
    flashmask::comm::singleton().wait_sr_buffer_empty(as_stream(compute_stream));
}

// Total bytes of the registered symmetric windows (for memory accounting).
uint64_t fm4_overlap_symmetric_buffer_bytes() noexcept {
    return flashmask::comm::singleton().symmetric_buffer_bytes();
}

void fm4_overlap_start_bwd_segment(int segment_idx, uint64_t compute_stream) noexcept {
    auto& c = flashmask::comm::singleton();
    cudaStream_t stream = as_stream(compute_stream);
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

// Reset the AG scheduler on the compute stream; pairs with
// fm4_overlap_wait_wptr_init on the comm stream.
void fm4_overlap_reset_ag_counter(uint64_t compute_stream) noexcept {
    reset_ag_counter(flashmask::comm::singleton(), as_stream(compute_stream));
}

// comm_stream waits on wptr_init so the remote-get kernel reads the counter only
// after the reset lands. Pairs with fm4_overlap_reset_ag_counter.
void fm4_overlap_wait_wptr_init() noexcept {
    flashmask::comm::singleton().wait_wptr_init();
}

// compute stream waits until the comm kernel is actually scheduled onto SMs.
void fm4_overlap_wait_reset_stream_coordinator(uint64_t stream) noexcept {
    flashmask::comm::singleton().wait_reset_stream_coordinator(as_stream(stream));
}

}  // extern "C"
