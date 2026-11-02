// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 The FlashMask Authors. All rights reserved.

/**
 * Separated S-R buffer for a2a based all-gather 
*/
#pragma once

#include "nccl_gin_backend.cuh"
#include <cassert>
#include <stdexcept>
#include <cstring>
#include <vector>

namespace flashmask {

// Separate send/recv buffers: [send 0..K_s-1][recv 0..ns-1][semaphores], K_s = min(ns, 2).
// Send is a ring (segment s uses slot s % K_s); recv and semaphores are per segment.
template <typename KVType>
class SepSRBuffer {
using SemaphoreType = int64_t;

private:
    gin::Context& _context;
    KVType* _dk_data;
    KVType* _dv_data;
    SemaphoreType* _semaphores;
    bool _allocated;

    size_t _buf_offset;     // slot size in elements
    int _semaphore_size;

    const int _chunks_per_seg;
    const int _num_segments;
    const int _num_send_slots;
    std::vector<cudaEvent_t> _empty_states;     // one per send slot

    // this object cannot be moved or copied
    SepSRBuffer(const SepSRBuffer&) = delete;
    SepSRBuffer(SepSRBuffer&&) = delete;
    SepSRBuffer& operator=(const SepSRBuffer&) = delete;
    SepSRBuffer& operator=(SepSRBuffer&&) = delete;

    inline size_t send_slot(int seg_idx) const {
        assert(seg_idx >= 0 && seg_idx < _num_segments);
        return seg_idx % _num_send_slots;
    }
    inline size_t recv_slot(int seg_idx) const {
        assert(seg_idx >= 0 && seg_idx < _num_segments);
        return _num_send_slots + seg_idx;
    }
public:
    explicit SepSRBuffer(
        gin::Context& context,
        size_t single_k_numel,
        int semaphore_size,
        int chunks_per_seg,
        int num_segments,
        int kv_components = 2
    );

    void release();
    void release_for_realloc() { release(); }

    ~SepSRBuffer() noexcept;

    inline KVType* k_send(int seg_idx) const { return _dk_data + send_slot(seg_idx) * _buf_offset; }
    inline KVType* v_send(int seg_idx) const { return _dv_data + send_slot(seg_idx) * _buf_offset; }
    inline KVType* k_recv(int seg_idx) const { return _dk_data + recv_slot(seg_idx) * _buf_offset; }
    inline KVType* v_recv(int seg_idx) const { return _dv_data + recv_slot(seg_idx) * _buf_offset; }
    inline SemaphoreType* semaphores(int seg_idx) const {
        assert(seg_idx >= 0 && seg_idx < _num_segments);
        return _semaphores + seg_idx * _semaphore_size;
    }

    // Zero every recv slot; the sparse RS puts leave holes the reduce would otherwise read.
    void initialize_buffer();

    // Orders `stream` after all reads of send(seg)'s slot issued so far.
    void wait_buffer(int seg_idx, cudaStream_t stream) {
        cudaStreamWaitEvent(stream, _empty_states[send_slot(seg_idx)]);
    }

    // Orders `stream` after all reads of every send slot issued so far.
    void wait_all_send_slots(cudaStream_t stream) {
        for (cudaEvent_t event : _empty_states) cudaStreamWaitEvent(stream, event);
    }

    void release_buffer(int seg_idx, cudaStream_t stream) {
        cudaEventRecord(_empty_states[send_slot(seg_idx)], stream);
    }

    // Re-zeroes the part of recv(seg) one pass wrote (dK, plus dV when with_v).
    void zero_recv_buf(int seg_idx, size_t single_k_numel, bool with_v, cudaStream_t comm_stream);
};

}   // namespace flashmask
