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

// RAII object of separate send/recv buffer: one send/recv slot per backward segment.
template <typename KVType>
class SepSRBuffer {
using SemaphoreType = int64_t;

private:
    gin::Context& _context;
    KVType* _dk_data;
    KVType* _dv_data;
    SemaphoreType* _semaphores;
    bool _allocated;

    // offset to the recv buffer (2 * chunks_per_seg * k_numel)
    size_t _buf_offset;
    int _semaphore_size;

    const int _num_segments;
    std::vector<cudaEvent_t> _empty_states; 

    // this object cannot be moved or copied
    SepSRBuffer(const SepSRBuffer&) = delete;
    SepSRBuffer(SepSRBuffer&&) = delete;
    SepSRBuffer& operator=(const SepSRBuffer&) = delete;
    SepSRBuffer& operator=(SepSRBuffer&&) = delete;

    inline int slot(int seg_idx) const {
        assert(seg_idx >= 0 && seg_idx < _num_segments);
        return seg_idx;
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

    void team_bar() const {
        gin::barrier(_context);
    }

    void release();
    void release_for_realloc() { release(); }

    ~SepSRBuffer() noexcept;

    // [K_send, V_send] --> buf_offset size, therefore 2 * buf_offset is the double buffer offset
    // (shared K/V drops the V halves, so buf_offset is then just the dK slab)
    inline KVType* k_send(int seg_idx) const { return _dk_data + slot(seg_idx) * 2 * _buf_offset; }
    inline KVType* v_send(int seg_idx) const { return _dv_data + slot(seg_idx) * 2 * _buf_offset; }
    inline KVType* k_recv(int seg_idx) const { return _dk_data + (slot(seg_idx) * 2 + 1) * _buf_offset; }
    inline KVType* v_recv(int seg_idx) const { return _dv_data + (slot(seg_idx) * 2 + 1) * _buf_offset; }
    inline SemaphoreType* semaphores(int seg_idx) const { return _semaphores + slot(seg_idx) * _semaphore_size; }

    // Zero every recv slot; the sparse RS puts leave holes the reduce would otherwise read.
    void initialize_buffer(int self_rank);

    void wait_buffer(int seg_idx, cudaStream_t stream) {
        cudaStreamWaitEvent(stream, _empty_states[slot(seg_idx)]);
    }

    void release_buffer(int seg_idx, cudaStream_t stream) {
        cudaEventRecord(_empty_states[slot(seg_idx)], stream);
    }

    // clear recv buffer (so that reduce won't op on dirty data)
    void zero_recv_buf(int seg_idx, cudaStream_t comm_stream);
};

}   // namespace flashmask
