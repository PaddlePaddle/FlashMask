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

// RAII object of separate send/recv buffers. Window layout, each slot _buf_offset elements
// ([dK nc chunks][dV nc chunks], dV absent when K/V are shared):
//   [send 0 .. K_s-1][recv 0 .. ns-1][semaphores ns x semaphore_size]
// Send is a ring of K_s = min(ns, 2) slots: segment s writes send(s % K_s), and the slot's
// event, recorded after the reads of segment s, gates the compute writes of segment s + K_s.
// Each read of a slot follows (via done_event) a compute write that waited on the slot's
// previous record, so the latest record covers every earlier read of that slot.
// Recv, semaphores and signals stay per segment: peers address them by segment index.
template <typename KVType>
class SepSRBuffer {
using SemaphoreType = int64_t;

private:
    gin::Context& _context;
    KVType* _dk_data;
    KVType* _dv_data;
    SemaphoreType* _semaphores;
    bool _allocated;

    size_t _buf_offset;     // slot size: kv_components * chunks_per_seg * single_k_numel
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

    void team_bar() const {
        gin::barrier(_context);
    }

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
    void initialize_buffer(int self_rank);

    // Orders `stream` after every read of send(seg)'s slot issued so far.
    void wait_buffer(int seg_idx, cudaStream_t stream) {
        cudaStreamWaitEvent(stream, _empty_states[send_slot(seg_idx)]);
    }

    // Orders `stream` after every read of every send slot issued so far.
    void wait_all_send_slots(cudaStream_t stream) {
        for (cudaEvent_t event : _empty_states) cudaStreamWaitEvent(stream, event);
    }

    void release_buffer(int seg_idx, cudaStream_t stream) {
        cudaEventRecord(_empty_states[send_slot(seg_idx)], stream);
    }

    // Re-zero the extent of recv(seg) one pass used: single_k_numel elements per chunk of
    // dK, and of dV when with_v. The rest of the slot is never written, so stays zero.
    void zero_recv_buf(int seg_idx, size_t single_k_numel, bool with_v, cudaStream_t comm_stream);
};

}   // namespace flashmask
