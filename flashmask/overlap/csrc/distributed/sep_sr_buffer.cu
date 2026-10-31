// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 The FlashMask Authors. All rights reserved.

/**
 * Statistics: for extreme use case, CP16 N4
 *  K shape (1, 128K, 8, 128) -> CP -> (1, 8192, 8, 128)
 *
 * Each slot holds nc chunks of dK (+ dV); the window holds min(ns, 2) send slots and ns
 * recv slots, i.e. (min(ns, 2) + ns) * nc chunks per component.
*/

#include "sep_sr_buffer.cuh"
#include "debug_logger.cuh"
#include <algorithm>
#include <cutlass/bfloat16.h>

namespace flashmask {

template <typename KVType>
SepSRBuffer<KVType>::SepSRBuffer(
    gin::Context& context,
    size_t single_k_numel,
    int semaphore_size,
    int chunks_per_seg,
    int num_segments,
    int kv_components
) :
    _context(context), _dk_data(nullptr), _dv_data(nullptr), _semaphores(nullptr),
    _allocated(false),
    _buf_offset(kv_components * chunks_per_seg * single_k_numel),
    _semaphore_size(semaphore_size),
    _chunks_per_seg(chunks_per_seg),
    _num_segments(num_segments),
    _num_send_slots(std::min(num_segments, 2))
{
    if (single_k_numel & 31) {
        throw std::invalid_argument("SepSRBuffer: numel should be the positive multiple of 32");
    }
    if (num_segments <= 0) {
        throw std::invalid_argument("SepSRBuffer: num_segments must be > 0, got: " + std::to_string(num_segments));
    }

    const size_t total_elements = (_num_send_slots + num_segments) * _buf_offset +
            num_segments * semaphore_size * sizeof(SemaphoreType) / sizeof(KVType);
    _empty_states.resize(_num_send_slots);

    const size_t total_bytes = total_elements * sizeof(KVType);
    _dk_data = static_cast<KVType*>(gin::alloc(_context, total_bytes));
    // Shared K/V has no dV slab; null so that a stray dV access faults instead of
    // silently landing in the dK slab.
    _dv_data = kv_components == 2 ? _dk_data + chunks_per_seg * single_k_numel : nullptr;
    for (cudaEvent_t& event : _empty_states) {
        FM_CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    }
    _semaphores = reinterpret_cast<SemaphoreType*>(_dk_data + (_num_send_slots + num_segments) * _buf_offset);
    _allocated = true;
}

template <typename KVType>
void SepSRBuffer<KVType>::release() {
    if (!_allocated || _dk_data == nullptr) {
        return;
    }

    gin::free(_context, _dk_data);
    for (cudaEvent_t event : _empty_states) {
        CUDA_DEBUG_CHECK(cudaEventDestroy(event));
    }
    _empty_states.clear();
    _dk_data = nullptr;
    _dv_data = nullptr;
    _semaphores = nullptr;
    _allocated = false;
    _buf_offset = 0;
    _semaphore_size = 0;
}

template <typename KVType>
SepSRBuffer<KVType>::~SepSRBuffer() noexcept {
    if (!_allocated || _dk_data == nullptr) {
        return;
    }
    gin::free_noexcept(_context, _dk_data);
    for (cudaEvent_t event : _empty_states) {
        cudaEventDestroy(event);
    }
}

template <typename KVType>
void SepSRBuffer<KVType>::zero_recv_buf(int seg_idx, size_t single_k_numel, bool with_v, cudaStream_t comm_stream) {
    const size_t used_bytes = sizeof(KVType) * _chunks_per_seg * single_k_numel;
    cudaMemsetAsync(k_recv(seg_idx), 0, used_bytes, comm_stream);
    if (with_v) cudaMemsetAsync(v_recv(seg_idx), 0, used_bytes, comm_stream);
}

template <typename KVType>
void SepSRBuffer<KVType>::initialize_buffer() {
    // RS puts are sparse, so the reduce reads recv rows no producer wrote; every slot must
    // start at zero, and zero_recv_buf only recycles a slot after its first use.
    // all the GPU ops uses the default blocking stream, since this is only called during initialization (one-off)
    // recv slots and semaphores are contiguous: zero them all at once
    const size_t bytes = _num_segments * (sizeof(KVType) * _buf_offset + _semaphore_size * sizeof(SemaphoreType));
    FM_CUDA_CHECK(cudaMemset(k_recv(0), 0, bytes));
}

template class SepSRBuffer<cutlass::bfloat16_t>;

}   // namespace flashmask

