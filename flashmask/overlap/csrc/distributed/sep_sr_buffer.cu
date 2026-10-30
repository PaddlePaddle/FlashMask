// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 The FlashMask Authors. All rights reserved.

/**
 * Statistics: for extreme use case, CP16 N4
 *  K shape (1, 128K, 8, 128) -> CP -> (1, 8192, 8, 128)
 * 
 *  We need (4 or 2, fp32 or bf16) * 2 (KV) * 2 (SR) * 4 (num_chunks) * 
 *  numel = 256M (bf16) or 512M (bf16). If double buffering, x2. 
 * So I suppose we should not use double buffering to alleviate GMEM consumption
*/

#include "sep_sr_buffer.cuh"
#include "debug_logger.cuh"
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
    _num_segments(num_segments)
{
    if (single_k_numel & 31) {
        throw std::invalid_argument("SepSRBuffer: numel should be the positive multiple of 32");
    }
    if (num_segments <= 0) {
        throw std::invalid_argument("SepSRBuffer: num_segments must be > 0, got: " + std::to_string(num_segments));
    }

    // 2 = send + recv, kv_components = dK (+ dV unless K and V are shared)
    size_t total_elements = 2 * _buf_offset +
            semaphore_size * sizeof(SemaphoreType) / sizeof(KVType);

    total_elements *= num_segments;
    _empty_states.resize(num_segments);

    const size_t total_bytes = total_elements * sizeof(KVType);
    _dk_data = static_cast<KVType*>(gin::alloc(_context, total_bytes));
    // Shared K/V has no dV slab; null so that a stray dV access faults instead of
    // silently landing in the dK slab.
    _dv_data = kv_components == 2 ? _dk_data + chunks_per_seg * single_k_numel : nullptr;
    for (int i = 0; i < num_segments; i++) {
        cudaEventCreateWithFlags(&_empty_states[i], cudaEventDisableTiming);
    }
    _semaphores = reinterpret_cast<SemaphoreType*>(_dk_data + 2 * num_segments * _buf_offset);
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
void SepSRBuffer<KVType>::zero_recv_buf(int seg_idx, cudaStream_t comm_stream) {
    cudaMemsetAsync(_dk_data + _buf_offset * (1 + 2 * slot(seg_idx)), 0, sizeof(KVType) * _buf_offset, comm_stream);
}

template <typename KVType>
void SepSRBuffer<KVType>::initialize_buffer(int self_rank) {
    // RS puts are sparse, so the reduce reads recv rows no producer wrote; every slot must
    // start at zero, and zero_recv_buf only recycles a slot after its first use.
    // all the GPU ops uses the default blocking stream, since this is only called during initialization (one-off)
    size_t recv_buffer_sz = sizeof(KVType) * _buf_offset;
    for (int stage = 0; stage < _num_segments; stage++) {
        cudaMemset(_dk_data + _buf_offset * (1 + 2 * stage), 0, recv_buffer_sz);
    }
    size_t semaphore_bytes = _semaphore_size * sizeof(SemaphoreType);
    // set recv buffer and semaphores to be 0 all at once
    cudaMemset(_semaphores, 0, semaphore_bytes * _num_segments);
}

template class SepSRBuffer<cutlass::bfloat16_t>;

}   // namespace flashmask

