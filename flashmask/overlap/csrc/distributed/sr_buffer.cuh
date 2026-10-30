// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 The FlashMask Authors. All rights reserved.

#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

#include "nccl_gin_backend.cuh"

namespace flashmask {

using SemaphoreType = int64_t;

template <typename KVType>
class SRBuffer {
private:
    gin::Context& _context;
    KVType* _k_sr = nullptr;
    KVType* _v_sr = nullptr;
    SemaphoreType* _semaphores = nullptr;
    bool _allocated = false;
    size_t _numel = 0;

    SRBuffer(const SRBuffer&) = delete;
    SRBuffer& operator=(const SRBuffer&) = delete;
    SRBuffer(SRBuffer&&) = delete;
    SRBuffer& operator=(SRBuffer&&) = delete;

public:
    explicit SRBuffer(gin::Context& context, size_t numel, int semaphore_size = 0,
                      int kv_components = 2)
        : _context(context) {
        if (numel == 0) {
            throw std::invalid_argument("SRBuffer: numel must be positive");
        }
        if (numel & 31) {
            throw std::invalid_argument("SRBuffer: numel should be a multiple of 32");
        }

        const size_t total_bytes = kv_components * numel * sizeof(KVType) +
                                   semaphore_size * sizeof(SemaphoreType);
        _k_sr = static_cast<KVType*>(gin::alloc(_context, total_bytes));
        // ncclMemAlloc returns uninitialized memory. The compute side can load rows of
        // skipped (never fetched) works, e.g. the 2-CTA keep-alive tile, and 0 * NaN
        // garbage would poison dQ/dK; zeroing also clears the semaphores.
        const cudaError_t memset_result = cudaMemset(_k_sr, 0, total_bytes);
        if (memset_result != cudaSuccess) {
            gin::free_noexcept(_context, _k_sr);
            throw std::runtime_error(
                std::string("SRBuffer: zeroing failed: ") + cudaGetErrorString(memset_result));
        }
        // Shared K/V has no V region; null so that a stray V access faults instead of
        // silently landing in the semaphores.
        _v_sr = kv_components == 2 ? _k_sr + numel : nullptr;
        _semaphores = reinterpret_cast<SemaphoreType*>(_k_sr + kv_components * numel);
        _allocated = true;
        _numel = numel;
    }

    ~SRBuffer() noexcept {
        if (_allocated && _k_sr != nullptr) {
            gin::free_noexcept(_context, _k_sr);
            reset();
        }
    }

    void team_bar() const { gin::barrier(_context); }

    void release() {
        if (_allocated && _k_sr != nullptr) {
            gin::free(_context, _k_sr);
            reset();
        }
    }

    void release_for_realloc() { release(); }

    KVType* k_data() const noexcept { return _k_sr; }
    KVType* v_data() const noexcept { return _v_sr; }
    SemaphoreType* semaphores() const noexcept { return _semaphores; }

    bool is_valid() const noexcept {
        return _allocated && _k_sr != nullptr && _semaphores != nullptr;
    }

    size_t capacity() const noexcept { return _numel; }

private:
    void reset() noexcept {
        _k_sr = nullptr;
        _v_sr = nullptr;
        _semaphores = nullptr;
        _allocated = false;
        _numel = 0;
    }
};

}  // namespace flashmask
