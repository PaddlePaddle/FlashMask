// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 The FlashMask Authors. All rights reserved.

#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

// #define FLASHMASK_DEBUG

inline void CheckCudaErrorAux(const char *file, unsigned line,
                                       const char *statement, cudaError_t err) {
    if (err == cudaSuccess)
        return;
    printf("%s returned %s(%d) at %s:%u\n", statement, cudaGetErrorString(err),
           err, file, line);
    exit(1);
}

// The release build still evaluates the wrapped call; only FLASHMASK_DEBUG checks it.
#if defined(FLASHMASK_DEBUG)
#define CUDA_DEBUG_CHECK(value) CheckCudaErrorAux(__FILE__, __LINE__, #value, value)
#else
#define CUDA_DEBUG_CHECK(value) (value)
#endif

namespace flashmask {

// Host setup calls (allocation, stream / event creation, synchronization) whose results
// later launches depend on. Throws like every other host failure; the noexcept bridge
// entry points turn the escape into std::terminate, which prints what().
inline void check_cuda(cudaError_t err, const char* call) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("[FlashMask Overlap] ") + call + " failed: "
            + cudaGetErrorString(err));
    }
}

}  // namespace flashmask

#define FM_CUDA_CHECK(call) ::flashmask::check_cuda((call), #call)
