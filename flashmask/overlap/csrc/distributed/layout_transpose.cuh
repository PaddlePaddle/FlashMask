// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 The FlashMask Authors. All rights reserved.

#pragma once
/**
 * High-performance BSHD → BHSD transpose-copy kernel for bf16.
 *
 * transpose_copy_to_sr: local KV (B,S_local,H,D) → SR buffer (B,H,S_total,D) at row
 * s_offset (FWD: S_total - S_local, BWD: 0).
 *
 * D is a runtime argument. It only feeds the TMA descriptors, the dynamic smem
 * size and the expect_tx byte count, so no code path specializes on it. TMA needs
 * D*2 bytes to be a multiple of 16 and D <= 256, which every head dim satisfies.
 */

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <cuda.h>
#include "debug_logger.cuh"

namespace flashmask {
namespace layout {

inline void check_cu(CUresult result, const char* call) {
    if (result != CUDA_SUCCESS) {
        throw std::runtime_error(std::string("[FlashMask Overlap] ") + call
            + " failed: CUresult " + std::to_string(static_cast<int>(result)));
    }
}

// Runtime-resolve cuTensorMapEncodeTiled via cudaGetDriverEntryPointByVersion.
// This avoids linking against libcuda.so which is a driver-provided library.
inline CUresult call_cuTensorMapEncodeTiled(
    CUtensorMap *tensorMap, CUtensorMapDataType tensorDataType,
    cuuint32_t tensorRank, void *globalAddress,
    const cuuint64_t *globalDim, const cuuint64_t *globalStrides,
    const cuuint32_t *boxDim, const cuuint32_t *elementStrides,
    CUtensorMapInterleave interleave, CUtensorMapSwizzle swizzle,
    CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill
) {
    using FuncPtr = CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*,
        const cuuint64_t*, const cuuint64_t*, const cuuint32_t*, const cuuint32_t*,
        CUtensorMapInterleave, CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
    static FuncPtr pfn = nullptr;
    if (!pfn) {
        void* fn = nullptr;
        cudaDriverEntryPointQueryResult status = cudaDriverEntryPointSymbolNotFound;
        // Ask for the CUDA 12.0 ABI of the symbol (where it was introduced): the unversioned call
        // requests the runtime's own version and fails on an older user-mode driver.
        FM_CUDA_CHECK(cudaGetDriverEntryPointByVersion(
            "cuTensorMapEncodeTiled", &fn, 12000, cudaEnableDefault, &status));
        if (status != cudaDriverEntryPointSuccess || !fn) {
            throw std::runtime_error(
                "[FlashMask Overlap] cudaGetDriverEntryPointByVersion(cuTensorMapEncodeTiled) failed: "
                "driver status " + std::to_string(static_cast<int>(status)));
        }
        pfn = reinterpret_cast<FuncPtr>(fn);
    }
    return pfn(tensorMap, tensorDataType, tensorRank, globalAddress,
               globalDim, globalStrides, boxDim, elementStrides,
               interleave, swizzle, l2Promotion, oobFill);
}

// TMA load (strided BSHD global → smem) + TMA store (smem → contiguous BHSD global).
// Smem layout: [tile_data: TILE_S × D bf16] [mbarrier: 8 bytes]
// Both TMA ops share the same smem — load fills it as (S, D) row-major,
// store reads it in the same order. No per-thread data movement needed.
template <int TILE_S = 64>
__global__ void __launch_bounds__(128)
transpose_copy_kernel_tma(
    const __grid_constant__ CUtensorMap src_tma_map,   // (B, S_local, H, D) BSHD
    const __grid_constant__ CUtensorMap dst_tma_map,   // (B, H, S_dst, D) BHSD
    const int S_local,
    const int H,
    const int S_dst,
    const int tile_bytes,                              // TILE_S * D * sizeof(bf16)
    const int s_offset
) {
    extern __shared__ char smem_raw[];
    uint64_t* mbar_ptr = reinterpret_cast<uint64_t*>(smem_raw + tile_bytes);

    const int bh = blockIdx.y;
    const int s_tile_idx = blockIdx.x;
    const int b = bh / H;
    const int h = bh % H;
    const int s_start = s_tile_idx * TILE_S;
    if (s_start >= S_local) return;

    uint32_t mbar_addr = (uint32_t)__cvta_generic_to_shared(mbar_ptr);
    uint32_t smem_addr = (uint32_t)__cvta_generic_to_shared(smem_raw);

    if (threadIdx.x == 0) {
        asm volatile("mbarrier.init.shared.b64 [%0], %1;"
            :: "r"(mbar_addr), "r"(1));
    }
    __syncthreads();

    // TMA load: src (B, S_local, H, D) — dims {D, H, S, B}, box {D, 1, TILE_S, 1}
    // Coords: {d=0, h, s_start, b}
    if (threadIdx.x == 0) {
        asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;"
            :: "r"(mbar_addr), "r"(tile_bytes));
        asm volatile(
            "cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
            " [%0], [%1, {%2, %3, %4, %5}], [%6];"
            :: "r"(smem_addr), "l"(&src_tma_map),
               "r"(0), "r"(h), "r"(s_start), "r"(b),
               "r"(mbar_addr));
    }

    // Wait for TMA load completion
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "LAB_WAIT:\n\t"
        "mbarrier.try_wait.parity.shared.b64 p, [%0], 0;\n\t"
        "@!p bra LAB_WAIT;\n\t"
        "}\n\t"
        :: "r"(mbar_addr));
    __syncthreads();

    // TMA store: dst (B, H, S_dst, D) — dims {D, S_dst, H, B}, box {D, TILE_S, 1, 1}
    // Coords: {d=0, s=s_offset+s_start, h, b}
    if (threadIdx.x == 0) {
        asm volatile(
            "cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group"
            " [%0, {%1, %2, %3, %4}], [%5];"
            :: "l"(&dst_tma_map),
               "r"(0), "r"(s_offset + s_start), "r"(h), "r"(b),
               "r"(smem_addr));
        // Wait for TMA store to complete before CTA exits
        asm volatile("cp.async.bulk.wait_group 0;");
    }
}

inline void create_src_tma_descriptor(
    CUtensorMap* tma_map, const __nv_bfloat16* src, int B, int S_local, int H, int D
) {
    // Source (B, S_local, H, D), TMA dims (innermost→outermost): D, H, S, B
    uint64_t globalDim[4] = {
        (uint64_t)D, (uint64_t)H, (uint64_t)S_local, (uint64_t)B
    };
    uint64_t globalStrides[3] = {
        (uint64_t)(D * sizeof(__nv_bfloat16)),
        (uint64_t)((size_t)H * D * sizeof(__nv_bfloat16)),
        (uint64_t)((size_t)S_local * H * D * sizeof(__nv_bfloat16))
    };
    constexpr int TILE_S = 64;
    uint32_t boxDim[4] = {(uint32_t)D, 1, (uint32_t)TILE_S, 1};
    uint32_t elementStrides[4] = {1, 1, 1, 1};

    check_cu(call_cuTensorMapEncodeTiled(
        tma_map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, (void*)src,
        globalDim, globalStrides, boxDim, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    ), "cuTensorMapEncodeTiled (src)");
}

// s_end bounds the S coordinate: a partial last tile is clipped there instead of writing
// into the rows after this rank's chunk.
inline void create_dst_tma_descriptor(
    CUtensorMap* tma_map, __nv_bfloat16* dst, int B, int H, int S_dst, int D, int s_end
) {
    // Destination (B, H, S_dst, D), TMA dims (innermost→outermost): D, S_dst, H, B
    uint64_t globalDim[4] = {
        (uint64_t)D, (uint64_t)s_end, (uint64_t)H, (uint64_t)B
    };
    uint64_t globalStrides[3] = {
        (uint64_t)(D * sizeof(__nv_bfloat16)),                      // stride for S (row-to-row within (b,h) slice)
        (uint64_t)((size_t)S_dst * D * sizeof(__nv_bfloat16)),      // stride for H
        (uint64_t)((size_t)H * S_dst * D * sizeof(__nv_bfloat16))   // stride for B
    };
    constexpr int TILE_S = 64;
    uint32_t boxDim[4] = {(uint32_t)D, (uint32_t)TILE_S, 1, 1};
    uint32_t elementStrides[4] = {1, 1, 1, 1};

    check_cu(call_cuTensorMapEncodeTiled(
        tma_map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, (void*)dst,
        globalDim, globalStrides, boxDim, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    ), "cuTensorMapEncodeTiled (dst)");
}

inline void launch_copy_tma(
    const __nv_bfloat16* src, __nv_bfloat16* dst,
    int B, int S_local, int H, int D, int S_dst, int s_offset,
    cudaStream_t stream = 0
) {
    constexpr int TILE_S = 64;
    const int tile_bytes = TILE_S * D * sizeof(__nv_bfloat16);
    const int smem_size = tile_bytes + sizeof(uint64_t);

    CUtensorMap src_tma_map, dst_tma_map;
    create_src_tma_descriptor(&src_tma_map, src, B, S_local, H, D);
    create_dst_tma_descriptor(&dst_tma_map, dst, B, H, S_dst, D, s_offset + S_local);

    dim3 grid((S_local + TILE_S - 1) / TILE_S, B * H);
    dim3 block(128);

    FM_CUDA_CHECK(cudaFuncSetAttribute(
        transpose_copy_kernel_tma<TILE_S>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    transpose_copy_kernel_tma<TILE_S><<<grid, block, smem_size, stream>>>(
        src_tma_map, dst_tma_map, S_local, H, S_dst, tile_bytes, s_offset);
    FM_CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// Public API
// ============================================================================

/// Transpose-copy local KV (BSHD) into SR buffer (BHSD) at given S offset.
///   src: (B, S_local, H, D), contiguous BSHD
///   dst: (B, H, S_dst, D), BHSD (SR buffer)
///   s_offset: FWD = S_dst - S_local, BWD = 0
inline void transpose_copy_to_sr(
    const __nv_bfloat16* src, __nv_bfloat16* dst,
    int B, int S_local, int H, int D, int S_dst, int s_offset,
    cudaStream_t stream = 0
) {
    launch_copy_tma(src, dst, B, S_local, H, D, S_dst, s_offset, stream);
}

}  // namespace layout
}  // namespace flashmask
