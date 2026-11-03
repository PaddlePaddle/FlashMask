// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 The FlashMask Authors. All rights reserved.

#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>
#include <cutlass/bfloat16.h>
#include <cutlass/array.h>

namespace flashmask {

using bf16 = cutlass::bfloat16_t;
using bf16x4 = cutlass::Array<bf16, 4>;

// bf16x4 and float4 conversion, so that we can use fp32 accumulation
__device__ __forceinline__ float4 to_float4(bf16x4 in) {
    return {
        static_cast<float>(in[0]),
        static_cast<float>(in[1]),
        static_cast<float>(in[2]),
        static_cast<float>(in[3])
    };
}

__device__ __forceinline__ bf16x4 to_bf16x4(float4 in) {
    bf16x4 out;
    out[0] = static_cast<bf16>(in.x);
    out[1] = static_cast<bf16>(in.y);
    out[2] = static_cast<bf16>(in.z);
    out[3] = static_cast<bf16>(in.w);
    return out;
}

// One component (dK or dV) of a reduce step. Pointers are (B, nc * S_chunk, H, D) recv slots;
// out is (B, S_chunk, H, D).
//   home      recv(0); chunk 0 holds the low 8 bytes of each fp32 quad of the running sum
//   home_hi   high 8 bytes: recv(0) chunk 1 if nc >= 2, else recv(1) chunk 0
//   recv      this step's recv(s); null skips the component (kv_shared dV)
// Not __restrict__: home_hi aliases home; recv may alias home or home_hi.
struct ReduceOperand {
    bf16* home;
    bf16* home_hi;
    const bf16* recv;
    bf16* out;
};

// Where a reduce step's fp32 sum starts.
enum class ReduceStart {
    kZero,      // single segment: recv(0) holds every chunk, nothing to fold
    kFoldHome,  // segment 1: fold the parked recv(0) chunks first
    kRunning,   // later segments: resume the running sum in the home bytes
};

/**
 * fp32 reduce of the RS recv slots; the running sum lives in the home bytes, not a scratch.
 * One thread owns each (b, quad), so the add order is 0 + recv(0) chunks + recv(1) chunks + ...
*/
template <int num_chunks = 4, ReduceStart start = ReduceStart::kFoldHome, bool is_last = true>
__global__ __launch_bounds__(128, 8)
void ReducedKdVKernel(
    const ReduceOperand dk,
    const ReduceOperand dv,
    const int num_tasks_per_batch       // S_chunk * H * D / 512
) {
    static_assert(start != ReduceStart::kZero || is_last, "a single segment is also the last");
    static constexpr int elem_per_block = 512;
    const int b = blockIdx.y;           // batch

    const int64_t elem_per_chunk = int64_t(num_tasks_per_batch) * elem_per_block;    // chunk stride
    const int64_t b_offset_accum = b * elem_per_chunk;
    const int64_t b_offset_sr = b_offset_accum * num_chunks;

    auto add_chunks = [&](float4& acc, const bf16* const src, const int64_t base_offset) {
        #pragma unroll
        for (int c = 0; c < num_chunks; ++c) {
            float4 temp_v = to_float4(
                *reinterpret_cast<const bf16x4*>(src + c * elem_per_chunk + base_offset)
            );
            acc.x += temp_v.x;
            acc.y += temp_v.y;
            acc.z += temp_v.z;
            acc.w += temp_v.w;
        }
    };

    // task offset is small_chunk offset + thread offset
    auto reduce_op = [&](const ReduceOperand& op, int64_t task_offset) {
        const int64_t base_offset = b_offset_sr + task_offset;
        uint2* const lo = reinterpret_cast<uint2*>(op.home + base_offset);
        uint2* const hi = reinterpret_cast<uint2*>(op.home_hi + base_offset);
        float4 acc = make_float4(0, 0, 0, 0);
        if constexpr (start == ReduceStart::kFoldHome) {
            add_chunks(acc, op.home, base_offset);
        } else if constexpr (start == ReduceStart::kRunning) {
            const uint2 l = *lo, h = *hi;
            acc = make_float4(__uint_as_float(l.x), __uint_as_float(l.y),
                              __uint_as_float(h.x), __uint_as_float(h.y));
        }
        add_chunks(acc, op.recv, base_offset);

        if constexpr (is_last) {
            *reinterpret_cast<bf16x4*>(op.out + b_offset_accum + task_offset) = to_bf16x4(acc);
        } else {
            *lo = make_uint2(__float_as_uint(acc.x), __float_as_uint(acc.y));
            *hi = make_uint2(__float_as_uint(acc.z), __float_as_uint(acc.w));
        }
    };

    for (int task_idx = blockIdx.x; task_idx < num_tasks_per_batch; task_idx += gridDim.x) {
        const int64_t task_offset = int64_t(task_idx) * elem_per_block + 4 * threadIdx.x;

        reduce_op(dk, task_offset);
        if (dv.recv) reduce_op(dv, task_offset);
    }
}

#define ReduceKernelLaunch(_num_chunk, _start, _is_last)                                 \
    ReducedKdVKernel<_num_chunk, _start, _is_last><<<grid, 128, 0, stream>>>(            \
        dk, dv, num_tasks_per_chunk)

#define ChunkDipatchKernelLaunch(num_chunk, start, is_last)                              \
    switch (num_chunk) {                                                                \
        case 4: { ReduceKernelLaunch(4, start, is_last); break; }                       \
        case 2: { ReduceKernelLaunch(2, start, is_last); break; }                       \
        case 8: { ReduceKernelLaunch(8, start, is_last); break; }                       \
        case 1: { ReduceKernelLaunch(1, start, is_last); break; }                       \
    default:                                                                            \
        throw std::invalid_argument(                                                    \
            "[FlashMask Overlap] num_chunks must be one of {1, 2, 4, 8}, got: "         \
            + std::to_string(num_chunk));                                               \
    }

/**
 * Reduce step of segment `segment_idx` >= 1, or of a lone segment 0. A null dv.recv skips dV
 * (kv_shared merges dV into dK).
*/
void launch_dk_dv_reduce(
    const ReduceOperand& dk,
    const ReduceOperand& dv,
    int B, int S_chunk, int H, int D,
    int num_chunks, int segment_idx, int num_segments, cudaStream_t stream
) {
    // 128 threads, each reduces 4 bf16
    static constexpr int elem_per_block = 512;
    size_t elem_per_chunk = static_cast<size_t>(S_chunk) * H * D;
    // a typical value: 8192 * 8 * 128 / 512 = 16384
    int num_tasks_per_chunk = elem_per_chunk / elem_per_block;

    // typically, B = 1, so we have 2048 CTAs --> 16 CTAs per SM = 128 SMs
    // the reduce speed shouldn't be a bottleneck, so it's OK to allocate more SMs
    dim3 grid(std::max(2048 / B, 128), B);

    const bool is_last = segment_idx == num_segments - 1;
    if (segment_idx == 0) {
        if (!is_last) {
            throw std::invalid_argument(
                "[FlashMask Overlap] segment 0 reduces only as the single segment, got num_segments: "
                + std::to_string(num_segments));
        }
        ChunkDipatchKernelLaunch(num_chunks, ReduceStart::kZero, true);
    } else if (segment_idx == 1) {
        if (is_last) { ChunkDipatchKernelLaunch(num_chunks, ReduceStart::kFoldHome, true); }
        else { ChunkDipatchKernelLaunch(num_chunks, ReduceStart::kFoldHome, false); }
    } else {
        if (is_last) { ChunkDipatchKernelLaunch(num_chunks, ReduceStart::kRunning, true); }
        else { ChunkDipatchKernelLaunch(num_chunks, ReduceStart::kRunning, false); }
    }
}

#undef ChunkDipatchKernelLaunch
#undef ReduceKernelLaunch

}   // namespace flashmask