// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 The FlashMask Authors. All rights reserved.

#pragma once

#include <cuda_runtime.h>

#include "nccl_gin_backend.cuh"

namespace flashmask::shmem {

// Per-thread register cap of the comm kernels. Comm CTAs never share an SM with compute
// CTAs; 64 regs at 512 threads fits 2 comm CTAs (one AG + one RS) per SM.
constexpr int comm_regs_per_thread = 64;
// 16 B loads each thread keeps in flight per lsa_copy_block iteration.
constexpr int lsa_copy_unroll = 8;

__host__ __device__ constexpr int comm_min_blocks_per_sm(int num_warps) {
    return 65536 / (num_warps * 32 * comm_regs_per_thread);
}

// A null dst2 means the caller has a single component to move (shared K/V); src2 is then
// unused. Offsetting keeps it null, so callers need no branch of their own.
template <typename T>
__device__ __forceinline__ T* offset_or_null(T* base, int64_t offset) {
    return base != nullptr ? base + offset : nullptr;
}

// Block-wide copy of `count` int4s; stages `unroll` loads before storing to hide NVLink
// latency. No __restrict__: the source can change mid-kernel, so ld.global.nc is illegal.
template <int unroll>
__device__ __forceinline__ void lsa_copy_block(int4* dst, const int4* src, int count) {
    const int stride = blockDim.x;
    const int tile = unroll * stride;
    const int full = count - count % tile;
    int i = threadIdx.x;
    for (; i < full; i += tile) {
        int4 staged[unroll];
#pragma unroll
        for (int u = 0; u < unroll; ++u) {
            staged[u] = src[i + u * stride];
        }
#pragma unroll
        for (int u = 0; u < unroll; ++u) {
            dst[i + u * stride] = staged[u];
        }
    }
    for (; i < count; i += stride) {
        dst[i] = src[i];
    }
}

// Out of line to avoid register spills in the comm kernels (see network_get_two_buffers).
// Copies whole int4s only; a trailing bytes % 16 is dropped.
static __device__ __noinline__ void copy_two_buffers_block(
    void* dst1, void* dst2, const void* src1, const void* src2, int bytes) {
    const int count = bytes / sizeof(int4);
    lsa_copy_block<lsa_copy_unroll>(
        static_cast<int4*>(dst1), static_cast<const int4*>(src1), count);
    if (dst2 != nullptr) {
        lsa_copy_block<lsa_copy_unroll>(
            static_cast<int4*>(dst2), static_cast<const int4*>(src2), count);
    }
}

// Out of line: building a GIN request needs ~126 registers, and inlining it would make
// the register-capped comm kernels spill around the copy loop.
// `static` is required: with external linkage the ABI register count is fixed
// before ptxas sees the caller, and separable compilation then rejects a
// budget-capped entry calling a 126-register function.
static __device__ __noinline__ void network_get_two_buffers(
    void* dst1, void* dst2, const void* src1, const void* src2,
    int bytes, int world_peer) {
    const auto& src_region = gin::find_region(src1);
    const auto& dst_region = gin::find_region(dst1);
    const int peer = gin::team_peer(world_peer);
    auto transport = gin::make_gin();
    // The doorbell carries a producer index, so when a second get follows, its ring
    // publishes this WQE too. Skipping the first ring saves a lock + release fence + two
    // BAR writes; DOCA's in-order sq_ready_index gate guarantees a later ring
    // never exposes a half-written WQE.
    transport.get(
        gin::network_team(), peer,
        src_region.window, gin::region_offset(src_region, src1),
        dst_region.window, gin::region_offset(dst_region, dst1),
        bytes, ncclCoopThread(), ncclGin_None(),
        dst2 != nullptr ? ncclGinOptFlagsAggregateRequests : ncclGinOptFlagsDefault,
        ncclGin_SegmentDevice());
    if (dst2 != nullptr) {
        transport.get(
            gin::network_team(), peer,
            src_region.window, gin::region_offset(src_region, src2),
            dst_region.window, gin::region_offset(dst_region, dst2),
            bytes, ncclCoopThread(), ncclGin_None(),
            ncclGinOptFlagsDefault, ncclGin_SegmentDevice());
    }
    // Settle only the peer we just read from. flush() instead walks all team
    // ranks of this context, issuing a CST + CQ wait per peer from this single
    // thread, and would also block on unrelated in-flight reads posted by CTAs
    // that share this context.
    ncclGinRequest_t request;
    transport.flushAsync(gin::network_team(), peer, &request, ncclCoopThread());
    transport.wait(request, ncclCoopThread());
}

__device__ __forceinline__ void two_buffers_getmem_block(
    void* dst1, void* dst2, const void* src1, const void* src2,
    int bytes, int world_peer) {
    __syncthreads();
    if (auto* peer_src1 = gin::get_lsa_ptr(src1, world_peer); peer_src1 != nullptr) {
        // get_lsa_ptr's find_region traps on a null pointer.
        auto* peer_src2 = dst2 != nullptr ? gin::get_lsa_ptr(src2, world_peer) : nullptr;
        copy_two_buffers_block(dst1, dst2, peer_src1, peer_src2, bytes);
        __syncthreads();
        return;
    }

    if (threadIdx.x == 0) {
        network_get_two_buffers(dst1, dst2, src1, src2, bytes, world_peer);
    }
    __syncthreads();
}

// Out of line for the same reason as network_get_two_buffers.
static __device__ __noinline__ void network_put_two_buffers(
    void* dst1, void* dst2, const void* src1, const void* src2,
    int bytes, int world_peer) {
    const auto& src_region = gin::find_region(src1);
    const auto& dst_region = gin::find_region(dst1);
    // Route every CTA's put through the shared ordered context (one QP per
    // peer) so that the winner CTA's later remote_strong_add on the same
    // context can settle ALL these puts to this peer -- see try_commit_rank.
    auto transport = gin::make_gin_ordered();
    transport.put(
        gin::network_team(), gin::team_peer(world_peer),
        dst_region.window, gin::region_offset(dst_region, dst1),
        src_region.window, gin::region_offset(src_region, src1),
        bytes, ncclGin_None(), ncclGin_None(), ncclCoopThread(),
        ncclGin_None(), cuda::thread_scope_thread, cuda::thread_scope_device,
        // A following second put's doorbell publishes this WQE too -- see network_get_two_buffers.
        dst2 != nullptr ? ncclGinOptFlagsAggregateRequests : ncclGinOptFlagsDefault);
    if (dst2 != nullptr) {
        transport.put(
            gin::network_team(), gin::team_peer(world_peer),
            dst_region.window, gin::region_offset(dst_region, dst2),
            src_region.window, gin::region_offset(src_region, src2),
            bytes, ncclGin_None(), ncclGin_None(), ncclCoopThread(),
            ncclGin_None(), cuda::thread_scope_thread, cuda::thread_scope_device,
            ncclGinOptFlagsDefault);
    }
}

__device__ __forceinline__ void two_buffers_putmem_block(
    void* dst1, void* dst2, const void* src1, const void* src2,
    int bytes, int world_peer) {
    __syncthreads();
    if (auto* peer_dst1 = gin::get_lsa_ptr(dst1, world_peer); peer_dst1 != nullptr) {
        auto* peer_dst2 = dst2 != nullptr ? gin::get_lsa_ptr(dst2, world_peer) : nullptr;
        copy_two_buffers_block(peer_dst1, peer_dst2, src1, src2, bytes);
        // System scope: cross-node, the commit signal for these stores goes through the NIC.
        __threadfence_system();
        __syncthreads();
        return;
    }

    if (threadIdx.x == 0) {
        network_put_two_buffers(dst1, dst2, src1, src2, bytes, world_peer);
    }
    __syncthreads();
}

}  // namespace flashmask::shmem
