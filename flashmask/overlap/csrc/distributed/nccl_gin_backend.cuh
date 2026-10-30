// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 The FlashMask Authors. All rights reserved.

#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#if defined(FLASHMASK_GIN_DEVICE_API)

#include <cuda/atomic>
#include <nccl.h>
#include <nccl_device.h>

#if NCCL_VERSION_CODE < NCCL_VERSION(2, 30, 5)
#error "FlashMask overlap requires NCCL 2.30.5 or newer"
#endif

#endif  // FLASHMASK_GIN_DEVICE_API

namespace flashmask::gin {

#if defined(FLASHMASK_GIN_DEVICE_API)

constexpr int kMaxRegions = 4;
constexpr int kDefaultNumQPs = 17;

struct DeviceRegion {
    uintptr_t base;
    size_t bytes;
    ncclWindow_t window;
};

struct DeviceState {
    ncclDevComm_t dev_comm;
    DeviceRegion regions[kMaxRegions];
    int num_regions;
    int world_rank;
    int world_size;
    int num_lsa_ranks;
    int lsa_rank;
    int rail_rank;
    int num_qps;
    bool use_rail;
};

#endif  // FLASHMASK_GIN_DEVICE_API

enum class Compare {
    Equal,
    NotEqual,
    GreaterThan,
};

class Context;

struct ContextDeleter {
    void operator()(Context* context) const noexcept;
};

using ContextPtr = std::unique_ptr<Context, ContextDeleter>;

std::vector<uint8_t> get_unique_id();
ContextPtr create_context(const uint8_t* unique_id, int rank, int nranks);
int rank(const Context& context);
int nranks(const Context& context);
int num_lsa_ranks(const Context& context);
int lsa_rank(const Context& context);
bool use_rail(const Context& context);

void* alloc(Context& context, size_t bytes);
void free(Context& context, void* ptr);
void free_noexcept(Context& context, void* ptr) noexcept;
void barrier(Context& context);
void wait_until_on_stream(int64_t* ptr, Compare compare, int64_t value, cudaStream_t stream);

#if defined(FLASHMASK_GIN_DEVICE_API)

extern __device__ __constant__ DeviceState g_device_state;

__device__ __forceinline__ const DeviceRegion& find_region(const void* ptr) {
    const auto address = reinterpret_cast<uintptr_t>(ptr);
    for (int i = 0; i < g_device_state.num_regions; ++i) {
        const auto& region = g_device_state.regions[i];
        if (address >= region.base && address < region.base + region.bytes) {
            return region;
        }
    }
    asm volatile("trap;");
    return g_device_state.regions[0];
}

__device__ __forceinline__ size_t region_offset(const DeviceRegion& region, const void* ptr) {
    return reinterpret_cast<uintptr_t>(ptr) - region.base;
}

__device__ __forceinline__ bool is_lsa_peer(int world_peer) {
    return world_peer / g_device_state.num_lsa_ranks ==
           g_device_state.world_rank / g_device_state.num_lsa_ranks;
}

__device__ __forceinline__ void* get_lsa_ptr(const void* local_ptr, int world_peer) {
    if (!is_lsa_peer(world_peer)) {
        return nullptr;
    }
    const auto& region = find_region(local_ptr);
    const int peer_lsa_rank = world_peer % g_device_state.num_lsa_ranks;
    return ncclGetLsaPointer(region.window, region_offset(region, local_ptr), peer_lsa_rank);
}

__device__ __forceinline__ int qp_index() {
    if (g_device_state.num_qps <= 1) {
        return 0;
    }
    return 1 + static_cast<int>(blockIdx.x % (g_device_state.num_qps - 1));
}

// Cross-node transport always uses the World team so the NIC path can reach
// every peer (same-node included) on a single coherence domain. Hierarchical
// vs flat differs only in fetch schedule, not in transport.
__device__ __forceinline__ int team_peer(int world_peer) {
    return world_peer;
}

__device__ __forceinline__ ncclTeam network_team() {
    return ncclTeamWorld(g_device_state.dev_comm);
}

// One GIN context is shared by several CTAs (launch geometry uses more CTAs
// than we allocate contexts), so the resource-sharing mode must be
// device-scope: NCCL_GIN_RESOURCE_SHARING_CTA lowers to block-scope locks and
// atomics on the shared QP submission queue, which provides no mutual
// exclusion between CTAs.
__device__ __forceinline__ ncclGin make_gin() {
    return ncclGin(g_device_state.dev_comm, qp_index(), NCCL_GIN_RESOURCE_SHARING_GPU);
}

// Dedicated context for the *ordered* remote-put path.
//
// qp_index() only ever returns 1..num_qps-1 (see above), so context 0 is unused
// by the per-CTA path and is reserved here. All CTAs share this single context
// with GPU-wide resource sharing, so every put to a given peer traverses the
// context's one-QP-per-peer connection. 
constexpr int kOrderedPutContext = 0;

__device__ __forceinline__ ncclGin make_gin_ordered() {
    return ncclGin(g_device_state.dev_comm, 0, NCCL_GIN_RESOURCE_SHARING_GPU);
}

// Drain THIS thread's outstanding puts on the ordered context (context 0) to
// LOCAL completion -- i.e. the put source buffers become safe to reuse.
// Out of line: see network_get_two_buffers in gin_copy_utils.cuh.
static __device__ __noinline__ void flush_ordered() {
    make_gin_ordered().flush(ncclCoopThread());
}

// Transport dispatch is one binary rule:
//   single node -> all peers directly addressable -> GPU system-scope atomics.
//   cross node   -> route EVERY peer (same-node included) through the World/NIC
//       path. Mixing NIC and GPU atomics on the same word can lose updates, so
//       we never split by peer locality once the mesh spans nodes.

__device__ __forceinline__ bool mesh_is_cross_node() {
    return g_device_state.world_size > g_device_state.num_lsa_ranks;
}

__device__ __forceinline__ bool force_network_atomics() {
    return mesh_is_cross_node();
}

template <typename T>
__device__ __forceinline__ void remote_add(T* ptr, T value, int world_peer) {
    if (!force_network_atomics()) {
        if (auto* peer_ptr = static_cast<T*>(get_lsa_ptr(ptr, world_peer)); peer_ptr != nullptr) {
            cuda::atomic_ref<T, cuda::thread_scope_system> atom(*peer_ptr);
            atom.fetch_add(value, cuda::memory_order_release);
            return;
        }
    }

    static_assert(sizeof(T) == sizeof(uint64_t), "GIN VA signal add requires a 64-bit value");
    const auto& region = find_region(ptr);
    auto transport = make_gin();
    transport.signal(
        network_team(), team_peer(world_peer),
        ncclGin_VASignalAdd{region.window, region_offset(region, ptr), static_cast<uint64_t>(value)},
        ncclCoopThread(), ncclGin_None(),
        cuda::thread_scope_thread, cuda::thread_scope_device,
        ncclGinOptFlagsDefault);
}


// Out of line: see network_get_two_buffers in gin_copy_utils.cuh. Called from
// the RS put kernel's commit path, which is register-starved for the same reason.
template <typename T>
static __device__ __noinline__ void network_strong_signal_add(T* ptr, T value, int world_peer) {
    static_assert(sizeof(T) == sizeof(uint64_t), "GIN VA signal add requires a 64-bit value");
    const auto& region = find_region(ptr);
    auto transport = make_gin_ordered();
    transport.signal(
        network_team(), team_peer(world_peer),
        ncclGin_StrongVASignalAdd{
            region.window, region_offset(region, ptr), static_cast<uint64_t>(value)},
        ncclCoopThread(), ncclGin_None(),
        cuda::thread_scope_thread, cuda::thread_scope_device,
        ncclGinOptFlagsDefault);
}

// Ordered variant of remote_add: routes the aggregate signal through the shared
// kOrderedPutContext (one QP per peer, GPU-wide sharing) and uses a *strong* VA
// signal. Because every per-CTA put also travels the ordered context (see
// two_buffers_putmem_block), the strong signal's arrival guarantees that ALL
// preceding puts to this peer -- from any CTA -- have settled in remote memory.
// This is the cross-CTA, cross-node ordering that gin::fence() (a GPU-local
// __threadfence_system) could never provide on the RDMA path.
template <typename T>
__device__ __forceinline__ void remote_strong_add(T* ptr, T value, int world_peer) {
    if (!force_network_atomics()) {
        if (auto* peer_ptr = static_cast<T*>(get_lsa_ptr(ptr, world_peer)); peer_ptr != nullptr) {
            cuda::atomic_ref<T, cuda::thread_scope_system> atom(*peer_ptr);
            atom.fetch_add(value, cuda::memory_order_release);
            return;
        }
    }
    network_strong_signal_add(ptr, value, world_peer);
}

template <typename T>
__device__ __forceinline__ void remote_store(T* ptr, T value, int world_peer) {
    if (!force_network_atomics()) {
        if (auto* peer_ptr = static_cast<T*>(get_lsa_ptr(ptr, world_peer)); peer_ptr != nullptr) {
            cuda::atomic_ref<T, cuda::thread_scope_system> atom(*peer_ptr);
            atom.store(value, cuda::memory_order_release);
            return;
        }
    }

    const auto& region = find_region(ptr);
    auto transport = make_gin();
    transport.putValue(
        network_team(), team_peer(world_peer),
        region.window, region_offset(region, ptr), value,
        ncclGin_None(), ncclCoopThread(), ncclGin_None(),
        cuda::thread_scope_thread, cuda::thread_scope_device,
        ncclGinOptFlagsDefault);
}

__device__ __forceinline__ void remote_or(int64_t* ptr, int64_t value, int world_peer) {
    auto* peer_ptr = static_cast<int64_t*>(get_lsa_ptr(ptr, world_peer));
    if (peer_ptr == nullptr) {
        asm volatile("trap;");
        return;
    }
    cuda::atomic_ref<int64_t, cuda::thread_scope_system> atom(*peer_ptr);
    atom.fetch_or(value, cuda::memory_order_release);
}

__device__ __forceinline__ int64_t load_acquire(const int64_t* ptr) {
    cuda::atomic_ref<const int64_t, cuda::thread_scope_system> atom(*ptr);
    return atom.load(cuda::memory_order_acquire);
}

__device__ __forceinline__ void wait_until(const int64_t* ptr, Compare compare, int64_t value) {
    while (true) {
        const int64_t current = load_acquire(ptr);
        if ((compare == Compare::Equal && current == value) ||
            (compare == Compare::NotEqual && current != value) ||
            (compare == Compare::GreaterThan && current > value)) {
            return;
        }
    }
}

__device__ __forceinline__ void fence() {
    __threadfence_system();
}

#endif  // FLASHMASK_GIN_DEVICE_API

}  // namespace flashmask::gin
