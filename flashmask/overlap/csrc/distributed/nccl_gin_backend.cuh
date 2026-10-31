// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 The FlashMask Authors. All rights reserved.

#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <vector>

#include <cuda/atomic>
#include <nccl.h>
#include <nccl_device.h>

#if NCCL_VERSION_CODE < NCCL_VERSION(2, 30, 5)
#error "FlashMask overlap requires NCCL 2.30.5 or newer"
#endif

namespace flashmask::gin {

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
    unsigned long long spin_timeout_ns;   // 0: no time-based trap (see SpinWatchdog)
};

enum class Compare {
    NotEqual,
    GreaterThan,
};

class Context;

struct ContextDeleter {
    void operator()(Context* context) const noexcept;
};

using ContextPtr = std::unique_ptr<Context, ContextDeleter>;

ContextPtr create_context(const uint8_t* unique_id, int rank, int nranks);
int rank(const Context& context);
int nranks(const Context& context);
int num_lsa_ranks(const Context& context);

// Signal windows hold VA signals, which NCCL requires to be registered with
// NCCL_WIN_STRICT_ORDERING once they are reached through GIN (cross-node meshes);
// data windows (payload and putValue flags) use NCCL_WIN_DEFAULT.
enum class WindowKind {
    Data,
    Signal,
};

void* alloc(Context& context, size_t bytes, WindowKind kind = WindowKind::Data);
void free(Context& context, void* ptr);
void free_noexcept(Context& context, void* ptr) noexcept;
void barrier(Context& context);
// Collective. On return, every put, putValue and signal that any rank issued before
// the call has settled in its target's memory, so windows may be freed or rezeroed.
// `drain_signal` is a VA-signal word reserved for this call and `drained` its current
// cumulative count; returns the new count.
uint64_t quiesce(Context& context, uint64_t* drain_signal, uint64_t drained);
// Sum of the requested sizes of every registered window.
size_t window_bytes(const Context& context);

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

// VA signals. Every signal word lives in the communicator's signal window and only
// ever counts up; it is touched only by the adds below, by the loads in
// signal_wait_exact, and by the one-off zeroing before the first barrier. The waiter
// knows the exact cumulative count it expects, so no word is ever reset.

// Refcount add: orders nothing but the caller's own earlier accesses, so the weak
// signal suffices.
__device__ __forceinline__ void remote_add(uint64_t* sig, uint64_t value, int world_peer) {
    if (!force_network_atomics()) {
        if (auto* peer_sig = static_cast<uint64_t*>(get_lsa_ptr(sig, world_peer)); peer_sig != nullptr) {
            cuda::atomic_ref<uint64_t, cuda::thread_scope_system>(*peer_sig)
                .fetch_add(value, cuda::memory_order_release);
            return;
        }
    }

    const auto& region = find_region(sig);
    make_gin().signal(
        network_team(), team_peer(world_peer),
        ncclGin_WeakVASignalAdd{region.window, region_offset(region, sig), value},
        ncclCoopThread(), ncclGin_None(),
        cuda::thread_scope_thread, cuda::thread_scope_device,
        ncclGinOptFlagsDefault);
}

// Out of line: see network_get_two_buffers in gin_copy_utils.cuh. Called from
// the RS put kernel's commit path, which is register-starved for the same reason.
static __device__ __noinline__ void network_strong_signal_add(
        uint64_t* sig, uint64_t value, int world_peer) {
    const auto& region = find_region(sig);
    make_gin_ordered().signal(
        network_team(), team_peer(world_peer),
        ncclGin_StrongVASignalAdd{region.window, region_offset(region, sig), value},
        ncclCoopThread(), ncclGin_None(),
        cuda::thread_scope_thread, cuda::thread_scope_device,
        ncclGinOptFlagsDefault);
}

// Data-ordering add: routed through the shared kOrderedPutContext (one QP per peer,
// GPU-wide sharing) as a *strong* VA signal. Every per-CTA put also travels the
// ordered context (see two_buffers_putmem_block), so the signal's arrival implies
// that ALL preceding puts to this peer -- from any CTA -- have settled in remote
// memory.
__device__ __forceinline__ void remote_strong_add(uint64_t* sig, uint64_t value, int world_peer) {
    if (!force_network_atomics()) {
        if (auto* peer_sig = static_cast<uint64_t*>(get_lsa_ptr(sig, world_peer)); peer_sig != nullptr) {
            cuda::atomic_ref<uint64_t, cuda::thread_scope_system>(*peer_sig)
                .fetch_add(value, cuda::memory_order_release);
            return;
        }
    }
    network_strong_signal_add(sig, value, world_peer);
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

// ---------------------------------------------------------------------------
// Spin watchdog. Every device-side spin in the comm runtime ticks one of these.
// The budget (FLASHMASK_SPIN_TIMEOUT_S, read once at context init) must exceed the
// longest legitimate rank skew, like a framework collective timeout, so a trip means
// a lost signal or a peer that stopped participating.
// Not covered: NIC error completions. They are handled inside NCCL's GIN wait, which
// drops the status without an abort flag (the fetch counts as landed) or retries until
// ncclCommAbort; DOCA only prints the error CQE. The compute-side gates
// (flashmask/cute/barrier.py) apply the same budget.
// ---------------------------------------------------------------------------

constexpr unsigned kSpinCheckInterval = 1024;   // polls between timer reads

static __device__ __noinline__ void spin_trap(const char* site, const char* reason,
        const void* addr, unsigned long long expected, unsigned long long actual,
        unsigned long long waited_ns) {
    printf("[FlashMask spin] %s: %s, rank %d/%d block %d thread %d addr %p "
           "expected %llu actual %llu waited %llu ms\n",
           site, reason, g_device_state.world_rank, g_device_state.world_size,
           static_cast<int>(blockIdx.x), static_cast<int>(threadIdx.x), addr,
           expected, actual, waited_ns / 1000000ull);
    __trap();
}

// Returns the spin's start time. It is taken lazily on the first check, so a wait
// that ends within one interval never reads the timer.
static __device__ __noinline__ unsigned long long spin_watchdog_check(
        unsigned long long start_ns, const char* site, const void* addr,
        unsigned long long expected, unsigned long long actual) {
    const unsigned long long budget_ns = g_device_state.spin_timeout_ns;
    if (budget_ns == 0) {
        return start_ns;
    }
    unsigned long long now;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
    if (start_ns == 0) {
        return now;
    }
    if (now - start_ns > budget_ns) {
        spin_trap(site, "timeout", addr, expected, actual, now - start_ns);
    }
    return start_ns;
}

struct SpinWatchdog {
    unsigned long long start_ns = 0;
    unsigned steps = 0;

    __device__ __forceinline__ void tick(const char* site, const void* addr,
                                         unsigned long long expected, unsigned long long actual) {
        if (++steps % kSpinCheckInterval == 0) {
            start_ns = spin_watchdog_check(start_ns, site, addr, expected, actual);
        }
    }
};

// Spins until done(*ptr). Polls with relaxed system-scope loads, which do not
// invalidate L1 per poll, then re-reads once with an acquire load; that load is what
// orders the caller's later accesses. Returns the acquired value, for which done holds.
template <typename T, typename Done>
__device__ __forceinline__ T spin_until(const T* ptr, Done done, const char* site,
                                        unsigned long long expected) {
    SpinWatchdog watchdog;
    for (;;) {
        const T seen = cuda::atomic_ref<const T, cuda::thread_scope_system>(*ptr)
            .load(cuda::memory_order_relaxed);
        if (done(seen)) {
            const T acquired = cuda::atomic_ref<const T, cuda::thread_scope_system>(*ptr)
                .load(cuda::memory_order_acquire);
            if (done(acquired)) {
                return acquired;
            }
            continue;
        }
        watchdog.tick(site, ptr, expected, static_cast<unsigned long long>(seen));
    }
}

// Flag wait. Out of line so the watchdog state stays out of the register-capped comm
// kernel frames (see network_get_two_buffers in gin_copy_utils.cuh).
static __device__ __noinline__ void wait_until(
        const int64_t* ptr, Compare compare, int64_t value, const char* site) {
    spin_until(ptr, [=](int64_t current) {
        return (compare == Compare::NotEqual && current != value) ||
               (compare == Compare::GreaterThan && current > value);
    }, site, value);
}

// Waits until a monotonic signal counter reaches `want`, the exact cumulative number
// of adds the protocol issued to it (rolling compare, as ncclGin::waitSignal). The
// loads are those of ncclGin::readSignal on a VA signal (gin__funcs.h: a system-scope
// cuda::atomic_ref load), relaxed while polling and acquire for the final read. A count
// past `want` is a duplicated or misrouted signal: trap instead of releasing the waiter
// early.
static __device__ __noinline__ void signal_wait_exact(
        const uint64_t* sig, uint64_t want, const char* site) {
    const uint64_t got = spin_until(sig, [=](uint64_t current) {
        return static_cast<int64_t>(current - want) >= 0;
    }, site, want);
    if (got != want) {
        spin_trap(site, "overshoot", sig, want, got, 0);
    }
}

}  // namespace flashmask::gin
