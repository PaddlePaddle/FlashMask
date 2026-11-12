// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 The FlashMask Authors. All rights reserved.

#include "nccl_gin_backend.cuh"
#include "debug_logger.cuh"

#include <dlfcn.h>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace flashmask::gin {

__device__ __constant__ DeviceState g_device_state;

namespace {

// The libnccl this process actually bound, so a version mismatch names the
// offending file instead of leaving the caller to guess.
const char* loaded_nccl_path() {
    Dl_info info{};
    return dladdr(reinterpret_cast<const void*>(&ncclGetVersion), &info) ? info.dli_fname : "?";
}

#define FM_NCCL_CHECK(call)                                                                    \
    do {                                                                                        \
        const ncclResult_t result = (call);                                                     \
        if (result != ncclSuccess) {                                                            \
            throw std::runtime_error(std::string("NCCL error: ") + ncclGetErrorString(result)); \
        }                                                                                       \
    } while (0)

struct HostRegion {
    void* raw_ptr = nullptr;
    void* mapped_ptr = nullptr;
    size_t bytes = 0;
    ncclWindow_t window = nullptr;
};

// FLASHMASK_SPIN_TIMEOUT_S: device spin budget in seconds (0 disables it).
unsigned long long spin_timeout_ns_from_env() {
    constexpr unsigned long long kDefaultSeconds = 300;
    constexpr unsigned long long kNsPerSecond = 1000ull * 1000ull * 1000ull;
    const char* text = std::getenv("FLASHMASK_SPIN_TIMEOUT_S");
    if (text == nullptr || *text == '\0') {
        return kDefaultSeconds * kNsPerSecond;
    }
    errno = 0;
    char* end = nullptr;
    const unsigned long long seconds = std::strtoull(text, &end, 10);
    if (*text == '-' || *end != '\0' || errno != 0 || seconds > ~0ull / kNsPerSecond) {
        throw std::invalid_argument(
            std::string("FLASHMASK_SPIN_TIMEOUT_S must be a non-negative integer number of "
                        "seconds, got: ") + text);
    }
    return seconds * kNsPerSecond;
}

// One strong signal per (GIN context, peer): it lands only after every earlier put on that
// context to that peer, so counting all of them means the drain is complete.
__global__ void DrainSendKernel(uint64_t* drain_signal) {
    if (static_cast<int>(threadIdx.x) == g_device_state.world_rank) {
        return;
    }
    const auto& region = find_region(drain_signal);
    ncclGin(g_device_state.dev_comm, static_cast<int>(blockIdx.x), NCCL_GIN_RESOURCE_SHARING_GPU)
        .signal(network_team(), team_peer(static_cast<int>(threadIdx.x)),
                ncclGin_StrongVASignalAdd{region.window, region_offset(region, drain_signal), 1},
                ncclCoopThread(), ncclGin_None(),
                cuda::thread_scope_thread, cuda::thread_scope_device,
                ncclGinOptFlagsDefault);
}

__global__ void DrainWaitKernel(const uint64_t* drain_signal, uint64_t want) {
    signal_wait_exact(drain_signal, want, "drain");
}

}  // namespace

class Context {
public:
    Context() = default;

    ~Context() noexcept { destroy(); }

    void initialize(const uint8_t* unique_id, int rank, int nranks) {
        if (unique_id == nullptr) {
            throw std::invalid_argument("NCCL unique ID must not be null");
        }
        rank_ = rank;
        nranks_ = nranks;
        regions_.reserve(kMaxRegions);
        spin_timeout_ns_ = spin_timeout_ns_from_env();

        int runtime_version = 0;
        FM_NCCL_CHECK(ncclGetVersion(&runtime_version));
        if (runtime_version != NCCL_VERSION_CODE) {
            throw std::runtime_error(
                "FlashMask GIN requires matching compile-time and runtime NCCL versions: compiled=" +
                std::to_string(NCCL_VERSION_CODE) + ", runtime=" + std::to_string(runtime_version) +
                ", loaded=" + loaded_nccl_path() +
                ". nccl_device.h inlines the GIN device API, so the process must bind exactly the "
                "libnccl it was built against: put that directory first in LD_LIBRARY_PATH.");
        }

        ncclUniqueId id;
        std::memcpy(&id, unique_id, sizeof(id));
        FM_NCCL_CHECK(ncclCommInitRank(&comm_, nranks_, id, rank_));

        const ncclTeam_t lsa = ncclTeamLsa(comm_);
        num_lsa_ranks_ = lsa.nRanks;
        lsa_rank_ = lsa.rank;
        if (num_lsa_ranks_ <= 0 || nranks_ % num_lsa_ranks_ != 0 ||
            lsa_rank_ != rank_ % num_lsa_ranks_) {
            throw std::runtime_error("NCCL LSA topology does not match FlashMask rank layout");
        }
        rail_rank_ = rank_ / num_lsa_ranks_;
        use_rail_ = nranks_ > num_lsa_ranks_;

        ncclCommProperties props = NCCL_COMM_PROPERTIES_INITIALIZER;
        FM_NCCL_CHECK(ncclCommQueryProperties(comm_, &props));

        ncclDevCommRequirements_t requirements = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
        const bool needs_network = use_rail_;
        if (needs_network) {
            // Cross-node transport always uses full-mesh GIN so the World team can
            // reach every peer (same-node included) on a single coherence domain.
            if (props.ginType == NCCL_GIN_TYPE_NONE) {
                throw std::runtime_error("NCCL full-mesh GIN is unavailable for FlashMask overlap");
            }
            requirements.ginContextCount = kDefaultNumQPs;
            requirements.ginExclusiveContexts = true;
            requirements.ginQueueDepth = 256;
            requirements.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
        }

        FM_NCCL_CHECK(ncclDevCommCreate(comm_, &requirements, &dev_comm_));
        dev_comm_created_ = true;
        // ginContextCount is only a hint: NCCL may grant fewer contexts than
        // requested (older GIN backends clamp it to the connection count).
        // qp_index() indexes contexts modulo num_qps, so it must reflect what
        // was actually granted, otherwise we address contexts out of range.
        num_qps_ = needs_network ? static_cast<int>(dev_comm_.ginContextCount) : 0;
        log_topology(props);
        FM_CUDA_CHECK(cudaMalloc(&barrier_value_, sizeof(int)));
        FM_CUDA_CHECK(cudaMemset(barrier_value_, 0, sizeof(int)));
        publish_device_state();
    }

    void* allocate(size_t bytes, WindowKind kind) {
        if (bytes == 0 || regions_.size() >= kMaxRegions) {
            throw std::invalid_argument("Invalid FlashMask NCCL symmetric allocation");
        }

        HostRegion region;
        region.bytes = bytes;
        FM_NCCL_CHECK(ncclMemAlloc(&region.raw_ptr, bytes));

        // STRICT_ORDERING only affects NIC traffic; single node uses LSA atomics only.
        const bool strict = kind == WindowKind::Signal && use_rail_;
        const ncclResult_t register_result = ncclCommWindowRegister(
            comm_, region.raw_ptr, bytes, &region.window,
            strict ? NCCL_WIN_STRICT_ORDERING : NCCL_WIN_DEFAULT);
        if (register_result != ncclSuccess) {
            ncclMemFree(region.raw_ptr);
            throw std::runtime_error(
                std::string("NCCL error: ") + ncclGetErrorString(register_result));
        }

        const ncclResult_t pointer_result = ncclGetLsaDevicePointer(
            region.window, 0, lsa_rank_, &region.mapped_ptr);
        if (pointer_result != ncclSuccess) {
            ncclCommWindowDeregister(comm_, region.window);
            ncclMemFree(region.raw_ptr);
            throw std::runtime_error(
                std::string("NCCL error: ") + ncclGetErrorString(pointer_result));
        }

        regions_.push_back(region);
        publish_device_state();
        return region.mapped_ptr;
    }

    void release(void* ptr) {
        const auto it = std::find_if(regions_.begin(), regions_.end(),
            [ptr](const HostRegion& region) { return region.mapped_ptr == ptr; });
        if (it == regions_.end()) {
            throw std::invalid_argument("Unknown FlashMask NCCL symmetric allocation");
        }

        FM_NCCL_CHECK(ncclCommWindowDeregister(comm_, it->window));
        FM_NCCL_CHECK(ncclMemFree(it->raw_ptr));
        regions_.erase(it);
        publish_device_state();
    }

    void release_noexcept(void* ptr) noexcept {
        const auto it = std::find_if(regions_.begin(), regions_.end(),
            [ptr](const HostRegion& region) { return region.mapped_ptr == ptr; });
        if (it == regions_.end()) {
            return;
        }
        ncclCommWindowDeregister(comm_, it->window);
        ncclMemFree(it->raw_ptr);
        regions_.erase(it);
    }

    void barrier_on_stream(cudaStream_t stream) {
        FM_NCCL_CHECK(ncclAllReduce(
            barrier_value_, barrier_value_, 1, ncclInt, ncclSum, comm_, stream));
    }

    void barrier() {
        barrier_on_stream(nullptr);
        FM_CUDA_CHECK(cudaStreamSynchronize(nullptr));
    }

    // Single node: all remote accesses came from finished kernels, so a device sync suffices.
    uint64_t quiesce(uint64_t* drain_signal, uint64_t drained) {
        FM_CUDA_CHECK(cudaDeviceSynchronize());
        if (num_qps_ > 0) {
            DrainSendKernel<<<num_qps_, nranks_>>>(drain_signal);
            FM_CUDA_CHECK(cudaGetLastError());
            drained += static_cast<uint64_t>(num_qps_) * static_cast<uint64_t>(nranks_ - 1);
            DrainWaitKernel<<<1, 1>>>(drain_signal, drained);
            FM_CUDA_CHECK(cudaGetLastError());
            FM_CUDA_CHECK(cudaDeviceSynchronize());
        }
        barrier();
        return drained;
    }

    int rank() const { return rank_; }
    int nranks() const { return nranks_; }
    int num_lsa_ranks() const { return num_lsa_ranks_; }
    size_t window_bytes() const {
        size_t total = 0;
        for (const HostRegion& region : regions_) total += region.bytes;
        return total;
    }

    void destroy() noexcept {
        if (comm_ == nullptr) {
            return;
        }
        cudaDeviceSynchronize();
        for (const HostRegion& region : regions_) {
            ncclCommWindowDeregister(comm_, region.window);
            ncclMemFree(region.raw_ptr);
        }
        regions_.clear();
        if (dev_comm_created_) {
            ncclDevCommDestroy(comm_, &dev_comm_);
            dev_comm_created_ = false;
        }
        if (barrier_value_ != nullptr) {
            cudaFree(barrier_value_);
            barrier_value_ = nullptr;
        }
        // ncclCommDestroy waits in an intra-node barrier for every local rank of the communicator;
        // ncclCommAbort releases this rank's share without them.
        if (local_teardown_) {
            ncclCommAbort(comm_);
        } else {
            ncclCommDestroy(comm_);
        }
        comm_ = nullptr;
    }

    void set_local_teardown() noexcept { local_teardown_ = true; }

private:
    // One line per rank at init. This is the only place that knows the real
    // transport shape, and every "why is the network path slow" question starts
    // with num_lsa_ranks (a degenerate LSA domain of 1 silently routes every
    // chunk over the NIC while staying bit-exact) and the granted context count.
    void log_topology(const ncclCommProperties& props) const {
        std::fprintf(stderr,
            "[FlashMask GIN] rank %d/%d lsa_rank %d/%d rail_rank %d use_rail %d "
            "ginType %d railedGinType %d ginContexts %u ginConnections %u\n",
            rank_, nranks_, lsa_rank_, num_lsa_ranks_, rail_rank_,
            static_cast<int>(use_rail_),
            static_cast<int>(props.ginType), static_cast<int>(props.railedGinType),
            static_cast<unsigned>(dev_comm_.ginContextCount),
            static_cast<unsigned>(dev_comm_.ginConnectionCount));
        std::fflush(stderr);
    }

    void publish_device_state() {
        DeviceState state{};
        state.dev_comm = dev_comm_;
        state.num_regions = static_cast<int>(regions_.size());
        state.world_rank = rank_;
        state.world_size = nranks_;
        state.num_lsa_ranks = num_lsa_ranks_;
        state.lsa_rank = lsa_rank_;
        state.rail_rank = rail_rank_;
        state.num_qps = num_qps_;
        state.use_rail = use_rail_;
        state.spin_timeout_ns = spin_timeout_ns_;
        for (int i = 0; i < state.num_regions; ++i) {
            state.regions[i] = {
                reinterpret_cast<uintptr_t>(regions_[i].mapped_ptr),
                regions_[i].bytes,
                regions_[i].window,
            };
        }
        FM_CUDA_CHECK(cudaMemcpyToSymbol(g_device_state, &state, sizeof(state)));
    }

    ncclComm_t comm_ = nullptr;
    ncclDevComm_t dev_comm_{};
    bool dev_comm_created_ = false;
    int* barrier_value_ = nullptr;
    int rank_ = 0;
    int nranks_ = 0;
    int num_lsa_ranks_ = 0;
    int lsa_rank_ = 0;
    int rail_rank_ = 0;
    int num_qps_ = 0;
    bool use_rail_ = false;
    unsigned long long spin_timeout_ns_ = 0;
    bool local_teardown_ = false;
    std::vector<HostRegion> regions_;
};

ContextPtr create_context(const uint8_t* unique_id, int rank, int nranks) {
    ContextPtr context(new Context());
    context->initialize(unique_id, rank, nranks);
    return context;
}

void ContextDeleter::operator()(Context* context) const noexcept {
    delete context;
}

int rank(const Context& context) { return context.rank(); }
int nranks(const Context& context) { return context.nranks(); }
int num_lsa_ranks(const Context& context) { return context.num_lsa_ranks(); }
void set_local_teardown(Context& context) noexcept { context.set_local_teardown(); }
void* alloc(Context& context, size_t bytes, WindowKind kind) { return context.allocate(bytes, kind); }
void free(Context& context, void* ptr) { context.release(ptr); }
void free_noexcept(Context& context, void* ptr) noexcept { context.release_noexcept(ptr); }
void barrier(Context& context) { context.barrier(); }
uint64_t quiesce(Context& context, uint64_t* drain_signal, uint64_t drained) {
    return context.quiesce(drain_signal, drained);
}
size_t window_bytes(const Context& context) { return context.window_bytes(); }

}  // namespace flashmask::gin
