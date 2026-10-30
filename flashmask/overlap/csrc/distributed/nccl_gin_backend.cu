// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 The FlashMask Authors. All rights reserved.

#include "nccl_gin_backend.cuh"

#include <dlfcn.h>

#include <algorithm>
#include <cstdio>
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

#define FM_CUDA_CHECK(call)                                                                    \
    do {                                                                                        \
        const cudaError_t result = (call);                                                      \
        if (result != cudaSuccess) {                                                            \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(result)); \
        }                                                                                       \
    } while (0)

struct HostRegion {
    void* raw_ptr = nullptr;
    void* mapped_ptr = nullptr;
    size_t bytes = 0;
    ncclWindow_t window = nullptr;
};

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
        const bool needs_network = nranks_ > num_lsa_ranks_;
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

    void* allocate(size_t bytes) {
        if (bytes == 0 || regions_.size() >= kMaxRegions) {
            throw std::invalid_argument("Invalid FlashMask NCCL symmetric allocation");
        }

        HostRegion region;
        region.bytes = bytes;
        FM_NCCL_CHECK(ncclMemAlloc(&region.raw_ptr, bytes));

        const ncclResult_t register_result = ncclCommWindowRegister(
            comm_, region.raw_ptr, bytes, &region.window, NCCL_WIN_DEFAULT);
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

    int rank() const { return rank_; }
    int nranks() const { return nranks_; }
    int num_lsa_ranks() const { return num_lsa_ranks_; }
    int lsa_rank() const { return lsa_rank_; }
    bool use_rail() const { return use_rail_; }

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
        ncclCommDestroy(comm_);
        comm_ = nullptr;
    }

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
    std::vector<HostRegion> regions_;
};

namespace {

__global__ void WaitUntilKernel(int64_t* ptr, Compare compare, int64_t value) {
    if (threadIdx.x == 0) {
        wait_until(ptr, compare, value);
    }
}

}  // namespace

std::vector<uint8_t> get_unique_id() {
    ncclUniqueId id;
    FM_NCCL_CHECK(ncclGetUniqueId(&id));
    std::vector<uint8_t> result(sizeof(id));
    std::memcpy(result.data(), &id, sizeof(id));
    return result;
}

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
int lsa_rank(const Context& context) { return context.lsa_rank(); }
bool use_rail(const Context& context) { return context.use_rail(); }
void* alloc(Context& context, size_t bytes) { return context.allocate(bytes); }
void free(Context& context, void* ptr) { context.release(ptr); }
void free_noexcept(Context& context, void* ptr) noexcept { context.release_noexcept(ptr); }
void barrier(Context& context) { context.barrier(); }

void wait_until_on_stream(
    int64_t* ptr, Compare compare, int64_t value, cudaStream_t stream) {
    WaitUntilKernel<<<1, 1, 0, stream>>>(ptr, compare, value);
    FM_CUDA_CHECK(cudaGetLastError());
}

}  // namespace flashmask::gin
