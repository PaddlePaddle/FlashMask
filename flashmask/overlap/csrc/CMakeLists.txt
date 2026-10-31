# FM-4 Overlap bridge: standalone sub-CMake.
#
# This is invoked by setup.py via subprocess (cmake configure + build).
# It compiles four TUs into one shared lib:
#   - fm4_overlap_bridge.cu   (extern "C" surface consumed by overlap_runtime.py)
#   - distributed/overlap_comm.cu     (SM-level comm runtime)
#   - distributed/sep_sr_buffer.cu    (send/recv buffer)
#   - distributed/nccl_gin_backend.cu (NCCL communicator/window lifecycle)
# The target links the selected NCCL shared library and keeps the device-link
# step because the device state is shared by the kernel translation units.
#
# 3.18 is the floor: CUDA_ARCHITECTURES accepting arch strings like "90a"/"100a"
# landed in 3.18. The dev box has 3.31.
cmake_minimum_required(VERSION 3.18)
project(fm4_overlap LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

# ---------------------------------------------------------------------------
# Inputs (all passed by setup.py via -D...). No absolute paths baked in here:
# every path is either derived from CMAKE_CURRENT_SOURCE_DIR (relative to this
# file) or supplied by the caller. This keeps the tree relocatable across the
# A100 dev box and the H100/B200 build boxes.
# ---------------------------------------------------------------------------
set(FM4_OVERLAP_CUDA_ARCH "90a" CACHE STRING "target sm arch or list, e.g. 100a or 100a;103a")
set(NCCL_INSTALL_DIR "" CACHE PATH "NCCL install dir (include/ + lib/)")
set(FM4_OVERLAP_CUTLASS_INC "" CACHE PATH "cutlass include dir (must contain cutlass/bfloat16.h)")

if(NOT NCCL_INSTALL_DIR)
    message(FATAL_ERROR "NCCL_INSTALL_DIR is required. Set it to the NCCL installation root.")
endif()

find_path(NCCL_INCLUDE_DIR
    NAMES nccl.h nccl_device.h
    PATHS "${NCCL_INSTALL_DIR}/include"
    NO_DEFAULT_PATH
)
# [NCCL-WHEEL] the pip wheel (nvidia-nccl-cu13) ships only the versioned
# libnccl.so.2 with no libnccl.so dev symlink, so match that name too.
find_library(NCCL_LIBRARY
    NAMES nccl libnccl.so.2
    PATHS "${NCCL_INSTALL_DIR}/lib" "${NCCL_INSTALL_DIR}/lib64"
    NO_DEFAULT_PATH
)
if(NOT NCCL_INCLUDE_DIR OR NOT NCCL_LIBRARY)
    message(FATAL_ERROR "NCCL headers (nccl.h, nccl_device.h) and libnccl.so[.2] are required under ${NCCL_INSTALL_DIR}")
endif()
if(NOT FM4_OVERLAP_CUTLASS_INC)
    message(FATAL_ERROR "FM4_OVERLAP_CUTLASS_INC is required (dir containing cutlass/bfloat16.h).")
endif()

# One SASS per listed arch: sm_100a and sm_103a binaries are not interchangeable,
# so a .so for both needs both. setup.py passes a ';' list; ',' is accepted too.
# The gencode 'compute_' must drop the trailing 'a'
# (compute_90a is illegal; the correct pair is arch=compute_90,code=sm_90a).
string(REPLACE "," ";" FM4_OVERLAP_CUDA_ARCH "${FM4_OVERLAP_CUDA_ARCH}")
set(FM4_OVERLAP_GENCODE "")
foreach(_arch IN LISTS FM4_OVERLAP_CUDA_ARCH)
    string(REGEX REPLACE "a$" "" _arch_num "${_arch}")
    # One token per arch: CMake de-duplicates repeated options such as a bare -gencode.
    list(APPEND FM4_OVERLAP_GENCODE "-gencode=arch=compute_${_arch_num},code=sm_${_arch}")
endforeach()

# ---------------------------------------------------------------------------
# Source locations (relative to this CMakeLists.txt).
#   this file lives at: flashmask/overlap/csrc/CMakeLists.txt
#   the comm runtime lives next to it in distributed/
# ---------------------------------------------------------------------------
set(DIST_DIR "${CMAKE_CURRENT_SOURCE_DIR}/distributed")

# Fail loudly now (configure time) rather than with a cryptic nvcc error later.
foreach(_req
        "${DIST_DIR}/overlap_comm.cu"
        "${DIST_DIR}/sep_sr_buffer.cu"
        "${DIST_DIR}/nccl_gin_backend.cu"
        "${DIST_DIR}/overlap_comm.cuh"
        "${DIST_DIR}/nccl_gin_backend.cuh")
    if(NOT EXISTS "${_req}")
        message(FATAL_ERROR "Required comm runtime source not found: ${_req}\n"
                            "overlap/csrc/distributed/ is part of this repository; "
                            "check that the working tree is complete.")
    endif()
endforeach()

if(NOT EXISTS "${FM4_OVERLAP_CUTLASS_INC}/cutlass/bfloat16.h")
    message(FATAL_ERROR "cutlass/bfloat16.h not found under FM4_OVERLAP_CUTLASS_INC="
                        "${FM4_OVERLAP_CUTLASS_INC}\n"
                        "Install nvidia-cutlass, or point FM4_OVERLAP_CUTLASS_INC at "
                        "a cutlass include dir.")
endif()

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# The bridge shared library.
# ---------------------------------------------------------------------------
add_library(fm4_overlap SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/fm4_overlap_bridge.cu
    ${DIST_DIR}/overlap_comm.cu
    ${DIST_DIR}/sep_sr_buffer.cu
    ${DIST_DIR}/nccl_gin_backend.cu
)

target_compile_options(fm4_overlap PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    -Xcompiler=-fPIC
    -Xcompiler=-O3
    -std=c++17
    --use_fast_math
    ${FM4_OVERLAP_GENCODE}
    -lineinfo
    -DNDEBUG
    --expt-relaxed-constexpr
>)

target_include_directories(fm4_overlap PRIVATE
    ${NCCL_INCLUDE_DIR}
    ${FM4_OVERLAP_CUTLASS_INC}
    ${DIST_DIR}                       # overlap_comm.cuh, sr_buffer.cuh, ...
    ${CMAKE_CURRENT_SOURCE_DIR}       # fm4_overlap_bridge.cu's own dir
)

target_link_libraries(fm4_overlap PRIVATE
    CUDA::cudart
    ${NCCL_LIBRARY}
)

# Export only the fm4_overlap_* C ABI and keep the bridge's implementation
# symbols local to avoid interposing with libpaddle.so or DeepEP.
target_link_options(fm4_overlap PRIVATE
    -Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/fm4_overlap.map
    -Wl,-Bsymbolic
    -Wl,--exclude-libs,ALL
)

set_target_properties(fm4_overlap PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD_REQUIRED ON
    CXX_STANDARD 17
    CUDA_STANDARD 17
    CUDA_SEPARABLE_COMPILATION ON     # = -rdc=true (shared GIN device state)
    CUDA_RESOLVE_DEVICE_SYMBOLS ON    # = device-link, avoids runtime PTX JIT
    CUDA_ARCHITECTURES "${FM4_OVERLAP_CUDA_ARCH}"
    BUILD_RPATH "$ORIGIN;${NCCL_INSTALL_DIR}/lib;${NCCL_INSTALL_DIR}/lib64"
    INSTALL_RPATH "$ORIGIN;${NCCL_INSTALL_DIR}/lib;${NCCL_INSTALL_DIR}/lib64"
    BUILD_WITH_INSTALL_RPATH ON
)

