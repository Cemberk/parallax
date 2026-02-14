# NVBit binary instrumentation backend for gpu-diffdbg
#
# This builds libprlx_nvbit.so, a shared library loaded via LD_PRELOAD
# that instruments SASS instructions at runtime using NVBit.
#
# Usage:
#   cmake -B build -DPRLX_BUILD_NVBIT=ON -DPRLX_NVBIT_PATH=/path/to/nvbit_release
#   cmake --build build
#   LD_PRELOAD=./build/lib/nvbit_tool/libprlx_nvbit.so ./my_cuda_app

cmake_minimum_required(VERSION 3.20)

# NVBit SDK location: user must provide via PRLX_NVBIT_PATH or we try FetchContent
set(PRLX_NVBIT_PATH "" CACHE PATH "Path to NVBit release directory (contains nvbit.h, libnvbit.a)")

if(PRLX_NVBIT_PATH AND EXISTS "${PRLX_NVBIT_PATH}/core/nvbit.h")
    message(STATUS "Using NVBit SDK at: ${PRLX_NVBIT_PATH}")
    set(NVBIT_ROOT "${PRLX_NVBIT_PATH}")
else()
    # Try to download NVBit 1.7.1 release
    include(FetchContent)
    set(NVBIT_VERSION "1.7.1")
    set(NVBIT_URL "https://github.com/NVlabs/NVBit/releases/download/${NVBIT_VERSION}/nvbit-Linux-x86_64-${NVBIT_VERSION}.tar.bz2")

    FetchContent_Declare(
        nvbit_sdk
        URL "${NVBIT_URL}"
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(nvbit_sdk)
    set(NVBIT_ROOT "${nvbit_sdk_SOURCE_DIR}")
    message(STATUS "Downloaded NVBit ${NVBIT_VERSION} to: ${NVBIT_ROOT}")
endif()

# Verify NVBit SDK structure
if(NOT EXISTS "${NVBIT_ROOT}/core/nvbit.h")
    message(FATAL_ERROR "NVBit SDK not found at ${NVBIT_ROOT}/core/nvbit.h\n"
        "Set PRLX_NVBIT_PATH to the extracted NVBit release directory.")
endif()

# Import NVBit static library
add_library(nvbit_core STATIC IMPORTED)
set_target_properties(nvbit_core PROPERTIES
    IMPORTED_LOCATION "${NVBIT_ROOT}/core/libnvbit.a"
    INTERFACE_INCLUDE_DIRECTORIES "${NVBIT_ROOT}/core"
)

# Device-side injected functions (compiled with special NVBit flags)
# These are the functions that NVBit inserts at instrumentation points
add_library(prlx_nvbit_inject OBJECT
    inject_funcs.cu
)
target_include_directories(prlx_nvbit_inject PRIVATE
    "${NVBIT_ROOT}/core"
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_SOURCE_DIR}/lib/common"
)
set_target_properties(prlx_nvbit_inject PROPERTIES
    CUDA_ARCHITECTURES "${PRLX_CUDA_ARCHITECTURES}"
    CUDA_SEPARABLE_COMPILATION ON
)
# NVBit requires specific compilation flags for injected functions:
# -maxrregcount=24: limit register usage for injected code
# --keep-device-functions: prevent dead code elimination of injected functions
target_compile_options(prlx_nvbit_inject PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -maxrregcount=24
        -Xptxas -astoolspatch
        --keep-device-functions
        -Xcompiler -Wall
        -Xcompiler -fPIC
    >
)

# Main NVBit tool shared library
add_library(prlx_nvbit SHARED
    nvbit_prlx.cu
    trace_writer.cu
    site_table.cu
)
target_include_directories(prlx_nvbit PRIVATE
    "${NVBIT_ROOT}/core"
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_SOURCE_DIR}/lib/common"
)
target_link_libraries(prlx_nvbit PRIVATE
    nvbit_core
    prlx_nvbit_inject
    CUDA::cuda_driver
    CUDA::cudart_static
    pthread
    dl
)
set_target_properties(prlx_nvbit PROPERTIES
    CUDA_ARCHITECTURES "${PRLX_CUDA_ARCHITECTURES}"
    CUDA_SEPARABLE_COMPILATION ON
    OUTPUT_NAME "prlx_nvbit"
    PREFIX "lib"
)
target_compile_options(prlx_nvbit PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -Xcompiler -Wall
        -Xcompiler -fPIC
    >
)
