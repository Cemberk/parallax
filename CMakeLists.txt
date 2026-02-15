cmake_minimum_required(VERSION 3.20)

# GPU backend options (set before project() so languages are configured correctly)
option(PRLX_ENABLE_CUDA "Build CUDA runtime" ON)
option(PRLX_ENABLE_HIP "Build HIP/ROCm runtime" OFF)

# Configure project languages based on enabled backends
set(_PRLX_LANGUAGES CXX)
if(PRLX_ENABLE_CUDA)
    list(APPEND _PRLX_LANGUAGES CUDA)
endif()

project(prlx LANGUAGES ${_PRLX_LANGUAGES})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(PRLX_ENABLE_CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    # Configurable CUDA architectures
    # SM_70=Volta/V100, SM_80=Ampere/A100, SM_90=Hopper/H100, SM_120=Blackwell/50xx
    set(PRLX_CUDA_ARCHITECTURES "70;80;90" CACHE STRING "CUDA architectures to build for")
endif()

# Export compile commands for editor integration
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find CUDA (when enabled)
if(PRLX_ENABLE_CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()

# Find HIP/ROCm (when enabled)
if(PRLX_ENABLE_HIP)
    list(APPEND CMAKE_PREFIX_PATH /opt/rocm)
    find_package(hip QUIET)
    if(hip_FOUND)
        message(STATUS "Found HIP: ${hip_VERSION}")
    else()
        message(WARNING "HIP not found. HIP runtime will not be built.")
        set(PRLX_ENABLE_HIP OFF)
    endif()
endif()

# Workaround for missing zstd CMake config
# LLVM was built with zstd but the development package may not be installed
# Create the imported target manually BEFORE finding LLVM
# Look for versioned library since libzstd.so symlink requires -dev package
find_library(ZSTD_LIBRARY
    NAMES libzstd.so.1 libzstd.so.1.5.5 zstd
    PATHS /usr/lib/x86_64-linux-gnu /usr/lib /usr/local/lib
)
if(ZSTD_LIBRARY)
    add_library(zstd::libzstd_shared SHARED IMPORTED GLOBAL)
    set_target_properties(zstd::libzstd_shared PROPERTIES
        IMPORTED_LOCATION "${ZSTD_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "/usr/include"
    )
    message(STATUS "Manually configured zstd: ${ZSTD_LIBRARY}")
else()
    message(WARNING "Could not find libzstd, LLVM detection may fail")
endif()

# Find LLVM (for the pass)
# User can skip auto-detection by setting PRLX_LLVM_DIR to the directory
# containing LLVMConfig.cmake (e.g., -DPRLX_LLVM_DIR=/opt/llvm-20/lib/cmake/llvm)
set(PRLX_LLVM_DIR "" CACHE PATH "Explicit path to LLVMConfig.cmake directory (skips auto-detection)")
set(BUILD_LLVM_PASS OFF)

if(PRLX_LLVM_DIR)
    find_package(LLVM REQUIRED CONFIG PATHS "${PRLX_LLVM_DIR}" NO_DEFAULT_PATH)
    if(LLVM_VERSION_MAJOR LESS 18)
        message(FATAL_ERROR "PRLX requires LLVM >= 18, found ${LLVM_PACKAGE_VERSION}")
    endif()
    message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION} (user-specified)")
    message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
    set(BUILD_LLVM_PASS ON)
    set(PRLX_LLVM_VERSION ${LLVM_VERSION_MAJOR})
else()
    # Auto-detect: prefer LLVM 20 (closest to Triton's bundled LLVM 22), then 18 as fallback
    foreach(_LLVM_VER 20 19 18)
        if(NOT BUILD_LLVM_PASS)
            # Don't pass version to find_package — LLVM's cmake version compatibility
            # checks are too strict. We verify the major version ourselves.
            unset(LLVM_FOUND CACHE)
            unset(LLVM_DIR CACHE)
            find_package(LLVM CONFIG
                PATHS "/usr/lib/llvm-${_LLVM_VER}/cmake"
                NO_DEFAULT_PATH
            )
            if(LLVM_FOUND AND LLVM_VERSION_MAJOR EQUAL ${_LLVM_VER})
                message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
                message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
                set(BUILD_LLVM_PASS ON)
                set(PRLX_LLVM_VERSION ${_LLVM_VER})
            endif()
        endif()
    endforeach()

    # Last resort: let CMake search its default paths (Spack, Conda, Homebrew, etc.)
    if(NOT BUILD_LLVM_PASS)
        unset(LLVM_FOUND CACHE)
        unset(LLVM_DIR CACHE)
        find_package(LLVM CONFIG QUIET)
        if(LLVM_FOUND AND LLVM_VERSION_MAJOR GREATER_EQUAL 18)
            message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION} via default search")
            message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
            set(BUILD_LLVM_PASS ON)
            set(PRLX_LLVM_VERSION ${LLVM_VERSION_MAJOR})
        endif()
    endif()
endif()

if(NOT BUILD_LLVM_PASS)
    message(WARNING "LLVM 18+ not found. LLVM pass will not be built.")
    message(WARNING "Set -DPRLX_LLVM_DIR=<path> to specify a custom LLVM installation.")
    message(WARNING "Runtime and tests will still be built.")
endif()

# Build runtime library
add_subdirectory(lib/runtime)

# Build LLVM pass (if LLVM is available)
if(BUILD_LLVM_PASS)
    add_subdirectory(lib/pass)
endif()

# NVBit binary instrumentation backend (optional, experimental)
option(PRLX_BUILD_NVBIT "Build NVBit binary instrumentation tool (experimental)" OFF)
if(PRLX_BUILD_NVBIT)
    add_subdirectory(lib/nvbit_tool)
endif()

# Build tests
enable_testing()
add_subdirectory(tests)

# Build tools
add_subdirectory(tools)

# Build examples (validation scenarios)
add_subdirectory(examples)
