cmake_minimum_required(VERSION 3.20)

project(gpu-diffdbg LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Export compile commands for editor integration
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

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
# Accept any LLVM installation, then check version
find_package(LLVM CONFIG)
if(LLVM_FOUND)
    if(LLVM_VERSION_MAJOR EQUAL 18)
        message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
        message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
        set(BUILD_LLVM_PASS ON)
    else()
        message(WARNING "Found LLVM ${LLVM_PACKAGE_VERSION}, but need version 18.x")
        message(WARNING "Runtime and tests will still be built.")
        set(BUILD_LLVM_PASS OFF)
    endif()
else()
    message(WARNING "LLVM not found. LLVM pass will not be built.")
    message(WARNING "Runtime and tests will still be built.")
    set(BUILD_LLVM_PASS OFF)
endif()

# Build runtime library
add_subdirectory(lib/runtime)

# Build LLVM pass (if LLVM is available)
if(BUILD_LLVM_PASS)
    add_subdirectory(lib/pass)
endif()

# Build tests
enable_testing()
add_subdirectory(tests)

# Build tools
add_subdirectory(tools)
