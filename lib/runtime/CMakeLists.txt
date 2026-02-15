cmake_minimum_required(VERSION 3.20)

project(prlx_runtime LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Optional zstd compression
option(PRLX_ENABLE_COMPRESSION "Enable zstd trace compression" ON)
if(PRLX_ENABLE_COMPRESSION)
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(ZSTD QUIET libzstd)
    endif()
    if(NOT ZSTD_FOUND)
        find_library(ZSTD_LIBRARIES NAMES zstd)
        find_path(ZSTD_INCLUDE_DIRS zstd.h)
        if(ZSTD_LIBRARIES AND ZSTD_INCLUDE_DIRS)
            set(ZSTD_FOUND TRUE)
        endif()
    endif()
endif()

# Runtime library sources
set(RUNTIME_SOURCES
    prlx_host.cu
    prlx_runtime.cu
)

# Create static library with both host and device code
add_library(prlx_runtime STATIC ${RUNTIME_SOURCES})

# Include directories
target_include_directories(prlx_runtime PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
)

# Link CUDA runtime
target_link_libraries(prlx_runtime PUBLIC
    CUDA::cudart
)

# Set CUDA architecture and enable separable compilation
# Do NOT resolve device symbols in the library - let the final executable do it
set_target_properties(prlx_runtime PROPERTIES
    CUDA_ARCHITECTURES "${PRLX_CUDA_ARCHITECTURES}"
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS OFF
)

# Shared library for Python FFI (ctypes)
add_library(prlx_runtime_shared SHARED ${RUNTIME_SOURCES})

target_include_directories(prlx_runtime_shared PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
)

target_link_libraries(prlx_runtime_shared PUBLIC
    CUDA::cudart
)

set_target_properties(prlx_runtime_shared PROPERTIES
    CUDA_ARCHITECTURES "${PRLX_CUDA_ARCHITECTURES}"
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    OUTPUT_NAME "prlx_runtime_shared"
)

# Link zstd to both static and shared targets when available
if(PRLX_ENABLE_COMPRESSION AND ZSTD_FOUND)
    target_compile_definitions(prlx_runtime PRIVATE PRLX_HAS_ZSTD)
    target_include_directories(prlx_runtime PRIVATE ${ZSTD_INCLUDE_DIRS})
    target_link_libraries(prlx_runtime PRIVATE ${ZSTD_LIBRARIES})

    target_compile_definitions(prlx_runtime_shared PRIVATE PRLX_HAS_ZSTD)
    target_include_directories(prlx_runtime_shared PRIVATE ${ZSTD_INCLUDE_DIRS})
    target_link_libraries(prlx_runtime_shared PRIVATE ${ZSTD_LIBRARIES})

    message(STATUS "zstd compression enabled")
else()
    message(STATUS "zstd compression disabled (PRLX_ENABLE_COMPRESSION=${PRLX_ENABLE_COMPRESSION}, ZSTD_FOUND=${ZSTD_FOUND})")
endif()

# NVPTX bitcode for Triton integration (device-only runtime)
# Compiled with the highest available clang to match Triton's LLVM IR format.
# User can skip auto-detection: -DPRLX_CLANG=/opt/llvm-20/bin/clang++
set(PRLX_CLANG "" CACHE FILEPATH "Clang compiler for NVPTX bitcode (skips auto-detection)")
if(NOT PRLX_CLANG)
    # Auto-detect: prefer clang-20 (closer to Triton's bundled LLVM 22), fall back to clang-18.
    foreach(_CLANG_VER clang++-20 clang++-18)
        if(NOT PRLX_CLANG)
            find_program(_FOUND_CLANG ${_CLANG_VER})
            if(_FOUND_CLANG)
                set(PRLX_CLANG ${_FOUND_CLANG} CACHE FILEPATH "Clang compiler for NVPTX bitcode (auto-detected)" FORCE)
            endif()
            unset(_FOUND_CLANG CACHE)
        endif()
    endforeach()
endif()

if(PRLX_CLANG)
    set(PRLX_RUNTIME_BC "${CMAKE_CURRENT_BINARY_DIR}/prlx_runtime_nvptx.bc")
    add_custom_command(
        OUTPUT ${PRLX_RUNTIME_BC}
        COMMAND ${PRLX_CLANG}
            --cuda-device-only
            --cuda-gpu-arch=sm_70
            -emit-llvm -c
            -fcuda-flush-denormals-to-zero
            -I${CMAKE_CURRENT_SOURCE_DIR}
            -I${CMAKE_CURRENT_SOURCE_DIR}/../common
            ${CMAKE_CURRENT_SOURCE_DIR}/prlx_runtime.cu
            -o ${PRLX_RUNTIME_BC}
        DEPENDS
            ${CMAKE_CURRENT_SOURCE_DIR}/prlx_runtime.cu
            ${CMAKE_CURRENT_SOURCE_DIR}/prlx_runtime.h
            ${CMAKE_CURRENT_SOURCE_DIR}/../common/trace_format.h
        COMMENT "Building NVPTX bitcode for Triton integration (${PRLX_CLANG})"
    )
    add_custom_target(prlx_runtime_bitcode ALL DEPENDS ${PRLX_RUNTIME_BC})
    message(STATUS "NVPTX bitcode target enabled (${PRLX_CLANG})")
else()
    message(STATUS "No suitable clang found, NVPTX bitcode target disabled")
endif()

# Installation
install(TARGETS prlx_runtime prlx_runtime_shared
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(FILES prlx_runtime.h
    DESTINATION include
)

install(FILES ../common/trace_format.h
    DESTINATION include
)
