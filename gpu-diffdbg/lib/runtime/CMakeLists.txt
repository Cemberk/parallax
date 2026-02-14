cmake_minimum_required(VERSION 3.20)

project(gddbg_runtime LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Optional zstd compression
option(GDDBG_ENABLE_COMPRESSION "Enable zstd trace compression" ON)
if(GDDBG_ENABLE_COMPRESSION)
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
    gddbg_host.cu
    gddbg_runtime.cu
)

# Create static library with both host and device code
add_library(gddbg_runtime STATIC ${RUNTIME_SOURCES})

# Include directories
target_include_directories(gddbg_runtime PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
)

# Link CUDA runtime
target_link_libraries(gddbg_runtime PUBLIC
    CUDA::cudart
)

# Set CUDA architecture and enable separable compilation
# Do NOT resolve device symbols in the library - let the final executable do it
set_target_properties(gddbg_runtime PROPERTIES
    CUDA_ARCHITECTURES "${GDDBG_CUDA_ARCHITECTURES}"
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS OFF
)

# Shared library for Python FFI (ctypes)
add_library(gddbg_runtime_shared SHARED ${RUNTIME_SOURCES})

target_include_directories(gddbg_runtime_shared PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
)

target_link_libraries(gddbg_runtime_shared PUBLIC
    CUDA::cudart
)

set_target_properties(gddbg_runtime_shared PROPERTIES
    CUDA_ARCHITECTURES "${GDDBG_CUDA_ARCHITECTURES}"
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    OUTPUT_NAME "gddbg_runtime_shared"
)

# Link zstd to both static and shared targets when available
if(GDDBG_ENABLE_COMPRESSION AND ZSTD_FOUND)
    target_compile_definitions(gddbg_runtime PRIVATE GDDBG_HAS_ZSTD)
    target_include_directories(gddbg_runtime PRIVATE ${ZSTD_INCLUDE_DIRS})
    target_link_libraries(gddbg_runtime PRIVATE ${ZSTD_LIBRARIES})

    target_compile_definitions(gddbg_runtime_shared PRIVATE GDDBG_HAS_ZSTD)
    target_include_directories(gddbg_runtime_shared PRIVATE ${ZSTD_INCLUDE_DIRS})
    target_link_libraries(gddbg_runtime_shared PRIVATE ${ZSTD_LIBRARIES})

    message(STATUS "zstd compression enabled")
else()
    message(STATUS "zstd compression disabled (GDDBG_ENABLE_COMPRESSION=${GDDBG_ENABLE_COMPRESSION}, ZSTD_FOUND=${ZSTD_FOUND})")
endif()

# NVPTX bitcode for Triton integration (device-only runtime)
# Compiled with the highest available clang to match Triton's LLVM IR format.
# Prefer clang-20 (closer to Triton's bundled LLVM 22), fall back to clang-18.
set(GDDBG_CLANG "")
foreach(_CLANG_VER clang++-20 clang++-18)
    if(NOT GDDBG_CLANG)
        find_program(_FOUND_CLANG ${_CLANG_VER})
        if(_FOUND_CLANG)
            set(GDDBG_CLANG ${_FOUND_CLANG})
        endif()
        unset(_FOUND_CLANG CACHE)
    endif()
endforeach()

if(GDDBG_CLANG)
    set(GDDBG_RUNTIME_BC "${CMAKE_CURRENT_BINARY_DIR}/gddbg_runtime_nvptx.bc")
    add_custom_command(
        OUTPUT ${GDDBG_RUNTIME_BC}
        COMMAND ${GDDBG_CLANG}
            --cuda-device-only
            --cuda-gpu-arch=sm_70
            -emit-llvm -c
            -fcuda-flush-denormals-to-zero
            -I${CMAKE_CURRENT_SOURCE_DIR}
            -I${CMAKE_CURRENT_SOURCE_DIR}/../common
            ${CMAKE_CURRENT_SOURCE_DIR}/gddbg_runtime.cu
            -o ${GDDBG_RUNTIME_BC}
        DEPENDS
            ${CMAKE_CURRENT_SOURCE_DIR}/gddbg_runtime.cu
            ${CMAKE_CURRENT_SOURCE_DIR}/gddbg_runtime.h
            ${CMAKE_CURRENT_SOURCE_DIR}/../common/trace_format.h
        COMMENT "Building NVPTX bitcode for Triton integration (${GDDBG_CLANG})"
    )
    add_custom_target(gddbg_runtime_bitcode ALL DEPENDS ${GDDBG_RUNTIME_BC})
    message(STATUS "NVPTX bitcode target enabled (${GDDBG_CLANG})")
else()
    message(STATUS "No suitable clang found, NVPTX bitcode target disabled")
endif()

# Installation
install(TARGETS gddbg_runtime gddbg_runtime_shared
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(FILES gddbg_runtime.h
    DESTINATION include
)

install(FILES ../common/trace_format.h
    DESTINATION include
)
