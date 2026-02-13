cmake_minimum_required(VERSION 3.20)

project(gddbg_runtime LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

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
    CUDA_ARCHITECTURES "80"
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS OFF
)

# Installation
install(TARGETS gddbg_runtime
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(FILES gddbg_runtime.h
    DESTINATION include
)

install(FILES ../common/trace_format.h
    DESTINATION include
)
