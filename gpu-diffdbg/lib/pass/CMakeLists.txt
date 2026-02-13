cmake_minimum_required(VERSION 3.20)

# Find LLVM
find_package(LLVM REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

# Set C++ standard to match LLVM
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add LLVM definitions and include directories
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
add_definitions(${LLVM_DEFINITIONS_LIST})
include_directories(${LLVM_INCLUDE_DIRS})

# Pass sources
set(PASS_SOURCES
    GpuDiffDbgPass.cpp
    SiteTable.cpp
)

# Build the pass as a shared library (plugin)
add_library(GpuDiffDbgPass MODULE ${PASS_SOURCES})

# Include common headers
target_include_directories(GpuDiffDbgPass PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
)

# Link against LLVM libraries
target_link_libraries(GpuDiffDbgPass PRIVATE
    LLVMCore
    LLVMSupport
    LLVMPasses
)

# Set output name
set_target_properties(GpuDiffDbgPass PROPERTIES
    PREFIX ""
    OUTPUT_NAME "libGpuDiffDbgPass"
)

# On Linux, we need -fPIC and avoid linking against unnecessary LLVM libs
if(UNIX AND NOT APPLE)
    target_compile_options(GpuDiffDbgPass PRIVATE -fPIC)
    target_link_options(GpuDiffDbgPass PRIVATE -Wl,-znodelete)
endif()

# Installation
install(TARGETS GpuDiffDbgPass
    LIBRARY DESTINATION lib
)
