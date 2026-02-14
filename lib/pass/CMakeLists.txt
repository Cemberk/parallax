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
    PrlxPass.cpp
    SiteTable.cpp
)

# Build the pass as a shared library (plugin)
add_library(PrlxPass MODULE ${PASS_SOURCES})

# Include common headers
target_include_directories(PrlxPass PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
)

# Link against LLVM: use shared libLLVM if available to avoid symbol conflicts
# when loaded into opt/clang (which also use libLLVM.so)
if(LLVM_LINK_LLVM_DYLIB)
    # LLVM was built with shared libLLVM - link against it
    target_link_libraries(PrlxPass PRIVATE LLVM)
else()
    # Fallback: don't link LLVM libs at all for MODULE plugins.
    # The pass plugin will resolve symbols from the host process (opt/clang).
    # This avoids duplicate LLVM symbol registration errors.
    # We only need the headers (already included via LLVM_INCLUDE_DIRS).
endif()

# Set output name
set_target_properties(PrlxPass PROPERTIES
    PREFIX ""
    OUTPUT_NAME "libPrlxPass"
)

# On Linux, we need -fPIC and undefined symbols are resolved at load time
if(UNIX AND NOT APPLE)
    target_compile_options(PrlxPass PRIVATE -fPIC)
    target_link_options(PrlxPass PRIVATE -Wl,-znodelete)
    # Allow undefined symbols - they come from the host process (opt/clang)
    set_target_properties(PrlxPass PROPERTIES
        LINK_FLAGS "-Wl,--allow-shlib-undefined"
    )
endif()

# Installation
install(TARGETS PrlxPass
    LIBRARY DESTINATION lib
)
