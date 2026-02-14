# Test executables

# Manual trace test (Week 1A - tests runtime without LLVM pass)
add_executable(manual_trace_test
    kernels/manual_trace_test.cu
)

# Link against runtime library (contains both host and device code)
target_link_libraries(manual_trace_test PRIVATE prlx_runtime)

# Include runtime headers
target_include_directories(manual_trace_test PRIVATE
    ${CMAKE_SOURCE_DIR}/lib/runtime
    ${CMAKE_SOURCE_DIR}/lib/common
)

# Set CUDA architecture and enable device linking
set_target_properties(manual_trace_test PROPERTIES
    CUDA_ARCHITECTURES "${PRLX_CUDA_ARCHITECTURES}"
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Add as a CTest
add_test(NAME manual_trace_test COMMAND manual_trace_test)
set_tests_properties(manual_trace_test PROPERTIES
    ENVIRONMENT "PRLX_TRACE=${CMAKE_CURRENT_BINARY_DIR}/manual_trace.prlx"
)

# Divergence test (generates two traces for differ testing)
add_executable(divergence_test
    kernels/divergence_test.cu
)

target_link_libraries(divergence_test PRIVATE prlx_runtime)

target_include_directories(divergence_test PRIVATE
    ${CMAKE_SOURCE_DIR}/lib/runtime
    ${CMAKE_SOURCE_DIR}/lib/common
)

set_target_properties(divergence_test PROPERTIES
    CUDA_ARCHITECTURES "${PRLX_CUDA_ARCHITECTURES}"
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
