# Example executables for validation (Phase 5)

# Scenario 1: Loop Desync - Tests bounded lookahead algorithm
add_executable(loop_divergence
    loop_divergence.cu
)

target_link_libraries(loop_divergence PRIVATE gddbg_runtime)

target_include_directories(loop_divergence PRIVATE
    ${CMAKE_SOURCE_DIR}/lib/runtime
    ${CMAKE_SOURCE_DIR}/lib/common
)

set_target_properties(loop_divergence PROPERTIES
    CUDA_ARCHITECTURES "80"
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Scenario 2: Occupancy Test - Tests memory and performance limits
add_executable(occupancy_test
    occupancy_test.cu
)

target_link_libraries(occupancy_test PRIVATE gddbg_runtime)

target_include_directories(occupancy_test PRIVATE
    ${CMAKE_SOURCE_DIR}/lib/runtime
    ${CMAKE_SOURCE_DIR}/lib/common
)

set_target_properties(occupancy_test PROPERTIES
    CUDA_ARCHITECTURES "80"
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Scenario 3: Shared Memory Hazard - Tests bug detection
add_executable(matmul_divergence
    matmul_divergence.cu
)

target_link_libraries(matmul_divergence PRIVATE gddbg_runtime)

target_include_directories(matmul_divergence PRIVATE
    ${CMAKE_SOURCE_DIR}/lib/runtime
    ${CMAKE_SOURCE_DIR}/lib/common
)

set_target_properties(matmul_divergence PROPERTIES
    CUDA_ARCHITECTURES "80"
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Scenario 4: Instrumented Divergence - Full pipeline with manual instrumentation
add_executable(instrumented_divergence
    instrumented_divergence.cu
)

target_link_libraries(instrumented_divergence PRIVATE gddbg_runtime)

target_include_directories(instrumented_divergence PRIVATE
    ${CMAKE_SOURCE_DIR}/lib/runtime
    ${CMAKE_SOURCE_DIR}/lib/common
)

set_target_properties(instrumented_divergence PROPERTIES
    CUDA_ARCHITECTURES "80"
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Optional: Add as CTests
add_test(NAME example_loop_divergence COMMAND loop_divergence 10)
set_tests_properties(example_loop_divergence PROPERTIES
    ENVIRONMENT "GDDBG_TRACE=${CMAKE_CURRENT_BINARY_DIR}/loop_test.gddbg"
)

add_test(NAME example_occupancy_light COMMAND occupancy_test light)
set_tests_properties(example_occupancy_light PROPERTIES
    ENVIRONMENT "GDDBG_TRACE=${CMAKE_CURRENT_BINARY_DIR}/occupancy_test.gddbg"
)

add_test(NAME example_matmul_correct COMMAND matmul_divergence correct)
set_tests_properties(example_matmul_correct PROPERTIES
    ENVIRONMENT "GDDBG_TRACE=${CMAKE_CURRENT_BINARY_DIR}/matmul_test.gddbg"
)
