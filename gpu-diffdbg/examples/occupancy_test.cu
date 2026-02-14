// Scenario 2: The "Occupancy" Test
//
// This tests the tool's behavior under heavy load:
//   1. Memory pressure: Can it handle large grid sizes without OOM?
//   2. Buffer overflow: Do the circular buffers work correctly?
//   3. Performance: Does instrumentation cause TDR (timeout)?
//
// Expected behavior:
//   - Should complete without crashing
//   - Overflow counters should be non-zero for warps with >4096 events
//   - Execution time should be reasonable (<30s even with instrumentation)
//
// This validates production readiness on large-scale GPU workloads.

#include "../../lib/runtime/prlx_runtime.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <ctime>

// Kernel with significant branching and multiple loop iterations
// This generates many trace events per warp
__global__ void stress_kernel(int* data, int* out, int iterations, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int value = data[idx];

    // Nested loops to generate many events
    for (int i = 0; i < iterations; i++) {
        // Branch 1: Value comparison
        if (value > idx) {
            value = value * 2;

            // Nested branch
            if (value % 3 == 0) {
                value += 10;
            }
        } else {
            value = value + idx;

            // Nested branch
            if (value % 2 == 0) {
                value -= 5;
            }
        }

        // Branch 2: Loop iteration parity
        if (i % 2 == 0) {
            value ^= 0x55555555;  // XOR pattern
        } else {
            value ^= 0xAAAAAAAA;  // Alternate XOR pattern
        }

        // Branch 3: Thread ID based
        if (threadIdx.x < 16) {
            value += 100;
        } else {
            value -= 50;
        }

        // Branch 4: Block ID based
        if (blockIdx.x % 2 == 0) {
            value *= 3;
        }
    }

    out[idx] = value;

    // Final branch
    if (out[idx] < 0) {
        out[idx] = 0;  // Clamp negative values
    }
}

void print_memory_info() {
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    printf("GPU Memory: %.2f MB free / %.2f MB total\n",
           free_mem / 1024.0 / 1024.0,
           total_mem / 1024.0 / 1024.0);
}

int main(int argc, char** argv) {
    // Configurable parameters
    int num_blocks = 256;      // Default: moderate load
    int threads_per_block = 256;
    int iterations = 50;       // Iterations per thread

    // Parse command line arguments for stress levels
    if (argc > 1) {
        if (strcmp(argv[1], "light") == 0) {
            num_blocks = 64;
            threads_per_block = 128;
            iterations = 20;
        } else if (strcmp(argv[1], "medium") == 0) {
            num_blocks = 256;
            threads_per_block = 256;
            iterations = 50;
        } else if (strcmp(argv[1], "heavy") == 0) {
            num_blocks = 512;
            threads_per_block = 512;
            iterations = 100;
        } else if (strcmp(argv[1], "extreme") == 0) {
            num_blocks = 1024;
            threads_per_block = 1024;
            iterations = 200;
        }
    }

    const int N = num_blocks * threads_per_block;
    const int warps_per_block = (threads_per_block + 31) / 32;
    const int total_warps = num_blocks * warps_per_block;
    const int events_per_warp_estimated = iterations * 5;  // ~5 branches per iteration

    printf("=== Occupancy / Stress Test ===\n");
    printf("Grid:              %d blocks\n", num_blocks);
    printf("Threads per block: %d\n", threads_per_block);
    printf("Total threads:     %d\n", N);
    printf("Total warps:       %d\n", total_warps);
    printf("Iterations:        %d\n", iterations);
    printf("Estimated events per warp: %d\n", events_per_warp_estimated);

    if (events_per_warp_estimated > 4096) {
        printf("⚠️  WARNING: Expected events/warp (%d) exceeds buffer size (4096)\n",
               events_per_warp_estimated);
        printf("    Overflow counter will be tested.\n");
    }
    printf("\n");

    print_memory_info();
    printf("\n");

    // Initialize runtime
    prlx_init();

    // Allocate host data
    int* h_data = new int[N];
    int* h_out = new int[N];

    // Initialize data
    for (int i = 0; i < N; i++) {
        h_data[i] = i % 1000;
    }

    // Allocate device memory
    int *d_data, *d_out;
    cudaMalloc(&d_data, N * sizeof(int));
    cudaMalloc(&d_out, N * sizeof(int));
    cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice);

    dim3 blockDim(threads_per_block);
    dim3 gridDim(num_blocks);

    // Pre-launch
    printf("Launching kernel...\n");
    prlx_pre_launch("stress_kernel", gridDim, blockDim);

    // Time the kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    stress_kernel<<<gridDim, blockDim>>>(d_data, d_out, iterations, N);
    cudaEventRecord(stop);

    cudaError_t err = cudaDeviceSynchronize();
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    if (err != cudaSuccess) {
        printf("❌ KERNEL FAILED: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("✓ Kernel completed successfully\n");
    printf("Execution time: %.2f ms\n\n", milliseconds);

    // Post-launch
    prlx_post_launch();

    // Copy results
    cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Show sample results
    printf("Sample results (first 10 threads):\n");
    for (int i = 0; i < 10; i++) {
        printf("  Thread %4d: in=%d, out=%d\n", i, h_data[i], h_out[i]);
    }

    // Cleanup
    cudaFree(d_data);
    cudaFree(d_out);
    delete[] h_data;
    delete[] h_out;

    prlx_shutdown();

    printf("\n");
    print_memory_info();
    printf("\n");

    printf("✓ Stress test complete\n\n");

    printf("Validation checklist:\n");
    printf("  [%s] Kernel executed without crash\n", err == cudaSuccess ? "✓" : "✗");
    printf("  [%s] Execution time reasonable (< 30s)\n", milliseconds < 30000 ? "✓" : "✗");
    printf("  [ ] Check trace file size (should be reasonable)\n");
    printf("  [ ] Check for overflow counters in trace (if events > 4096/warp)\n");
    printf("  [ ] Verify differ can load and process the large trace\n");

    printf("\nTo test:\n");
    printf("  1. Check trace file: ls -lh *.prlx\n");
    printf("  2. Run twice with different stress levels:\n");
    printf("       PRLX_TRACE=trace_light.prlx ./occupancy_test light\n");
    printf("       PRLX_TRACE=trace_medium.prlx ./occupancy_test medium\n");
    printf("  3. Compare: prlx diff trace_light.prlx trace_medium.prlx\n");

    return 0;
}
