// Scenario 1: The "Loop Desync" Test
//
// This tests the bounded lookahead algorithm with variable loop counts.
// The challenge: Different runs execute different numbers of loop iterations.
//
// Expected behavior:
//   - The differ should detect "ExtraEvents" for the extra iteration(s)
//   - It should NOT flag every instruction after loop exit as a mismatch
//   - It should re-sync after the loop and continue comparing
//
// This is the CRITICAL test for the bounded lookahead algorithm.

#include "../../lib/runtime/prlx_runtime.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>

// Kernel with data-dependent loop count
// The loop count is determined by the 'iterations' parameter
__global__ void loop_kernel(int* data, int* out, int iterations, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int value = data[idx];

    // Data-dependent loop - different runs may have different iteration counts
    for (int i = 0; i < iterations; i++) {
        // Branch inside loop - will be recorded each iteration
        if (value > 50) {
            value = value * 2;
        } else {
            value = value + 1;
        }

        // Another branch for complexity
        if (i % 2 == 0) {
            value += 10;
        }
    }

    out[idx] = value;

    // Post-loop branch - should re-sync here after loop desync
    if (out[idx] > 100) {
        out[idx] = 100;  // Clamp to 100
    }
}

int main(int argc, char** argv) {
    const int N = 128;
    int iterations = 10;  // Default: 10 iterations

    // Allow overriding iteration count via command line
    if (argc > 1) {
        iterations = atoi(argv[1]);
    }

    printf("=== Loop Divergence Test ===\n");
    printf("Iterations: %d\n", iterations);
    printf("This kernel has a data-dependent loop.\n");
    printf("Run with different iteration counts to test bounded lookahead:\n");
    printf("  Run A: PRLX_TRACE=trace_a.prlx ./loop_divergence 10\n");
    printf("  Run B: PRLX_TRACE=trace_b.prlx ./loop_divergence 11\n");
    printf("  Diff:  prlx diff trace_a.prlx trace_b.prlx\n\n");

    // Initialize runtime
    prlx_init();

    // Allocate host data
    int* h_data = new int[N];
    int* h_out = new int[N];

    // Initialize data: 0, 1, 2, ..., 127
    for (int i = 0; i < N; i++) {
        h_data[i] = i;
    }

    // Allocate device memory
    int *d_data, *d_out;
    cudaMalloc(&d_data, N * sizeof(int));
    cudaMalloc(&d_out, N * sizeof(int));
    cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice);

    dim3 blockDim(32);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x);

    // Pre-launch
    prlx_pre_launch("loop_kernel", gridDim, blockDim);

    // Launch kernel
    loop_kernel<<<gridDim, blockDim>>>(d_data, d_out, iterations, N);
    cudaDeviceSynchronize();

    // Post-launch
    prlx_post_launch();

    // Copy results
    cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Verify some results
    printf("Sample results (first 5 threads):\n");
    for (int i = 0; i < 5; i++) {
        printf("  Thread %d: in=%d, out=%d\n", i, h_data[i], h_out[i]);
    }

    // Cleanup
    cudaFree(d_data);
    cudaFree(d_out);
    delete[] h_data;
    delete[] h_out;

    prlx_shutdown();

    printf("\n");
    if (iterations == 10) {
        printf("✓ Run A complete (10 iterations)\n");
        printf("Next: Run with 11 iterations for Run B\n");
    } else if (iterations == 11) {
        printf("✓ Run B complete (11 iterations)\n");
        printf("Next: Compare traces with: prlx diff trace_a.prlx trace_b.prlx\n");
    } else {
        printf("✓ Run complete (%d iterations)\n", iterations);
    }

    printf("\nExpected diff behavior:\n");
    printf("  - Differ should detect 'ExtraEvents' for the 11th iteration\n");
    printf("  - Should re-sync at post-loop branch\n");
    printf("  - Should NOT flag every instruction after loop as divergent\n");

    return 0;
}
