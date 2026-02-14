// Fully instrumented divergence example
//
// This example manually calls __prlx_record_branch() and __prlx_record_value()
// to simulate what the LLVM pass does automatically. This lets us test the full
// trace pipeline without needing the pass at compile time.
//
// Usage:
//   PRLX_TRACE=trace_a.prlx PRLX_HISTORY_DEPTH=64 ./instrumented_divergence 0
//   PRLX_TRACE=trace_b.prlx PRLX_HISTORY_DEPTH=64 ./instrumented_divergence 50
//   prlx-diff trace_a.prlx trace_b.prlx --history

#include "../../lib/runtime/prlx_runtime.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

__global__ void divergent_kernel(int* data, int* out, int threshold, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int value = data[idx];

    // Record the value feeding into the branch (time-travel history)
    __prlx_record_value(0xAAAA0001, (uint32_t)value);
    __prlx_record_value(0xAAAA0002, (uint32_t)threshold);

    // Main branch: different thresholds -> different paths taken
    if (value > threshold) {
        // Record: branch taken
        __prlx_record_branch(0xBBBB0001, 1, (uint32_t)value);
        value = value * 2;

        // Nested branch
        __prlx_record_value(0xAAAA0003, (uint32_t)value);
        if (value > 200) {
            __prlx_record_branch(0xBBBB0002, 1, (uint32_t)value);
            value = 200;  // Clamp
        } else {
            __prlx_record_branch(0xBBBB0002, 0, (uint32_t)value);
            value = value + 10;
        }
    } else {
        // Record: branch not taken
        __prlx_record_branch(0xBBBB0001, 0, (uint32_t)value);
        value = -value;

        // Different nested branch on this path
        __prlx_record_value(0xAAAA0004, (uint32_t)value);
        if (value < -100) {
            __prlx_record_branch(0xBBBB0003, 1, (uint32_t)value);
            value = -100;  // Clamp
        } else {
            __prlx_record_branch(0xBBBB0003, 0, (uint32_t)value);
            value = value - 5;
        }
    }

    // Final convergence point
    __prlx_record_value(0xAAAA0005, (uint32_t)value);
    out[idx] = value;
}

int main(int argc, char** argv) {
    const int N = 128;
    int threshold = 50;

    if (argc > 1) {
        threshold = atoi(argv[1]);
    }

    printf("=== Instrumented Divergence Test ===\n");
    printf("Threshold: %d\n", threshold);
    printf("Each warp records branches + value history\n\n");

    prlx_init();

    // Host data: values 0..127
    int* h_data = new int[N];
    int* h_out  = new int[N];
    for (int i = 0; i < N; i++) h_data[i] = i;

    int *d_data, *d_out;
    cudaMalloc(&d_data, N * sizeof(int));
    cudaMalloc(&d_out,  N * sizeof(int));
    cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice);

    dim3 block(32);
    dim3 grid((N + 31) / 32);

    prlx_pre_launch("divergent_kernel", grid, block);
    divergent_kernel<<<grid, block>>>(d_data, d_out, threshold, N);
    cudaDeviceSynchronize();
    prlx_post_launch();

    cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Sample results:\n");
    for (int i = 0; i < 8; i++)
        printf("  [%3d] in=%3d  out=%4d  %s\n",
               i, h_data[i], h_out[i],
               h_data[i] > threshold ? "TAKEN" : "NOT-TAKEN");
    printf("  ...\n");
    for (int i = N-4; i < N; i++)
        printf("  [%3d] in=%3d  out=%4d  %s\n",
               i, h_data[i], h_out[i],
               h_data[i] > threshold ? "TAKEN" : "NOT-TAKEN");

    cudaFree(d_data);
    cudaFree(d_out);
    delete[] h_data;
    delete[] h_out;
    prlx_shutdown();

    return 0;
}
