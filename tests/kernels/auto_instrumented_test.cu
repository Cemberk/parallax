// Test case for automatic instrumentation via LLVM pass
// This kernel will be automatically instrumented without manual __prlx_record_* calls

#include "../../lib/runtime/prlx_runtime.h"
#include <cuda_runtime.h>
#include <cstdio>

// Simple kernel with branches (will be auto-instrumented by LLVM pass)
__global__ void auto_test_kernel(int* data, int* out, int threshold, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Branch 1: threshold comparison (will be auto-instrumented)
    if (data[idx] > threshold) {
        out[idx] = data[idx] * 2;
    } else {
        out[idx] = data[idx] + 1;
    }

    // Branch 2: parity check (will be auto-instrumented)
    if (out[idx] % 2 == 0) {
        out[idx] += 10;
    }
}

int main() {
    const int N = 128;
    const int threshold = 50;

    printf("=== Automatic Instrumentation Test ===\n");
    printf("This kernel should be automatically instrumented by the LLVM pass\n\n");

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
    prlx_pre_launch("auto_test_kernel", gridDim, blockDim);

    // Launch kernel (should be automatically instrumented)
    auto_test_kernel<<<gridDim, blockDim>>>(d_data, d_out, threshold, N);
    cudaDeviceSynchronize();

    // Post-launch
    prlx_post_launch();

    // Copy results
    cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Verify some results
    printf("Sample results:\n");
    for (int i = 0; i < 10; i++) {
        printf("  data[%d] = %d, out[%d] = %d\n", i, h_data[i], i, h_out[i]);
    }

    // Cleanup
    cudaFree(d_data);
    cudaFree(d_out);
    delete[] h_data;
    delete[] h_out;

    prlx_shutdown();

    printf("\nSUCCESS: Check trace file and prlx-sites.json for instrumentation\n");
    printf("Expected: Multiple branch events auto-recorded at each conditional\n");

    return 0;
}
