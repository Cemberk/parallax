// Test kernel for Week 1A: Manual trace recording (no LLVM pass yet)
// This tests the runtime independently before integrating with the LLVM pass

#include "../../lib/runtime/gddbg_runtime.h"
#include <cuda_runtime.h>
#include <cstdio>

// Simple kernel with a data-dependent branch
__global__ void simple_branch(int* data, int* out, int threshold, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Manually record branch event (site_id is hardcoded for this test)
    uint32_t site_id = 0x12345678;  // In real pass, this will be hash of source location
    uint32_t condition = (data[idx] > threshold) ? 1 : 0;
    uint32_t operand_a = data[idx];

    __gddbg_record_branch(site_id, condition, operand_a);

    // Execute the branch
    if (condition) {
        out[idx] = data[idx] * 2;
    } else {
        out[idx] = data[idx] + 1;
    }
}

int main() {
    const int N = 256;
    const int threshold = 50;

    // Allocate and initialize host data
    int* h_data = new int[N];
    int* h_out = new int[N];

    for (int i = 0; i < N; i++) {
        h_data[i] = i;  // 0, 1, 2, ..., 255
    }

    // Allocate device memory
    int *d_data, *d_out;
    cudaMalloc(&d_data, N * sizeof(int));
    cudaMalloc(&d_out, N * sizeof(int));

    // Copy data to device
    cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice);

    // Launch configuration
    dim3 blockDim(32);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x);

    printf("Launching kernel with grid=%d, block=%d, threshold=%d\n",
           gridDim.x, blockDim.x, threshold);

    // Pre-launch: set up trace buffer
    gddbg_pre_launch("simple_branch", gridDim, blockDim);

    // Launch kernel
    simple_branch<<<gridDim, blockDim>>>(d_data, d_out, threshold, N);

    // Wait for completion
    cudaDeviceSynchronize();

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Post-launch: copy trace buffer and write to file
    gddbg_post_launch();

    // Copy results back
    cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Verify results
    int errors = 0;
    for (int i = 0; i < N; i++) {
        int expected = (h_data[i] > threshold) ? (h_data[i] * 2) : (h_data[i] + 1);
        if (h_out[i] != expected) {
            errors++;
            if (errors <= 5) {
                printf("ERROR at %d: got %d, expected %d\n", i, h_out[i], expected);
            }
        }
    }

    if (errors == 0) {
        printf("SUCCESS: All %d outputs correct\n", N);
    } else {
        printf("FAILED: %d errors\n", errors);
    }

    // Cleanup
    cudaFree(d_data);
    cudaFree(d_out);
    delete[] h_data;
    delete[] h_out;

    return errors > 0 ? 1 : 0;
}
