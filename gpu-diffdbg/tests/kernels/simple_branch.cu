// Simple branch kernel for testing GPU DiffDbg
// This kernel can be compiled with the LLVM pass to test end-to-end instrumentation

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// Simple kernel with a data-dependent branch
__global__ void simple_branch(int* data, int* out, int threshold, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Data-dependent branch - this will be instrumented by the LLVM pass
    if (data[idx] > threshold) {
        out[idx] = data[idx] * 2;
    } else {
        out[idx] = data[idx] + 1;
    }
}

int main(int argc, char** argv) {
    const int N = 256;
    int threshold = 50;  // Default threshold

    // Parse command-line arguments
    for (int i = 1; i < argc; i++) {
        if (strncmp(argv[i], "--threshold=", 12) == 0) {
            threshold = atoi(argv[i] + 12);
        }
    }

    printf("Running simple_branch kernel with N=%d, threshold=%d\n", N, threshold);

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

    printf("Launching kernel with grid=%d, block=%d\n", gridDim.x, blockDim.x);

    // NOTE: When compiled with gddbg runtime, the pre/post launch hooks
    // are automatically called via constructor/destructor injection

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
