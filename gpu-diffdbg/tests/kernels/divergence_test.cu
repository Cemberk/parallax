// Test case to demonstrate divergence detection
// Generates two traces with a known divergence point

#include "../../lib/runtime/prlx_runtime.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>

// Kernel with configurable behavior
__global__ void parametrized_kernel(int* data, int* out, int threshold, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    uint32_t condition = (data[idx] > threshold) ? 1 : 0;
    uint32_t operand_a = data[idx];

    // Record branch event
    __prlx_record_branch(0xDEADBEEF, condition, operand_a);

    if (condition) {
        out[idx] = data[idx] * 2;
    } else {
        out[idx] = data[idx] + 1;
    }
}

void run_trace(const char* output_file, int threshold, int* h_data, int* h_out, int N) {
    // Set output path
    setenv("PRLX_TRACE", output_file, 1);
    prlx_init(); // Re-initialize to pick up new path

    // Allocate device memory
    int *d_data, *d_out;
    cudaMalloc(&d_data, N * sizeof(int));
    cudaMalloc(&d_out, N * sizeof(int));
    cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice);

    dim3 blockDim(32);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x);

    // Pre-launch
    prlx_pre_launch("parametrized_kernel", gridDim, blockDim);

    // Launch kernel
    parametrized_kernel<<<gridDim, blockDim>>>(d_data, d_out, threshold, N);
    cudaDeviceSynchronize();

    // Post-launch
    prlx_post_launch();

    // Copy results
    cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_data);
    cudaFree(d_out);
}

int main() {
    const int N = 128;

    // Allocate host data
    int* h_data = new int[N];
    int* h_out_a = new int[N];
    int* h_out_b = new int[N];

    // Initialize data: 0, 1, 2, ..., 127
    for (int i = 0; i < N; i++) {
        h_data[i] = i;
    }

    printf("Generating divergent traces...\n");
    printf("Data range: 0 to %d\n\n", N-1);

    // Run A: threshold=50
    printf("Trace A: threshold=50\n");
    run_trace("trace_a.prlx", 50, h_data, h_out_a, N);
    printf("  -> Diverges at value=51 (taken)\n\n");

    // Run B: threshold=60
    printf("Trace B: threshold=60\n");
    run_trace("trace_b.prlx", 60, h_data, h_out_b, N);
    printf("  -> Diverges at value=51 (NOT taken)\n\n");

    printf("SUCCESS: Generated traces trace_a.prlx and trace_b.prlx\n");
    printf("Expected divergence: Warp with idx=51 (warp 1)\n");
    printf("  Trace A: branch TAKEN (51 > 50)\n");
    printf("  Trace B: branch NOT-TAKEN (51 <= 60 is false, so 51 > 60 is the condition)\n");
    printf("\nRun: ./differ/target/release/prlx-diff trace_a.prlx trace_b.prlx\n");

    delete[] h_data;
    delete[] h_out_a;
    delete[] h_out_b;

    return 0;
}
