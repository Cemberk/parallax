// Loop divergence: variable iteration counts between runs.
// Tests bounded lookahead re-sync after extra iterations.

#include "../../lib/runtime/prlx_runtime.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>

__global__ void loop_kernel(int* data, int* out, int iterations, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int value = data[idx];

    for (int i = 0; i < iterations; i++) {
        if (value > 50) {
            value = value * 2;
        } else {
            value = value + 1;
        }

        if (i % 2 == 0) {
            value += 10;
        }
    }

    out[idx] = value;

    if (out[idx] > 100) {
        out[idx] = 100;
    }
}

int main(int argc, char** argv) {
    const int N = 128;
    int iterations = 10;  // Default: 10 iterations

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

    prlx_init();

    int* h_data = new int[N];
    int* h_out = new int[N];
    for (int i = 0; i < N; i++) {
        h_data[i] = i;
    }

    int *d_data, *d_out;
    cudaMalloc(&d_data, N * sizeof(int));
    cudaMalloc(&d_out, N * sizeof(int));
    cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice);

    dim3 blockDim(32);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x);

    prlx_pre_launch("loop_kernel", gridDim, blockDim);
    loop_kernel<<<gridDim, blockDim>>>(d_data, d_out, iterations, N);
    cudaDeviceSynchronize();
    prlx_post_launch();

    cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Sample results (first 5 threads):\n");
    for (int i = 0; i < 5; i++) {
        printf("  Thread %d: in=%d, out=%d\n", i, h_data[i], h_out[i]);
    }

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

    return 0;
}
