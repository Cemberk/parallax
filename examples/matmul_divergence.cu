// Tiled matmul with a shared memory indexing bug.
// Demonstrates catching data corruption via branch divergence.

#include "../../lib/runtime/prlx_runtime.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

#define TILE_SIZE 16

// Tiled matrix multiplication kernel
// Matrix C = A * B
// A: M x K, B: K x N, C: M x N
__global__ void matmul_kernel(
    const float* A, const float* B, float* C,
    int M, int N, int K, bool inject_bug)
{
    // Shared memory tiles
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    // Loop over tiles
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tile into shared memory
        int a_col = t * TILE_SIZE + threadIdx.x;
        int b_row = t * TILE_SIZE + threadIdx.y;

        if (row < M && a_col < K) {
            // BUG INJECTION: Wrong index when inject_bug is true
            int bug_offset = (inject_bug && threadIdx.x == 7) ? 1 : 0;
            As[threadIdx.y][threadIdx.x + bug_offset] = A[row * K + a_col];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (col < N && b_row < K) {
            Bs[threadIdx.y][threadIdx.x] = B[b_row * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Compute partial product
        for (int k = 0; k < TILE_SIZE; k++) {
            float a_val = As[threadIdx.y][k];
            float b_val = Bs[k][threadIdx.x];

            // Branch based on loaded values
            // This branch will diverge if shared memory data is corrupted
            if (a_val > 0.5f && b_val > 0.5f) {
                sum += a_val * b_val;
            } else if (a_val > 0.5f) {
                sum += a_val * 0.1f;
            } else if (b_val > 0.5f) {
                sum += b_val * 0.1f;
            }

            // Additional branch for more divergence opportunity
            if (fabs(sum) > 10.0f) {
                sum *= 0.9f;  // Damping
            }
        }

        __syncthreads();
    }

    // Write result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

void init_matrix(float* mat, int rows, int cols, float scale) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = ((float)rand() / RAND_MAX) * scale;
    }
}

int main(int argc, char** argv) {
    bool inject_bug = false;

    // Check if we should inject the bug
    if (argc > 1 && strcmp(argv[1], "buggy") == 0) {
        inject_bug = true;
    }

    const int M = 64;  // Rows of A and C
    const int N = 64;  // Cols of B and C
    const int K = 64;  // Cols of A, Rows of B

    printf("=== Shared Memory Hazard Test ===\n");
    printf("Matrix dimensions: %dx%d * %dx%d = %dx%d\n", M, K, K, N, M, N);
    printf("Mode: %s\n", inject_bug ? "BUGGY (shared memory bug)" : "CORRECT");
    printf("\nThis kernel performs tiled matrix multiplication.\n");
    printf("In BUGGY mode, thread 7 writes to wrong shared memory index.\n\n");

    printf("To test:\n");
    printf("  Run A: PRLX_TRACE=trace_a.prlx ./matmul_divergence correct\n");
    printf("  Run B: PRLX_TRACE=trace_b.prlx ./matmul_divergence buggy\n");
    printf("  Diff:  prlx diff trace_a.prlx trace_b.prlx\n\n");

    prlx_init();

    float *h_A = new float[M * K];
    float *h_B = new float[K * N];
    float *h_C = new float[M * N];

    srand(42);
    init_matrix(h_A, M, K, 2.0f);
    init_matrix(h_B, K, N, 2.0f);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    // Copy to device
    cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch configuration
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    printf("Grid: %dx%d blocks of %dx%d threads\n", gridDim.x, gridDim.y, blockDim.x, blockDim.y);
    printf("Total warps: %d\n\n", gridDim.x * gridDim.y * ((blockDim.x * blockDim.y + 31) / 32));

    // Pre-launch
    prlx_pre_launch("matmul_kernel", gridDim, blockDim);

    // Launch kernel
    matmul_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K, inject_bug);
    cudaDeviceSynchronize();

    // Post-launch
    prlx_post_launch();

    // Copy results
    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Show sample results
    printf("Sample results (C[0:5, 0:5]):\n");
    for (int i = 0; i < 5; i++) {
        printf("  [");
        for (int j = 0; j < 5; j++) {
            printf(" %6.2f", h_C[i * N + j]);
        }
        printf(" ]\n");
    }

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    prlx_shutdown();

    printf("\n");
    if (inject_bug) {
        printf("✓ BUGGY run complete\n");
        printf("Expected: Branch divergence due to corrupted shared memory reads\n");
    } else {
        printf("✓ CORRECT run complete\n");
        printf("Expected: Consistent branch behavior\n");
    }

    printf("\nExpected diff behavior:\n");
    printf("  - Differ should detect branch divergence in computation loop\n");
    printf("  - This proves control flow tracing can catch data bugs\n");
    printf("  - Even without explicit shared memory value tracing\n");

    return 0;
}
