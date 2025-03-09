#include <metal_stdlib>
using namespace metal;

// Kernel function for matrix multiplication
kernel void matrix_mul(
    device const float* A [[buffer(0)]], // Input matrix A
    device const float* B [[buffer(1)]], // Input matrix B
    device float* C [[buffer(2)]],       // Output matrix C
    constant uint& M [[buffer(3)]],      // Number of rows in A and C
    constant uint& N [[buffer(4)]],      // Number of columns in B and C
    constant uint& K [[buffer(5)]],      // Number of columns in A and rows in B
    uint2 gid [[thread_position_in_grid]] // Thread ID in 2D grid
) {
    // Check if the thread is within the bounds of the output matrix
    if (gid.x < M && gid.y < N) {
        float sum = 0.0f;

        // Compute the dot product of the row from A and the column from B
        for (uint i = 0; i < K; ++i) {
            sum += A[gid.x * K + i] * B[i * N + gid.y];
        }

        // Store the result in the output matrix
        C[gid.x * N + gid.y] = sum;
    }
}