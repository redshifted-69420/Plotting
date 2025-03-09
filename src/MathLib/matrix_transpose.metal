#include <metal_stdlib>
using namespace metal;

// Kernel function for matrix transpose
kernel void matrix_transpose(
    device const float* input [[buffer(0)]],  // Input matrix
    device float* output [[buffer(1)]],       // Output matrix (transposed)
    constant uint& rows [[buffer(2)]],        // Number of rows in the input matrix
    constant uint& cols [[buffer(3)]],        // Number of columns in the input matrix
    uint2 gid [[thread_position_in_grid]]     // Thread ID in 2D grid
) {
    // Check if the thread is within the bounds of the output matrix
    if (gid.x < cols && gid.y < rows) {
        // Transpose the matrix: output[col][row] = input[row][col]
        output[gid.x * rows + gid.y] = input[gid.y * cols + gid.x];
    }
}   