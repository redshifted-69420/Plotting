#include <metal_stdlib>
using namespace metal;

kernel void add_arrays(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    // Calculate base index (each thread handles 4 elements)
    uint base_idx = id * 4;
    
    // Skip if we're completely out of bounds
    if (base_idx >= size) return;
    
    // Process 4 elements at once if we have a complete vector
    if (base_idx + 3 < size) {
        // Vector processing for aligned data
        float4 a4 = *reinterpret_cast<const device float4*>(&A[base_idx]);
        float4 b4 = *reinterpret_cast<const device float4*>(&B[base_idx]);
        *reinterpret_cast<device float4*>(&C[base_idx]) = a4 + b4;
    } else {
        // Handle remaining elements one by one
        for (uint i = 0; i < 4; i++) {
            uint idx = base_idx + i;
            if (idx < size) {
                C[idx] = A[idx] + B[idx];
            }
        }
    }
}