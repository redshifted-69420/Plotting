#include <metal_stdlib>
using namespace metal;

kernel void sub_arrays(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    uint base_idx = id * 4;
    if (base_idx >= size) return;
    if ((base_idx % 4 == 0) && (base_idx + 3 < size)) {
        float4 a4 = *reinterpret_cast<const device float4*>(&A[base_idx]);
        float4 b4 = *reinterpret_cast<const device float4*>(&B[base_idx]);
        *reinterpret_cast<device float4*>(&C[base_idx]) = a4 - b4;
    } else {
        for (uint i = 0; i < 4; i++) {
            uint idx = base_idx + i;
            if (idx < size) {
                C[idx] = A[idx] - B[idx];
            }
        }
    }
}