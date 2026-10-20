// Micro-benchmarks that measure this GPU's real ceilings for the roofline:
//   * DRAM bandwidth: a float4 device-to-device copy over buffers far larger than L2
//   * FP32 peak:      independent FMA chains, no memory traffic at all
#pragma once
#include <cuda_runtime.h>

__global__ void probe_copy(const float4* __restrict__ src, float4* __restrict__ dst, size_t n) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x)
        dst[i] = src[i];
}

// 8 independent accumulators per thread hide the 4-cycle FMA latency.
template <int ITERS>
__global__ void probe_fma(float* out, float seed) {
    float a0 = seed + threadIdx.x, a1 = a0 + 1, a2 = a0 + 2, a3 = a0 + 3;
    float a4 = a0 + 4, a5 = a0 + 5, a6 = a0 + 6, a7 = a0 + 7;
    const float m = 0.999f, c = 0.001f;
    #pragma unroll 16
    for (int i = 0; i < ITERS; ++i) {
        a0 = fmaf(a0, m, c); a1 = fmaf(a1, m, c); a2 = fmaf(a2, m, c); a3 = fmaf(a3, m, c);
        a4 = fmaf(a4, m, c); a5 = fmaf(a5, m, c); a6 = fmaf(a6, m, c); a7 = fmaf(a7, m, c);
    }
    // a data dependency on the result so the compiler can't delete the loop
    const float r = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
    if (r == 12345.678f) out[threadIdx.x] = r;
}
