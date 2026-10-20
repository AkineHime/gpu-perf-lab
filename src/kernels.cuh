// SGEMM kernels, from naive to vectorized 2D register tiling.
//
// All matrices are row-major:  C (MxN) = alpha * A (MxK) @ B (KxN) + beta * C
//
// The progression follows Simon Boehm's "How to Optimize a CUDA Matmul Kernel for
// cuBLAS-like Performance" (siboehm.com/articles/22/CUDA-MMM). Each step removes the
// bottleneck that the previous one exposed:
//
//   v0 naive         one thread per output; warps read A with a K-float stride
//   v1 coalesced     remap threads so a warp reads/writes 32 consecutive floats
//   v2 smem tiling   stage 32x32 tiles of A and B in shared memory (32x data reuse)
//   v3 1D blocktile  each thread computes 8 outputs in a column: fewer smem loads per FMA
//   v4 2D blocktile  each thread computes an 8x8 outer product held in registers
//   v5 vectorized    float4 global loads, A tile transposed in smem for wide reads
//
// v3-v5 assume M, N divisible by their block tile (64 or 128) and K by 8; the
// benchmark only runs sizes that are multiples of 128.
#pragma once
#include <cuda_runtime.h>

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

// ---------------------------------------------------------------- v0 naive
// threadIdx.x walks rows: the 32 threads of a warp touch 32 different rows of A and C,
// so every A load and C store is a separate memory transaction.
__global__ void sgemm_naive(int M, int N, int K, float alpha, const float* A,
                            const float* B, float beta, float* C) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < N) {
        float acc = 0.f;
        for (int i = 0; i < K; ++i) acc += A[row * K + i] * B[i * N + col];
        C[row * N + col] = alpha * acc + beta * C[row * N + col];
    }
}

// ---------------------------------------------------------------- v1 coalesced
// Same math; consecutive threads now own consecutive columns. A warp's B loads and C
// stores become one 128-byte transaction, and its A load is a single broadcast.
template <int BS>
__global__ void sgemm_coalesced(int M, int N, int K, float alpha, const float* A,
                                const float* B, float beta, float* C) {
    const int row = blockIdx.y * BS + threadIdx.x / BS;
    const int col = blockIdx.x * BS + threadIdx.x % BS;
    if (row < M && col < N) {
        float acc = 0.f;
        for (int i = 0; i < K; ++i) acc += A[row * K + i] * B[i * N + col];
        C[row * N + col] = alpha * acc + beta * C[row * N + col];
    }
}

// ---------------------------------------------------------------- v2 shared memory
// Each block loads a BSxBS tile of A and of B into shared memory once, then every
// thread reuses it BS times. Global traffic drops by a factor of BS.
template <int BS>
__global__ void sgemm_smem(int M, int N, int K, float alpha, const float* A,
                           const float* B, float beta, float* C) {
    __shared__ float As[BS * BS];
    __shared__ float Bs[BS * BS];
    const int tRow = threadIdx.x / BS, tCol = threadIdx.x % BS;
    const int row = blockIdx.y * BS + tRow, col = blockIdx.x * BS + tCol;

    float acc = 0.f;
    for (int k0 = 0; k0 < K; k0 += BS) {
        As[tRow * BS + tCol] = (row < M && k0 + tCol < K) ? A[row * K + k0 + tCol] : 0.f;
        Bs[tRow * BS + tCol] = (k0 + tRow < K && col < N) ? B[(k0 + tRow) * N + col] : 0.f;
        __syncthreads();
        #pragma unroll
        for (int i = 0; i < BS; ++i) acc += As[tRow * BS + i] * Bs[i * BS + tCol];
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = alpha * acc + beta * C[row * N + col];
}

// ---------------------------------------------------------------- v3 1D blocktiling
// v2 does 2 shared-memory loads per FMA. Here a thread computes TM outputs stacked in
// one column: one Bs value is loaded once and reused TM times from a register.
template <int BM, int BN, int BK, int TM>
__global__ void __launch_bounds__(BM * BN / TM)
sgemm_blocktile_1d(int M, int N, int K, float alpha, const float* A, const float* B,
                   float beta, float* C) {
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];
    static_assert(BM * BK == BM * BN / TM && BK * BN == BM * BN / TM,
                  "one global load per thread per tile");

    const int tCol = threadIdx.x % BN, tRow = threadIdx.x / BN;
    const int aRow = threadIdx.x / BK, aCol = threadIdx.x % BK;
    const int bRow = threadIdx.x / BN, bCol = threadIdx.x % BN;

    A += blockIdx.y * BM * K;
    B += blockIdx.x * BN;
    C += blockIdx.y * BM * N + blockIdx.x * BN;

    float acc[TM] = {0.f};
    for (int k0 = 0; k0 < K; k0 += BK) {
        As[aRow * BK + aCol] = A[aRow * K + aCol];
        Bs[bRow * BN + bCol] = B[bRow * N + bCol];
        __syncthreads();
        A += BK;
        B += BK * N;
        #pragma unroll
        for (int d = 0; d < BK; ++d) {
            const float b = Bs[d * BN + tCol];
            #pragma unroll
            for (int m = 0; m < TM; ++m) acc[m] += As[(tRow * TM + m) * BK + d] * b;
        }
        __syncthreads();
    }
    #pragma unroll
    for (int m = 0; m < TM; ++m) {
        float* c = &C[(tRow * TM + m) * N + tCol];
        *c = alpha * acc[m] + beta * *c;
    }
}

// ---------------------------------------------------------------- v4 2D blocktiling
// Each thread owns a TMxTN patch. Per k-step it loads TM values of A and TN of B into
// registers and does TM*TN FMAs: 64 FMAs per 16 shared loads instead of 8 per 9.
template <int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__(BM * BN / (TM * TN))
sgemm_blocktile_2d(int M, int N, int K, float alpha, const float* A, const float* B,
                   float beta, float* C) {
    constexpr int THREADS = BM * BN / (TM * TN);
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    const int tCol = threadIdx.x % (BN / TN), tRow = threadIdx.x / (BN / TN);
    const int aRow = threadIdx.x / BK, aCol = threadIdx.x % BK;
    constexpr int A_STRIDE = THREADS / BK;
    const int bRow = threadIdx.x / BN, bCol = threadIdx.x % BN;
    constexpr int B_STRIDE = THREADS / BN;

    A += blockIdx.y * BM * K;
    B += blockIdx.x * BN;
    C += blockIdx.y * BM * N + blockIdx.x * BN;

    float acc[TM * TN] = {0.f};
    float regA[TM], regB[TN];
    for (int k0 = 0; k0 < K; k0 += BK) {
        #pragma unroll
        for (int off = 0; off < BM; off += A_STRIDE)
            As[(aRow + off) * BK + aCol] = A[(aRow + off) * K + aCol];
        #pragma unroll
        for (int off = 0; off < BK; off += B_STRIDE)
            Bs[(bRow + off) * BN + bCol] = B[(bRow + off) * N + bCol];
        __syncthreads();
        A += BK;
        B += BK * N;
        #pragma unroll
        for (int d = 0; d < BK; ++d) {
            #pragma unroll
            for (int m = 0; m < TM; ++m) regA[m] = As[(tRow * TM + m) * BK + d];
            #pragma unroll
            for (int n = 0; n < TN; ++n) regB[n] = Bs[d * BN + tCol * TN + n];
            #pragma unroll
            for (int m = 0; m < TM; ++m)
                #pragma unroll
                for (int n = 0; n < TN; ++n) acc[m * TN + n] += regA[m] * regB[n];
        }
        __syncthreads();
    }
    #pragma unroll
    for (int m = 0; m < TM; ++m)
        #pragma unroll
        for (int n = 0; n < TN; ++n) {
            float* c = &C[(tRow * TM + m) * N + tCol * TN + n];
            *c = alpha * acc[m * TN + n] + beta * *c;
        }
}

// ---------------------------------------------------------------- v5 vectorized
// Like v4, but global memory is moved as float4 (16-byte) loads/stores, and the A tile
// is stored transposed so the inner loop reads TM consecutive floats from As.
template <int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__(BM * BN / (TM * TN))
sgemm_vectorized(int M, int N, int K, float alpha, const float* A, const float* B,
                 float beta, float* C) {
    constexpr int THREADS = BM * BN / (TM * TN);
    static_assert(BM * BK / 4 == THREADS && BK * BN / 4 == THREADS,
                  "one float4 of A and one of B per thread per tile");
    __shared__ float As[BK * BM];  // transposed: As[k][m]
    __shared__ __align__(16) float Bs[BK * BN];  // written with float4 stores

    const int tCol = threadIdx.x % (BN / TN), tRow = threadIdx.x / (BN / TN);
    const int aRow = threadIdx.x / (BK / 4), aCol = threadIdx.x % (BK / 4);
    const int bRow = threadIdx.x / (BN / 4), bCol = threadIdx.x % (BN / 4);

    A += blockIdx.y * BM * K;
    B += blockIdx.x * BN;
    C += blockIdx.y * BM * N + blockIdx.x * BN;

    float acc[TM * TN] = {0.f};
    float regA[TM], regB[TN];
    for (int k0 = 0; k0 < K; k0 += BK) {
        const float4 a = reinterpret_cast<const float4*>(&A[aRow * K + aCol * 4])[0];
        As[(aCol * 4 + 0) * BM + aRow] = a.x;
        As[(aCol * 4 + 1) * BM + aRow] = a.y;
        As[(aCol * 4 + 2) * BM + aRow] = a.z;
        As[(aCol * 4 + 3) * BM + aRow] = a.w;
        reinterpret_cast<float4*>(&Bs[bRow * BN + bCol * 4])[0] =
            reinterpret_cast<const float4*>(&B[bRow * N + bCol * 4])[0];
        __syncthreads();
        A += BK;
        B += BK * N;
        #pragma unroll
        for (int d = 0; d < BK; ++d) {
            #pragma unroll
            for (int m = 0; m < TM; ++m) regA[m] = As[d * BM + tRow * TM + m];
            #pragma unroll
            for (int n = 0; n < TN; ++n) regB[n] = Bs[d * BN + tCol * TN + n];
            #pragma unroll
            for (int m = 0; m < TM; ++m)
                #pragma unroll
                for (int n = 0; n < TN; ++n) acc[m * TN + n] += regA[m] * regB[n];
        }
        __syncthreads();
    }
    #pragma unroll
    for (int m = 0; m < TM; ++m)
        #pragma unroll
        for (int n = 0; n < TN; n += 4) {
            float4* p = reinterpret_cast<float4*>(&C[(tRow * TM + m) * N + tCol * TN + n]);
            float4 c = *p;
            c.x = alpha * acc[m * TN + n + 0] + beta * c.x;
            c.y = alpha * acc[m * TN + n + 1] + beta * c.y;
            c.z = alpha * acc[m * TN + n + 2] + beta * c.z;
            c.w = alpha * acc[m * TN + n + 3] + beta * c.w;
            *p = c;
        }
}

// ---------------------------------------------------------------- registry
using Launcher = void (*)(int, int, int, float, const float*, const float*, float, float*);

struct KernelInfo {
    const char* id;
    const char* name;
    const char* idea;
    // Global-load intensity: FLOPs per byte the kernel *requests* from global memory,
    // derived from its tiling (caches can make real DRAM traffic lower).
    double load_intensity;
    int size_multiple;  // M, N must be multiples of this
    Launcher launch;
};

inline void launch_naive(int M, int N, int K, float al, const float* A, const float* B,
                         float be, float* C) {
    dim3 block(32, 32), grid(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
    sgemm_naive<<<grid, block>>>(M, N, K, al, A, B, be, C);
}
inline void launch_coalesced(int M, int N, int K, float al, const float* A, const float* B,
                             float be, float* C) {
    dim3 grid(CEIL_DIV(N, 32), CEIL_DIV(M, 32));
    sgemm_coalesced<32><<<grid, 32 * 32>>>(M, N, K, al, A, B, be, C);
}
inline void launch_smem(int M, int N, int K, float al, const float* A, const float* B,
                        float be, float* C) {
    dim3 grid(CEIL_DIV(N, 32), CEIL_DIV(M, 32));
    sgemm_smem<32><<<grid, 32 * 32>>>(M, N, K, al, A, B, be, C);
}
inline void launch_1d(int M, int N, int K, float al, const float* A, const float* B,
                      float be, float* C) {
    dim3 grid(N / 64, M / 64);
    sgemm_blocktile_1d<64, 64, 8, 8><<<grid, 64 * 64 / 8>>>(M, N, K, al, A, B, be, C);
}
inline void launch_2d(int M, int N, int K, float al, const float* A, const float* B,
                      float be, float* C) {
    dim3 grid(N / 128, M / 128);
    sgemm_blocktile_2d<128, 128, 8, 8, 8><<<grid, 256>>>(M, N, K, al, A, B, be, C);
}
inline void launch_vec(int M, int N, int K, float al, const float* A, const float* B,
                       float be, float* C) {
    dim3 grid(N / 128, M / 128);
    sgemm_vectorized<128, 128, 8, 8, 8><<<grid, 256>>>(M, N, K, al, A, B, be, C);
}

// Tile of BMxBN outputs loads (BM + BN) * K floats: intensity = 2*BM*BN*K / (4*(BM+BN)*K)
constexpr double tile_intensity(int bm, int bn) { return 2.0 * bm * bn / (4.0 * (bm + bn)); }

inline const KernelInfo KERNELS[] = {
    {"v0_naive", "Naive", "one thread per output, uncoalesced", 0.25, 1, launch_naive},
    {"v1_coalesced", "Coalesced", "warp reads consecutive floats", 0.25, 1, launch_coalesced},
    {"v2_smem", "Shared-memory tiling", "32x32 tiles staged in smem", tile_intensity(32, 32), 1, launch_smem},
    {"v3_tile1d", "1D register tiling", "8 outputs per thread", tile_intensity(64, 64), 64, launch_1d},
    {"v4_tile2d", "2D register tiling", "8x8 outer product per thread", tile_intensity(128, 128), 128, launch_2d},
    {"v5_vectorized", "Vectorized float4", "128-bit loads, transposed A tile", tile_intensity(128, 128), 128, launch_vec},
};
