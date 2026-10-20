// gemm_bench: time every SGEMM kernel against cuBLAS and measure the GPU's ceilings.
//
//   gemm_bench [--sizes 256,512,1024,2048,4096] [--kernels v2_smem,v5_vectorized]
//              [--min-ms 300] [--no-probes] [--no-verify]
//
// Prints one JSON object per line on stdout (device, probe, result, done), so a
// dashboard can stream results as they arrive. Human-readable progress goes to stderr.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "kernels.cuh"
#include "probes.cuh"

#define CUDA_CHECK(x)                                                                \
    do {                                                                             \
        cudaError_t e_ = (x);                                                        \
        if (e_ != cudaSuccess) {                                                     \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_),      \
                    __FILE__, __LINE__);                                             \
            printf("{\"type\":\"error\",\"message\":\"%s\"}\n", cudaGetErrorString(e_)); \
            exit(1);                                                                 \
        }                                                                            \
    } while (0)

#define CUBLAS_CHECK(x)                                                     \
    do {                                                                    \
        cublasStatus_t s_ = (x);                                            \
        if (s_ != CUBLAS_STATUS_SUCCESS) {                                  \
            fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)s_, __FILE__, __LINE__); \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

struct Options {
    std::vector<int> sizes{256, 512, 1024, 2048, 4096};
    std::vector<std::string> kernels;  // empty = all
    float min_ms = 300.f;
    bool probes = true, verify = true;
};

static std::vector<std::string> split(const char* s) {
    std::vector<std::string> out;
    std::string cur;
    for (const char* p = s; ; ++p) {
        if (*p == ',' || *p == '\0') {
            if (!cur.empty()) out.push_back(cur);
            cur.clear();
            if (!*p) break;
        } else {
            cur += *p;
        }
    }
    return out;
}

static Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        auto next = [&] {
            if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", argv[i]); exit(2); }
            return argv[++i];
        };
        if (!strcmp(argv[i], "--sizes")) {
            o.sizes.clear();
            for (auto& s : split(next())) o.sizes.push_back(atoi(s.c_str()));
        } else if (!strcmp(argv[i], "--kernels")) {
            o.kernels = split(next());
        } else if (!strcmp(argv[i], "--min-ms")) {
            o.min_ms = (float)atof(next());
        } else if (!strcmp(argv[i], "--no-probes")) {
            o.probes = false;
        } else if (!strcmp(argv[i], "--no-verify")) {
            o.verify = false;
        } else {
            fprintf(stderr, "unknown argument %s\n", argv[i]);
            exit(2);
        }
    }
    return o;
}

// Median time of a launch, after warm-up; repeats until min_ms of samples (>= 3 runs).
template <class F>
static float time_ms(F&& launch, float min_ms, int max_reps = 100) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    launch();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> samples;
    float total = 0.f;
    while ((int)samples.size() < max_reps && (total < min_ms || samples.size() < 3)) {
        CUDA_CHECK(cudaEventRecord(start));
        launch();
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        samples.push_back(ms);
        total += ms;
    }
    CUDA_CHECK(cudaGetLastError());
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

static void print_device() {
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp p{};
    CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
    int cuda_rt = 0;
    cudaRuntimeGetVersion(&cuda_rt);
    printf("{\"type\":\"device\",\"name\":\"%s\",\"cc\":\"%d.%d\",\"sms\":%d,"
           "\"vram_mb\":%zu,\"l2_kb\":%d,\"smem_per_block_kb\":%zu,\"cuda_runtime\":%d}\n",
           p.name, p.major, p.minor, p.multiProcessorCount, p.totalGlobalMem >> 20,
           p.l2CacheSize >> 10, p.sharedMemPerBlock >> 10, cuda_rt);
    fflush(stdout);
    fprintf(stderr, "%s (sm_%d%d, %d SMs)\n", p.name, p.major, p.minor, p.multiProcessorCount);
}

static void run_probes(float min_ms) {
    // Bandwidth: 2 x 256 MB, far beyond L2.
    const size_t n4 = (256u << 20) / sizeof(float4);
    float4 *src, *dst;
    CUDA_CHECK(cudaMalloc(&src, n4 * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&dst, n4 * sizeof(float4)));
    CUDA_CHECK(cudaMemset(src, 0, n4 * sizeof(float4)));
    int sms = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    const float copy_ms = time_ms([&] { probe_copy<<<sms * 8, 256>>>(src, dst, n4); }, min_ms);
    const double gbps = 2.0 * n4 * sizeof(float4) / (copy_ms * 1e-3) / 1e9;
    CUDA_CHECK(cudaFree(src));
    CUDA_CHECK(cudaFree(dst));
    printf("{\"type\":\"probe\",\"name\":\"dram_bandwidth\",\"gbps\":%.1f}\n", gbps);
    fflush(stdout);
    fprintf(stderr, "  DRAM bandwidth  %8.1f GB/s\n", gbps);

    // FP32: 8 FMAs (16 FLOPs) per iteration per thread.
    constexpr int ITERS = 4096;
    float* out;
    CUDA_CHECK(cudaMalloc(&out, 1024 * sizeof(float)));
    const int blocks = sms * 32, threads = 256;
    const float fma_ms = time_ms([&] { probe_fma<ITERS><<<blocks, threads>>>(out, 1.f); }, min_ms);
    const double gflops = 16.0 * ITERS * blocks * threads / (fma_ms * 1e-3) / 1e9;
    CUDA_CHECK(cudaFree(out));
    printf("{\"type\":\"probe\",\"name\":\"fp32_peak\",\"gflops\":%.1f}\n", gflops);
    fflush(stdout);
    fprintf(stderr, "  FP32 FMA peak   %8.1f GFLOPS\n", gflops);
}

int main(int argc, char** argv) {
    const Options opt = parse(argc, argv);
    print_device();
    if (opt.probes) run_probes(opt.min_ms);

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    const float alpha = 1.f, beta = 0.f;

    for (int n : opt.sizes) {
        if (n % 128) {
            fprintf(stderr, "skipping size %d: must be a multiple of 128\n", n);
            continue;
        }
        const int M = n, N = n, K = n;
        const size_t elems = (size_t)n * n, bytes = elems * sizeof(float);
        const double flops = 2.0 * M * N * K;
        fprintf(stderr, "\nsize %d x %d x %d\n", M, N, K);

        std::vector<float> hA(elems), hB(elems), hC(elems), hRef(elems);
        for (auto& x : hA) x = dist(rng);
        for (auto& x : hB) x = dist(rng);
        float *A, *B, *C;
        CUDA_CHECK(cudaMalloc(&A, bytes));
        CUDA_CHECK(cudaMalloc(&B, bytes));
        CUDA_CHECK(cudaMalloc(&C, bytes));
        CUDA_CHECK(cudaMemcpy(A, hA.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(B, hB.data(), bytes, cudaMemcpyHostToDevice));

        // cuBLAS is column-major: row-major C = A @ B  <=>  column-major C^T = B^T @ A^T.
        auto cublas = [&] {
            CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, N,
                                     A, K, &beta, C, N));
        };
        CUDA_CHECK(cudaMemset(C, 0, bytes));
        const float ref_ms = time_ms(cublas, opt.min_ms);
        const double ref_gflops = flops / (ref_ms * 1e-3) / 1e9;
        CUDA_CHECK(cudaMemcpy(hRef.data(), C, bytes, cudaMemcpyDeviceToHost));
        double ref_max = 0;
        for (float v : hRef) ref_max = std::max(ref_max, (double)std::fabs(v));
        printf("{\"type\":\"result\",\"kernel\":\"cublas\",\"name\":\"cuBLAS\",\"n\":%d,"
               "\"ms\":%.4f,\"gflops\":%.1f,\"pct_cublas\":100.0,\"max_rel_err\":0,"
               "\"load_intensity\":null}\n", n, ref_ms, ref_gflops);
        fflush(stdout);
        fprintf(stderr, "  %-22s %9.3f ms %9.1f GFLOPS\n", "cuBLAS", ref_ms, ref_gflops);

        for (const KernelInfo& k : KERNELS) {
            if (!opt.kernels.empty() &&
                std::find(opt.kernels.begin(), opt.kernels.end(), k.id) == opt.kernels.end())
                continue;
            if (n % k.size_multiple) continue;
            CUDA_CHECK(cudaMemset(C, 0, bytes));
            const float ms = time_ms([&] { k.launch(M, N, K, alpha, A, B, beta, C); }, opt.min_ms);
            const double gflops = flops / (ms * 1e-3) / 1e9;
            double err = -1;
            if (opt.verify) {
                CUDA_CHECK(cudaMemcpy(hC.data(), C, bytes, cudaMemcpyDeviceToHost));
                double worst = 0;
                for (size_t i = 0; i < elems; ++i)
                    worst = std::max(worst, (double)std::fabs(hC[i] - hRef[i]));
                err = worst / ref_max;
            }
            const bool ok = !opt.verify || err < 1e-3;
            printf("{\"type\":\"result\",\"kernel\":\"%s\",\"name\":\"%s\",\"idea\":\"%s\","
                   "\"n\":%d,\"ms\":%.4f,\"gflops\":%.1f,\"pct_cublas\":%.1f,"
                   "\"max_rel_err\":%.2e,\"correct\":%s,\"load_intensity\":%.3f}\n",
                   k.id, k.name, k.idea, n, ms, gflops, 100.0 * gflops / ref_gflops, err,
                   !opt.verify ? "null" : ok ? "true" : "false", k.load_intensity);
            fflush(stdout);
            fprintf(stderr, "  %-22s %9.3f ms %9.1f GFLOPS  %5.1f%% of cuBLAS  err %.1e%s\n",
                    k.name, ms, gflops, 100.0 * gflops / ref_gflops, err, ok ? "" : "  WRONG");
        }
        CUDA_CHECK(cudaFree(A));
        CUDA_CHECK(cudaFree(B));
        CUDA_CHECK(cudaFree(C));
    }
    cublasDestroy(handle);
    printf("{\"type\":\"done\"}\n");
    fflush(stdout);
    return 0;
}
