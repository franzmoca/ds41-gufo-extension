/* The batched BF16 matmul (ds41_v41_matmul_bf16) on the prefill shapes: the
 * router, the compressor KV and the indexer q_b at prompt batch sizes, with
 * an FNV hash of each output so two builds can be compared bit for bit
 * (build the file at both revisions and diff the output). Arguments:
 * repetitions (default 40), then token counts to time the router at
 * instead of the shape list. */
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "../../kernels/rocm/ds41_v41.hip.cpp"

int main(int argc, char** argv) {
  const int reps = argc > 1 ? std::atoi(argv[1]) : 40;
  struct Shape {
    const char* name;
    uint32_t rows, k, n;
  };
  std::vector<Shape> list = {
      {"router", 384, 5120, 2048},       {"router", 384, 5120, 1100},   {"router", 384, 5120, 140},
      {"router", 384, 5120, 64},         {"router", 384, 5120, 20},     {"compressor", 512, 5120, 2048},
      {"indexer q_b", 4096, 1280, 2048}, {"indexer q_b", 4096, 1280, 140},
  };
  if (argc > 2) {
    list.clear();
    for (int i = 2; i < argc; ++i) list.push_back({"router", 384, 5120, (uint32_t)std::atoi(argv[i])});
  }
  std::mt19937 rng(11);
  std::normal_distribution<float> nd;
  hipEvent_t a, b;
  (void)hipEventCreate(&a);
  (void)hipEventCreate(&b);
  for (const auto& s : list) {
    std::vector<uint16_t> wh((size_t)s.rows * s.k);
    for (auto& v : wh) {
      const float f = nd(rng) * 0.05f;
      uint32_t u;
      __builtin_memcpy(&u, &f, 4);
      v = (uint16_t)(u >> 16);
    }
    /* Activations over several binades, with exact zeros and denormal-range values. */
    std::vector<float> xh((size_t)s.n * s.k);
    for (size_t i = 0; i < xh.size(); ++i) {
      const float scale = std::ldexp(1.0f, (int)(rng() % 12) - 6);
      xh[i] = i % 97 == 0 ? 0.0f : i % 1013 == 0 ? 1e-30f : nd(rng) * scale;
    }
    uint16_t* w{};
    float *x{}, *y{};
    (void)hipMalloc(&w, wh.size() * 2);
    (void)hipMalloc(&x, xh.size() * 4);
    (void)hipMalloc(&y, (size_t)s.n * s.rows * 4);
    (void)hipMemcpy(w, wh.data(), wh.size() * 2, hipMemcpyHostToDevice);
    (void)hipMemcpy(x, xh.data(), xh.size() * 4, hipMemcpyHostToDevice);
    if (ds41_v41_matmul_bf16(w, x, y, s.rows, s.k, s.n)) return 1;
    (void)hipDeviceSynchronize();
    std::vector<uint32_t> yh((size_t)s.n * s.rows);
    (void)hipMemcpy(yh.data(), y, yh.size() * 4, hipMemcpyDeviceToHost);
    uint64_t hash = 1469598103934665603ull;
    for (uint32_t v : yh) hash = (hash ^ v) * 1099511628211ull;
    (void)hipEventRecord(a);
    for (int i = 0; i < reps; ++i) (void)ds41_v41_matmul_bf16(w, x, y, s.rows, s.k, s.n);
    (void)hipEventRecord(b);
    (void)hipEventSynchronize(b);
    float ms = 0.0f;
    (void)hipEventElapsedTime(&ms, a, b);
    ms /= reps;
    std::printf("%-12s %4u x %4u, %4u tokens: %8.1f us  %5.2f TFLOP/s (one pass)  hash %016llx\n", s.name, s.rows, s.k,
                s.n, ms * 1e3, 2.0 * s.rows * s.k * s.n / ms / 1e9, (unsigned long long)hash);
    (void)hipFree(w);
    (void)hipFree(x);
    (void)hipFree(y);
  }
  return 0;
}
