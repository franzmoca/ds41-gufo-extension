/* The batched hyper-connection state pass (ds41_v41_hc_state_pass) at the
 * prefill batch size: the attention-expand form (rows 0) and the FFN-expand
 * form (six routed rows), with the bytes each moves. Prints an FNV hash of
 * the outputs so variants can be compared bit for bit. Arguments: tokens
 * (default 2048), repetitions (default 30). */
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "../../kernels/rocm/ds41_v41.hip.cpp"

int main(int argc, char** argv) {
  const uint32_t n = argc > 1 ? std::atoi(argv[1]) : 2048;
  const int reps = argc > 2 ? std::atoi(argv[2]) : 30;
  constexpr uint32_t kDim = 5120, kHc = 4, kMix = 24, kRows = 6;
  std::mt19937 rng(7);
  std::normal_distribution<float> nd;
  const auto dev = [&](size_t count, float scale) {
    std::vector<float> host(count);
    for (auto& v : host) v = nd(rng) * scale;
    float* p{};
    (void)hipMalloc(&p, count * 4);
    (void)hipMemcpy(p, host.data(), count * 4, hipMemcpyHostToDevice);
    return p;
  };
  float* y = dev((size_t)n * kDim, 1.0f);
  float* x = dev((size_t)n * kRows * kDim, 1.0f);
  float* h = dev((size_t)n * kHc * kDim, 1.0f);
  float* h_out = dev((size_t)n * kHc * kDim, 0.0f);
  float* xn = dev((size_t)n * kDim, 0.0f);
  float* post = dev((size_t)n * kHc, 0.5f);
  float* comb = dev((size_t)n * kHc * kHc, 0.5f);
  float* pre = dev((size_t)n * kHc, 0.5f);
  float* norm_w = dev(kDim, 1.0f);
  float* fn = dev((size_t)kMix * kHc * kDim, 0.02f);
  float* mixes = dev((size_t)n * kMix, 0.0f);
  hipEvent_t a, b;
  (void)hipEventCreate(&a);
  (void)hipEventCreate(&b);
  const auto hash = [&](const float* p, size_t count, uint64_t hsh) {
    std::vector<uint32_t> host(count);
    (void)hipMemcpy(host.data(), p, count * 4, hipMemcpyDeviceToHost);
    for (uint32_t v : host) hsh = (hsh ^ v) * 1099511628211ull;
    return hsh;
  };
  struct Form {
    const char* name;
    bool expand;
    uint32_t rows;
  };
  for (const Form f : {Form{"collapse only", false, 0}, Form{"attention expand", true, 0}, Form{"ffn expand", true, kRows}}) {
    const auto run = [&] {
      return ds41_v41_hc_state_pass(f.expand ? y : nullptr, f.rows ? x : nullptr, f.rows, post, comb, h,
                                    f.expand ? h_out : nullptr, pre, norm_w, xn, fn, mixes, kHc, kDim, 1e-6f, n);
    };
    if (run()) return 1;
    (void)hipDeviceSynchronize();
    uint64_t hsh = hash(xn, (size_t)n * kDim, 1469598103934665603ull);
    hsh = hash(mixes, (size_t)n * kMix, hsh);
    if (f.expand) hsh = hash(h_out, (size_t)n * kHc * kDim, hsh);
    (void)hipEventRecord(a);
    for (int i = 0; i < reps; ++i) (void)run();
    (void)hipEventRecord(b);
    (void)hipEventSynchronize(b);
    float ms = 0.0f;
    (void)hipEventElapsedTime(&ms, a, b);
    ms /= reps;
    const double read = (double)n * kDim * 4 * (kHc + (f.expand ? 1 : 0) + f.rows);
    const double write = (double)n * kDim * 4 * (1 + (f.expand ? kHc : 0));
    std::printf("%-17s n %u: %7.3f ms  read %5.1f MB write %5.1f MB  %6.1f GB/s  hash %016llx\n", f.name, n, ms,
                read / 1e6, write / 1e6, (read + write) / ms / 1e6, (unsigned long long)hsh);
  }
  return 0;
}
