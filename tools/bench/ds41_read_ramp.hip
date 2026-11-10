/* Read rate against launch size on gfx1151: what a decode-sized launch
 * (8-64 MB) can reach against the 2 GiB stream the 240 GB/s ceiling was
 * measured on (Gufo's tools/bench/gfx1151_peak.hip). Buffers rotate through
 * a pool larger than the caches, launches are dependent (one stream, back to
 * back) as in a decode step, and every pattern's checksum is compared with
 * the host's. Patterns:
 *   stride  every thread strides over the whole buffer (the peak bench)
 *   rows    workgroup g reads its own contiguous span, one 16-byte vector
 *           per lane per step (the GEMV layout: a wave walks its rows)
 * A last table rereads one buffer of growing size, which shows the cache
 * capacities (2 MiB and 32 MiB here) and their read rates.
 * Arguments: repetitions (default 200), pool MiB (default 1536). */
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

using u32x4 = __attribute__((__vector_size__(16))) std::uint32_t;

__global__ void read_stride(const u32x4* __restrict__ src, std::uint32_t* __restrict__ sink, std::size_t n) {
  const std::size_t id = (std::size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t stride = (std::size_t)gridDim.x * blockDim.x;
  u32x4 acc{};
  for (std::size_t i = id; i < n; i += stride) acc += src[i];
  sink[id] = acc[0] + acc[1] + acc[2] + acc[3];
}

/* Workgroup g owns vectors [g * span, (g + 1) * span); its threads stride
 * through it, kUnroll loads in flight per lane. */
template <int kUnroll>
__global__ void read_rows(const u32x4* __restrict__ src, std::uint32_t* __restrict__ sink, std::size_t n,
                          std::size_t span) {
  const std::size_t id = (std::size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t begin = (std::size_t)blockIdx.x * span;
  const std::size_t end = begin + span < n ? begin + span : n;
  u32x4 acc{};
  std::size_t i = begin + threadIdx.x;
  for (; i + (kUnroll - 1) * blockDim.x < end; i += kUnroll * blockDim.x) {
    u32x4 v[kUnroll];
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) v[u] = src[i + u * blockDim.x];
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) acc += v[u];
  }
  for (; i < end; i += blockDim.x) acc += src[i];
  sink[id] = acc[0] + acc[1] + acc[2] + acc[3];
}

}  // namespace

int main(int argc, char** argv) {
  const int reps = argc > 1 ? std::atoi(argv[1]) : 200;
  const std::size_t pool_bytes = (std::size_t)(argc > 2 ? std::atoi(argv[2]) : 1536) << 20;
  hipDeviceProp_t prop{};
  (void)hipGetDeviceProperties(&prop, 0);
  const int cus = prop.multiProcessorCount;
  u32x4* pool{};
  std::uint32_t* sink{};
  if (hipMalloc(&pool, pool_bytes) != hipSuccess || hipMalloc(&sink, (64u << 20)) != hipSuccess) return 1;
  (void)hipMemset(pool, 1, pool_bytes);
  (void)hipDeviceSynchronize();
  hipEvent_t a, b;
  (void)hipEventCreate(&a);
  (void)hipEventCreate(&b);
  const auto time_us = [&](std::size_t bytes, auto launch) {
    const std::size_t copies = pool_bytes / bytes;
    for (std::size_t i = 0; i < std::min<std::size_t>(copies, 8); ++i) launch(pool + i * (bytes / 16));
    (void)hipDeviceSynchronize();
    (void)hipEventRecord(a);
    for (int i = 0; i < reps; ++i) launch(pool + (i % copies) * (bytes / 16));
    (void)hipEventRecord(b);
    (void)hipEventSynchronize(b);
    float ms = 0.0f;
    (void)hipEventElapsedTime(&ms, a, b);
    return (double)ms * 1e3 / reps;
  };
  const auto check = [&](std::size_t bytes, std::size_t threads) {
    std::vector<std::uint32_t> host(threads);
    (void)hipMemcpy(host.data(), sink, threads * 4, hipMemcpyDeviceToHost);
    std::uint64_t sum = 0;
    for (auto v : host) sum += v;
    return (std::uint32_t)sum == (std::uint32_t)(bytes / 16 * 4 * 0x01010101ull);
  };
  std::printf("device %s, %d CUs, reps %d, pool %zu MiB\n", prop.name, cus, reps, pool_bytes >> 20);
  std::printf("%-22s", "pattern \\ MiB");
  const std::vector<std::size_t> sizes = {4, 8, 12, 16, 24, 32, 44, 64, 128, 256, 512};
  for (auto s : sizes) std::printf(" %7zu", s);
  std::printf("   (us per launch, then GB/s)\n");
  struct Row {
    const char* name;
    int kind, wgs_per_cu, threads, unroll;
    std::size_t span_bytes;
  };
  const std::vector<Row> rows = {
      {"stride 32wg/cu x256", 0, 32, 256, 1, 0},
      {"stride 16wg/cu x256", 0, 16, 256, 1, 0},
      {"stride 8wg/cu x256", 0, 8, 256, 1, 0},
      {"rows 64K u1", 1, 0, 256, 1, 64 << 10},
      {"rows 64K u2", 1, 0, 256, 2, 64 << 10},
      {"rows 64K u4", 1, 0, 256, 4, 64 << 10},
      {"rows 16K u2", 1, 0, 256, 2, 16 << 10},
      {"rows 256K u2", 1, 0, 256, 2, 256 << 10},
      {"rows 1M u2", 1, 0, 256, 2, 1 << 20},
      {"rows 1M u4", 1, 0, 256, 4, 1 << 20},
  };
  for (const Row& r : rows) {
    std::vector<double> us;
    bool ok = true;
    for (auto mib : sizes) {
      const std::size_t bytes = mib << 20, n = bytes / 16;
      if (r.kind == 0) {
        const int blocks = cus * r.wgs_per_cu;
        us.push_back(time_us(bytes, [&](const u32x4* p) { read_stride<<<blocks, r.threads>>>(p, sink, n); }));
        ok = ok && check(bytes, (std::size_t)blocks * r.threads);
      } else {
        const std::size_t span = r.span_bytes / 16;
        const int blocks = (int)((n + span - 1) / span);
        us.push_back(time_us(bytes, [&](const u32x4* p) {
          if (r.unroll == 1)
            read_rows<1><<<blocks, r.threads>>>(p, sink, n, span);
          else if (r.unroll == 2)
            read_rows<2><<<blocks, r.threads>>>(p, sink, n, span);
          else
            read_rows<4><<<blocks, r.threads>>>(p, sink, n, span);
        }));
        ok = ok && check(bytes, (std::size_t)blocks * r.threads);
      }
    }
    std::printf("%-22s", r.name);
    for (double u : us) std::printf(" %7.1f", u);
    std::printf("%s\n%-22s", ok ? "" : "  CHECKSUM MISMATCH", "");
    for (std::size_t i = 0; i < sizes.size(); ++i) std::printf(" %7.1f", (double)(sizes[i] << 20) / us[i] / 1e3);
    std::printf("\n");
  }
  /* The same buffer read again by every launch: what the caches hold. */
  std::printf("\nreread of one buffer (stride pattern, %d workgroups):\n", cus * 8);
  for (double mib : {0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 8.0, 16.0, 24.0, 32.0, 48.0, 64.0}) {
    const std::size_t bytes = (std::size_t)(mib * 1048576), n = bytes / 16;
    const int blocks = cus * 8;
    read_stride<<<blocks, 256>>>(pool, sink, n);
    (void)hipDeviceSynchronize();
    (void)hipEventRecord(a);
    for (int i = 0; i < reps; ++i) read_stride<<<blocks, 256>>>(pool, sink, n);
    (void)hipEventRecord(b);
    (void)hipEventSynchronize(b);
    float ms = 0.0f;
    (void)hipEventElapsedTime(&ms, a, b);
    std::printf("  %6.2f MiB: %8.1f us per launch, %7.1f GB/s\n", mib, ms * 1e3 / reps, bytes * (double)reps / ms / 1e6);
  }
  return 0;
}
