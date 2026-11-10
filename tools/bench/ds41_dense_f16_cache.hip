/* Workgroup order of the dense FP16 prefill kernel (dense_f16_kernel).
 * gfx1151 reads its 32 MiB last-level cache at ~950 GB/s against ~239 GB/s
 * from DRAM, and a 2,048-token batch sweeps each weight matrix once per
 * 128-token tile. For every prefill shape this times the orders "chunks of
 * T token tiles, inside each chunks of R row blocks" over a grid of R and T
 * (R = all rows, T = 1 is one sweep per tile) and the order dense_chunks
 * picks, and checks that every order gives the same output bits. Weights
 * rotate through copies so the first touch of each is DRAM-cold.
 * Arguments: type (8, 14, 13; default 8), tokens (default 2048), a list of
 * shape names (default all), and any fourth argument to time only the
 * unchunked order and the production choice. */
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "../../kernels/rocm/ds41_dense_f16.hip.cpp"

int main(int argc, char** argv) {
  setbuf(stdout, nullptr);
  const int type = argc > 1 ? std::atoi(argv[1]) : 8;
  const uint32_t n = argc > 2 ? std::atoi(argv[2]) : 2048;
  const int block = type == 8 ? 272 : type == 14 ? 210 : 176;
  const struct {
    const char* name;
    uint32_t m, k, groups;
    uint32_t io;
  } shapes[] = {
      {"q_b", 32768, 1280, 1, DS41_V41_DENSE_X_HALF},
      {"o_b", 5120, 8192, 1, DS41_V41_DENSE_X_HALF},
      {"wo_a", 1024, 4096, 8, DS41_V41_DENSE_X_HALF | DS41_V41_DENSE_OUT_HALF},
      {"sh_gate+up", 4608, 5120, 1, 0},
      {"sh_down", 5120, 2304, 1, DS41_V41_DENSE_X_HALF},
      {"q_a+kv", 1792, 5120, 1, DS41_V41_DENSE_X_HALF},
      {"engram", 25600, 6144, 1, 0},
  };
  std::mt19937 rng(41);
  hipEvent_t a, b;
  (void)hipEventCreate(&a);
  (void)hipEventCreate(&b);
  for (const auto& s : shapes) {
    const std::size_t row_bytes = (std::size_t)s.k / 256 * block;
    const std::size_t bytes = (std::size_t)s.groups * s.m * row_bytes;
    const int copies = (int)std::max<std::size_t>(3, (512u << 20) / bytes);
    if (argc > 3 && !std::strstr(argv[3], s.name)) continue;
    std::vector<uint8_t> host(bytes);
    for (auto& v : host) v = (uint8_t)rng();
    for (std::size_t i = 0; i < bytes / block; ++i) {
      const uint16_t d = 0x1c00 | (rng() & 0x3ff);
      if (type == 8) {
        for (int j = 0; j < 8; ++j) std::memcpy(&host[i * block + 34 * j], &d, 2);
      } else {
        std::memcpy(&host[i * block + (type == 14 ? 208 : 0)], &d, 2);
        if (type == 13) std::memcpy(&host[i * block + 2], &d, 2);
      }
    }
    std::vector<uint8_t*> w(copies);
    for (auto& p : w) {
      (void)hipMalloc(&p, bytes);
      (void)hipMemcpy(p, host.data(), bytes, hipMemcpyHostToDevice);
    }
    const std::size_t x_n = (std::size_t)s.groups * s.k * n;
    const bool x_half = s.io & DS41_V41_DENSE_X_HALF;
    void *x = nullptr, *y = nullptr;
    (void)hipMalloc(&x, x_n * 4);
    (void)hipMemset(x, 0x3c, x_n * 4);
    (void)hipMalloc(&y, (std::size_t)s.groups * s.m * n * 4);
    const double flops = 2.0 * s.m * s.k * n * s.groups;
    std::printf("%-10s %5ux%-5u g%u W %5.1f MB X %5.1f MB:", s.name, s.m, s.k, s.groups, bytes / 1e6,
                x_n * (x_half ? 2 : 4) / 1e6);
    const uint32_t row_blocks = s.m / 256, tiles = n / 128;
    const double tile_w = 256.0 * row_bytes / 1048576, tile_x = 128.0 * s.k * (x_half ? 2 : 4) / 1048576;
    std::printf(" row block %.2f MiB, token tile %.2f MiB, %u x %u workgroups per group\n", tile_w, tile_x, row_blocks,
                tiles);
    uint64_t want = 0;
    const size_t y_bytes = (std::size_t)s.groups * s.m * n * ((s.io & DS41_V41_DENSE_OUT_HALF) ? 2 : 4);
    std::vector<uint8_t> yh(y_bytes);
    for (uint32_t cr : {row_blocks, 32u, 16u, 8u, 5u, 4u, 2u, 1u}) {
      if (cr > row_blocks || (cr != row_blocks && argc > 4)) continue;
      std::printf("   rows/chunk %3u (%5.1f MiB):", cr, cr * tile_w);
      for (uint32_t ct : {1u, 2u, 4u, 8u, 16u}) {
        if (ct > tiles || (ct != 1 && argc > 4)) continue;
        const DenseChunks order{cr, ct};
        const auto run = [&](const uint8_t* p) {
          (void)dense_run(type, {p, x, y, s.m, n, s.k, s.groups, nullptr, nullptr, 0}, s.io, &order);
        };
        (void)hipMemset(y, 0, y_bytes);
        run(w[0]);
        (void)hipDeviceSynchronize();
        (void)hipMemcpy(yh.data(), y, y_bytes, hipMemcpyDeviceToHost);
        uint64_t hash = 1469598103934665603ull;
        for (size_t i = 0; i + 8 <= y_bytes; i += 8) {
          uint64_t v;
          std::memcpy(&v, &yh[i], 8);
          hash = (hash ^ v) * 1099511628211ull;
        }
        if (!want) want = hash;
        for (int i = 0; i < copies; ++i) run(w[i]);
        (void)hipDeviceSynchronize();
        const int iters = std::max(copies, 9);
        (void)hipEventRecord(a, 0);
        for (int i = 0; i < iters; ++i) run(w[i % copies]);
        (void)hipEventRecord(b, 0);
        (void)hipEventSynchronize(b);
        float ms = 0;
        (void)hipEventElapsedTime(&ms, a, b);
        ms /= iters;
        std::printf("  x%-2u %5.2f ms%s", ct, ms, hash == want ? "" : " MISMATCH");
      }
      std::printf("\n");
    }
    {
      const auto run = [&](const uint8_t* p) { (void)ds41_v41_dense_f16(type, p, x, y, s.m, n, s.k, s.groups, s.io); };
      (void)hipMemset(y, 0, y_bytes);
      run(w[0]);
      (void)hipDeviceSynchronize();
      (void)hipMemcpy(yh.data(), y, y_bytes, hipMemcpyDeviceToHost);
      uint64_t hash = 1469598103934665603ull;
      for (size_t i = 0; i + 8 <= y_bytes; i += 8) {
        uint64_t v;
        std::memcpy(&v, &yh[i], 8);
        hash = (hash ^ v) * 1099511628211ull;
      }
      for (int i = 0; i < copies; ++i) run(w[i]);
      (void)hipDeviceSynchronize();
      const int iters = std::max(copies, 9);
      (void)hipEventRecord(a, 0);
      for (int i = 0; i < iters; ++i) run(w[i % copies]);
      (void)hipEventRecord(b, 0);
      (void)hipEventSynchronize(b);
      float ms = 0;
      (void)hipEventElapsedTime(&ms, a, b);
      const DenseChunks c = dense_chunks(256 * row_bytes, 128ull * s.k * (x_half ? 2 : 4), row_blocks, std::max(tiles, 1u));
      std::printf("   production (%u rows x %u tiles): %5.2f ms%s, hash %016llx", c.rows, c.tiles, ms / iters,
                  hash == want ? "" : " MISMATCH", (unsigned long long)want);
    }
    std::printf("\n");
    for (auto* p : w) (void)hipFree(p);
    (void)hipFree(x);
    (void)hipFree(y);
  }
  return 0;
}
