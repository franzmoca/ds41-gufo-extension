/* Single-token decode projections for the V4.1 Flash dense Q5_K, Q6_K and
 * Q8_0 weights (and the DSpark drafter's Q4_K): weights stay in their GGUF
 * blocks, activations stay FP32. A
 * lane's unit of work is one 16-byte vector of quantized values; lanes
 * holding the same task of different rows read the same activations, which
 * the hardware broadcasts. A workgroup's eight waves form 8 / S row groups
 * of S waves; the S waves of a group take interleaved super-blocks of its
 * rows and are summed in wave order. */
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>

#include "ds41_gemv_tasks.hpp"
#include "ds41_v41.h"

namespace {

using namespace ds41_gemv_tasks;

constexpr int kWave = 32;
constexpr int kWaves = 8;

template <int kType>
struct Format;
template <>
struct Format<13> {  // Q5_K
  static constexpr int kBlockBytes = 176;
  static constexpr int kTasks = 8;
  using Raw = RawQ45K;
  using SplitRaw = Raw;
  static __device__ __forceinline__ Raw Load(const uint8_t* b, int t) { return LoadQ45K<true>(b, t); }
  static __device__ __forceinline__ Raw LoadSplit(const uint8_t* b, int t) { return Load(b, t); }
  static __device__ __forceinline__ void DecodeSplit(const Raw& r, int t, float (&w)[32]) { Decode(r, t, w); }
  static __device__ __forceinline__ void Decode(const Raw& r, int t, float (&w)[32]) { DecodeQ45K<true>(r, t, w); }
  static __device__ __forceinline__ void Decode(const uint8_t* b, int t, float (&w)[32]) { DecodeQ45K<true>(b, t, w); }
  static __device__ __forceinline__ float Dot(float acc, const float (&w)[32], int t, const float* x) {
    return DotQ45K(acc, w, t, x);
  }
};
template <>
struct Format<12> {  // Q4_K (DSpark drafter projections)
  static constexpr int kBlockBytes = 144;
  static constexpr int kTasks = 8;
  using Raw = RawQ45K;
  using SplitRaw = Raw;
  static __device__ __forceinline__ Raw Load(const uint8_t* b, int t) { return LoadQ45K<false>(b, t); }
  static __device__ __forceinline__ Raw LoadSplit(const uint8_t* b, int t) { return Load(b, t); }
  static __device__ __forceinline__ void DecodeSplit(const Raw& r, int t, float (&w)[32]) { Decode(r, t, w); }
  static __device__ __forceinline__ void Decode(const Raw& r, int t, float (&w)[32]) { DecodeQ45K<false>(r, t, w); }
  static __device__ __forceinline__ void Decode(const uint8_t* b, int t, float (&w)[32]) { DecodeQ45K<false>(b, t, w); }
  static __device__ __forceinline__ float Dot(float acc, const float (&w)[32], int t, const float* x) {
    return DotQ45K(acc, w, t, x);
  }
};
template <>
struct Format<14> {  // Q6_K (Unsloth's dense projections)
  static constexpr int kBlockBytes = 210;
  static constexpr int kTasks = 8;
  using SplitRaw = RawQ6K32;
  static __device__ __forceinline__ SplitRaw LoadSplit(const uint8_t* b, int t) { return LoadQ6K32(b, t); }
  static __device__ __forceinline__ void DecodeSplit(const SplitRaw& r, int t, float (&w)[32]) {
    DecodeQ6K32(r, t, w);
  }
  static __device__ __forceinline__ void Decode(const uint8_t* b, int t, float (&w)[32]) { DecodeQ6K32(b, t, w); }
  static __device__ __forceinline__ float Dot(float acc, const float (&w)[32], int t, const float* x) {
    const float* xv = x + OffsetQ6K32(t);
    acc = Accumulate<16>(acc, w, xv);
    return Accumulate<16>(acc, w + 16, xv + 32);
  }
};
struct RawQ8_0 {
  uint32_t r[9];
};
template <>
struct Format<8> {  // Q8_0, eight 34-byte blocks per 256 values
  static constexpr int kBlockBytes = 272;
  static constexpr int kTasks = 8;
  /* The split kernel's register ring: the nine words around block t, decoded
   * as DecodeQ8_0 does. */
  struct SplitRaw : RawQ8_0 {};
  static __device__ __forceinline__ SplitRaw LoadSplit(const uint8_t* b, int t) {
    const auto* words = reinterpret_cast<const uint32_t*>(b + 34 * t - (t & 1 ? 2 : 0));
    SplitRaw r;
#pragma unroll
    for (int i = 0; i < 9; ++i) r.r[i] = words[i];
    return r;
  }
  static __device__ __forceinline__ void DecodeSplit(const SplitRaw& raw, int t, float (&w)[32]) {
    const bool odd = (t & 1) != 0;
    const float d = HalfBits(odd ? raw.r[0] >> 16 : raw.r[0]);
    uint32_t qs[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) qs[i] = odd ? raw.r[i + 1] : __builtin_amdgcn_alignbyte(raw.r[i + 1], raw.r[i], 2);
    const auto* q = reinterpret_cast<const int8_t*>(qs);
#pragma unroll
    for (int i = 0; i < 32; ++i) w[i] = d * static_cast<float>(q[i]);
  }
  static __device__ __forceinline__ void Decode(const uint8_t* b, int t, float (&w)[32]) {
    DecodeQ8_0(b + 34 * t, (t & 1) != 0, w);
  }
  static __device__ __forceinline__ float Dot(float acc, const float (&w)[32], int t, const float* x) {
    return Accumulate<32>(acc, w, x + 32 * t);
  }
};

/* Row r of token c reads x + c * x_stride + (r / rows_per_x) * k, for the
 * first n <= kCols tokens. Rows from m1 on come from w2 and go to y2 (row
 * r - m1, token stride m - m1); the others to y (token stride y_stride). With
 * kSwiGLU, w_up holds a second matrix of the same shape whose rows are taken
 * in the same pass, and row r stores silu(min(gate, limit)) * clamp(up,
 * +-limit). Each lane decodes its weights once for all tokens; every token
 * sees the FMA order of a single-token pass. */
template <int kType, int S, bool kSwiGLU, int kCols>
__global__ void __launch_bounds__(kWave* kWaves)
    gemv_kernel(const uint8_t* __restrict__ w, const uint8_t* __restrict__ w_up, float limit,
                const float* __restrict__ x, uint64_t x_stride, float* __restrict__ y, uint32_t y_stride, uint32_t m,
                uint32_t k, uint32_t rows_per_x, const uint8_t* __restrict__ w2, float* __restrict__ y2, uint32_t m1,
                uint32_t n) {
  using F = Format<kType>;
  constexpr int kRowsPerWave = kWave / F::kTasks;
  constexpr int kGroups = kWaves / S;
  __shared__ float partial[S > 1 ? kCols : 1][kWaves][kWave];
  __shared__ float partial_up[S > 1 && kSwiGLU ? kCols : 1][kWaves][kWave];
  const int lane = threadIdx.x % kWave;
  const int wave = threadIdx.x / kWave;
  const int group = wave / S;
  const int slice = wave % S;
  const int t = lane % F::kTasks;
  const uint32_t row = (blockIdx.x * kGroups + group) * kRowsPerWave + lane / F::kTasks;
  const uint32_t blocks_per_row = k / 256;
  float acc[kCols] = {};
  float acc_up[kCols] = {};
  const bool second = row >= m1;
  if (row < m) {
    const uint64_t offset = (uint64_t)(second ? row - m1 : row) * blocks_per_row * F::kBlockBytes;
    const uint8_t* base = (second ? w2 : w) + offset;
    const float* xr = x + (uint64_t)(row / rows_per_x) * k;
    if constexpr (kCols > 1 && requires { typename F::Raw; }) {
      /* Several tokens make each block's FMAs long; the next block's
       * weights load before this block is decoded. */
      const uint8_t* base_up = kSwiGLU ? w_up + offset : nullptr;
      typename F::Raw next{}, next_up{};
      if (slice < (int)blocks_per_row) {
        next = F::Load(base + (uint64_t)slice * F::kBlockBytes, t);
        if constexpr (kSwiGLU) next_up = F::Load(base_up + (uint64_t)slice * F::kBlockBytes, t);
      }
      for (uint32_t blk = slice; blk < blocks_per_row; blk += S) {
        const typename F::Raw raw = next, raw_up = next_up;
        if (blk + S < blocks_per_row) {
          next = F::Load(base + (uint64_t)(blk + S) * F::kBlockBytes, t);
          if constexpr (kSwiGLU) next_up = F::Load(base_up + (uint64_t)(blk + S) * F::kBlockBytes, t);
        }
        float wv[32];
        F::Decode(raw, t, wv);
#pragma unroll
        for (int c = 0; c < kCols; ++c)
          if (c < (int)n) acc[c] = F::Dot(acc[c], wv, t, xr + c * x_stride + blk * 256);
        if constexpr (kSwiGLU) {
          F::Decode(raw_up, t, wv);
#pragma unroll
          for (int c = 0; c < kCols; ++c)
            if (c < (int)n) acc_up[c] = F::Dot(acc_up[c], wv, t, xr + c * x_stride + blk * 256);
        }
      }
    } else
#pragma unroll 2
    for (uint32_t blk = slice; blk < blocks_per_row; blk += S) {
      float wv[32];
      F::Decode(base + (uint64_t)blk * F::kBlockBytes, t, wv);
#pragma unroll
      for (int c = 0; c < kCols; ++c)
        if (kCols == 1 || c < (int)n) acc[c] = F::Dot(acc[c], wv, t, xr + c * x_stride + blk * 256);
      if constexpr (kSwiGLU) {
        F::Decode(w_up + offset + (uint64_t)blk * F::kBlockBytes, t, wv);
#pragma unroll
        for (int c = 0; c < kCols; ++c)
          if (kCols == 1 || c < (int)n) acc_up[c] = F::Dot(acc_up[c], wv, t, xr + c * x_stride + blk * 256);
      }
    }
  }
#pragma unroll
  for (int c = 0; c < kCols; ++c) {
#pragma unroll
    for (int offset = F::kTasks / 2; offset > 0; offset >>= 1) {
      acc[c] += __shfl_xor(acc[c], offset, kWave);
      if constexpr (kSwiGLU) acc_up[c] += __shfl_xor(acc_up[c], offset, kWave);
    }
  }
  if constexpr (S > 1) {
#pragma unroll
    for (int c = 0; c < kCols; ++c) {
      partial[c][wave][lane] = acc[c];
      if constexpr (kSwiGLU) partial_up[c][wave][lane] = acc_up[c];
    }
    __syncthreads();
    if (slice != 0) return;
#pragma unroll
    for (int c = 0; c < kCols; ++c) {
#pragma unroll
      for (int s = 1; s < S; ++s) {
        acc[c] += partial[c][wave + s][lane];
        if constexpr (kSwiGLU) acc_up[c] += partial_up[c][wave + s][lane];
      }
    }
  }
  if (t != 0 || row >= m) return;
#pragma unroll
  for (int c = 0; c < kCols; ++c) {
    if (kCols > 1 && c >= (int)n) break;
    if constexpr (kSwiGLU) {
      float g = acc[c];
      float u = acc_up[c];
      if (limit > 0.0f) {
        u = fminf(fmaxf(u, -limit), limit);
        g = fminf(g, limit);
      }
      y[(uint64_t)c * y_stride + row] = g / (1.0f + expf(-g)) * u;
    } else if (second) {
      y2[(uint64_t)c * (m - m1) + row - m1] = acc[c];
    } else {
      y[(uint64_t)c * y_stride + row] = acc[c];
    }
  }
}


/* Several tokens over Q5_K rows. The lane layout above re-reads each
 * block's activations in every wave, eight tasks per load, and at 8 tokens
 * those vector loads, not the weights, bound it. Here wave w takes task w of
 * 32 rows (one per lane), so a wave's activations are uniform and come
 * through scalar loads; the vector unit loads only weights. Workgroup
 * (tile, slice) runs blocks slice, slice + S, ... in order, as a slice wave
 * above does, reduces the tasks in its xor-tree order
 * ((a0 + a4) + (a2 + a6)) + ((a1 + a5) + (a3 + a7)) and stores the slice
 * result; the tile's last workgroup (atomic counter) adds the S slice
 * results in wave order. With kSwiGLU, w_up's rows run in the same pass
 * (their slice results follow the gate's in part) and the last workgroup
 * stores the SwiGLU as the kernel above does. Every row and token matches
 * the kernel above bit for bit. */
template <int kType, int S, bool kSwiGLU, int kCols>
__global__ void __launch_bounds__(kWave* kWaves)
    split_gemv_kernel(const uint8_t* __restrict__ w, const uint8_t* __restrict__ w_up, float limit,
                      const float* __restrict__ x, uint64_t x_stride, float* __restrict__ y, uint32_t y_stride,
                      uint32_t m, uint32_t k, uint32_t rows_per_x, const uint8_t* __restrict__ w2,
                      float* __restrict__ y2, uint32_t m1, uint32_t n, float* __restrict__ part,
                      uint32_t* __restrict__ counter) {
  using F = Format<kType>;
  static_assert(F::kTasks == kWaves, "one wave per task");
  constexpr int kMats = kSwiGLU ? 2 : 1;
  __shared__ float red[kWaves / 2][kMats][kCols][kWave];
  const int lane = threadIdx.x % kWave;
  const int t = __builtin_amdgcn_readfirstlane(threadIdx.x / kWave);
  const uint32_t tile = blockIdx.x / S;
  const uint32_t slice = blockIdx.x % S;
  const uint32_t row0 = tile * kWave;
  const uint32_t row = min(row0 + lane, m - 1);
  const uint32_t blocks_per_row = k / 256;
  // m1 is a multiple of 32, so a workgroup's rows come from one matrix.
  const bool second = row0 >= m1;
  const uint64_t offset = (uint64_t)(second ? row - m1 : row) * blocks_per_row * F::kBlockBytes;
  const uint8_t* base[kMats];
  base[0] = (second ? w2 : w) + offset;
  if constexpr (kSwiGLU) base[kMats - 1] = w_up + offset;
  // rows_per_x is m or a multiple of 32, so a workgroup shares its x row.
  const float* xr = x + (uint64_t)(row0 / rows_per_x) * k;
  float acc[kMats][kCols] = {};
  typename F::SplitRaw next[kMats] = {};
  if (slice < blocks_per_row)
#pragma unroll
    for (int i = 0; i < kMats; ++i) next[i] = F::LoadSplit(base[i] + (uint64_t)slice * F::kBlockBytes, t);
  for (uint32_t blk = slice; blk < blocks_per_row; blk += S) {
    typename F::SplitRaw raw[kMats];
#pragma unroll
    for (int i = 0; i < kMats; ++i) {
      raw[i] = next[i];
      if (blk + S < blocks_per_row) next[i] = F::LoadSplit(base[i] + (uint64_t)(blk + S) * F::kBlockBytes, t);
    }
#pragma unroll
    for (int i = 0; i < kMats; ++i) {
      float wv[32];
      F::DecodeSplit(raw[i], t, wv);
#pragma unroll
      for (int c = 0; c < kCols; ++c)
        if (c < (int)n) acc[i][c] = F::Dot(acc[i][c], wv, t, xr + c * x_stride + blk * 256);
    }
  }
  /* Xor tree over the tasks: partner offset 4, then 2, then 1. */
#pragma unroll
  for (int offset = kWaves / 2; offset > 0; offset >>= 1) {
    if (t >= offset && t < 2 * offset)
#pragma unroll
      for (int i = 0; i < kMats; ++i)
#pragma unroll
        for (int c = 0; c < kCols; ++c) red[t - offset][i][c][lane] = acc[i][c];
    __syncthreads();
    if (t < offset)
#pragma unroll
      for (int i = 0; i < kMats; ++i)
#pragma unroll
        for (int c = 0; c < kCols; ++c) acc[i][c] += red[t][i][c][lane];
    __syncthreads();
  }
  if (t != 0) return;
  const bool valid = row0 + lane < m;  // part rows are exactly m apart
#pragma unroll
  for (int i = 0; i < kMats; ++i)
#pragma unroll
    for (int c = 0; c < kCols; ++c)
      if (c < (int)n && valid) part[(((uint64_t)i * S + slice) * kCols + c) * m + row0 + lane] = acc[i][c];
  __threadfence();
  uint32_t done = 0;
  if (lane == 0) done = atomicAdd(&counter[tile], 1u);
  if (__shfl(done, 0, kWave) != S - 1) return;
  __threadfence();
  if (lane == 0) counter[tile] = 0;  // ready for the next launch
  if (!valid) return;
#pragma unroll
  for (int c = 0; c < kCols; ++c) {
    if (c >= (int)n) break;
    float total[kMats];
#pragma unroll
    for (int i = 0; i < kMats; ++i) {
#pragma unroll
      for (int s = 0; s < S; ++s) {
        const float v = __hip_atomic_load(&part[(((uint64_t)i * S + s) * kCols + c) * m + row0 + lane],
                                          __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        total[i] = s ? total[i] + v : v;
      }
    }
    if constexpr (kSwiGLU) {
      float g = total[0];
      float u = total[1];
      if (limit > 0.0f) {
        u = fminf(fmaxf(u, -limit), limit);
        g = fminf(g, limit);
      }
      y[(uint64_t)c * y_stride + row0 + lane] = g / (1.0f + expf(-g)) * u;
    } else if (second) {
      y2[(uint64_t)c * (m - m1) + row0 + lane - m1] = total[0];
    } else {
      y[(uint64_t)c * y_stride + row0 + lane] = total[0];
    }
  }
}

/* Slice results and per-tile counters of split_gemv_kernel: every launch
 * runs on the null stream, so one buffer serves them in turn. 64 MiB holds
 * the 129,280-row output head's 8 slices for 16 tokens (DSpark verification),
 * which took 43-47% less time than the lane layout there. */
constexpr uint64_t kSplitPartFloats = 16ull * 16 * 65536;
struct SplitScratch {
  float* part = nullptr;
  uint32_t* counter = nullptr;
};
const SplitScratch& split_scratch() {
  static const SplitScratch scratch = [] {
    SplitScratch s;
    const uint64_t tiles = kSplitPartFloats / 32;
    if (hipMalloc(&s.part, kSplitPartFloats * sizeof(float)) != hipSuccess ||
        hipMalloc(&s.counter, tiles * sizeof(uint32_t)) != hipSuccess ||
        hipMemset(s.counter, 0, tiles * sizeof(uint32_t)) != hipSuccess ||
        hipDeviceSynchronize() != hipSuccess) {
      (void)hipGetLastError();
      s = {};
    }
    return s;
  }();
  return scratch;
}

/* The second matrix of a pair (w2, y2 from row m1), or none (m1 = m). */
struct Second {
  const void* w;
  float* y;
  uint32_t m1;
};

/* Tokens of one launch: x rows x_stride floats apart, y rows y_stride apart. */
struct Tokens {
  uint32_t n;
  uint64_t x_stride;
  uint32_t y_stride;
};

template <int kType, int S, int kCols>
void launch(const void* w, const void* w_up, float limit, const float* x, float* y, uint32_t m, uint32_t k,
            uint32_t rows_per_x, Second second, Tokens tokens) {
  constexpr uint32_t kRows = (kWave / Format<kType>::kTasks) * (kWaves / S);
  const uint32_t blocks = (m + kRows - 1) / kRows;
  const auto* w2 = static_cast<const uint8_t*>(second.w);
  if constexpr ((kType == 13 || kType == 12 || kType == 8 || kType == 14) && kCols > 1) {
    /* Plain Q5_K and Q4_K rows from 2 tokens, plain and paired Q6_K/Q8_0 rows from 4,
     * the fused SwiGLU from 6; below those the lane layout is as fast or
     * faster. */
    const bool split = w_up ? kCols >= 6 : kCols >= (kType == 13 || kType == 12 ? 2 : 4);
    if (split && second.m1 % kWave == 0 &&
        (rows_per_x == m || rows_per_x % kWave == 0) &&
        (uint64_t)m * S * kCols * (w_up ? 2 : 1) <= kSplitPartFloats) {
      const SplitScratch& scratch = split_scratch();
      if (scratch.part) {
        const auto* wu = static_cast<const uint8_t*>(w_up);
        const auto* wg = static_cast<const uint8_t*>(w);
        const uint32_t grid = (m + kWave - 1) / kWave * S;
        if (w_up)
          split_gemv_kernel<kType, S, true, kCols><<<grid, kWave * kWaves>>>(
              wg, wu, limit, x, tokens.x_stride, y, tokens.y_stride, m, k, rows_per_x, w2, second.y, second.m1,
              tokens.n, scratch.part, scratch.counter);
        else
          split_gemv_kernel<kType, S, false, kCols><<<grid, kWave * kWaves>>>(
              wg, nullptr, 0.0f, x, tokens.x_stride, y, tokens.y_stride, m, k, rows_per_x, w2, second.y, second.m1,
              tokens.n, scratch.part, scratch.counter);
        return;
      }
    }
  }
  if (w_up)
    gemv_kernel<kType, S, true, kCols><<<blocks, kWave * kWaves>>>(
        static_cast<const uint8_t*>(w), static_cast<const uint8_t*>(w_up), limit, x, tokens.x_stride, y,
        tokens.y_stride, m, k, rows_per_x, w2, second.y, second.m1, tokens.n);
  else
    gemv_kernel<kType, S, false, kCols><<<blocks, kWave * kWaves>>>(
        static_cast<const uint8_t*>(w), nullptr, 0.0f, x, tokens.x_stride, y, tokens.y_stride, m, k, rows_per_x, w2,
        second.y, second.m1, tokens.n);
}

template <int kType, int kCols>
bool dispatch_slices(const void* w, const void* w_up, float limit, const float* x, float* y, uint32_t m, uint32_t k,
                     uint32_t rows_per_x, int slices, Second second, Tokens tokens) {
  switch (slices) {
  case 1:
    launch<kType, 1, kCols>(w, w_up, limit, x, y, m, k, rows_per_x, second, tokens);
    return true;
  case 2:
    launch<kType, 2, kCols>(w, w_up, limit, x, y, m, k, rows_per_x, second, tokens);
    return true;
  case 4:
    launch<kType, 4, kCols>(w, w_up, limit, x, y, m, k, rows_per_x, second, tokens);
    return true;
  case 8:
    launch<kType, 8, kCols>(w, w_up, limit, x, y, m, k, rows_per_x, second, tokens);
    return true;
  default:
    return false;
  }
}

template <int kType>
int dispatch(const void* w, const void* w_up, float limit, const float* x, float* y, uint32_t m, uint32_t k,
             uint32_t rows_per_x, int slices, Second second, Tokens tokens) {
  bool ok = false;
  if (tokens.n == 1)
    ok = dispatch_slices<kType, 1>(w, w_up, limit, x, y, m, k, rows_per_x, slices, second, tokens);
  else if (tokens.n == 2)
    ok = dispatch_slices<kType, 2>(w, w_up, limit, x, y, m, k, rows_per_x, slices, second, tokens);
  else if (tokens.n <= 4)
    ok = dispatch_slices<kType, 4>(w, w_up, limit, x, y, m, k, rows_per_x, slices, second, tokens);
  else if (tokens.n <= 6)
    ok = dispatch_slices<kType, 6>(w, w_up, limit, x, y, m, k, rows_per_x, slices, second, tokens);
  else if (tokens.n <= 8)
    ok = dispatch_slices<kType, 8>(w, w_up, limit, x, y, m, k, rows_per_x, slices, second, tokens);
  else if (tokens.n <= 16)
    ok = dispatch_slices<kType, 16>(w, w_up, limit, x, y, m, k, rows_per_x, slices, second, tokens);
  if (!ok) return 1;
  const hipError_t err = hipGetLastError();
  if (err == hipSuccess) return 0;
  fprintf(stderr, "ds41: gemv launch failed: %s\n", hipGetErrorString(err));
  return 1;
}

}  // namespace

namespace {

int run(int type, const void* w, const void* w_up, float limit, const float* x, float* y, uint32_t m, uint32_t k,
        uint32_t rows_per_x, int slices, Second second, Tokens tokens) {
  if (k % 256 != 0 || m == 0 || tokens.n == 0 || tokens.n > DS41_V41_GEMV_MAX_TOKENS) return 1;
  if (rows_per_x == 0) rows_per_x = m;
  if (slices <= 0) slices = 8;
  if (!second.w) second.m1 = m;
  switch (type) {
  case 13:
    return dispatch<13>(w, w_up, limit, x, y, m, k, rows_per_x, slices, second, tokens);
  case 8:
    return dispatch<8>(w, w_up, limit, x, y, m, k, rows_per_x, slices, second, tokens);
  case 12:
    return dispatch<12>(w, w_up, limit, x, y, m, k, rows_per_x, slices, second, tokens);
  case 14:
    return dispatch<14>(w, w_up, limit, x, y, m, k, rows_per_x, slices, second, tokens);
  default:
    return 1;
  }
}

}  // namespace

extern "C" int ds41_v41_gemv(int type, const void* w, const float* x, float* y, uint32_t m, uint32_t k,
                             uint32_t rows_per_x, int slices, uint32_t n, uint64_t x_stride, uint32_t y_stride) {
  return run(type, w, nullptr, 0.0f, x, y, m, k, rows_per_x, slices, Second{nullptr, nullptr, 0},
             Tokens{n, x_stride, y_stride});
}

extern "C" int ds41_v41_gemv_pair(int type, const void* w1, float* y1, uint32_t m1, const void* w2, float* y2,
                                  uint32_t m2, const float* x, uint32_t k, int slices, uint32_t n) {
  if (!w2 || m2 == 0) return 1;
  return run(type, w1, nullptr, 0.0f, x, y1, m1 + m2, k, 0, slices, Second{w2, y2, m1}, Tokens{n, k, m1});
}

extern "C" int ds41_v41_gemv_swiglu(int type, const void* gate, const void* up, const float* x, float* y, uint32_t m,
                                    uint32_t k, float limit, int slices, uint32_t n) {
  if (!up) return 1;
  return run(type, gate, up, limit, x, y, m, k, 0, slices, Second{nullptr, nullptr, 0}, Tokens{n, k, m});
}

namespace {

template<int n>
__device__ __forceinline__ float UByteD(uint32_t v) {
  float r;
  if constexpr (n == 0)
    asm("v_cvt_f32_ubyte0 %0, %1" : "=v"(r) : "v"(v));
  else if constexpr (n == 1)
    asm("v_cvt_f32_ubyte1 %0, %1" : "=v"(r) : "v"(v));
  else if constexpr (n == 2)
    asm("v_cvt_f32_ubyte2 %0, %1" : "=v"(r) : "v"(v));
  else
    asm("v_cvt_f32_ubyte3 %0, %1" : "=v"(r) : "v"(v));
  return r;
}

template<int kType>
__device__ __forceinline__ void ScatterDecoded(const float (&wv)[32], int t,
                                               float* dst) {
  if constexpr (kType == 12 || kType == 13) {
    const int base = OffsetQ45K(t);
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      dst[base + i] = wv[i];
      dst[base + 32 + i] = wv[16 + i];
    }
  } else if constexpr (kType == 14) {
    const int base = OffsetQ6K32(t);
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      dst[base + i] = wv[i];
      dst[base + 32 + i] = wv[16 + i];
    }
  } else {
#pragma unroll
    for (int i = 0; i < 32; ++i)
      dst[32 * t + i] = wv[i];
  }
}

__device__ __forceinline__ float Group64Sum(float v, int wave, int g,
                                            float (&wsum)[8]) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    v += __shfl_xor(v, off, kWave);
  if ((threadIdx.x & 31) == 0)
    wsum[wave] = v;
  __syncthreads();
  const float total = wsum[2 * g] + wsum[2 * g + 1];
  __syncthreads();
  return total;
}

/* Quantizes each 256-value super-block of src (Q8_0, Q4_K, Q5_K or Q6_K) into
 * Q2_S64 (72 bytes): four FP16 scales for the 64-value groups and 64 bytes of
 * Lloyd-Max 2-bit symmetric codes {-3, -1, +1, +3}. */
template<int kType>
__global__ void __launch_bounds__(256)
    quantize_q2s64_kernel(const uint8_t* __restrict__ src,
                          uint8_t* __restrict__ dst, uint32_t m,
                          uint32_t blocks_per_row) {
  using F = Format<kType>;
  const uint32_t row = blockIdx.x;
  if (row >= m)
    return;
  const int tid = (int)threadIdx.x;
  const int wave = tid >> 5;
  const int g = tid >> 6;
  __shared__ float vals[256];
  __shared__ float wsum[8];
  __shared__ uint16_t hscales[4];
  __shared__ uint8_t q_arr[256];
  const uint8_t* srow = src + (uint64_t)row * blocks_per_row * F::kBlockBytes;
  uint8_t* drow =
      dst + (uint64_t)row * blocks_per_row * DS41_V41_Q2S64_BLOCK_BYTES;
  for (uint32_t blk = 0; blk < blocks_per_row; ++blk) {
    if (tid < 8) {
      float wv[32];
      F::Decode(srow + (uint64_t)blk * F::kBlockBytes, tid, wv);
      ScatterDecoded<kType>(wv, tid, vals);
    }
    __syncthreads();
    const float v = vals[tid];
    const float sum_sq = Group64Sum(v * v, wave, g, wsum);
    float s = sqrtf(sum_sq * (1.0f / 64.0f)) * 0.5f;
#pragma unroll
    for (int it = 0; it < 2; ++it) {
      const float c = s > 0.0f ? ((v >= 0.0f ? 1.0f : -1.0f) *
                                  (fabsf(v) >= 2.0f * s ? 3.0f : 1.0f))
                               : 0.0f;
      const float num = Group64Sum(v * c, wave, g, wsum);
      const float den = Group64Sum(c * c, wave, g, wsum);
      s = den > 0.0f ? fmaxf(num / den, 0.0f) : 0.0f;
    }
    const __half hs = __float2half(s);
    const float sf = __half2float(hs);
    if ((tid & 63) == 0) {
      uint16_t bits;
      __builtin_memcpy(&bits, &hs, 2);
      hscales[g] = bits;
    }
    uint8_t q = 2u;
    if (sf > 0.0f) {
      const float thr = 2.0f * sf;
      q = v < -thr ? 0u : (v < 0.0f ? 1u : (v < thr ? 2u : 3u));
    }
    q_arr[tid] = q;
    __syncthreads();
    uint8_t* out = drow + (uint64_t)blk * DS41_V41_Q2S64_BLOCK_BYTES;
    if (tid < 4)
      __builtin_memcpy(out + 2 * tid, &hscales[tid], 2);
    if (tid < 16) {
      const int base = tid * 16;
      uint32_t word = 0u;
#pragma unroll
      for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int b = 0; b < 4; ++b)
          word |= (uint32_t)q_arr[base + 4 * j + b] << (8 * b + 2 * j);
      __builtin_memcpy(out + 8 + 4 * tid, &word, 4);
    }
    __syncthreads();
  }
}

using Fp16x2 = _Float16 __attribute__((ext_vector_type(2)));

constexpr uint32_t kQ2S64MaxXFloats = DS41_V41_Q2S64_MAX_TOKENS * 16384u;

Fp16x2* q2s64_xh_scratch() {
  static Fp16x2* buf = []() -> Fp16x2* {
    Fp16x2* p = nullptr;
    if (hipMalloc(&p, (uint64_t)kQ2S64MaxXFloats * sizeof(_Float16)) !=
        hipSuccess) {
      (void)hipGetLastError();
      return nullptr;
    }
    return p;
  }();
  return buf;
}

__global__ void __launch_bounds__(256)
    f32_to_fp16x2_kernel(const float* __restrict__ src,
                         Fp16x2* __restrict__ dst, uint32_t pairs) {
  const uint32_t i = blockIdx.x * 256u + threadIdx.x;
  if (i >= pairs)
    return;
  const float2 f = reinterpret_cast<const float2*>(src)[i];
  dst[i] = {(_Float16)f.x, (_Float16)f.y};
}

__device__ __forceinline__ void DecodeQ2S64HalfChunk(uint32_t word, int shift,
                                                     Fp16x2* dst) {
  const uint32_t v = (word >> shift) & 0x03030303u;
  const Fp16x2 a = {(_Float16)UByteD<0>(v), (_Float16)UByteD<1>(v)};
  const Fp16x2 b = {(_Float16)UByteD<2>(v), (_Float16)UByteD<3>(v)};
  const Fp16x2 two = {(_Float16)2.0f, (_Float16)2.0f};
  const Fp16x2 m3 = {(_Float16)-3.0f, (_Float16)-3.0f};
  dst[0] = a * two + m3;
  dst[1] = b * two + m3;
}

__device__ __forceinline__ void DecodeQ2S64Half(uint2 qw, Fp16x2 (&wh)[16]) {
  DecodeQ2S64HalfChunk(qw.x, 0, wh + 0);
  DecodeQ2S64HalfChunk(qw.x, 2, wh + 2);
  DecodeQ2S64HalfChunk(qw.x, 4, wh + 4);
  DecodeQ2S64HalfChunk(qw.x, 6, wh + 6);
  DecodeQ2S64HalfChunk(qw.y, 0, wh + 8);
  DecodeQ2S64HalfChunk(qw.y, 2, wh + 10);
  DecodeQ2S64HalfChunk(qw.y, 4, wh + 12);
  DecodeQ2S64HalfChunk(qw.y, 6, wh + 14);
}

__device__ __forceinline__ float DotQ2S64Half(const Fp16x2 (&wh)[16],
                                              const Fp16x2* xh) {
  float d0 = 0.0f, d1 = 0.0f;
#pragma unroll
  for (int i = 0; i < 16; i += 2) {
    d0 = __builtin_amdgcn_fdot2(wh[i], xh[i], d0, false);
    d1 = __builtin_amdgcn_fdot2(wh[i + 1], xh[i + 1], d1, false);
  }
  return d0 + d1;
}

/* Q2_S64 screening GEMV: 8 waves x 4 rows = 32 rows per workgroup, S = 1;
 * uses v_dot2_f32_f16 with exact {-3,-1,+1,+3} FP16 weights and pre-converted
 * FP16x2 activations, bit-identical per token across n = 1..16. */
template<int kCols>
__global__ void __launch_bounds__(kWave* kWaves)
    gemv_q2s64_kernel(const uint8_t* __restrict__ w,
                      const Fp16x2* __restrict__ xh, float* __restrict__ y,
                      uint32_t m, uint32_t k, uint32_t n) {
  constexpr int kTasks = 8;
  constexpr int kRowsPerWave = kWave / kTasks;
  const int lane = (int)(threadIdx.x % kWave);
  const int wave = (int)(threadIdx.x / kWave);
  const int t = lane % kTasks;
  const uint32_t row =
      (blockIdx.x * kWaves + wave) * kRowsPerWave + (lane / kTasks);
  const uint32_t blocks_per_row = k / 256;
  const uint32_t kh = k / 2;
  float acc[kCols] = {};
  if (row < m) {
    const uint8_t* base =
        w + (uint64_t)row * blocks_per_row * DS41_V41_Q2S64_BLOCK_BYTES;
    const uint32_t scale_off = 2u * (uint32_t)(t >> 1);
    const uint32_t q_off = 8u + 8u * (uint32_t)t;
    for (uint32_t blk = 0; blk < blocks_per_row; ++blk) {
      const uint8_t* bp = base + (uint64_t)blk * DS41_V41_Q2S64_BLOCK_BYTES;
      uint16_t hs;
      __builtin_memcpy(&hs, bp + scale_off, 2);
      const float s = HalfBits(hs);
      const uint2 qw = *reinterpret_cast<const uint2*>(bp + q_off);
      Fp16x2 wh[16];
      DecodeQ2S64Half(qw, wh);
      const Fp16x2* xb = xh + (uint64_t)blk * 128 + 16 * t;
#pragma unroll
      for (int c = 0; c < kCols; ++c) {
        if (kCols == 1 || c < (int)n)
          acc[c] = __builtin_fmaf(s, DotQ2S64Half(wh, xb + (uint64_t)c * kh),
                                  acc[c]);
      }
    }
  }
#pragma unroll
  for (int c = 0; c < kCols; ++c) {
#pragma unroll
    for (int off = kTasks / 2; off > 0; off >>= 1)
      acc[c] += __shfl_xor(acc[c], off, kWave);
  }
  if (t != 0 || row >= m)
    return;
#pragma unroll
  for (int c = 0; c < kCols; ++c) {
    if (kCols > 1 && c >= (int)n)
      break;
    y[(uint64_t)c * m + row] = acc[c];
  }
}

/* Task-major Q2_S64 GEMV for n >= 2: wave t owns task t (32 values) of 32 rows
 * (one per lane), so FP16x2 x loads are wave-uniform scalar loads while the
 * vector unit loads only Q2_S64 weights (software-prefetched one block ahead)
 * and decodes each 32-weight block once into 16 FP16x2 pairs for dual-accum
 * v_dot2_f32_f16 across all n tokens. The 8 waves reduce in LDS in the exact
 * xor-tree order of gemv_q2s64_kernel<1>, matching n = 1 bit for bit. */
template<int kCols, bool kExact>
__global__ void __launch_bounds__(kWave* kWaves)
    task_gemv_q2s64_kernel(const uint8_t* __restrict__ w,
                           const Fp16x2* __restrict__ xh, float* __restrict__ y,
                           uint32_t m, uint32_t k, uint32_t n) {
  __shared__ float red[kWaves / 2][kCols][kWave];
  const int lane = (int)(threadIdx.x % kWave);
  const int t = __builtin_amdgcn_readfirstlane((int)(threadIdx.x / kWave));
  const uint32_t row0 = blockIdx.x * kWave;
  const uint32_t row = min(row0 + (uint32_t)lane, m - 1);
  const uint32_t blocks_per_row = k / 256;
  const uint32_t kh = k / 2;
  const uint8_t* base =
      w + (uint64_t)row * blocks_per_row * DS41_V41_Q2S64_BLOCK_BYTES;
  const uint32_t scale_off = 2u * (uint32_t)(t >> 1);
  const uint32_t q_off = 8u + 8u * (uint32_t)t;
  float acc[kCols] = {};
  uint16_t next_hs = 0;
  uint2 next_qw{};
  if (blocks_per_row > 0) {
    __builtin_memcpy(&next_hs, base + scale_off, 2);
    next_qw = *reinterpret_cast<const uint2*>(base + q_off);
  }
  for (uint32_t blk = 0; blk < blocks_per_row; ++blk) {
    const uint16_t hs = next_hs;
    const uint2 qw = next_qw;
    if (blk + 1 < blocks_per_row) {
      const uint8_t* nb =
          base + (uint64_t)(blk + 1) * DS41_V41_Q2S64_BLOCK_BYTES;
      __builtin_memcpy(&next_hs, nb + scale_off, 2);
      next_qw = *reinterpret_cast<const uint2*>(nb + q_off);
    }
    const float s = HalfBits(hs);
    Fp16x2 wh[16];
    DecodeQ2S64Half(qw, wh);
    const Fp16x2* xb = xh + (uint64_t)blk * 128 + 16 * t;
#pragma unroll
    for (int c = 0; c < kCols; ++c) {
      if (kExact || c < (int)n)
        acc[c] =
            __builtin_fmaf(s, DotQ2S64Half(wh, xb + (uint64_t)c * kh), acc[c]);
    }
  }
#pragma unroll
  for (int off = kWaves / 2; off > 0; off >>= 1) {
    if (t >= off && t < 2 * off)
#pragma unroll
      for (int c = 0; c < kCols; ++c)
        red[t - off][c][lane] = acc[c];
    __syncthreads();
    if (t < off)
#pragma unroll
      for (int c = 0; c < kCols; ++c)
        acc[c] += red[t][c][lane];
    __syncthreads();
  }
  if (t != 0 || row0 + (uint32_t)lane >= m)
    return;
#pragma unroll
  for (int c = 0; c < kCols; ++c) {
    if (c >= (int)n)
      break;
    y[(uint64_t)c * m + row0 + (uint32_t)lane] = acc[c];
  }
}

/* Quantizes rank-256 FP16 Markov W2 into symmetric 4-bit codes per 32 values:
 * 128 B/row qs (32 u32 words, 8 nibbles in [1..15] representing [-7..+7] per
 * lane) + 16 B/row FP16 scales (8 per row, 1 per 4 lanes). Total 18.6 MB for
 * 129,280 rows, fitting inside gfx1151's 32 MiB MALL across sequential slots.
 */
__global__ void __launch_bounds__(256)
    quantize_markov_q4_kernel(const _Float16* __restrict__ w2,
                              uint32_t* __restrict__ dst_qs,
                              uint16_t* __restrict__ dst_scales,
                              uint32_t vocab) {
  const uint32_t lane = threadIdx.x & 31u;
  const uint32_t wave = threadIdx.x >> 5;
  const uint32_t row = blockIdx.x * 8u + wave;
  if (row >= vocab)
    return;
  const uint4 packed =
      reinterpret_cast<const uint4*>(w2 + (uint64_t)row * 256)[lane];
  const _Float16* h = reinterpret_cast<const _Float16*>(&packed);
  float v[8];
  float amax = 0.0f;
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    v[e] = (float)h[e];
    amax = fmaxf(amax, fabsf(v[e]));
  }
  amax = fmaxf(amax, __shfl_xor(amax, 1, kWave));
  amax = fmaxf(amax, __shfl_xor(amax, 2, kWave));
  const __half hs = __float2half(amax * (1.0f / 7.0f));
  const float sf = __half2float(hs);
  const float inv = sf > 0.0f ? 1.0f / sf : 0.0f;
  if ((lane & 3u) == 0) {
    uint16_t bits;
    __builtin_memcpy(&bits, &hs, 2);
    dst_scales[(uint64_t)row * 8 + (lane >> 2)] = bits;
  }
  uint32_t q[8];
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    int qi = sf > 0.0f ? (int)rintf(v[e] * inv) : 0;
    qi = qi < -7 ? -7 : (qi > 7 ? 7 : qi);
    q[e] = (uint32_t)(qi + 8);
  }
  uint32_t word = 0u;
#pragma unroll
  for (int b = 0; b < 4; ++b)
    word |= (q[b] | (q[b + 4] << 4)) << (8 * b);
  dst_qs[(uint64_t)row * 32 + lane] = word;
}

constexpr uint32_t kMarkovQ4Blocks = 128;
constexpr uint32_t kMarkovQ4PerBlock = 2;
constexpr uint32_t kMarkovRerankTop = 128;

struct MarkovRerankScratch {
  float block_score[kMarkovQ4Blocks * kMarkovQ4PerBlock];
  float block_bias[kMarkovQ4Blocks * kMarkovQ4PerBlock];
  int32_t block_id[kMarkovQ4Blocks * kMarkovQ4PerBlock];
  int32_t cand_ids[kMarkovRerankTop];
  float cand_bias[kMarkovRerankTop];
  float exact_score[kMarkovRerankTop];
};
static_assert(sizeof(MarkovRerankScratch) <=
                  DS41_V41_MARKOV_RERANK_SCRATCH_BYTES,
              "MarkovRerankScratch exceeds scratch budget");

__device__ __forceinline__ bool markov_better(float va, int32_t ia, float vb,
                                              int32_t ib) {
  const bool a_ok = ia >= 0 && isfinite(va);
  const bool b_ok = ib >= 0 && isfinite(vb);
  if (!a_ok)
    return false;
  if (!b_ok)
    return true;
  return va > vb || (va == vb && ia < ib);
}

/* Single pass over all vocab rows: 128 blocks (1,024 waves) strided in
 * coalesced 8-row chunks with 4x unroll and fast-path top-2 tracking per wave,
 * emitting 128 x 2 = 256 (score, bias, id) candidates for exact top-128 rerank.
 */
__global__ void __launch_bounds__(256)
    markov_q4_screen_kernel(const uint32_t* __restrict__ w2_qs,
                            const uint16_t* __restrict__ w2_scales,
                            const float* __restrict__ x,
                            const float* __restrict__ approx_logits,
                            uint32_t vocab, MarkovRerankScratch* sc) {
  const uint32_t lane = threadIdx.x & 31u;
  const uint32_t wave = threadIdx.x >> 5;
  const uint32_t b = blockIdx.x;
  float xv[8];
  {
    const float4* xp = reinterpret_cast<const float4*>(x) + 2 * lane;
    const float4 a = xp[0], bx = xp[1];
    xv[0] = a.x;
    xv[1] = a.y;
    xv[2] = a.z;
    xv[3] = a.w;
    xv[4] = bx.x;
    xv[5] = bx.y;
    xv[6] = bx.z;
    xv[7] = bx.w;
  }
  const float xsum8 = -8.0f * (((xv[0] + xv[1]) + (xv[2] + xv[3])) +
                               ((xv[4] + xv[5]) + (xv[6] + xv[7])));
  const uint32_t sub = lane >> 2;
  float s0 = -INFINITY, b0 = 0.0f;
  int32_t i0 = -1;
  float s1 = -INFINITY, b1 = 0.0f;
  int32_t i1 = -1;
  constexpr uint32_t kUnroll = 4;
  constexpr uint32_t kChunkStride = kMarkovQ4Blocks * 8u;
  const uint32_t r_start = b * 8u + wave;
  for (uint32_t r0 = r_start; r0 < vocab; r0 += kChunkStride * kUnroll) {
    uint32_t qs[kUnroll];
    uint16_t scs[kUnroll];
    float lg[kUnroll];
#pragma unroll
    for (uint32_t u = 0; u < kUnroll; ++u) {
      const uint32_t r = r0 + u * kChunkStride;
      if (r < vocab) {
        qs[u] = w2_qs[(uint64_t)r * 32 + lane];
        scs[u] = w2_scales[(uint64_t)r * 8 + sub];
        lg[u] = approx_logits[r];
      } else {
        qs[u] = 0u;
        scs[u] = 0u;
        lg[u] = -INFINITY;
      }
    }
#pragma unroll
    for (uint32_t u = 0; u < kUnroll; ++u) {
      const uint32_t r = r0 + u * kChunkStride;
      const uint32_t q = qs[u];
      const float scale = HalfBits(scs[u]);
      const uint32_t lo = q & 0x0F0F0F0Fu;
      const uint32_t hi = (q >> 4) & 0x0F0F0F0Fu;
      float qdot = xsum8;
      qdot = __builtin_fmaf(UByteD<0>(lo), xv[0], qdot);
      qdot = __builtin_fmaf(UByteD<1>(lo), xv[1], qdot);
      qdot = __builtin_fmaf(UByteD<2>(lo), xv[2], qdot);
      qdot = __builtin_fmaf(UByteD<3>(lo), xv[3], qdot);
      qdot = __builtin_fmaf(UByteD<0>(hi), xv[4], qdot);
      qdot = __builtin_fmaf(UByteD<1>(hi), xv[5], qdot);
      qdot = __builtin_fmaf(UByteD<2>(hi), xv[6], qdot);
      qdot = __builtin_fmaf(UByteD<3>(hi), xv[7], qdot);
      float bias = scale * qdot;
#pragma unroll
      for (int off = 16; off > 0; off >>= 1)
        bias += __shfl_xor(bias, off, kWave);
      if (r >= vocab)
        continue;
      const float score = lg[u] + bias;
      if (score >= s1 && isfinite(score)) {
        const int32_t id = (int32_t)r;
        if (markov_better(score, id, s0, i0)) {
          s1 = s0;
          b1 = b0;
          i1 = i0;
          s0 = score;
          b0 = bias;
          i0 = id;
        } else if (markov_better(score, id, s1, i1)) {
          s1 = score;
          b1 = bias;
          i1 = id;
        }
      }
    }
  }
  __shared__ float ss[8][2];
  __shared__ float sb[8][2];
  __shared__ int32_t si[8][2];
  if (lane == 0) {
    ss[wave][0] = s0;
    sb[wave][0] = b0;
    si[wave][0] = i0;
    ss[wave][1] = s1;
    sb[wave][1] = b1;
    si[wave][1] = i1;
  }
  __syncthreads();
  if (threadIdx.x != 0)
    return;
  uint32_t head[8] = {};
  for (uint32_t j = 0; j < kMarkovQ4PerBlock; ++j) {
    int win = -1;
    for (int w = 0; w < 8; ++w) {
      if (head[w] >= 2 || si[w][head[w]] < 0)
        continue;
      if (win < 0 || markov_better(ss[w][head[w]], si[w][head[w]],
                                   ss[win][head[win]], si[win][head[win]]))
        win = w;
    }
    const uint32_t dst = b * kMarkovQ4PerBlock + j;
    if (win < 0) {
      sc->block_score[dst] = -INFINITY;
      sc->block_bias[dst] = 0.0f;
      sc->block_id[dst] = -1;
    } else {
      sc->block_score[dst] = ss[win][head[win]];
      sc->block_bias[dst] = sb[win][head[win]];
      sc->block_id[dst] = si[win][head[win]];
      ++head[win];
    }
  }
}

/* Merges the 256 block candidates into the top-128 for exact head reranking. */
__global__ void __launch_bounds__(256)
    markov_q4_merge_kernel(MarkovRerankScratch* sc) {
  const uint32_t tid = threadIdx.x;
  __shared__ float ss[256];
  __shared__ float sb[256];
  __shared__ int32_t si[256];
  ss[tid] = sc->block_score[tid];
  sb[tid] = sc->block_bias[tid];
  si[tid] = sc->block_id[tid];
  __syncthreads();
  for (uint32_t k = 2; k <= 256; k <<= 1) {
    for (uint32_t j = k >> 1; j > 0; j >>= 1) {
      const uint32_t ixj = tid ^ j;
      if (ixj > tid) {
        const bool desc = (tid & k) == 0;
        const bool a_better = markov_better(ss[tid], si[tid], ss[ixj], si[ixj]);
        if (desc ? !a_better : a_better) {
          const float ts = ss[tid], tb = sb[tid];
          const int32_t ti = si[tid];
          ss[tid] = ss[ixj];
          sb[tid] = sb[ixj];
          si[tid] = si[ixj];
          ss[ixj] = ts;
          sb[ixj] = tb;
          si[ixj] = ti;
        }
      }
      __syncthreads();
    }
  }
  if (tid < kMarkovRerankTop) {
    sc->cand_ids[tid] = si[tid];
    sc->cand_bias[tid] = sb[tid];
    sc->exact_score[tid] = ss[tid];
  }
}

/* Exact target-head dot product over the 128 screened candidates + bias. */
template<int kType>
__global__ void __launch_bounds__(256)
    exact_rerank_markov_kernel(const uint8_t* __restrict__ exact_w,
                               const float* __restrict__ x, uint32_t k,
                               MarkovRerankScratch* sc) {
  using F = Format<kType>;
  const uint32_t r = blockIdx.x;
  const int32_t tok = sc->cand_ids[r];
  if (tok < 0) {
    if (threadIdx.x == 0)
      sc->exact_score[r] = -INFINITY;
    return;
  }
  const uint32_t blocks_per_row = k / 256;
  const uint32_t grp = threadIdx.x / 8;
  const int t = (int)(threadIdx.x % 8);
  const uint8_t* row_base =
      exact_w + (uint64_t)tok * blocks_per_row * F::kBlockBytes;
  float acc = 0.0f;
  for (uint32_t blk = grp; blk < blocks_per_row; blk += 32) {
    float wv[32];
    F::Decode(row_base + (uint64_t)blk * F::kBlockBytes, t, wv);
    acc = F::Dot(acc, wv, t, x + (uint64_t)blk * 256);
  }
#pragma unroll
  for (int off = 4; off > 0; off >>= 1)
    acc += __shfl_xor(acc, off, kWave);
  __shared__ float partial[32];
  if (t == 0)
    partial[grp] = acc;
  __syncthreads();
  if (threadIdx.x == 0) {
    float sum = 0.0f;
    const uint32_t used = blocks_per_row < 32 ? blocks_per_row : 32;
    for (uint32_t g = 0; g < used; ++g)
      sum += partial[g];
    sc->exact_score[r] = sum + sc->cand_bias[r];
  }
}

/* Sorts the 128 exact biased scores descending and writes either greedy argmax
 * or a sampled top-8 ds41_v41_proposal + drawn token. */
__global__ void __launch_bounds__(kMarkovRerankTop)
    markov_q4_finalize_kernel(MarkovRerankScratch* sc, int stochastic,
                              const uint32_t* draw, float temperature,
                              ds41_v41_proposal* proposal, int32_t* tokens,
                              uint32_t token_index) {
  const uint32_t tid = threadIdx.x;
  __shared__ float sv[kMarkovRerankTop];
  __shared__ int32_t si[kMarkovRerankTop];
  sv[tid] = sc->exact_score[tid];
  si[tid] = sc->cand_ids[tid];
  __syncthreads();
  for (uint32_t k = 2; k <= kMarkovRerankTop; k <<= 1) {
    for (uint32_t j = k >> 1; j > 0; j >>= 1) {
      const uint32_t ixj = tid ^ j;
      if (ixj > tid) {
        const bool desc = (tid & k) == 0;
        const bool a_better = markov_better(sv[tid], si[tid], sv[ixj], si[ixj]);
        if (desc ? !a_better : a_better) {
          const float tv = sv[tid];
          const int32_t ti = si[tid];
          sv[tid] = sv[ixj];
          si[tid] = si[ixj];
          sv[ixj] = tv;
          si[ixj] = ti;
        }
      }
      __syncthreads();
    }
  }
  if (tid != 0)
    return;
  if (!stochastic) {
    tokens[token_index] = (si[0] >= 0 && isfinite(sv[0])) ? si[0] : -1;
    return;
  }
  constexpr uint32_t K = 8;
  ds41_v41_proposal q{};
  if (si[K - 1] < 0 || !isfinite(sv[K - 1])) {
    q.token = -1;
    *proposal = q;
    tokens[token_index] = -1;
    return;
  }
  float e[K], total = 0.0f;
  for (uint32_t j = 0; j < K; ++j)
    total += e[j] = expf((sv[j] - sv[0]) / temperature);
  constexpr uint32_t kUnits = 1u << 24;
  if (e[0] / total >= 0.7f) {
    q.size = 1;
    q.ids[0] = q.token = si[0];
    q.masses[0] = kUnits;
  } else {
    uint32_t sum = 0;
    for (uint32_t j = 0; j < K; ++j) {
      q.ids[j] = si[j];
      sum += q.masses[j] = (uint32_t)floorf(e[j] / total * (float)kUnits);
    }
    q.masses[0] += kUnits - sum;
    q.size = K;
    uint32_t cumulative = 0;
    q.token = si[K - 1];
    const uint32_t d = *draw;
    for (uint32_t j = 0; j < K; ++j) {
      cumulative += q.masses[j];
      if (d < cumulative) {
        q.token = si[j];
        break;
      }
    }
  }
  *proposal = q;
  tokens[token_index] = q.token;
}

}  // namespace

extern "C" int ds41_v41_quantize_q2s64(int src_type, const void* src, void* dst,
                                       uint32_t m, uint32_t k) {
  if (!src || !dst || m == 0 || k == 0 || k % 256 != 0)
    return 1;
  const uint32_t blocks_per_row = k / 256;
  const auto* s = static_cast<const uint8_t*>(src);
  auto* d = static_cast<uint8_t*>(dst);
  switch (src_type) {
    case 8:
      quantize_q2s64_kernel<8><<<m, 256>>>(s, d, m, blocks_per_row);
      break;
    case 12:
      quantize_q2s64_kernel<12><<<m, 256>>>(s, d, m, blocks_per_row);
      break;
    case 13:
      quantize_q2s64_kernel<13><<<m, 256>>>(s, d, m, blocks_per_row);
      break;
    case 14:
      quantize_q2s64_kernel<14><<<m, 256>>>(s, d, m, blocks_per_row);
      break;
    default:
      return 1;
  }
  const hipError_t err = hipGetLastError();
  if (err == hipSuccess)
    return 0;
  fprintf(stderr, "ds41: quantize q2s64 launch failed: %s\n",
          hipGetErrorString(err));
  return 1;
}

extern "C" int ds41_v41_gemv_q2s64(const void* w, const float* x, float* y,
                                   uint32_t m, uint32_t k, uint32_t n) {
  if (!w || !x || !y || m == 0 || k == 0 || k % 256 != 0 || n == 0 ||
      n > DS41_V41_Q2S64_MAX_TOKENS || (uint64_t)n * k > kQ2S64MaxXFloats)
    return 1;
  Fp16x2* xh = q2s64_xh_scratch();
  if (!xh)
    return 1;
  const uint32_t pairs = (n * k) / 2u;
  f32_to_fp16x2_kernel<<<(pairs + 255u) / 256u, 256>>>(x, xh, pairs);
  const uint32_t blocks = (m + 31) / 32;
  const auto* wp = static_cast<const uint8_t*>(w);
  if (n == 1)
    gemv_q2s64_kernel<1><<<blocks, kWave * kWaves>>>(wp, xh, y, m, k, n);
  else if (n == 2)
    task_gemv_q2s64_kernel<2, true>
        <<<blocks, kWave * kWaves>>>(wp, xh, y, m, k, n);
  else if (n <= 4)
    task_gemv_q2s64_kernel<4, false>
        <<<blocks, kWave * kWaves>>>(wp, xh, y, m, k, n);
  else if (n == 5)
    task_gemv_q2s64_kernel<5, true>
        <<<blocks, kWave * kWaves>>>(wp, xh, y, m, k, n);
  else if (n <= 8)
    task_gemv_q2s64_kernel<8, false>
        <<<blocks, kWave * kWaves>>>(wp, xh, y, m, k, n);
  else if (n == 10)
    task_gemv_q2s64_kernel<10, true>
        <<<blocks, kWave * kWaves>>>(wp, xh, y, m, k, n);
  else
    task_gemv_q2s64_kernel<16, false>
        <<<blocks, kWave * kWaves>>>(wp, xh, y, m, k, n);
  const hipError_t err = hipGetLastError();
  if (err == hipSuccess)
    return 0;
  fprintf(stderr, "ds41: gemv q2s64 launch failed: %s\n",
          hipGetErrorString(err));
  return 1;
}

extern "C" int ds41_v41_quantize_markov_q4(const uint16_t* w2_f16,
                                           uint32_t* dst_qs,
                                           uint16_t* dst_scales,
                                           uint32_t vocab) {
  if (!w2_f16 || !dst_qs || !dst_scales || vocab == 0)
    return 1;
  quantize_markov_q4_kernel<<<(vocab + 7) / 8, 256>>>(
      reinterpret_cast<const _Float16*>(w2_f16), dst_qs, dst_scales, vocab);
  const hipError_t err = hipGetLastError();
  if (err == hipSuccess)
    return 0;
  fprintf(stderr, "ds41: quantize markov q4 launch failed: %s\n",
          hipGetErrorString(err));
  return 1;
}

extern "C" int ds41_v41_markov_q4_rerank(
    const uint32_t* w2_qs, const uint16_t* w2_scales, const float* markov_x,
    const float* approx_logits, int exact_type, const void* exact_w,
    const float* head_x, uint32_t vocab, uint32_t k, int stochastic,
    const uint32_t* draw, float temperature, void* scratch,
    ds41_v41_proposal* proposal, int32_t* tokens, uint32_t token_index) {
  if (!w2_qs || !w2_scales || !markov_x || !approx_logits || !scratch ||
      !tokens || vocab == 0)
    return 1;
  if (exact_w && (!head_x || k == 0 || k % 256 != 0))
    return 1;
  if (stochastic && (!draw || !proposal || !(temperature > 0.0f)))
    return 1;
  auto* sc = static_cast<MarkovRerankScratch*>(scratch);
  markov_q4_screen_kernel<<<kMarkovQ4Blocks, 256>>>(w2_qs, w2_scales, markov_x,
                                                    approx_logits, vocab, sc);
  markov_q4_merge_kernel<<<1, 256>>>(sc);
  if (exact_w) {
    const auto* ew = static_cast<const uint8_t*>(exact_w);
    switch (exact_type) {
      case 8:
        exact_rerank_markov_kernel<8>
            <<<kMarkovRerankTop, 256>>>(ew, head_x, k, sc);
        break;
      case 12:
        exact_rerank_markov_kernel<12>
            <<<kMarkovRerankTop, 256>>>(ew, head_x, k, sc);
        break;
      case 13:
        exact_rerank_markov_kernel<13>
            <<<kMarkovRerankTop, 256>>>(ew, head_x, k, sc);
        break;
      case 14:
        exact_rerank_markov_kernel<14>
            <<<kMarkovRerankTop, 256>>>(ew, head_x, k, sc);
        break;
      default:
        return 1;
    }
  }
  markov_q4_finalize_kernel<<<1, kMarkovRerankTop>>>(
      sc, stochastic, draw, temperature, proposal, tokens, token_index);
  const hipError_t err = hipGetLastError();
  if (err == hipSuccess)
    return 0;
  fprintf(stderr, "ds41: markov q4 rerank launch failed: %s\n",
          hipGetErrorString(err));
  return 1;
}
