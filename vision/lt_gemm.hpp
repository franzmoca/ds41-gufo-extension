// Cached hipBLASLt BF16 GEMM plans for the vision encoder: the heuristic path of
// Gufo's Qwen HipblasLtGemm (src/families/qwen/kernels/hipblaslt_gemm.hip), with
// the same rank selection and no plan database, tuning or telemetry.
#ifndef DEEPSEEK41FLASH_VISION_LT_GEMM_HPP_
#define DEEPSEEK41FLASH_VISION_LT_GEMM_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <memory>

namespace gufo::models::deepseek_v41_flash::vision {

class LtGemm {
public:
  LtGemm();
  ~LtGemm();
  LtGemm(const LtGemm&) = delete;
  LtGemm& operator=(const LtGemm&) = delete;

  /// Computes Y[B, M] = X_bf16[B, K] * A_bf16[M, K]^T. False when hipBLASLt has
  /// no supported plan; the caller falls back to hipBLAS.
  [[nodiscard]] bool RunBf16(const void* a_bf16, const void* x_bf16, float* y,
                             std::size_t batch_size, std::size_t m,
                             std::size_t k, hipStream_t stream);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::models::deepseek_v41_flash::vision

#endif  // DEEPSEEK41FLASH_VISION_LT_GEMM_HPP_
