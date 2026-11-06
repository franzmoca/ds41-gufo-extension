#include "vision/lt_gemm.hpp"

#include <hipblaslt/hipblaslt.h>

#include <algorithm>
#include <ranges>
#include <span>
#include <vector>

namespace gufo::models::deepseek_v41_flash::vision {
namespace {

constexpr std::size_t kMaxAlgorithms = 16;
constexpr std::size_t kWideFfnThreshold = 3;
constexpr std::size_t kLargeBatch = 256;
constexpr std::size_t kLargeWideRank = 4;
constexpr std::size_t kTallRank = 5;

struct Plan {
  std::size_t batch_size{0};
  std::size_t m{0};
  std::size_t k{0};
  hipblasLtMatmulDesc_t operation{nullptr};
  hipblasLtMatrixLayout_t a_layout{nullptr};
  hipblasLtMatrixLayout_t x_layout{nullptr};
  hipblasLtMatrixLayout_t y_layout{nullptr};
  hipblasLtMatmulAlgo_t algorithm{};

  Plan() = default;
  Plan(const Plan&) = delete;
  Plan& operator=(const Plan&) = delete;
  ~Plan() {
    if (a_layout != nullptr)
      (void)hipblasLtMatrixLayoutDestroy(a_layout);
    if (x_layout != nullptr)
      (void)hipblasLtMatrixLayoutDestroy(x_layout);
    if (y_layout != nullptr)
      (void)hipblasLtMatrixLayoutDestroy(y_layout);
    if (operation != nullptr)
      (void)hipblasLtMatmulDescDestroy(operation);
  }
};

[[nodiscard]] std::size_t PreferredRank(std::size_t batch_size, std::size_t m,
                                        std::size_t k,
                                        std::size_t result_count) noexcept {
  if (batch_size >= kLargeBatch) {
    if (m <= 1024)
      return kLargeWideRank;
    return k >= kWideFfnThreshold * m ? kTallRank : kLargeWideRank;
  }
  const std::size_t tall_m_threshold = batch_size >= 128 ? k : 2 * k;
  if (m > tall_m_threshold || k >= kWideFfnThreshold * m)
    return kTallRank;
  return result_count - 1;
}

// The encoder runs with no workspace, as its Qwen source does with no database.
[[nodiscard]] bool Usable(const hipblasLtMatmulHeuristicResult_t& r) noexcept {
  return r.state == HIPBLAS_STATUS_SUCCESS && r.workspaceSize == 0;
}

[[nodiscard]] std::unique_ptr<Plan> CreatePlan(hipblasLtHandle_t handle,
                                               std::size_t batch_size,
                                               std::size_t m, std::size_t k) {
  auto plan = std::make_unique<Plan>();
  plan->batch_size = batch_size;
  plan->m = m;
  plan->k = k;
  if (hipblasLtMatmulDescCreate(&plan->operation, HIPBLAS_COMPUTE_32F,
                                HIP_R_32F) != HIPBLAS_STATUS_SUCCESS)
    return nullptr;
  const hipblasOperation_t transpose = HIPBLAS_OP_T;
  if (hipblasLtMatmulDescSetAttribute(
          plan->operation, HIPBLASLT_MATMUL_DESC_TRANSA, &transpose,
          sizeof(transpose)) != HIPBLAS_STATUS_SUCCESS)
    return nullptr;
  if (hipblasLtMatrixLayoutCreate(&plan->a_layout, HIP_R_16BF, k, m, k) !=
          HIPBLAS_STATUS_SUCCESS ||
      hipblasLtMatrixLayoutCreate(&plan->x_layout, HIP_R_16BF, k, batch_size,
                                  k) != HIPBLAS_STATUS_SUCCESS ||
      hipblasLtMatrixLayoutCreate(&plan->y_layout, HIP_R_32F, m, batch_size,
                                  m) != HIPBLAS_STATUS_SUCCESS)
    return nullptr;

  hipblasLtMatmulPreference_t preference = nullptr;
  if (hipblasLtMatmulPreferenceCreate(&preference) != HIPBLAS_STATUS_SUCCESS)
    return nullptr;
  std::size_t workspace_capacity = 0;
  const auto preference_status = hipblasLtMatmulPreferenceSetAttribute(
      preference, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
      &workspace_capacity, sizeof(workspace_capacity));
  std::vector<hipblasLtMatmulHeuristicResult_t> results(kMaxAlgorithms);
  int result_count = 0;
  const auto status =
      preference_status == HIPBLAS_STATUS_SUCCESS
          ? hipblasLtMatmulAlgoGetHeuristic(
                handle, plan->operation, plan->a_layout, plan->x_layout,
                plan->y_layout, plan->y_layout, preference,
                static_cast<int>(results.size()), results.data(), &result_count)
          : preference_status;
  (void)hipblasLtMatmulPreferenceDestroy(preference);
  if (status != HIPBLAS_STATUS_SUCCESS || result_count <= 0)
    return nullptr;

  const auto found =
      std::span(results.data(), static_cast<std::size_t>(result_count));
  const auto rank = PreferredRank(batch_size, m, k, found.size());
  if (rank < found.size() && Usable(found[rank])) {
    plan->algorithm = found[rank].algo;
    return plan;
  }
  const auto fallback = std::ranges::find_if(found, Usable);
  if (fallback == found.end())
    return nullptr;
  plan->algorithm = fallback->algo;
  return plan;
}

}  // namespace

struct LtGemm::Impl {
  hipblasLtHandle_t handle{nullptr};
  std::vector<std::unique_ptr<Plan>> plans;

  Impl() {
    if (hipblasLtCreate(&handle) != HIPBLAS_STATUS_SUCCESS)
      handle = nullptr;
  }
  ~Impl() {
    plans.clear();
    if (handle != nullptr)
      (void)hipblasLtDestroy(handle);
  }
};

LtGemm::LtGemm() : impl_(std::make_unique<Impl>()) {}
LtGemm::~LtGemm() = default;

bool LtGemm::RunBf16(const void* a_bf16, const void* x_bf16, float* y,
                     std::size_t batch_size, std::size_t m, std::size_t k,
                     hipStream_t stream) {
  if (impl_->handle == nullptr || batch_size == 0 || m == 0 || k == 0)
    return false;
  auto it = std::ranges::find_if(impl_->plans, [&](const auto& plan) {
    return plan->batch_size == batch_size && plan->m == m && plan->k == k;
  });
  Plan* plan = nullptr;
  if (it != impl_->plans.end()) {
    plan = it->get();
  } else {
    auto created = CreatePlan(impl_->handle, batch_size, m, k);
    if (created == nullptr)
      return false;
    plan = created.get();
    impl_->plans.push_back(std::move(created));
  }
  const float alpha = 1.0F;
  const float beta = 0.0F;
  return hipblasLtMatmul(impl_->handle, plan->operation, &alpha, a_bf16,
                         plan->a_layout, x_bf16, plan->x_layout, &beta, y,
                         plan->y_layout, y, plan->y_layout, &plan->algorithm,
                         nullptr, 0, stream) == HIPBLAS_STATUS_SUCCESS;
}

}  // namespace gufo::models::deepseek_v41_flash::vision
