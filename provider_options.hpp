#ifndef GUFO_MODELS_DEEPSEEK_V41_FLASH_PROVIDER_OPTIONS_HPP_
#define GUFO_MODELS_DEEPSEEK_V41_FLASH_PROVIDER_OPTIONS_HPP_

#include <cstdint>
#include <gufo/ext/options.hpp>
#include <limits>
#include <stdexcept>

#include "engine.hpp"

namespace gufo::models::deepseek_v41_flash {

/// Applies the options `extension.json` declares to `options`; an option that
/// was not given keeps the default. Throws std::invalid_argument for values
/// out of range. The engine validates the prefill batch against its limits.
inline void ApplyProviderOptions(const ext::ParsedOptions& given,
                                 ModelOptions& options) {
  using ext::FindOption;
  if (const auto* batch = FindOption<std::int64_t>(given, "--prefill-batch")) {
    if (*batch <= 0 || *batch > std::numeric_limits<std::uint32_t>::max()) {
      throw std::invalid_argument("--prefill-batch must be 16 to 4096");
    }
    options.prefill_batch = static_cast<std::uint32_t>(*batch);
  }
  if (const auto* exact = FindOption<bool>(given, "--exact-prefill")) {
    options.exact_prefill = *exact;
  }
  if (const auto* stream = FindOption<bool>(given, "--stream-experts")) {
    options.stream_experts = *stream;
  }
  if (const auto* bytes =
          FindOption<std::int64_t>(given, "--expert-cache-bytes")) {
    if (*bytes <= 0) {
      throw std::invalid_argument("--expert-cache-bytes must be positive");
    }
    options.expert_cache_bytes = static_cast<std::uint64_t>(*bytes);
  }
}

}  // namespace gufo::models::deepseek_v41_flash

#endif  // GUFO_MODELS_DEEPSEEK_V41_FLASH_PROVIDER_OPTIONS_HPP_
