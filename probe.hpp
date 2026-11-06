#ifndef DEEPSEEK41FLASH_PROBE_HPP_
#define DEEPSEEK41FLASH_PROBE_HPP_

#include <gufo/ext/gguf_reader.hpp>
#include <string_view>

namespace gufo::models::deepseek_v41_flash {

/// A `deepseek41` backbone GGUF. DSpark drafters and the vision sidecar are
/// artifacts of their backbone, not models of their own, so they do not match.
[[nodiscard]] bool ProbeDeepSeek41Gguf(const core::GgufReader& reader);
/// `model_root` is the backbone's first GGUF shard.
[[nodiscard]] bool ProbeDeepSeek41(std::string_view model_root);

}  // namespace gufo::models::deepseek_v41_flash

#endif  // DEEPSEEK41FLASH_PROBE_HPP_
