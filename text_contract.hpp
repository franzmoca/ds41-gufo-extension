#ifndef GUFO_MODELS_DEEPSEEK_V41_FLASH_TEXT_CONTRACT_HPP_
#define GUFO_MODELS_DEEPSEEK_V41_FLASH_TEXT_CONTRACT_HPP_

#include <gufo/ext/gguf_reader.hpp>
#include <gufo/ext/text.hpp>
#include <optional>
#include <string_view>

namespace gufo::models::deepseek_v41_flash {

/// Identity of the registered V4.1 DSML dialect, which writes each tag name
/// after a space and names the call block `calls`.
inline constexpr std::string_view kToolDialectId = "deepseek41";

/// The native call grammar the dialect constrains generation with.
[[nodiscard]] const sampling::JsonConstraint::ToolSyntax& ToolGrammarSyntax();

/// The dialect the extension registers.
[[nodiscard]] ext::ToolDialect ToolDialect();

/// DeepSeek's API defaults, with top-p left open at 1.
[[nodiscard]] sampling::TextSamplingDefaults SamplingDefaults();
/// `TextProvider::sampling_defaults`.
[[nodiscard]] std::optional<sampling::TextSamplingDefaults>
ProbeSamplingDefaults(const core::GgufReader& reader);

}  // namespace gufo::models::deepseek_v41_flash

#endif  // GUFO_MODELS_DEEPSEEK_V41_FLASH_TEXT_CONTRACT_HPP_
