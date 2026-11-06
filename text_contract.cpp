#include "text_contract.hpp"

#include <string>

#include "tool_calls.hpp"

namespace gufo::models::deepseek_v41_flash {

const sampling::JsonConstraint::ToolSyntax& ToolGrammarSyntax() {
  static const sampling::JsonConstraint::ToolSyntax syntax{
      .open = std::string(DeepSeekToolSyntax::kCallsOpen),
      .name_prefix = "\n<｜DSML｜ invoke name=\"",
      .name_suffix = "\">\n",
      .close = "</｜DSML｜ invoke>",
      .block_close = "\n</｜DSML｜ calls>",
      .tagged =
          sampling::JsonConstraint::TaggedArguments{
              .parameter_open = "<｜DSML｜ parameter name=\"",
              .raw_value_open = "\" string=\"true\">",
              .typed_value_open = "\" string=\"false\">",
              .value_close = "</｜DSML｜ parameter>",
              .parameter_separator = "\n",
          },
      .reasoning_header = true,
  };
  return syntax;
}

ext::ToolDialect ToolDialect() {
  return {
      .markers =
          {
              DeepSeekToolSyntax::kCallsOpenWithSeparator,
              DeepSeekToolSyntax::kCallsOpen,
          },
      .reasoning_markers =
          {
              DeepSeekToolSyntax::kNativeCallOpenWithSeparator,
              DeepSeekToolSyntax::kNativeCallOpen,
          },
      // As in llama.cpp, DeepSeek content is everything outside its native
      // call block, which ends the output: there is no echo to strip, and
      // client envelope syntax the model writes stays visible text.
      .closers = {},
      .scan_tool_calls = &ScanDeepSeekToolCalls,
      .id = std::string(kToolDialectId),
      .syntax = ToolGrammarSyntax(),
  };
}

sampling::TextSamplingDefaults SamplingDefaults() {
  sampling::TextSamplingValues values;
  values.temperature = 1.0F;
  values.top_p = 1.0F;
  return {values, values, values};
}

std::optional<sampling::TextSamplingDefaults> ProbeSamplingDefaults(
    const core::GgufReader&) {
  return SamplingDefaults();
}

}  // namespace gufo::models::deepseek_v41_flash
