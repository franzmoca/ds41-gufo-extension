// The registered V4.1 tool dialect: its markers, grammar and parser agree on
// the wire format, and the sampling defaults stay at DeepSeek's.
#include "text_contract.hpp"

#include <cstdio>
#include <cstdlib>
#include <gufo/ext/json.hpp>
#include <gufo/ext/text.hpp>
#include <string>
#include <vector>

#include "tool_calls.hpp"

namespace ds41 = gufo::models::deepseek_v41_flash;

#define CHECK(condition)                                                   \
  do {                                                                     \
    if (!(condition)) {                                                    \
      std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #condition); \
      std::exit(1);                                                        \
    }                                                                      \
  } while (0)

int main() {
  const auto dialect = ds41::ToolDialect();
  CHECK(dialect.id == "deepseek41");
  CHECK(dialect.syntax.has_value());
  const auto& syntax = *dialect.syntax;
  CHECK(syntax.open == ds41::DeepSeekToolSyntax::kCallsOpen);
  CHECK(syntax.open + syntax.name_prefix ==
        ds41::DeepSeekToolSyntax::kNativeCallOpen);
  CHECK(syntax.close == ds41::DeepSeekToolSyntax::kInvokeCloses[0]);
  CHECK(syntax.block_close ==
        "\n" + std::string(ds41::DeepSeekToolSyntax::kCallsClose));
  CHECK(syntax.tagged.has_value());
  CHECK(syntax.tagged->value_close ==
        ds41::DeepSeekToolSyntax::kParameterCloses[0]);
  // V4's spelling is another dialect's.
  CHECK(syntax.open.find("tool_calls") == std::string::npos);

  gufo::ext::RegisterToolDialect(dialect);
  CHECK(gufo::ext::FindToolDialect("deepseek41") != nullptr);

  // What the grammar admits, the parser reads back.
  const auto schema = gufo::json::parse(
      R"({"type":"object","properties":{"city":{"type":"string"},)"
      R"("days":{"type":"integer"}},"required":["city"]})");
  CHECK(gufo::sampling::JsonConstraint::ToolParameters(schema, false, syntax) !=
        nullptr);
  const std::string text =
      syntax.open + syntax.name_prefix + "get_weather" + syntax.name_suffix +
      syntax.tagged->parameter_open + "city" + syntax.tagged->raw_value_open +
      "Rome" + syntax.tagged->value_close + syntax.tagged->parameter_separator +
      syntax.tagged->parameter_open + "days" + syntax.tagged->typed_value_open +
      "3" + syntax.tagged->value_close + "\n" + syntax.close +
      syntax.block_close;
  std::vector<gufo::ext::ScannedToolCall> calls;
  std::vector<std::pair<std::size_t, std::size_t>> spans;
  dialect.scan_tool_calls(text, dialect.markers, &calls, &spans, false);
  CHECK(calls.size() == 1);
  CHECK(calls[0].name == "get_weather");
  CHECK(calls[0].valid && calls[0].complete);
  CHECK(calls[0].arguments.size() == 2);
  CHECK(calls[0].arguments[0].value == "Rome" &&
        calls[0].arguments[0].is_string);
  CHECK(calls[0].arguments[1].value == "3" && !calls[0].arguments[1].is_string);
  CHECK(spans.size() == 1 && spans[0].first == 0 &&
        spans[0].second == text.size());

  const auto defaults = ds41::SamplingDefaults();
  for (const auto thinking : {std::optional<bool>{}, std::optional<bool>{true},
                              std::optional<bool>{false}}) {
    CHECK(defaults.For(thinking).temperature == 1.0F);
    CHECK(defaults.For(thinking).top_p == 1.0F);
    CHECK(!defaults.For(thinking).top_k.has_value());
  }
  return 0;
}
