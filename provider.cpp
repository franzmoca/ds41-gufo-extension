#include <chrono>
#include <gufo/ext/generated/deepseek41flash_constants.hpp>
#include <gufo/ext/registry.hpp>
#include <gufo/ext/text.hpp>
#include <memory>

#include "probe.hpp"
#include "runner.hpp"
#include "text_contract.hpp"

namespace gufo::models::deepseek_v41_flash {

int RunDeepSeekBenchmark(const ext::TextBenchOptions& options,
                         const std::shared_ptr<const core::GgufReader>& reader,
                         std::chrono::steady_clock::time_point load_start);

int RunDeepSeekPrompt(const ext::TextPromptOptions& options,
                      const core::GgufReader& reader,
                      std::chrono::steady_clock::time_point load_start);

int RunDeepSeekChat(const ext::TextPromptOptions& options,
                    const core::GgufReader& reader,
                    std::chrono::steady_clock::time_point load_start);

}  // namespace gufo::models::deepseek_v41_flash

void deepseek41flash_provider_register(gufo::ext::Registry& registry) {
  namespace ds41 = gufo::models::deepseek_v41_flash;
  registry.AddProvider(
      gufo::ext::generated::extension_deepseek41flash::kProvider_deepseek41);
  registry.AddTextProvider({
      .id = "deepseek41",
      .probe = &ds41::ProbeDeepSeek41,
      .probe_gguf = &ds41::ProbeDeepSeek41Gguf,
      .create = &ds41::CreateDeepSeekTextRunner,
      .bench_driver = &ds41::RunDeepSeekBenchmark,
      .prompt_driver = &ds41::RunDeepSeekPrompt,
      .chat_driver = &ds41::RunDeepSeekChat,
      .sampling_defaults = &ds41::ProbeSamplingDefaults,
      // MXFP8 Engram tables (type 43): the runtime's own GGUF parser sizes
      // and loads them.
      .accepts_unsized_tensors = true,
      .speculative_defaults = {{.backend = "dspark",
                                .max_draft_tokens = 7,
                                .min_draft_tokens = 1}},
  });
}
