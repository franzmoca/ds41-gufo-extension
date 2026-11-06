#include <gufo/ext/generated/deepseek41flash_constants.hpp>
#include <gufo/ext/registry.hpp>
#include <gufo/ext/text.hpp>

#include "probe.hpp"

void deepseek41flash_register(gufo::ext::Registry&) {}

// Metadata and probe only: this revision has no execution factory yet.
void deepseek41flash_provider_register(gufo::ext::Registry& registry) {
  namespace ds41 = gufo::models::deepseek_v41_flash;
  registry.AddProvider(
      gufo::ext::generated::extension_deepseek41flash::kProvider_deepseek41);
  registry.AddTextProvider({
      .id = "deepseek41",
      .probe = &ds41::ProbeDeepSeek41,
      .probe_gguf = &ds41::ProbeDeepSeek41Gguf,
      .create = nullptr,
      .bench_driver = nullptr,
      .prompt_driver = nullptr,
      .chat_driver = nullptr,
  });
}
