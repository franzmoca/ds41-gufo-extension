#include <gufo/ext/registry.hpp>
#include <gufo/ext/text.hpp>

#include "probe.hpp"
#include "text_contract.hpp"

// The tool dialect is host-side and registers in every build, before any
// runner is created. HIP builds register the provider from
// gufo_deepseek41flash_text (provider.cpp); without HIP the probes register
// with no execution factories.
void deepseek41flash_register(gufo::ext::Registry& registry) {
  gufo::ext::RegisterToolDialect(
      gufo::models::deepseek_v41_flash::ToolDialect());
#if !defined(ENGINE_ENABLE_HIP)
  namespace ds41 = gufo::models::deepseek_v41_flash;
  registry.AddTextProvider({
      .id = "deepseek41",
      .probe = &ds41::ProbeDeepSeek41,
      .probe_gguf = &ds41::ProbeDeepSeek41Gguf,
      .create = nullptr,
      .bench_driver = nullptr,
      .prompt_driver = nullptr,
      .chat_driver = nullptr,
      .sampling_defaults = &ds41::ProbeSamplingDefaults,
  });
#else
  (void)registry;
#endif
}
