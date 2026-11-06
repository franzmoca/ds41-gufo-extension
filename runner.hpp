#ifndef GUFO_MODELS_DEEPSEEK_V41_FLASH_RUNNER_HPP_
#define GUFO_MODELS_DEEPSEEK_V41_FLASH_RUNNER_HPP_

#include <gufo/ext/text.hpp>
#include <memory>
#include <string>

namespace gufo::models::deepseek_v41_flash {

std::shared_ptr<ext::TextModelRunner> CreateDeepSeekTextRunner(
    const ext::TextRunnerOptions& options, std::string* error);

}  // namespace gufo::models::deepseek_v41_flash

#endif  // GUFO_MODELS_DEEPSEEK_V41_FLASH_RUNNER_HPP_
