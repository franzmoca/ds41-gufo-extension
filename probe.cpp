#include "probe.hpp"

#include <filesystem>
#include <string>
#include <system_error>

namespace gufo::models::deepseek_v41_flash {

bool ProbeDeepSeek41Gguf(const core::GgufReader& reader) {
  return reader.GetMetadataString("general.architecture").value_or("") ==
         "deepseek41";
}

bool ProbeDeepSeek41(std::string_view model_root) {
  try {
    const std::filesystem::path path(model_root);
    std::error_code ec;
    if (path.extension() != ".gguf" ||
        !std::filesystem::is_regular_file(path, ec)) {
      return false;
    }
    std::string error;
    const auto reader = core::GgufReader::OpenFile(path.string(), &error);
    return reader != nullptr && ProbeDeepSeek41Gguf(*reader);
  } catch (...) {
    return false;
  }
}

}  // namespace gufo::models::deepseek_v41_flash
