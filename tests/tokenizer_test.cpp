// DeepSeek V4.1 Flash tokenizer: request text is counted bytes, so a NUL
// tokenizes as a byte instead of ending the prompt.
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <string_view>
#include <vector>

#include "runtime/native_internal.h"

namespace {

using namespace std::string_view_literals;

int failures = 0;

void Expect(bool condition, const std::string& what) {
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++failures;
  }
}

void PutU32(std::vector<std::uint8_t>& out, std::uint32_t value) {
  for (int i = 0; i < 4; ++i)
    out.push_back(static_cast<std::uint8_t>(value >> (8 * i)));
}

void PutU64(std::vector<std::uint8_t>& out, std::uint64_t value) {
  for (int i = 0; i < 8; ++i)
    out.push_back(static_cast<std::uint8_t>(value >> (8 * i)));
}

void PutString(std::vector<std::uint8_t>& out, const std::string& text) {
  PutU64(out, text.size());
  out.insert(out.end(), text.begin(), text.end());
}

void PutStringArray(std::vector<std::uint8_t>& out, const std::string& key,
                    const std::vector<std::string>& items) {
  PutString(out, key);
  PutU32(out, 9);  // array
  PutU32(out, 8);  // of strings
  PutU64(out, items.size());
  for (const auto& item : items)
    PutString(out, item);
}

// GPT-2 byte-level BPE spells byte `b` as this codepoint, in UTF-8.
std::string ByteSymbol(unsigned b) {
  const auto printable = [](unsigned x) {
    return (x >= 33 && x <= 126) || (x >= 161 && x <= 172) || x >= 174;
  };
  unsigned cp = b;
  if (!printable(b)) {
    cp = 256;
    for (unsigned x = 0; x < b; ++x)
      cp += printable(x) ? 0 : 1;
  }
  std::string out;
  if (cp < 0x80) {
    out += static_cast<char>(cp);
  } else {
    out += static_cast<char>(0xc0 | (cp >> 6));
    out += static_cast<char>(0x80 | (cp & 0x3f));
  }
  return out;
}

constexpr std::string_view kSpecials[] = {"<｜begin▁of▁sentence｜>",
                                          "<｜end▁of▁sentence｜>",
                                          "<｜User｜>",
                                          "<｜Assistant｜>",
                                          "<think>",
                                          "</think>",
                                          "｜DSML｜",
                                          "<｜deepseek_image｜>"};
constexpr int kUser = 256 + 2;
constexpr int kAssistant = 256 + 3;

// A vocabulary of the 256 byte tokens (token `b` is byte `b`) and the
// control tokens after them, without merges.
std::vector<std::uint8_t> VocabGguf() {
  std::vector<std::string> tokens;
  for (unsigned b = 0; b < 256; ++b)
    tokens.push_back(ByteSymbol(b));
  for (const auto special : kSpecials)
    tokens.emplace_back(special);
  std::vector<std::uint8_t> out;
  PutU32(out, 0x46554747);  // "GGUF"
  PutU32(out, 3);
  PutU64(out, 0);  // tensors
  PutU64(out, 2);  // metadata
  PutStringArray(out, "tokenizer.ggml.tokens", tokens);
  PutStringArray(out, "tokenizer.ggml.merges", {});
  out.resize((out.size() + 31) / 32 * 32, 0);
  return out;
}

std::vector<int> Take(ds41_tokens* tokens) {
  std::vector<int> out(tokens->v, tokens->v + tokens->len);
  ds41_tokens_free(tokens);
  return out;
}

std::vector<int> Bytes(std::string_view text) {
  std::vector<int> out;
  for (const char c : text)
    out.push_back(static_cast<unsigned char>(c));
  return out;
}

}  // namespace

int main() {
  char path[] = "/tmp/ds41_tokenizer_XXXXXX";
  const int fd = mkstemp(path);
  const auto bytes = VocabGguf();
  if (fd < 0 || write(fd, bytes.data(), bytes.size()) !=
                    static_cast<ssize_t>(bytes.size())) {
    std::perror("ds41 tokenizer_test: temp file");
    return 1;
  }
  close(fd);
  ds41_model model{};
  ds41_model_open(&model, path);
  ds41_vocab* vocab = ds41_vocab_create(&model);

  const auto plain = [&](std::string_view text) {
    ds41_tokens tokens{};
    ds41_vocab_tokenize(vocab, text.data(), text.size(), &tokens);
    return Take(&tokens);
  };
  const auto rendered = [&](std::string_view text) {
    ds41_tokens tokens{};
    ds41_vocab_tokenize_rendered_chat(vocab, text.data(), text.size(), &tokens);
    return Take(&tokens);
  };

  const std::string_view text = "A\0B\0\0C"sv;
  Expect(plain(text) == Bytes(text), "plain text tokenizes past a NUL");

  std::vector<int> chat = {kUser};
  for (int b : Bytes("x\0y"sv))
    chat.push_back(b);
  chat.push_back(kAssistant);
  Expect(rendered("<｜User｜>x\0y<｜Assistant｜>"sv) == chat,
         "a control token after a NUL is still read");

  // The counted text ends before the spelling does.
  const std::string_view think = "<think>";
  Expect(rendered(think.substr(0, think.size() - 1)) ==
             Bytes(think.substr(0, think.size() - 1)),
         "a spelling cut by the text's end stays text");

  ds41_vocab_destroy(vocab);
  ds41_model_close(&model);
  unlink(path);
  if (failures != 0)
    return 1;
  std::printf("tokenizer: ok\n");
  return 0;
}
