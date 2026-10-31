#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace qwen35::dflash {
class TextTokenizer {
 public:
  explicit TextTokenizer(const std::string& bundle);
  ~TextTokenizer();
  TextTokenizer(const TextTokenizer&) = delete;
  TextTokenizer& operator=(const TextTokenizer&) = delete;
  std::vector<std::uint32_t> Encode(const std::string& text, const std::string& mode) const;
  std::string Decode(const std::vector<std::uint32_t>& ids, bool skip_special = true) const;
  std::vector<std::uint32_t> Unmapped(const std::vector<std::uint32_t>& ids) const;
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}
