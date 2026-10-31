#pragma once
#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>

namespace qwen35::dflash::detail {
// Optional startup experiment: read the same open file sequentially into a
// bounded scratch buffer. No second whole-model copy, hash, mlock, cache purge
// or change to the mapping handed to the SDK. The OS may evict pages afterward.
inline void SequentialModelPreread(int fd, std::uint64_t size, std::uint64_t& bytes_read) {
  if (!size || size > UINT32_MAX || bytes_read)
    throw std::invalid_argument("model preread requires one nonempty uint32-sized file");
  std::vector<unsigned char> scratch(static_cast<std::size_t>(std::min<std::uint64_t>(size, 1U << 20)));
  while (bytes_read < size) {
    const auto count = static_cast<std::size_t>(std::min<std::uint64_t>(scratch.size(), size - bytes_read));
    const auto n = pread(fd, scratch.data(), count, static_cast<off_t>(bytes_read));
    if (n < 0 && errno == EINTR) continue;
    if (n < 0) throw std::runtime_error(std::string("preread OMC: ") + std::strerror(errno));
    if (!n) throw std::runtime_error("preread OMC: file ended before declared extent");
    bytes_read += static_cast<std::uint64_t>(n);
  }
}
}  // namespace qwen35::dflash::detail
