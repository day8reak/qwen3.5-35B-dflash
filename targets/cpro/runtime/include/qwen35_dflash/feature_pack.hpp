#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace qwen35::dflash {
// Source project_features: [1,16,16384] -> [1,1024,16,16].
// Each token supplies one 4x4 spatial patch per channel to stride-4 Conv.
// Copy storage bytes only, including signed zero/NaN payloads. Distinct,
// preallocated buffers are required; no allocation or arithmetic on FP16.
inline void PackDraftFeaturesC16(const std::vector<std::uint8_t>& source,
                                 std::vector<std::uint8_t>& destination) {
  constexpr std::size_t bytes = 16 * 16384 * 2;
  if (&source == &destination || source.size() != bytes || destination.size() != bytes)
    throw std::invalid_argument("feature packing requires distinct 512 KiB buffers");
  for (std::size_t channel = 0; channel < 1024; ++channel)
    for (std::size_t tile_y = 0; tile_y < 4; ++tile_y)
      for (std::size_t tile_x = 0; tile_x < 4; ++tile_x)
        for (std::size_t y = 0; y < 4; ++y) {
          const auto src = (((tile_y * 4 + tile_x) * 1024 + channel) * 16 + y * 4) * 2;
          const auto dst = ((channel * 16 + tile_y * 4 + y) * 16 + tile_x * 4) * 2;
          std::memcpy(destination.data() + dst, source.data() + src, 8);
        }
}
// Bounded projection experiment only. Same K=16384 terms and token order,
// carried by stride-2 2x2 patches with 4096 channels. No FP16 arithmetic.
inline void PackDraftFeaturesC16Kernel2(const std::vector<std::uint8_t>& source,
                                       std::vector<std::uint8_t>& destination) {
  constexpr std::size_t bytes = 16 * 16384 * 2;
  if (&source == &destination || source.size() != bytes || destination.size() != bytes)
    throw std::invalid_argument("feature packing requires distinct 512 KiB buffers");
  for (std::size_t channel = 0; channel < 4096; ++channel)
    for (std::size_t tile_y = 0; tile_y < 4; ++tile_y)
      for (std::size_t tile_x = 0; tile_x < 4; ++tile_x)
        for (std::size_t y = 0; y < 2; ++y) {
          const auto src = (((tile_y * 4 + tile_x) * 4096 + channel) * 4 + y * 2) * 2;
          const auto dst = ((channel * 8 + tile_y * 2 + y) * 8 + tile_x * 2) * 2;
          std::memcpy(destination.data() + dst, source.data() + src, 4);
        }
}
}  // namespace qwen35::dflash
