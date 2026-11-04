// Host-only executable specification. This is NOT an Ascend C kernel.
// Decode binary16 ordering without depending on a host half type or FP flags.
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

constexpr int rows = 15;
constexpr int vocabulary = 248320;

static uint16_t key(uint16_t bits) {
    if ((bits & 0x7fff) == 0) bits = 0; // +0 and -0 compare equal.
    return (bits & 0x8000) ? static_cast<uint16_t>(~bits)
                           : static_cast<uint16_t>(bits ^ 0x8000);
}

int main(int argc, char** argv) {
    try {
        if (argc != 3) throw std::runtime_error("usage: top1_cpu input.bin output.bin");
        std::ifstream input(argv[1], std::ios::binary | std::ios::ate);
        constexpr size_t bytes = size_t(rows) * vocabulary * 2;
        if (!input || input.tellg() != static_cast<std::streamoff>(bytes))
            throw std::runtime_error("expected exactly 7449600 input bytes");
        input.seekg(0);
        std::vector<unsigned char> data(bytes);
        if (!input.read(reinterpret_cast<char*>(data.data()), bytes))
            throw std::runtime_error("input read failed");
        std::vector<unsigned char> output(rows * 4);
        for (int row = 0; row < rows; ++row) {
            uint16_t best = 0;
            int32_t winner = 0;
            for (int index = 0; index < vocabulary; ++index) {
                const size_t offset = (size_t(row) * vocabulary + index) * 2;
                const uint16_t bits = uint16_t(data[offset]) | (uint16_t(data[offset + 1]) << 8);
                if ((bits & 0x7c00) == 0x7c00)
                    throw std::runtime_error("finite FP16 contract violated");
                const uint16_t candidate = key(bits);
                if (index == 0 || candidate > best) {
                    best = candidate;
                    winner = index;
                }
            }
            for (int byte = 0; byte < 4; ++byte)
                output[row * 4 + byte] = static_cast<unsigned char>(uint32_t(winner) >> (byte * 8));
        }
        // Open output only after validating all rows. No padding/extra writes.
        std::ofstream destination(argv[2], std::ios::binary);
        if (!destination.write(reinterpret_cast<const char*>(output.data()), output.size()))
            throw std::runtime_error("output write failed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
}
