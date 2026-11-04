// TEST ONLY. Execute the source kernel against the synchronous API shim.
#include "../op_kernel/stable_arg_max.cpp"
#include <fstream>
#include <iostream>
#include <iterator>

int main(int argc, char **argv)
{
    if (argc != 5) return 2;
    try {
        const auto rows = static_cast<uint32_t>(std::stoul(argv[1]));
        const auto columns = static_cast<uint32_t>(std::stoul(argv[2]));
        std::ifstream input(argv[3], std::ios::binary);
        std::vector<uint8_t> x((std::istreambuf_iterator<char>(input)), {});
        if (x.size() != static_cast<uint64_t>(rows) * columns * 2) return 2;
        // Guard bytes check the physical write extent independently of shape.
        std::vector<uint8_t> output(rows * 4 + 128, 0xa5);
        TestTiling tiling{rows, columns, 8192};
        stable_arg_max(x.data(), output.data() + 64, nullptr,
                       reinterpret_cast<uint8_t *>(&tiling));
        for (size_t i = 0; i < 64; ++i) {
            if (output[i] != 0xa5 || output[64 + rows * 4 + i] != 0xa5) return 3;
        }
        if (std::any_of(AscendC::events.begin(), AscendC::events.end(), [](bool b) { return b; })) return 4;
        std::ofstream file(argv[4], std::ios::binary);
        file.write(reinterpret_cast<char *>(output.data() + 64), rows * 4);
        return file ? 0 : 5;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 6;
    }
}
