// Last-axis argmax for finite contiguous FP16. Equal values select first index.
// No CPU fallback; scalar control/merge below executes on the NPU core.
#include "kernel_operator.h"

using namespace AscendC;

class StableArgMaxKernel {
public:
    __aicore__ inline void Init(GM_ADDR logits, GM_ADDR tokens,
                               uint32_t rows, uint32_t vocabulary, uint32_t tileElements)
    {
        rows_ = rows;
        vocabulary_ = vocabulary;
        tileElements_ = tileElements;
        logits_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(logits), rows * vocabulary);
        tokens_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(tokens), rows);
        pipe_.InitBuffer(input_, tileElements * sizeof(half));
        pipe_.InitBuffer(scratch_, tileElements * sizeof(half));
        pipe_.InitBuffer(result_, 32);
        pipe_.InitBuffer(output_, 64);
    }

    __aicore__ inline void Process()
    {
        auto input = input_.Get<half>();
        auto scratch = scratch_.Get<half>();
        auto result = result_.Get<half>();
        auto resultBits = result.ReinterpretCast<uint16_t>();
        auto output = output_.Get<int32_t>();
        for (uint32_t row = 0; row < rows_; ++row) {
            float bestValue = 0.0f;
            uint32_t bestIndex = 0;
            for (uint32_t start = 0; start < vocabulary_; start += tileElements_) {
                const uint32_t remaining = vocabulary_ - start;
                const uint32_t count = remaining < tileElements_ ? remaining : tileElements_;
                // Arbitrary row starts and tails; count excludes local padding.
                DataCopyExtParams read = {};
                read.blockCount = 1;
                read.blockLen = count * sizeof(half);
                DataCopyPadExtParams<half> pad = {};
                DataCopyPad(input, logits_[row * vocabulary_ + start], read, pad);
                Sync<HardEvent::MTE2_V>();
                ReduceMax<half>(result, input, scratch, static_cast<int32_t>(count), true);
                Sync<HardEvent::V_S>();
                const float value = static_cast<float>(result.GetValue(0));
                // ReduceMax stores the local index as raw uint16 bits in the
                // second half slot. A numeric half-to-int conversion is wrong.
                const uint32_t index = start + static_cast<uint32_t>(resultBits.GetValue(1));
                if (start == 0 || value > bestValue || (value == bestValue && index < bestIndex)) {
                    bestValue = value;
                    bestIndex = index;
                }
                // Protect reused reduction results from pending scalar reads,
                // and reused input storage from the preceding vector read.
                Sync<HardEvent::S_V>();
                Sync<HardEvent::V_MTE2>();
            }
            const uint32_t slot = row % 16;
            output.SetValue(slot, static_cast<int32_t>(bestIndex));
            if (slot == 15 || row + 1 == rows_) {
                Sync<HardEvent::S_MTE3>();
                DataCopyExtParams copy = {};
                copy.blockCount = 1;
                copy.blockLen = (slot + 1) * sizeof(int32_t);
                DataCopyPad(tokens_[row - slot], output, copy);
                // Output UB slots can now be reused safely by scalar stores.
                Sync<HardEvent::MTE3_S>();
            }
        }
    }

private:
    template <HardEvent event>
    __aicore__ inline void Sync()
    {
        const auto id = static_cast<TEventID>(pipe_.FetchEventID(event));
        SetFlag<event>(id);
        WaitFlag<event>(id);
    }
    TPipe pipe_;
    TBuf<TPosition::VECIN> input_;
    TBuf<TPosition::VECCALC> scratch_, result_;
    TBuf<TPosition::VECOUT> output_;
    GlobalTensor<half> logits_;
    GlobalTensor<int32_t> tokens_;
    uint32_t rows_, vocabulary_, tileElements_;
};

extern "C" __global__ __aicore__ void stable_arg_max(
    GM_ADDR x, GM_ADDR indices, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(data, tiling);
    StableArgMaxKernel kernel;
    kernel.Init(x, indices, data.rows, data.vocabulary, data.tileElements);
    kernel.Process();
}
