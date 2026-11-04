// TEST ONLY: synchronous host shim, not an AscendC SDK or hardware simulator.
// Compiles the actual kernel loops; ReduceMax is an ideal first-maximum oracle.
#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>
#define __aicore__
#define __global__
#define __gm__
struct half {
    uint16_t bits;
    operator float() const {
        const uint16_t exp = (bits >> 10) & 31;
        const uint16_t frac = bits & 1023;
        const float magnitude = exp == 0 ? std::ldexp(static_cast<float>(frac), -24) :
            exp == 31 ? (frac ? std::numeric_limits<float>::quiet_NaN() :
                               std::numeric_limits<float>::infinity()) :
            std::ldexp(static_cast<float>(1024 + frac), static_cast<int>(exp) - 25);
        return bits & 0x8000 ? -magnitude : magnitude;
    }
};
static_assert(sizeof(half) == 2, "test binary16 storage");
using GM_ADDR = uint8_t *;
struct TestTiling { uint32_t rows, vocabulary, tileElements; };
#define GET_TILING_DATA(name, address) TestTiling name; std::memcpy(&name, address, sizeof(name))

namespace AscendC {
enum class TPosition { VECIN, VECCALC, VECOUT };
enum class HardEvent { MTE2_V, V_S, S_V, V_MTE2, S_MTE3, MTE3_S };
using TEventID = int;
inline std::array<bool, 6> events{};
template <HardEvent event> void SetFlag(TEventID) {
    auto &set = events.at(static_cast<int>(event));
    if (set) throw std::runtime_error("event set twice");
    set = true;
}
template <HardEvent event> void WaitFlag(TEventID) {
    auto &set = events.at(static_cast<int>(event));
    if (!set) throw std::runtime_error("unmatched event wait");
    set = false;
}
template <class T> class LocalTensor {
public:
    explicit LocalTensor(std::shared_ptr<std::vector<uint8_t>> data) : data_(data) {}
    T GetValue(size_t index) const {
        T value;
        check((index + 1) * sizeof(T));
        std::memcpy(&value, data_->data() + index * sizeof(T), sizeof(T));
        return value;
    }
    void SetValue(size_t index, T value) const {
        check((index + 1) * sizeof(T));
        std::memcpy(data_->data() + index * sizeof(T), &value, sizeof(T));
    }
    template <class U> LocalTensor<U> ReinterpretCast() const { return LocalTensor<U>(data_); }
    uint8_t *bytes(size_t n) const { check(n); return data_->data(); }
private:
    void check(size_t n) const { if (n > data_->size()) throw std::runtime_error("UB bounds"); }
    std::shared_ptr<std::vector<uint8_t>> data_;
};
template <class T> class GlobalTensor {
public:
    void SetGlobalBuffer(T *data, size_t size) { data_ = data; size_ = size; }
    GlobalTensor operator[](size_t offset) const {
        if (offset > size_) throw std::runtime_error("GM offset");
        GlobalTensor result;
        result.SetGlobalBuffer(data_ + offset, size_ - offset);
        return result;
    }
    uint8_t *bytes(size_t n) const {
        if (n > size_ * sizeof(T)) throw std::runtime_error("GM bounds");
        return reinterpret_cast<uint8_t *>(data_);
    }
private:
    T *data_ = nullptr;
    size_t size_ = 0;
};
template <TPosition position> class TBuf {
public:
    void Init(size_t bytes) { data_ = std::make_shared<std::vector<uint8_t>>(bytes, 0xa5); }
    template <class T> LocalTensor<T> Get() { return LocalTensor<T>(data_); }
private:
    std::shared_ptr<std::vector<uint8_t>> data_;
};
class TPipe {
public:
    template <TPosition position> void InitBuffer(TBuf<position> &buffer, size_t bytes) {
        buffer.Init(bytes);
    }
    TEventID FetchEventID(HardEvent) { return 0; }
};
struct DataCopyExtParams { uint16_t blockCount; uint32_t blockLen, srcStride, dstStride, rsv; };
template <class T> struct DataCopyPadExtParams { bool isPad; uint8_t leftPadding, rightPadding; T paddingValue; };
template <class T> void DataCopyPad(LocalTensor<T> dst, GlobalTensor<T> src,
                                    DataCopyExtParams p, DataCopyPadExtParams<T>) {
    if (p.blockCount != 1 || p.srcStride || p.dstStride) throw std::runtime_error("copy geometry");
    std::memcpy(dst.bytes(p.blockLen), src.bytes(p.blockLen), p.blockLen);
}
template <class T> void DataCopyPad(GlobalTensor<T> dst, LocalTensor<T> src, DataCopyExtParams p) {
    if (p.blockCount != 1 || p.srcStride || p.dstStride) throw std::runtime_error("copy geometry");
    std::memcpy(dst.bytes(p.blockLen), src.bytes(p.blockLen), p.blockLen);
}
template <class T> void ReduceMax(LocalTensor<T> dst, LocalTensor<T> src,
                                  LocalTensor<T> scratch, int32_t count, bool calIndex) {
    if (!calIndex || count < 1 || count > 8192) throw std::runtime_error("reduction contract");
    scratch.bytes(static_cast<size_t>(count) * sizeof(T));
    T best = src.GetValue(0);
    uint16_t index = 0;
    for (int32_t i = 1; i < count; ++i) {
        T value = src.GetValue(i);
        if (value > best) { best = value; index = static_cast<uint16_t>(i); }
    }
    dst.SetValue(0, best);
    dst.template ReinterpretCast<uint16_t>().SetValue(1, index);
}
}  // namespace AscendC
