// Source candidate for the selected CANN Kit; target compilation is pending.
#include "stable_arg_max_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <cstddef>
#include <cstdint>

namespace {
constexpr uint32_t kTileElements = 8192;
// One input tile, a separate equally sized reduction workspace, 32-byte
// reduction result, and sixteen INT32 output slots. Output copies use valid bytes.
constexpr uint64_t kUbBytes = 2 * kTileElements * sizeof(uint16_t) + 32 + 64;

bool GetGeometry(const gert::Shape &shape, uint32_t &rows, uint32_t &vocabulary)
{
    const auto rank = shape.GetDimNum();
    if (rank < 1 || rank > 8) {
        return false;
    }
    uint64_t elements = 1;
    for (size_t i = 0; i < rank; ++i) {
        const auto dim = shape.GetDim(i);
        if (dim <= 0 || dim > 2147483647 || elements > 2147483647ULL / dim) {
            return false;  // Static, nonempty shapes; INT32-bounded element offsets.
        }
        elements *= dim;
    }
    vocabulary = static_cast<uint32_t>(shape.GetDim(rank - 1));
    rows = static_cast<uint32_t>(elements / vocabulary);
    return true;
}
}  // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    if (context == nullptr || context->GetInputShape(0) == nullptr ||
        context->GetPlatformInfo() == nullptr || context->GetRawTilingData() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto *shape = context->GetInputShape(0);
    // The kernel accepts only contiguous ND, with no hidden physical padding.
    uint32_t rows = 0, vocabulary = 0, storageRows = 0, storageVocabulary = 0;
    const auto &origin = shape->GetOriginShape();
    const auto &storage = shape->GetStorageShape();
    if (!GetGeometry(origin, rows, vocabulary) ||
        !GetGeometry(storage, storageRows, storageVocabulary) ||
        origin.GetDimNum() != storage.GetDimNum()) {
        return ge::GRAPH_FAILED;
    }
    for (size_t i = 0; i < origin.GetDimNum(); ++i) {
        if (origin.GetDim(i) != storage.GetDim(i)) {
            return ge::GRAPH_FAILED;
        }
    }
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t ubBytes = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
    if (ubBytes < kUbBytes) {
        return ge::GRAPH_FAILED;
    }
    StableArgMaxTilingData data;
    data.set_rows(rows);
    data.set_vocabulary(vocabulary);
    data.set_tileElements(kTileElements);
    auto *raw = context->GetRawTilingData();
    if (raw->GetData() == nullptr || raw->GetCapacity() < data.GetDataSize()) {
        return ge::GRAPH_FAILED;
    }
    auto *workspace = context->GetWorkspaceSizes(1);
    if (workspace == nullptr) {
        return ge::GRAPH_FAILED;
    }
    // CANN Kit documents BlockDim=1 for KirinX90. No multi-core assumption.
    context->SetBlockDim(1);
    context->SetTilingKey(1);
    data.SaveToBuffer(raw->GetData(), raw->GetCapacity());
    raw->SetDataSize(data.GetDataSize());
    workspace[0] = 0;  // Basic ReduceMax uses the explicit UB scratch buffer.
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context)
{
    if (context == nullptr || context->GetInputShape(0) == nullptr ||
        context->GetOutputShape(0) == nullptr) {
        return GRAPH_FAILED;
    }
    uint32_t rows = 0, vocabulary = 0;
    const auto &input = *context->GetInputShape(0);
    if (!GetGeometry(input, rows, vocabulary)) {
        return GRAPH_FAILED;
    }
    auto *output = context->GetOutputShape(0);
    output->SetDimNum(input.GetDimNum() - 1);
    for (size_t i = 0; i + 1 < input.GetDimNum(); ++i) {
        output->SetDim(i, input.GetDim(i));
    }
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    if (context == nullptr || context->GetInputDataType(0) != DT_FLOAT16) {
        return GRAPH_FAILED;
    }
    context->SetOutputDataType(0, DT_INT32);
    return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class StableArgMax : public OpDef {
public:
    explicit StableArgMax(const char *name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
        this->Output("indices").ParamType(REQUIRED)
            .DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape);
        this->SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        // Add a 310P configuration only after its SDK/build/runtime validation.
        this->AICore().AddConfig("kirinx90");
    }
};
OP_ADD(StableArgMax);
}  // namespace ops
