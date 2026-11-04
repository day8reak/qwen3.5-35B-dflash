#ifndef STABLE_ARG_MAX_TILING_H
#define STABLE_ARG_MAX_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(StableArgMaxTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, rows);
    TILING_DATA_FIELD_DEF(uint32_t, vocabulary);
    TILING_DATA_FIELD_DEF(uint32_t, tileElements);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(StableArgMax, StableArgMaxTilingData)
}  // namespace optiling
#endif
