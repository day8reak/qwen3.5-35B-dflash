#include "qwen35_dflash/hiai_executor.hpp"
#include <map>
#include <stdexcept>

// Independent test enums establish template behavior only; they are not vendor
// API declarations or claims about any selected SDK's actual raw values.
enum class HasInt8 { INT8 = 71 };
enum class LacksInt8 { OTHER = 4 };
static_assert(qwen35::dflash::detail::OptionalInt8Enum<HasInt8>::available);
static_assert(!qwen35::dflash::detail::OptionalInt8Enum<LacksInt8>::available);
struct HasTensorName { std::string tensorName; };
struct LacksTensorName { int unrelated; };
static_assert(qwen35::dflash::detail::OptionalTensorName<HasTensorName>::available);
static_assert(!qwen35::dflash::detail::OptionalTensorName<LacksTensorName>::available);

int main() {
  if (qwen35::dflash::detail::OptionalTensorName<HasTensorName>::Read({"exact raw name"}) != "exact raw name"
      || !qwen35::dflash::detail::OptionalTensorName<HasTensorName>::Read({""}).empty()
      || !qwen35::dflash::detail::OptionalTensorName<LacksTensorName>::Read({1}).empty())
    throw std::runtime_error("optional tensorName must preserve actual names and tolerate SDKs without the field");
  std::map<std::string, int> present{{"FLOAT16", 2}};
  qwen35::dflash::detail::OptionalInt8Enum<HasInt8>::Append(present);
  if (present.size() != 2 || present.at("INT8") != 71 || present.at("FLOAT16") != 2)
    throw std::runtime_error("optional INT8 must use the actual enum member value");
  std::map<std::string, int> absent{{"FLOAT16", 2}};
  qwen35::dflash::detail::OptionalInt8Enum<LacksInt8>::Append(absent);
  if (absent.size() != 1 || absent.count("INT8"))
    throw std::runtime_error("absent INT8 must remain unreported without guessing a raw ID");
}
