# OHOS Native SDK 与 HIAI 构建

[环境配置](environment-setup.md) · [板端执行](board-guide.md)

HIAI runner 需要 Linux 主机可执行的 OHOS 编译器、sysroot、CMake toolchain，以及匹配的 HIAI 头文件和 HarmonyOS ARM64 库。板端的 `.so` 或 AscendC 编译环境不能替代 Native SDK。

## 复用已有安装

优先使用内部已验证示例的工具链。典型目录为 `command-line-tools/sdk/default/openharmony/native`，但应以实际文件为准：

```bash
export OHOS_NATIVE_SDK='/your/openharmony/native'
export HIAI_INCLUDE_DIR='/your/hiai/include'
export HIAI_LIBRARIES='/your/hiai/ohos/libhiai.so'
test -f "$OHOS_NATIVE_SDK/build/cmake/ohos.toolchain.cmake"
test -x "$OHOS_NATIVE_SDK/llvm/bin/clang++"
test -d "$OHOS_NATIVE_SDK/sysroot"
test -f "$HIAI_INCLUDE_DIR/HiAiModelManagerService.h"
test -f "$HIAI_LIBRARIES"
"$OHOS_NATIVE_SDK/llvm/bin/clang++" --version
cmake --version
```

这里只举单个 HIAI 库；多个链接库以冒号分隔。资源 JSON 中非空的 `hiai_libraries` 优先于环境变量。不要用 root shell 的 HOME 推断其他用户的 SDK 目录，也不要只拷贝一份 toolchain 文件。

## 没有安装时

使用内部已验证的 `*-ohos-sdk-public.tar.gz`，或从[华为官方下载中心](https://developer.huawei.com/consumer/cn/download/command-line-tools-for-hmos)获取与固件、HIAI 库匹配的 Linux 主机包；版本兼容性需按内部示例核对。下载与解压放在仓库外，已有可用安装无需重复安装。

内部 public 包通常先解压 tar.gz，再解压其 `linux/` 下的组件 zip；Command Line Tools 包按自身目录结构解压。只在本次解压目录内寻找 `ohos.toolchain.cmake`，取 `build/cmake/ohos.toolchain.cmake` 之前的目录作为 `OHOS_NATIVE_SDK`，完成上述检查。

## 构建与结果含义

构建命令统一见[板端执行](board-guide.md)。`build_hiai.py` 默认使用 SDK 自带的兼容 API 策略；如内部环境要求，显式增加 `--compatible-sdk-version <正整数>`。它不会自动检测板端 API，任意降低该数值不能解决 HIAI/固件 ABI 不匹配。

`command.json` / `result.json` 记录工具链、库及构建结果。`--headers-only` 仅验证头文件编译；`LINKED` 表示得到链接产物，实际兼容性仍以板端加载与执行为准。构建记录应与 runner 一起保留，上传脚本会据此校验配套运行库。
