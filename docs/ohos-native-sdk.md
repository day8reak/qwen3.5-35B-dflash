# 服务器缺少 OHOS Native SDK

HIAI 程序的交叉编译需要 **Linux 主机可执行的 OHOS 编译器、sysroot 和 CMake toolchain**。`libs/ohos/libhiai.so` 与 `libc++_shared.so` 是链接/运行库，不能替代这些组件。AscendC 环境用于算子编译，也不等于已配置 OHOS C++ 应用工具链。

SDK 只放在服务器；板端使用编译后的程序、OMC 和匹配的运行库。已有成功的 target/draft OMC 与 prepared 保留，无需重跑 AIR/OMG。

## 先复用已有工具链配置

若内部已有能在同一 CPro 固件上运行的 `model_run_tool`，优先查看其 README、CMakeLists、构建脚本中指定的编译器、sysroot、SDK/API 和 C++ 运行库。只有 `libs/ohos` 目录不足以证明 SDK 存在；只有 `file libhiai.so` 显示 `ARM aarch64` 也不足以证明其系统 ABI 与板端匹配。

复用时仍通过本项目的 `OHOS_NATIVE_SDK`、`HIAI_INCLUDE_DIR`、`HIAI_LIBRARIES` 指定资源，不修改共享 DDK。

用户提供的内部示例配置现已确认：`scripts/build_ohos.sh` 使用 `toolchain/openharmony-toolchain.cmake`，其中 target 为 `aarch64-linux-ohos`，SDK 写死在 `$HOME/hmos_sdk/openharmony/native`，链接 `c++_shared`；`src/CMakeLists.txt` 取示例的 `inc`、`libs/ohos` 并链接 `hiai`。该配置未指定兼容 API，也不能证明其引用的工具已安装。root shell 的 `$HOME` 往往是 `/root`，不能用它推断其他用户目录下 SDK 的位置。

已有解压后的 Command Line Tools 时，可直接配置本项目。将下面两个根目录换成实际目录，随后执行检查和构建；本项目使用 SDK 自带的 toolchain，无需复制内部示例中写死路径的文件：

```bash
export COMMAND_LINE_TOOLS_ROOT='/your/command-line-tools'
export HIAI_EXAMPLE_ROOT='/your/model_run_tool'
export OHOS_NATIVE_SDK="$COMMAND_LINE_TOOLS_ROOT/sdk/default/openharmony/native"
export HIAI_INCLUDE_DIR="$HIAI_EXAMPLE_ROOT/inc"
export HIAI_LIBRARIES="$HIAI_EXAMPLE_ROOT/libs/ohos/libhiai.so"

test -f "$OHOS_NATIVE_SDK/build/cmake/ohos.toolchain.cmake"
test -x "$OHOS_NATIVE_SDK/llvm/bin/clang++"
test -d "$OHOS_NATIVE_SDK/sysroot"
test -f "$HIAI_INCLUDE_DIR/HiAiModelManagerService.h"
test -f "$HIAI_LIBRARIES"
```

全部通过后，按下文“公共检查：CMake”和“指定兼容 API 并构建”继续。若头文件缺失，应使用与这套库匹配的 HIAI 头文件目录。构建脚本也会在启动 CMake 前检查 Native SDK；仅存在 `command-line-tools` 顶层目录不代表 Native 组件齐全。示例库是已知的链接候选，仍需本项目实际链接和板端运行验证。

## 方式一：内部示例所用的 public tar.gz 包

2026-10-22 用户提供的 `model_run_tool` README 要求：先解压 `*-ohos-sdk-public.tar.gz`，再进入其中 `linux` 目录解压组件 zip，最后执行 `scripts/build_ohos.sh`。这确认了包的安装流程，但 README 未指定 SDK 版本/API；需继续核对构建脚本与固件。

如果已经取得与内部示例匹配的 SDK 包，使用这条路径，无需再下载下一节的 Command Line Tools。将 `OHOS_SDK_ARCHIVE` 换成实际文件名，使用新的仓库外目录，避免覆盖已有工具链：

```bash
export OHOS_SDK_ARCHIVE='/your/downloads/matching-ohos-sdk-public.tar.gz'
export OHOS_SDK_INSTALL="${AI_RUN_DIR:?先设置本次输出目录}/ohos-public-$(date +%Y%m%d-%H%M%S)"
mkdir "$OHOS_SDK_INSTALL"
tar -xzf "$OHOS_SDK_ARCHIVE" -C "$OHOS_SDK_INSTALL"
(
  set -eu
  cd "$OHOS_SDK_INSTALL/linux"
  for component in *.zip; do
    test -f "$component"
    unzip -n "$component"
  done
)
find "$OHOS_SDK_INSTALL/linux" -type f -name ohos.toolchain.cmake
```

如果输出路径为 `linux/native/build/cmake/ohos.toolchain.cmake`，配置：

```bash
export OHOS_NATIVE_SDK="$OHOS_SDK_INSTALL/linux/native"
test -f "$OHOS_NATIVE_SDK/build/cmake/ohos.toolchain.cmake"
test -x "$OHOS_NATIVE_SDK/llvm/bin/clang++"
test -d "$OHOS_NATIVE_SDK/sysroot"
"$OHOS_NATIVE_SDK/llvm/bin/clang++" --version
```

若实际路径不同，`OHOS_NATIVE_SDK` 应取 `build/cmake/ohos.toolchain.cmake` 之前的目录。如果已按示例解压到 `$HOME/ohos_sdk` 且检查通过，可直接使用 `$HOME/ohos_sdk/linux/native`，无需再次解压。

然后跳至下文“公共检查：CMake”和“指定兼容 API 并构建”。内部示例的 `out/ohos` 是它的产物目录，不是 SDK 根目录；`scripts/build_ohos.sh` 编译的是 `model_run_tool`，本项目正式 DFlash 程序仍通过 `targets/cpro/scripts/build_hiai.py` 构建。

## 方式二：Command Line Tools 的 Linux zip 包

没有可复用工具链时，从[华为官方下载中心](https://developer.huawei.com/consumer/cn/download/command-line-tools-for-hmos)获取 **Command Line Tools 的 Linux x64 包**。可在能联网的电脑下载后传到服务器；本工程使用其中 Native SDK，无需启动完整 IDE。官方获取与解压说明见 [Obtaining Command Line Tools](https://developer.huawei.com/consumer/en/doc/harmonyos-guides/ide-commandline-get)。版本应与内部 CPro 固件、HIAI 库和已验证示例匹配，不能将“最新版本”视为已验证兼容。

如果下载页提供可直接访问的文件链接，可在服务器下载；将 URL 换成页面实际链接，不能把下载页本身当作 zip：

```bash
export OHOS_SDK_URL='填写官方下载页的 Linux x64 zip 文件链接'
export OHOS_SDK_INSTALL="$AI_RUN_DIR/ohos-sdk-$(date +%Y%m%d-%H%M%S)"
mkdir "$OHOS_SDK_INSTALL"
export OHOS_SDK_ARCHIVE="$OHOS_SDK_INSTALL/sdk.zip"
curl --fail --location --retry 3 \
  "$OHOS_SDK_URL" -o "$OHOS_SDK_ARCHIVE"
sha256sum "$OHOS_SDK_ARCHIVE"
```

若包已经传入服务器，跳过上述下载代码，改用：

```bash
export OHOS_SDK_INSTALL="$AI_RUN_DIR/ohos-sdk-$(date +%Y%m%d-%H%M%S)"
mkdir "$OHOS_SDK_INSTALL"
export OHOS_SDK_ARCHIVE='/your/uploaded/commandline-tools-linux-x64.zip'
sha256sum "$OHOS_SDK_ARCHIVE"
```

两种方式都按下载页给出的摘要校验后，再执行解压：

```bash
unzip "$OHOS_SDK_ARCHIVE" -d "$OHOS_SDK_INSTALL/unpacked"
find "$OHOS_SDK_INSTALL/unpacked" -type f -name ohos.toolchain.cmake
```

仅搜索刚解压的目录。常见布局如下，**以实际找到的文件为准**；SDK 根目录必须满足下列三个检查，不能指向压缩包、`libs/ohos` 或整个 `command-line-tools`。官方 [C++ 链接排障](https://developer.huawei.com/consumer/en/doc/harmonyos-faqs-V14/faqs-compiling-and-building-116-V14)也给出了 `sdk/default/openharmony/native/llvm/bin` 工具路径。

```bash
export OHOS_NATIVE_SDK="$OHOS_SDK_INSTALL/unpacked/command-line-tools/sdk/default/openharmony/native"
test -f "$OHOS_NATIVE_SDK/build/cmake/ohos.toolchain.cmake"
test -x "$OHOS_NATIVE_SDK/llvm/bin/clang++"
test -d "$OHOS_NATIVE_SDK/sysroot"
"$OHOS_NATIVE_SDK/llvm/bin/clang++" --version
```

检查失败先纠正目录或补齐完整 Native 组件。不要只复制 `ohos.toolchain.cmake`，它依赖同包的编译器、系统头文件和库。

## 公共检查：CMake

两种安装方式都需主机 CMake 3.16 或更新版本；服务器已有时直接复用。若没有，且 SDK 提供下列可执行文件，可只在当前 shell 使用它：

```bash
if ! command -v cmake >/dev/null 2>&1; then
  test -x "$OHOS_NATIVE_SDK/build-tools/cmake/bin/cmake" && \
    export PATH="$OHOS_NATIVE_SDK/build-tools/cmake/bin:$PATH"
fi
cmake --version
```

## 指定兼容 API 并构建

`build_hiai.py` 已取消硬编码 API 21。未指定 `--compatible-sdk-version` 时，不传 `OHOS_COMPATIBLE_SDK_VERSION`，采用所选 SDK 自带工具链的行为，**不会自动检测板端 API**。若内部环境要求明确版本，可追加 `--compatible-sdk-version <正整数>`；应同时确认所选 SDK 支持它且板端满足要求。不要通过任意降低此参数掩盖 HIAI 库或系统 ABI 不匹配。

可核对的[官方 toolchain 源码](https://github.com/openharmony/build/blob/961ba684a2c9a1feaa063cbe597ebf04642f297d/ohos/ndk/cmake/ohos.toolchain.cmake#L110-L154)仅在指定兼容版本时向编译 target 追加版本；未指定不等于自动选择板端版本。本机实际安装包的 toolchain 仍是最终依据。

`command.json` 记录请求的 API（未指定为 null）、API 参数策略和 toolchain 文件哈希，实际编译器参数保存在构建目录；这些信息不表示板端兼容性已通过。

Native 包带有 `oh-uni-package.json` 时，可查看其版本/API 元数据；这说明已装 SDK 的版本，不能代替板端固件信息：

```bash
if test -f "$OHOS_NATIVE_SDK/oh-uni-package.json"; then
  cat "$OHOS_NATIVE_SDK/oh-uni-package.json"
fi
```

沿用[上板指南](direct-ge-board.md)中的 `MODEL_PYTHON`、`REPO_DIR`、`GE_BOARD_RUN`。已有上述示例资源配置时保留它们；否则先设置实际匹配的 `HIAI_INCLUDE_DIR` 与 `HIAI_LIBRARIES`，再构建：

```bash
"$MODEL_PYTHON" "$REPO_DIR/targets/cpro/scripts/build_hiai.py" \
  --output "$GE_BOARD_RUN/hiai-build"
cat "$GE_BOARD_RUN/hiai-build/result.json"
```

多个 HIAI 库用冒号分隔。用户提供的示例只显式链接 `hiai`，无需据此额外添加 `hiai_ir` 或 `hiai_ir_build`；C++ 运行库由工具链的 `c++_shared` 配置处理。`HIAI_LIBRARIES` 应选同一套 HarmonyOS ARM64 库；不要仅因目录名为 `lib64` 就选用。若资源 JSON 的 `hiai_libraries` 非空，它优先于环境变量。

结果为 `LINKED` 后按[上板指南第3节](direct-ge-board.md#3-传到板端的文件)传输正式程序和 OMC。`--headers-only` 只能证明头文件编译成功；链接成功仍需板端 `--probe` 和 tiny 数值/生成验证。缺少匹配 SDK 时，可以继续服务器参考验证，但不能把主机或 fake runner 当成板端二进制。
