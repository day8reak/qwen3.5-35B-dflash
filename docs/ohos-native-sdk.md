# 服务器缺少 OHOS Native SDK

HIAI 程序的交叉编译需要 **Linux 主机可执行的 OHOS 编译器、sysroot 和 CMake toolchain**。`libs/ohos/libhiai.so` 与 `libc++_shared.so` 是链接/运行库，不能替代这些组件。AscendC 环境用于算子编译，也不等于已配置 OHOS C++ 应用工具链。

SDK 只放在服务器；板端使用编译后的程序、OMC 和匹配的运行库。已有成功的 target/draft OMC 与 prepared 保留，无需重跑 AIR/OMG。

## 先复用已有工具链配置

若内部已有能在同一 CPro 固件上运行的 `model_run_tool`，优先查看其 README、CMakeLists、构建脚本中指定的编译器、sysroot、SDK/API 和 C++ 运行库。只有 `libs/ohos` 目录不足以证明 SDK 存在；只有 `file libhiai.so` 显示 `ARM aarch64` 也不足以证明其系统 ABI 与板端匹配。

复用时仍通过本项目的 `OHOS_NATIVE_SDK`、`HIAI_INCLUDE_DIR`、`HIAI_LIBRARIES` 指定资源，不修改共享 DDK。

用户随后提供的 `scripts/build_ohos.sh` 实际将 `CMAKE_TOOLCHAIN_FILE` 指向示例内的 `toolchain/openharmony-toolchain.cmake`，并设置 `INC_PATH=<示例>/inc`、`LIB_PATH=<示例>/libs/ohos`。因此，仅搜索名为 `ohos.toolchain.cmake` 的文件不能排除已有自定义工具链；还需读取这份 `openharmony-toolchain.cmake` 和 `src/CMakeLists.txt`，确认编译器、sysroot、C++ ABI 与链接库。脚本未给出 SDK 版本/API，也不能证明其引用的工具已安装。当前本项目的构建脚本使用 Native SDK 自带的 toolchain，尚不能直接把这个自定义 `.cmake` 文件填入 `OHOS_NATIVE_SDK`。

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

`build_hiai.py` 当前默认 `--compatible-sdk-version 21`，**这不是已确认的 CPro API 版本**。应按内部示例/板端固件确认最低兼容 API，并确认所选 SDK 支持它；旧 SDK 不一定支持 21。不要通过任意降低此参数掩盖 HIAI 库或系统 ABI 不匹配。

Native 包带有 `oh-uni-package.json` 时，可查看其版本/API 元数据；这说明已装 SDK 的版本，不能代替板端固件信息：

```bash
if test -f "$OHOS_NATIVE_SDK/oh-uni-package.json"; then
  cat "$OHOS_NATIVE_SDK/oh-uni-package.json"
fi
```

沿用[上板指南](direct-ge-board.md)中的 `MODEL_PYTHON`、`REPO_DIR`、`GE_BOARD_RUN`。填写实际确认的 API 和库路径：

```bash
export OHOS_COMPATIBLE_API='填写确认的 API 整数'
export HIAI_INCLUDE_DIR="$CPRO_DDK_ROOT/ddk/ai_ddk_lib/include"
export HIAI_LIBRARIES='/your/matching/ohos-arm64/libhiai.so'

"$MODEL_PYTHON" "$REPO_DIR/targets/cpro/scripts/build_hiai.py" \
  --compatible-sdk-version "$OHOS_COMPATIBLE_API" \
  --output "$GE_BOARD_RUN/hiai-build"
cat "$GE_BOARD_RUN/hiai-build/result.json"
```

头文件路径适用于已确认含 `HiAiModelManagerService.h` 的内部布局；其他布局按实际调整。多个 HIAI 库用冒号分隔。`HIAI_LIBRARIES` 应选同一套 HarmonyOS ARM64 库；不要仅因目录名为 `lib64` 就选用。若资源 JSON 的 `hiai_libraries` 非空，它优先于环境变量。

结果为 `LINKED` 后按[上板指南第3节](direct-ge-board.md#3-传到板端的文件)传输正式程序和 OMC。`--headers-only` 只能证明头文件编译成功；链接成功仍需板端 `--probe` 和 tiny 数值/生成验证。缺少匹配 SDK 时，可以继续服务器参考验证，但不能把主机或 fake runner 当成板端二进制。
