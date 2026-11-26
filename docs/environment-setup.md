# 环境配置

## Decoder 对比的两机配置

78 运行 310P 参考、CPU golden 和 CPro AIR 导出。35 读取 AIR 包、用目标 DDK 编译并通过 HDC 上板。下面的共享范围指**同一台服务器**。两台服务器可以使用不同的绝对路径。

| 谁维护 | 路径或设置 | 如何使用 |
|---|---|---|
| 78 共享 | CANN 版本目录和匹配的 conda 环境 | 软件可装一次。每个进程仍要加载对应环境。在配置中填写 `DECODER_CANN_SETUP` 指向本站的 `set_env.sh`。310P 和 CPro 导出如需不同安装，分别填 `DECODER_NZ_SETUP`、`DECODER_EXPORT_SETUP`。 |
| 78 共享 | 原 310P 工程 `TARGET_SOURCE`、原 CPro 工程 `CPRO_SOURCE`；模型权重和 tokenizer | 可共用同一版本的原始输入。YAML 的 `model_file` 指向模型目录，`quanted_pth` 指向 W4 分片目录；原生 40 层 prefill 还读取 `embedding_weight_path`、`embedding_scale_path`。给使用者读取文件、遍历父目录的权限。统一入口在自己的结果目录导出；如果自行修改的 310P 驱动往源码目录写文件，应给它单独的工作副本。 |
| 35 共享 | `CPRO_DDK_ROOT`、DDK 内的 OMG、算子注册和配套实现 | 使用已安装本项目算子的同一版 DDK。将 `DECODER_DDK_SETUP` 指向该版本的 `tools/tools_ascendc/set_ascendc_env.sh`，脚本在编译进程里加载。如果它导出正确的 `DDK_PATH`，可以省略 `CPRO_DDK_ROOT`。正在开发不同算子版本时要选各自的安装或注册环境。 |
| 35 共享 | HarmonyOS Command Line Tools（command-line-tools）、`HDC`、`OHOS_NATIVE_SDK` | `HDC` 填 **35 上 hdc 可执行文件**的绝对路径，已在 PATH 时可省略；编译自有 ARM64 runner 才需 Native SDK，此时 `OHOS_NATIVE_SDK` 填含 `build/cmake/ohos.toolchain.cmake` 的 `native/`。日常用板上已有工具运行 decoder 可直接复用它。 |
| 35 共享 | `RUNTIME_CONFIG`、`TBE_DIAGNOSIS` | 可指向团队现有的只读产物。`runtime.json` 和它引用的 assets JSON 需一同可读，并匹配所用固件与 DDK。`TBE_DIAGNOSIS` 指向与所选 DDK 匹配的原诊断目录，保留其中的 `report.json` 及当时记录的绝对路径。编译时生成的适配文件写在本次 case 中。 |
| 按板选择 | `BIP`、`SN`、`CPRO_BOARD_TOOL` | `BIP` 指 HDC 服务端，`SN` 指板卡，`CPRO_BOARD_TOOL` 是**板上**的工具路径。同板可用相同值，共享板端锁会阻止同一时刻重复占用。这些值按实际选板填写。 |
| 各执行用户 | `AI_RUN_DIR`、`DECODER_RUN_ROOT`、`CPRO_PATCHED_SOURCE` | 给每人独立可写的运行空间。同一 case 重跑会覆盖旧结果。手改 CPro 的人应使用自己的可编辑源码目录；团队确认的已补丁源码也可作为共享只读输入。多个操作者都用 root 登录时，按操作者名字建目录。 |
| 各执行用户 | `decoder.env`、个人 YAML、SSH 配置和私钥 | 每人可单独改 `prompts[0]`、模式和设备。复制的是小 YAML，其权重路径继续指向共享文件。35 通过 SSH 拉取时，默认读取本账号的 `~/.config/qwen35-cpro/transfer.json`。可用 `CPRO_TRANSFER_CONFIG` 指向个人文件（权限 600）。登录 78 的账号须能读取该人的 `export/`。 |

**从已有安装开始。** 在两台主机的仓库根目录分别执行一次，把模板中本机需要的路径填好。入口通过 PATH 中的 `python3` 启动（现有环境使用 Python 3.11）。各阶段可以用已配置的 conda 环境名，也可以填共享环境的绝对 prefix；如果新登录的 shell 找不到 conda，在配置中设 `CONDA_EXE=/绝对路径/bin/conda`。两个角色的日常启动都只需要一条命令。

```bash
mkdir -p ~/.config/qwen35-cpro
cp -n targets/cpro/scripts/decoder.env.example ~/.config/qwen35-cpro/decoder.env
```

管理员可在每台主机各放一份只读 Bash 文件（例如 `/srv/qwen35/site/decoder.env`）填写表中的共享项。用户在自己的 `decoder.env` **先**加载本站公共项，再设置个人项。例如 78 上的个人部分可以是

```bash
source /srv/qwen35/site/decoder.env
AI_RUN_DIR=/data/runs/alice/qwen35
CPRO_PATCHED_SOURCE=/data/work/alice/cpro-edit
NZ_CONFIG="$HOME/.config/qwen35-cpro/model.yaml"
TARGET_CONFIG="$NZ_CONFIG"  # 两个工程使用同一份模型 YAML 时
DECODER_MODE=40
DECODER_PHASE=prefill
```

35 选相同的 mode 和 phase。若本机 `AI_RUN_DIR` 与 78 不同，在 35 的个人文件中设 `DECODER_REMOTE_RUN_ROOT=/data/runs/alice/qwen35/target/tests`。它应指向 **78 的 case 父目录**。新建个人运行根时，35 原先默认的 `runtime/runtime.json` 和 `common/tbe-diagnosis/` 可能为空；将 `RUNTIME_CONFIG` 和 `TBE_DIAGNOSIS` 指回现有公共版本即可。35 使用导出包中的模型及外置权重，运行这个入口无需另存 78 的整套原始 checkpoint。

如果多人使用同一登录账号，为每人保留单独的配置目录。启动时可在同一条 bash 命令前指定 `DECODER_ENV=/data/config/alice/decoder.env`。35 初次配置 SSH 且使用自定义路径时，执行 `python3 targets/cpro/scripts/pull_artifacts.py --config /data/config/alice/transfer.json --init 78主机名 35主机名`，并在个人配置中填写相同的 `CPRO_TRANSFER_CONFIG`。已有可用的传输配置可直接复用。

**78** 执行 `bash targets/cpro/scripts/decoder_compare.sh prepare`。**35** 在 78 完成后执行 `bash targets/cpro/scripts/decoder_compare.sh board`。[模式、目录覆盖和输出说明](decoder-compare.md)

## 其它 CPro 构建工具的初始化

下面保留工程级 `env.sh` 和离线词表的配置方法，供需要构建 runner 或使用旧工具的环境使用。

## 1. 恢复工具链

```bash
conda activate custom_pypto
source '/实际DDK根目录/tools/tools_ascendc/set_ascendc_env.sh'
export REPO_DIR='/实际路径/qwen3.5-35B-dflash'
```

## 2. 加载项目配置

仅首次填写路径和设备。多人共用一个 checkout 时，用个人的 `QWEN35_ENV_CONFIG`；已有配置可直接指向它。

```bash
export QWEN35_ENV_CONFIG="${QWEN35_ENV_CONFIG:-$HOME/.config/qwen35-cpro/build.env}"
if ! test -f "$QWEN35_ENV_CONFIG"; then
  export AI_RUN_DIR="/data/runs/alice/qwen35"
  export CANN_ROOT='/实际CANN版本目录'
  export CPRO_DDK_ID='100.520.020.010'
  export OHOS_NATIVE_SDK='/实际SDK目录/openharmony/native'
  export BIP='实际HDC服务器IP:端口'
  export SN='实际设备序列号'
  # 首次创建配置会提示再次加载；下一行source负责加载。
  source "$REPO_DIR/targets/cpro/scripts/env.sh"
fi
source "$REPO_DIR/targets/cpro/scripts/env.sh"
```

## 3. 离线准备词表

在可联网电脑下载[离线词表包（3.52 MB）](https://raw.githubusercontent.com/day8reak/qwen3.5-35B-dflash/main/docs/assets/qwen35-tokenizer-offline.tar.gz)和[SHA256清单](https://raw.githubusercontent.com/day8reak/qwen3.5-35B-dflash/main/docs/assets/qwen35-tokenizer-offline.tar.gz.sha256)，再传到服务器。包内是锁定版本的三个原始词表文件，不含权重或Python依赖。已有source和prepared目录时跳过解压／准备，只设置TOKENIZER_DIR。

```bash
mkdir -p "$AI_RUN_DIR/common/tokenizer-offline"
tar -xzf /实际路径/qwen35-tokenizer-offline.tar.gz \
  -C "$AI_RUN_DIR/common/tokenizer-offline"
(cd "$AI_RUN_DIR/common/tokenizer-offline" && sha256sum -c SHA256SUMS)

export TOKENIZER_DIR="$AI_RUN_DIR/common/tokenizer-offline/source"
python3 "$REPO_DIR/targets/cpro/scripts/prepare_board_tokenizer.py" \
  --source-dir "$TOKENIZER_DIR" \
  --output "$AI_RUN_DIR/common/tokenizer-offline/prepared"

python3 -c 'import numpy, tokenizers; print(tokenizers.__version__)'
cmake --version
```

## 4. 仅缺少tokenizers时

在与服务器相同Python版本、OS和CPU架构的联网环境下载wheel，整个目录传入服务器。不要复制Windows环境默认下载的wheel到Linux。

```bash
# 联网电脑
python3 -m pip download --only-binary=:all: \
  --dest tokenizer-wheelhouse 'tokenizers==0.22.2'

# DDK服务器的custom_pypto环境
python3 -m pip install --no-index \
  --find-links /实际路径/tokenizer-wheelhouse 'tokenizers==0.22.2'
```
