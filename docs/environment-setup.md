# 环境配置

## Decoder 对比的四个环境

同一版仓库提供四个执行入口和一个传输入口。角色决定配置，与主机名无关。一台机器承担多个角色时可以复用软件安装，分别选择 Python 环境。

| 环境 | 同机可共享的只读资源 | 应按执行用户或任务选择 |
|---|---|---|
| CPU、310P golden 生成服务器 | 310P、CANN、匹配的 PyTorch/torch_npu 环境、原始工程 `TARGET_SOURCE`、模型权重和 tokenizer | `NZ_CONFIG` 的 `prompts[0]`、`DECODER_NZ_ENV` 和 `DECODER_NZ_SETUP` |
| CPro AIR 生成服务器 | CANN、TorchAir、CPro 原工程 `CPRO_SOURCE`、模型与 W4 分片 | `TARGET_CONFIG`、`DECODER_EXPORT_ENV`、`DECODER_EXPORT_SETUP` 和个人可编辑的 `CPRO_PATCHED_SOURCE` |
| CPro OMC 生成服务器 | `CPRO_DDK_ROOT` 下的 OMG、原模型算子与 FP16 SiluPatch/SoftplusPatch、匹配的 `TBE_DIAGNOSIS` | `DECODER_COMPILE_ENV`、`DECODER_DDK_SETUP` 和本次编译空间 |
| CPro 板端执行服务器 | HarmonyOS Command Line Tools 的 `HDC`，匹配的 `RUNTIME_CONFIG` 与其引用的 assets 文件 | `DECODER_EXECUTE_ENV`、可选 `DECODER_EXECUTE_SETUP`、HDC 服务 `BIP`、设备 `SN` 和板端 `CPRO_BOARD_TOOL` |

golden 与 AIR 两端使用同一份模型和量化权重。当前 checkpoint 校验要求 `model_file` 和 `quanted_pth` 在这两个环境解析为采集清单记录的绝对路径。整网 prefill 的 AIR 端还需要 YAML 中的 embedding 权重和 scale。OMC 和上板环境从交接的产物读权重。运行已编译 OMC 所需的输入输出描述由 OMC 生成服务器保存。自建 runner 时用到的 `OHOS_NATIVE_SDK` 也可共享。

## 首次配置

在每个环境把模板复制到个人配置目录。按角色命名为 `golden.env`、`air.env`、`compile.env`、`execute.env`。示例为 OMC 生成服务器

```bash
mkdir -p ~/.config/qwen35-cpro
cp -n targets/cpro/scripts/decoder.env.example ~/.config/qwen35-cpro/compile.env
```

入口按以下顺序选取**一份**配置文件：显式 `DECODER_ENV`、本角色文件、公共的 `~/.config/qwen35-cpro/decoder.env`。同机两角色若已经配置了 `decoder.env`，可以继续使用。需要组合多份配置时，在个人文件中用 `source /srv/qwen35/site/common.env` 引入本站公共项。conda 环境可用名称、绝对 prefix 或 `current`；新 shell 找不到 conda 时填写 `CONDA_EXE`。独立的接收脚本使用下游角色配置，SSH 传输需要 Paramiko，可指定 `DECODER_TRANSFER_ENV`。

每位操作者独立设置可写的 `AI_RUN_DIR`、个人 YAML、SSH 凭据及手改源码目录。不同服务器的运行根目录可以不同。多个操作者共用同一登录账号时，给每人单独的路径并用 `DECODER_ENV` 选择个人配置。四个角色的 `DECODER_MODE`、`DECODER_PHASE`、`DECODER_GDN_CONV` 应一致。已有设置 `DECODER_CANN_SETUP`、`DECODER_BOARD_ENV` 和 `DECODER_REMOTE_RUN_ROOT` 可继续使用。共享权重目录及上游运行产物需对接收账号可读。

## 环境间传输

golden 和 AIR 生成成功时，脚本对本次 case 目录递归执行 `chmod -R 777`，然后发布同样权限的 `.handoff-golden.json` 或 `.handoff-air.json`。已有 case 手动修复权限后，在接收方重试即可复用这份数据。

在接收方运行 `decoder_transfer.sh golden|air|omc`。SSH 配置默认为 `~/.config/qwen35-cpro/transfer.json`（权限 600），可用 `CPRO_TRANSFER_CONFIG` 更改。已有 `air` 端点的两机部署可直接沿用。新环境可交互创建配置

```bash
python3 targets/cpro/scripts/pull_artifacts.py --init GOLDEN_HOST AIR_HOST OMC_HOST
```

每个接收环境只需自己会访问的上游端点。可手动配置，如 AIR 服务器接收 golden 用 `{"golden":{"host":"golden.example","port":22,"user":"alice"}}`，支持 SSH 密钥或 agent。上游目录不同于本机时，按产物填写 `DECODER_GOLDEN_REMOTE_ROOT`、`DECODER_AIR_REMOTE_ROOT` 或 `DECODER_OMC_REMOTE_ROOT`，其值是生产端的 **case 父目录**。共享文件系统也可设置对应的 `*_LOCAL_ROOT`，从已挂载的上游读取。磁盘预算应覆盖本机接收的文件、编译产物和 OMG 临时空间；重复接收同一份清单时，脚本复用已收齐的文件。

通过 SSH 接收时，脚本用生成服务器上的 GNU `dd` 分段读取大文件，连接中断会自动重试。若重试后仍退出，在接收服务器再次执行同一条命令即可。接收进度保存在本次 case 的 `.receive-air/`、`.receive-golden/` 或 `.receive-omc/` 下。同一清单下按已收字节数续传，文件收齐后发布产物并清理进度目录。SSH 和 HDC 收发都跳过文件内容的哈希校验。

四环境分步指令和两机组合命令见[Decoder 运行说明](decoder-compare.md)。

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
