# HDC 自动传输、运行和结果回传

在能通过 HDC 访问 CPro 的**服务器**执行 `targets/cpro/scripts/run_hdc.py`。脚本统一从环境读取设备 ID `SN` 和远程连接地址 `BIP`，每次调用均为：

```bash
hdc -t "$SN" -s "$BIP" ...
```

`-s` 是 HDC 服务端地址，填写你现有连接实际使用的 `IP:端口`，不能只填 IP；`SN` 选择该服务端连接的设备，见[官方远程连接说明](https://github.com/openharmony/docs/blob/master/en/application-dev/dfx/hdc.md#remote-connection)。本脚本使用已有连接，不配置网络转发或启动远端 HDC 服务。

## 服务器准备

先完成[直接 GE tiny 上板指南的第1、2节](direct-ge-board.md)：保留成功的 OMC bundle、同一批 prepared 生成的 `reference.json`，以及正式链接的 `qwen35_dflash_hiai`。无需重新生成 AIR/OMC。

需要的资源为：

| 参数/变量 | 含义 |
|---|---|
| `SN` | 设备 ID，必填 |
| `BIP` | 远程 HDC 服务端连接地址，必填 |
| `HDC` | 可选，主机 HDC 可执行文件路径；默认从 PATH 查找 `hdc` |
| `--omc` | 含成功 `manifest.json` 的 OMC bundle 目录 |
| `--runner` | 正式 HarmonyOS ARM64 程序；不接受主机 fake runner |
| `--reference` | 与这批 OMC 权重匹配、状态为 PASS 的参考 JSON |
| `--lib-dir` | 可选，可重复；上传其中的 `.so` / `.so.*` 库，保留库文件名 |
| `--board-cleanup` | 成功后的清理方式：`all`（默认）、`omc`、`none`，见下表 |
| `--output` | 本次服务器结果目录，必须在 `AI_RUN_DIR` 下且尚不存在 |

板端须能通过 shell 执行 `sh`、`mkdir`、`chmod`、`sha256sum` 等基础命令，且部署位置允许执行程序与访问 HIAI。`--lib-dir` 应包含匹配的 HarmonyOS ARM64 HIAI/C++ 运行库及所需依赖，不要传服务器的 x86_64 CANN 库。

## 一条命令完成上板步骤

沿用已有 `REPO_DIR`、`AI_RUN_DIR`、`GE_OMC_DIR`、`GE_BOARD_RUN`。如果你用上一轮独立构建命令生成了程序，使用 `$HIAI_BUILD_DIR/build/qwen35_dflash_hiai`；如果按上板指南第2节构建，则用 `$GE_BOARD_RUN/hiai-build/build/qwen35_dflash_hiai`。

```bash
cd "$REPO_DIR"
git pull --ff-only
export PYTHONDONTWRITEBYTECODE=1

export SN='填写设备ID'
export BIP='填写远程HDC服务端IP:端口'
export HIAI_RUNTIME_LIB_DIR='/your/model_run_tool/libs/ohos'
export HDC_RUN_DIR="$AI_RUN_DIR/hdc-$(date +%Y%m%d-%H%M%S)"

python3 "$REPO_DIR/targets/cpro/scripts/run_hdc.py" \
  --omc "$GE_OMC_DIR" \
  --runner "$HIAI_BUILD_DIR/build/qwen35_dflash_hiai" \
  --reference "$GE_BOARD_RUN/reference.json" \
  --lib-dir "$HIAI_RUNTIME_LIB_DIR" \
  --output "$HDC_RUN_DIR"
```

若 `hdc` 不在 PATH，先设置 `export HDC=/your/host/toolchains/hdc`。板端已提供全部匹配运行库时可省略 `--lib-dir`；需要补充其他库目录时再加一个 `--lib-dir`。脚本不自动搜集或上传整个 DDK。

默认板端目录为 `/data/local/tmp/qwen35-dflash-<唯一标识>`，实际路径记录在结果中。需要自选位置时加 `--board-root /data/local/tmp/你的新目录`，该目录必须尚不存在。每次重跑都换新的 `--output`。成功完成运行、结果与日志回传及参考比较后，脚本自动删除本次创建的板端目录；失败时保留供排查。服务器结果与已有目录均保留。

以后需要保留大体积的独立权重文件时，在运行命令末尾加 `--board-cleanup omc`：

| 参数 | 成功后板端保留内容 | `board_cleanup.status` |
|---|---|---|
| `--board-cleanup all`（默认） | 删除整个本次部署目录 | `REMOVED` |
| `--board-cleanup omc` | 仅删除 manifest 中 target/draft 的 OMC 文件；保留独立权重、所有目录、manifest、程序、库及日志 | `OMC_FILES_REMOVED` |
| `--board-cleanup none` | 全部保留 | `RETAINED` |

`omc` 按清单中的确切文件路径删除，不用通配符扫描权重文件。已嵌入 OMC 内部的权重会随 OMC 一起删除；只有独立文件形式的权重能够单独保留。保留目录不代表当前脚本支持复用旧目录，下一次运行仍需新部署目录。

固件标识默认读取板端 `param get const.ohos.fullname`。若内部系统没有这个属性，可显式追加 `--firmware-id '实际固件版本'`。该值用于报告身份记录，不能用“unknown”等占位符代替实际版本。长时间运行可用 `--timeout-seconds` 调整单次 HDC 调用超时，默认3600秒。

## 自动执行的内容

1. 检查 OMC 清单与文件哈希、参考权重身份、prompt 和序列长度，检查程序是 ARM64 ELF。
2. 创建新的板端目录，按文件精确上传程序、manifest、清单中的 OMC/附属文件及指定运行库，保留相对目录结构。清单中包含的编译诊断文件也可能随 bundle 一并传输。
3. 在板端用 `sha256sum` 核对上传内容。
4. 设置本次运行的 `LD_LIBRARY_PATH=<部署目录>/lib:<板端原值>`，执行 `--probe` 并回传 JSON；要求 `npu_supported=true`。
5. 根据 manifest 和 probe 自动生成 benchmark 命令，执行普通生成与 DFlash 一致性检查，并各进行3次预热、10次测量。prompt 与生成长度直接取参考 JSON。
6. 回传运行结果、stdout/stderr 和退出码，调用项目现有 `compare` 与服务器参考比较。
7. 仅在上述步骤全部成功后，按 `--board-cleanup` 执行清理；删除前验证本次运行归属标记，结果记录在 `result.json` 的 `board_cleanup.mode/status`。

环境变量仅作用于本次板端脚本，不修改系统环境文件。生成的脚本和 HDC 调用记录保存在服务器结果目录；AIR、prepared、原始权重、Python、CANN、SDK 留在服务器。

## 结果与失败排查

```bash
cat "$HDC_RUN_DIR/result.json"
cat "$HDC_RUN_DIR/comparison.json"
```

主要文件：

| 文件 | 内容 |
|---|---|
| `result.json` | 本次自动流程状态与身份信息 |
| `runtime-probe.json` | 板端 HIAI 版本和 NPU 能力 |
| `benchmark.json` | 普通/DFlash 生成结果与测量报告 |
| `comparison.json` | 与参考的 token、停止原因和产物身份比较 |
| `probe.stderr`、`benchmark.stderr` | 板端错误日志 |
| `probe.exitcode`、`benchmark.exitcode` | 板端程序实际退出码 |
| `calls/` | 每次 HDC 的 argv、输出和返回码 |
| `failure.json` | 失败阶段与原因，仅失败时生成 |

脚本以非零退出码报告失败，并尽可能拉回已生成日志；断连时部分文件可能无法取回，保留 `calls/` 和板端目录供后续排查。它还检查远端 shell 的独立退出状态，避免把 HDC 客户端返回0误当作板端程序成功。

清理前若发现归属标记不匹配、目录被替换成符号链接，脚本会拒绝删除。清理命令失败或无法确认删除结果时也返回非零，`board_cleanup.status` 为 `FAILED`；已回传的服务器文件仍保留，板端目录可能完整保留、部分清理或已删除，需按日志确认。只有运行与比较已通过而清理失败时，`device_validation` 可为 `PASS`，整体 `status` 仍为 `FAILED`。

`comparison.json` 为 PASS 表示这次普通/DFlash token 与停止原因匹配参考，并通过既有产物身份检查。当前 tiny 测试不证明完整35B量化、内存、质量或性能，也不能仅由 OMC 成功运行推断所有节点都在 NPU 上执行。脚本开发使用模拟 HDC 验证流程；真实 CPro 运行结果由你执行后提供。
