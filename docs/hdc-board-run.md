# HDC 自动传输、运行和结果回传

在能通过 HDC 访问 CPro 的**服务器**执行 `targets/cpro/scripts/run_hdc.py`。脚本统一从环境读取设备 ID `SN` 和远程连接地址 `BIP`，每次调用均为：

```bash
hdc -t "$SN" -s "$BIP" ...
```

`-s` 是 HDC 服务端地址，填写你现有连接实际使用的 `IP:端口`，不能只填 IP；`SN` 选择该服务端连接的设备，见[官方远程连接说明](https://github.com/openharmony/docs/blob/master/en/application-dev/dfx/hdc.md#remote-connection)。本脚本使用已有连接，不配置网络转发或启动远端 HDC 服务。

<a id="当前优先重新导出后打通双图执行"></a>

<a id="当前优先诊断已有分区回退产物"></a>

<a id="当前优先本轮-matmul-归约候选上板"></a>
<a id="明确修复后重新导出并执行双图"></a>

## 当前：双图采用可选 MatMul 归约

用户回传 `reduce-mm-replay-20261026-104025`：六个 MatMul 归约小图的 Load、Process、数值检查全部 PASS，编译分区均 NPU=1、CPU=0，实际执行位置仍为 `UNVERIFIED`。此前五项 ND 前补维 IO 拒绝已完成复测。现在用显式 `export --fp16-reductions matmul` 生成新的 target/draft，验证完整链路；默认归约模式仍是 `native`，整数和不符合条件的归约继续保留原实现。

保留 `$GE_PAIR_RUN/prepared`、`$GE_BOARD_RUN/reference.json` 和原专家模式；沿用原 `CANN_ROOT`、DDK、resources/SDK/头文件/链接库配置，以及 `SN`、`BIP`，无需重新 prepare 或增加库参数。新建三个输出目录，四步用 `&&` 串联，失败就不继续后续构建或上板：

```bash
git -C "$REPO_DIR" pull --ff-only
export GE_MODEL_TOOL="$REPO_DIR/targets/cpro/scripts/direct_model_air.py"
export GE_REDUCE_MODEL_RUN="$AI_RUN_DIR/ge-reduce-model-$(date +%Y%m%d-%H%M%S)"
export HIAI_REDUCE_BUILD_DIR="$GE_REDUCE_MODEL_RUN/hiai-build"
export HDC_REDUCE_RUN_DIR="$GE_REDUCE_MODEL_RUN/board"

python3 "$GE_MODEL_TOOL" export \
  --prepared "$GE_PAIR_RUN/prepared" --ge-root "$CANN_ROOT" \
  --fp16-reductions matmul --output "$GE_REDUCE_MODEL_RUN/air" &&
python3 "$GE_MODEL_TOOL" compile \
  --air "$GE_REDUCE_MODEL_RUN/air" \
  --ddk-root "$CPRO_DDK_ROOT" --ddk-id "$CPRO_DDK_ID" \
  --output "$GE_REDUCE_MODEL_RUN/omc" --reject-cpu-fallback &&
python3 "$REPO_DIR/targets/cpro/scripts/build_hiai.py" \
  --output "$HIAI_REDUCE_BUILD_DIR" &&
python3 "$REPO_DIR/targets/cpro/scripts/run_hdc.py" \
  --omc "$GE_REDUCE_MODEL_RUN/omc" \
  --runner "$HIAI_REDUCE_BUILD_DIR/build/qwen35_dflash_hiai" \
  --reference "$GE_BOARD_RUN/reference.json" \
  --output "$HDC_REDUCE_RUN_DIR" --execution-only
```

`build_hiai.py` 必须生成新的正式 `qwen35_dflash_hiai`，使本次执行包含 `97dd374` 的 ND 前补1适配；已刷新过的 `qwen35_hiai_op_probe` 不能代替正式 runner。构建沿用原 resources/环境；若此前显式指定 resources 文件或兼容 SDK 版本，继续使用同一配置，不换 SDK 或库。原 runner 和所有历史结果保留。

新 AIR 每张图都有 `reduction-lowering.json`，其中 `mode` 记录归约模式，`counts` 给出替换/保留计数。tiny/static 预期 target/draft 分别替换421/10个 FP16 归约，保留16/1个整数归约；以这次生成的报告为准。反馈以下文件：

```bash
python3 - <<'PYCOUNTS'
import json
import os
from pathlib import Path

for name in ("target", "draft"):
    path = Path(os.environ["GE_REDUCE_MODEL_RUN"]) / "air" / name / "reduction-lowering.json"
    report = json.loads(path.read_text())
    print(name, report["mode"], report["counts"])
PYCOUNTS
cat "$HDC_REDUCE_RUN_DIR/result.json"
cat "$HDC_REDUCE_RUN_DIR/benchmark.stderr"
```

候选保留归约前已有的 FP16 乘积，Sum 使用全1右权重、Mean 使用 `1/N` 右权重，避免先形成 FP16 总和再缩放。MatMul 累加舍入仍可能不同，六项通过并不能替代全图精度验证；当前不承诺消除超过9个 NPUCL 分区的回退，也不承诺全 NPU 或2倍加速。

若编译仍回退，保留这一次新 AIR/OMC、日志和 `failure.json`，用只读诊断定位，不直接重复编译：

```bash
export GE_REDUCE_INSPECT_RUN="$AI_RUN_DIR/ge-reduce-inspect-$(date +%Y%m%d-%H%M%S)"
python3 "$GE_MODEL_TOOL" inspect-compile \
  --air "$GE_REDUCE_MODEL_RUN/air" --omc "$GE_REDUCE_MODEL_RUN/omc" \
  --graph target --output "$GE_REDUCE_INSPECT_RUN"
cat "$GE_REDUCE_INSPECT_RUN/report.txt"
```

上例对应 target 失败；draft 失败时改 `--graph draft` 并使用新的诊断目录。小图编译/复测、此前 rank 对照及支持表的历史命令保存在[直接 GE 第7节](direct-model-air.md#7-当前现场双图采用可选-matmul-归约)，当前无需重跑。

`--reject-cpu-fallback` 是本项目 compile 包装器参数，不传给厂商 OMG。它遇到主图或归属未知的 NPUCL 分区超限回退时停止，保留已生成的 OMC、编译日志和诊断，写入 `failure.json`，不发布成功 manifest。常量折叠图的 CPU 执行允许保留；没有发现这类回退也不证明所有算子都在 NPU 上。

此模式沿用文件名 `benchmark.json`，但报告的 `mode` 是 `smoke`：强制 target 和 draft 各调用一次，将 target 特征传给 draft，保存两图 Top1 输出。即使 prompt 的下一 token 是 EOS，也会执行 draft。成功标为 `EXECUTION_COMPLETE`；`accuracy`、`performance` 为 `NOT_EVALUATED`，不产生 `comparison.json`。参考 JSON 仍用于身份和输入配置检查；其历史 PASS 不是本次板端精度结论。

程序保留模型哈希、版本、张量数量、格式、大小和 token 范围检查。ND 描述允许原有尾部补1到四维，例如 `[1,S] → [1,S,1,1]`、`[1,S,F] → [1,S,F,1]`；源码另支持逻辑 rank3 `[a,b,c] → [1,a,b,c]`，使用运行库返回的描述初始化张量。新增规则仅扩展 rank3 特征，rank2 token 合同及宽度推导不变；仍不按元素总数放行任意布局。本轮命令在新目录重新构建正式 runner，以实际取得这项兼容。成功和失败都会尝试回传 HiLog，日志缺失单独记录。

当前没有完整模型板端通过证据；这个命令用于取得实际加载/执行结果。去掉 `--execution-only` 后仍是下文的严格 benchmark + 参考比较。

## 服务器准备

首次准备参考与 runner 可查[直接 GE tiny 上板指南的第1、2节](direct-ge-board.md)。当前现场已有原 prepared 和匹配的 `reference.json`，按本文开头可选导出 MatMul 归约双图，并在新目录构建当前正式 `qwen35_dflash_hiai`。

需要的资源为：

| 参数/变量 | 含义 |
|---|---|
| `SN` | 设备 ID，必填 |
| `BIP` | 远程 HDC 服务端连接地址，必填 |
| `HDC` | 可选，主机 HDC 可执行文件路径；默认从 PATH 查找 `hdc` |
| `--omc` | 含成功 `manifest.json` 的 OMC bundle 目录 |
| `--runner` | 正式 HarmonyOS ARM64 程序；不接受主机 fake runner |
| `--reference` | 与这批 OMC 身份匹配的参考 JSON；默认要求 PASS，execution-only 不以历史精度状态作门槛 |
| `--board-cleanup` | 成功后的清理方式：`all`（默认）、`omc`、`none`，见下表 |
| `--output` | 本次服务器结果目录，必须在 `AI_RUN_DIR` 下且尚不存在 |
| `--execution-only` | 只跑 target/draft 各一次并回传，精度与性能后续验证 |

板端须能通过 shell 执行 `sh`、`mkdir`、`chmod`、`sha256sum` 等基础命令，且部署位置允许执行程序与访问 HIAI。脚本自动定位构建记录中的 HIAI 库和 OHOS SDK 的 ARM64 C++ 运行库。

## 后续严格 benchmark 入口

新 OMC 的双图 smoke 通过后，再使用本节严格 benchmark 与参考比较。沿用已有 `REPO_DIR`、`AI_RUN_DIR`、`GE_BOARD_RUN` 和本轮新构建的 runner；`GE_OMC_DIR` 指向本轮编译成功的目录。

```bash
cd "$REPO_DIR"
git pull --ff-only
export PYTHONDONTWRITEBYTECODE=1

export SN='填写设备ID'
export BIP='填写远程HDC服务端IP:端口'
export GE_OMC_DIR="$GE_REDUCE_MODEL_RUN/omc"
export HDC_RUN_DIR="$AI_RUN_DIR/hdc-$(date +%Y%m%d-%H%M%S)"

python3 "$REPO_DIR/targets/cpro/scripts/run_hdc.py" \
  --omc "$GE_OMC_DIR" \
  --runner "$HIAI_REDUCE_BUILD_DIR/build/qwen35_dflash_hiai" \
  --reference "$GE_BOARD_RUN/reference.json" \
  --output "$HDC_RUN_DIR"
```

若 `hdc` 不在 PATH，先设置 `export HDC=/your/host/toolchains/hdc`。无需设置 `HIAI_RUNTIME_LIB_DIR`，无需手动拷贝 C++ 库，也无需传库目录或构建目录参数。

脚本在 `--runner` 相邻的标准构建目录查找 `command.json` / `result.json`（例如 `hiai-build/build/qwen35_dflash_hiai` 对应 `hiai-build/`）。确认构建为 `LINKED`、程序 SHA256 匹配后，检查记录中的链接库 SHA256，并收集这些库所在目录第一层的 `.so` / `.so.*`。不会扫描整个 DDK，也不会从示例目录混入另一份 HIAI 库。

如果同套库目录没有 `libc++_shared.so`，脚本从构建命令记录的 OHOS Native SDK 中定位 ARM64 版本；找不到时会记录未补充，后续由板端 probe 判断装载是否成功。自动收集的库须为 ARM64 ELF；同名库内容冲突时在上传前停止。最终来源、哈希和查找结果记录在 `preflight.json` / `result.json` 中。保留原构建目录、DDK 和 SDK 路径即可；如果这些文件搬迁或变化，会显示具体缺失路径或哈希不匹配，不会悄悄替换为另一套库。

默认部署复用已在本板探针验证的 vendor HCL 候选：排除 `libhiai_hcl_model_runtime.so` 与 `libhiai_hcl_model_runtime_impl.so` 两个 DDK 库，并在进程的库搜索路径中加入 `/vendor/lib64/chipsetsdk`。上传前检查保留文件的 ELF 依赖及 SONAME；若存在对排除库的硬依赖，明确失败，不上传替代空壳库。此策略只影响本次隔离目录和进程，不修改系统文件；实际后端以日志为准。

新构建记录绑定 `command.json` 的 SHA256；旧记录缺少该字段时仍可使用，标记为 `legacy-unbound`，并检查现有程序及库的身份。该检查不保证全部传递依赖与板端 ABI 兼容，最终以 probe 和实际运行结果为准。旧版显式库目录参数仍兼容已有脚本，推荐直接使用上面的四个参数。

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

以下是默认 benchmark 流程；当前执行联调使用上文 `--execution-only`，第5–6步改为一次双图 smoke 及执行报告校验。

1. 检查 OMC 清单与文件哈希、参考权重身份、prompt 和序列长度，检查程序是 ARM64 ELF。
2. 创建新的板端目录，按文件精确上传程序、manifest、清单中的 OMC 模型载荷、附属权重及自动定位的运行库，保留相对目录结构。编译诊断文件留在服务器，不随模型上传。
3. 在板端用 `sha256sum` 核对上传内容。
4. 设置本次运行的 `LD_LIBRARY_PATH=<部署目录>/lib:/vendor/lib64/chipsetsdk:<板端原值>`，执行 `--probe` 并回传 JSON；要求 `npu_supported=true`。
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
| `benchmark.json` | 默认是生成测量；`--execution-only` 时为 mode=smoke 的两图执行输出 |
| `comparison.json` | 默认模式的参考比较；execution-only 不生成 |
| `probe.stderr`、`benchmark.stderr` | 板端错误日志 |
| `probe.exitcode`、`benchmark.exitcode` | 板端程序实际退出码 |
| `hiai-hilog.txt`、`hiai-hilog.stderr`、`hiai-hilog.exitcode` | probe / benchmark 失败后尝试抓取的板端系统日志与抓取状态，独立于推理状态 |
| `calls/` | 每次 HDC 的 argv、输出和返回码 |
| `failure.json` | 失败阶段与原因，仅失败时生成 |

脚本以非零退出码报告失败，并尽可能拉回已生成日志；probe / benchmark 失败时会直接打印已回传且校验成功的 stderr/stdout 末尾及文件路径。断连时部分文件可能无法取回，保留 `calls/` 和板端目录供后续排查。它还检查远端 shell 的独立退出状态，避免把 HDC 客户端返回0误当作板端程序成功。

若第一步 `create-root` 提示 `missing unique remote exit marker`，先看同屏 HDC stdout/stderr 或 `calls/0001-create-root.json`。例如 `[Fail]Device not founded or connected` 表示所选设备未找到或未连接，shell 未执行；当前尚未上传模型或运行推理。用下列只读命令检查连接：

```bash
"${HDC:-hdc}" -s "$BIP" list targets -v
# SN 应与上述列表中的连接标识完全一致，并且设备已连接。
"${HDC:-hdc}" -t "$SN" -s "$BIP" shell echo HDC_OK
```

如果列表为空或设备状态异常，先确认设备是否连接到 `BIP` 指定的 HDC 服务端、板端调试及授权是否就绪。若列表中的连接标识不同，用该标识更新 `SN`；网络连接的标识可能是 `IP:端口`，并非 USB 序列号，见[官方设备选择说明](https://github.com/openharmony/docs/blob/master/en/application-dev/dfx/hdc.md#connecting-to-the-specified-target-device)。确认 shell 输出 `HDC_OK` 后，换一个新的 `HDC_RUN_DIR` 再运行；不需要重新生成 OMC。

## 历史加载接口与运行库排障

以下保留此前加载接口及运行库问题的排障记录。其中重编 runner、沿用 OMC 的命令仅对应当时的问题；归约小图现已通过，当前按本文开头可选导出新的双图，并构建当前正式 runner 执行。

若 `probe` 返回127，先读 `$HDC_RUN_DIR/probe.stderr`。曾出现程序链接 DDK 库，却上传示例中的另一份 `libhiai.so`，导致缺少 `libhiai_ir_build.so` 和 `CheckNPUSupport` 符号。现在自动按构建记录选择库，这一历史库问题本身不要求重建 OMC。

用户随后反馈 `probe.exitcode=0`、`probe.stderr` 为空，但 `benchmark.exitcode=1`；这表示程序探测通过，推理尚未通过。此时应看 benchmark 自己的日志，不再用 probe 日志解释推理失败：

```bash
cat "$HDC_RUN_DIR/benchmark.stderr"
tail -n 60 "$HDC_RUN_DIR/benchmark.stdout"
```

用户补充的错误为 `HIAI CheckModelCompatibility failed: 1`，发生在 target 的独立兼容性检查阶段，尚未执行张量推理。随后提供的内部 `model_run_tool` 示例以相同的 `(HIGH, 0, 1, 0)` 参数创建描述，调用 `SetModelPath` 后直接 `Load`，没有独立调用 `CheckModelCompatibility`。先前框架据此改用文件加载；后续 HCL 日志显示还必须显式传入模型缓冲区（见下节），现以 `Load` 返回状态判断能否加载；模型哈希、运行库版本、NPU 查询及后续张量数量/形状/格式检查继续执行。API 检查返回错误不能直接推断 OMC 本身无效，也不能据此次源码修正认定上板成功。

文件加载及下述 HCL 缓冲区修正都需要重新交叉编译推理程序；无需重新生成 AIR、OMC 或参考。沿用原来的编译环境变量与资源配置，在服务器执行：

```bash
cd "$REPO_DIR"
git pull --ff-only
export HIAI_BUILD_DIR="$AI_RUN_DIR/hiai-load-$(date +%Y%m%d-%H%M%S)"
python3 "$REPO_DIR/targets/cpro/scripts/build_hiai.py" \
  --output "$HIAI_BUILD_DIR"
```

编译成功后执行下面的四参数命令：

```bash
export HDC_RUN_DIR="$AI_RUN_DIR/hdc-load-$(date +%Y%m%d-%H%M%S)"
python3 "$REPO_DIR/targets/cpro/scripts/run_hdc.py" \
  --omc "$GE_OMC_DIR" \
  --runner "$HIAI_BUILD_DIR/build/qwen35_dflash_hiai" \
  --reference "$GE_BOARD_RUN/reference.json" \
  --output "$HDC_RUN_DIR"
```

新脚本会在失败时直接打印已回传日志的末尾；完整内容仍保留在结果目录。模型加载错误还会注明模型名称、路径和运行库版本。当前只完成主机流程与 HIAI 测试替身验证，修正后的真实板端加载与推理等待实测。

### 实际 Load 仍返回 1

用户在上述修正后报告：`HIAI Load failed: 1 [model=qwen35_target, ... runtime=800.631.120.010]`。重新过滤完整 HiLog 后，多次出现同一组加载错误：

```text
CreateModelManager: DDK pipe is HCL.
CreateLocalBuffer: data != nullptr && size > 0 false, return nullptr.
AiModelManagerClientImpl::Load: CreateLocalBuffer return null, load failed
```

这将当前故障定位到模型缓冲区为空或长度为零。只设置 `SetModelPath` 未给此 HCL 加载分支提供有效数据；尚不能据此判断 OMC 内容、算子或固件兼容性。代码现保留路径，并将 OMC 只读映射后显式 `SetModelBuffer(data, size)`，检查设置状态和描述中的地址/长度，再调用 `Load`。映射和描述均保留到 `UnLoadModel` 及客户端销毁之后；不修改模型类型、设备类型或计算精度，也不跳过加载结果与张量验证。

蓝区头文件及[华为公开接口声明](https://github.com/huaweicodelabs/HiAI-Foundation/blob/master/IRBuild/ddk/ai_ddk_lib/include/HiAiModelManagerType.h)的 `SetModelBuffer` 长度是 `uint32_t`。空文件、非普通文件、超过 `UINT32_MAX` 字节的单个 OMC 会提前明确失败，禁止长度截断。这是当前 tiny OMC 的加载修正，尚不提供已验证的完整 35B 大文件加载路线；只读映射也不代表 HIAI 内部不会再分配或复制数据。

此修正要按上节命令重新构建 HIAI 二进制；AIR、OMC、参考文件可以沿用。主机测试要求实际提供正确的模型字节，并在推理、卸载和客户端析构时继续检查原始缓冲区，覆盖 target/draft 部分加载失败、空文件及超过 4 GiB 的稀疏文件拒绝。主机测试及新增缓冲区接口的蓝区 OHOS 编译检查均不能代替内部 DDK 链接和真实板端复测。本地蓝区头文件缺少项目原先使用的 `CheckNPUSupport`，完整适配器未在该蓝区版本编译通过；此限制不通过删除内部 DDK 的 NPU 探测检查来绕过。

脚本在 probe / benchmark 失败时自动尝试 `hilog -x`，使用短超时，将快照及命令自身的 stderr/退出码保存到本次板端目录，然后按已有 SHA256 流程回传；命令参数含义见 [OpenHarmony HiLog 文档](https://github.com/openharmony/docs/blob/master/zh-cn/application-dev/dfx/hilog.md#命令行说明)。不会清空系统日志、修改日志级别或启动持续抓取任务。HiLog 不可用、无权限、超时或回传失败只记录诊断失败，不覆盖原始模型错误，不将推理失败变成成功。

本板端拒绝组合 `-x` 与 `-z`，返回 `Mutlti commands can't be used in combination [CODE: -31]`。此前组合命令有误，现已改为单独的 `hilog -x`：读完当前缓冲区后退出，完整输出保存在文件中，终端只显示有限摘要。如果抓取程序把该参数错误写到 stdout 且返回 0，也会标记为不可用，不能当成成功取得 HIAI 日志。

`hiai-hilog.txt` 是共享日志缓冲区快照，可能包含其他进程或较早的日志；需要按时间、进程和模型名关联，不能把所有匹配行都当成本次错误。程序扫描整个已校验快照，优先选择最新观察到的 `qwen35_dflash_hiai` 会话候选，按 PID、日期和 foundation 初始化分隔，保留头部、首条 E/F、Restore 失败前后及尾部，去重后最多60行。同 PID 的 MUSL 加载错误也纳入，其他服务进程不混入。候选身份尚未独立绑定本次执行，首条 E 可能是已处理的库回退，摘要不自动判定根因。无法解析会话时沿用最后60条关键词候选；完整快照留在服务器输出目录。不会仅检查文件最后 64 KiB，避免后续系统流量掩盖之前的模型错误。摘要排除 `HDC_LOG` 传输日志，`NPU` 按完整单词匹配，并单独匹配 `NPUCL` 等组件，避免误匹配 `input`。候选条数为 0 表示本次筛选没有找到相关内容，不表示模型运行成功或没有底层错误。

仅日志抓取和筛选修改无需重编译；上述 HCL 模型缓冲区修正需要重编译二进制，OMC 可以沿用。更新后使用上面的一条四参数命令，换新的 `HDC_RUN_DIR` 即可；不再传旧命令中的 `--lib-dir`，自动按构建记录准备库。

已有失败现场也可以立即手动取日志：

```bash
hdc -t "$SN" -s "$BIP" shell "hilog -x" > "$HDC_RUN_DIR/hiai-hilog-manual.txt"
grep -n -v 'HDC_LOG' "$HDC_RUN_DIR/hiai-hilog-manual.txt" | \
  grep -i -B 3 -A 8 -E \
  'hiai|ai_fmk|cpucl|npucl|(^|[^[:alnum:]_])npu([^[:alnum:]_]|$)|qwen35_dflash_hiai|CheckModelCompatibility' | tail -n 160
```

最初宽泛筛选只显示新目录创建和文件传输，还混入另一个模型的传输日志；收紧筛选后才发现前面的 HCL 空缓冲区错误。板端日志时间为 `03-06`，而主机目录包含 `20261023`；两端时钟未对齐，不能直接用主机时间排除板端日志。核对现场时以本次 `upload-plan.json` / `result.json` 的 `board_root` 为准，避免重跑先前目录后查看另一次运行的日志。

如果系统日志仍不能定位，再用内部 `model_run_tool` 加载同一份 `target.omc`，保持相同的 HIAI 库集合做对照。内部示例也加载失败时优先核对 OMC/DDK/运行库/固件；示例能加载时再比较客户端初始化。当前尚无这项对照结果。

清理前若发现归属标记不匹配、目录被替换成符号链接，脚本会拒绝删除。清理命令失败或无法确认删除结果时也返回非零，`board_cleanup.status` 为 `FAILED`；已回传的服务器文件仍保留，板端目录可能完整保留、部分清理或已删除，需按日志确认。只有运行与比较已通过而清理失败时，`device_validation` 可为 `PASS`，整体 `status` 仍为 `FAILED`。

`comparison.json` 为 PASS 表示这次普通/DFlash token 与停止原因匹配参考，并通过既有产物身份检查。当前 tiny 测试不证明完整35B量化、内存、质量或性能，也不能仅由 OMC 成功运行推断所有节点都在 NPU 上执行。脚本开发使用模拟 HDC 验证流程；真实 CPro 运行结果由你执行后提供。


## 2026-10-26：完整图分区超限回退已定位

当前执行优先的 smoke 已使用排除 DDK HCL 两库的部署。用户补齐 PID25399 的日志后，终止链条为 `input1 dtype should be fp32, but now it is fp16` → `op:MatMul not support` → `compatibleHelper CheckCompatibility failed by cl CPUCL` → `Recompile failed` → `Load failed`。另外记录了 activation.mode9 和 Cast 类型检查失败。前序缺少已排除 DDK HCL 库的报错之后，运行时已打开板端 `libcpucl_itf.so`，不能将该报错当成新的缺库修复目标。

这次 target 在恢复时遇到 CPUCL 算子兼容性阻断，尚未执行 IO 适配、Process 或 draft。用户随后核对 manifest SHA256 与失败记录相同，并提供该 OMC 的 target OMG 第110650行：`Partition of type: NPUCL is more than 9,`，紧接着 `will rollback to CPU`。这已定位主图编译时的分区超限回退；较早第19509、31879行的 CPU 分区属于 `constNodeGraph`。以下只读命令用于核对历史现场：

```bash
sha256sum "$GE_OMC_DIR/manifest.json"
grep -n -E 'partition type|run[_ -]*by[_ -]*cl|PreGraphSplit|CompileGraph.*(target|SubGraph)' "$GE_OMC_DIR/target/omg.log" | tail -n 100
cat "$GE_OMC_DIR/target/command.json"
```

已有 OMC 不会随代码或 runner 更新获得图改动。用户随后已按 `78d94ef` 重新 export，新的主图窗口 L91094–L127615 再次回退；随后已完成 `inspect-compile` 与六项 rank/keep_dims 对照，六个 MatMul 候选也已全部通过 Load、Process 和数值检查；当前按本文开头可选导出新的 MatMul 归约双图，重新构建正式 runner 后执行 smoke。保持原 prepared、FP16 和已通过小图验证的部署策略；即使后续未触发回退，也要检查最终分区与板端执行，不能宣称已修复或全 NPU。
