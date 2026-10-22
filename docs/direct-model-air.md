# Target / Draft 直接 GE 构图 → AIR

新增入口为 `targets/cpro/scripts/direct_model_air.py`。target 和 draft 的图定义分别在 `ge_target.py`、`ge_draft.py`；通过主机 GE C++ 接口保存 AIR，生成端不导入 PyTorch、TorchNPU、TorchAir，也不初始化 NPU。原有 PyTorch 模型、参考、`export-air` 与 `export-tiny-air` 全部保留。

用户已反馈 FP16 Add 小图的 AIR 和内部 OMG 转 OMC 成功，以及本页 tiny target/draft 的 AIR 生成成功。修正 Select 广播后，最新 tiny target OMG 日志推进到 `SoftplusV2` 缺少形状推导函数，并出现 CPUCL 常量输入缓冲区大小不匹配。对应源码修正及重试命令见第 3 节；真实 OMG 转换和上板仍待验证。源码数值检查不代表内部 OMG 已支持全部算子。

## 实现范围

target 覆盖 40 层完整结构定义：门控全注意力、部分 RoPE、chunk-64 GDN 和卷积、动态 TopK 路由、全部专家与共享专家、最终归一化、八层特征拼接及完整词表 Top1。draft 覆盖六层结构：目标特征投影、共享 embedding/head、动态 anchor、滑窗因果注意力与最后一层双向块注意力、RMSNorm、RoPE、SwiGLU 和 15-token 输出。

保持现有固定 S、batch 1、纯文本、前缀重算、HIAI ABI v2。静态图计算所有专家，不是稀疏 MoE 性能实现。Python 浮点标量采用 GE 属性保留原值，避免提前转 FP16 常量造成额外舍入；所有浮点张量仍为 FP16。原始 BF16/FP32 模型的质量对比是另一道未完成门槛。

**本次直接 GE 路径先提供 FP16 正确性基线，尚未映射 DOPT 的压缩整数权重算子。** `prepare` 生成的 `.bin` 是 FP16，不是 INT4。完整 35B 只能显式使用 `--float-baseline` 做服务器转换诊断，不能作为 32GB 板机部署包。原有量化代码保留；后续需要补充直接 GE 的原生量化算子映射，再验证压缩存储、质量和峰值内存。

## 1. 服务器：先准备结构小模型

在之前成功生成 Add AIR 的环境执行。保留已有 `CANN_ROOT`、`AI_RUN_DIR`；输出目录使用新的名称。这里的 NumPy 随机权重与旧 `cli demo` 即使种子相同也不同，后续参考必须使用本页的 `reference-tiny`。

```bash
cd "$REPO_DIR"
git pull --ff-only
export PYTHONDONTWRITEBYTECODE=1
export GE_MODEL_TOOL="$REPO_DIR/targets/cpro/scripts/direct_model_air.py"
export GE_PAIR_RUN="$AI_RUN_DIR/ge-pair-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$GE_PAIR_RUN"

python3 "$GE_MODEL_TOOL" prepare --tiny --sequence-length 8 \
  --output "$GE_PAIR_RUN/prepared"
```

准备只需要 Python 和 NumPy。成功状态为 `PREPARED`，产生 target/draft 的 GE 图描述、常量、C++ 生成器源码和按哈希锁定的 FP16 权重，不会把图描述文件冒充 AIR。生成器使用紧凑的二进制图描述，完整 target 的数十万节点不会展开成巨型 C++ 源文件。

tiny 为 target 4 层、draft 2 层、hidden 16、词表 64、block 4；它保留 GDN/MoE/两类注意力结构，没有可读文本质量意义。

## 2. 服务器：生成两张 AIR

```bash
export GE_AIR_DIR="$GE_PAIR_RUN/air"
python3 "$GE_MODEL_TOOL" export \
  --prepared "$GE_PAIR_RUN/prepared" \
  --ge-root "$CANN_ROOT" \
  --output "$GE_AIR_DIR"
```

使用与你已跑通的小图相同的 CANN GE 头文件和真实库。脚本编译并运行两个主机生成器；每张图保存前和加载后都检查实际计算图节点。成功时 `air/manifest.json` 为 `EXPORTED`，包含 `target/target.air` 和 `draft/draft.air`。失败保留阶段、`build.log` / `export.log` 和 `failure.json`，重试 `export` 只需换新输出目录，可复用 `prepared`。

如果旧版报 `BuildGraph: tensor_desc is null`、随后报输入不包含 Data，原因是生成器缺少 Data 算子工厂注册。修复版补齐该注册，并在构图前检查工厂原型。已有 `prepared` 的权重和图配置可直接复用；导出时会重新生成当前版本的 C++ 与图描述。保留旧失败目录，执行：

如果旧修复版报 `Data factory input x status=4294967295`，且日志已显示 `Register creator v2 for Data`，说明工厂已注册，失败来自额外检查误用了 `TryGetInputDesc`。GE 9.1 的该接口要求形状维度数大于零，不能用于检查工厂原型是否声明了端口。当前版本已移除这项误判，保留工厂及实际计算图检查；同样按下面命令更新并复用 prepared。

```bash
cd "$REPO_DIR"
git pull --ff-only
export GE_AIR_DIR="$GE_PAIR_RUN/air-datafix-$(date +%Y%m%d-%H%M%S)"
python3 "$GE_MODEL_TOOL" export \
  --prepared "$GE_PAIR_RUN/prepared" --ge-root "$CANN_ROOT" \
  --output "$GE_AIR_DIR"
```

新日志先显示 `Checking GE Data factory...`，随后进入 `SetInputs` 和计算图检查；只有两张图均保存、回读成功，最终才输出 `EXPORTED`。失败报告同时保留前几个错误和日志尾部，完整日志仍在 `export.log`。实际修复效果以服务器重试结果为准。

默认 tiny 内嵌权重。需要单独测试内部 OMG 对外部权重的支持时，在新的准备目录加 `--external-weights`，再导出、转换。完整模型自动使用 `FileConstant`；该方案依据 [TorchAir 官方外部权重实现](https://github.com/Ascend/torchair/blob/e1118e8b0fc1b95190ab53b732c0872989550e20/python/torchair/_utils/export_utils.py)，但本入口不依赖其 Python 包。AIR 中含绝对权重路径，转换完成前不要移动或清理 AIR bundle。

## 3. 服务器：使用内部 OMG 转 OMC

切回已经 source 内部 AscendC 配置脚本的终端，恢复同一 `REPO_DIR`、`GE_MODEL_TOOL`、`GE_PAIR_RUN`、`GE_AIR_DIR` 和 `AI_RUN_DIR`。`GE_AIR_DIR` 必须指向刚刚成功导出的目录。不重复拼接脚本已有环境变量。下面的 DDK 版本是你此前反馈的版本，若安装包已更新，请填写实际版本；脚本不会自动检测该版本号。

```bash
export CPRO_DDK_ROOT="$(realpath "${DDK_PATH:?先加载内部 DDK 环境}")"
export CPRO_DDK_ID=100.520.020.010
export GE_OMC_DIR="$GE_PAIR_RUN/omc"

python3 "$GE_MODEL_TOOL" compile \
  --air "$GE_AIR_DIR" \
  --ddk-root "$CPRO_DDK_ROOT" --ddk-id "$CPRO_DDK_ID" \
  --output "$GE_OMC_DIR"
```

沿用现有 `compile_air`：`--framework=1 --target=omc --platform=kirinx90 --weight_data_type=FP16`，两个 OMG 子进程各有独立的 run 内临时目录和日志。成功为 `COMPILED`，设备状态仍是 `PENDING`。这一步可能暴露内部 DDK 的算子、属性或布局差异，应按具体 `omg.log` 修图。

### 内部 OMG 兼容性修正与重试

最新日志（进程 157178）明确报 `v52/SoftplusV2` 缺少 infershape 函数。当前模型固定使用 FP16、`beta=1`、`threshold=20`，因此生成器改发原生 `Softplus`（x/y，无额外属性）；[官方 GE 文档](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/programug/graphdevg/atlasag_25_0009.html)有该算子的构图用法。穷举全部 63,488 个有限 FP16 输入，理想函数的阈值版本与非阈值版本舍入到 FP16 后逐位一致。其他 beta/threshold 设置会明确拒绝，不擅自改变公式；真实内部 kernel 数值仍需上板比较。

同一日志还报 CPUCL 输入 32 字节、预期 64 字节，且 `fp16Flag:0`。这与小型 FP16 常量算术进入 FP32 执行路径相符，但日志未给出该失败节点，暂不把具体归属当作定论。生成器现在只提前计算小常量的 `scalar_add/scalar_mul/exp/neg`，保留每个原始节点的 FP16 输出舍入，避免把归一化偏移和 A_log 衰减参数计算交给这条 CPUCL 路径。原始 prepared 权重及其哈希不变；不展开专家矩阵、切片或 reshape，不加载完整 35B。单张量上限 64 KiB、累计缓存/读取预算 8 MiB。

此前的 Select 和 RMSNorm 修正仍保留；这次 RMSNorm 的常量 Power 会进一步变成直接存储的 FP16 结果：

2026-10-22 用户的 tiny target 日志中，`v7/Select` 报 `x1Shape.GetDimNum() != x2Shape.GetDimNum()`；源码对应四维条件和两个标量分支。现已在 target/draft 的每个 Select 前显式广播三个输入到相同形状，保留真正的选择操作及 `-inf`/NaN 语义。

同一日志的 `v23/Add` 对应 RMSNorm 的 `1 + weight`，CPUCL 报输入类型不一致。保存并回读的 AIR 描述中二者均为 FP16，现有日志不能确定 OMG 内部发生了什么类型变化；源码将此处改为单输入 Power 的 `shift=1`。全部有限 FP16 输入的本地检查与原加法逐位一致，真实转换仍须重试。节点编号会随重新构图变化，应以新日志为准。

这次改动影响 AIR 内容，不能只重跑旧 AIR 的 `compile`。可复用旧 `prepared`，不用重新下载或生成权重；保留原目录并运行：

```bash
cd "$REPO_DIR"
git pull --ff-only
export GE_AIR_DIR="$GE_PAIR_RUN/air-compat-$(date +%Y%m%d-%H%M%S)"
export GE_OMC_DIR="$GE_PAIR_RUN/omc-compat-$(date +%Y%m%d-%H%M%S)"

python3 "$GE_MODEL_TOOL" export \
  --prepared "$GE_PAIR_RUN/prepared" --ge-root "$CANN_ROOT" \
  --output "$GE_AIR_DIR"

python3 "$GE_MODEL_TOOL" compile \
  --air "$GE_AIR_DIR" \
  --ddk-root "$CPRO_DDK_ROOT" --ddk-id "$CPRO_DDK_ID" \
  --output "$GE_OMC_DIR"
```

只有 export 输出 `EXPORTED` 后再执行 compile。新版本的转换失败会直接显示 graph、阶段、日志路径、首错和尾部，并在 `failure.json` 保留退出码或超时值；draft 失败时不会发布成功的双图 manifest。摘要识别内部 OMG 的 `E/AI_FMK` 等格式，不将 W/I 级别的 `failed to load` 插件提示列作首错；也会跳过本次已确认的 shape 回显及明确返回 SUCCESS 的空配置消息，保留真正的 CPUCL 缓冲区错误。完整日志保留所有消息；尾部不应用上述过滤。

若仍失败，提供终端摘要，或提取完整日志的尾部；target 通过而 draft 失败时将下列路径中的 target 改为 draft：

```bash
tail -n 100 "$GE_OMC_DIR/target/omg.log"
```

## 4. 服务器参考与板端运行

在已有 PyTorch/Transformers 的 CPU 参考环境执行。此命令才需要 PyTorch，会沿用项目的 CPU 后端隔离；它读取本次 prepared 的实际权重，而非重新生成另一份随机模型。切换环境后再设置下面的解释器与项目路径；后续复用板端指南时保持这些变量。

```bash
export MODEL_PYTHON="$(command -v python3)"
export PYTHONPATH="$REPO_DIR/model:$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
python3 "$GE_MODEL_TOOL" reference-tiny \
  --prepared "$GE_PAIR_RUN/prepared" \
  --prompt-token-ids 1,2,3 --max-new-tokens 4 \
  --output "$GE_PAIR_RUN/tiny-reference.json"
python3 - <<'PY'
import json, os
from pathlib import Path
run = Path(os.environ["GE_PAIR_RUN"])
with (run / "tiny-prompt.json").open("x") as out:
    json.dump({"prompt_token_ids": [1, 2, 3]}, out)
PY
```

参考应为 `PASS`。双图输出与现有 HIAI runner 一致：target 为 INT32 `[1,8]` 和 FP16 `[1,8,32]`，draft 为 INT32 `[1,3]`。板端编译、传输、运行及回传比较沿用 [tiny 验证指南第 2–5 节](tiny-validation.md#2-服务器编译程序并传文件)，将路径对应如下；不要使用那个页面第 1 节重新生成旧随机权重。

| 旧指南路径 | 本次路径 |
|---|---|
| `$AI_RUN_DIR/tiny-omc` | `$GE_OMC_DIR` |
| `$AI_RUN_DIR/tiny-reference.json` | `$GE_PAIR_RUN/tiny-reference.json` |
| `$AI_RUN_DIR/tiny-prompt.json` | `$GE_PAIR_RUN/tiny-prompt.json` |

传板端的是整个 OMC bundle、交叉编译的 `qwen35_dflash_hiai` 和板端缺少的匹配运行库。复用旧指南时，把服务器 `$GE_OMC_DIR/` 的内容放到板端 `$BOARD_ROOT/models/tiny-omc/`，保持板端命令中的目录名一致。AIR、prepared 权重、图描述、C++ 生成器、原始 safetensors 和 Python 环境都留在服务器。参考与 OMC 的 `source` 包含所有 FP16 权重内容的聚合哈希，可拒绝混用不同随机权重的报告。

## 5. 完整 35B：先查结构，再做服务器诊断

无需加载权重即可核对完整结构和空间估算：

```bash
python3 "$GE_MODEL_TOOL" plan --full --sequence-length 8 \
  --output "$GE_PAIR_RUN/full-plan"
```

完整 target 的 693 个参数定义已经与锁定模型结构/检查点索引核对。此项不代表实际权重加载或量化完成。完整 target/draft 的唯一 FP16 权重约 70.09 GB；原始下载之外还要为准备权重、AIR 和 OMC 留出空间。准备阶段按块读取 [safetensors 格式](https://github.com/safetensors/safetensors#format)，转换 BF16→FP16，无需把整个模型载入主机内存；转换/编译工具自身仍可能需要大量 RAM。

只在需要排查完整模型转换且服务器资源充足时执行：

```bash
python3 "$GE_MODEL_TOOL" prepare --full --float-baseline --sequence-length 8 \
  --target-weights "$QWEN35_TARGET_WEIGHTS" \
  --draft-weights "$QWEN35_DRAFT_WEIGHTS" \
  --output "$GE_PAIR_RUN/full-prepared"
python3 "$GE_MODEL_TOOL" export \
  --prepared "$GE_PAIR_RUN/full-prepared" --ge-root "$CANN_ROOT" \
  --output "$GE_PAIR_RUN/full-air"
```

不要将这份未量化 FP16 全模型传到 32GB 板机。完成 tiny 双图 AIR/OMC/HIAI 验证后，下一步是原生量化映射、全权重数值验证和内存实测；原 PyTorch/TorchAir 路径随时可继续联调。
