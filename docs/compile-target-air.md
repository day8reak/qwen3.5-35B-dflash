# 从实际 AIR 输入生成 OMG 编译参数

适用于已经分别导出的 40 层静态 Prefill64、Decode1 AIR。使用本次导出的输入名，不复用旧图的 `argN` 或固定输入数。脚本只读主图 `Data`／`RefData` 的输出描述，排除 `Const`、`FileConstant` 和子图内部输入；源图顺序不代表编译后的运行槽位。

```bash
python "$REPO_DIR/targets/cpro/tools/air_input_shape.py" \
  --air "$TARGET_SOURCE/glm_export_air_cpro_40block_64_quant/export.air"
```

stdout 只输出 `name:dim,dim;name:dim,...`，便于赋给 shell 变量；输入数量输出到 stderr。可加 `--report /新的路径/inputs.json` 留存逐输入描述，父目录需存在。工具使用仓库附带的 GE protobuf 描述，不导入 PyTorch、TorchAir、torch_npu，也不加载权重。若导出器 schema 不同，传入 `--torchair-pb2 /实际的/ge_ir_by_protoc_XXX_pb2.py`；该文件只作 AST 数据解析，不执行。

动态维度、零维度、缺失或空 shape 会报错，不能自动把它们猜成序列长度。脚本不会把 T=55 的静态 AIR 改成 T=64，也不会把单分支静态 AIR 变成 Prefill／Decode 双分支图。

以下命令针对原导出目录：两个 `glm_export_air_cpro_40block_*_quant` 目录及其原始外部权重保持相对位置。在原 Target 根目录执行 OMG，以解析原始 FileConstant 相对路径。已规范化成 `air/weight/` 的 bundle 应从其 AIR 所在目录编译。

```bash
(
set -e
: "${REPO_DIR:?设置当前仓库根目录}"
: "${TARGET_SOURCE:?设置原 Target 工程根目录}"
: "${CPRO_DDK_ROOT:?设置本次使用的 DDK 根目录}"
export AI_RUN_DIR="${AI_RUN_DIR:-${REPO_DIR}-run}"
export PYTHONDONTWRITEBYTECODE=1
source "$CPRO_DDK_ROOT/tools/tools_ascendc/set_ascendc_env.sh"

# 使用仓库现有的 DDK 环境设置，调用 master/omg ELF。
# 环境辅助函数不运行编译；下面两次 "$OMG" 是完整编译调用。
omg_environment=$(PYTHONPATH="$REPO_DIR/model${PYTHONPATH:+:$PYTHONPATH}" \
  python - "$CPRO_DDK_ROOT" <<'PY'
import os, shlex, sys
from qwen35_moe_dflash.toolchain.compile import omg_environment
omg, env = omg_environment(sys.argv[1])
for key in sorted(set(os.environ) - set(env)):
    print('unset ' + key)
for key, value in env.items():
    if os.environ.get(key) != value:
        print('export ' + key + '=' + shlex.quote(value))
print('OMG=' + shlex.quote(str(omg)))
PY
)
eval "$omg_environment"

cd "$TARGET_SOURCE"
for phase in prefill decode; do
  if test "$phase" = prefill; then seq=64; else seq=1; fi
  air="$TARGET_SOURCE/glm_export_air_cpro_40block_${seq}_quant/export.air"
  out="$AI_RUN_DIR/$phase/omc"
  mkdir -p "$out"
  input_shape=$(python "$REPO_DIR/targets/cpro/tools/air_input_shape.py" --air "$air")
  printf '%s\n' "[$phase] --input_shape=$input_shape"
  printf '%s\n' "$input_shape" > "$out/input-shape.txt"

  if "$OMG" \
    --model="$air" \
    --output="$out/model" \
    --framework=1 \
    --target=omc \
    --platform=kirinx90 \
    --limit_partition_size=false \
    --save_weights_as_external_data=true \
    --input_shape="$input_shape" \
    > "$out/omg.log" 2>&1; then
    printf '%s\n' "$phase: OMG 返回成功，输出目录 $out"
  else
    tail -n 60 "$out/omg.log"
    exit 1
  fi
done
)
```

这里不传 `--dynamic_dims`：两个 AIR 分别走自己的静态导出分支。编译后一起保留 OMC 及其外置权重，再读取新 OMC 的输入输出 ABI；源 AIR 输入检查和 OMG 返回成功都不代表整网精度或性能已验证。

本地已用合成 ModelDef／GraphDef 检查常量排除、输入改名、81 输入图、动态／缺失形状拒绝、报告覆盖保护及无第三方包 CLI。实际 40 层 AIR／DDK 编译由目标环境执行。

## 直接查看 OMC 的输入输出

使用已初始化的 CPro DDK 环境，设置 `CPRO_DDK_ROOT` 后只需传入 OMC 路径：

```bash
python "$REPO_DIR/targets/cpro/tools/parse_model_json.py" "$OMC"
```

脚本自动调用 OMG 转换元数据，打印逐槽位名称、dtype、shape、逻辑字节数，以及 `--input_shape`、`--input_type`、`--output_shape`、`--output_type`。OMC 输入默认使用本项目已核对的 `ge-proto` dtype 枚举；其他 schema 可用 `--dtype-map` 覆盖。

这是独立的单文件标准库脚本，可以单独复制运行，不需要 `PYTHONPATH`、Torch 或项目 Python 包。DDK 依次从 `--ddk-root`、`CPRO_DDK_ROOT`、`DDK_PATH`、`TOOLCHAIN_HOME` 获取，未配置时在 PATH 查找 OMG；也可传 `--omg /实际路径/master/omg`。沿用该 DDK 的环境初始化方式。默认转换超时为 120 秒，可用 `--timeout` 调整。

转换在临时目录使用相对的 `input.omc`、`model.json`，避免 DDK 的绝对 JSON 路径校验问题。模型和相邻外置权重通过符号链接读取，不复制大权重；临时目录在结束后自动清理。有 `AI_RUN_DIR` 时使用其 `target/omc-inspection`，否则使用系统临时目录。需要保留解析结果时加 `--output-dir /新的报告目录`，生成下述五个报告文件。

动态模型显示 JSON 中的描述，不自动选择 Prefill／Decode 档位；缺失输出名称时仍打印按槽位排列的 shape，不编造运行时名称。图输入顺序与 NetOutput 输出顺序仍需与实际运行槽位核对。

## 已有 model.json 的解析方式

OMC 可通过 OMG 的 `--mode=1` 导出 JSON。若 `--json` 的绝对路径被 DDK 拒绝，从 OMC 目录执行并传相对文件名，例如 `--om=实际模型名.omc --json=model_io.json`。外置权重保持在原位置。

`targets/cpro/tools/parse_model_json.py` 是独立标准库脚本，单文件复制到服务器也能运行，不需要安装项目包、Torch 或 NPU 扩展。对当前已核对的 GE protobuf dtype 数值，执行：

```bash
export OMC_JSON="$(dirname "$OMC")/model_io.json"
python "$REPO_DIR/targets/cpro/tools/parse_model_json.py" "$OMC_JSON" \
  --dtype-enum ge-proto \
  --output-dir "$AI_RUN_DIR/target/omc-inspection/io"
```

输出目录需是新目录；重跑到同一目录时添加 `--overwrite`，只覆盖下面五个报告文件。省略 `--output-dir` 时只打印结果。生成：

- `io.json`：完整输入输出描述、按槽位排列的 `input_shapes` / `output_shapes`，以及四个 CLI 参数；`input_order_candidates` 保存图输入列表、Data.index 和 JSON 节点顺序。
- `inputs.csv`、`outputs.csv`：名称、图中索引、shape、dtype、格式、逻辑字节数。
- `summary.txt`：可读表格和候选参数。
- `args.sh`：`INPUT_SHAPE`、`INPUT_TYPE`、`OUTPUT_SHAPE`、`OUTPUT_TYPE` 四个 shell 变量；未确定的项目为空。

字符串类型如 `DT_FLOAT16` 可直接解析；数字 dtype 默认不猜。`--dtype-enum ge-proto` 使用 TensorDescriptor 的 protobuf 枚举（1=FP32、2=FP16、7=INT32、8=INT64），不是 C++ GE/HIAI 的枚举。不同 schema 可通过 `--dtype-map /path/dtypes.json` 提供显式数值映射，例如 `{"1":"FP32","2":"FP16","8":"INT64"}`。

脚本只选择一个根图，排除常量和内部子图。输入优先按完整的 `GraphDef.input` 列表排列；缺少有效列表时，才用完整、连续且唯一的 `Data/RefData.attr.index`。编译 OMC 的 Data.index 可能保留源图编号，因此两者不同不再阻断 `input_type`，报告保留 `input_order_conflict=true` 和两套映射。表格同时显示选定槽位、Data.index、Graph.input 槽位。

可用 `--input-order-source graph-input` 或 `data-index` 显式选定已有完整映射；`json-order` 仅供已核对 JSON 节点顺序的情况，不会被自动选中。也可用 `--input-order '节点A,节点B,...'` 指定已核对的全部输入节点顺序。多个根图需加 `--graph graph_1` 或报告中的 JSON 路径。更新脚本后可直接重用原报告目录：

```bash
python "$REPO_DIR/targets/cpro/tools/parse_model_json.py" "$OMC_JSON" \
  --dtype-enum ge-proto --input-order-source graph-input \
  --output-dir "$IO_DIR" --overwrite
```

输出按单个 NetOutput 的数据端口排列，忽略控制边。名称依次读取描述中的 name、完整 NetOutput 输入名称映射，再尝试模型输出名称属性及 `src_name/src_index`。后者按[公开 GE GetOutputDescInfo 的名称规则](https://github.com/Ascend/graphengine/blob/d2ab1c2819c90e61d98b5f88699bcd6b6de4423c/src/ge/graph/load/new_model_manager/davinci_model.cc)生成候选名称，报告记录 `output_name_basis`；这不等同于查询当前 DDK 的运行时名称。

输出名称缺失或含有 shape CLI 分隔符时，仍输出全部 `output_shapes_by_slot`，只把命名的 CLI 参数标为 `output_shape_cli=UNAVAILABLE`，不会再将已解析的 shape 报成 `UNRESOLVED`。可用 `--output-names '实际名称A,实际名称B,...'` 补充。`status` 表示张量描述是否解析完整，`cli_status` 单独表示四个命名／位置参数能否全部生成。静态模型通常不需要传运行工具的 shape 覆盖参数；工具帮助中的 shape 参数用于 flexible enum shape 场景。

这些是编译图给出的候选参数，`runtime_order_verified=false`，不能自动证明 HIAI 的实际槽位顺序。输入文件列表必须与实际运行槽位对应；逻辑字节数不包含布局 padding 或分配对齐。动态维度的字节数保留为空，不替换成 1。

本地已检查标准库独立执行、图索引排序、数值 dtype 映射、主图边界、缺失与动态 shape、多图选择、控制边、输出名称缺失和 shell 转义。真实 DDK JSON 与设备输入绑定仍需目标环境核对。

针对 88 输入、81 输出的整网结构，回归覆盖两套输入顺序冲突、INT64[1] 的 8 字节描述、显式顺序选择、输出名称元数据和重复生成报告。修改解析工具不需要重新导出 AIR 或编译 OMC。

## 原 CPro 四层 S=64 对照

本节的诊断脚本按两两共用主机编排：golden 与 AIR 在同一台服务器，OMC 编译与板端执行在另一台。四个环境独立部署时，完整 decoder 使用[分步入口](decoder-compare.md)。

`original-air` 在 CPro AIR 生成服务器直接运行 `CPRO_SOURCE` 下的 `export_glm_model_qwen3_5_moe.py`。原量化 wrapper 在图内把 INT8 embedding 乘以 FP32 scale，执行 L0–L3，按 `output_pos` 选行，再过 final norm 和 LM head，返回 logits 和各层 cache。另一个入口 `air` 接收 310P 的 FP16 embedding，只导出 L3 的全部 64 行 hidden。

脚本在 `config=...` 后显式传入 `quant_bits=4` 选择 A8W4。支持位宽选择的原导出器默认用 8 位权重。对 L0 的 `in_proj_qkv`，W4 加载器将权重打包为 INT32 `[2048,1024]`，将分组权重 scale 整理为 FP32 `[8,8192]`。激活的逐 token scale 仍为 FP32 `[64,1]`。

如需单独核对 CPro AIR 生成服务器的 `torch_npu` 是否接受这组 shape 和 dtype，可在读取模型权重前运行下面的 meta 检查。通过时会打印 `torch.Size([64, 8192]) torch.float16`。若报 `scale's type should be torch.int64 in A8W4`，需核对当前 CPro 导出环境对分组权重 scale 的 dtype 要求。

```bash
conda run --no-capture-output -n qwen3_5_cpro python -c '
import torch, torch_npu
def meta(shape, dtype):
    return torch.empty(shape, device="meta", dtype=dtype)
y = torch_npu.npu_quant_matmul(meta((64,2048), torch.int8), meta((2048,1024), torch.int32),
                               meta((8,8192), torch.float32), pertoken_scale=meta((64,1), torch.float32),
                               output_dtype=torch.float16)
print(y.shape, y.dtype)'
```

CPro AIR 生成服务器导出，用现有 valid19 采集准备输入。可用 `ORIGINAL4_REFERENCE` 指定含 `manifest.json` 的采集目录。脚本打印图内乘法在主机重算后舍入到 FP16 与 310P 层入口的逐位差异数。运行时选择 `output_pos=[18]`，初始 cache 为零。`conv_state_pos` 默认按已记录的 valid64 格式推定为 `[19, 0 × 127]`，可用 `ORIGINAL4_CONV_CONTROL` 指向实测的 128 元素 INT32 文件。

```bash
cd "${REPO_DIR:?}" && git pull --ff-only origin main
bash targets/cpro/scripts/decoder4_t64.sh original-air
```

CPro OMC 生成服务器使用现有 DDK、HDC、板上工具和运行配置。脚本拉取 AIR，编译一份带外置权重的 OMC，按编译图的输入顺序运行 HIAI V2，并取回各输出。控制量和 embedding 的数据类型按原版接口生成。默认目录是 `$AI_RUN_DIR/target/tests/decoder4-original-t64-01`，本次运行的 `replay-*/io.json` 和 `result.json` 记录槽位与结果文件。两台机器可分别设置 `ORIGINAL4_RUN`，目录不同时在 CPro OMC 生成服务器设置 `ORIGINAL4_REMOTE_AIR` 为 CPro AIR 生成服务器的 AIR 目录。

```bash
cd "${REPO_DIR:?}" && git pull --ff-only origin main
bash targets/cpro/scripts/decoder4_t64.sh original-board
```

### 四层原版、改版与 310P 的精度对比

三个模型入口的区别如下。每个四层模型都连续执行 L0、L1、L2 的 linear attention decoder 和 L3 的 full attention decoder，包括每层的 MoE 和残差。

| 项目 | 已跑通的 `original-air` | 旧诊断入口 `air` | 本次 `aligned-air` |
|---|---|---|---|
| 输入 | INT8 embedding、FP32 embedding scale、位置、mask、长度、ConvState 控制及各层初始 cache | 310P 的 FP16 embedding、位置、mask、长度和初始 cache | 沿用原版 wrapper 和全部输入，板端按原版成功记录逐槽核对实际输入 SHA-256 |
| 输出 | `output_pos` 选行，经 final norm、LM head 返回 FP32 logits 和八个状态 | L3 全部 64 行 FP16 hidden | 同原版，S=64、valid19 时取下标 18 的 248320 个 logits |
| W4 投影与 RMSNorm | `quant_bits=4`，本次核对的原版 `utils.py` 从 NZ 模型导入 `QLinear` | 显式指定 `CproQLinear`，适配 scale 与 GDR 导出类型，并预计算 RMSNorm gamma | 选用 CPro QLinear，在 W4 的 GE 入口使用 FP16 权重 scale 和逐 token scale；RMSNorm 沿用原版 |
| GDN 历史更新 | `LinearAttentionConvstateUpdate` 读取原控制张量，随后 conv1d | 拼接、切片和 conv2d | 使用原版历史更新和控制张量，卷积和门控处保留 FP16 `SiluPatch`。三个 GDR 的衰减输入和状态输出按已验证的单层接口导出 |
| L3 RoPE 与 MoE | 原实现 | FP32 RoPE、位置 gather、专家量化 reshape 及门控 dtype 适配 | 沿用这些内部改动。L3 attention 的 sigmoid 保留原实现，共享专家和 W4 路由专家的 SiLU 使用 `SiluPatch` |
| L3 K/V 回写 | 更新传入的 K/V，再返回两个 cache | 在副本上更新 K/V，作为计算 hidden 的中间值 | 沿用原版两路 K/V 回写与返回顺序 |

**CPro AIR 生成服务器导出改版并准备四层 310P 参考。** 脚本从原版 AIR 清单恢复导出器和 YAML 路径，逐文件校验并复用既有输入，包括 ConvState 控制量。私有源码或 YAML 移动后可分别指定 `CPRO_SOURCE`、`TARGET_CONFIG`。命令会打印 `silu-source-*` 路径，该目录的 `source.diff` 可直接 review。导出时核对原版和改版 AIR 的输入、输出顺序、状态来源和可见的 shape/dtype。可用 `ORIGINAL4_CAPTURE` 指向已有的四层 310P 采集目录，内含 `manifest.json`。默认先查找已有采集，缺少时才运行 310P 首次 prefill。给 CPro OMC 生成服务器传输的参考包只含本次需要的张量和清单。

```bash
cd "${REPO_DIR:?}" && git pull --ff-only origin main
bash targets/cpro/scripts/decoder4_t64.sh aligned-air
```

**CPro OMC 生成服务器编译一份四层 OMC，上板一次并比较。** 使用已注册 FP16 输入和 FP16 输出版本 `SiluPatch` 的 X90 DDK。如果 CPro OMC 生成服务器保留了成功的原版 replay，脚本还会核对两份 OMC 的端口和输入文件哈希。缺少原版 replay 时，脚本按改版 OMC 的编译槽位读取导出包中的输入。默认目录为 `$AI_RUN_DIR/target/tests/decoder4-original-t64-01-aligned-silu-softplus-patch-fp16`。如两台机器的目录不同，CPro OMC 生成服务器可用 `ALIGNED4_REMOTE_AIR` 和 `ORIGINAL4_REMOTE_GOLDEN` 指定 CPro AIR 生成服务器的改版 AIR 和四层参考包路径。

```bash
cd "${REPO_DIR:?}" && git pull --ff-only origin main
bash targets/cpro/scripts/decoder4_t64.sh aligned-board
```

CPro OMC 生成服务器已有改版 AIR 时，直接执行 `aligned-board` 即可。此入口让 OMG 省略 `--platform`，保留 `--target=omc`，并从当前 DDK 环境继承 `SOC_VERSION`。控制台和 `compile.log` 记录实际命令以及该变量的值。编译复用上一步的 AIR 和权重，本次 OMC、板端结果分别写入改版目录的 `no-platform/omc/` 和 `no-platform/replay-*/`。若只编译，使用 `aligned-compile`。

如内嵌和外置权重的四层模型都在 `OpUpdateKernelArgs` 加载失败，可在 CPro OMC 生成服务器用保留的 AIR 核对单层已经验证的端口。将 `ALIGNED4_RUN` 设为原四层运行目录，执行以下命令。

```bash
bash targets/cpro/scripts/decoder4_t64.sh aligned-audit
```

`MISMATCH` 会列出具体 W4 或 GDR 节点和端口。先在 CPro AIR 生成服务器为 `ALIGNED4_RUN` 选择新目录，再执行 `aligned-air`。新导出核对 36 个普通 W4 投影、3 个 GDR、原版输入输出边界和 14 处 FP16 SiluPatch。在 CPro OMC 生成服务器选择同一新目录执行 `aligned-board`。已有 AIR 报 `MATCH` 时，保留原产物，结合失败 replay 的服务线程日志进一步定位。

若 V2 在加载四层模型时打印 `Initialize model manager client failed`，先在 CPro 板端执行服务器上读取那次 replay 的 `window.log` 和上传清单。`FAILED_REPLAY` 取报错中的 `result.json` 所在目录。以下命令需在 `custom_pypto` 环境执行。

```bash
python targets/cpro/tools/split_stage_log.py --four-layer-replay "$FAILED_REPLAY"
```

若同一服务线程出现 `OpUpdateKernelArgs failed`，可用已导出的 AIR 单独测试内嵌权重。命令核对已有回执、DDK 路径、OMG 参数、SOC_VERSION、设备以及逐个具名输入的 SHA-256，在 `no-platform/embedded/` 编译和运行一份四层 OMC。两种权重模式由 OMG 分别编译。通过加载后脚本继续推理并比较九个输出。如果仍在同一处失败，可对照两次 replay 的服务线程日志。

```bash
bash targets/cpro/scripts/decoder4_t64.sh aligned-board --retry-load "$FAILED_REPLAY"
```

CPro OMC 生成服务器的 OMG 编译失败时，在同一环境执行 `bash targets/cpro/scripts/decoder4_t64.sh aligned-log` 即可读取最近一次编译的错误摘录和完整日志路径。对仅替换 SiLU 的实验使用 `original-log --silu-patch`，对原版使用 `original-log`。这些命令直接读取已保存的日志。

若错误为 `SiluPatch getformatconfigs failed` 且子图出口报告 `outputTensorFormats ... is 0`，在 CPro OMC 生成服务器执行 `bash targets/cpro/scripts/decoder4_t64.sh aligned-diagnose`。它直接读取四层 AIR 中每个 SiLU 的 x/z 形状、类型和布局，打印失败节点的上下游、OMG 的实现选择日志，以及当前 DDK 注册文件中的端口。随后用四层编译入口在相同固定长度下编译 v3 单算子与复制了失败节点属性和端口描述的单算子；两者相同就只编一次。小图也报 `EMPTY_OUTPUT_FORMAT` 可把问题缩小到该编译环境下的单算子；小图 `COMPILED` 而四层失败，应进一步查整图分区的出口和跨分区消费者。完整记录在 `no-platform/silu-diagnosis-*/`。

**CPro OMC 生成服务器一次编译多个小图探针。** `ALIGNED4_RUN` 指向本次四层运行目录。以下命令读取已有 AIR 和 OMG 日志，首次运行时自动生成单算子对照。使用同一套 DDK 时，再次执行可复用本次诊断中小图字节相同、已确认的结果。

```bash
export ALIGNED4_RUN="${AI_RUN_DIR:?}/target/tests/decoder4-original-t64-01-aligned-silu-patch-native-shape-fp16"
bash targets/cpro/scripts/decoder4_t64.sh aligned-diagnose-next
```

对原形状的 L0 卷积分支，命令分别编译下列对照。某个小图的 OMG 退出失败后会继续运行其他小图。

| 要验证的因素 | 比较的小图 |
| --- | --- |
| 张量秩 | 相同元素数的 `[8192,64]` 和 `[1,8192,64]`，各自单独输出及接相同属性的静态 ND Cast |
| AIR 端口元数据 | 同形状下，标准 ND 描述与四层图的 SiluPatch 端口及属性，分别直接输出和接源图消费者 |
| 卷积上下游 | 读取源图实际相连的 Reshape 或 Cast，并用同 shape、同 dtype 的 Data 对照。如 AIR 的直接生产者是可确定输入形状的一元 Squeeze，还会测试 Squeeze 单独、接 SiluPatch、再接 Cast |
| 门控分支 | `[2048,128]` 的 SiluPatch 单独输出、通过源图的 FP16→FP32 Cast 接原 FP32 Mul，并用同 dtype 的 Data 替换 SiLU 做对照。RMSNorm 输出用同形状的 FP32 Data 接入原槽位 |

汇总文件是命令打印的 `silu-diagnosis-*/probes.json`。`COMPILED` 表示 OMG 成功并生成 OMC。`EMPTY_OUTPUT_FORMAT` 表示该小图复现了 SiluPatch 的同一条格式错误。`OTHER_FAILURE` 需读对应的 `compile.log`。`SKIPPED` 记录未能从源图确定的连线。`format_failure_contrasts` 列出一边编译成功、另一边复现错误的成对结果。若所有小图均成功，定位范围仍包括整图分区及未保留的上游上下文。历史一维用例继续使用其下游 Reshape 对照。`SILU_DIAGNOSIS` 可指定已有诊断目录，`aligned-diagnose-report` 可只读历史结果。

**检查两个已复现的子图出口。** 在 CPro OMC 生成服务器使用相同的 `ALIGNED4_RUN` 执行 `bash targets/cpro/scripts/decoder4_t64.sh aligned-format-check`。`gate-mul` 和 `output-cast-reshape` 已在子图出口复现空格式列表，对应的两个 FP32 Data 对照均编译成功。该命令读取最近的 `probes.json`，列出当前 DDK 的算子注册、所选配置目录中出现两个算子名字的文件，以及 DDK 内能找到的 `FindOpFormatConfig` 源码位置。磁盘上出现名字仅说明文件包含该项；查询是否使用它仍需结合实现判断。

脚本先在当前 DDK 重编这两个保留的小 AIR。若仍复现同一错误，分别在 `SiluPatch.z`、其直接消费者的输入、或两端同时补上静态 shape 和 ND 描述，保留原连线及 dtype；字节等价的变体只编一次。结果写在 `format-lookup-*/result.json`。两个原小图均成功时可以重试原四层编译；两处均有成功的端口变体时，把对应的描述带入四层候选再验证。若某个小图在所有变体下都报相同错误，将该小图的 `export.air` 和 `compile.log` 交给 DDK 的 NPUCL 格式转换维护者。

在 CPro OMC 生成服务器上执行 `bash targets/cpro/scripts/decoder4_t64.sh aligned-format-trace`，可以复用同一目录的小 AIR，在相同 DDK 下分别显式设置 `--use_origin_format=true` 和 `false`。第四个探针沿用门控 Mul，只把 SiluPatch 节点替换为单算子生成器的静态定义，可检验四层导出的自定义节点是否为复现所必需。结果保存在 `format-trace-*/result.json`。若环境支持 strace，命令还会独立记录成功和失败门控用例中 OMG 及其子进程实际打开的配置和 DDK 库。打开某个文件只能证明编译期间访问过它。小图通过编译后，还需验证四层 OMC 的实际 I/O 布局和板端精度。

修改或重装 SiluPatch host 后，在 CPro OMC 生成服务器上运行 `bash targets/cpro/scripts/decoder4_t64.sh aligned-format-host`。选择带有原 `probes.json` 的 `ALIGNED4_RUN`。命令比较两份已安装的算子注册表，列出其他声明 FP32 输出的算子，再从新编的小图 OMC 读取实际输出端口。两表和保留 AIR 的输出 dtype 一致时复查门控 Mul，发现 INT32 项时另编端口声明一致的 INT32 单算子。可选的 `SILU_HOST_SO` 指向新构建的 host 库，用于和编译进程树中观测到的库比对 SHA256。结果写入 `host-check-*/result.json`。库的内容匹配只能确认该版本被加载，具体调用了哪个回调还需要回调日志。

旧版 FP32 输出算子在 CPro OMC 生成服务器的实测结果中，两份注册表均声明 `SiluPatch.z` 为 ND FP32，单算子 OMC 的输出为 `FP32[2048,128]`，同一个算子连接原门控 Mul 时在格式转换阶段失败。执行 `python targets/cpro/tools/silu_format_source.py "$H/result.json"`，其中 `H` 为该次 `host-check-*` 目录。命令只读已有记录，核对 AIR 哈希，打印成功和失败小图的日志、两表中的 AddCustom 端口，以及加载记录中包含对应报错文本的 ELF 和哈希。有同版本 DDK 源码时，可追加 `--source` 和源码目录或 `format_converter_pass.cc` 路径，打印查表和取输出格式的代码。

**单独比较已有结果。** 若只想先看成功原版的精度，在 CPU、310P golden 生成服务器执行 `bash targets/cpro/scripts/decoder4_t64.sh original-golden` 准备参考，然后在 CPro 板端执行服务器执行下方第一条。改版也已上板时执行第二条。两条命令都只读已保存的板端输出。可用 `ORIGINAL4_REPLAY` 和 `MODIFIED4_REPLAY` 选择指定的成功记录。

```bash
bash targets/cpro/scripts/decoder4_t64.sh original-compare
bash targets/cpro/scripts/decoder4_t64.sh aligned-compare
```

终端和脚本打印的 `comparison-*/comparison.json` 给出原版对 310P、改版对 310P、改版对原版的分项结果。logits 使用 310P 在 **L3 token18 hidden 上运行 final norm 和 LM head** 的实测值并比较 top1。每层 ConvState 和 recurrent state 比较完整返回张量。L3 K/V 比较第 0 页的有效位置 0–18，另外检查未写入的后续页。相对 L2 和逐元素超差比例在终端按百分数显示，JSON 保存原始比值。

如需量出 **仅替换 SiLU** 的影响，CPro AIR 生成服务器和 CPro OMC 生成服务器分别运行 `original-air --silu-patch` 和 `original-board --silu-patch`，均通过同一个 `decoder4_t64.sh`。该组合使用独立目录，复用同一原版结果和参考。默认 `aligned` 组同时包含表中 L3 RoPE 和 MoE 的内部改动。

### SoftplusPatch 与 GDN 衰减

新导出的 `aligned-air` 在 L0、L1、L2 各调用一次 `SoftplusPatch`。保持 `a.float() + self.dt_bias` 的计算顺序，随后转成连续 FP16（T64 的形状为 `[1,64,32]`），调用 `torch.ops.npu.softplus_patch`。输出 `z` 转 FP32，再与 `-exp(A_log)` 相乘。算子和 GDR 共用 npu Meta 注册方式；导出检查实际 x/z 类型、两个 Cast、消费端和节点数，记录在 `air/manifest.json` 的 `softplus_patch.nodes`。编译机同时需要 FP16 版本的 SiluPatch 和 SoftplusPatch。独立的原版及 SiLU 对照仍可从原入口运行。L0 分模块的重跑命令见 [GDN 逐算子对比](guides/gdn-layer-t64.md)。

### SiluPatch 在四层图中的调用

导出脚本把 `silu_patch` 的注册放在 `npu_chunk_gated_delta_rule` 旁边，两者共用 `Library("npu", "IMPL", "Meta")`。若当前 PyTorch 缺少 `npu::silu_patch` 的 schema，就用已有的 `m_fragment` 定义它。`@torch.library.impl(m, "silu_patch")` 返回与输入同形状的 FP16 Meta 张量。每次调用将激活转为连续 FP16，直接传入 `torch.ops.npu.silu_patch`，TorchAir 导出为 `SiluPatch(x) → z`。GDN 门控在乘法前将 FP16 的 z 显式转为 FP32，与 RMSNorm 的 FP32 输出相乘。卷积与两处专家按原激活 dtype 继续运算。四层实验复用原版的 valid19 输入和参考，默认写入新的 native-shape 目录。

在 CPro AIR 生成服务器的导出日志和 `air/manifest.json` 的 `silu_patch.nodes` 中可以查看 14 个节点的输入形状、每个直接消费者的输入槽与读入和输出 dtype。若下一步是 Cast，`after_cast` 记录接收转型结果的下游节点和 dtype。形状来自静态 TorchAir 转换时的输入，AIR 显式记录的维度同时参与核对。`unrecorded` 表示该消费者的 AIR 描述未记录 dtype。

| S=64 的调用位置 | SiluPatch 的 x/z 形状 | z 如何进入后续运算 | 次数 |
|---|---|---|---:|
| L0–L2 GDN 卷积 | `[1,8192,64]` | 转回卷积激活 dtype，再 transpose 和拆分 Q/K/V | 3 |
| L0–L2 GDN RMSNorm 门控 | `[2048,128]` | 将 FP16 输出转为 FP32，乘 RMSNorm 输出，乘完按原逻辑转型 | 3 |
| L0–L3 共享专家 | `[64,512]` | 转回 gate 激活 dtype，再与 up 投影相乘 | 4 |
| L0–L3 W4 路由专家 | `[512,512]` | 转回 gate 切片 dtype，再与另一半切片相乘 | 4 |
