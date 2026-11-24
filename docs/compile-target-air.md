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

`original-air` 在 78 直接运行 `CPRO_SOURCE` 下的 `export_glm_model_qwen3_5_moe.py`。原量化 wrapper 在图内把 INT8 embedding 乘以 FP32 scale，执行 L0–L3，按 `output_pos` 选行，再过 final norm 和 LM head，返回 logits 和各层 cache。另一个入口 `air` 接收 310P 的 FP16 embedding，只导出 L3 的全部 64 行 hidden。

脚本在 `config=...` 后显式传入 `quant_bits=4` 选择 A8W4。支持位宽选择的原导出器默认用 8 位权重。对 L0 的 `in_proj_qkv`，W4 加载器将权重打包为 INT32 `[2048,1024]`，将分组权重 scale 整理为 FP32 `[8,8192]`。激活的逐 token scale 仍为 FP32 `[64,1]`。

如需单独核对 78 的 `torch_npu` 是否接受这组 shape 和 dtype，可在读取模型权重前运行下面的 meta 检查。通过时会打印 `torch.Size([64, 8192]) torch.float16`。若报 `scale's type should be torch.int64 in A8W4`，需核对当前 CPro 导出环境对分组权重 scale 的 dtype 要求。

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

78 导出，用现有 valid19 采集准备输入。可用 `ORIGINAL4_REFERENCE` 指定含 `manifest.json` 的采集目录。脚本打印图内乘法在主机重算后舍入到 FP16 与 310P 层入口的逐位差异数。运行时选择 `output_pos=[18]`，初始 cache 为零。`conv_state_pos` 默认按已记录的 valid64 格式推定为 `[19, 0 × 127]`，可用 `ORIGINAL4_CONV_CONTROL` 指向实测的 128 元素 INT32 文件。

```bash
cd "${REPO_DIR:?}" && git pull --ff-only origin main
bash targets/cpro/scripts/decoder4_t64.sh original-air
```

53 使用现有 DDK、HDC、板上工具和运行配置。脚本拉取 AIR，编译一份带外置权重的 OMC，按编译图的输入顺序运行 HIAI V2，并取回各输出。控制量和 embedding 的数据类型按原版接口生成。默认目录是 `$AI_RUN_DIR/target/tests/decoder4-original-t64-01`，本次运行的 `replay-*/io.json` 和 `result.json` 记录槽位与结果文件。两台机器可分别设置 `ORIGINAL4_RUN`，目录不同时在 53 设置 `ORIGINAL4_REMOTE_AIR` 为 78 的 AIR 目录。

```bash
cd "${REPO_DIR:?}" && git pull --ff-only origin main
bash targets/cpro/scripts/decoder4_t64.sh original-board
```
