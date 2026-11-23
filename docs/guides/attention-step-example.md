# 单测 L0 的 `qkv_conv = F.silu(conv_out)`

在 78 上抓取 310P 原始 L0 forward 中实际送入 SiLU 的 `conv_out`，保存 310P 的 SiLU 输出，并用同一份输入在 CPU 上执行 `F.silu`。在 35 上把原 CPro attention AIR 中这一步编译成单独的 OMC，给板端灌入相同的 `conv_out`。比较边界从卷积输出开始，到 SiLU 返回值结束，后面才进行类型转换、转置和 Q/K/V 拆分。

### 1. 在 78 和 35 的 Bash 终端各执行一次

沿用之前四层 attention 测试的 `REPO_DIR`、`AI_RUN_DIR`。两台机器选择相同的 `STEP` 和 `TAG`。下面四个步骤都有 CPU、310P、CPro 三端对比入口。

```bash
cd "${REPO_DIR:?}"
git pull --ff-only origin main

STEP=qkv_conv_silu
TAG=01
T="${AI_RUN_DIR:?}/target/tests"
TOOLS="$REPO_DIR/targets/cpro/tools"

case "$STEP" in
  qkv_conv_silu)
    P="$TOOLS/attention_step.py"
    PREPARE=prepare-qkv
    JOB="l0-qkv-conv-silu-t64-valid19-$TAG"
    ;;
  rms_output|silu_output|out_proj_only)
    P="$TOOLS/gdn_op_matrix.py"
    PREPARE=prepare
    JOB="l0-three-ops-t64-valid19-$TAG"
    ;;
esac
R="$T/$JOB"
```

| 将 `STEP` 设为 | 三端使用的输入 | 对比的输出 |
|---|---|---|
| `qkv_conv_silu` | 310P 的 `conv_out` | 卷积后的 `F.silu(conv_out)` |
| `rms_output` | 310P 的 GDR core | 使用原层 gamma 和 epsilon 的 RMSNorm 输出 |
| `silu_output` | 310P 的门控投影 z | `F.silu(z)` |
| `out_proj_only` | 310P 的门控乘积回转 dtype 后的投影输入 | 动态量化和矩阵乘之后的 attention 输出 |

后三项由同一个入口一次采集、编译并执行三张独立 OMC。在同一批次切换这三项时，直接用第 4 节筛选已有 CSV。要以新输入或新版编译环境重新测试，请在两端使用新的 `TAG`。

### 2. 在 78 采集输入、生成参考并截取 AIR

`TARGET_SOURCE` 是含 `inference_segment.py` 的原始 310P 源码目录。`ATTENTION_AIR` 可指向以前四层 attention 批次的 `air/` 目录，其中有 `manifest.json` 和 `models/l00_c00_gdn_layer_hidden_o0/export.air`。如果旧目录已删，下面的回退步骤使用原四层流程，导出环境沿用此前的 `CPRO_SOURCE` 和 `TARGET_CONFIG`。

```bash
conda activate qwen3_5_w4
A="${ATTENTION_AIR:-$T/attention4-t64-valid19-v1024-qlate-01/air}"

(
  set -e
  if [[ ! -f "$A/manifest.json" || ! -f "$A/models/l00_c00_gdn_layer_hidden_o0/export.air" ]]; then
    S="$REPO_DIR/targets/cpro/scripts/check_attention4_tmatrix.sh"
    A="$T/attention4-source-$TAG/air"
    bash "$S" capture "${A%/air}"
    (conda activate qwen3_5_cpro && bash "$S" air "${A%/air}")
  fi

  python "$P" "$PREPARE" --run "$R" --source-air "$A" \
    --source-root "${TARGET_SOURCE:?}" --config "${NZ_CONFIG:-config/qwen3_5_moe.yaml}"
  cat "$R/air/boundaries.json"
  echo "35 上使用的远端 AIR 目录为 $R/air"
)
```

对 `qkv_conv_silu`，`boundaries.json` 中的 `inputs`、`source_edge` 和 `nodes` 给出真实输入边、输出边和截取后保留的节点。310P 驱动沿用 `--quant_mode=enable --quant_bits=4`。采集器会核对 310P 的 L0 输出在 valid19 上与源 AIR 配套参考逐位一致。相关文件在 `$R/air/models/l00_c00_tail_qkv_conv_silu_o0/`：

| 文件 | 内容 |
|---|---|
| `export.air` | 将在 35 编译的 SiLU 子图 |
| `conv_out.bin` | 板端唯一输入，数值取自 310P 的 SiLU 入口 |
| `qkv_conv_silu.bin` | 310P 原始 SiLU 输出 |
| `cpu.bin` | CPU 在相同输入上执行 SiLU 的输出 |
| `export.json` | 各 bin 的 dtype、shape、大小和 SHA256 |

同批次已有 `$R/air/manifest.json` 时可直接做第 3 节。

### 3. 在 35 编译 OMC、上板并回收结果

在已配置 `CPRO_DDK_ROOT`、`BIP`、`SN` 的 DDK 终端执行。默认两台机上的运行目录同路径；若不同，把 `AIR_ON_78` 设为第 2 节最后打印的绝对路径。首次执行会从 78 拉取整批 AIR 和 bin，编译、上板、回收并打印三端矩阵。

```bash
conda activate custom_pypto
python "$P" board --run "$R" \
  --remote-air "${AIR_ON_78:-$R/air}" \
  --ddk-root "${CPRO_DDK_ROOT:?}" \
  --tbe-diagnosis "${TBE_DIAGNOSIS:-$AI_RUN_DIR/common/tbe-diagnosis}" \
  --runtime-config "${RUNTIME_CONFIG:-$AI_RUN_DIR/runtime/runtime.json}"
```

编译结果见 `$R/omc/compile.json`，板端执行及回收状态见 `$R/recovery/result.json`。同一次请求中断时，用同一 `R` 重复本节命令可复用已有编译和执行请求。

### 4. 在 35 查看结果

`$R/matrices.csv` 每张矩阵按行取 actual、按列取 reference。下面只显示所选步骤的 CPro 对 310P；要重印完整 3×3 矩阵，用 `python "$P" report --run "$R"` 从已回收输出重算。

```bash
python - "$R/matrices.csv" "$STEP" <<'PY'
import csv
import sys

with open(sys.argv[1], newline="") as stream:
    for row in csv.DictReader(stream):
        if (row["operator"] == sys.argv[2] and row["actual"] == "cpro"
                and row["scope"] in ("valid19", "token18")
                and row["metric"] in ("cosine", "relative_l2", "rmse", "error_ratio")):
            value = float(row["310p"])
            text = f"{value:.4%}" if row["metric"] in ("relative_l2", "error_ratio") else f"{value:.9g}"
            print(f"{row['scope']:8} {row['metric']:12} {text}")
PY
```

`SiLU(conv_out)` 的原始输入是 `[1,8192,64]`，token 在最后一维。valid19 比较 `x[..., :19]` 的 155,648 个元素，token18 比较 `x[..., 18]` 的 8,192 个元素。CSV 还包含每个有效 token 的五项指标和逐元素统计。`error_ratio` 是超出 `atol=1e-5, rtol=1e-3` 的元素比例，门限为 0.1%。

### 换成其他 L0 或 L3 步骤

从[预设表](../../specs/attention-step-presets.csv)选 `LAYER` 和 `STEP`。例如单独导出 L0 输出投影的 A8 整数码、逐 token scale 或矩阵乘，可分别选 `out_proj_a8`、`out_proj_token_scale`、`out_proj_matmul`。下面这段在两台机器上各执行一次。

```bash
LAYER=0
STEP=out_proj_a8
TAG=01
P="${REPO_DIR:?}/targets/cpro/tools/attention_step.py"
T="${AI_RUN_DIR:?}/target/tests"
R="$T/l${LAYER}-$STEP-$TAG"
```

在 78，用第 2 节选定的原始 attention 批次。若用过回退导出，请将 `ATTENTION_AIR` 设为 `$T/attention4-source-$TAG/air`。

```bash
conda activate qwen3_5_w4
A="${ATTENTION_AIR:-$T/attention4-t64-valid19-v1024-qlate-01/air}"
python "$P" catalog --source-air "$A" --layer "$LAYER" --presets-only \
  --output "$T/attention-l${LAYER}-steps.csv"
python "$P" extract --source-air "$A" --layer "$LAYER" --step "$STEP" --run "$R" &&
  cat "$R/slice/slice.json"
```

在 35 编译该切片。如果两端目录不同，`SLICE_ON_78` 指向 78 本次运行的 `slice/`。

```bash
conda activate custom_pypto
python "$P" compile --run "$R" --remote-slice "${SLICE_ON_78:-$R/slice}" \
  --ddk-root "${CPRO_DDK_ROOT:?}" \
  --tbe-diagnosis "${TBE_DIAGNOSIS:-$AI_RUN_DIR/common/tbe-diagnosis}"
```

通用模式产出 AIR、OMC 和边界清单。上板执行自选步骤需要该边界的真实输入 bin，按编译 OMC 的输入槽位绑定并与 `slice.json` 的边名、dtype、shape 核对。构建三端矩阵还需要在相同输入上得到该步骤的 CPU 和 310P 输出。上面四个具备专用采集器的步骤可直接用第 1–4 节完成这些操作。

例如要执行刚编好的 `out_proj_a8`，可复用三算子批次里 `air/models/l00_c00_tail_out_proj_only_o0/export.json` 的 `source_inputs[0].path` 登记的投影输入。该张量是量化前的整份 T64 投影输入。将本次编译打印的 OMC（连同同目录的外置权重，如有）和该输入 bin 传到板端同一工作目录后，可在已分配给本次测试的**板端 shell**使用以下模板。示例为单输入 FP16、单输出 INT8，实际 dtype 以本次 OMC 为准。多输入步骤的 `INPUTS` 和 `INPUT_TYPES` 按 OMC 槽位一一对应，以逗号分隔。

```sh
cd "${BOARD_STEP_DIR:?}"
MODEL="$PWD/model.omc"
INPUTS="$PWD/proj_input.bin"
INPUT_TYPES=FP16
OUTPUT_TYPE=INT8
TOOL_DIR=/data/local/tmp/tools_sysdbg
OUT=$(mktemp -d "$PWD/output.XXXXXX") &&
  LD_LIBRARY_PATH="$TOOL_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  "$TOOL_DIR/model_run_tool_internal" \
    --model="$MODEL" --weight_dir="$PWD" \
    --input="$INPUTS" --input_type="$INPUT_TYPES" --output_type="$OUTPUT_TYPE" \
    --times=1 --modelmanager_itf=V2 --direct_from_file=true --enable_item=1 \
    --output_dir="$OUT" &&
  ls -l "$OUT"
```

这个整数输出应与同一份 `proj_input` 产生的 310P DynamicQuant A8 码逐元素比较。完整 `out_proj` 的浮点结果用于投影输出边界。
