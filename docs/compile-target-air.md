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
