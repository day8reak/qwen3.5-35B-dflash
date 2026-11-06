# 完整流程执行

从源权重生成Draft、Prefill和Decode，再准备配置并运行板端文字闭环。原始Target Cpro导出工程、量化权重与源模型权重由用户提供，不在GitHub中。当前首次生成运行ABI配置的入口尚未补齐，具体边界见第5步；已有成功闭环的配置可由当前阶段整理命令保留。

当前阶段为[归档约60ms Draft并转向Target精度](current-status.md)。已有模型直接归档，不重新编译。以下是当前packed基线的从零生成配方；其独立设备调用已执行，原数值门槛仍FAILED，旧token-ID常驻接口尚未接入该host embedding模型。

## 1. 公共目录和源权重

先加载conda/DDK环境，再加载项目配置。以下变量填写实际输入位置，不依赖已有AIR或OMC。

```bash
source "$REPO_DIR/targets/cpro/scripts/env.sh"
export TARGET_WEIGHTS='/实际路径/原Target配置model_file对应的原始权重目录'
export DRAFT_WEIGHTS='/实际路径/DFlash原始权重'
export TARGET_SOURCE='/实际路径/原始Cpro导出工程'
export TARGET_YAML='/实际路径/原始量化配置.yaml'
export MAX_DRAFT_TOKENS=15

mkdir -p "$AI_RUN_DIR"/{draft,prefill,decode}/tests
mkdir -p "$AI_RUN_DIR"/{common,runtime,fused}
export INCREMENTAL_AIR="$AI_RUN_DIR/draft/air"
export INCREMENTAL_COMPILED="$AI_RUN_DIR/draft/omc"
```

`TARGET_WEIGHTS`对应原Target YAML的`model_file`目录；Draft只从中读取原始embedding和LM head。`DRAFT_WEIGHTS`指向独立的DFlash原始检查点。Target YAML另含`quanted_pth`、`embedding_weight_path`、`embedding_scale_path`，供后面的Target量化导出使用。导出主机与DDK服务器不同时，传输整个AIR输入目录及引用权重，不能只传AIR。

## 2. Draft权重 → 公共AIR → 公共OMC

准备六层Draft的71份FP16权重；Target提供原始embedding和LM head。当前归档档位固定C16／cap192／K15，host FP16 embedding、合并投影、GQA reshape、StableArgMax、FP32 RoPE系数表与CPU feature pack均保留。DDK端需按[算子说明](../targets/cpro/ops/draft-top1/README.md)完成StableArgMax安装及已有TBE兼容诊断。

```bash
python3 "$REPO_DIR/targets/cpro/scripts/prepare_draft.py" \
  --target-weights "$TARGET_WEIGHTS" --draft-weights "$DRAFT_WEIGHTS" \
  --output "$AI_RUN_DIR/draft/source/prepared"

# TorchAir权重服务器；完整传输air目录及引用权重到DDK服务器。
python3 "$REPO_DIR/targets/cpro/scripts/export_incremental_draft.py" \
  --prepared "$AI_RUN_DIR/draft/source/prepared" --frontend pytorch \
  --optimization packed-spatial-host-embedding --top1 stable-arg-max \
  --context-rows 16 --cache-capacity 192 --max-draft-tokens 15 \
  --output "$AI_RUN_DIR/draft/air"

# 所选DDK服务器；不使用离线转置参数。
python3 "$REPO_DIR/targets/cpro/scripts/compile_incremental_draft.py" \
  --air-export "$AI_RUN_DIR/draft/air" \
  --source "$AI_RUN_DIR/draft/source/prepared" \
  --ddk-root "$CPRO_DDK_ROOT" --ddk-id "$CPRO_DDK_ID" \
  --tbe-diagnosis "$AI_RUN_DIR/common/tbe-diagnosis" \
  --rope-fp32-table --cpu-feature-pack --output "$AI_RUN_DIR/draft/omc"
```

导出最终目录为`draft/air`，编译最终目录为`draft/omc`。固定暂存槽为`.air-build`／`.omc-build`，成功后发布并收起；编译失败不覆盖已有公共模型。失败重试使用`--replace-failed`，有意替换已有成功模型使用`--replace`。新模型未验证前保留`.omc-previous`；不会自动改成生产常驻模型。编译前磁盘下界约6.44GB，工具会按真实权重计算。

编译成功状态为`PYTORCH_DRAFT_OMC_COMPILED`、物理ABI为`BOUND_DIAGNOSTIC`。紧凑bundle保留`pytorch-abi.json`和`compiler.air`，重新准备独立验证不再依赖旧test编译快照或DDK动态库。完整原始prepared权重仍用于CPU参考和host embedding。

仅在需要重新验收时运行以下诊断；模型直接引用公共目录，不复制到tests。

```bash
export VECTOR_CONFIG="${VECTOR_CONFIG:-$AI_RUN_DIR/runtime/runtime.json}"
python3 "$REPO_DIR/targets/cpro/tools/validate_incremental_draft.py" \
  --compiled "$AI_RUN_DIR/draft/omc" --air-export "$AI_RUN_DIR/draft/air" \
  --source "$AI_RUN_DIR/draft/source/prepared" --config "$VECTOR_CONFIG" \
  --output "$AI_RUN_DIR/draft/tests/accuracy-run" --apply
python3 "$REPO_DIR/targets/cpro/tools/validate_incremental_draft.py" \
  --recover "$AI_RUN_DIR/draft/tests/accuracy-run" \
  --output "$AI_RUN_DIR/draft/tests/accuracy-result"
```

PENDING只回收同一请求；不得重复启动。上述公开位置是模型存放与独立诊断入口，后面的旧常驻循环仍需其兼容的token-ID Draft，不应直接指向这份packed OMC。

## 3. Target源工程 → Prefill64／Decode1 AIR → OMC

分别导出40层静态64和静态1，将输出规范化为logits、80路状态和Target features。原始导出使用TorchAir，编译使用指定DDK。公共AIR目录统一为`air/export.air`＋`air/weight/`；同步改写AIR的权重路径，同盘移动导出产物，不再复制一份大权重。

```bash
(
set -e
for role in prefill decode; do
  if test "$role" = prefill; then sequence=64; else sequence=1; fi
  export_job="$AI_RUN_DIR/$role/source/export"
  canonical="$AI_RUN_DIR/$role/source/canonical"
  air_bundle="$AI_RUN_DIR/$role/air"
  test ! -e "$air_bundle"

  python3 "$REPO_DIR/targets/cpro/scripts/prepare_target_pytorch.py" \
    --source-root "$TARGET_SOURCE" --config "$TARGET_YAML" \
    --sequence-length "$sequence" --output "$export_job"
  bash "$export_job/export.sh"

  original=$(python3 - "$export_job/target-pytorch-source/receipt.json" <<'PY'
import json, sys
from pathlib import Path
print(Path(json.loads(Path(sys.argv[1]).read_text())['expected_air']).parent)
PY
)
  python3 "$REPO_DIR/targets/cpro/scripts/canonicalize_target_export.py" \
    --air "$original/export.air" --discover-torchair-schema \
    --sequence-length "$sequence" --hash-weights --output "$canonical"

  cp "$canonical/export.air" "$original/export.air"
  python3 "$REPO_DIR/targets/cpro/scripts/normalize_target_air.py" \
    --air-dir "$original" --apply
  mv "$original" "$air_bundle"

  python3 "$REPO_DIR/targets/cpro/scripts/compile_full_target.py" \
    --air "$air_bundle/export.air" \
    --input-layout air-directory --profile "cpro-40block-$role-$sequence" \
    --sequence-length "$sequence" --target-features \
    --ddk-root "$CPRO_DDK_ROOT" --ddk-id "$CPRO_DDK_ID" \
    --output "$AI_RUN_DIR/$role/omc"
done
)
```

两个模型分别编译，不使用旧双gear方案。产物在各自omc目录，独立测试放各自tests目录。保留当前AIR与对应输入权重，不把旧失败任务搬进公共目录。

## 4. 离线词表、embedding与共享运行器

词表包已随仓库提供，也可从[原始下载链接](https://raw.githubusercontent.com/day8reak/qwen3.5-35B-dflash/main/docs/assets/qwen35-tokenizer-offline.tar.gz)下载再复制。下面不访问网络。

```bash
mkdir -p "$AI_RUN_DIR/common/tokenizer-offline" "$AI_RUN_DIR/common/embedding"
tar -xzf "$REPO_DIR/docs/assets/qwen35-tokenizer-offline.tar.gz" \
  -C "$AI_RUN_DIR/common/tokenizer-offline"
(cd "$AI_RUN_DIR/common/tokenizer-offline" && sha256sum -c SHA256SUMS)
export TOKENIZER_DIR="$AI_RUN_DIR/common/tokenizer-offline/source"

python3 "$REPO_DIR/targets/cpro/scripts/prepare_board_tokenizer.py" \
  --source-dir "$TOKENIZER_DIR" --output "$AI_RUN_DIR/common/tokenizer-offline/prepared"

cp '/原量化配置的embedding_weight_path' "$AI_RUN_DIR/common/embedding/weight.bin"
cp '/原量化配置的embedding_scale_path' "$AI_RUN_DIR/common/embedding/scale.bin"
wc -c "$AI_RUN_DIR/common/embedding/"{weight,scale}.bin

python3 "$REPO_DIR/targets/cpro/scripts/build_hiai.py" \
  --output "$AI_RUN_DIR/runtime/native" --replace
```

embedding大小分别为508559360、993280字节。所有测试共用runtime/native；C++代码更新时重编运行器，模型未改变时不重编OMC。

## 5. 运行配置

统一入口runtime/runtime.json通过相对路径引用公共模型、词表、embedding及native运行器；runtime-assets.json保存实际89输入／82输出描述和80路状态映射。

**当前尚缺从全新编译产物自动生成runtime-assets.json的生产入口。** 首次部署需要获取两个模型的实际SDK命名描述，并完成源图状态所有者、编译图和运行时槽位的对应，不能按大小或序号猜测。上面的生成/编译从源权重开始；下面的启动要求该配置已就绪，不代表当前能从空目录自动完成全流程。

已有成功闭环的runtime.json继续绑定其兼容模型；归档若替换了公共目录，会重定向原配置中对应模型的路径。重新编译Target后须核对新ABI与状态映射，并更新cache_key。新生成的packed Draft尚需常驻接口适配，不能把它直接填入旧配置。

## 6. 文字输入、板端闭环、结果回收

以下仅适用于已有、兼容token-ID Draft的成功runtime配置；运行器从该配置识别原增量模型，不使用本页新生成的packed OMC。配置就绪后，板端自主执行文字编码→Prefill一次→常驻Draft/Decode循环→文字解码。服务器只准备、下发、启动和最终回收。当前模板后的prompt＋生成预算须≤64。

```bash
export RUN_DIR="$AI_RUN_DIR/fused/run"
export RECOVERY_DIR="$AI_RUN_DIR/fused/result"

python3 "$REPO_DIR/targets/cpro/scripts/run_board_serial_dflash.py" \
  --config "$AI_RUN_DIR/runtime/runtime.json" \
  --max-draft-tokens "$MAX_DRAFT_TOKENS" \
  --text '你好，请介绍一下自己。' \
  --max-new-tokens 8 --max-proposals-per-round 2 \
  --apply --detach --output "$RUN_DIR"

# 完成后回收同一请求；断线后也不重新启动模型。
python3 "$REPO_DIR/targets/cpro/scripts/run_board_serial_dflash.py" \
  --recover "$RUN_DIR" --output "$RECOVERY_DIR"
```

输出包含文字、tokens、接受率、各阶段和加载时延。复测使用相同run/result目录，已结束的旧临时文件自动清理；未知退出状态先回收。Target精度仍待闭合，流程接通不代表文本正确或获得加速。
