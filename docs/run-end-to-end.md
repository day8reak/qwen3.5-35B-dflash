# 完整流程执行

从源权重生成Draft、Prefill和Decode，再准备配置并运行板端文字闭环。原始Target Cpro导出工程、量化权重与源模型权重由用户提供，不在GitHub中。当前首次生成运行ABI配置的入口尚未补齐，具体边界见第5步；已有成功闭环的配置可由当前阶段整理命令保留。

## 1. 公共目录和源权重

先加载conda/DDK环境，再加载项目配置。以下变量填写实际输入位置，不依赖已有AIR或OMC。

```bash
source "$REPO_DIR/targets/cpro/scripts/env.sh"
export TARGET_WEIGHTS='/实际路径/Target原始权重'
export DRAFT_WEIGHTS='/实际路径/DFlash原始权重'
export TARGET_SOURCE='/实际路径/原始Cpro导出工程'
export TARGET_YAML='/实际路径/原始量化配置.yaml'
export MAX_DRAFT_TOKENS=15

mkdir -p "$AI_RUN_DIR"/{draft,prefill,decode}/{air,omc,tests}
mkdir -p "$AI_RUN_DIR"/{common,runtime,fused}
export INCREMENTAL_AIR="$AI_RUN_DIR/draft/air/kv-c16-cap192-k$MAX_DRAFT_TOKENS"
export INCREMENTAL_COMPILED="$AI_RUN_DIR/draft/omc/kv-c16-cap192-k$MAX_DRAFT_TOKENS"
```

Target YAML包含有效的model_file、quanted_pth、embedding_weight_path、embedding_scale_path。导出主机与DDK服务器不同时，传输整个AIR输入目录及引用权重，不能只传AIR。

## 2. Draft权重 → AIR → OMC

准备六层Draft的71份FP16权重，只读取Target的embedding和LM head。K支持15／7／3／1；新导出不依赖旧S64 AIR或OMC。编译预检至少需要6.23GiB空闲，编译器临时文件可能额外占用空间。

```bash
python3 "$REPO_DIR/targets/cpro/scripts/prepare_draft.py" \
  --target-weights "$TARGET_WEIGHTS" --draft-weights "$DRAFT_WEIGHTS" \
  --output "$AI_RUN_DIR/draft/source/prepared"

python3 "$REPO_DIR/targets/cpro/scripts/export_incremental_draft.py" \
  --prepared "$AI_RUN_DIR/draft/source/prepared" --ge-root "$CANN_ROOT" \
  --context-rows 16 --cache-capacity 192 \
  --max-draft-tokens "$MAX_DRAFT_TOKENS" --output "$INCREMENTAL_AIR"

python3 "$REPO_DIR/targets/cpro/scripts/compile_incremental_draft.py" \
  --air-export "$INCREMENTAL_AIR" \
  --ddk-root "$CPRO_DDK_ROOT" --ddk-id "$CPRO_DDK_ID" \
  --replace-failed --output "$INCREMENTAL_COMPILED"
```

需要DRAFT_INCREMENTAL_OMC_COMPILED_ABI_BOUND、compiled_precision和compiled_abi均PASSED。当前新增量模型的实机结果待补，不能把主机测试写成板端精度通过。

## 3. Target源工程 → Prefill64／Decode1 AIR → OMC

分别导出40层静态64和静态1，将输出规范化为logits、80路状态和Target features。原始导出使用TorchAir，编译使用指定DDK。cp优先reflink，不支持时会复制权重，须预留磁盘空间。

```bash
(
set -e
for role in prefill decode; do
  if test "$role" = prefill; then sequence=64; else sequence=1; fi
  export_job="$AI_RUN_DIR/$role/source/export"
  canonical="$AI_RUN_DIR/$role/source/canonical"
  prefix="glm_export_air_cpro_40block_${sequence}_quant"

  python3 "$REPO_DIR/targets/cpro/scripts/prepare_target_pytorch.py" \
    --source-root "$TARGET_SOURCE" --config "$TARGET_YAML" \
    --sequence-length "$sequence" --output "$export_job"
  bash "$export_job/export.sh"

  original="$export_job/target-pytorch-source/source/$prefix"
  python3 "$REPO_DIR/targets/cpro/scripts/canonicalize_target_export.py" \
    --air "$original/export.air" --discover-torchair-schema \
    --sequence-length "$sequence" --hash-weights --output "$canonical"

  # 保持FileConstant引用的原目录前缀。
  test ! -e "$AI_RUN_DIR/$role/air/$prefix"
  cp -a --reflink=auto "$original" "$AI_RUN_DIR/$role/air/$prefix"
  cp "$canonical/export.air" "$AI_RUN_DIR/$role/air/$prefix/export.air"

  python3 "$REPO_DIR/targets/cpro/scripts/compile_full_target.py" \
    --air "$AI_RUN_DIR/$role/air/$prefix/export.air" \
    --input-layout export-directory --profile "cpro-40block-$role-$sequence" \
    --sequence-length "$sequence" --target-features \
    --ddk-root "$CPRO_DDK_ROOT" --ddk-id "$CPRO_DDK_ID" \
    --output "$AI_RUN_DIR/$role/omc/current"
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

对于已有成功闭环的本项目，当前阶段整理命令会迁移此配置，旧测试prepared不再是依赖。重新编译Target后须核对新ABI与状态映射，并更新cache_key；不能把旧模型缓存当作新产物。

## 6. 文字输入、板端闭环、结果回收

配置就绪后，板端自主执行文字编码→Prefill一次→常驻Draft/Decode循环→文字解码。服务器只准备、下发、启动和最终回收。当前模板后的prompt＋生成预算须≤64。

```bash
export RUN_DIR="$AI_RUN_DIR/fused/run"
export RECOVERY_DIR="$AI_RUN_DIR/fused/result"

python3 "$REPO_DIR/targets/cpro/scripts/run_board_serial_dflash.py" \
  --config "$AI_RUN_DIR/runtime/runtime.json" \
  --incremental-draft "$INCREMENTAL_COMPILED" \
  --max-draft-tokens "$MAX_DRAFT_TOKENS" \
  --text '你好，请介绍一下自己。' \
  --max-new-tokens 8 --max-proposals-per-round 2 \
  --apply --detach --output "$RUN_DIR"

# 完成后回收同一请求；断线后也不重新启动模型。
python3 "$REPO_DIR/targets/cpro/scripts/run_board_serial_dflash.py" \
  --recover "$RUN_DIR" --output "$RECOVERY_DIR"
```

输出包含文字、tokens、接受率、各阶段和加载时延。复测使用相同run/result目录，已结束的旧临时文件自动清理；未知退出状态先回收。Target精度仍待闭合，流程接通不代表文本正确或获得加速。
