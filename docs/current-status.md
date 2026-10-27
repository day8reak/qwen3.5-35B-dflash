# 当前更新与执行命令

[首页](README.md) · [环境配置](environment-setup.md) · [验证记录](validation.md)

更新日期：2026-10-27。当前主线是完整40层target、6层DFlash和可选融合算子。

## 当前状态

| 项目 | 状态 |
|---|---|
| PyTorch完整结构 | 全尺寸meta模型及层级FX图；不加载权重 |
| GE完整结构 | target/draft完整节点、连边、C++和recipe；融合可选 |
| 融合算子探针 | GDN prefill/decode已真实编译通过，名称适配已生效；上板待验证。conv/RMS接口问题待处理 |
| 真实权重AIR导出 | 旧双图支持完整FP16；新GE target支持显式缓存 |
| 状态执行 | 主机fake prefill→decode接续通过；真实板端待验证 |
| 尚未完成 | W4A8映射、常驻缓存与native DFlash调度、完整模型设备精度和性能 |

## 用户现在执行什么

**复用已编译的两份GDN OMC，上板执行并回收结果。** 更新代码后重建runner，它已支持本DDK的FP16末态；无需重建AIR/OMC或重跑平台探针。以下操作使用[环境配置](environment-setup.md#5-ohos--hiai--hdc)中的OHOS SDK、HIAI和HDC变量。

**1. 更新runner并上传。** 保留刚成功编译时的 `FUSED_PROBE_RUN`；`FUSED_BOARD_JOB` 是本次上板的新目录，`BOARD_HIAI_LIB_DIR` 沿用已验证的板端私有运行库目录。

```bash
cd "$REPO_DIR"
git pull --ff-only
: "${FUSED_PROBE_RUN:?设置刚成功编译的probes目录}"
: "${BOARD_HIAI_LIB_DIR:?设置板端已验证的私有库目录}"
: "${HIAI_RUNTIME_VERSION:?设置实际运行库版本}" "${CPRO_DDK_ID:?}" "${FIRMWARE_ID:?}"
"$HDC" -t "$SN" -s "$BIP" shell "echo qwen35-fused-board-ready"
export FUSED_BOARD_JOB="$(mktemp -d "$AI_RUN_DIR/fused-board-XXXXXXXX")"
"$MODEL_PYTHON" "$REPO_DIR/targets/cpro/scripts/build_hiai.py" \
  --output "$FUSED_BOARD_JOB/native"
export FUSED_PROBE_RUNNER="$FUSED_BOARD_JOB/native/build/qwen35_dflash_hiai"
export BOARD_FUSED_ROOT="/data/local/tmp/qwen35_fused_$(date +%Y%m%d_%H%M%S)_$RANDOM"
tar -C "$FUSED_PROBE_RUN" -cf "$FUSED_BOARD_JOB/bundle.tar" bundle
"$HDC" -t "$SN" -s "$BIP" shell "mkdir -p '$BOARD_FUSED_ROOT'"
"$HDC" -t "$SN" -s "$BIP" file send "$FUSED_BOARD_JOB/bundle.tar" "$BOARD_FUSED_ROOT/bundle.tar"
"$HDC" -t "$SN" -s "$BIP" file send "$FUSED_PROBE_RUNNER" "$BOARD_FUSED_ROOT/runner"
```

**2. 执行两例。** 使用本批次生成的 `run-board.sh`，逐例保存输出、stderr和退出码。

```bash
"$HDC" -t "$SN" -s "$BIP" shell \
  "cd '$BOARD_FUSED_ROOT' && tar -xf bundle.tar && chmod +x runner && LD_LIBRARY_PATH='$BOARD_HIAI_LIB_DIR:/vendor/lib64/chipsetsdk' sh bundle/run-board.sh '$BOARD_FUSED_ROOT/runner' '$BOARD_FUSED_ROOT/results' '$HIAI_RUNTIME_VERSION' '$CPRO_DDK_ID' '$FIRMWARE_ID'"
```

**3. 回收并比较，执行失败也继续回收。** 回传 `comparison/report.txt`；失败时下方命令同时打印对应日志。数值误差目前作诊断，尚未设验收阈值。

```bash
"$HDC" -t "$SN" -s "$BIP" shell \
  "tar -cf '$BOARD_FUSED_ROOT/results.tar' -C '$BOARD_FUSED_ROOT' results"
"$HDC" -t "$SN" -s "$BIP" file recv "$BOARD_FUSED_ROOT/results.tar" "$FUSED_BOARD_JOB/results.tar"
tar -xf "$FUSED_BOARD_JOB/results.tar" -C "$FUSED_BOARD_JOB"
"$MODEL_PYTHON" "$REPO_DIR/targets/cpro/scripts/direct_model_air.py" compare-fused-probes \
  --probe-run "$FUSED_PROBE_RUN" --board-run "$FUSED_BOARD_JOB/results" \
  --output "$FUSED_BOARD_JOB/comparison"
cat "$FUSED_BOARD_JOB/comparison/report.txt"
for name in gdn_prefill gdn_decode; do
  if [ "$(cat "$FUSED_BOARD_JOB/results/$name.exitcode")" != 0 ]; then
    tail -n 40 "$FUSED_BOARD_JOB/results/$name.stdout" "$FUSED_BOARD_JOB/results/$name.stderr"
  fi
done
```

## 已确认与剩余问题

| 项目 | 结论 |
|---|---|
| GDN两例 | 用户回传构图、AIR、OMC均通过；每例名称适配 `APPLIED=1`、拒绝0 |
| 编译目标 | 始终为 `kirinx90`；`Ascend310B1` 是DDK初始化后返回的内部名称，`Ascend310P3` 是原工程平台 |
| GDN状态 | 安装版初态/末态为FP16；本次runner修正了末态只接受FP32的旧校验 |
| conv / RMS | 状态宽度4与文档3不一致 / 当前维推注册缺失，分别待处理 |
| 尚未验证 | 真实设备加载、执行、输出数值和NPU执行位置；完整模型与连续状态验证另行进行 |

## 完整模型构图（按需）

需要完整模型结构时选择下面一条路径；`sequence-length=8`只限定输入长度，模型层数、专家和词表均为完整尺寸。

**PyTorch：** 生成完整meta模型和层级FX结构，不需要检查点或TorchAir。

```bash
"$MODEL_PYTHON" -m qwen35_moe_dflash.cli build-full-pytorch \
  --sequence-length 8 --output "$AI_RUN_DIR/torch-full-$(date +%Y%m%d-%H%M%S)"
```

**GE：** 生成完整target/draft构图源码，默认不使用融合算子，不需要检查点或GE库。

```bash
"$MODEL_PYTHON" "$REPO_DIR/targets/cpro/scripts/direct_model_air.py" build-full-structure \
  --sequence-length 8 --expert-dispatch selected --fused-ops none \
  --output "$AI_RUN_DIR/ge-full-$(date +%Y%m%d-%H%M%S)"
```

这些是未绑定真实权重的结构产物。实际导出AIR：按 [PyTorch](pytorch-guide.md)或 [GE](ge-guide.md)指南继续；无需重跑旧探针。

## 文档融合接口

上面的 `GE_CUSTOM_BINDINGS` 可复用于GE融合构图。绑定来自用户文档，不代表kernel已安装或已在目标芯片验证；融合构图命令见[GE指南](ge-guide.md)。

完整W4A8 + FP16 DFlash先按24–28GiB可用RAM规划；这不是实测峰值，也不适用于当前FP16权重导出，见[内存预算](environment-setup.md#内存预算文档-w4a8--dflash)。
