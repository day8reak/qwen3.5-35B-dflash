# 先核验 Target golden

当前 T8 捕获的连接与 QKV→ConvState、V→缓存逐位一致；完整模块 AIR 的 19 对 weight/scale 也与源数据一致。这些检查尚未证明投影计算正确。先独立重算 L0–L2 QKV 和 L3 V，暂缓 CPro 五图重跑。

## 1. CPU 独立重算

在保存原 NZ 源码、量化权重和 `GOLDEN_CAPTURE` 的主机执行，沿用 [Prefill 环境与变量](target-prefill-fixture.md#1-准备)，先同步 `main`。需要 Python、NumPy、PyYAML 和 CPU 可用的 PyTorch；不导入模型、不调用 NPU、不编译 AIR/OMC。无需重新运行 Prefill。

`TARGET_SOURCE` 在本步骤指生成 golden 的**原 NZ/310P 源码根目录**，必须包含 `models/modeling_qwen3_5_moe_hiai_nz.py`，不是 CPro 目录。脚本核对它与 capture 的源文件哈希；YAML 的 `quanted_pth` 相对路径也按该根目录解析。

```bash
export GOLDEN_CAPTURE="$PREFILL_FIXTURE/capture"
export PAIR_ROOT="${PAIR_ROOT:-$AI_RUN_DIR/target/tests/attention-pair}"
export GOLDEN_CHECK="$PAIR_ROOT/golden-check"

python "$REPO_DIR/targets/cpro/tools/target_layer_probe.py" verify-qlinear-golden \
  --capture "$GOLDEN_CAPTURE" \
  --source-root "$TARGET_SOURCE" --target-config "$TARGET_CONFIG" \
  --output "$GOLDEN_CHECK/cpu"

cat "$GOLDEN_CHECK/cpu/summary.txt"
```

输出目录须尚不存在。脚本按原始 INT4 权重 `[N,K]`、原 FP32 group scale 计算，先核对重新打包的权重和 scale 与 capture 完全一致，再做每组 256 元素的精确整数点积、FP64 缩放求和、最终转 FP16。另报告固定 FP32 累加结果与 FP64 参考的差异，不按 golden 选择更接近的一份。

CPU 动态量化使用明确约定：FP32 逐行 `absmax/127`、除 scale 后 ties-to-even 舍入、零行输出零。逐行缩放和舍入步骤参考[官方测试](https://github.com/Ascend/op-plugin/blob/master/test/test_custom_ops/test_npu_dynamic_quant.py)，零行策略由本工具显式定义。这不能认证当前定制 310P 内核的舍入；官方测试本身允许量化整数有 1 的差异。当前成功 capture 未保存实际 A8 激活与 token scale，小偏差不能直接判 golden 错误。

`INSPECTED` 表示计算与记录完成。各行 `AGREES/DIFFERS` 使用现有精度指标，仅描述所选投影与声明数学参考的一致性；整网 golden 保持 `UNVERIFIED`，原输入和 golden 不改写。

## 2. 有偏差时，仅重放 DynamicQuant

在**原来生成 golden 的 310P/torch_npu 环境**执行，保留原自定义算子环境配置。只对四份已保存的 hidden 调用动态量化，不加载完整模型、权重，不调用 `npu_quant_matmul`。

```bash
python "$REPO_DIR/targets/cpro/tools/target_layer_probe.py" replay-dynamic-quant \
  --capture "$GOLDEN_CAPTURE" --device npu:0 \
  --output "$GOLDEN_CHECK/dynamic-quant"

python "$REPO_DIR/targets/cpro/tools/target_layer_probe.py" verify-qlinear-golden \
  --capture "$GOLDEN_CAPTURE" \
  --source-root "$TARGET_SOURCE" --target-config "$TARGET_CONFIG" \
  --quant-replay "$GOLDEN_CHECK/dynamic-quant" \
  --output "$GOLDEN_CHECK/with-dynamic-quant"

cat "$GOLDEN_CHECK/with-dynamic-quant/summary.txt"
```

第二次核验同时报告纯 CPU 参考、使用新 A8/scale 的 CPU 矩阵参考，以及两套量化中间值的差异。新调用恢复的是冻结输入的逻辑值，不是历史同次调用的 NPU format/stride，也不冒充原调用 A8。

| 结果 | 定位方向 |
| --- | --- |
| 原始权重或 scale 绑定失败 | 先核对 checkpoint、capture 与 YAML，不作精度归因 |
| CPU 量化参考有偏差，新量化加 CPU 矩阵参考一致 | 检查动态量化规则／实现差异 |
| 两套矩阵参考均与原投影明显不符 | 查原 QuantMatmul 的 group、packing、调用环境；暂不把这些 golden 当正确参考 |
| 所选投影一致 | 再推进 GDN 后续计算、FullAttention 及整网参考核验；四个投影不覆盖全部模型 |

先回传 `cpu/summary.txt`；如执行第二步，再带上 `with-dynamic-quant/report.json`。无需回传权重或大 tensor。
