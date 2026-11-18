# Target T8 Prefill：定位投影与缓存链路

当前同次 T8 捕获的完整 Linear／Full Attention 均有数值偏差；先复用现有输入和 golden 检查连接，再用五张小图区分投影计算与缓存写入。无需重新生成整网 Prefill。

环境与变量沿用 [Attention 分层验证](target-attention-pair.md)：`GOLDEN_CAPTURE` 必须指向本次 T8 的 `prefill-fixture/capture`，`CPRO_SOURCE` 仍是保留原调用的私有 CPro 源码。导出适配由工具自动应用，原源码与 golden 不改写。先同步 `main`。

## 1. 离线检查捕获连接

只需要 Python 和 NumPy，不加载 NPU、不编译模型。输出目录必须尚不存在。

```bash
export PAIR_ROOT="${PAIR_ROOT:-$AI_RUN_DIR/target/tests/attention-pair}"
python "$REPO_DIR/targets/cpro/tools/target_layer_probe.py" inspect-prefill-links \
  --capture "$GOLDEN_CAPTURE" \
  --output "$PAIR_ROOT/prefill-links"
cat "$PAIR_ROOT/prefill-links/summary.txt"
```

检查 GDN 捕获的 QKV→concat/tail，以及 Full Attention 的 hidden→V 投影输入、V 投影输出→CacheUpdate updates、cache 输出→core 输入→模块返回。另将顺序写入缓存的 NumPy 数学假设与捕获缓存比较；它不替代安装内核的 ABI 说明。发现不一致先看报告，不覆盖 golden。全部一致只说明这些连接成立，不证明 QLinear、Attention 或整网数值正确。

在保存 DDK、已有 AIR／OMC 和执行 fixture 的编译主机，还可以只读核对完整模块中每个 QLinear 的 weight／scale 配对：

```bash
for group in linear full-fp32-pos; do
  python "$REPO_DIR/targets/cpro/tools/target_layer_probe.py" inspect-qlinear \
    --batch "$PAIR_ROOT/$group/omc" \
    --fixture "$PAIR_ROOT/$group/execution/fixture" \
    --ddk-root "$CPRO_DDK_ROOT" \
    --output "$PAIR_ROOT/$group/qlinear-inspection"
  cat "$PAIR_ROOT/$group/qlinear-inspection/summary.txt"
done
```

这一步不调用编译器。旧导出记录中的投影候选没有语义名称，工具按匿名候选组核对完整常量字节和 x2／scale 配对；不因此声称已证明 Q/K/V 的语义接线或编译后内部存储正确。文件缺失或不透明的中间链会保留未确认状态。

## 2. 导出五张小图

在此前成功导出 AIR 的 CPro Python 环境执行。使用同一次 T8 捕获的三层 QKV 投影、L3 V 投影和 L3 V CacheUpdate；独立 CacheUpdate 直接使用捕获的 updates／block／offset，不经过 RoPE 或浮点索引转换。

```bash
export DIAG_ROOT="$PAIR_ROOT/projection-t8"
python "$REPO_DIR/targets/cpro/tools/target_layer_probe.py" export \
  --capture "$GOLDEN_CAPTURE" \
  --source-root "$CPRO_SOURCE" --target-config "$TARGET_CONFIG" \
  --converter-script "$CPRO_EXPORT_SCRIPT" \
  --points linear_attn_in_proj_qkv self_attn_v_proj cache_v \
  --phases prefill --qlinear-scales fp16 \
  --output "$DIAG_ROOT/air"
```

预期 5 个 case 全部 `EXPORTED`。若已有**本次同一捕获**的这些小图结果，可直接使用其比较报告，不必重复执行；旧 T256／T1 结果不覆盖 T8。

## 3. 编译、执行与比较

编译主机沿用已验证的 DDK 环境：

```bash
python "$REPO_DIR/targets/cpro/tools/target_layer_probe.py" compile \
  --batch "$DIAG_ROOT/air" --output "$DIAG_ROOT/omc" \
  --ddk-root "$CPRO_DDK_ROOT" --ddk-id "$CPRO_DDK_ID" \
  --compiler-profile isolated-kirinx90 \
  --tbe-diagnosis "$TBE_DIAGNOSIS" --use-origin-format
```

确认 5 个 case 均为 `COMPILED` 后再上板；部分编译成功不能覆盖全部诊断。上板主机沿用 [HDC 与 runtime.json 配置](target-batch-rerun.md#4-上板与回收)。跨主机时复制整个 `projection-t8`，保留 `air/omc` 相对位置，并重新设置 `DIAG_ROOT`：

```bash
export RUNTIME_CONFIG="${RUNTIME_CONFIG:-$AI_RUN_DIR/runtime/runtime.json}"
python "$REPO_DIR/targets/cpro/tools/validate_draft_vector.py" \
  --compiled "$DIAG_ROOT/omc" --config "$RUNTIME_CONFIG" \
  --ddk-root "$CPRO_DDK_ROOT" --output "$DIAG_ROOT/execution" --apply

python "$REPO_DIR/targets/cpro/tools/validate_draft_vector.py" \
  --recover "$DIAG_ROOT/execution" --output "$DIAG_ROOT/recovery"

python "$REPO_DIR/targets/cpro/tools/target_layer_probe.py" compare \
  --fixture "$DIAG_ROOT/execution/fixture" \
  --native "$DIAG_ROOT/recovery/native" \
  --output "$DIAG_ROOT/comparison"

cat "$DIAG_ROOT/comparison/summary.csv"
```

`PENDING` 只重复 recover；已有 native 输出时，即使精度失败也继续 compare。保持原精度门槛。

| 结果 | 下一步 |
| --- | --- |
| 离线 V 投影→updates 不一致 | 查参考端捕获时序／布局，暂不把问题归给 CPro |
| 顺序写入假设与捕获缓存不一致 | 查原 CacheUpdate 的更新语义／数据；保留原 golden |
| T8 独立投影失败 | 检查同次 DynamicQuant、scale、packed weight 与短序列实现；不先改 Attention 核心 |
| V 投影通过，独立 CacheUpdate 失败 | 检查 T8 updates、写入行和缓存布局 |
| 五张小图与离线连接均通过，完整模块仍失败 | 对比整图内部边、输出槽位及存储复用；小图通过不能排除串联问题 |

先回传离线检查的 `summary.txt`；若离线链路及权重配对没有发现异常，再执行五图并回传 `projection-t8/comparison/summary.csv`。有异常时同时保留对应 `report.json`。当前根因未确认，不据这些诊断修改已捕获参考或放宽阈值。
