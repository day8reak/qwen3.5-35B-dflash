# 当前阶段：以约60ms的packed Draft为基线继续优化

2026-11-05：完整模型对照的488个区域逐字节一致，覆盖isolated／feedback原两序列的有效Top1、hidden和完整KV。38次调用按次数加权，准备＋API从66.4573ms降到60.7651ms，减少5.6922ms（8.565%）。用户已选择这份packed模型作为后续性能基线，继续暂缓精度排查；原数值状态仍为FAILED。[完整结果](results.md)。

| 模式／阶段 | baseline ms | packed ms | 净省 ms |
|---|---:|---:|---:|
| isolated catchup，11次 | 66.6092 | 60.9815 | 5.6276 |
| isolated proposal，8次 | 66.3970 | 61.2258 | 5.1713 |
| feedback catchup，11次 | 66.4160 | 60.1885 | 6.2275 |
| feedback proposal，8次 | 66.3655 | 60.7998 | 5.5657 |

当前OMC为 `draft/tests/top1-integration/feature-pack/compiled/model.omc`，SHA256 `1105ce436ed4dd889ce4c26d0a5eeb620382b51ec5c98d8b810e3875af37600b`。计时包含端侧CPU打包及输入设置／复用和同步API，排除文件I/O、诊断回读、快照、commit、加载；不是整轮端到端或预热后十次重复的基准。常驻入口仍为旧token-ID接口，不能仅改OMC路径就使用host embedding＋packed候选。

## 当前profile与下一步

2026-11-06用户回传packed模型表：model inference **60.214ms**，入口原有的 `Transpose`／`Transpose_1` 已不在算子表中。`MatMul_30` 为19.960ms，`Conv2D` 为8.691ms，StableArgMax为0.416ms。算子duration合计56.223ms，不能当作关键路径；NPU start晚于API inference end，不能跨表计算等待时间。表格未附请求身份／OMC hash，按本轮case18请求关联，未在本地读取原CSV。

下一步只比较特征投影前缀：当前packed **4×4／1024通道**与 **2×2／4096通道**。两者每个输出都保留全部16384项点积、原权重字节、16个token和2048通道及原FP16输出边界；没有缩减词表或有效行。2×2只是另一种卷积表达，实际DDK支持、累加舍入和性能尚待验证。

先做这个约64MiB权重的两模型实验，比直接改约1.017GB权重的head更节省构建空间。主机已检查全部262144项索引映射、实际C++打包字节、两侧Data描述符和AIR改写、旧4×4兼容性、六组计划及失败重试，共11项通过；没有本地DDK编译或设备执行结果。AIR编码沿用导出schema，类型按[固定版本TorchAir协议](https://github.com/Ascend/torchair/blob/e1118e8b0fc1b95190ab53b732c0872989550e20/proto/ge_ir.proto)检查。

## 先清运行盘，再编译两个小前缀

用户报告仅剩1.7G。以下清理保留当前packed编译目录、所有OMC、原始AIR／权重、回放输入／golden和已回收输出。第一步删除已嵌入OMC的外置权重重复副本；第二步核对完整packed等价、两次任务完成及释放锁记录，再删除上一版RoPE的 `compilation/compile-work` 和 `compilation/tbe-compat/wheel-python`。若仍被当前导出／编译输入或运行配置引用则拒绝。旧RoPE的保留fixture仍可回放、对比和profiling；重新做该旧编译目录的ABI绑定需要重建已删除的编译副本。

沿用服务器现有环境。先更新脚本并设置目录：

```bash
cd "$REPO_DIR" && git pull --ff-only origin main
TOP1_DIR="$AI_RUN_DIR/draft/tests/top1-integration"
FULL_PACK_DIR="$TOP1_DIR/feature-pack"
PROJECTION_DIR="$AI_RUN_DIR/draft/tests/feature-projection"
```

清理成功后才编译；已清理的目录可重复执行。清理结果中的 `free_bytes_after` 是实测剩余空间，本地尚未收到这个值。探针编译前至少需1GiB可用空间，编译scratch可能另需空间；每个成功模型会回收自己的重复编译副本，不生成新的1.79GB完整Draft。

```bash
python3 "$REPO_DIR/targets/cpro/tools/clean_draft_probes.py" \
  --root "$AI_RUN_DIR" --config "$AI_RUN_DIR/runtime/runtime.json" \
  --integration-sidecars --apply &&
python3 "$REPO_DIR/targets/cpro/tools/clean_draft_probes.py" \
  --root "$AI_RUN_DIR" --config "$AI_RUN_DIR/runtime/runtime.json" \
  --retired-rope-scratch --apply &&
python3 "$REPO_DIR/targets/cpro/tools/probe_draft_feature_pack.py" \
  --air-export "$TOP1_DIR/export" --result "$FULL_PACK_DIR/result" \
  --projection-kernel 2 \
  --ddk-root "$CPRO_DDK_ROOT" --ddk-id "$CPRO_DDK_ID" \
  --tbe-diagnosis "$AI_RUN_DIR/common/tbe-diagnosis" \
  --output "$PROJECTION_DIR/compiled"
```

只有两个前缀均为COMPILED才执行下面一步。新2×2输入需要重建native runner；当前完整OMC保持为性能基线。

```bash
python3 "$REPO_DIR/targets/cpro/scripts/build_hiai.py" \
  --output "$AI_RUN_DIR/runtime/native" --replace &&
python3 "$REPO_DIR/targets/cpro/tools/validate_draft_vector.py" \
  --compiled "$PROJECTION_DIR/compiled" \
  --config "$AI_RUN_DIR/runtime/runtime.json" --ddk-root "$CPRO_DDK_ROOT" \
  --output "$PROJECTION_DIR/run" --apply
```

启动确认后回收同一次请求；未完成时继续recover，不重新apply：

```bash
python3 "$REPO_DIR/targets/cpro/tools/validate_draft_vector.py" \
  --recover "$PROJECTION_DIR/run" --output "$PROJECTION_DIR/result"
```

编译失败先看 `models[].compilation_diagnostics`。仅有已记录的OMG失败退出、没有运行中的任务时，可对原编译命令加 `--retry-failed`，保留成功模型和fixture。不要删除整个测试树或重跑完整模型编译。

## 对照判定

使用原case2／16／17的逻辑输入和同一CPU参考，每种实现各预热3次、记录10次。报告沿用 `feature_pack` 字段，`experiment=full-dot-kernel2-v1`，baseline为当前packed4×4，candidate为kernel2。检查30对输出全部逐字节一致、有限、重复稳定，再比较各组 `prepared_call_us`：它包含预分配CPU打包、SetInput和同步API，排除快照与加载。CPU参考FAILED单独保留，不能被baseline等价覆盖。

交回清理摘要和验证JSON即可判断这条路线是否值得整图集成。前缀有净收益才继续完整模型；输出出现新差异或没有净收益则保留当前60ms基线。完整词表head为下一候选；零新增KV专用proposal和StableArgMax继续低优先级。尚无匹配X90能力实测画像，不推断带宽瓶颈或可实现的加速上限。
