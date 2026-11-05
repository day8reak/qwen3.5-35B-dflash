# 当前阶段：以约60ms的packed Draft为基线继续优化

2026-11-05：完整模型对照的488个区域逐字节一致，覆盖isolated／feedback原两序列的有效Top1、hidden和完整KV。38次调用按次数加权，准备＋API从66.4573ms降到60.7651ms，减少5.6922ms（8.565%）。用户已选择这份packed模型作为后续性能基线，继续暂缓精度排查；原数值状态仍为FAILED。[完整结果](results.md)。

| 模式／阶段 | baseline ms | packed ms | 净省 ms |
|---|---:|---:|---:|
| isolated catchup，11次 | 66.6092 | 60.9815 | 5.6276 |
| isolated proposal，8次 | 66.3970 | 61.2258 | 5.1713 |
| feedback catchup，11次 | 66.4160 | 60.1885 | 6.2275 |
| feedback proposal，8次 | 66.3655 | 60.7998 | 5.5657 |

当前OMC为 `draft/tests/top1-integration/feature-pack/compiled/model.omc`，SHA256 `1105ce436ed4dd889ce4c26d0a5eeb620382b51ec5c98d8b810e3875af37600b`。计时包含端侧CPU打包及输入设置／复用和同步API，排除文件I/O、诊断回读、快照、commit、加载；不是整轮端到端或预热后十次重复的基准。常驻入口仍为旧token-ID接口，不能仅改OMC路径就使用host embedding＋packed候选。

## 先清运行盘，再采集当前模型热点

用户报告仅剩1.7G。以下清理保留当前packed编译目录、所有OMC、原始AIR／权重、回放输入／golden和已回收输出。第一步删除已嵌入OMC的外置权重重复副本；第二步核对完整packed等价、两次任务完成及释放锁记录，再删除上一版RoPE的 `compilation/compile-work` 和 `compilation/tbe-compat/wheel-python`。若仍被当前导出／编译输入或运行配置引用则拒绝。旧RoPE的保留fixture仍可回放、对比和profiling；重新做该旧编译目录的ABI绑定需要重建已删除的编译副本。

沿用服务器现有环境，不重编模型或runner。先更新脚本并设置目录：

```bash
cd "$REPO_DIR" && git pull --ff-only origin main
TOP1_DIR="$AI_RUN_DIR/draft/tests/top1-integration"
FULL_PACK_DIR="$TOP1_DIR/feature-pack"
```

清理成功后才启动一次当前OMC的case18 profiling；`&&`保证前一步失败时不会继续。清理结果中的 `free_bytes_after` 是实测剩余空间，不能把估算的副本大小当成已释放量。

```bash
python3 "$REPO_DIR/targets/cpro/tools/clean_draft_probes.py" \
  --root "$AI_RUN_DIR" --config "$AI_RUN_DIR/runtime/runtime.json" \
  --integration-sidecars --apply &&
python3 "$REPO_DIR/targets/cpro/tools/clean_draft_probes.py" \
  --root "$AI_RUN_DIR" --config "$AI_RUN_DIR/runtime/runtime.json" \
  --retired-rope-scratch --apply &&
python3 "$REPO_DIR/targets/cpro/tools/profile_draft.py" \
  --accuracy "$FULL_PACK_DIR/result" --case-index 18 \
  --allow-numerical-failure --output "$FULL_PACK_DIR/profile-run" --apply
```

启动确认后回收同一次请求，不重新启动模型：

```bash
python3 "$REPO_DIR/targets/cpro/tools/profile_draft.py" \
  --recover "$FULL_PACK_DIR/profile-run" --output "$FULL_PACK_DIR/profile-result"
```

profiling入口按物理语义槽位识别 `target_features`，将原逻辑fixture转成独立的512KiB物理字节文件，其他输入直接复用。服务器离线打包不计入vendor模型／算子耗时；完整模型收益仍以上表含端侧打包的对照为准。工具复用已部署、哈希匹配的OMC，不上传另一份大模型。交回清理摘要和本次 `*_model.csv`、`*_op.csv` 即可。

本地主机检查：11项profiling适配检查（含全部元素映射、与实际C++打包函数逐字节对照、语义槽位、原fixture保持和恢复不重打包），9项清理检查（含实际删除范围、重复执行、活动锁／配置依赖保护）。未连接用户服务器或设备，实际释放空间及新profiling待回传。

## 后续优化顺序

先确认原入口转置热点是否消失，再按新CSV检查约8.8ms的特征投影和约19.8ms的完整词表head；这两个数字来自上一版RoPE模型，不能直接当成当前热点耗时。

- 特征投影权重约64MiB，优先考虑少量前缀对照，保留完整K点积和FP16边界。旧DDK路径曾拒绝MatMul的K=16384，不直接退回该失败表达。
- 完整词表head权重约1.017GB，先确认布局／实现和实测成本；不先铺开多个完整模型。
- 零新增KV专用proposal在当前合并追加的调度中覆盖有限，继续暂缓。StableArgMax旧测仅0.416ms，同样低优先级。

继续使用原词表、层数、权重和精度门槛；不以这次加速替代数值验收，也不将尚未实现的常驻接入或端到端加速写成完成。
