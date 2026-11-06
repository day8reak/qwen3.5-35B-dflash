# 当前阶段：以约60ms的packed Draft为基线继续优化

2026-11-05：完整模型对照的488个区域逐字节一致，覆盖isolated／feedback原两序列的有效Top1、hidden和完整KV。38次调用按次数加权，准备＋API从66.4573ms降到60.7651ms，减少5.6922ms（8.565%）。用户已选择这份packed模型作为后续性能基线，继续暂缓精度排查；原数值状态仍为FAILED。[完整结果](results.md)。

| 模式／阶段 | baseline ms | packed ms | 净省 ms |
|---|---:|---:|---:|
| isolated catchup，11次 | 66.6092 | 60.9815 | 5.6276 |
| isolated proposal，8次 | 66.3970 | 61.2258 | 5.1713 |
| feedback catchup，11次 | 66.4160 | 60.1885 | 6.2275 |
| feedback proposal，8次 | 66.3655 | 60.7998 | 5.5657 |

当前OMC为 `draft/tests/top1-integration/feature-pack/compiled/model.omc`，SHA256 `1105ce436ed4dd889ce4c26d0a5eeb620382b51ec5c98d8b810e3875af37600b`。计时包含端侧CPU打包及输入设置／复用和同步API，排除文件I/O、诊断回读、快照、commit、加载；不是整轮端到端或预热后十次重复的基准。常驻入口仍为旧token-ID接口，不能仅改OMC路径就使用host embedding＋packed候选。

## 2×2投影实验结束，保留当前基线

2026-11-06用户回传：六组执行／回收成功、锁已释放，输出有限且重复稳定。每组预热3次、记录10次，30对设备输出均不满足逐字节等价；原CPU参考门槛也仍失败。

| source case | 当前4×4准备＋API ms | 2×2准备＋API ms | 净省 ms |
|---|---:|---:|---:|
| 2 | 10.5075 | 10.8774 | −0.3699 |
| 16 | 10.8752 | 10.9697 | −0.0945 |
| 17 | 10.9199 | 10.9672 | −0.0473 |

三组平均10.7675→10.9381ms，观测总耗时增加0.1706ms。API平均省0.1761ms，CPU打包平均增加0.3623ms；测量范围重叠，不据此声称统计显著变慢，但没有值得整图集成的净收益。2×2对CPU参考的L2略低，不能把逐字节差异说成精度恶化，也不据此推翻当前不集成的决定。停止这条路线，不重跑或编译完整2×2 Draft。[详细结果](results.md)。

## 下一步只读检查完整词表head

当前packed profile中，inference为60.214ms，head方向的 `MatMul_30` 为19.960ms，特征投影 `Conv2D` 为8.691ms，StableArgMax为0.416ms；原入口 `Transpose`／`Transpose_1` 已消失。算子duration合计56.223ms不是关键路径，NPU start晚于API inference end，不能跨表推算等待时间。表格按本轮case18请求关联，未在本地核对原CSV及OMC hash。

源码已经把权重转置折入MatMul属性，下一步核对当前导出AIR、真正送入编译器的prepared AIR和保留IR中的head矩阵方向／格式，避免重复已做的转置优化。检查器通过完整词表权重和Top1消费路径定位节点，不依赖 `MatMul_30` 这个数字后缀。所有15行、248320个词表列、每个logit的2048项点积都属于当前声明合同；不从某次候选数较少推导可删除的词表。

沿用服务器环境运行：

```bash
cd "$REPO_DIR" && git pull --ff-only origin main
TOP1_DIR="$AI_RUN_DIR/draft/tests/top1-integration"
FULL_PACK_DIR="$TOP1_DIR/feature-pack"
HEAD_DIR="$AI_RUN_DIR/draft/tests/head-layout"
mkdir -p "$HEAD_DIR"
python3 "$REPO_DIR/targets/cpro/tools/inspect_draft_head.py" \
  --air-export "$TOP1_DIR/export" --compiled "$FULL_PACK_DIR/compiled" \
  --ddk-root "$CPRO_DDK_ROOT" > "$HEAD_DIR/inspection.json"
cat "$HEAD_DIR/inspection.json"
```

此步不连接板端、不启动编译、不读取或复制约1.017GB的head权重，只输出小型JSON。报告的OMC hash来自已有packing记录，不重新计算；源AIR及prepared AIR核对原编译记录。`source`／`compiler_input`是编译前图，`retained_compiler_ir`单独列出可读IR及缺失部分，不能混写成最终内核实现。没有IR时返回PARTIAL，仍保留源图检查结果，不自动重新编译。

交回该JSON后，依据实际矩阵方向选择一个小head表达对照；当前不铺开大模型构建。服务器最近剩余空间仍只有用户先前报告的1.7G，未收到实际清理字节数；下一次编译前须重新核对。原数值门槛及60ms性能基线保持，常驻接入与端到端加速尚未完成。
