# 当前阶段测试命令

目标为C16／cap192／K15常驻预热后平均≤50ms。完整Draft仍为103.720ms／NUMERICAL_FAILED，host embedding准备未计入API。

## Top1局部归约已确认在CPU，本轮成组验证形状和组合

用户回传的已编译OMC检查确认：direct和flat的ArgMaxExt2均在CPU；tiled的local_argmax与local_value（ReduceMax）也都在CPU，输入仍是完整FP32[15,485,512]。tiled另有NPU边界FP16[15,485]→INT32[15]，内部算子未解码。当前结构尚未消除完整词表的CPU归约。

本轮固定21项，用同一条命令编译并自动检查落点：17个独立归约，4个完整词表组合。复用`draft/tests/top1-probes`，保留完整Draft和已有性能基线。

| 对照 | 覆盖 | 要回答的问题 |
|---|---|---|
| ArgMaxExt2与ReduceMax分别运行 | 每个几何使用相同输入 | 哪个归约单独落CPU，是否由组合影响 |
| 行数 | 宽512：1／15／485／7275行 | 小批量与完整局部数据量有何区别 |
| 归约长度 | 宽485：15／7680行；宽512的上述对照；单独ReduceMax宽248320 | 小轴长度及总元素量是否影响放置 |
| rank | [7275,512]与[15,485,512]，输入字节一致 | 只改变前导维度的影响 |
| 轴方向 | [512,7275]沿axis0，为二维输入的精确转置 | 非末轴归约能否改变放置 |
| 完整组合 | 原3D控制、局部展平、局部转置、512×485分块 | 单算子结果能否保留到完整Top1链 |

[7275,512]和[7680,485]均为3724800元素，完整组合均保留248320词表、无padding。已经确认的全宽CPU ArgMax不重复新增测试。独立算子使用有限FP16值，覆盖并列、负数、零与正子正规数；完整组合保留NaN、Inf、全相等、跨块并列及65535／65536／大索引。485宽组合的跨块并列行改为484／485，golden仍由直接全词表NumPy ArgMax生成并与PyTorch交叉检查。

## 一次执行

沿用上一轮已成功编译的环境：`REPO_DIR`是代码仓库，`AI_RUN_DIR`是运行目录，`CANN_ROOT`是已有GE环境，`CPRO_DDK_ROOT`和`CPRO_DDK_ID`指向所选X90 DDK。同步main后执行：

```bash
python3 "$REPO_DIR/targets/cpro/tools/probe_draft_vector.py" \
  --top1-matrix \
  --ddk-root "$CPRO_DDK_ROOT" \
  --output "$AI_RUN_DIR/draft/tests/top1-probes" --replace
```

命令逐例编译，单例编译失败会记录诊断并继续其余例。随后在同一目录锁内自动读取成功编译的OMC，终端输出各例编译状态、CPU操作、GraphOp实际输入输出和owner。无需再跑一轮只读命令；贴回最后JSON即可。

`result.json`保存原编译报告，`placement-inspection.json`保存完整检查及源报告哈希，逐例`omc/compile.log`保留原日志。某例失败仍明确标出，不能把部分覆盖算整体通过。NPU分区数量不证明指定归约的NPU实现；下一步按CPU操作和完整输入边界比较17个独立图与4个组合，再选候选做设备数值和时延验证。

本地主机检查450项通过：57次图语义回放（其中40次完整词表组合）、210次模拟native回收、真实ABI准备入口的模拟元数据绑定，以及自动编译／检查调度。原六图的manifest、C++和fixture字节保持。独立ReduceMax的FP32结果也要求精确字节相等，差一个ULP会失败；原其他浮点门槛不变。此处未运行所选DDK或设备，详见[结果](results.md)。
