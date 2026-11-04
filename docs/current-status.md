# 当前阶段测试命令

目标为C16／cap192／K15常驻预热后平均≤50ms。完整Draft仍为103.720ms／NUMERICAL_FAILED，host embedding准备未计入API。

## Top1前端修正已通过编译，归约落点待核对

用户回传283da71的三个探针全部COMPILED，设备和数值验证尚未运行。

| 探针 | 编译分区 | 当前结论 |
|---|---|---|
| direct | CPUCL 1 | 仍无NPU图 |
| flat | NPUCL 1＋CPUCL 1 | NPU边界FP16→FP16，CPU边界FP32→INT32；尚未证明ArgMax上NPU |
| tiled | NPUCL 2＋CPUCL 2 | 归约链仍混合，存在FP32→FP16边界转换 |

分块版的CPU子图输出INT32和FP32，随后NPU子图输出INT32，末尾CPU接两个INT32输入。这与局部索引／最大值留CPU、获胜块归约落NPU的结构相符，但目前仅是推测。需要CPU节点清单及实际边界shape；不能用NPU_GRAPH_PRESENT或描述中的device_type=NPU直接判定算子归属。

## 只读本轮已编译OMC

同步main后运行，无需重新编译或替换测试槽：

```bash
python3 "$REPO_DIR/targets/cpro/tools/probe_draft_vector.py" \
  --inspect-top1-placement "$AI_RUN_DIR/draft/tests/top1-probes" \
  --ddk-root "$CPRO_DDK_ROOT"
```

每例输出main_nodes（形状、类型、原始连接及GraphOp owner）与cpu_subgraphs.operations（CPU节点名称、类型和描述）。检查器先核对编译状态、kirinx90和保存DDK路径，再绑定report.json中的OMC路径、字节数和SHA256；读取CPU任务前检查范围、不重叠和图owner。NPU任务内部不解码，CPU中缺少某名称不能直接当作NPU映射证明。

本地主机run `20261104T032006Z-4074-bb512d`：70项合成schema／OMC／CLI检查通过，涵盖只读保证、哈希与DDK错配、损坏任务范围、owner和重复节点拒绝，并保持已有完整Draft参考检查输出。未调用所选DDK或设备；这次没有改探针数学表达、fixture或完整Draft。详细记录见[结果](results.md)。
