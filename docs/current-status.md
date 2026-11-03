# 当前阶段测试命令

2026-11-03，用户明确接受本轮组合attention完整Draft作为性能优化基线：最差hidden相对L2约2.61%，Top1共7/172不一致，新增KV相对L2约0.44%～0.83%。未写KV逐位保持，两个feedback重复逐位稳定。当前OMC SHA256为`288644ea39913fc49466ed4b6d9329825b6ea819259c266c4601cba391b5e182`。此次接受针对已观测结果；原0.01／0.01逐元素及Top1零差异报告仍为NUMERICAL_FAILED，golden不改，生产未提升。

四个代表case的真实权重CPU审计已由用户执行：两种RMS gamma边界的host_vs_reference均通过，剩余板端差异尚未定位具体算子。按用户要求暂停精度追查，先优化性能。

当前composed-attention-v1为CPUCL=3／NPUCL=2，共5个子图。用户回传同次厂商profile：模型推理627.647ms，算子耗时加和134.940ms，两者相差492.707ms。加载231.187ms、卸载7.652ms另计，表底866.486ms是三项合计。inference begin的`09:47:25.89418`按未补齐六位的微秒字段解释为`.089418`，与结束`.717065`相减吻合。算子start字段不能直接拼成统一执行时间线；差额尚不能归因为CPU、搬运或IPC。此前760.223／773.596ms是混合输入精度调用的Process均值，不与本次单次profile直接相减。

Conv69.038ms、末尾MatMul19.975ms、CPU ArgMax16.199ms是当前算子热点。现有native运行器已一次Load、多次Process，请求HIGH档位，外部输出快照及KV回写不计入Process时间。先检查现有分区边界的数据描述和CPU节点，再决定运行时或图结构实验；Conv空间排布保留为优化候选，尚未实测提速。

## 读取当前OMC的子图边界

在DDK服务器custom_pypto环境运行。复用现有OMC和已回收结果，只读所选DDK的protobuf schema，不调用OMG、不加载厂商库、不访问板子，也无需重新编译。

```bash
git -C "$REPO_DIR" pull --ff-only
python3 "$REPO_DIR/targets/cpro/tools/profile_draft.py" \
  --inspect-boundaries "$AI_RUN_DIR/draft/tests/pytorch-omc" \
  --accuracy-result "$AI_RUN_DIR/draft/tests/pytorch-result" \
  --ddk-root "$CPRO_DDK_ROOT" \
  --output "$AI_RUN_DIR/draft/tests/pytorch-boundaries.json"
```

工具核对OMC完整hash、保留夹具身份及编译打包schema。stdout列出各分区输入／输出的已知逻辑字节数、最大输入、明确的同图producer和CPU算子计数；完整节点信息保留在固定JSON。缺失或动态shape不猜大小，引用无法对齐则标为未解析。CPU task按明确offset／size且图名匹配后读取，NPU task保持不透明。

逻辑字节数不等于实测搬运量：重复消费按端口计数，物理padding、缓冲区复用、同步等待均未知。此检查没有匹配OMG JSON交叉验证，CPU task结果仅作有边界检查的元数据诊断；不能单独确定493ms差额的来源。INSPECTED／PARTIAL不是精度或性能通过。

完整Draft仍为C16／cap192／K15、17输入／14输出，组合attention使用208行逻辑KV。当前只新增诊断入口，尚未修改执行图或实现加速。证据、原数值门槛和后续候选见[当前结果](results.md)。
