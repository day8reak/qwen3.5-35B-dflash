# 当前阶段测试命令

两条路线并行：GE试验FP32 `AddCustom`；PyTorch实现同一套增量Draft，权重服务器导出AIR、DDK服务器编译OMC。当前已上板GE模型继续放在公共`draft/air`、`draft/omc`；新候选使用固定测试槽位，验证前不覆盖公共模型。

目标仍为C16／cap192／K15，常驻预热后单次proposal同步`Process`平均≤50ms。当前整模型profiling为518.990ms，目标未达到。已有精度、计时与失败原因见[当前结果](results.md)，算子展开图见[模型结构](model-architecture.md)。

## 1. GE：AddCustom FP32小图

用户回传的DDK注册列出`AddCustom`输入输出支持`float16,float32,int32`，同时有`precision_reduce=true`和JIT-only提示。它不同于此前普通Add的`executor_add_eltwise_half_nd`，必须实际编译和运行。下面保留大值相消输入，不能只看总体L2。

本次AddCustom已在OMG编译阶段退出1，未上板。当前回传只有进程退出摘要，没有具体编译错误；不能据此认定FP32不支持，也不能把之前发现的缺失库直接当成本次根因。**先检查本次保留目录，不重复编译**：

```bash
git -C "$REPO_DIR" pull --ff-only
python3 "$REPO_DIR/targets/cpro/tools/probe_draft_vector.py" \
  --inspect-compile "$AI_RUN_DIR/draft/tests/vector"
```

这个入口只读取已有失败case的`omc/report.json`及编译日志，返回真实stage、returncode和优先错误行／末尾日志；不编译、不上板、不修改产物。回传其中`compilation_diagnostics`即可。尚未更新脚本时也可直接提取：

```bash
tail -n 120 "$AI_RUN_DIR/draft/tests/vector/residual_addcustom_fp32/omc/compile.log"
```

定位并修复具体原因后，才使用下面的原编译命令复测。在DDK服务器沿用已配置的`REPO_DIR`、`AI_RUN_DIR`、`CANN_ROOT`、`CPRO_DDK_ROOT`、`CPRO_DDK_ID`，复用同名已结束目录：

```bash
git -C "$REPO_DIR" pull --ff-only
python3 "$REPO_DIR/targets/cpro/tools/probe_draft_vector.py" \
  --ge-root "$CANN_ROOT" \
  --ddk-root "$CPRO_DDK_ROOT" --ddk-id "$CPRO_DDK_ID" \
  --cases residual_addcustom_fp32 \
  --output "$AI_RUN_DIR/draft/tests/vector" --replace
```

若本例编译为COMPILED，再启动板端小图验证：

```bash
python3 "$REPO_DIR/targets/cpro/tools/validate_draft_vector.py" \
  --compiled "$AI_RUN_DIR/draft/tests/vector" \
  --config "$AI_RUN_DIR/runtime/runtime.json" \
  --ddk-root "$CPRO_DDK_ROOT" \
  --cases residual_addcustom_fp32 \
  --output "$AI_RUN_DIR/draft/tests/vector-run" --apply

python3 "$REPO_DIR/targets/cpro/tools/validate_draft_vector.py" \
  --recover "$AI_RUN_DIR/draft/tests/vector-run" \
  --output "$AI_RUN_DIR/draft/tests/vector-result"

python3 "$REPO_DIR/targets/cpro/tools/validate_draft_vector.py" \
  --analyze "$AI_RUN_DIR/draft/tests/vector-result"
```

检查`precision_boundaries`中的输入降精度Cast与GraphOp dtype，以及`sum`全部输出、相消行误差、重复稳定性和同步API时间。原阈值不变；即使NPUCL归属正确，也不能单凭归属宣称FP32通过。若编译失败，回传该例`omc/compile.log`末尾和`result.json`，先处理JIT／实现选择，不继续重跑已确定降精度的普通Add。

## 2. PyTorch：权重服务器导出AIR

使用能导出Target CPro的PyTorch／torch_npu／TorchAir环境。新模块只装载六层Draft、Target embedding和LM head，不装载40层Target。复用`draft/source/prepared`；如果权重服务器还没有它，先从原始权重准备：

```bash
# 仅在尚无prepared权重时执行；两个变量指向该服务器上的实际源权重。
python3 "$REPO_DIR/targets/cpro/scripts/prepare_draft.py" \
  --target-weights "$TARGET_WEIGHTS" --draft-weights "$DRAFT_WEIGHTS" \
  --output "$AI_RUN_DIR/draft/source/prepared"
```

在权重服务器更新仓库后导出：

```bash
git -C "$REPO_DIR" pull --ff-only
python3 "$REPO_DIR/targets/cpro/scripts/export_incremental_draft.py" \
  --frontend pytorch \
  --prepared "$AI_RUN_DIR/draft/source/prepared" \
  --context-rows 16 --cache-capacity 192 --max-draft-tokens 15 \
  --output "$AI_RUN_DIR/draft/tests/pytorch-air" --replace
```

预期`PYTORCH_DRAFT_AIR_EXPORTED`、17输入、14输出、32个`RmsNorm`。这里用已安装converter的`torch_npu.npu_rms_norm → RmsNorm`（[公开converter](https://github.com/Ascend/torchair/blob/master/python/torchair/_ge_concrete_graph/ge_converter/custom/npu_rms_norm.py)），与Target CPro接口一致；不使用ND工程的`AdnRmsNorm`。Draft gamma直接使用，不加Target零中心权重的1。mask复制与GQA继续使用GE已采用的Gather表达，避免重新引入BOOL查询轴广播和五维KV展开。PyTorch候选在FP32完成gamma乘法后转FP16，与旧GE RMS的中间舍入不同，需要重新做数值验证。

输出整目录：

```text
draft/tests/pytorch-air/
  export.air
  weight/                 # AIR用相对路径引用
  air-schema.pb           # 本次安装的TorchAir schema
  manifest.json
  air-inspection.json
```

把**整个目录**传到DDK服务器相同角色的位置，不能只传`export.air`。服务器根目录可不同，AIR不依赖权重服务器的绝对路径。跨机传输使用现有文件通道；DDK服务器不需要安装TorchAir。两次`--replace`只替换本工具已结束的候选槽位，拒绝活动或未知目录。

## 3. PyTorch：DDK服务器转OMC

确认收到完整`draft/tests/pytorch-air`后：

```bash
git -C "$REPO_DIR" pull --ff-only
python3 "$REPO_DIR/targets/cpro/scripts/compile_incremental_draft.py" \
  --air-export "$AI_RUN_DIR/draft/tests/pytorch-air" \
  --ddk-root "$CPRO_DDK_ROOT" --ddk-id "$CPRO_DDK_ID" \
  --output "$AI_RUN_DIR/draft/tests/pytorch-omc" --replace
```

编译自动识别PyTorch AIR。使用外置权重编译后流式打包成`pytorch-omc/model.omc`，避免重走大模型内嵌序列化失败路径。此候选编译器需要输入快照、编译权重与打包模型同时存在，预检为额外三份权重大小＋1GiB，编译临时空间可能更大；不会把旧的6GiB估计用于这里。

预期状态为`PYTORCH_DRAFT_OMC_COMPILED`，但`compiled_abi=NOT_BOUND`、`compiled_precision=NOT_VALIDATED`。检查`compiled-interface.json`的子图归属和边界Cast，再绑定实际输入输出槽位、做Load及独立KV验证。这一候选尚不能直接传给现有`--incremental-draft`。此前内置RmsNorm小图有Load失败，换PyTorch前端不保证自动解决。

## 4. 核对当前Target AIR的RMS

需要对比实际Prefill／Decode的RMS类型、端口dtype、shape和epsilon时，读取规范化后的公共AIR即可，不编译、不上板，也不需要`rg`：

```bash
python3 "$REPO_DIR/targets/cpro/tools/probe_draft_vector.py" \
  --inspect-rms "$AI_RUN_DIR/prefill/air/export.air" \
                "$AI_RUN_DIR/decode/air/export.air"
```

默认使用仓库的TorchAir schema；若源AIR使用不同schema，可传`--torchair-pb2`指定实际导出环境中的文件。输出按RMS类型／描述分组；源AIR端口不等于编译内核精度证明。

## 已完成的主机检查

AddCustom探针及分析入口7项检查通过。PyTorch模型5项检查通过：K15／7／3／1共37步独立前缀对照，Top1无差异、KV最大差0、hidden最大绝对差0.001953125；覆盖零新增行、拒绝换anchor、重复、无效padding和容量边界。全宽16384→2048入口确认是4×4完整dot表达。另6项检查覆盖可移植AIR、源码RMS识别、替换保护及导出／编译／打包错误传播。

这些是CPU数值、`torch.export`捕获与模拟工具边界检查；实际TorchAir导出、OMG编译、板端精度和性能尚待上述命令验证。成功后继续减少CPU/NPU分段、优化入口投影与词表head，并重新测量同输入的完整Draft。
