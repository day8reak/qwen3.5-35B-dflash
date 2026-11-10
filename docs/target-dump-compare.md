# 四层 Target dump 批量比较

当前工具只进行主机离线分析，不调用 PyTorch/NPU、不编译 OMC、不连接设备。用户确认：**DDK／OMG 编译在“35”服务器，310P 推理和 golden 采集仍在此前的权重服务器**，沿用已跑通的四层 inference 入口。“35”指服务器，不是 Python 环境名；此前的 147 服务器仅是参考命令来源。CPro dump 尚无样例，因此当前交付是**可测试的比较核心和显式映射清单**；设备工具格式适配、自动节点映射仍待真实样例。

## 310P golden 从哪里来

沿用 [inference_target_steps.py](../targets/cpro/tools/inference_target_steps.py) 调用原 Target 的加载、W4 量化、prefill 和 decoder；原模型 forward 不改写。采集器使用 hook 和调用包装，保存真实执行中的输入输出，不把 CPro 输出或 CPU 重算值当作 golden。已有同输入的完成态 capture 可直接复用。

这不等于自动得到所有底层算子的内部张量。现有采集覆盖选定边界，包含 Norm、投影、Conv/GDN、状态、Attention 及 residual；MoE 内部和融合内部需要另行增加观测。各层 residual_mlp 的输出可用于观察对应层输出。wrapper 输出虽另有保存，当前映射生成器只导入 manifest.regions，尚不代替最终模型输出验收。

运行条件见[四层采集说明](target-layer-accuracy.md#备选入口直接调用原-inference-的步骤)。当前步骤入口至少执行一次 decode；比较四层 Prefill 时，可从映射清单中选出 phase=prefill 的完整 region，报告会列出省略的 region 数，不会把未比较 Decode 记为通过。

T64 OMC 必须配同用例的 T64 golden；T256 capture 不能简单裁切作为替代。现有 --input-ids 接口接受无 padding 的 INT32 batch=1 token 文件，不代表已支持任意 padding/mask 合同。

## 使用方式

输出都位于 AI_RUN_DIR，原 golden 和设备 dump 只读；每次选新的输出目录。脚本依赖现有 NumPy，无需新增 DDK 或安装另一套环境。

先从已有 capture 创建映射模板。GOLDEN_CAPTURE 是含 manifest.json、regions/ 的目录；直接步骤入口通常在其输出的 capture/ 下：

```bash
python "$REPO_DIR/targets/cpro/tools/compare_target_dump.py" prepare \
  --capture "$GOLDEN_CAPTURE" \
  --output "$AI_RUN_DIR/target/tests/four-layer-compare/map-001"
```

生成 mapping.json，保留层号、调用、阶段、原始顺序、golden 路径／shape／dtype／哈希。所有 actual 初始为 null，布局和合同保持未确认。它不会按照文件名、大小或 shape 自动配对。

根据真实节点、端口和布局证据填写映射，随后执行：

```bash
python "$REPO_DIR/targets/cpro/tools/compare_target_dump.py" compare \
  --mapping "$AI_RUN_DIR/target/tests/four-layer-compare/map-001/mapping.json" \
  --golden-root "$GOLDEN_CAPTURE" \
  --actual-root "$CPRO_DUMP" \
  --output "$AI_RUN_DIR/target/tests/four-layer-compare/compare-001"
```

可以先只映射少数边界，其余保留 null；工具仍输出已比较项及未覆盖项。也可以选择完整 region 子集，但不能删除已选 region 的输入／输出端口来绕过缺失。

输出：

- report.json：全部张量指标、覆盖缺失、输入／输出诊断，以及候选区域。
- summary.csv：按捕获顺序列出层、调用、阶段、输入／输出、误差及状态，可按层筛选。
- local_candidates：合同确认、输入完整且逐位一致、输出明确失败的区域。仍只是候选，不能替代权重／属性和实现核对。
- first_observed_output_failure：按原 capture 顺序最早看到的失败输出。没有核对编译拓扑，不宣称它就是整图根因。

## mapping.json 中需要填写的内容

每个 region 的 inputs/outputs 元素包含 name、golden、actual、gate。golden 原有 path/shape/dtype/bytes/sha256 必须保留，布局需按实际语义确认。下面只展示一个 actual 描述的格式，文件名和尺寸是示例：

```json
{
  "path": "dump_relative_path.bin",
  "format": "raw",
  "dtype": "float16",
  "shape": [1, 64, 2048],
  "layout": "ND"
}
```

path 相对 actual-root。raw 必须是无头、小端数组；npy 使用 format=npy，脚本核对 NPY 头中的 shape/dtype，禁止 pickle。可填 bytes/sha256 绑定文件；未填写时仍记录实际大小和 SHA256。支持 float16/32/64、常用有符号／无符号整数和 bool，不支持直接把 bfloat16 或带头私有格式当 float16 读取。

golden.layout 默认为 UNVERIFIED。确认其轴和数据含义后才能改为 logical 或 ND；已捕获 shape 并不证明该张量是逻辑 ND，例如 attention_nz。物理 shape 描述的是文件中的数组；两边经过显式视图后必须具有相同 shape。

仅支持经确认的 permute（轴重排）和 slice（有效范围裁剪）。view.permute 是完整轴排列；view.slice 按重排后的每一维填写 [start, stop]，同时提供非空 view_evidence。不自动 squeeze、reshape、广播、转 dtype 或解包。NZ、packed INT4、编码 scale 需要经验证的解码／映射适配，当前版本明确拒绝未知布局。

合同与覆盖字段：

| 字段 | 何时设置 |
|---|---|
| contract_confirmed=true | 已核对同权重、输入、长度、mask、位置、初态、阶段及调用；contract_evidence 写对应记录／哈希 |
| region.inputs_complete=true | 该区域所需数据、控制和状态输入均已覆盖；mapping_evidence 写编译节点／端口映射依据 |
| pair.gate | 每个输入／输出各自的比较规则，不能由观察到的误差反推宽松阈值 |

布尔字段是显式声明，脚本不会把文字说明当成已经独立验证的模型合同。两边文件有哈希也不能证明加载了同一 checkpoint。

## 指标和结论

浮点默认 gate.mode=metrics，只记录 max-abs、RMSE、relative-L2、cosine、非有限值和不一致坐标，不自动判 PASS。需要验收时使用 mode=allclose 并声明非负有限的 atol、rtol，判定为 abs(actual-golden) <= atol + rtol*abs(golden)。mode=bitwise 要求相同 dtype，逐位核对，包括正负零。

整数控制量始终精确比较，不受浮点容差放宽；int64 不先转 float64 再判断，避免大整数差异消失。NaN/Inf 即使在 metrics 模式也失败。零参考范数对应非零误差时 relative-L2 为 null，RMSE/max-abs 仍保留；不能把 null 当零。

| 状态／诊断 | 含义 |
|---|---|
| UNMAPPED / MISSING_FILE / INVALID_TENSOR / EMPTY_TENSOR | 未映射、缺文件、哈希／格式／布局异常或空张量；整体覆盖 PARTIAL |
| METRICS_RECORDED | 数值有限，但缺少验收阈值或合同／输入覆盖确认 |
| FAILED | 非有限值、整数不一致，或违反显式数值／逐位门槛 |
| LOCAL_CANDIDATE_EXACT_INPUTS | 输入逐位一致、输入覆盖和合同已声明完整，输出失败 |
| OUTPUT_FAILURE_WITH_INPUT_DIFFERENCES | 输出失败且输入已有差异，可能包含误差传播 |
| OUTPUT_FAILURE_INPUTS_OR_CONTRACT_UNVERIFIED | 输出失败，但不能确认局部输入／状态／合同 |
| MAPPED_BOUNDARIES_PASSED | 所选映射边界均满足显式门槛，合同与输入覆盖已声明；不是整模型验收 |

输入在 allclose 容差内也不等同完全相同输入，不会据此自动晋升为局部候选。转轴或裁剪后的逻辑坐标用于报告；逐位模式另列 gate_mismatch_count，避免正负零被“数值相等”掩盖。

脚本用只读 mmap 和分块统计，不把整套 dump 同时转为 float64。每个文件计算一次 SHA256；首次比较仍需顺序读取文件。未知／损坏张量会逐项报告，其余边界继续比较。

CLI 退出码：0 表示生成模板或完成无明确失败的比较（可能仍为 METRICS_RECORDED，**不等于精度通过**）；1 表示 FAILED/PARTIAL；2 表示映射或调用请求无效。

## 还需要的信息

当前已确认：DDK 编译使用 35 服务器；310P 使用此前的权重服务器和已跑通的四层推理入口。离线比较可放在能够读取两份 dump 的主机上，不依赖 NPU。现在无需执行板端任务，也无需提供全部权重或所有 dump。

1. 在采集前确定本轮实际输入长度、padding／有效长度、token 和初态；保留原工作命令与 config。已有同输入 capture 就直接复用。
2. 首次 CPro dump 产生后，提供一小段目录清单、对应算子的节点／端口、shape/dtype/layout 元数据，以及一个普通 ND 张量的文件格式说明或样例。也保留 model_run_tool/data_proc_tool 的实际命令和版本；有状态或融合节点时再补一个对应样例。
3. 指定沿用的数值验收标准。暂时没有标准时先看指标，默认不补造容差。

拿到第二项后再实现该工具版本的解析与批量映射，避免现在猜文件结构、随后要求用户手工填写数百个端口。当前 19 项主机测试覆盖误差传播分类、大整数精度、非有限值、正负零、缺失数据、布局确认、哈希绑定和源目录保护；尚无真实 DDK dump 的解析或数值验收结果。
