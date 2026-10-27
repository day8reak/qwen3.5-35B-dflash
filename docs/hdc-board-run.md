# HDC 自动传输、运行和结果回传

在能通过 HDC 访问 CPro 的**服务器**执行 `targets/cpro/scripts/run_hdc.py`。脚本统一从环境读取设备 ID `SN` 和远程连接地址 `BIP`，每次调用均为：

```bash
hdc -t "$SN" -s "$BIP" ...
```

`-s` 是 HDC 服务端地址，填写你现有连接实际使用的 `IP:端口`，不能只填 IP；`SN` 选择该服务端连接的设备，见[官方远程连接说明](https://github.com/openharmony/docs/blob/master/en/application-dev/dfx/hdc.md#remote-connection)。本脚本使用已有连接，不配置网络转发或启动远端 HDC 服务。

<a id="当前优先重新导出后打通双图执行"></a>

<a id="当前优先诊断已有分区回退产物"></a>

<a id="当前优先本轮-matmul-归约候选上板"></a>
<a id="明确修复后重新导出并执行双图"></a>

<a id="当前双图采用可选-matmul-归约"></a>

<a id="当前读取-gdn-整图加载失败证据"></a>

## 当前：板端生成已完成，定位与主机参考的 token 差异

用户随后完成严格benchmark，native报告通过，但主机比较的 `tokens`、`dflash_tokens`、`ten_stable_runs` 为False；来源、权重、精度、量化、ABI、OMC身份、prompt、生成长度及stop检查均通过。精度门槛仍失败，不能据此发布提速结论。记录见 `specs/draft-rope-comparison-user-report.json`。

`ten_stable_runs` 的历史实现要求两种模式各10次结果都等于主机参考，不单独代表板端重复性。用户已回传：ordinary和DFlash均为 `[55,54,49,9]`，各10次全部相同；参考两模式均为 `[45,42,60,42]`。第一差异是共享prompt `[1,2,3]` 后的首token：45→55，已发生在target首轮，不能归因于后续draft验收。

prepared/reference已确认是seed17、S8、static dispatch、未量化FP16 tiny随机权重，`quantization=null`，并非完整35B或INT4。相同payload哈希的主机PyTorch、直接图及MatMul/compact改写投影均得首token45，原始target特征完全一致；尚不能证明板端GDN正确或确定设备偏差来源。主机证据见 `specs/target-prefill-host-verification.json`。

比较器现新增诊断字段，分别记录两模式一致性、各模式重复性、汇总与测量一致性、首个参考差异位置/值和长度差异；原有检查及PASS/FAIL条件全部保持。更新源码后可对已有文件离线重比，输出使用新文件：

```bash
export GE_COMPARE_DETAILS="$GE_DRAFT_ROPE_CHECK_RUN/comparison-details-$(date +%Y%m%d-%H%M%S).json"
PYTHONPATH="$REPO_DIR/model${PYTHONPATH:+:$PYTHONPATH}" python3 -m qwen35_moe_dflash.cli compare \
  --omc "$GE_DRAFT_ROPE_RUN/omc" \
  --report "$GE_DRAFT_ROPE_CHECK_RUN/benchmark.json" \
  --reference "$GE_BOARD_RUN/reference.json" \
  --output "$GE_COMPARE_DETAILS"
```

存在参考偏差时命令仍返回FAIL，并保留完整诊断JSON；终端会显示每种模式的测量数、重复性及首差异。它只读取已有OMC/报告并核验哈希，不调用OMG/HDC，不改旧报告或参考。首差异的 `generated_index` 从0计数，长度不同还会明确哪一侧已结束。已有smoke的有效三行target Top1为 `[41,41,55]`，同权重主机为 `[41,41,45]`；只比较有效prompt位置，padding输出不计入。

## 当前：修复拼接并一次验证布局、精度和完整第一层

26案例结果已回传：20个Load/Process成功，其中7个案例通过原数值门槛、13个失败；另6个Load失败。关键证据见 `specs/gdn-comprehensive-user-report.json`：

- 实际值和标记值的IO均逐项相等，已记录分配大小都等于逻辑大小；继续使用当前输入输出打包。
- 卷积转置正确，三维左补零后只剩8/384项非零；给BMM独立输入正确窗口时卷积通过。value的四维补零逐项正确，beta/decay的三维补零却把一个head移到序列尾部，导致后续乘法丢失整个head。
- Q/K平方逐项相等，MatMul求和输出丢失Q的两行、K的一行小值；独立二维对照同样失败，正常幅度对照通过。后续gain达到992是小sum丢失的结果，不能只改Rsqrt。
- 最终BMM实际输入仍有36/128个非零结果丢失，relative L2约0.038；原绝对容差使其PASSED，因此不能把它视为完全正确。裁剪/转置未再增加这组误差。

本次将所有FP16三维Concat的输入 `[a,b,c]` 精确视为 `[a,b,c,1]`，沿原轴执行四维Concat，再恢复原形状；普通图和前缀和内部拼接共同采用。权重、数值、图输出合同与舍入边界不变。设备是否保留该carrier及是否修复错误，需本批结果确认。

六个Load失败用例共同包含直接Data或可折叠Const输出，这只是结构关联，尚未证明是Load根因。新版删去常量kernel的非必要输出；预填充输入的观察支路改为精确Neg，标签带 `_neg`、参考相应取负，实际BMM/Mul/scan消费者不变。

**只运行下面一个41案例批次**：包含修正后的原26案例、11个精度/边界控制、4个完整第一层回归。11个控制覆盖原生ReduceSum的Q/K实际值及正常值、独立正确sum输入、正负次正规数/最小正常数、MatMul乘积和抵消。4个整层案例分别观察最终输出/所有core阶段，并对比原MatMul策略和仅Q/K sum、core输出mean三个位置恢复原生算子的策略。其他归约与scan维持原策略；该局部原生策略仅用于诊断候选，生产导出不自动采用。没有增加FP32路径或重新定义参考。

```bash
git -C "$REPO_DIR" pull --ff-only &&
export GE_GDN_REPAIR_RUN="$AI_RUN_DIR/ge-gdn-repair-$(date +%Y%m%d-%H%M%S)" &&
export GE_GDN_REPAIR_BOARD_RUN="${GE_GDN_REPAIR_RUN}-board" &&
python3 "$REPO_DIR/targets/cpro/scripts/probe_hdc_ops.py" \
  --suite gdn-repair --prepared "$GE_PAIR_RUN/prepared" \
  --reference "$GE_BOARD_RUN/reference.json" --omc "$GE_DRAFT_ROPE_RUN/omc" \
  --compile-only --output "$GE_GDN_REPAIR_RUN" &&
python3 "$REPO_DIR/targets/cpro/scripts/probe_hdc_runtime.py" \
  --probe-run "$GE_GDN_REPAIR_RUN" --native-from "$OP_PROBE_RUN" \
  --refresh-runner --replay-all --output "$GE_GDN_REPAIR_BOARD_RUN"
```

保留原prepared、reference和OMC作为来源及回退证据，暂不重建完整双图。个别native候选编译或Load失败时，其他已编译案例仍能运行；失败本身也用于判定替代路径能否使用。新构造的可精确表示边界控制采用零容差，原模型/实际算术容差保持。不能从native候选通过推断最终BMM的小值问题也已解决。

回传这两份摘要即可，无需按案例逐个找日志：

```bash
cat "$GE_GDN_REPAIR_BOARD_RUN/runtime-comparison.txt"
cat "$GE_GDN_REPAIR_BOARD_RUN/arms/candidate_without_ddk_hcl/gdn-summary.txt"
```

<a id="当前一次覆盖-gdn-故障链"></a>

## 已完成：26案例覆盖 GDN 故障链

用户已完成两个core诊断，均Load、Process、回收成功，但数值失败。原第一层的projected_qkv通过原门槛，convolved_qkv已出现大偏差，384项只有173项非零；chunk_0_intra在裁剪前就全零，因而末端裁剪不能单独解释本次归零。单独core使用主机Q/K/V等正确输入也失败：normalized_query/key幅度异常，weighted_value仅64/128项非零，decay_prefix为72/128项非零。记录见`specs/gdn-core-stages-user-report.json`。目前未确认某个算子或公共布局是唯一根因。

根据用户要求，上一轮一次生成 **26个小案例**，覆盖原值、正常幅度、独立布局以及串联上下文。以下是同一批次的覆盖范围，不需要逐组手动运行：

| 范围 | 案例数 | 对照内容 |
| --- | ---: | --- |
| 输入输出搬运 | 2 | 实际值/非周期标记，五种core输入及窄卷积输出形状，精确Neg |
| 卷积 | 6 | 转置/左填充/窗口构造；图内窗口与正确运行时窗口；常量/动态权重；后续view/SiLU/转置 |
| Q/K归一化 | 4 | 实际小值/所有平方正常的变范数行；square/sum/epsilon/Rsqrt/Mul及独立二维分支 |
| core布局 | 2 | value/beta/decay的转置、双head补零、view，实际值/标记值 |
| 逐行乘法 | 4 | 布局串联/正确预填充输入，实际值/标记值，并列独立二维控制 |
| 前缀和 | 4 | 布局串联/正确预填充输入，实际小值/较大负值，观察完整64个前缀 |
| core后段 | 4 | 实际/正常正值的scores×value；跳过Q/K归一化的精确切点；末端BMM到裁剪/转置的串联 |

主机实际序列化recipe与源图的六组执行对照全部位级一致，目前没有可确认的源码边连接错误。实际Q/K平方包含大量次正规数，冲零实验能够产生大于1的归一化结果，因此不能仅凭该现象判定布局问题。平方冲零没有复现摘要。进一步仅模拟sum→epsilon边界的次正规数丢失，并对epsilon-only行使用已观察到的gain992，得到Q/K最大值3.984375/3.490234375，与设备一致，相对L2误差只差约4.1e-7/7.7e-7。这是较强的局部假设，尚无设备原始张量逐元素确认，也不能区分MatMul输出或后续Power输入处丢失；正常域及各步骤输出用于进一步区分。

新版runner记录每个张量的逻辑形状/字节数、运行时形状/format及实际分配字节数，成功失败都保留已有记录。分配更大只说明存储更大，不能直接推断行stride；当前输入拷贝及输出读取方式不变，不读取未初始化的padding。新增报告按逻辑轴统计非零分布，帮助定位具体head/位置丢失。精确复制案例使用零容差；算术案例保持原FP16门槛，信号全部消失仍会明确显示。

下面26案例命令保留供历史复现，当前使用上面的41案例修复验证批次：

```bash
git -C "$REPO_DIR" pull --ff-only &&
export GE_GDN_ALL_RUN="$AI_RUN_DIR/ge-gdn-comprehensive-$(date +%Y%m%d-%H%M%S)" &&
export GE_GDN_ALL_BOARD_RUN="${GE_GDN_ALL_RUN}-board" &&
python3 "$REPO_DIR/targets/cpro/scripts/probe_hdc_ops.py" \
  --suite gdn-comprehensive --prepared "$GE_PAIR_RUN/prepared" \
  --reference "$GE_BOARD_RUN/reference.json" --omc "$GE_DRAFT_ROPE_RUN/omc" \
  --compile-only --output "$GE_GDN_ALL_RUN" &&
python3 "$REPO_DIR/targets/cpro/scripts/probe_hdc_runtime.py" \
  --probe-run "$GE_GDN_ALL_RUN" --native-from "$OP_PROBE_RUN" \
  --refresh-runner --replay-all --output "$GE_GDN_ALL_BOARD_RUN"
```

以下两份摘要已经回传。`gdn-summary.txt`包含各阶段误差、非零数量、变化轴分布及IO大小，省去重复编译器环境警告；完整统计和日志仍保留在`report.json`、`report.txt`及`results.tar.gz`。若个别案例编译失败，其他已编译案例仍会回放；传输失败仍保留原错误，不计为完成。

```bash
cat "$GE_GDN_ALL_BOARD_RUN/runtime-comparison.txt"
cat "$GE_GDN_ALL_BOARD_RUN/arms/candidate_without_ddk_hcl/gdn-summary.txt"
```

原三案例、两个core案例和完整模型不需要重复。新增探针的切点值来自相同权重的主机计算；增加输出或切开图仍可能改变编译，不能把其结果无条件推广为原OMC所有内部行为。生产计算、原始OMC、库部署与完整精度门槛保持不变。

<a id="当前定位-core-首次归零的计算或裁剪边界"></a>

## 已完成：定位 core 首次归零的计算或裁剪边界

用户 `ge-gdn-accuracy-20261027-082151` 的三个探针均编译、Load、Process及回收完成。第一层单输出和多输出探针数值失败，独立Rsqrt通过。多输出图的 `input_norm` 相对L2误差约0.00174，仍有128个非零值；`chunk_core` 却是128个零，后面的门控和输出投影也全零。`norm_input`只剩FP16舍入后的epsilon，`norm_gain=992`与这种输入相符，不能把该增益相对主机参考的差异单独归为Rsqrt故障。独立Rsqrt对实际主机norm输入的相对L2误差约0.00185。记录见 `specs/gdn-core-zero-user-report.json`。

当时最早**已观察**的错误边界是core输出；后续两个core案例已将大偏差提前定位到卷积，并在独立core中观察到归一化异常。另一个需要区分的情况是裁剪：GDN保持64行chunk，实际S8的结果要从中裁出8行，填充区本来全零。主机core有92个正常FP16数，仅冲零次正规数不能解释128个零。首token也不依赖历史状态，源图明确非零。这些检查缩小范围，但不确定具体设备算子或编译pass。

用户随后完成已有文件对照：单输出及多输出探针的前3个有效位置共48个特征，均与原OMC逐元素相等，max_abs和relative L2都为0。当前两图复现了原故障在该边界的结果；这不等于其内部优化全部相同。该对照无需再跑。

以下两个案例已经运行，命令仅保留作历史复现；它们复用原prepared实际权重、参考、OMC记录及相同DDK/OMG：

- `gdn_layer0_core_stages`：观察原第一层投影、卷积/激活、core内部、裁剪前后及块结束输出。
- `gdn_core_replay`：把同权重主机求得的query/key/value/decay/beta作为运行时输入，单独观察相同core，区分core之前的计算和core自身。

这些观察点只引用已有节点，不改变默认模型计算，也不增加新的中间算术。把它们作为新模型输出仍可能改变编译优化，因此保留最终core及第一层输出作对照。原三个案例和独立Rsqrt无需重跑。新报告明确标出 `all_zero_with_nonzero_reference`，原精度门槛保持；新套件终端摘要简化，完整统计仍在 `report.json`。

```bash
git -C "$REPO_DIR" pull --ff-only &&
export GE_GDN_CORE_RUN="$AI_RUN_DIR/ge-gdn-core-$(date +%Y%m%d-%H%M%S)" &&
export GE_GDN_CORE_BOARD_RUN="${GE_GDN_CORE_RUN}-board" &&
python3 "$REPO_DIR/targets/cpro/scripts/probe_hdc_ops.py" \
  --suite gdn-core-accuracy --prepared "$GE_PAIR_RUN/prepared" \
  --reference "$GE_BOARD_RUN/reference.json" --omc "$GE_DRAFT_ROPE_RUN/omc" \
  --compile-only --output "$GE_GDN_CORE_RUN" &&
python3 "$REPO_DIR/targets/cpro/scripts/probe_hdc_runtime.py" \
  --probe-run "$GE_GDN_CORE_RUN" --native-from "$OP_PROBE_RUN" \
  --refresh-runner --replay-all --output "$GE_GDN_CORE_BOARD_RUN"
```

下面两份报告已回传。原图上下文对照也已完成；下节的 `vs_original_omc_valid_layer0` 命令仅保留供历史复现：

```bash
cat "$GE_GDN_CORE_BOARD_RUN/runtime-comparison.txt"
cat "$GE_GDN_CORE_BOARD_RUN/arms/candidate_without_ddk_hcl/report.txt"
```

<a id="当前检查第一层-gdn-分支为何贡献接近零"></a>

## 已完成：第一层 GDN 分支分阶段检查

用户已完成原OMC特征采集：有效prompt的第0层relative L2约0.129，第2层约0.229；首token仍为55，源图和参考均为45。这里使用未量化FP16 tiny权重。对用户回传的512字节原始特征做主机干预实验，把三层GDN的输出投影结果置零，可以复现板端完整序列 `[55,54,49,9]`；第0、2层有效特征最大差仅约 `3.05e-5`、`6.10e-5`。这强烈指向GDN分支的有效贡献消失，但没有证明是哪一个算子或编译pass出错；跳过部分归一化也会造成类似现象。置零只用于定位，没有应用到生产图。证据见 `specs/gdn-accuracy-localization.json`。

以下三个诊断案例已经执行，命令保留用于历史复现。它们使用原prepared实际权重、原prompt和原OMC的lowering选项；核验来源、权重、ABI、DDK/OMG身份，不重新生成随机权重或完整模型：

- `gdn_layer0_final`：原第一层完整计算，仅导出块结束特征。
- `gdn_layer0_stages`：同一层同时观察输入归一化、GDN core、归一化输入/增益、门控结果、输出投影及块结束特征。
- `gdn_layer0_rsqrt`：实际主机norm输入与同形状正常正数对照，检查Rsqrt的数值范围。

```bash
git -C "$REPO_DIR" pull --ff-only &&
export GE_GDN_ACCURACY_RUN="$AI_RUN_DIR/ge-gdn-accuracy-$(date +%Y%m%d-%H%M%S)" &&
export GE_GDN_ACCURACY_BOARD_RUN="${GE_GDN_ACCURACY_RUN}-board" &&
python3 "$REPO_DIR/targets/cpro/scripts/probe_hdc_ops.py" \
  --suite gdn-accuracy --prepared "$GE_PAIR_RUN/prepared" \
  --reference "$GE_BOARD_RUN/reference.json" --omc "$GE_DRAFT_ROPE_RUN/omc" \
  --compile-only --output "$GE_GDN_ACCURACY_RUN" &&
python3 "$REPO_DIR/targets/cpro/scripts/probe_hdc_runtime.py" \
  --probe-run "$GE_GDN_ACCURACY_RUN" --native-from "$OP_PROBE_RUN" \
  --refresh-runner --replay-all --output "$GE_GDN_ACCURACY_BOARD_RUN"
```

`OP_PROBE_RUN`指向此前含native构建的探针批次，复用了已验证的candidate库部署。第一层单输出/多输出结果与原OMC捕获的第0层有效特征比较已经完成，均为48/48相等；仍不能把插桩图内部结果直接认定为原OMC内部结果。Rsqrt输入来自同权重主机求值，不是板端内部采集。源码不调整原数值门槛；原绝对容差0.002可能掩盖很小的GDN core全零，因此报告对每个输出打印幅度、非零数、RMS和relative L2，PASS也要检查这些指标。全部8行用于探针原比较，归因时原capture只比较前3个有效prompt位置。

以下两个报告已回传，读取命令仅供历史复现：

```bash
cat "$GE_GDN_ACCURACY_BOARD_RUN/runtime-comparison.txt"
cat "$GE_GDN_ACCURACY_BOARD_RUN/arms/candidate_without_ddk_hcl/report.txt"
```

已有输出也已确认两个诊断图复现原图第0层有效特征。下面保留只读比较命令供历史复现，它检查回收哈希并打印有效位置误差：

```bash
python3 - "$GE_FEATURE_RUN/board/benchmark.json" "$GE_GDN_ACCURACY_RUN/manifest.json" "$GE_GDN_ACCURACY_BOARD_RUN" <<'PY'
import hashlib, json, sys
from pathlib import Path
import numpy as np
capture, probe = [json.loads(Path(p).read_text()) for p in sys.argv[1:3]]
assert capture['prompt_token_ids'] == probe['provenance']['prompt_token_ids']
value = capture['execution']['target_features']
assert (value['dtype'], value['encoding'], value['shape'], value['bytes']) == ('float16', 'hex-le', [1, 8, 32], 512)
raw = bytes.fromhex(value['data'])
assert len(raw) == 512
length = len(capture['prompt_token_ids'])
original = np.frombuffer(raw, dtype='<f2').reshape(1, 8, 32)[:, :length, :16].astype(np.float64)
arm = Path(sys.argv[3]) / 'arms/candidate_without_ddk_hcl'
report = json.loads((arm / 'report.json').read_text())
for name in ('gdn_layer0_final', 'gdn_layer0_stages'):
    case = next(c for c in report['cases'] if c['name'] == name)
    item = case['numerical']['outputs'][0]
    assert case['run'] == 'PASSED' and item['label'] == 'layer0_final'
    data = (arm / 'board' / name / item['path']).read_bytes()
    assert hashlib.sha256(data).hexdigest() == item['actual_sha256']
    actual = np.frombuffer(data, dtype='<f2').reshape(1, 8, 16)[:, :length].astype(np.float64)
    assert np.isfinite(actual).all() and np.isfinite(original).all()
    difference = actual - original
    norm = float(np.linalg.norm(original))
    print(name, json.dumps({'vs_original_omc_valid_layer0': {
        'max_abs': float(np.abs(difference).max()),
        'relative_l2': float(np.linalg.norm(difference)) / norm if norm else None}}, allow_nan=False))
PY
```

## 采集原始 OMC 的 target 特征

本次已完成，下面保留采集方法供复现，无需重跑。只重建runner，复用已有OMC、prepared及参考。选项在target Process之后、draft Process之前保存原始FP16字节，随已有 `benchmark.json` 的远端SHA256流程回传；不会改变OMC输出接口。仅允许execution-only，最多65536字节。tiny S8输出为 `[1,8,32]`、512字节；前后16列分别是第0、2层结束后的特征。每层还包含归一化、MoE和残差，所以单个边界的偏差不能独立归因GDN。

```bash
cd "$REPO_DIR"
git pull --ff-only
export GE_FEATURE_RUN="$AI_RUN_DIR/ge-target-features-$(date +%Y%m%d-%H%M%S)"
python3 "$REPO_DIR/targets/cpro/scripts/build_hiai.py" \
  --output "$GE_FEATURE_RUN/hiai-build" &&
python3 "$REPO_DIR/targets/cpro/scripts/run_hdc.py" \
  --omc "$GE_DRAFT_ROPE_RUN/omc" \
  --runner "$GE_FEATURE_RUN/hiai-build/build/qwen35_dflash_hiai" \
  --reference "$GE_BOARD_RUN/reference.json" \
  --output "$GE_FEATURE_RUN/board" --execution-only --capture-target-features &&
python3 "$REPO_DIR/targets/cpro/scripts/direct_model_air.py" compare-target-features \
  --prepared "$GE_PAIR_RUN/prepared" \
  --omc "$GE_DRAFT_ROPE_RUN/omc" \
  --board-run "$GE_FEATURE_RUN/board" \
  --reference "$GE_BOARD_RUN/reference.json" \
  --output "$GE_FEATURE_RUN/comparison"
```

离线工具只依赖NumPy，读取并核验原prepared权重、两份OMC、参考及已完成capture报告的哈希。支持已有tiny配置、S=2–16；按有效prompt token及layer0/layer2分别输出finite数、max_abs、RMS、relative L2和数值相等性，同时列出主机末个有效位置的Top5。它使用当前源码中立图，不模拟OMG内部优化，不改变精度门槛或旧参考；输出 `DIAGNOSTIC_COMPLETE` 不代表精度通过。绑定覆盖相同权重identity/config与运行文件，未读取AIR，因此不声称完整prepared→AIR文档哈希链。当前源码哈希也保存在报告中。

用户已回传 `comparison/report.txt` 及原始特征hex，确认本次仍复现target的55且两个边界均有偏差。记录为用户提供的终端证据，远端完整文件尚未独立获取；源码及测试证据见 `specs/target-feature-capture-verification.json`。首轮三个分阶段探针也已完成，当前使用页首两个core内部探针继续定位。

## 已完成：tiny 双图执行与严格 benchmark


用户 `ge-draft-rope-20261026-194345` 已导出、编译并构建成功，板端返回 `EXECUTION_COMPLETE`。当前脚本只有在 target/draft 均 Load、初始化张量、各调用一次 Process，target 特征传给 draft，必要输出回收及哈希校验、smoke 报告身份与输出范围检查都通过后才设置此状态。记录为用户回传通过，原始 `benchmark.json`、`result.json` 和哈希尚未在本工作区独立读取。证据见 `specs/draft-rope-execution-user-report.json`。

这次日志已匹配记录的 PID11937；仍保留PID复用限制。HCL dlopen警告和Binder清理信息与本次执行成功同时出现，无需仅凭这些信息补库或重跑 smoke。`EXECUTION_COMPLETE` 不包括精度、NPU运行位置或性能判定。

以下benchmark命令已经执行并得到上述参考差异，保留供历史复现；当前检查第一层中间结果，不重复benchmark：

```bash
export GE_DRAFT_ROPE_CHECK_RUN="$AI_RUN_DIR/ge-draft-rope-check-$(date +%Y%m%d-%H%M%S)"
python3 "$REPO_DIR/targets/cpro/scripts/run_hdc.py" \
  --omc "$GE_DRAFT_ROPE_RUN/omc" \
  --runner "$GE_DRAFT_ROPE_RUN/hiai-build/build/qwen35_dflash_hiai" \
  --reference "$GE_BOARD_RUN/reference.json" \
  --output "$GE_DRAFT_ROPE_CHECK_RUN"
```

去掉 `--execution-only` 后，脚本要求参考状态为PASS，来源/精度/量化/序列长度匹配，并满足 `len(prompt)+max_new_tokens-1≤S`；不满足时在上传前拒绝，不修改旧参考来绕过。默认每种模式3次预热、10次测量，比较 ordinary/DFlash 生成token与停止原因。结果及比较见新目录的 `benchmark.json`、`comparison.json`、`result.json`；若失败保留该次输出继续定位。tiny计时只用于诊断，不能代表完整35B质量或最终优化后的DFlash至少2倍收益。

## 已完成的修复：draft 动态 RoPE 频率计算

用户 `ge-rope-chain-20261026-192630` 已完成双图导出、编译及 runner 构建。target Load 和输入输出维度查询成功；draft Load 失败，因此两图都尚未调用 Process。完整静态 target RoPE 区域折叠已通过这次编译和 target 加载，但不能据此认定精度、实际 NPU 执行或完整模型通过。

候选 HiLog 的直接失败链为：`SubGraph_2` 中 `v29/Mul` 的两个输入类型不一致，随后 GenerateOp、Prepare 和 Load 失败。此前 draft 的第一个 CPU 子图已初始化成功。不能把前面的 HCL 库加载警告当作本次直接失败原因，也不能由这段日志确定是哪一个输入被内部编译器改了类型。原始 `result.json` 的 PID 绑定未回传，日志仍保留候选身份。用户回传记录见 `specs/rope-chain-model-load-user-report.json`。

源码的 `v29` 是动态 RoPE 频率外积：tiny 的 FP16 `[1,12,1] × [1,1,4]`。本轮改用 K=1 BatchMatMul，结果仍为 `[1,12,4]`。保留 INT32 位置先转 FP16 的舍入边界，以及后续重复拼接和 Cos/Sin；位置继续依赖运行时 mask，不预计算 draft 表。非空有效前缀使位置落在 `[0,S+B-2]`；只在上界不超过65504且存储的 FP16 inverse 有限、非负（不含负零）时使用 BMM，其他合同保留 Mul。此条件避免把有符号零、非有限值等未经验证的通用 Mul 改成 MatMul。双图 ABI、权重和 target 静态 RoPE 折叠保持。

已有 `rope_frequency_bmm` 小图完成 Load、Process 和数值比较，但它是 `[1,2,1] @ [1,1,8]` 且两端都是运行时输入。它支持选用 BMM 的方向；实际 draft 形状与常量右端现已由完整 tiny 图单次执行验证，但数值比较仍待完成。主机验证见 `specs/ge-draft-rope-frequency-verification.json`。

同时修复终端 HiLog 摘要：保留最近一次 Init/GenerateOp/Prepare/Load 等失败及前后文，避免长清理日志挤掉中间的算子错误。完整候选文件继续保存；匹配已记录 PID 与启发式选取的提示分开，PID 复用限制仍保留。

以下是本次已成功的重导出命令，保留供历史复现；当前使用上面的现有 OMC 比较入口，不重复执行：

```bash
git -C "$REPO_DIR" pull --ff-only &&
export GE_MODEL_TOOL="$REPO_DIR/targets/cpro/scripts/direct_model_air.py" &&
export GE_DRAFT_ROPE_RUN="$AI_RUN_DIR/ge-draft-rope-$(date +%Y%m%d-%H%M%S)" &&
python3 "$GE_MODEL_TOOL" export \
  --prepared "$GE_PAIR_RUN/prepared" --ge-root "$CANN_ROOT" \
  --fp16-reductions matmul --fp16-row-mul compact --gdn-indexed-copy gather \
  --output "$GE_DRAFT_ROPE_RUN/air" &&
python3 "$GE_MODEL_TOOL" compile \
  --air "$GE_DRAFT_ROPE_RUN/air" \
  --ddk-root "$CPRO_DDK_ROOT" --ddk-id "$CPRO_DDK_ID" \
  --output "$GE_DRAFT_ROPE_RUN/omc" --reject-cpu-fallback &&
python3 "$REPO_DIR/targets/cpro/scripts/build_hiai.py" \
  --output "$GE_DRAFT_ROPE_RUN/hiai-build" &&
python3 "$REPO_DIR/targets/cpro/scripts/run_hdc.py" \
  --omc "$GE_DRAFT_ROPE_RUN/omc" \
  --runner "$GE_DRAFT_ROPE_RUN/hiai-build/build/qwen35_dflash_hiai" \
  --reference "$GE_BOARD_RUN/reference.json" \
  --output "$GE_DRAFT_ROPE_RUN/board" --execution-only
```

上述重导出候选已由用户完成双图 Load/Process；无需再重复 prepare、诊断转换或算子探针。完整精度检查和最终 DFlash 至少2倍目标仍未验证。

历史 `ge-gdn-model-20261026-171048` 在 target 静态 RoPE BMM 后的 `insert_cast_4` 初始化失败，主图 NPU5/CPU6 与 NPU3/CPU4 仅属于该旧产物。`ge-rope-model-20261026-190121` 的单节点折叠又暴露编译器静态链问题，随后改为七节点原子折叠，通过上述新一次 target 加载。历史证据见 `specs/gdn-model-load-user-report.json`、`specs/rope-model-compile-user-report.json` 和 `specs/ge-constant-rope-region-verification.json`。`--reject-cpu-fallback` 检查显式分区超限回退，不保证 CPU 分区为0，也不证明 NPU 执行。

<a id="当前将已验证的-gdn-gather-复制接入整图"></a>

## 已完成：将已验证的 GDN Gather 复制接入整图

以下是较早的 GDN 候选链路，当时完成到 target Load 失败；当前定位页首的已有比较失败报告。

最新`ge-row-chain-20261026-144144`已通过ReshapeFusion，19个row Mul CPU-only候选消失；已识别CPU-only计算从117减到98，主图仍因NPUCL分区数超过9回退，尚未执行双图。保留本次产物与诊断，不重新运行旧整图。

用户重放[remaining-cpu批次](cpu-gap-probes.md)后，42个已编译案例全部通过Load、Process和原数值比较。唯一未上板的是`moe_cast_bool_roundtrip`，它在编译时失败。最后`hilog.stderr`回收失败导致旧脚本把整批标成`INFRASTRUCTURE_FAILED`，但42项结果已完整回收；无需为此重跑。新脚本将这种附加日志失败标为`INCOMPLETE`和`REPLAY_COMPLETE_WITH_DIAGNOSTIC_WARNINGS`，保留远端目录，不伪造日志或修改旧报告；必需结果/哈希/退出码失败仍中止。

这批路由Gather、TopK/ArgMax布局与attention mask候选均未消除编译CPU分区，暂不替换。当时 RoPE BMM/二维MatMul及原形位置Cast因合同差异未直接接入整图；上文现已给出 draft 频率的有界适配及待验证范围。GDN接入此前4个已验证的Gather候选，新增显式`--gdn-indexed-copy gather`，默认仍为`native`。它仅覆盖FP16 `[2,1,64,64]`三角掩码及对应outer Sub；其他形状保留原实现并报告原因。Exp前后两个普通掩码和Neg前的严格掩码均保留，Sub仍只做一次原FP16减法。权重、专家策略、GDN递推顺序与双图ABI不变。

复用已有prepared和reference，在新目录一次运行以下链路；无需再跑算子批次：

```bash
git -C "$REPO_DIR" pull --ff-only
export GE_MODEL_TOOL="$REPO_DIR/targets/cpro/scripts/direct_model_air.py"
export GE_GDN_MODEL_RUN="$AI_RUN_DIR/ge-gdn-model-$(date +%Y%m%d-%H%M%S)"

python3 "$GE_MODEL_TOOL" export \
  --prepared "$GE_PAIR_RUN/prepared" --ge-root "$CANN_ROOT" \
  --fp16-reductions matmul --fp16-row-mul compact --gdn-indexed-copy gather \
  --output "$GE_GDN_MODEL_RUN/air" &&
python3 "$GE_MODEL_TOOL" compile \
  --air "$GE_GDN_MODEL_RUN/air" \
  --ddk-root "$CPRO_DDK_ROOT" --ddk-id "$CPRO_DDK_ID" \
  --output "$GE_GDN_MODEL_RUN/omc" --reject-cpu-fallback &&
python3 "$REPO_DIR/targets/cpro/scripts/build_hiai.py" \
  --output "$GE_GDN_MODEL_RUN/hiai-build" &&
python3 "$REPO_DIR/targets/cpro/scripts/run_hdc.py" \
  --omc "$GE_GDN_MODEL_RUN/omc" \
  --runner "$GE_GDN_MODEL_RUN/hiai-build/build/qwen35_dflash_hiai" \
  --reference "$GE_BOARD_RUN/reference.json" \
  --output "$GE_GDN_MODEL_RUN/board" --execution-only
```

每图新增并哈希绑定`gdn-copy-lowering.json`，当前tiny预期target替换9个mask与3个outer Sub，draft为0；这些源图替换数不是最终分区数。已有两种lowering报告继续保留。编译失败或仍回退时，链路自动停在构建/上板之前，对同一产物做只读检查：

```bash
python3 "$GE_MODEL_TOOL" inspect-compile \
  --air "$GE_GDN_MODEL_RUN/air" --omc "$GE_GDN_MODEL_RUN/omc" \
  --graph target --output "$GE_GDN_MODEL_RUN/inspect-target"
```

若失败发生在draft，改`--graph draft`及新诊断目录。不要重复编译同一失败产物。此候选现已编译通过并上板到target Load失败，当前排查入口见页首。实际NPU位置、完整精度及DFlash至少2倍尚未验证。

## 历史：MatMul归约与row Mul compact双图复测

以下整图重跑和诊断已完成，当前无需重复。

历史`ge-row-mul-20261026-121239`已确认归约421/10、row Mul19/0，但target在OMG的ReshapeFusion/ByPassNode阶段失败，尚未到分区检查；没有构建runner或上板。本轮在导出端合并涉及row Mul的连续静态reshape，保留共享分支和图输出，避免把这部分链交给厂商pass处理；后续用户重跑已通过该pass，但仍触发分区超限。新结果使用`GE_ROW_CHAIN_RUN`，旧`GE_ROW_MUL_RUN`保留。已修复的`inspect-compile`可直接读取这份早期失败留档，命令见[当前导出指南](direct-model-air.md#当前下一步可选-row-mul-整图适配)。

用户回传 `layout-probe-20261026-113524` / `layout-board-20261026-113529`：9项均通过 Load、Process 和原数值比较。两种 row Mul 原形为 NPU=0、CPU=1，compact 均为 NPU=1、CPU=0，因此只将 row Mul compact 推进整图候选。outer Sub compact 仍为 NPU=2、CPU=1；Concat64 原形、tree8、tree2 都是 NPU=1、CPU=0，不能将整图中的Concat拒绝归因于64路扇入，也没有依据用树替换原形。Sub和Concat保持原实现。

以上是用户粘贴报告，未独立获取并核对原始产物哈希；此前建议 revision 为 `37ba5d3`，实际运行 revision 未报告。编译分区与实际执行位置分开记录，9项的 runtime placement 仍为 `UNVERIFIED`。

复用 `$GE_PAIR_RUN/prepared` 和 `$GE_BOARD_RUN/reference.json`，显式叠加 `--fp16-reductions matmul --fp16-row-mul compact`，导出到新目录。两个选项默认均为 `native`；不重新 prepare、不改专家模式、不换库。构建沿用原 resources/环境中的 SDK、头文件和链接资源；若此前显式指定 resources 或兼容 SDK 版本，继续使用同一配置。

```bash
git -C "$REPO_DIR" pull --ff-only
export GE_MODEL_TOOL="$REPO_DIR/targets/cpro/scripts/direct_model_air.py"
export GE_ROW_CHAIN_RUN="$AI_RUN_DIR/ge-row-chain-$(date +%Y%m%d-%H%M%S)"
export HIAI_ROW_CHAIN_BUILD_DIR="$GE_ROW_CHAIN_RUN/hiai-build"
export HDC_ROW_CHAIN_RUN_DIR="$GE_ROW_CHAIN_RUN/board"

python3 "$GE_MODEL_TOOL" export \
  --prepared "$GE_PAIR_RUN/prepared" --ge-root "$CANN_ROOT" \
  --fp16-reductions matmul --fp16-row-mul compact \
  --output "$GE_ROW_CHAIN_RUN/air" &&
python3 "$GE_MODEL_TOOL" compile \
  --air "$GE_ROW_CHAIN_RUN/air" \
  --ddk-root "$CPRO_DDK_ROOT" --ddk-id "$CPRO_DDK_ID" \
  --output "$GE_ROW_CHAIN_RUN/omc" --reject-cpu-fallback &&
python3 "$REPO_DIR/targets/cpro/scripts/build_hiai.py" \
  --output "$HIAI_ROW_CHAIN_BUILD_DIR" &&
python3 "$REPO_DIR/targets/cpro/scripts/run_hdc.py" \
  --omc "$GE_ROW_CHAIN_RUN/omc" \
  --runner "$HIAI_ROW_CHAIN_BUILD_DIR/build/qwen35_dflash_hiai" \
  --reference "$GE_BOARD_RUN/reference.json" \
  --output "$HDC_ROW_CHAIN_RUN_DIR" --execution-only
```

四步用 `&&` 串联；导出或编译失败就停止正式 runner 构建和上板。新的 `qwen35_dflash_hiai` 包含现有 ND 补维兼容；旧独立 probe 或旧正式二进制不能代替该次构建。旧 prepared、OMC、runner、探针与板端结果均保留。

row Mul compact 只将已验证的两种 FP16 广播形状压成二维，保持行顺序、原操作数及一次 FP16 Mul，再恢复原输出形状。当前 tiny/static 源码预期 target 替换19项（rows16为10项、rows128为9项）、draft为0项；用户已确认上一轮实际导出计数为19/0，链处理后的新重跑通过ReshapeFusion，19项Mul退出CPU-only列表；主图仍分区超限，完整执行待验证。每图新增 `row-mul-lowering.json`，同时保留 `reduction-lowering.json`；反馈 `mode` 和 `counts` 即可：

```bash
python3 - <<'PYCOUNTS'
import json
import os
from pathlib import Path

for name in ("target", "draft"):
    folder = Path(os.environ["GE_ROW_CHAIN_RUN"]) / "air" / name
    for filename in ("reduction-lowering.json", "row-mul-lowering.json"):
        report = json.loads((folder / filename).read_text())
        print(name, filename, report["mode"], report["counts"])
        if "reshape_chains" in report:
            print("  reshape_chains", report["reshape_chains"]["counts"])
PYCOUNTS
# 仅上板已尝试并生成对应文件时读取；编译失败不会进入该阶段。
for file in result.json benchmark.stderr; do
  if [ -f "$HDC_ROW_CHAIN_RUN_DIR/$file" ]; then
    cat "$HDC_ROW_CHAIN_RUN_DIR/$file"
  fi
done
```

历史首次导出在OMG图优化阶段失败；链处理后的重跑通过ReshapeFusion，但主图仍超过9分区，draft编译和上板未运行。整数归约、路由、TopK及GDN掩码等候选仍待处理。编译通过后 `--execution-only` 强制 target/draft 各执行一次并回传输出；精度和性能为 `NOT_EVALUATED`。完整精度后续验证，最终目标仍为同板同目标权重下 DFlash 至少提速2倍。

若新编译失败或仍回退，保留本次 AIR/OMC、日志和 `failure.json`，对同一产物运行只读诊断，不直接重复编译：

```bash
export GE_ROW_CHAIN_INSPECT_RUN="$AI_RUN_DIR/ge-row-chain-inspect-$(date +%Y%m%d-%H%M%S)"
python3 "$GE_MODEL_TOOL" inspect-compile \
  --air "$GE_ROW_CHAIN_RUN/air" --omc "$GE_ROW_CHAIN_RUN/omc" \
  --graph target --output "$GE_ROW_CHAIN_INSPECT_RUN"
cat "$GE_ROW_CHAIN_INSPECT_RUN/report.txt"
```

上例对应 target 失败；draft 失败时改 `--graph draft` 并使用新诊断目录。候选支持与构造时边不等于最终分区或设备执行位置。

九项布局探针与此前归约探针的历史命令保存在[直接 GE 第7节](direct-model-air.md#当前下一步可选-row-mul-整图适配)，当前无需重跑。

`--reject-cpu-fallback` 是本项目 compile 包装器参数，不传给厂商 OMG。它遇到主图或归属未知的 NPUCL 分区超限回退时停止，保留已生成的 OMC、编译日志和诊断，写入 `failure.json`，不发布成功 manifest。常量折叠图的 CPU 执行允许保留；没有发现这类回退也不证明所有算子都在 NPU 上。

此模式沿用文件名 `benchmark.json`，但报告的 `mode` 是 `smoke`：强制 target 和 draft 各调用一次，将 target 特征传给 draft，保存两图 Top1 输出。即使 prompt 的下一 token 是 EOS，也会执行 draft。成功标为 `EXECUTION_COMPLETE`；`accuracy`、`performance` 为 `NOT_EVALUATED`，不产生 `comparison.json`。参考 JSON 仍用于身份和输入配置检查；其历史 PASS 不是本次板端精度结论。

程序保留模型哈希、版本、张量数量、格式、大小和 token 范围检查。ND 描述允许原有尾部补1到四维，例如 `[1,S] → [1,S,1,1]`、`[1,S,F] → [1,S,F,1]`；源码另支持逻辑 rank3 `[a,b,c] → [1,a,b,c]`，使用运行库返回的描述初始化张量。新增规则仅扩展 rank3 特征，rank2 token 合同及宽度推导不变；仍不按元素总数放行任意布局。后续整图编译成功后需在新目录重建正式runner以取得这项兼容；此前remaining-cpu批次只刷新独立probe。成功和失败都会尝试回传 HiLog，日志缺失单独记录。

当前 tiny 双图执行已取得用户通过反馈；完整权重模型仍未验证。去掉 `--execution-only` 后进入严格 benchmark + 参考比较，不能用 smoke 的成功替代比较结果。

## 服务器准备

首次准备参考与 runner 可查[直接 GE tiny 上板指南的第1、2节](direct-ge-board.md)。当前复用 `GE_DRAFT_ROPE_RUN` 中成功执行的 OMC 和正式 runner，以及匹配的 `reference.json`；无需重复首次准备。

需要的资源为：

| 参数/变量 | 含义 |
|---|---|
| `SN` | 设备 ID，必填 |
| `BIP` | 远程 HDC 服务端连接地址，必填 |
| `HDC` | 可选，主机 HDC 可执行文件路径；默认从 PATH 查找 `hdc` |
| `--omc` | 含成功 `manifest.json` 的 OMC bundle 目录 |
| `--runner` | 正式 HarmonyOS ARM64 程序；不接受主机 fake runner |
| `--reference` | 与这批 OMC 身份匹配的参考 JSON；默认要求 PASS，execution-only 不以历史精度状态作门槛 |
| `--board-cleanup` | 成功后的清理方式：`all`（默认）、`omc`、`none`，见下表 |
| `--output` | 本次服务器结果目录，必须在 `AI_RUN_DIR` 下且尚不存在 |
| `--execution-only` | 只跑 target/draft 各一次并回传，精度与性能后续验证 |

板端须能通过 shell 执行 `sh`、`mkdir`、`chmod`、`sha256sum` 等基础命令，且部署位置允许执行程序与访问 HIAI。脚本自动定位构建记录中的 HIAI 库和 OHOS SDK 的 ARM64 C++ 运行库。

## 后续严格 benchmark 入口

当前 smoke 和native benchmark均已通过，参考比较失败；先执行页首已有报告的离线诊断。保留现有 `SN`、`BIP`、`GE_DRAFT_ROPE_RUN` 和 `GE_BOARD_RUN`；不重新设置设备占位符或使用旧的回退 OMC。

若 `hdc` 不在 PATH，先设置 `export HDC=/your/host/toolchains/hdc`。无需设置 `HIAI_RUNTIME_LIB_DIR`，无需手动拷贝 C++ 库，也无需传库目录或构建目录参数。

脚本在 `--runner` 相邻的标准构建目录查找 `command.json` / `result.json`（例如 `hiai-build/build/qwen35_dflash_hiai` 对应 `hiai-build/`）。确认构建为 `LINKED`、程序 SHA256 匹配后，检查记录中的链接库 SHA256，并收集这些库所在目录第一层的 `.so` / `.so.*`。不会扫描整个 DDK，也不会从示例目录混入另一份 HIAI 库。

如果同套库目录没有 `libc++_shared.so`，脚本从构建命令记录的 OHOS Native SDK 中定位 ARM64 版本；找不到时会记录未补充，后续由板端 probe 判断装载是否成功。自动收集的库须为 ARM64 ELF；同名库内容冲突时在上传前停止。最终来源、哈希和查找结果记录在 `preflight.json` / `result.json` 中。保留原构建目录、DDK 和 SDK 路径即可；如果这些文件搬迁或变化，会显示具体缺失路径或哈希不匹配，不会悄悄替换为另一套库。

默认部署复用已在本板探针验证的 vendor HCL 候选：排除 `libhiai_hcl_model_runtime.so` 与 `libhiai_hcl_model_runtime_impl.so` 两个 DDK 库，并在进程的库搜索路径中加入 `/vendor/lib64/chipsetsdk`。上传前检查保留文件的 ELF 依赖及 SONAME；若存在对排除库的硬依赖，明确失败，不上传替代空壳库。此策略只影响本次隔离目录和进程，不修改系统文件；实际后端以日志为准。

新构建记录绑定 `command.json` 的 SHA256；旧记录缺少该字段时仍可使用，标记为 `legacy-unbound`，并检查现有程序及库的身份。该检查不保证全部传递依赖与板端 ABI 兼容，最终以 probe 和实际运行结果为准。旧版显式库目录参数仍兼容已有脚本，推荐直接使用上面的四个参数。

默认板端目录为 `/data/local/tmp/qwen35-dflash-<唯一标识>`，实际路径记录在结果中。需要自选位置时加 `--board-root /data/local/tmp/你的新目录`，该目录必须尚不存在。每次重跑都换新的 `--output`。成功完成运行、结果与日志回传及参考比较后，脚本自动删除本次创建的板端目录；失败时保留供排查。服务器结果与已有目录均保留。

以后需要保留大体积的独立权重文件时，在运行命令末尾加 `--board-cleanup omc`：

| 参数 | 成功后板端保留内容 | `board_cleanup.status` |
|---|---|---|
| `--board-cleanup all`（默认） | 删除整个本次部署目录 | `REMOVED` |
| `--board-cleanup omc` | 仅删除 manifest 中 target/draft 的 OMC 文件；保留独立权重、所有目录、manifest、程序、库及日志 | `OMC_FILES_REMOVED` |
| `--board-cleanup none` | 全部保留 | `RETAINED` |

`omc` 按清单中的确切文件路径删除，不用通配符扫描权重文件。已嵌入 OMC 内部的权重会随 OMC 一起删除；只有独立文件形式的权重能够单独保留。保留目录不代表当前脚本支持复用旧目录，下一次运行仍需新部署目录。

固件标识默认读取板端 `param get const.ohos.fullname`。若内部系统没有这个属性，可显式追加 `--firmware-id '实际固件版本'`。该值用于报告身份记录，不能用“unknown”等占位符代替实际版本。长时间运行可用 `--timeout-seconds` 调整单次 HDC 调用超时，默认3600秒。

## 自动执行的内容

以下是默认 benchmark 流程；当前执行联调使用上文 `--execution-only`，第5–6步改为一次双图 smoke 及执行报告校验。

1. 检查 OMC 清单与文件哈希、参考权重身份、prompt 和序列长度，检查程序是 ARM64 ELF。
2. 创建新的板端目录，按文件精确上传程序、manifest、清单中的 OMC 模型载荷、附属权重及自动定位的运行库，保留相对目录结构。编译诊断文件留在服务器，不随模型上传。
3. 在板端用 `sha256sum` 核对上传内容。
4. 设置本次运行的 `LD_LIBRARY_PATH=<部署目录>/lib:/vendor/lib64/chipsetsdk:<板端原值>`，执行 `--probe` 并回传 JSON；要求 `npu_supported=true`。
5. 根据 manifest 和 probe 自动生成 benchmark 命令，执行普通生成与 DFlash 一致性检查，并各进行3次预热、10次测量。prompt 与生成长度直接取参考 JSON。
6. 回传运行结果、stdout/stderr 和退出码，调用项目现有 `compare` 与服务器参考比较。
7. 仅在上述步骤全部成功后，按 `--board-cleanup` 执行清理；删除前验证本次运行归属标记，结果记录在 `result.json` 的 `board_cleanup.mode/status`。

环境变量仅作用于本次板端脚本，不修改系统环境文件。生成的脚本和 HDC 调用记录保存在服务器结果目录；AIR、prepared、原始权重、Python、CANN、SDK 留在服务器。

## 结果与失败排查

```bash
cat "$HDC_RUN_DIR/result.json"
cat "$HDC_RUN_DIR/comparison.json"
```

主要文件：

| 文件 | 内容 |
|---|---|
| `result.json` | 本次自动流程状态与身份信息 |
| `runtime-probe.json` | 板端 HIAI 版本和 NPU 能力 |
| `benchmark.json` | 默认是生成测量；`--execution-only` 时为 mode=smoke 的两图执行输出 |
| `comparison.json` | 默认模式的参考比较；execution-only 不生成 |
| `probe.stderr`、`benchmark.stderr` | 板端错误日志 |
| `probe.exitcode`、`benchmark.exitcode` | 板端程序实际退出码 |
| `hiai-hilog.txt`、`hiai-hilog.stderr`、`hiai-hilog.exitcode` | probe / benchmark 失败后尝试抓取的板端系统日志与抓取状态，独立于推理状态 |
| `calls/` | 每次 HDC 的 argv、输出和返回码 |
| `failure.json` | 失败阶段与原因，仅失败时生成 |

脚本以非零退出码报告失败，并尽可能拉回已生成日志；probe / benchmark 失败时会直接打印已回传且校验成功的 stderr/stdout 末尾及文件路径。断连时部分文件可能无法取回，保留 `calls/` 和板端目录供后续排查。它还检查远端 shell 的独立退出状态，避免把 HDC 客户端返回0误当作板端程序成功。

若第一步 `create-root` 提示 `missing unique remote exit marker`，先看同屏 HDC stdout/stderr 或 `calls/0001-create-root.json`。例如 `[Fail]Device not founded or connected` 表示所选设备未找到或未连接，shell 未执行；当前尚未上传模型或运行推理。用下列只读命令检查连接：

```bash
"${HDC:-hdc}" -s "$BIP" list targets -v
# SN 应与上述列表中的连接标识完全一致，并且设备已连接。
"${HDC:-hdc}" -t "$SN" -s "$BIP" shell echo HDC_OK
```

如果列表为空或设备状态异常，先确认设备是否连接到 `BIP` 指定的 HDC 服务端、板端调试及授权是否就绪。若列表中的连接标识不同，用该标识更新 `SN`；网络连接的标识可能是 `IP:端口`，并非 USB 序列号，见[官方设备选择说明](https://github.com/openharmony/docs/blob/master/en/application-dev/dfx/hdc.md#connecting-to-the-specified-target-device)。确认 shell 输出 `HDC_OK` 后，换一个新的 `HDC_RUN_DIR` 再运行；不需要重新生成 OMC。

## 历史加载接口与运行库排障

以下保留此前加载接口及运行库问题的排障记录。其中重编 runner、沿用 OMC 的命令仅对应当时的问题；归约小图现已通过，九项布局探针也已通过，当前仅将row Mul compact接入本文开头的新整图流程。

若 `probe` 返回127，先读 `$HDC_RUN_DIR/probe.stderr`。曾出现程序链接 DDK 库，却上传示例中的另一份 `libhiai.so`，导致缺少 `libhiai_ir_build.so` 和 `CheckNPUSupport` 符号。现在自动按构建记录选择库，这一历史库问题本身不要求重建 OMC。

用户随后反馈 `probe.exitcode=0`、`probe.stderr` 为空，但 `benchmark.exitcode=1`；这表示程序探测通过，推理尚未通过。此时应看 benchmark 自己的日志，不再用 probe 日志解释推理失败：

```bash
cat "$HDC_RUN_DIR/benchmark.stderr"
tail -n 60 "$HDC_RUN_DIR/benchmark.stdout"
```

用户补充的错误为 `HIAI CheckModelCompatibility failed: 1`，发生在 target 的独立兼容性检查阶段，尚未执行张量推理。随后提供的内部 `model_run_tool` 示例以相同的 `(HIGH, 0, 1, 0)` 参数创建描述，调用 `SetModelPath` 后直接 `Load`，没有独立调用 `CheckModelCompatibility`。先前框架据此改用文件加载；后续 HCL 日志显示还必须显式传入模型缓冲区（见下节），现以 `Load` 返回状态判断能否加载；模型哈希、运行库版本、NPU 查询及后续张量数量/形状/格式检查继续执行。API 检查返回错误不能直接推断 OMC 本身无效，也不能据此次源码修正认定上板成功。

文件加载及下述 HCL 缓冲区修正都需要重新交叉编译推理程序；无需重新生成 AIR、OMC 或参考。沿用原来的编译环境变量与资源配置，在服务器执行：

```bash
cd "$REPO_DIR"
git pull --ff-only
export HIAI_BUILD_DIR="$AI_RUN_DIR/hiai-load-$(date +%Y%m%d-%H%M%S)"
python3 "$REPO_DIR/targets/cpro/scripts/build_hiai.py" \
  --output "$HIAI_BUILD_DIR"
```

编译成功后执行下面的四参数命令：

```bash
export HDC_RUN_DIR="$AI_RUN_DIR/hdc-load-$(date +%Y%m%d-%H%M%S)"
python3 "$REPO_DIR/targets/cpro/scripts/run_hdc.py" \
  --omc "$GE_OMC_DIR" \
  --runner "$HIAI_BUILD_DIR/build/qwen35_dflash_hiai" \
  --reference "$GE_BOARD_RUN/reference.json" \
  --output "$HDC_RUN_DIR"
```

新脚本会在失败时直接打印已回传日志的末尾；完整内容仍保留在结果目录。模型加载错误还会注明模型名称、路径和运行库版本。当前只完成主机流程与 HIAI 测试替身验证，修正后的真实板端加载与推理等待实测。

### 实际 Load 仍返回 1

用户在上述修正后报告：`HIAI Load failed: 1 [model=qwen35_target, ... runtime=800.631.120.010]`。重新过滤完整 HiLog 后，多次出现同一组加载错误：

```text
CreateModelManager: DDK pipe is HCL.
CreateLocalBuffer: data != nullptr && size > 0 false, return nullptr.
AiModelManagerClientImpl::Load: CreateLocalBuffer return null, load failed
```

这将当前故障定位到模型缓冲区为空或长度为零。只设置 `SetModelPath` 未给此 HCL 加载分支提供有效数据；尚不能据此判断 OMC 内容、算子或固件兼容性。代码现保留路径，并将 OMC 只读映射后显式 `SetModelBuffer(data, size)`，检查设置状态和描述中的地址/长度，再调用 `Load`。映射和描述均保留到 `UnLoadModel` 及客户端销毁之后；不修改模型类型、设备类型或计算精度，也不跳过加载结果与张量验证。

蓝区头文件及[华为公开接口声明](https://github.com/huaweicodelabs/HiAI-Foundation/blob/master/IRBuild/ddk/ai_ddk_lib/include/HiAiModelManagerType.h)的 `SetModelBuffer` 长度是 `uint32_t`。空文件、非普通文件、超过 `UINT32_MAX` 字节的单个 OMC 会提前明确失败，禁止长度截断。这是当前 tiny OMC 的加载修正，尚不提供已验证的完整 35B 大文件加载路线；只读映射也不代表 HIAI 内部不会再分配或复制数据。

此修正要按上节命令重新构建 HIAI 二进制；AIR、OMC、参考文件可以沿用。主机测试要求实际提供正确的模型字节，并在推理、卸载和客户端析构时继续检查原始缓冲区，覆盖 target/draft 部分加载失败、空文件及超过 4 GiB 的稀疏文件拒绝。主机测试及新增缓冲区接口的蓝区 OHOS 编译检查均不能代替内部 DDK 链接和真实板端复测。本地蓝区头文件缺少项目原先使用的 `CheckNPUSupport`，完整适配器未在该蓝区版本编译通过；此限制不通过删除内部 DDK 的 NPU 探测检查来绕过。

脚本在 probe / benchmark 失败时自动尝试 `hilog -x`，使用短超时，将快照及命令自身的 stderr/退出码保存到本次板端目录，然后按已有 SHA256 流程回传；命令参数含义见 [OpenHarmony HiLog 文档](https://github.com/openharmony/docs/blob/master/zh-cn/application-dev/dfx/hilog.md#命令行说明)。不会清空系统日志、修改日志级别或启动持续抓取任务。HiLog 不可用、无权限、超时或回传失败只记录诊断失败，不覆盖原始模型错误，不将推理失败变成成功。

本板端拒绝组合 `-x` 与 `-z`，返回 `Mutlti commands can't be used in combination [CODE: -31]`。此前组合命令有误，现已改为单独的 `hilog -x`：读完当前缓冲区后退出，完整输出保存在文件中，终端只显示有限摘要。如果抓取程序把该参数错误写到 stdout 且返回 0，也会标记为不可用，不能当成成功取得 HIAI 日志。

`hiai-hilog.txt` 是共享日志缓冲区快照，可能包含其他进程或较早的日志；需要按时间、进程和模型名关联，不能把所有匹配行都当成本次错误。程序扫描整个已校验快照，优先选择最新观察到的 `qwen35_dflash_hiai` 会话候选，按 PID、日期和 foundation 初始化分隔，保留头部、首条 E/F、Restore 失败前后及尾部，去重后最多60行。同 PID 的 MUSL 加载错误也纳入，其他服务进程不混入。候选身份尚未独立绑定本次执行，首条 E 可能是已处理的库回退，摘要不自动判定根因。无法解析会话时沿用最后60条关键词候选；完整快照留在服务器输出目录。不会仅检查文件最后 64 KiB，避免后续系统流量掩盖之前的模型错误。摘要排除 `HDC_LOG` 传输日志，`NPU` 按完整单词匹配，并单独匹配 `NPUCL` 等组件，避免误匹配 `input`。候选条数为 0 表示本次筛选没有找到相关内容，不表示模型运行成功或没有底层错误。

仅日志抓取和筛选修改无需重编译；上述 HCL 模型缓冲区修正需要重编译二进制，OMC 可以沿用。更新后使用上面的一条四参数命令，换新的 `HDC_RUN_DIR` 即可；不再传旧命令中的 `--lib-dir`，自动按构建记录准备库。

已有失败现场也可以立即手动取日志：

```bash
hdc -t "$SN" -s "$BIP" shell "hilog -x" > "$HDC_RUN_DIR/hiai-hilog-manual.txt"
grep -n -v 'HDC_LOG' "$HDC_RUN_DIR/hiai-hilog-manual.txt" | \
  grep -i -B 3 -A 8 -E \
  'hiai|ai_fmk|cpucl|npucl|(^|[^[:alnum:]_])npu([^[:alnum:]_]|$)|qwen35_dflash_hiai|CheckModelCompatibility' | tail -n 160
```

最初宽泛筛选只显示新目录创建和文件传输，还混入另一个模型的传输日志；收紧筛选后才发现前面的 HCL 空缓冲区错误。板端日志时间为 `03-06`，而主机目录包含 `20261023`；两端时钟未对齐，不能直接用主机时间排除板端日志。核对现场时以本次 `upload-plan.json` / `result.json` 的 `board_root` 为准，避免重跑先前目录后查看另一次运行的日志。

如果系统日志仍不能定位，再用内部 `model_run_tool` 加载同一份 `target.omc`，保持相同的 HIAI 库集合做对照。内部示例也加载失败时优先核对 OMC/DDK/运行库/固件；示例能加载时再比较客户端初始化。当前尚无这项对照结果。

清理前若发现归属标记不匹配、目录被替换成符号链接，脚本会拒绝删除。清理命令失败或无法确认删除结果时也返回非零，`board_cleanup.status` 为 `FAILED`；已回传的服务器文件仍保留，板端目录可能完整保留、部分清理或已删除，需按日志确认。只有运行与比较已通过而清理失败时，`device_validation` 可为 `PASS`，整体 `status` 仍为 `FAILED`。

`comparison.json` 为 PASS 表示这次普通/DFlash token 与停止原因匹配参考，并通过既有产物身份检查。当前 tiny 测试不证明完整35B量化、内存、质量或性能，也不能仅由 OMC 成功运行推断所有节点都在 NPU 上执行。脚本开发使用模拟 HDC 验证流程；真实 CPro 运行结果由你执行后提供。


## 2026-10-26：完整图分区超限回退已定位

当前执行优先的 smoke 已使用排除 DDK HCL 两库的部署。用户补齐 PID25399 的日志后，终止链条为 `input1 dtype should be fp32, but now it is fp16` → `op:MatMul not support` → `compatibleHelper CheckCompatibility failed by cl CPUCL` → `Recompile failed` → `Load failed`。另外记录了 activation.mode9 和 Cast 类型检查失败。前序缺少已排除 DDK HCL 库的报错之后，运行时已打开板端 `libcpucl_itf.so`，不能将该报错当成新的缺库修复目标。

这次 target 在恢复时遇到 CPUCL 算子兼容性阻断，尚未执行 IO 适配、Process 或 draft。用户随后核对 manifest SHA256 与失败记录相同，并提供该 OMC 的 target OMG 第110650行：`Partition of type: NPUCL is more than 9,`，紧接着 `will rollback to CPU`。这已定位主图编译时的分区超限回退；较早第19509、31879行的 CPU 分区属于 `constNodeGraph`。以下只读命令用于核对历史现场：

```bash
sha256sum "$GE_OMC_DIR/manifest.json"
grep -n -E 'partition type|run[_ -]*by[_ -]*cl|PreGraphSplit|CompileGraph.*(target|SubGraph)' "$GE_OMC_DIR/target/omg.log" | tail -n 100
cat "$GE_OMC_DIR/target/command.json"
```

已有 OMC 不会随代码或 runner 更新获得图改动。用户随后已按 `78d94ef` 重新 export，新的主图窗口 L91094–L127615 再次回退；随后已完成 `inspect-compile` 与六项 rank/keep_dims 对照，六个 MatMul 候选也已全部通过 Load、Process 和数值检查；后续MatMul归约双图仍超限，新支持表确认FP16归约已移除，九项布局探针也已通过，当前仅将row Mul compact接入本文开头的新整图流程。保持原 prepared、FP16 和已通过小图验证的部署策略；即使后续未触发回退，也要检查最终分区与板端执行，不能宣称已修复或全 NPU。
