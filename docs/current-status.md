# 当前阶段测试命令

当前集中推进**PyTorch Draft → TorchAir AIR → X90 OMG/OMC**，手写GE／AddCustom按用户要求暂缓。成功AIR放在`draft/air/`；PyTorch候选仍在`draft/tests/pytorch-omc/`，没有替换公共`draft/omc/`。

目标仍为C16／cap192／K15，常驻预热后proposal同步`Process`平均≤50ms，尚未达到。最新用户结果已完成Load及两组各19次调用；原比较门槛下38例均失败。两组Top1均为0/86，首轮全部0，后续proposal均为314；hidden相对L2为1.0～1.408，新增KV为0.44%～0.83%，未写区逐位保持。proposal诊断API均值约727ms，不能视为已完成预热的基准。详见[当前结果](results.md)。

最新诊断已确认两组空context的30720个hidden元素全零。context=18时，更换anchor 17／32／34／38，共同有效位置仍逐位相同；位置之间并非全部相同，context变化也影响hidden。用户随后回传独立Gather探针通过：原完整表、显式INT32 ID，10次embedding与ID回读均逐字节一致，NPUCL一个子图，API均值0.460ms。该结果只验证独立查表，完整Draft根因尚未确定。

最新六输出探针已完成设备执行与回收：ID、raw embedding、gamma观测支路和全1因子均十次逐字节正确；`rms_x`与RMS结果均为32768个有符号零，十次重复稳定。用回读x／gamma计算的RMS也为零，只说明观察值之间一致，不能视为独立精度通过。探针OMC为 `1995dc87eb59bdb4f548cfe03f2374745ed48c61c38c5c02c477ce3b20be5794`，尝试为 `qwen35_vector_cdfbec2849b74e03abac5b749b5c7309`。

当前优先定位正确embedding到错误x的路径：`Muls(value=1) → Cast_7(FP16) → Cast_11(FP32)`及其编译转换。Target CPro内置embedding分支直接查表并转为FP16；Draft多出的单位标量乘法是候选差异，尚未证明它就是根因。gamma支路正确也不等于RMS内核物理输入回读。完整Draft仍数值失败。

编译图检查已完成：CPUCL只包含anchor改形和ID拼接；NPUCL输出的RMS、x、gamma观测值均为FP16，公共接口再Cast回FP32。因子输入也先FP32→FP16。这证明存在接口降精度，不能证明内部每条指令精度，也不足以解释原非零FP16 embedding为何全部变零。当前无保留IR（`NO_DUMPS`），不能继续从主图确认Muls内部属性。

下一步做有界对照：`input-rms-no-scale`只旁路原Gather之后、系数严格为1的FP16 Muls，保持Cast_7／Cast_11、RMS及六个观测输出。只有唯一后续Cast消费者时才允许改线；源系数和参考配置必须都严格等于1，不接受近似相等。原权重、参考、输入文件和精度门槛保持；这是独立诊断副本，尚未修改完整Draft。

**以下命令在DDK服务器 `t00826215_atc_mobile` 的 `custom_pypto` 环境执行。** 原 `pytorch-input-rms-taps` 编译和结果目录保留用于对照；新对照使用固定目录 `pytorch-input-rms-no-scale`。现有AIR、prepared和native runner可复用，无需重新导出AIR。

```bash
git -C "$REPO_DIR" pull --ff-only
python3 "$REPO_DIR/targets/cpro/tools/probe_draft_embedding.py" \
  --stage input-rms-no-scale \
  --air-export "$AI_RUN_DIR/draft/air" \
  --source "$AI_RUN_DIR/draft/source/prepared" \
  --result "$AI_RUN_DIR/draft/tests/pytorch-result" \
  --ddk-root "$CPRO_DDK_ROOT" --ddk-id "$CPRO_DDK_ID" \
  --output "$AI_RUN_DIR/draft/tests/pytorch-input-rms-no-scale"
```

编译摘要应为 `COMPILED`，`extraction.policy=source-first-rms-unit-scale-bypass-v1`；`unit_scale_bypass`记录删除节点、唯一改线及前后hash。`baseline_probe_air_sha256`应与原六输出探针AIR `01cc6b789e756f765e901475979dadae6b90f3c64aa73a171dcbde922e60abe7` 一致。主机需约4.5 GB可用编译空间，脚本会检查。编译后启动同样3次预热、10次比较：

```bash
python3 "$REPO_DIR/targets/cpro/tools/validate_draft_vector.py" \
  --compiled "$AI_RUN_DIR/draft/tests/pytorch-input-rms-no-scale" \
  --cases pytorch_input_rms_no_scale \
  --ddk-root "$CPRO_DDK_ROOT" \
  --config "$AI_RUN_DIR/runtime/runtime.json" \
  --output "$AI_RUN_DIR/draft/tests/pytorch-input-rms-no-scale-run" --apply
```

返回 `VECTOR_VALIDATION_STARTED` 后回收同一次执行；若为 `VECTOR_VALIDATION_PENDING`，稍后重复recover，不重复apply：

```bash
python3 "$REPO_DIR/targets/cpro/tools/validate_draft_vector.py" \
  --recover "$AI_RUN_DIR/draft/tests/pytorch-input-rms-no-scale-run" \
  --output "$AI_RUN_DIR/draft/tests/pytorch-input-rms-no-scale-result"
```

完成后分析实际x及RMS：

```bash
python3 "$REPO_DIR/targets/cpro/tools/validate_draft_vector.py" \
  --analyze "$AI_RUN_DIR/draft/tests/pytorch-input-rms-no-scale-result"
```

通过条件仍为embedding、ID、x、gamma乘积、因子回读逐字节一致，RMS按独立参考 `atol=rtol=0.01` 逐元素通过，并检查十次重复稳定。若旁路后x恢复，支持原Muls所在编译路径有问题，仍不证明独立Muls内核普遍错误；若x仍为零，继续检查Cast／观测路径。对照探针通过后还需将合适修正用于完整Draft并重跑全部KV及候选测试，不能直接提升生产。

主机13项改写检查、23项原探针／对照执行模拟与分析、13项接口和14项只读图检查通过；包括遍历63488种有限FP16位模式的单位乘法等价、signed zero、近1拒绝、原节点字节保留，以及两探针输入／参考／权重hash一致。真实对照编译和设备执行尚待回传。

以下离线命令仍可复查原完整Draft结果：

```bash
git -C "$REPO_DIR" pull --ff-only
python3 "$REPO_DIR/targets/cpro/tools/validate_incremental_draft.py" \
  --diagnose "$AI_RUN_DIR/draft/tests/pytorch-result" \
  --source "$AI_RUN_DIR/draft/source/prepared"
```

该命令复核原比较结果，读取现有二进制快照并检查四个控制量文件的hash和值；不执行模型、不重编、不访问板端、不改阈值。可选`--source`核对PyTorch ABI绑定中的prepared身份及embedding文件hash，只读取mask／anchor行的数值，不装载完整Draft。

重点看`cases[].hidden`的`zero_rows`、`unique_rows_bitwise`、行范数和前8个数；`previous_proposal`比较相邻proposal的共同有效位置，`same_case_isolated_hidden`比较两组对应调用。`reference_hidden`提供独立参考的同类统计。相同Top1不代表hidden逐位相同，L2=1也不单独证明全零，须由这些实际字节检查确认。embedding字段仅反映源文件，不证明板端Gather读到了正确行；控制文件核验同样不是设备输入回读。

新增context KV由特征投影独立生成，不依赖候选分支的Attention结果。KV误差较小、未写区保持，只能验证该部分，不能替代embedding、临时block KV、Attention和hidden/head的逐段定位。源码已对照Target CPro的NZ转换、18个端口及长度属性，尚未找到能解释本次固定候选输出的源码差异。

以下为模型修正并复编后的验证流程，本轮保留现有结果用于上述离线诊断。新编译成功后，仍用原独立CPU参考和17／14 native runner：

```bash
python3 "$REPO_DIR/targets/cpro/tools/validate_incremental_draft.py" \
  --compiled "$AI_RUN_DIR/draft/tests/pytorch-omc" \
  --air-export "$AI_RUN_DIR/draft/air" \
  --source "$AI_RUN_DIR/draft/source/prepared" \
  --ddk-root "$CPRO_DDK_ROOT" \
  --config "$AI_RUN_DIR/runtime/runtime.json" \
  --output "$AI_RUN_DIR/draft/tests/pytorch-run" --apply
```

看到`DRAFT_ACCURACY_STARTED`后，在同一DDK服务器回收同一次执行：

```bash
python3 "$REPO_DIR/targets/cpro/tools/validate_incremental_draft.py" \
  --recover "$AI_RUN_DIR/draft/tests/pytorch-run" \
  --output "$AI_RUN_DIR/draft/tests/pytorch-result"
```

若为`DRAFT_ACCURACY_PENDING`，稍后重复回收命令，不重发`--apply`。runner为`runtime/native/build/qwen35_draft_accuracy`；仅在脚本报告缺少或构建身份不匹配时重建。绑定证据在`pytorch-run/fixture/pytorch-abi.json`，以源AIR语义连线与编译Data.index确定输入角色，KV不按尺寸猜顺序。两序列分别使用参考KV及连续板端KV反馈。

逐元素门槛`atol=0.01 / rtol=0.01`和Top1零差异规则保持；relative L2是诊断量，不是新增验收条件。候选未提升为生产模型，内部精度尚未证明。

以下为首次导出／工具链准备的说明，本轮无需重复执行。当前预处理继续保留Reshape链简化、静态ZerosLike物化和FP16 ND二维权重转置吸收到MatMul的`transpose_x2`标志。每个节点的来源、形状及修改前后标志保存在`compilation/air-preparation.json`中。

以下诊断恢复命令仅用于整个目录缺失的首次准备，已有`common/tbe-diagnosis`时跳过。它不读取权重、不编译模型、不上板，也不修改已安装包或DDK。诊断目录中含模块副本和逐例记录，不能只保留`report.json`，也不要移动：

```bash
python3 "$REPO_DIR/targets/cpro/scripts/direct_model_air.py" check-tbe-platform \
  --ddk-root "$CPRO_DDK_ROOT" \
  --python "$(command -v python3)" \
  --diagnose --jobs 4 \
  --output "$AI_RUN_DIR/common/tbe-diagnosis"
```

诊断返回`DIAGNOSIS_COMPLETE`表示12组执行完毕，不要求每组都成功。`--tbe-diagnosis`会核验候选、对照及DDK来源，兼容处理只在本次编译目录生效，OMG目标保持`kirinx90`。

编译入口在独立快照旁生成`prepared.air`：保留已有的静态Reshape链简化，并将能证明静态尺寸的`ZerosLike`替换为其**输出dtype**的零常量。缓存索引零值为INT32，六层mask零值为FP16；后者允许已观察到的FP32输入／FP16输出形式。空`shape: {}`视为尺寸未记录，不能当成标量；只读取AIR内的描述、shape／axis字面常量和明确的算子形状关系推导尺寸，不从期望配置补造shape。预处理现已移到权重快照和OMG调用之前，失败直接报告节点的dtype／layout等信息。

公共`draft/air`、原始快照、权重和17／14公共接口保持。进度日志的`materialized 7 ZerosLike constants`已由用户实际复编确认，详细证据在`compilation/air-preparation.json`。源码也改为固定零buffer，供以后重新导出使用；本轮已有AIR不必重新导出。TBE兼容选项已有实现，本轮复用；日志摘要补充有界traceback和worker失败节点，避免早期Cast诊断挤掉终止异常。下方第1、2节是首次生成AIR的完整流程，本轮跳过；成功条件和失败检查见第3节。

本轮把每层的QK／mask／Softmax／PV交给Target同款融合算子，使用原Target的converter端口映射；内部构造NZ分块KV，block_size=32，C16／cap192／K15的临时KV设计形状为`[7,64,32,16]`。padding全部屏蔽，保留Draft五层因果／滑窗与末层双向可见性。公共ND缓存仍只提交context，不直接套用Target的原地CacheUpdate。权重和公共接口不变，融合内核的舍入行为仍需实机比较。

## 1. 权重服务器：准备权重并导出AIR

在**保存原始Target／DFlash权重、此前成功导出Target AIR的服务器**执行本节。使用当时的PyTorch／torch_npu／TorchAir环境。该机无需DDK、HDC或板端连接，也无需加载DDK服务器的`env.sh`。

`TARGET_WEIGHTS`对应原Target YAML的 **`model_file`**。它是包含`config.json`和原始BF16 safetensors的目录；准备脚本只读取其中的embedding和LM head，不装载40层Target。

| 输入或YAML字段 | 本次用途 |
|---|---|
| `model_file` → `TARGET_WEIGHTS` | 原始Target目录，读取`model.language_model.embed_tokens.weight`、`lm_head.weight` |
| `quanted_pth` | Target量化权重目录，Draft准备不使用 |
| `embedding_weight_path`、`embedding_scale_path` | Target量化embedding及scale文件，Draft准备不使用 |
| `DRAFT_WEIGHTS` | 独立的`Qwen3.5-35B-A3B-DFlash`原始权重目录，含`config.json`和safetensors；来自该机已有DFlash权重 |

先设置**权重服务器上的路径**并更新代码。示例中的路径需要替换；`AI_RUN_DIR`放在仓库外，两台服务器可以取不同值。

```bash
# 权重服务器，已激活原Target导出环境
export REPO_DIR='/实际路径/qwen3.5-35B-dflash'
export AI_RUN_DIR='/实际路径/qwen3.5-35B-dflash-run'
export TARGET_YAML='/实际路径/原Target配置.yaml'
export DRAFT_WEIGHTS='/实际路径/Qwen3.5-35B-A3B-DFlash'
git -C "$REPO_DIR" pull --ff-only
```

已有`draft/source/prepared/prepared.json`时直接复用，无需重新准备或再设置原始权重路径。首次准备时，在**权重服务器同一终端**读取YAML的`model_file`，再转换所需权重：

```bash
if ! test -f "$AI_RUN_DIR/draft/source/prepared/prepared.json"; then
  TARGET_WEIGHTS=$(python3 - <<'PYCONFIG'
import os
from pathlib import Path
import yaml
p = Path(os.environ['TARGET_YAML']).expanduser().resolve()
config = yaml.safe_load(p.read_text())
value = Path(config['model_file']).expanduser()
print((value if value.is_absolute() else p.parent / value).resolve())
PYCONFIG
  ) && python3 "$REPO_DIR/targets/cpro/scripts/prepare_draft.py" \
    --target-weights "$TARGET_WEIGHTS" --draft-weights "$DRAFT_WEIGHTS" \
    --output "$AI_RUN_DIR/draft/source/prepared"
fi
```

这里按原配置读取规则，将相对`model_file`解析到YAML所在目录。如果目录只留了裁剪后的Target权重，需补齐上述两个原始BF16张量；不能用量化目录或embedding二进制文件代替。准备成功状态为`PREPARED`。

继续在**权重服务器同一终端**更新代码并导出六层增量Draft；本轮已有prepared时直接执行此块：

```bash
git -C "$REPO_DIR" pull --ff-only
python3 "$REPO_DIR/targets/cpro/scripts/export_incremental_draft.py" \
  --frontend pytorch \
  --prepared "$AI_RUN_DIR/draft/source/prepared" \
  --context-rows 16 --cache-capacity 192 --max-draft-tokens 15 \
  --output "$AI_RUN_DIR/draft/air" --replace
```

导出过程暂存在固定`draft/tests/export/`，成功检查后移动到`draft/air/`并清理旧AIR；失败保留原AIR，修复后`--replace`复用同一临时目录。预期`PYTORCH_DRAFT_AIR_EXPORTED`、17输入、14输出、`rms_nodes=32`、`attention_nodes=6`及`gather_index_dtype=int32`。实际逐节点描述见`air-inspection.json`的`attention`和`gather_indices`。这里用已安装converter的`torch_npu.npu_rms_norm → RmsNorm`，Draft gamma直接使用；融合注意力调用`torch.ops.npu.adn_fused_infer_attention.default`，在本次导出进程注册原Target的GE端口映射。使用已成功导出Target的环境，缺少该算子时会明确失败，不自动退回原语图。GQA交给融合算子的`num_heads=32 / num_key_value_heads=8`处理；query mask复制仍为INT32 Gather。PyTorch候选RMS与融合注意力的数值仍需重新验证。

## 2. 权重服务器 → DDK服务器：传输AIR目录

首次导出返回`PYTORCH_DRAFT_AIR_EXPORTED`，或本轮保留AIR完成入口返回`PYTORCH_DRAFT_AIR_INSTALLED`后再传输。权重服务器的整目录为：

```text
draft/air/
  export.air
  weight/                 # AIR用相对路径引用
  air-schema.pb           # 本次安装的TorchAir schema
  manifest.json
  air-inspection.json
```

在**DDK服务器**沿用现有`custom_pypto`与DDK环境、`REPO_DIR`和`AI_RUN_DIR`。用现有文件通道将权重服务器的`draft/air/`整目录接收到DDK的`$AI_RUN_DIR/draft/.air-incoming/`，最终应能直接访问`.air-incoming/export.air`。这只是传输中的临时目录，安装后整目录移动为`draft/air/`，不长期保留：

```bash
# DDK服务器：完整传输结束后执行
git -C "$REPO_DIR" pull --ff-only
python3 "$REPO_DIR/targets/cpro/scripts/export_incremental_draft.py" \
  --import-air "$AI_RUN_DIR/draft/.air-incoming" \
  --output "$AI_RUN_DIR/draft/air" --replace
```

`weight/`、`air-schema.pb`和全部JSON一起传输。两台服务器的根目录可以不同，AIR采用相对权重路径；无需传原始Target／DFlash检查点。本轮DDK上已安装融合版AIR，可直接复用。

## 3. DDK服务器：编译OMC

沿用已配置的`CPRO_DDK_ROOT`、`CPRO_DDK_ID`，确认已安装完整`draft/air`后，复用上述已完成的TBE诊断。旧目录丢失时先执行本页开头的恢复命令：

```bash
git -C "$REPO_DIR" pull --ff-only
python3 "$REPO_DIR/targets/cpro/scripts/compile_incremental_draft.py" \
  --air-export "$AI_RUN_DIR/draft/air" \
  --ddk-root "$CPRO_DDK_ROOT" --ddk-id "$CPRO_DDK_ID" \
  --tbe-diagnosis "$AI_RUN_DIR/common/tbe-diagnosis" \
  --output "$AI_RUN_DIR/draft/tests/pytorch-omc" --replace
```

编译自动识别公共目录中的PyTorch AIR。当前`draft/tests/pytorch-omc`只保存尚未绑定运行接口的编译工作；它不作为长期模型目录。正式OMC位置仍是`draft/omc`，现有可运行模型在新模型接通前保留。使用外置权重编译后流式打包成`pytorch-omc/model.omc`，避免重走大模型内嵌序列化失败路径。此候选编译器需要输入快照、编译权重与打包模型同时存在，预检为额外三份权重大小＋1GiB，编译临时空间可能更大；不会把旧的6GiB估计用于这里。

预期状态为`PYTORCH_DRAFT_OMC_COMPILED`，但`compiled_abi=NOT_BOUND`、`compiled_precision=NOT_VALIDATED`。检查`compiled-interface.json`的子图归属和边界Cast，再绑定实际输入输出槽位、做Load及独立KV验证。这一候选尚不能直接传给现有`--incremental-draft`。此前内置RmsNorm小图有Load失败，换PyTorch前端不保证自动解决。

若仍失败，先在**DDK服务器**读取同一目录的错误，避免未经定位连续重编：

```bash
python3 "$REPO_DIR/targets/cpro/scripts/compile_incremental_draft.py" \
  --inspect-compile "$AI_RUN_DIR/draft/tests/pytorch-omc"
```

## 4. 本阶段通过条件与后续

用户已回传融合版编译成功及38次板端调用完成，但数值失败；当前按本页开头命令检查保留张量。CPUCL3／NPUCL2及跨子图Cast属于编译结构观察，不证明内部精度或达到50ms。

前一轮主机检查覆盖C16／C64、K15／7／3／1、滑窗开关共280步：使用CPU模拟Target布局／调用，与原语参考的Top1、hidden、全部K/V完全一致，输入KV和未写区保持。真实宽度的布局索引检查覆盖212992个元素；四档完整形状图捕获均为6个测试融合节点、32个测试RMS节点、17／14接口及INT32 Gather。这些检查验证适配器和mask／缓存语义，不验证厂商融合内核。OMG及板端执行已由用户报告成功；最新板端数值失败，50ms目标未达到。
