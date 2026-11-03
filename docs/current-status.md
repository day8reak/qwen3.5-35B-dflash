# 当前阶段测试命令

当前集中推进**PyTorch Draft → TorchAir AIR → X90 OMG/OMC**，手写GE／AddCustom按用户要求暂缓。成功AIR放在`draft/air/`；PyTorch候选仍在`draft/tests/pytorch-omc/`，没有替换公共`draft/omc/`。

目标仍为C16／cap192／K15，常驻预热后proposal同步`Process`平均≤50ms，尚未达到。最新完整OMC为`52cf888e…a6d5eee`，用户回传38例均数值失败、两组Top1各0/86。空context的hidden已非零，但15行逐位相同，Top1全16；context=18时更换anchor，共同有效hidden前缀仍逐位相同。新增KV相对L2仍为0.44%～0.83%，未写区逐位保持，候选分支根因尚未确定。Proposal诊断API均值约743／732ms，不是预热稳态基准。完整结果与模型hash见[当前结果](results.md)。

此前原embedding→单位Muls→Cast→RMS前缀得到全零x。只旁路单位Muls的小图十次通过后，完整编译准备加入同一严格旁路（`draft-static-preparation-v5`）。新完整复测恢复非零hidden，但没有通过精度。保持此准备规则、源AIR、权重与原数值门槛；第一层mask／融合Attention小探针现已完成七组执行，正在从保留输出定位数值错误。

**以下命令均在DDK服务器 `t00826215_atc_mobile` 的 `custom_pypto` 环境执行。** 复用已有AIR、prepared、TBE诊断和支持INT64的native runner，不需要学习权重或重新导出完整模型。

直接INT64运行时长度对照已回传：OMC `cb1c94b0…d69471`，尝试 `qwen35_vector_03fa34f46ce34ac890dbd979dd964b36`。七组均十次逐位稳定，但四个新分支仍与原Attention逐字节相同，原golden全部失败。六组非加权输出精确符合无mask前16行签名，加权组最大签名误差0.00048828125；mask可见性和packed V辅助读回仍逐字节正确。改变常量／运行时来源未恢复结果，不能据此认定长度在物理内核中完全被忽略。

本轮改为一次成套实验。每组独立编译，避免把大量Attention分支放在同一图；对比带／不带回读的小图，检查诊断分支自身的影响。原0.01／0.01容差保持。

| 小模型 | 用例数 | 区分的问题 |
|---|---:|---|
| source | 7 | 原动态mask和Attention基线是否重现 |
| constant | 36 | 原Q16/KV224长度与identity table常量；运行时mask、Q/K响应、逐行V脉冲、有限／负无穷mask、逐query可见性 |
| runtime | 51 | 同样数据测试；三个长度端口和table均为运行时输入；KV长度1/15/16/17/31/32/33/63/64/65/192/208；页表旋转、反转和补偿置换 |
| constant_mask | 1 | 与运行时mask使用相同逐query选行值，只改变mask来源 |
| q32 | 10 | 保持Draft head配置和KV224，Q物理长度及两个Q长度值改为32 |
| target_prefill | 10 | Target的16 Q heads / 2 KV heads / head_dim256，Q64 |
| target_decode | 10 | 同一Target head配置，Q1 |
| observed | 4 | 与runtime对应例比较；回读K、V、有限mask、三个INT64长度和table，要求逐字节一致 |

合计8份小OMC、129例，每例3次预热＋10次保留输出。V编码区分行、head、channel；逐行脉冲覆盖0、15、16、31、32、63、64、127、191、192、207、208、223。KV长度与脉冲见证的预期差异超过原容差，避免原KV32均值与前16行过近导致的“通过”。页表用逐query选行测试，避免全页均值对页置换不敏感。

Target两组仅复现其head配置与Q64／Q1形状，KV仍固定224；没有抽取实际Target AIR，也没有验证Target完整cache或整模型。它们可以检查共用接口是否同样出错，不能单独归因Target精度。第一层Attention路径数值错误已经复现，但端口、布局、编译或内核哪一环导致，仍待本轮区分。

**本次更新后直接复用已编译矩阵，重新执行下方启动命令。** 旧请求 `qwen35_vector_297df57991e34b2a8ad76aed1118f437` 恢复后在HDC发送`attention_matrix_runtime__key_64/plan.txt`时报告180秒超时，`device_execution=NOT_RUN`、`lock_state=NO_KNOWN_OWNED_LOCK`，进程已退出。随后新启动的129例准备通过、`matrix_preparation_failures=[]`，但在打包1021个文件时被路径检查拒绝；同样未运行设备、未持有锁。这两次已知未启动的失败均可复用固定测试目录，无需删除目录或锁，无需重编OMC或native。

打包器原正则遗漏`+`，会误拒部署库`lib/libc++_shared.so`。已改为复用部署流程的相对路径规则，保留越界／重复／保留名称检查，并在错误中显示具体成员；完整清单保存在`payload-manifest.json`。包含实际运行库发现逻辑、合成SDK构建记录及129例夹具的主机检查已走通1021成员打包、解包和逐文件SHA校验；设备执行仍待本轮回传。

先更新代码，然后跳过下面的首次编译步骤，直接运行已有矩阵：

```bash
git -C "$REPO_DIR" pull --ff-only
```

以后若启动状态不明确，可用`validate_draft_vector.py --status "$AI_RUN_DIR/draft/tests/pytorch-attention-matrix-run"`只读检查本地主进程和最后一次通信记录。它不连接设备、不改文件、不获取执行锁；Ctrl-Z／Stopped表示挂起，需在原终端核对`jobs -l`后恢复对应job。request存在或缺少complete均不证明模型已开始／仍在运行，未确认完成的请求不能重复apply。

可以在另一个相同环境的终端使用下方recover查看同一次请求。新版允许独立回收目录重复使用；`PENDING`现在表示`completion-unconfirmed`，并附本地主进程状态、板端已完成例数、最后退出码和未完成例名。新版启动另提供当前例序号、归档阶段标记。上述标记不是进程存活证明，旧请求没有新标记也不等于尚未执行。

新启动会打印准备、打包、上传、校验、启动阶段，每30秒显示等待信息；输入／模型一次打包传输，再核验归档及每个文件SHA256。新请求的日志打包回收，旧请求保留逐文件回收且显示进度。所有例的3次预热、10次快照、原数值门槛和失败停止规则保持。无需重编OMC或native runner；尚未执行的矩阵才按下面两步编译、启动。

首次批量编译时，一组已知主机编译失败不会阻止其余组编译。目录为固定功能槽，不覆盖之前已执行的实验：

```bash
git -C "$REPO_DIR" pull --ff-only
python3 "$REPO_DIR/targets/cpro/tools/probe_draft_attention.py" \
  --matrix \
  --air-export "$AI_RUN_DIR/draft/air" \
  --source "$AI_RUN_DIR/draft/source/prepared" \
  --result "$AI_RUN_DIR/draft/tests/pytorch-result" \
  --ddk-root "$CPRO_DDK_ROOT" --ddk-id "$CPRO_DDK_ID" \
  --tbe-diagnosis "$AI_RUN_DIR/common/tbe-diagnosis" \
  --output "$AI_RUN_DIR/draft/tests/pytorch-attention-matrix"
```

返回 `COMPILED` 或 `PARTIAL` 后，批量运行已成功编译且ABI检查通过的组。编译后的描述无法可靠绑定时，该组单列在 `matrix-preparation.json` 和分析的 `preparation_failures` 中，其余组继续。缺失组不能算整个矩阵通过。此前直接INT64实验已成功执行，本轮复用其native，不需要重编runner：

```bash
python3 "$REPO_DIR/targets/cpro/tools/validate_draft_vector.py" \
  --compiled "$AI_RUN_DIR/draft/tests/pytorch-attention-matrix" \
  --ddk-root "$CPRO_DDK_ROOT" \
  --config "$AI_RUN_DIR/runtime/runtime.json" \
  --output "$AI_RUN_DIR/draft/tests/pytorch-attention-matrix-run" --apply
```

回收同一次执行；若返回 `VECTOR_VALIDATION_PENDING`，先查看附带状态，挂起的启动进程需要在原终端恢复。运行中的请求只重复recover，不重复apply。某例原生进程失败或超时仍停止后续设备调用，并回收已完成部分：

```bash
python3 "$REPO_DIR/targets/cpro/tools/validate_draft_vector.py" \
  --recover "$AI_RUN_DIR/draft/tests/pytorch-attention-matrix-run" \
  --output "$AI_RUN_DIR/draft/tests/pytorch-attention-matrix-result"
```

已有回收张量及`validated_cases`后再分析：

```bash
python3 "$REPO_DIR/targets/cpro/tools/validate_draft_vector.py" \
  --analyze "$AI_RUN_DIR/draft/tests/pytorch-attention-matrix-result"
```

分析重比全部十次输出，汇总覆盖范围、各例错误、常量／运行时来源差异、Q/K/各V行响应、页表响应、带回读／不带回读差异；失败例比较不同KV前缀、ND输出布局与GQA head映射签名，同时保留相关编译日志。数值签名不会自动宣告根因，也不会修改原接受门槛。

新增122例已与独立PyTorch FP64 SDPA核对；7组主机检查覆盖序列化图解释执行、完整8模型批处理、单组编译失败后继续、乱序ABI、十次输出比较、前16行故障识别、部分覆盖和禁止覆盖旧目录。原有7＋14＋12组探针检查通过。以上是主机／mock证据；本矩阵的DDK编译和板端结果尚未取得。

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
