# 当前阶段测试命令

当前集中推进**PyTorch Draft → TorchAir AIR → X90 OMG/OMC**。按用户要求暂缓手写GE和AddCustom优化，不再要求运行相关探针。成功导出的AIR统一放`draft/air/`，编译和测试直接引用它。当前可运行OMC仍放`draft/omc/`；新OMC完成接口绑定和板端验证后再替换运行模型。`tests/`用于测试记录和可清理的构建临时文件。

目标仍为C16／cap192／K15，常驻预热后单次proposal同步`Process`平均≤50ms，尚未达到。当前GE整模型profiling为518.990ms。融合版PyTorch AIR已进入OMG，本次终止在`ReshapeFusion`删除节点时的`ByPassNode PreCheck`，不是此前的Gather INT64检查。现有结果见[当前结果](results.md)，算子流程见[模型结构](model-architecture.md)。

**本轮只在DDK服务器／`custom_pypto`环境执行。无需重导出或重新传输AIR、权重。** 沿用现有`REPO_DIR`、`AI_RUN_DIR`及DDK变量：

```bash
git -C "$REPO_DIR" pull --ff-only
python3 "$REPO_DIR/targets/cpro/scripts/compile_incremental_draft.py" \
  --air-export "$AI_RUN_DIR/draft/air" \
  --ddk-root "$CPRO_DDK_ROOT" --ddk-id "$CPRO_DDK_ID" \
  --output "$AI_RUN_DIR/draft/tests/pytorch-omc" --replace
```

新编译入口在独立快照旁生成`prepared.air`，只合并满足静态shape、相同dtype／ND布局、单消费者且无控制依赖的连续Reshape。保留最终Reshape与目标shape，不跨Transpose／Cast／Identity合并；算术、注意力、mask、权重和17／14公共接口保持。公共`draft/air`和原始编译快照不改写。日志会显示`Prepared compiler AIR: APPLIED, removed N intermediate Reshapes`；具体节点保存在`compilation/air-preparation.json`。若没有满足条件的链，则显示`UNCHANGED`，不会猜测形状后强行改写。

这是针对失败阶段的等价简化候选，已通过主机检查，尚未在用户DDK验证。不能仅凭最后一行`removing Reshape_49`断言该节点本身非法；早期Cast常量图报错也保留为线索。下方第1、2节是首次生成AIR的完整流程，本轮跳过；成功条件和失败检查见第3节。

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

沿用已配置的`CPRO_DDK_ROOT`、`CPRO_DDK_ID`，确认已安装完整`draft/air`后：

```bash
git -C "$REPO_DIR" pull --ff-only
python3 "$REPO_DIR/targets/cpro/scripts/compile_incremental_draft.py" \
  --air-export "$AI_RUN_DIR/draft/air" \
  --ddk-root "$CPRO_DDK_ROOT" --ddk-id "$CPRO_DDK_ID" \
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

用户已完成融合版AIR导出并进入OMG，目前在ReshapeFusion阶段失败；下一阶段需要`PYTORCH_DRAFT_OMC_COMPILED`。随后核对编译后的RmsNorm、Cast和实际输入输出槽位，完成板端Load及独立KV精度验证，再采集同输入的完整模型profiling。PyTorch仍使用TorchAir和OMG，换前端不代表消除了内部降精度或加载问题。

前一轮主机检查覆盖C16／C64、K15／7／3／1、滑窗开关共280步：使用CPU模拟Target布局／调用，与原语参考的Top1、hidden、全部K/V完全一致，输入KV和未写区保持。真实宽度的布局索引检查覆盖212992个元素；四档完整形状图捕获均为6个测试融合节点、32个测试RMS节点、17／14接口及INT32 Gather。这些检查验证适配器和mask／缓存语义，不验证厂商融合内核。真实融合版AIR已保存并进入编译；OMG、板端精度与性能仍未通过。
