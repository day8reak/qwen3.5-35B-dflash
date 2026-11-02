# 当前阶段测试命令

当前集中推进**PyTorch Draft → TorchAir AIR → X90 OMG/OMC**。按用户要求暂缓手写GE和AddCustom优化，不再要求运行相关探针。公共`draft/air`、`draft/omc`继续保留现有可运行模型；PyTorch候选使用固定测试槽位，通过加载、ABI与数值验证后再替换。

目标仍为C16／cap192／K15，常驻预热后单次proposal同步`Process`平均≤50ms。当前GE整模型profiling为518.990ms，PyTorch候选尚无服务器导出、编译或板端结果。现有结果见[当前结果](results.md)，算子流程见[模型结构](model-architecture.md)。

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

继续在**权重服务器同一终端**导出六层增量Draft：

```bash
python3 "$REPO_DIR/targets/cpro/scripts/export_incremental_draft.py" \
  --frontend pytorch \
  --prepared "$AI_RUN_DIR/draft/source/prepared" \
  --context-rows 16 --cache-capacity 192 --max-draft-tokens 15 \
  --output "$AI_RUN_DIR/draft/tests/pytorch-air" --replace
```

预期`PYTORCH_DRAFT_AIR_EXPORTED`、17输入、14输出、32个`RmsNorm`。这里用已安装converter的`torch_npu.npu_rms_norm → RmsNorm`（[公开converter](https://github.com/Ascend/torchair/blob/master/python/torchair/_ge_concrete_graph/ge_converter/custom/npu_rms_norm.py)），与Target CPro接口一致；不使用ND工程的`AdnRmsNorm`。Draft gamma直接使用，不加Target零中心权重的1。mask复制与GQA继续使用GE已采用的Gather表达，避免重新引入BOOL查询轴广播和五维KV展开。PyTorch候选在FP32完成gamma乘法后转FP16，与旧GE RMS的中间舍入不同，需要重新做数值验证。

## 2. 权重服务器 → DDK服务器：传输AIR目录

只有第1步返回`PYTORCH_DRAFT_AIR_EXPORTED`后才传输。权重服务器生成的整目录为：

```text
draft/tests/pytorch-air/
  export.air
  weight/                 # AIR用相对路径引用
  air-schema.pb           # 本次安装的TorchAir schema
  manifest.json
  air-inspection.json
```

在**DDK服务器**沿用现有`custom_pypto`与DDK环境、`REPO_DIR`和`AI_RUN_DIR`，先准备接收位置：

```bash
# DDK服务器：使用该机已有的AI_RUN_DIR
mkdir -p "$AI_RUN_DIR/draft/tests"
```

用现有跨机文件通道，将权重服务器的`$AI_RUN_DIR/draft/tests/pytorch-air/`整目录传到**DDK服务器**的`$AI_RUN_DIR/draft/tests/pytorch-air/`。最终应能直接访问`pytorch-air/export.air`，不要多套一层同名目录。`weight/`、`air-schema.pb`和manifest等文件一起传，不只传AIR，也无需把原始Target／DFlash权重传到DDK服务器。

两台服务器的根目录可以不同；AIR使用相对权重路径。DDK服务器不需要安装TorchAir。两个构建入口的`--replace`只替换本工具已结束的候选槽位。

## 3. DDK服务器：编译OMC

沿用已配置的`CPRO_DDK_ROOT`、`CPRO_DDK_ID`，确认收到完整`draft/tests/pytorch-air`后：

```bash
git -C "$REPO_DIR" pull --ff-only
python3 "$REPO_DIR/targets/cpro/scripts/compile_incremental_draft.py" \
  --air-export "$AI_RUN_DIR/draft/tests/pytorch-air" \
  --ddk-root "$CPRO_DDK_ROOT" --ddk-id "$CPRO_DDK_ID" \
  --output "$AI_RUN_DIR/draft/tests/pytorch-omc" --replace
```

编译自动识别PyTorch AIR。使用外置权重编译后流式打包成`pytorch-omc/model.omc`，避免重走大模型内嵌序列化失败路径。此候选编译器需要输入快照、编译权重与打包模型同时存在，预检为额外三份权重大小＋1GiB，编译临时空间可能更大；不会把旧的6GiB估计用于这里。

预期状态为`PYTORCH_DRAFT_OMC_COMPILED`，但`compiled_abi=NOT_BOUND`、`compiled_precision=NOT_VALIDATED`。检查`compiled-interface.json`的子图归属和边界Cast，再绑定实际输入输出槽位、做Load及独立KV验证。这一候选尚不能直接传给现有`--incremental-draft`。此前内置RmsNorm小图有Load失败，换PyTorch前端不保证自动解决。

## 4. 本阶段通过条件与后续

先取得`PYTORCH_DRAFT_AIR_EXPORTED`和`PYTORCH_DRAFT_OMC_COMPILED`。随后核对编译后的RmsNorm、Cast和实际输入输出槽位，完成板端Load及独立KV精度验证，再采集同输入的完整模型profiling。PyTorch仍使用TorchAir和OMG，换前端不代表消除了内部降精度或加载问题。

已有主机检查覆盖K15／7／3／1共37步，Top1无差异、KV最大差0、hidden最大绝对差0.001953125；另检查了17输入／14输出、32个测试RMS节点、全宽4×4完整dot投影和可移植AIR打包。这些是CPU、图捕获及模拟工具检查，实际服务器与板端验证仍待完成。
