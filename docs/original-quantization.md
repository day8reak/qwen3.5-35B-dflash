# 复用原量化模型

用户于 2026-10-28 授权将本轮代码和文档更新发布到 GitHub。用户粘贴的六份原件另存于工作目录的私有参考区，未放入 Git，也未执行原导出脚本。原 AIR 编译与本页的模型重建是两个独立入口。

## YAML 是路径来源

`OriginalModelConfig.from_yaml(path)` 读取原 `qwen3_5_moe.yaml`，不要求新增环境变量。绝对路径保持原值，相对路径相对 YAML 所在目录解析。

| 字段 | 用途 |
|---|---|
| `model_file` | 原 HF `config.json` 和基础算子的权重；读取原始 norm、conv、门控参数，不载入 HF 专家库 |
| `quanted_pth` | 原 `data*.safetensors` 中的 INT8 载体 W4 数值和分组 scale；不重新量化 |
| `embedding_weight_path` | 原独立 INT8 `[vocab, hidden]` embedding 文件 |
| `embedding_scale_path` | 原独立 FP32 `[vocab, 1]` embedding scale 文件 |
| `kv_cache_max_len` | 本次缓存容量；与模型位置上限取较小值，原 YAML 为 2048 |

量化权重目录与 embedding 目录可以不同，加载器按各自字段读取，不把 embedding 路径从 `quanted_pth` 推导出来。YAML 中已有的 prompt、旧两段切分注释和示例导出命令不决定本次图的层数；层数取 `model_file/config.json`。

## 权重与计算契约

普通 Linear 的原 `[N,K]` 权重打包为 I4ZN INT32 `[K,N/8]`，scale 从 `[N,K/256,1]` 转为 FP32 `[K/256,N]`。专家权重保留专家轴；group size 为 256。主机实现已与私有原件的纯打包函数逐字节对照，未执行原模块。

专家 scale 必须由实际厂商 `npu_trans_quant_param(..., None, 1)` 或显式提供的等价厂商转换器生成 INT64 QPARAM。FP32 转 INT64、位重解释都不能替代这一步。未提供转换器时，准备器保留 FP32 scale 并报告 `QPARAM_PENDING`，实际 GE 导出拒绝继续。

target 按用户原 Cpro forward 对齐计算边界：embedding 的 INT8 行与 FP32 scale 相乘后，残差流保留 FP32；普通量化 Linear 输出 FP16。GDN 的衰减门计算、gated RMS 与 SiLU 乘积、RoPE 频率和三角函数使用 FP32。RoPE 表在进入 FP16 Q/K 运算前转换；gated RMS 在乘 gate 后转换为 FP16；最终 RMS 保留 FP32，再进入量化词表头。

main/q/k RMS 使用原 HF 权重，在计算时只加一次 1；gated RMS 不加 1。原 loader 和 forward 中重复偏移的组合没有照搬。Conv 历史拼接和 RMS 保持基础算子。prefill/decode 分开构图，GDN chunk 分别为 64/1，不依赖给静态导出图增加动态档位来重新执行 Python 分支。

这些是计算路径对齐，不是原 AIR 位级一致声明。当前基础权重缓存仍为 FP16；PyTorch 的逻辑 GDN 接口在 FP32 门计算后显式转 FP16，GE 根据绑定中的 GDN dtype 转换。GE DynamicQuant 允许显式声明 FP16 或 FP32 输入；所选绑定若要求 FP16，会保留转换节点并记录在 `precision_boundary_casts`。设备支持与数值误差需要实际编译、执行确认。

## 本地准备入口

更新仓库并在仓库外保存原 YAML 后，使用实际 YAML 路径：

```bash
python3 "$REPO_DIR/targets/cpro/scripts/adapt_original_target.py" \
  --config /实际保存位置/qwen3_5_moe.yaml --stage inspect
```

`inspect` 只核对配置、分片头、原权重键/shape/dtype 和 embedding 文件大小，不转换整套权重。需要构建本仓库的新图时再运行：

```bash
python3 "$REPO_DIR/targets/cpro/scripts/adapt_original_target.py" \
  --config /实际保存位置/qwen3_5_moe.yaml --stage prepare
```

每次创建新的 `original-quant-*` 目录。manifest 记录逐 tensor 源字节与输出 SHA256、真实载体类型、外部 embedding 路径及 hash。当前准备入口不支持续跑；失败时保留已完成记录，勿把半成品当成完整权重。

有实际厂商转换模块时可加 `--qparam-module 模块名`；该模块必须暴露 `convert_qparam(FP32 ndarray) -> INT64 ndarray`，保持 `[E,K/256,N]` 形状。本工程不提供猜测编码的替代模块。已有 AIR 编译不依赖此步骤，不需要为重试编译先生成这份权重缓存。

## PyTorch 路径

`OriginalModelConfig.load_pytorch(feature_layers=..., ops=..., quant_ops=..., device=...)` 创建原量化 target。`ops` 是现有显式基础/融合算子适配器；`quant_ops=OriginalQuantOps.from_torch_npu()` 检查动态量化、量化 matmul、QPARAM、MoE routing/GMM 等接口。

真实 torch_npu 绑定要求 NPU 设备，例如 `device="npu:0"`。只有 CPU/DDK 的服务器不能据此执行真实 PyTorch NPU 推理。CPU 测试后端须显式声明 reference，不会被自动选中。

`FusedTargetModel(..., quantization=...)` 保留原 Top1/features/state 接口，`forward(..., return_logits=True)` 可附带完整 FP32 logits。权重加载不建立完整反量化专家库。量化 backend 与 GDN/cache/attention 的物理布局适配仍各自需要实际目标验证。

普通 target 推理可使用新增的 `OriginalCproTargetWrapper`，由 YAML 工厂直接建立：

```python
from qwen35_moe_dflash.original_model_config import OriginalModelConfig

config = OriginalModelConfig.from_yaml(original_yaml)
target = config.load_pytorch_wrapper(
    feature_layers=feature_layers, ops=ops, quant_ops=quant_ops, device="npu:0",
)
prefill = target(input_ids, output_pos=-1)
decode = target(next_token_ids, past_key_values=prefill.state, output_pos=-1)
logits = decode.logits  # FP32 [1, 1, vocab_size]
```

`ops`、`quant_ops` 仍由调用者提供已验证适配器。支持原代码风格的外部量化 embedding：`target(inputs_embeds=int8_rows, embed_scale=fp32_scales, output_pos=-1)`，scale 只乘一次。也可传已经反量化的 FP16/FP32 embedding，此时不再传 scale。`input_ids` 和 `inputs_embeds` 二选一。

`output_pos` 使用本次输入中的零基索引，`-1` 取末 token，`None` 取全部，也接受原调用形式的 INT32/INT64 一维 Tensor。先选行再计算 LM head，features 和状态仍覆盖全部输入 token。拒绝原导出脚本中的越界 `output_pos=[seq_len]`；可选 `effective_length`、`allQLen` 必须等于本次非填充 token 数，不能沿用导出占位值零。

`past_key_values` 接受本仓库 `TargetState`，不接受原 AIR 的 RefData/NZ 列表。可选 `position_ids` 和 `new_kv_cache_pos` 必须与状态位置连续一致。前一轮状态保持不变，下一轮传入返回的 `.state`。这层接口服务 PyTorch 普通 prefill/decode，尚未证明可直接通过 TorchAir 导出。

## 直接 GE 路径

`build_quantized_target` 接入 DynamicQuant、量化 Linear、按 token 选择的专家 GMM。只计算选中的 TopK 专家；普通 scale 是 FP32，专家 scale 是厂商 INT64 QPARAM。`QuantOperatorBindings` 要求完整的 required/optional/dynamic 端口表、属性及来源 SHA256；测试中的 `Test*` 类型名仅用于主机回归，不能当作真实 DDK 绑定。

有实际 DDK 绑定且准备结果为 `PREPARED` 后：

```bash
python3 "$REPO_DIR/targets/cpro/scripts/adapt_original_target.py" \
  --config /实际保存位置/qwen3_5_moe.yaml --stage ge \
  --prepared /实际准备结果目录 \
  --quant-bindings /实际DDK量化绑定.json \
  --fused-bindings "$AI_RUN_DIR/fused-all-f6b81ead7ae8/bindings.json" \
  --sequence-length 64 --past-length 0 --output-pos -1 --full-logits
```

这条命令生成 C++、recipe 和图清单，状态为 `GE_SOURCE_GENERATED`，尚未生成 AIR/OMC。decode 对应 `--sequence-length 1 --past-length 64`；当前 past length 为静态特化，不是任意长度解码循环。省略 `--fused-bindings` 会构建基础 GDN，仅用于明确选择的结构路径。

此图沿用本仓库 functional stateful ABI：外部 embedding 行/scale、显式缓存、Top1/features/状态输出。`--output-pos -1` 只对末 token 计算词表头；`--full-logits` 在状态输出之后追加所选行的 FP32 `target_logits`，`output_contract` 记录实际选择。省略两项保持全部 Top1 行及原输出顺序。

Conv 历史为 K−1，full-attention 缓存为逻辑 ND。PyTorch 使用原 routing/GMM/finalize 调用，GE 当前使用逐 token 收集选中专家的 GMM 图；其物理契约和效果尚待实际验证。这些路径不同于原 AIR 的 89 输入/RefData/paged ABI，不能套用原 AIR 的形状表或旧探针 runner。

当前验证为主机权重打包、独立 CPU 数值参考、小模型状态接续、完整 40 层 64/1 构图和 recipe 检查。真实 DDK 量化端口契约、AIR/OMC 编译、整模设备数值及性能仍未通过验证。
