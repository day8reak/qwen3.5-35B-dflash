# Qwen3.5-MoE（35B-A3B）NPU 工程：算子与量化分析

> 分析对象目录：`qwen35_moe/`

> 适用平台：昇腾 310P（`Ascend310P3`）

> 相关文件：

> - `models/configuration_qwen3_5_moe.py` —— 模型配置

> - `models/modeling_qwen3_5_moe.py` —— HuggingFace 参考实现（含 Vision 多模态）

> - `models/modeling_qwen3_5_moe_hiai_{nz,nd,cpro}.py` —— 三个昇腾适配变体

> - `models/export_model_wrapper_qwen3_5_moe.py` —— 权重加载 + 4 个导出/推理 wrapper

> - `export_glm_model_qwen3_5_moe.py` —— 自定义算子 GE Converter 注册 + AIR 导出

> - `quant.py` —— 权重量化与保存

> - `utils.py` —— 量化权重装载 / 硬件格式转换工具

> - `inference.py` / `inference_segment.py` —— 单段 / 2 分段推理

---

## 目录

1. [模型结构速览](#1-模型结构速览)

2. [模型代码总结与大致步骤](#2-模型代码总结与大致步骤)

3. [用到的算子总览](#3-用到的算子总览)

4. [自定义算子详解（重点）](#4-自定义算子详解重点)

5. [torch_npu 内置算子详解](#5-torch_npu-内置算子详解)

6. [量化方案分析](#6-量化方案分析)

7. [评审发现的疑点 / 需确认项](#7-评审发现的疑点--需确认项)

---

## 1. 模型结构速览

| 配置项 | 值 | 说明 |

|---|---|---|

| `num_hidden_layers` | 40 | Transformer 层数 |

| `hidden_size` | 2048 | 隐藏维度 |

| `num_attention_heads` / `num_key_value_heads` | 16 / 2 | 全注意力头数（GQA，group=8） |

| `head_dim` | 256 | 每头维度 |

| `vocab_size` | 248320 | 词表 |

| `num_experts` / `num_experts_per_tok` | 256 / 8 | MoE 专家数 / 每 token 激活专家数 |

| `moe_intermediate_size` / `shared_expert_intermediate_size` | 512 / 512 | MoE / 共享专家中间维 |

| `layer_types` | `linear_attention` 30 层 + `full_attention` 10 层 | 混合注意力（每 4 层 1 个 full_attn，即 i+1≡0 mod 4） |

| `linear_num_key_heads` / `linear_num_value_heads` | 16 / 32 | 线性注意力 K/V 头数 |

| `linear_key_head_dim` / `linear_value_head_dim` | 128 / 128 | 线性注意力头维 |

| `linear_conv_kernel_dim` | 4 | 线性注意力 causal conv 卷积核 |

| `kv_cache_max_len` | 2048 | KV cache 长度（block_size=32） |

| `partial_rotary_factor` | 0.25 | 部分 RoPE（仅旋转头维的 1/4） |

| `hidden_act` / `tie_word_embeddings` | silu / False | 激活 / 不绑定 embedding |

单层 `DecoderLayer` 结构（NZ/Cpro/ND 一致）：

```

input_layernorm(RMSNorm)

  └─ Token Mixer（二选一）

       ├─ full_attention:  q_proj(输出2x → query+gate) → q/k RMSNorm → partial RoPE → PagedAttention → ×sigmoid(gate) → o_proj

       └─ linear_attention: in_proj_qkv/z/b/a → causal conv1d → split q/k/v → Gated Delta Rule → GatedRMSNorm → out_proj

  ＋ residual

post_attention_layernorm(RMSNorm)

  └─ MoE 块：router(gate) → topk+softmax → init_routing → grouped_matmul(w13) → swiglu → grouped_matmul(w2) → finalize_routing

              ＋ shared_expert(MLP) × sigmoid(shared_expert_gate)

  ＋ residual

```

三个变体（NZ/ND/Cpro）差异要点：

| 差异 | NZ (`..._hiai_nz.py`) | ND (`..._hiai_nd.py`) | Cpro (`..._hiai_cpro.py`) |

|---|---|---|---|

| full attention | `adn_fused_infer_attention`（分页） | eager attention + `npu_scatter_nd_update_` | `adn_fused_infer_attention` |

| KV 写入 | `npu_cache_update_` | `npu_scatter_nd_update_`② | `npu_cache_update_` |

| RMSNorm | `npu_rms_norm`（forward 里 `1.0+weight`①） | `adn_rms_norm`（自定义 GE，直接用折叠权重） | `npu_rms_norm`（forward 里 `weight+1.0`①） |

| linear attention | `causal_conv1d_update` + `npu_chunk_gated_delta_rule` | 同 NZ（CPU fallback 风格） | 自定义 `LinearAttentionConvstateUpdate` + `F.conv1d` + `npu_chunk_gated_delta_rule` |

| MoE 路由 gating | `softmax+topk`（PyTorch） | `softmax+topk`（PyTorch） | `npu_moe_gating_top_k_softmax`（融合） |

| 专家量化 | 默认 `CompressedTensorW4A8Int8MoEGMMMethod` | 默认 W4A8 GMM | 强制 W4A8 GMM |

| 布局 | NZ（310P） | ND | NZ（310P） |

> ① NZ/Cpro 的 `npu_rms_norm` 调用处传 `1.0 + self.weight`，而装载时 `weight` 已是折叠值 `1 + w_HF`，两者叠加会得到 `2 + w_HF`；ND 的 `adn_rms_norm` 直接使用 `self.weight`。该不一致见文末评审疑点。

> ② ND 的 attention 仍定义了用 `npu_cache_update_` 的 `update()` 辅助函数，但 `forward` 实际走 `npu_scatter_nd_update_` + eager 路径（fused 分支为死代码）。

---

## 2. 模型代码总结与大致步骤

### 2.1 工程流程总览

```

quant.py（离线权重量化）

   │ 产出 *_quant_weight(int8/int4)、*_quant_scale、embedding_weight/scale.bin

   ▼

export_glm_model_qwen3_5_moe.py（离线导出）

   注册 5 个自定义算子 GE Converter

   → torchair.dynamo_export → AIR（.air）

   → atc --framework=1 --soc_version=Ascend310P3 → .om（图编译产物）

   ▼

inference.py / inference_segment.py（NPU 推理）

   加载 HF 权重 + 量化权重 → Prefill → Decode 自回归

```

### 2.2 权重加载（`export_model_wrapper_qwen3_5_moe.py`）

1. `_load_text_config()` 直接从 `config.json` 读 `text_config`（绕过不识别该模型的旧版 AutoConfig），并补默认字段。

2. `_load_hf_state_dict()` 读 safetensors，去掉 `model.language_model.` / `model.` 前缀，跳过 `model.visual.*`（视觉）与 `mtp.*`（多 token 预测）。

3. `load_npu_model()` 按变体构建骨架（`nz/nd/cpro`），只构建需要的层（`layer_num` + `start_layer_idx` 支持分段）。

4. 权重映射：NPU 参数名 → HF key。关键转换：

   - **RMSNorm**：`weight_NPU = 1 + weight_HF`（HF 存零中心权重，NPU 端存折叠后的 1+w）。ND 的 `adn_rms_norm` 直接使用该值；**NZ/Cpro 的 `npu_rms_norm` forward 写成 `1.0 + self.weight`，等于对折叠权重再加 1（≈2+w）**，疑为遗留问题（各文件顶部注释均为“store gamma=1+weight 以便直接使用”），建议核对（见文末疑点）。

   - **专家 3D 权重**：`gate_up_proj [E, 2*inter, hidden] → w13_weight [E, hidden, 2*inter]`；`down_proj [E, hidden, inter] → w2_weight [E, inter, hidden]`（转置）。

   - 量化模式下 `w13/w2` 权重/scale/bias 不在此时拷入，留给 `combined_experts_weight_quant_3d`。

5. 构建 4 个 wrapper：`Full(NZ)`、`Cpro`、`First`（前 N 层）、`Second`（后段 + norm + lm_head）。

### 2.3 推理步骤（`inference.py` / `inference_segment.py`）

**单段模式（`inference.py`）**：

1. 加载模型（量化开关 → `quant_model_inplace` 装载量化权重）。

2. 初始化 KV cache（按 `layer_types`）：

   - `full_attention`：NZ block-table `[max_blk=64, expand_len=32, block_size=32, 16]` fp16（K、V 各一）。

   - `linear_attention`：`conv_state [1, 8192, 4]` fp16 + `recurrent_state [1, 32, 128, 128]` fp16。

3. **Prefill**：一次处理全部输入 token；逐层经 `DecoderLayer`；`npu_cache_update_` 写入 KV cache；取最后 token logits → argmax 得首 token。

4. **Decode 循环**：每次 1 个 token；`fill_attention_mask` 更新 mask；更新 KV cache；`apply_repetition_penalty` 惩罚重复；`argmax`/`topk` 采样；遇到 EOS 停止。

5. topK 出口：wrapper 支持返回 `topk_values / topk_indices`（导出推理常用），或全 logits。

**2 分段模式（`inference_segment.py`，`split_layer=20`）**：

- 段 1：NZ 变体前 20 层（`FirstWrapper`），输出 `hidden_states` + 各层 KV cache。

- 段 2：后段模型（`SecondWrapper`）接 `hidden_states`，继续剩余层 + `norm` + `lm_head` → logits。

- 显存策略：每段只在计算时 `.to(npu)`，算完 `.to('cpu')` + `empty_cache()`，NPU/CPU 交替搬移以跑下 35B 模型（310P 单卡显存受限）。

### 2.4 导出流程（`export_glm_model_qwen3_5_moe.py`）

1. 用 `torch.library.Library('npu', 'IMPL', 'Meta')` 注册 5 个自定义算子的 **Meta 实现**（fake 推测输出 shape/dtype），供 torch._dynamo 图捕获。

2. 为需要 Functionalize 的算子注册 FRAGMENT 变体（如 `npu_cache_update__functional`）。

3. 用 `@register_fx_node_ge_converter` 把 FX 节点映射为 `torchair.ge.custom_op('XXX', ...)`，落地为昇腾 GE 算子：

   - `CacheUpdate`、`AdnRmsNorm`、`AdnFusedInferAttention`（本工程自写 converter）；

   - `ChunkGatedDeltaRule`、`LinearAttentionConvstateUpdate`（同上）；

   - MoE 族算子（`npu_moe_init_routing_v2`、`npu_moe_finalize_routing`、`npu_grouped_matmul`、`npu_swiglu`）由 torchair 内置 converter 处理（见文件头注释）。

4. 构造 dummy 输入（`input_ids`、`attention_mask`、`position_ids`、`past_key_values`、`new_kv_cache_pos`、`inputs_embeds`、`embed_scale`、`allQLen`、`topK` 等）。

5. `torchair.dynamo_export(..., dynamic=False, export_path=...)` → 生成 AIR 图；再用 `atc` 转 `.om`（yaml 注释中给出命令）。

---

## 3. 用到的算子总览

### 3.1 按来源分类

| 类别 | 算子 | 用途 |

|---|---|---|

| **本工程自定义（GE 自研，见第 4 节）** | `CacheUpdate`、`AdnRmsNorm`、`AdnFusedInferAttention`、`ChunkGatedDeltaRule`、`LinearAttentionConvstateUpdate` | KV cache 写入 / RMSNorm / 分页注意力 / 线性注意力核心 / conv 状态更新 |

| **torch_npu 内置（见第 5 节）** | `npu_dynamic_quant`、`npu_quant_matmul`、`npu_grouped_matmul`、`npu_swiglu`、`npu_moe_init_routing_v2`、`npu_moe_finalize_routing`、`npu_moe_gating_top_k_softmax`、`npu_cache_update_`、`npu_scatter_nd_update_`、`npu_rms_norm`、`npu_trans_quant_param` | 量化 / GEMM / MoE 路由 / 归一化 / 工具 |

| **PyTorch 原生** | `nn.Embedding`、`nn.Linear`、`nn.Conv1d`(groups)、`F.conv1d`、`F.silu`、`F.softplus`、`F.softmax`、`torch.sigmoid`、`torch.topk`、`chunk`、`split`、`cat`、`transpose`、`reshape`、`pad`、`matmul`、`cumsum`、`argmax` | 通用算子 |

| **外部库 / 参考实现** | `causal_conv1d_update/ fn`（causal-conv1d）、`torch_causal_conv1d_update`、`torch_chunk_gated_delta_rule`、`torch_recurrent_gated_delta_rule`（flash-linear-attention 参考） | linear attention CPU 兜底 / 对齐用 |

### 3.2 按功能分类

| 功能 | 算子 |

|---|---|

| 归一化 | `AdnRmsNorm` / `npu_rms_norm` / `Qwen3_5MoeRMSNormGated`（RMSNorm+silu gate） |

| 注意力 | `AdnFusedInferAttention`（全注意力）、`ChunkGatedDeltaRule`（线性注意力）、`npu_cache_update_`/`npu_scatter_nd_update_`（KV 写入）、`causal_conv1d`系列（线性注意 conv） |

| 门控 / 投影 | `npu_quant_matmul`（QLinear A8W4）、`nn.Linear`（未量化）、`apply_rotary_pos_emb`（部分 RoPE） |

| MoE | `npu_moe_gating_top_k_softmax`（Cpro）、`F.softmax+topk`（NZ/ND）、`npu_moe_init_routing_v2`、`npu_grouped_matmul`、`npu_moe_finalize_routing` |

| 量化 | `npu_dynamic_quant`（激活 per-token）、`npu_quant_matmul`、`npu_trans_quant_param`（scale→int64 QPARAM） |

---

## 4. 自定义算子详解（重点）

> 本工程通过在 `export_glm_model_qwen3_5_moe.py` 中用 `torch.library` 注册 PyTorch 侧接口 + `torchair.register_fx_node_ge_converter` 注册 GE 映射，把 5 组计算收敛到昇腾**自研 C++ 算子**（编译进 .om 图）。每个算子定义了三层：

> 1. **Meta 实现**（`@torch.library.impl(m, ...)` / `register_fake`）：让 Dynamo 在 CPU/张量无关的情况下推断输出 shape/dtype；

> 2. **Apple/Fragment 实现**（如 `Functionalize`）：把就地算子包成 functional 版本；

> 3. **GE Converter**（`@register_fx_node_ge_converter`）：把 FX 节点翻译成 `torchair.ge.custom_op('XXX')`，即最终的 GE 图算节点。

### 4.1 CacheUpdate（KV Cache 分页写入）

- **PyTorch 侧接口**：`torch_npu.npu_cache_update_(self, updates, target_block, offset_in_block)`（就地）；

  导出版 `torch.ops.npu.npu_cache_update__functional`（Functionalize 包装，返回克隆）。

- **GE 算子名**：`CacheUpdate`

- **注册位置**：`export_glm_model_qwen3_5_moe.py:37-66`

**输入 / 输出格式**（以当前模型实际调用为准，`Qwen3_5MoeAttention.update`）：

| 参数 | Shape / 类型 | 说明 |

|---|---|---|

| `x`（self，就地=输出） | `[max_blk, expand_len, block_size, 16]` fp16 | NZ KV cache；`max_blk=64`，`expand_len=num_kv_heads*head_dim/16=32`，`block_size=32`，末尾 16 为 NZ 分形块 |

| `updates` | `[seq_len, expand_len, 16]` fp16 | 新 K 或 V，已按 16 分形切块（`k_flattened = new_k.reshape(1, s, -1, 16)`） |

| `target_block` | `[num_blk]` int32 | 由 `block_table[:, block_idx]` 查出的目标块 id |

| `offsetInBlock` | int32 | 块内偏移（`new_kv_cache_pos % block_size`） |

**功能**：把当前 step 计算出的新 K/V 写入分页式（PagedAttention）block-table KV cache 的指定块、指定偏移处。这是 decode 阶段 KV cache 更新的核心算子；prefill 时 `new_kv_cache_pos = 0..seq_len-1`，decode 时每个 step 写入 1 个 token。

---

### 4.2 AdnRmsNorm（RMS 归一化）

- **PyTorch 侧接口**：`torch_npu.adn_rms_norm(self, gamma, epsilon=1e-6)`（ND 变体的 `Qwen3_5MoeRMSNorm.forward` 使用）；

  NZ/Cpro 用 `torch_npu.npu_rms_norm`（内置），GE 侧同样可落到该实现。

- **GE 算子名**：`AdnRmsNorm`

- **注册位置**：`export_glm_model_qwen3_5_moe.py:70-89`

**输入 / 输出格式**：

| 参数 | Shape / 类型 | 说明 |

|---|---|---|

| `x`（self） | `[*, hidden_size]` fp16 | 待归一化输入（如 `[B,S,2048]` 或 `[B,head,S,256]`） |

| `gamma` | `[hidden_size]` fp16 | 权重；零中心版本已融合：NPU 直接存 `1 + weight_HF` |

| `epsilon`（attr） | float | `rms_norm_eps=1e-6` |

输出（2 个）：

| 输出 | Shape | 说明 |

|---|---|---|

| `y` | 同 x | 归一化结果 `x * (1/sqrt(mean(x²)+eps)) * gamma` |

| `rstd` | 前 `self.dim()-gamma.dim()` 维同 x，末尾 `gamma.dim()` 个维为 1 | `1/sqrt(var+eps)`，供后续 fused 使用 |

**功能**：RMSNorm。与标准实现不同（HF 存零中心 weight，forward 里 `×(1+w)`），本项目在**装载时**把 `1+w` 预计算进 `gamma`（`weight_NPU = 1 + weight_HF`）：

- **ND**：`adn_rms_norm(hidden_states, self.weight, eps)` 直接用折叠后的 `weight`，一步完成，导出图里无额外 `add`；

- **NZ/Cpro**：`npu_rms_norm(...)` 的 forward 里仍写 `1.0 + self.weight`（即对已折叠权重再加 1，等价于 `2 + w_HF`）——与其类注释“We store gamma = 1 + weight……”自相矛盾，疑为折叠改造前的遗留写法，**建议按 ND 口径核对数值**（见文末评审疑点）。

> 注意 `Qwen3_5MoeRMSNormGated`（线性注意力内 gated norm）在三个变体中都是直接用 `self.weight`、forward 不额外加 1。但该 gated 权重在装载端同样被 loader 按“所有 `norm.weight`”做了 `1+w` 折叠，而 HF 中它是非零中心、forward 直接乘 `w_g`（≈1 量级）——因此 NPU 侧等效 gamma≈`1+w_g` 可能偏离 HF 口径约一倍，属数值疑点（详见文末评审疑点 2），需 dump 权重或对拍 HF 确认。

---

### 4.3 AdnFusedInferAttention（融合分页注意力）

- **PyTorch 侧接口**：`torch_npu.adn_fused_infer_attention(query, key, value, ...)`（NZ/Cpro 的 full attention 核心）；

- **GE 算子名**：`AdnFusedInferAttention`

- **注册位置**：`export_glm_model_qwen3_5_moe.py:92-149`

**输入 / 输出格式**（当前模型 `attn_params` 实参）：

| 参数 | Shape / 类型 | 说明 |

|---|---|---|

| `query` | `[-1, num_heads*head_dim/16, q_seq_len, 16]` fp16 | Q，NZ 布局（`transform_nd_2_nz` 后 reshape），即 `[1, 256, seq_len, 16]` |

| `key` | list，`[1, expand_len, block_size, 16]` fp16 | K，KV cache（NZ，已含全部已写 token） |

| `value` | list，同上 | V |

| `actual_seq_lengths_q` | list[int] int64 | 经 `dtype_promote` 提升为 int64；`[q_seq_len]` |

| `actual_seq_lengths_kv` | list[int] int64 | `[kv_max_len]` |

| `atten_mask` | `[1, 1, q_seq_len, kv_max_len]` fp16 | 因果 mask（上三角 -inf） |

| `block_table` | `[1, max_blk]` int32 | 分页 block 表 |

| `pse_shift` / `all_seq_lengths_q` | int64 / list | 导出时传 `allQLen`（PSE 位置或全部 Q 长度） |

| quant 参数 | 可选 | `dequant_scale1/quant_scale1/ dequant_scale2/quant_scale2/quant_offset2/ antiquant_scale/antiquant_offset`（KV/输出量化） |

| attrs | - | `num_heads=16`、`num_key_value_heads=2`、`block_size=32`、`scale_value=1/sqrt(256)`、`input_layout='BNSD'`、`inner_precise=2` |

输出：

| 输出 | Shape | 说明 |

|---|---|---|

| `attention_out` | 同 query（int8 时 → fp16） | 注意力输出 `softmax(QKᵀ/√d + mask)V`，NZ 布局 |

**功能**：昇腾融合分页注意力（类 PagedAttention / FlashAttention）：

- 一次算子完成 `QKᵀ → scale → +mask → softmax → ×V`，支持 GQA（`num_heads > num_key_value_heads` 自动展开）、block-table 分页 cache、`inner_precise` 精度档位；

- 支持**量化 KV cache**（dequant/quant/antiquant 一组 param），为 W4A8 KV cache 预留；

- 是全注意力路径（每 4 层 1 个）的计算核心，NZ/Cpro 用它替代请 `transform_nz_2_nd` 后再 `o_proj`。

> 注意 converter 中 `all_seq_lengths_q` 与 `pse_shift` 的传参是工程约定（`'all_seq_lengths_q': pse_shift`），导出/运行模式二选一。

---

### 4.4 ChunkGatedDeltaRule（线性注意力：分块 Gated Delta Rule）

- **PyTorch 侧接口**：`torch_npu.npu_chunk_gated_delta_rule(query, key, value, g, beta, chunk_size, initial_state, effective_length, output_final_state, use_qk_l2norm_in_kernel)`；

- **GE 算子名**：`ChunkGatedDeltaRule`

- **注册位置**：`export_glm_model_qwen3_5_moe.py:151-182`

**输入 / 输出格式**（`Qwen3_5MoeGatedDeltaNet.forward` 实际调用，三个变体一致）：

| 参数 | Shape / 类型 | 说明 |

|---|---|---|

| `query` | `[B, S, 32, 128]` fp16 | Q，已 repeat 到 32 个 value 头（`num_v_heads=32`） |

| `key` | `[B, S, 32, 128]` fp16 | K |

| `value` | `[B, S, 32, 128]` fp16 | V（`.contiguous()`） |

| `g` | `[B, S, 32]` fp16 | 衰减 `g = -exp(A_log) * softplus(a + dt_bias)` |

| `beta` | `[B, S, 32]` fp16 | 记忆门 `beta = sigmoid(b)` |

| `initial_state` | `[1, 32, 128, 128]` fp16 缓存 / NZ、ND 调用前转 fp32 | recurrent_state（线性注意力循环状态；Cpro 直接传 fp16） |

| `effective_length` | `[1]` int16 | 有效长度（padding 支持） |

| attrs | - | `chunk_size=64`（prefill）/ `1`（decode，seq_len==1 时）；`output_final_state`；`use_qk_l2norm_in_kernel=1` |

输出（2 个）：

| 输出 | Shape | 说明 |

|---|---|---|

| `core_attn` | `[B, S, 32, 128]` | 线性注意力输出（delta-rule 算法） |

| `last_recurrent_state` | `[1, 32, 128, 128]` fp32 | 更新后的循环状态（写回 cache） |

**功能**：Gated DeltaNet 线性注意力的分块（chunked）delta rule 核心计算：

- 块内（intra-chunk）：用类似线性 attention 的矩阵运算 + decay mask 计算；

- 块间（inter-chunk）：通过 `recurrent_state` 传递上块累积状态；

- 支持 `use_qk_l2norm_in_kernel`（Q/K 进入 kernel 前做 L2 归一化）、`effective_length`（变长/截断）、`output_final_state`（是否回传状态）。

- prefill 用 `chunk_size=64`（一次处理多个 token），decode 用 `chunk_size=1`（逐 token 递归）；结果经 `Qwen3_5MoeRMSNormGated`（×silu(z)）后 `out_proj` 输出。

- 逻辑上等价于浮点参考实现 `torch_chunk_gated_delta_rule` / `torch_recurrent_gated_delta_rule`（`modeling_qwen3_5_moe.py:245/326`）。

---

### 4.5 LinearAttentionConvstateUpdate（线性注意力 Conv 状态更新，Cpro 专用）

- **PyTorch 侧接口**：`torch.ops.my_ops.linear_attention_convstate_update(x1, x2, x3) -> (y1, y2)`

  （用 `torch.library.define("my_ops::linear_attention_convstate_update", "(Tensor x1, Tensor x2, Tensor x3) -> (Tensor, Tensor)")` 自定义命名空间）；

- **GE 算子名**：`LinearAttentionConvstateUpdate`

- **注册位置**：`export_glm_model_qwen3_5_moe.py:185-218`；调用点 `modeling_qwen3_5_moe_hiai_cpro.py:644`

**输入 / 输出格式**：

| 参数 | Shape / 类型 | 说明 |

|---|---|---|

| `x1` = `mixed_qkv` | `[B, S, conv_dim]` fp16 | 未 transpose 的 QKV 投影输出；`conv_dim = key_dim*2 + value_dim = 2048+2048+4096 = 8192` |

| `x2` = `conv_state` | `[1, conv_dim, kernel-1]` = `[1, 8192, 3]` fp16 | causal conv 循环状态（kernel_size=4） |

| `x3` = `linear_update_tensor` | `[128]` int32 | 占位/更新控制张量（导出时传 `zeros([128], int32)`） |

输出（2 个）：

| 输出 | Shape | 说明 |

|---|---|---|

| `y1` = `new_conv_state` | `[1, conv_dim, kernel_size=4]` fp16 | 更新后的卷积状态 |

| `y2` = `padded` | `[1, conv_dim, seq_len + (kernel-1)]` fp16 | 拼接后的输入（state ‖ x），供后续 `F.conv1d` 消费 |

**功能**：把当前 step 的 QKV 写入 causal 卷积的循环状态，并返回与历史状态拼接的张量 —— 相当于把原来 PyTorch 里的「`cat([conv_state, x])` + 状态截断更新」组合收敛成一个自定义算子，便于整体编译进图。Cpro 变体随后用 `F.conv1d(padded, conv1d.weight, groups=8192)` + `F.silu` 完成 causal 卷积。

---

### 4.6 GE 端口信息速查（`custom_op` 注册口径）

> 上面各节是 **PyTorch 侧（建模代码调用）的 I/O**。真正落到 GE 图（.air → atc → .om）的算子端口，由 `export_glm_model_qwen3_5_moe.py` 中的 `torchair.ge.custom_op('XXX', inputs={...}, attrs={...}, outputs=[...])` 决定。阅读要点：

> - **输入端口**：`inputs` 字典的**键名即 GE 端口名**，**写入顺序即 GE 端口索引（0,1,2,…）**。昇腾侧 C++ 算子的 op-proto（`GetInputDesc` / 维推 `InferShapeAndType`）按同名端口取数，名字必须与算子定义一致（驼峰命名如 `targetBlock`、`offsetInBlock`）。

> - **属性**：`attrs` 用 `torchair.ge.attr.Int/Float/Str/...` 显式标类型，是编译期常量，不进数据端口。

> - **输出端口**：`outputs` 列表即 GE 输出端口（索引 0,1,…）。

> - 表格中「**实际使用**」列表示本模型真实填充的端口；标 None 的可选端口在图中不连实数据，但**端口序号仍与算子定义对齐**（C++ 侧按索引/名字取数，勿在定义端增删序号靠前的端口）。

**① CacheUpdate**（`export_glm_model_qwen3_5_moe.py:60-66`）

| 方向 | GE 端口名 | 索引 | GE 数据 / 形状要点 | 对应 PyTorch 实参 | 实际使用 |

|---|---|---|---|---|---|

| 输入 | `x` | 0 | Tensor fp16 `[max_blk, expand_len, block_size, 16]` | `self`（KV cache） | ✓ |

| 输入 | `updates` | 1 | Tensor fp16 `[s, expand_len, 16]` | `updates`（新 K/V） | ✓ |

| 输入 | `targetBlock` | 2 | Tensor int32 `[num_blk]` | `target_block` | ✓ |

| 输入 | `offsetInBlock` | 3 | Tensor int32 标量 | `offset_in_block` | ✓ |

| 输出 | `x` | 0 | 同输入 `x`（就地语义，单输出） | 结果写回 cache | ✓ |

- 属性：**无**。输出只有一个 `x`，就地更新的结果经该端口回写。

**② AdnRmsNorm**（`:82-89`）

| 方向 | GE 端口名 | 索引 | GE 数据 / 形状要点 | 对应 PyTorch 实参 | 实际使用 |

|---|---|---|---|---|---|

| 输入 | `self` | 0 | Tensor fp16 `[*, hidden]` | `hidden_states`（`[B,S,2048]` / `[B,head,S,256]`） | ✓ |

| 输入 | `gamma` | 1 | Tensor fp16 `[hidden]` | `self.weight`（折叠后 `1+w_HF`） | ✓ |

| 属性 | `epsilon` | - | attr.Float = `1e-6` | `layer_norm_epsilon` | ✓ |

| 输出 | `y` | 0 | 同 `self` fp16 | 归一化结果 | ✓ |

| 输出 | `rstd` | 1 | 前 `dim-gamma.dim()` 维同 x、尾 `gamma.dim()` 维=1，fp32 | 逆标准差（ND 弃用，取 `[0]`） | ✓ |

**③ AdnFusedInferAttention**（`:107-149`，端口最多的一个）

| 方向 | GE 端口名 | 索引 | GE 数据 / 形状要点 | 对应 PyTorch 实参 | 实际使用 |

|---|---|---|---|---|---|

| 输入 | `query` | 0 | Tensor fp16，NZ `[1, num_heads*head_dim/16, S, 16]` | `q2` | ✓ |

| 输入 | `key` | 1 | List[Tensor] fp16，NZ 块 `[1, expand_len, block_size, 16]` | `[k2]` | ✓ |

| 输入 | `value` | 2 | List[Tensor] fp16，同上 | `[v2]` | ✓ |

| 输入 | `pse_shift` | 3 | 可选（位置编码/移位） | `None` 或导出时传 `allQLen` | 导出分支 ✓ |

| 输入 | `atten_mask` | 4 | Tensor fp16 `[1,1,S,kv]`（因果 -inf） | `attention_mask` | ✓ |

| 输入 | `actual_seq_lengths_q` | 5 | Tensor int64 `[S]`（converter 内 `dtype_promote`） | `actual_seq_len_q` | ✓ |

| 输入 | `actual_seq_lengths_kv` | 6 | Tensor int64 `[kv]` | `actual_seq_len_kv` | ✓ |

| 输入 | `dequant_scale1` | 7 | 可选 fp16，Q 侧反量化 scale | - | ✗ |

| 输入 | `quant_scale1` | 8 | 可选，Q 侧量化 scale | - | ✗ |

| 输入 | `dequant_scale2` | 9 | 可选，K/V 侧反量化 scale | - | ✗ |

| 输入 | `quant_scale2` | 10 | 可选，K/V 侧量化 scale | - | ✗ |

| 输入 | `quant_offset2` | 11 | 可选 | - | ✗ |

| 输入 | `antiquant_scale` | 12 | 可选（antiquant 反量化） | - | ✗ |

| 输入 | `antiquant_offset` | 13 | 可选 | - | ✗ |

| 输入 | `block_table` | 14 | Tensor int32 `[1, max_blk]` 分页表 | `self.block_table` | ✓ |

| 输入 | `kv_padding_size` | 15 | 可选 | - | ✗ |

| 输入 | `all_seq_lengths_q` | 16 | Tensor（本模型复用 `pse_shift` 实参，见 `:137`） | `allQLen` | ✓ |

| 输入 | `actual_seq_lengths_q_back` | 17 | 可选（反向推断用） | - | ✗ |

| 属性 | `num_heads` | - | attr.Int = 16 | `self.num_heads` | - |

| 属性 | `scale_value` | - | attr.Float = `256**-0.5` | `self.scaling` | - |

| 属性 | `input_layout` | - | attr.Str = `'BNSD'` | `input_layout` | - |

| 属性 | `num_key_value_heads` | - | attr.Int = 2 | `self.num_key_value_heads` | - |

| 属性 | `block_size` | - | attr.Int = 32 | `block_size` | - |

| 属性 | `inner_precise` | - | attr.Int = 2 | `inner_precise` | - |

| 输出 | `attention_out` | 0 | Tensor fp16 同 Q 布局 | 注意力结果 | ✓ |

> 注意：导出分支（`export_flag`）把 `pse_shift` 接到 GE 端口 **3**；非导出调用则走端口 **16**（`all_seq_lengths_q`）。两个端口位置不同，C++ 侧若只实现了其中一个，需确认转换器走的是哪条。

**④ ChunkGatedDeltaRule**（`:151-182`，含 meta 与 converter）

| 方向 | GE 端口名 | 索引 | GE 数据 / 形状要点 | 对应 PyTorch 实参 | 实际使用 |

|---|---|---|---|---|---|

| 输入 | `query` | 0 | Tensor fp16 `[B,S,32,128]`（Q 已 repeat 到 v_heads） | `query` | ✓ |

| 输入 | `key` | 1 | Tensor fp16 `[B,S,32,128]` | `key` | ✓ |

| 输入 | `value` | 2 | Tensor fp16 `[B,S,32,128]` | `value` | ✓ |

| 输入 | `g` | 3 | Tensor fp16 `[B,S,32]`（衰减） | `g` | ✓ |

| 输入 | `beta` | 4 | Tensor fp16 `[B,S,32]`（记忆门） | `beta` | ✓ |

| 输入 | `initial_state` | 5 | Tensor fp32 `[1,32,128,128]`（Cpro 传 fp16） | `initial_state` | ✓ |

| 输入 | `effective_length` | 6 | Tensor int16 `[1]` | `effective_length` | ✓ |

| 属性 | `chunk_size` | - | attr.Int = 64（prefill）/ 1（decode） | `chunk_size` | - |

| 属性 | `output_final_state` | - | attr.Int 0/1（PyTorch bool → GE Int，Cpro 侧按 int 传） | `output_final_state` | - |

| 属性 | `use_qk_l2norm_in_kernel` | - | attr.Int = 1（Q/K 进 kernel 前 L2 归一化） | `use_qk_l2norm_in_kernel` | - |

| 输出 | `core_attn` | 0 | fp16 `[B,S,32,128]` | `core_attn_out` | ✓ |

| 输出 | `last_recurrent_state` | 1 | fp32 `[1,32,128,128]` | 写回 recurrent cache | ✓ |

**⑤ LinearAttentionConvstateUpdate**（`:208-218`）

| 方向 | GE 端口名 | 索引 | GE 数据 / 形状要点 | 对应 PyTorch 实参 | 实际使用 |

|---|---|---|---|---|---|

| 输入 | `x1` | 0 | Tensor fp16 `[1, S, 8192]`（QKV 拼接） | `mixed_qkv` | ✓ |

| 输入 | `x2` | 1 | Tensor fp16 `[1, 8192, 3]` | `conv_state` | ✓ |

| 输入 | `x3` | 2 | Tensor int32 `[128]`（占位，导出传 `zeros`） | `linear_update_tensor` | ✓ |

| 输出 | `y1` | 0 | fp16 `[1, 8192, 4]`（新卷积状态） | `new_conv_state` | ✓ |

| 输出 | `y2` | 1 | fp16 `[1, 8192, S+3]`（历史‖当前拼接） | `padded` | ✓ |

- 属性：**无**。属 `torch.library` 自命名空间 `my_ops::`（非 `npu::`），注册方式与其余 4 个不同。

---

## 5. torch_npu 内置算子详解

> 这些算子由 torchair / torch_npu 自带 converter 支持（无需写 GE converter），主要完成量化 GEMM 与 MoE 路由。

### 5.1 量化相关

#### npu_dynamic_quant（激活 per-token 动态量化）

- 输入：`x` fp16/fp32 `[M, K]`（可多 batch）

- 输出：`(x_quant int8 [M, K], pertoken_scale fp32)`，scale 为每个 token 一个（`pertoken_scale.reshape(-1,1)` 后 `[M,1]`）

- 功能：求 `absmax/M` → `round(x/scale)`，对称 per-token 量化。用于每次量化 GEMM 前的激活量化（QLinear 与 MoE grouped matmul 的 W4A8 都依赖它）；`quant.py` 中也用它量化 Embedding。

#### npu_quant_matmul（A8W4 量化矩阵乘，QLinear）

- 输入：

  - `x_quant`：int8 `[M, K]`（激活，动态量化）

  - `W_q`：int32 `[K, N//8]`（**I4ZN 打包 int4 权重**，8 个 4bit 存一个 int32）

  - `scale`：fp32 分组 scale（装载时 `[N, K//group, 1]` → `squeeze(2).t()` 后为 **`[K//group, N]`**，见 `replace_linear_to_QLinear`）

  - `pertoken_scale`：fp32 `[M,1]`

  - `output_dtype=torch.float16`

- 输出：`[M, N]` fp16

- 功能：A8W4 量化 GEMM（int8 激活 × int4 权重），在线性层 `QLinear` 中使用，反量化在算子内融合：`out = x_quant × W_quant_packed × (group_scale ⊙ per_token_scale)`。已量化替换的层：`self_attn.q/k/v/o_proj`、`mlp.gate`（router）、`mlp.shared_expert.*`、`linear_attn.in_proj*/out_proj`、`lm_head`。

#### npu_trans_quant_param（scale → int64 QPARAM）

- 输入：fp32 分组 scale；输出：int64 整型量化参数（QPARAM 格式）

- 用于专家权重 `w13_scale/w2_scale` 装载时（`utils.py` 用 `npu_trans_quant_param(scale, None, 1)`），把 float scale 编码成硬件 int64 参数供 `npu_grouped_matmul` 使用。

---

### 5.2 MoE 相关

#### npu_moe_gating_top_k_softmax（Cpro 路由，融合 topk+softmax）

- 输入：`router_logits [M, num_experts=256]`、`bias=None`、`k=num_experts_per_tok=8`

- 输出：`(routing_weights [M, k] fp16, selected_experts [M, k] int32, row_idx int32)`

- 功能：替代 NZ/ND 的 `F.softmax + torch.topk` 两步，融合为单算子。

#### npu_moe_init_routing_v2（MoE 路由初始化 / 数据重排）

- 输入：

  - `hidden_states`：`[B*S, hidden]` fp16（展平 token）

  - `expert_idx`：`[B*S, k]` int32（topk 选中的专家 id）

  - `active_num`：`B*S*k`；`expert_num=256`；`expert_tokens_num_flag/type`；`active_expert_range=[0,256]`；`quant_mode=-1`

- 输出：

  - `expanded_x`：`[num_active, hidden]` —— 按专家分桶重排后的 token（每个 token 出现 topk 次）

  - `expanded_row_idx`：`[num_active]` int32 —— 源行索引（finalize 时还原）

  - `tokens_per_expert`：`[num_experts]` int64 —— 每个专家的 token 数（作为 `group_list` 给 grouped_matmul）

  - `pertoken_scale`：动态量化 scale

- 功能：MoE 路由的数据搬运（gather/排序），把「token-专家」对应关系转换为 grouped matmul 需要的分组布局。

- 注意（代码注释）：310P 上异步下发会先落到 V3 无二进制，`synchronize()` 重试退回 V2 才可用——不能用 `ASCEND_LAUNCH_BLOCKING=1`。

#### npu_grouped_matmul（专家分组 GEMM）

- 输入：

  - `[x_quant]`：int8 `[num_active, K]`（动态量化）或 fp16 权重版

  - `[w]`：专家权重，量化版为 **int32 打包 int4**（`[E, K, N//8]` I4ZN），未量化版 fp16 `[E, K, N]`

  - `scale`：`[E, K//256, N]` int64（QPARAM，量化版）

  - `per_token_scale`：`[num_active]` fp32

  - `group_list`：`tokens_per_expert int64 [E]`

  - 参数：`split_item=3`（w13 一次算 gate/up 两个投影）、`group_type=0`、`group_list_type`、`act_type=0`、`output_dtype=fp16`

- 输出：`[num_active, N]` fp16（`split_item=3` 时为 `[num_active, 2*inter]`）

- 功能：按专家分组的一次量化 GEMM。第一跳：`w13_weight [E, hidden, 2*inter//8]` → 输出 `[sum_tokens, 2*inter]`（gate+up 融合）；第二跳：`w2_weight [E, inter, hidden//8]` → 输出 hidden。

#### npu_swiglu（SwiGLU 融合）

- 输入：`[M, 2*inter]` fp16 → 输出 `[M, inter]`

- 功能：`silu(x1) * x2`（gate×up），NZ/ND 用它替代 PyTorch 显式 split*silu*multiply（Cpro 用 `F.silu` 手动实现）。

#### npu_moe_finalize_routing（MoE 输出还原）

- 输入：`hidden_states_ordered_by_experts [num_active, hidden]`、`scales=topk_weight [B*S, k]`、`expanded_src_to_dst_row=expanded_row_idx`、`skip1/skip2/bias=None`、`drop_pad_mode=2`

- 输出：`[B*S, hidden]`

- 功能：按 `expanded_row_idx` 把专家输出加权重（topk_weight）回填到原始 token 行，完成 MoE 结果的合并。

---

### 5.3 归一化 / KV 写入

#### npu_rms_norm（NZ/Cpro）

- 输入：`x [*, H] fp16`、`gamma [H] fp16`、`eps`

- 输出：`y`（同 x）

- 功能：RMSNorm。装载时 `weight_NPU = 1 + weight_HF`；**NZ/Cpro 的 forward 里传的是 `1.0 + self.weight`**（对已折叠权重再加 1，等价 `2 + w_HF`，见 §2.2/文末疑点）；去掉 `1.0 + ` 后与 ND 的 `AdnRmsNorm`（直接用 `self.weight`）口径一致。

#### npu_cache_update_（就地 KV 写入）

- 输入/就地：`cache [max_blk, expand_len, block_size, 16]`、`updates [s, expand_len, 16]`、`target_block int32`、`offsetInBlock int32`

- 功能：NZ/Cpro 全注意力 KV 写入；导出时经 Functionalize 变成 `CacheUpdate` 自定义算子（见 4.1）。

#### npu_scatter_nd_update_（ND 变体 KV 写入）

- 输入/就地：`cache`、`indices = new_kv_cache_pos`、`updates`

- 功能：ND 变体（`modeling_qwen3_5_moe_hiai_nd.py:354-355`）用 scatter 方式写 KV cache。

---

## 6. 量化方案分析

### 6.1 量化总流程

```

quant.py（离线，CPU/fp16 原模型）

   ├─ Embedding            → npu_dynamic_quant → int8 weight + fp32 scale（存 .bin）

   ├─ nn.Linear（self_attn/shared_expert/linear_attn/mlp*/lm_head）

   │                        → symmetric_minmax_quantize(bits=4, group_size=256)

   ├─ 3D 专家权重 gate_up_proj/down_proj

   │                        → symmetric_minmax_quantize_3d(bits=4, group_size=256)

   └─ mlp.gate（2D router）→ bits=4, group_size=256

       ↓ 保存：*_quant_weight(int8/int4) + *_quant_scale + index.json（safetensors 分片）

运行时装载（utils.py / export 脚本）

   ├─ replace_linear_to_QLinear → 普通 Linear 替换为 QLinear（A8W4, npu_quant_matmul）

   ├─ combined_experts_weight_quant_3d → 专家 w13/w2 打包 I4ZN int32 + QPARAM scale

   └─ embedding：int8 lookup × fp32 scale（wrapper 中 ×embed_scale）

```

### 6.2 量化范围（“哪些地方量化”）

| 位置 | 模块 | 量化方式 |

|---|---|---|

| Embedding | `embed_tokens` | **int8**（`npu_dynamic_quant`，per-token scale），运行时 `int8 look up × scale`（支持 `mul_twice`） |

| 全注意力投影 | `self_attn.q/k/v/o_proj` | **W4 A8**（QLinear：int8 激活 × int4 权重，`npu_quant_matmul`） |

| MoE 路由 | `mlp.gate`（router，256*2048） | W4 A8（QLinear） |

| MoE 专家 | `mlp.experts`（gate_up_proj/down_proj，256 个 3D 权重） | **W4 A8**（`npu_grouped_matmul`：int4 打包 + int64 QPARAM scale） |

| 共享专家 | `mlp.shared_expert.*` | W4 A8（QLinear） |

| 线性注意力投影 | `linear_attn.in_proj_*/out_proj` | W4 A8（QLinear；注：`replace_linear_to_QLinear` 实际只跳过 `mlp.experts` 与 `shared_expert_gate`） |

| 输出头 | `lm_head` | W4 A8（QLinear） |

| 不量化 | `mlp.shared_expert_gate`（N=1，不满足 ZN 分形 N%16=0） | 保持 fp16 |

| 激活 | 每次量化 GEMM 前 | **per-token 动态 int8**（`npu_dynamic_quant`） |

### 6.3 量化算法细节（“做了什么量化”）

**对称 MinMax 分组量化（权重）—— `symmetric_minmax_quantize` / `_3d`**：

```

max_abs_per_group = |W|.max(沿 in_features 分 group_size=256 组)

scale = max_abs / (2^(bits-1) - 1)

q = clamp(round(W / scale), -2^(bits-1), 2^(bits-1)-1)

zero_point = 0（对称）

```

- bits：**W=4 bit**（int4，存于 int8 容器低 4 位）；激活 **A=8 bit**。

- group_size：**256**（沿 in_features / K 维分组；scale shape `[out, K//256, 1]`）。

- 专家 3D：`[E, out, in]` → reshape 2D 复用同函数，scale `[E, out, in//256]`。

**硬件格式（装载时二次处理）**：

| 张量 | 离线产物 | 装载后（NPU 参数） |

|---|---|---|

| 普通 Linear 权重 | `*_quant_weight [N, K] int8`（int4 值） | `nd_to_i4zn` → int32 `[K, N//8]`（**I4ZN** 分形，8×4bit 打包） |

| 专家权重 | `gate_up_proj [E,2 inter,hidden] int8`、`down_proj [E,hidden,inter] int8` | 转置 → `pack_int4_to_int32_simple` → `nd_to_packed_i4zn` → int32 `[E, hidden, 2 inter//8]` / `[E, inter, hidden//8]` |

| 普通 Linear scale | `*_quant_scale` | `[N, K//256, 1]` → `squeeze(2).t()` → fp32 `[K//group, N]`，给 `npu_quant_matmul.scale` |

| 专家 scale | `[E, out, in//256]` fp32 | transpose → `npu_trans_quant_param` → **int64 QPARAM** `[E, K//256, N]`（给 `npu_grouped_matmul`） |

> `utils.py` 提供多套格式工具：`nd_to_zn_int8`（int8→ZN）、`nd_to_packed_zn` / `nd_to_packed_i4zn` / `nd_to_i4zn`（int4→I4ZN）、`convert_int4_to_qbm_zn`、`pack_int4_to_int32_simple`，对应不同芯片/Roofline 打包需求。

**激活量化**：GEMM 前 `npu_dynamic_quant` per-token 对称 int8；`pertoken_scale` 与权重分组 scale 一起在算子内部反量化，输出 fp16。

**保精度对照路径**：`PseudoQuantLinear`（utils）直接读原始 int8 量化权重做 fp32 反量化 + `torch.matmul`，用于量化精度对比验证（`--pseudo_quant`）。

**fake-quant 验证**：quant.py 量化后立即反量化回填 `module.weight`，用 `model.generate` 离线验证量化方案正确性（见 quant.py 注释掉的 generate 段）。

### 6.4 量化在导出/推理中的落地

- 导出脚本 `quant_mode=enable` 时：`quant_model_inplace` 装载量化权重 + 量化 embedding 以 `int8` 张量进入图；`export_glm_model_qwen3_5_moe.py:314-322`。

- 推理 `quant_model_inplace`（`inference.py:724`）把同一套量化权重装入 NPU 模型，embedding 在 `prefill/decoder` 中按 `quant_embedding_weights[input_ids] × embed_scale` 取数。

- GE 图里量化算子（`npu_dynamic_quant`/`npu_quant_matmul`/`npu_grouped_matmul`/`npu_swiglu`）由 torchair 内置 converter 直接映射，无需自定义。

---

## 附：本工程 5 个自定义算子速查表

| GE 算子 | PyTorch 接口 | 变体 | 输入要点 | 输出 | 功能 |

|---|---|---|---|---|---|

| `CacheUpdate` | `npu_cache_update_` / `npu_cache_update__functional` | NZ/ND/Cpro | cache `[64,32,32,16]` + updates + targetBlock/offset | cache | 分页 KV 写入 |

| `AdnRmsNorm` | `adn_rms_norm` | ND | x `[*,H]`、gamma `[H]`、eps | y、rstd | RMSNorm（零中心 1+w） |

| `AdnFusedInferAttention` | `adn_fused_infer_attention` | NZ/Cpro | q NZ `[1,256,S,16]`、k/v list、block_table、mask、多种 scale | attention_out | 融合分页注意力 |

| `ChunkGatedDeltaRule` | `npu_chunk_gated_delta_rule` | 全部 | q/k/v `[B,S,32,128]`、g/beta `[B,S,32]`、state `[1,32,128,128]` | core_attn、state | 线性注意力（delta rule） |

| `LinearAttentionConvstateUpdate` | `my_ops::linear_attention_convstate_update` | Cpro | qkv `[B,S,8192]`、conv_state `[1,8192,3]`、update_tensor | new_conv_state、padded | causal conv 状态拼接 |

---

## 7. 评审发现的疑点 / 需确认项

> 以下为对照源码逐项核对后，文档中或代码中暴露出的不确定点，建议结合一次真实跑批/精度对齐确认。

1. **主 RMSNorm 双重加 1（NZ/Cpro）→ 已确认（静态代码层面）**

   - 三处装载都折叠：`load_npu_model`（`export_model_wrapper_qwen3_5_moe.py:162`）与 `utils.load_rmsnorm_weight`（`utils.py:544`）都把 `weight_NPU = 1 + weight_HF`；HF 主 RMSNorm 是**零中心**（weight≈0，forward `×(1+w)`，`modeling_qwen3_5_moe.py:824/833`），所以正确目标 gamma 就是 `1 + w_HF`。

   - **ND** `adn_rms_norm(x, self.weight, eps)` 直接用折叠值 → gamma=`1+w` ✓ 正确。

   - **NZ** `npu_rms_norm(..., 1.0 + self.weight.f...)`、**Cpro** `npu_rms_norm(..., self.weight + 1.0, ...)` 又加一次 → gamma=`2+w`（w≈0 时 ≈2）✗。

   - 影响范围不止 2 个主 norm：NZ/Cpro 的 `q_norm`、`k_norm`、`model.norm` 用的是同一个 `Qwen3_5MoeRMSNorm` 类，全部同样双加。

   - 数值影响：每个该 norm 的输出整体 ≈×2。对纯线性路径只是常数缩放（不改变 `argmax` 相对顺序），但会经过 attention softmax / delta rule 等非线性（分布尖锐化）、以及 `npu_quant_matmul` 的 per-token 量化（scale 与校准不一致），可能造成与 HF/ND 的可观测偏差。**建议**：取同一输入，对比 NZ vs ND vs HF 首层 output 或逐层输出，`gamma` 是否为 2 立即可见。

2. **GatedRMSNorm（线性注意力内）被统一折叠 → 新增疑点，静态上站得住但依赖数值确认**

   - HF `Qwen3_5MoeRMSNormGated` 的 weight 初始为 **ones** 且 forward **直接乘 weight 不加 1**（`modeling_qwen3_5_moe.py:189/198`），即过 gated norm 的等效 gamma = `w_g`（通常 O(1)，非零中心），**并不适用零中心的 1+w 折叠**。

   - 但两个装载器对所有后缀为 `norm.weight` 的参数都做 `1+w_g` 折叠 → NPU 侧 `linear_attn.norm.weight = 1 + w_g`。

   - NZ/Cpro/ND 的 gated norm forward 都直接用 `self.weight` → 实际 gamma = `1 + w_g`，若 `w_g≈1` 则 ≈2，与 HF 的 `w_g≈1` 差一倍。

   - 置信度说明：**取决于 checkpoint 中 gated weight 的真实数值**（初始 ones，训练后通常 O(1)，但 scale 类参数可能被后续 `out_proj` 吸收而偏离）。**建议验证**：dump 任一 `linear_attn.norm.weight` 的统计量（mean/std），或把 1 层 linear_attn 输出与 HF 参考逐元素对比；若该层 gamma 明显≈2 则同样应去掉折叠（改为直接装载 `w_g`）。

3. **ND 的 `npu_cache_update_` 与 `npu_scatter_nd_update_` 并存 → 确认（遗留代码，非当前生效路径）**

   - ND attention 定义了基于 `npu_cache_update_` 的 `update()`（`modeling_qwen3_5_moe_hiai_nd.py:301-316`），但 `forward` 实际用 `npu_scatter_nd_update_`（354-355）+ 手写 eager attention（`if True:`），`adn_fused_infer_attention` 分支是死代码。文档按“实际生效路径”描述，若后续要启用 fused 分支需注意这一点。

4. **导出脚本 block_size 参数不一致 → 确认（当前非活路径）**

   - `create_past_key_values` 的 `block_size` 由 `32 if segment=='cpro' else 128` 决定（`export_glm_model_qwen3_5_moe.py:330`），而 modeling/inference 内全为 `block_size=32`。当前主流程只导出 `segment='cpro'`（block_size=32，一致）；若启用被注释的 NZ/second 段导出，缓存 shape 会变成 `[max_blk=16, 32, 128, 16]`，与模型中 `block_size=32` 不符，需注意。

5. **`npu_quant_matmul` 的 scale 方向 → 确认（非问题）**

   - 装载时 `scale = qs.squeeze(2).t()` → `[K//group, N]`。与 `symmetric_minmax_quantize` 沿 in_features 分组的方向一致，属正常。

6. **router（`mlp.gate`）量化冗余 → 更正：无冗余（此前为误判）**

   - `quant.py` 的量化是 `if/elif` 链：HF 的 `mlp.gate` 是 `Qwen3_5MoeTopKRouter`（`nn.Module`，`weight` 为 2D `nn.Parameter`，非 `nn.Linear`，`modeling_qwen3_5_moe.py:785/801`），因此不命中 `isinstance(module, nn.Linear)` 分支，只命中专门的 2D `mlp.gate` 分支，每个权重只量化/保存一次。

   - NZ/Cpro 的 NPU 侧 `mlp.gate` 是 `nn.Linear`，`replace_linear_to_QLinear` 会替换为 QLinear；两侧 key（`model.layers.X.mlp.gate_quant_weight`）对应一致。

   - 保留提醒：NZ 路由目前是 `softmax+topk`（导出一条 fp16 门控矩阵乘路径），Cpro 用 `npu_moe_gating_top_k_softmax`；两套路由裁剪策略不同，改路由实现时需重新对齐权重量化格式。
