# L0 GatedDeltaNet 逐算子精度对比

测试边界是原模型 `layers[0].linear_attn.forward`。入口是 decoder 的 `input_layernorm` 返回给 GDN 的 `hidden_states`、进入该层的 `past_key_values` 和 `effective_length`。出口是 `output = self.out_proj(core_attn_out)`、卷积历史和递推状态。decoder 的第一次残差相加位于出口之后。

原生 310P 在同一次 T64 prefill 前向里记录入口、算子中间值和返回值。CPU 和 CPro 都从这份入口出发。四个投影读取同一份 hidden，之后每条分支都读取本端前序输出，直到汇合并产生 `output`。所以矩阵显示的是**传播到该点的累计误差**。沿同一依赖分支看误差变化，例如先看 `qkv_a8`、`qkv_scale`，再看 `qkv`、`conv`、`conv_silu`。

## 运行

沿用单层 linear 测试的环境变量。在 78 的代码仓库运行：

```bash
git pull --ff-only
export L0_SPLIT_RUN="${AI_RUN_DIR:?}/target/tests/l0-split-silupatch-t64-valid19-02"
bash targets/cpro/scripts/gdn_layer_t64.sh prepare
```

`prepare` 使用 `TARGET_SOURCE` 下的原版 310P NZ 驱动，以及 `NZ_CONFIG`（默认 `config/qwen3_5_moe.yaml`）。要求原有修正版本的 `$L0_SPLIT_RUN/air/manifest.json`。它检查 W4 权重、逐 token scale 的 FP16 转换、物理卷积历史、chunk 长度和 GDR 状态接口，并绑定两个原生 shape 的 `SiluPatch`。CPU 复用项目的 A8 ties-to-even / W4 group256 数学参考，组内整数点积、FP32 缩放和累加，最后转 FP16。五次投影各自核对原始 checkpoint 与本次 310P 调用实际读取的打包权重和组 scale。

在 53 的代码仓库运行：

```bash
git pull --ff-only
export L0_SPLIT_RUN="${AI_RUN_DIR:?}/target/tests/l0-split-silupatch-t64-valid19-02"
bash targets/cpro/scripts/gdn_layer_t64.sh board
bash targets/cpro/scripts/gdn_layer_t64.sh compare --tensor qkv --metric relative_l2
```

`board` 使用 53 已配置的 DDK、HDC 与板端 V2 runner，按需拉取 78 的参考包。两台服务器默认使用同名的 `$AI_RUN_DIR/target/tests/l0-gdn-ops-t64-valid19-01`，可以在两边设置 `GDN_LAYER_RUN` 改路径。再次运行 `board` 会复用输入、模型、设备和输出哈希均匹配的已完成阶段。中断后也可先运行 `compare` 查看已完成项。一个选定张量的 3×3 矩阵打印到终端，全部矩阵保存在 `board/matrices.csv`，原始指标和缺失阶段在 `board/comparison.json`。每个 `board/<阶段>/stage.json` 保存输入来源哈希和已取回的输出记录。

## OMC、代码变量与统计范围

合计 **24 个 OMC，33 个对比张量**。下面按源码中可观测的计算分段。reshape、切片、转置和 dtype 转换随相关阶段执行。GDR 的两个输出来自同一次融合算子调用。运行 `bash targets/cpro/scripts/gdn_layer_t64.sh plan` 可列出所有阶段的输入、输出和逻辑 shape。实际导出 AIR 的节点名和端口保存在参考包 `manifest.json` 的 `stages` 中。

| OMC 数量 | 报告字段 | 源码变量与含义 |
|---:|---|---|
| 8 | 对 `p=qkv,z,a,b` 各有 `p_a8`、`p_scale`、`p` | 每个 `self.in_proj_p(hidden_states)` 分为 DynamicQuant 和 W4 matmul。前两项是 `x_quant` 与转换前的 `pertoken_scale`。matmul 的四个输出分别对应 `mixed_qkv`、`z`、`a`、`b`。 |
| 1 | `window`、`new_history` | `padded` 是卷积使用的 67 个物理位置。310P 从本次 `F.conv1d` 实际输入取对应窗口。报告对齐窗口中本次 token 0–18。`new_conv_state` 是此次写回的四行卷积历史，比较完整 `[1,8192,4]`，对应当前 QKV 的物理行 60–63。 |
| 1 | `conv` | `conv_out`，逐通道因果卷积输出，SiLU 之前。 |
| 1 | `conv_silu` | `F.silu(conv_out)` 的激活，对应 `qkv_conv` 中的 SiLU 结果。CPro 调用独立的 FP16 入、FP32 出自定义 OMC；传给 Q/K/V 时执行模型内原有的回转。 |
| 3 | `query`、`key`、`value` | 卷积激活拆分、整理 heads 后送入 GDR 的三个实参。Q 和 K 已重复到 32 个 value heads。 |
| 3 | `biased_a`、`softplus`、`g` | 依次是 `a.float() + self.dt_bias`、`F.softplus(...)`、`-self.A_log.float().exp() * softplus`。权重侧的指数和负号在计算 `g` 的阶段内。 |
| 1 | `beta` | `beta = b.sigmoid()`，控制每个 head 的 delta 更新幅度。 |
| 1 | `core`、`final_state`、`recurrent_cache` | `npu_chunk_gated_delta_rule` 返回的 `core_attn_out` 和原始 `last_recurrent_state`，以及层实际返回的 FP16 `past_key_values[1]`。状态比较整个 `[1,32,128,128]`。CPro 的 GDR 原始状态已是 FP16；脚本校验其至公开缓存的路径。 |
| 1 | `rms` | `self.norm.forward` 中的 `hidden_states = npu_rms_norm(...)[0]`，已乘归一化权重，尚未乘门值。 |
| 1 | `gate_silu` | `self.norm.forward` 中的 `F.silu(gate.to(torch.float32))`，其中 `gate` 来自 z 投影。CPro 使用第二个独立 SiluPatch OMC。 |
| 1 | `product`、`gated` | `out = hidden_states * SiLU(gate)` 的 FP32 乘积，以及 `return out.to(input_dtype)` 整理成 `out_proj` 输入后的值。 |
| 2 | `out_a8`、`out_scale`、`out` | 最后一次 DynamicQuant 和 W4 matmul。`out` 对应整个 GDN 返回的 `output`，每个 token 有 2048 个分量。 |

激活一律按原张量的时间轴对齐前 19 个 token，卷积张量的时间轴在末维。浮点矩阵给出 cosine、relative L2、KL、RMSE、平均相对误差、最大和平均绝对误差，以及逐元素超差比例。行是实际值，列是参考值，例如 CPro 行 / 310P 列的 relative L2 分母是 310P 张量的 L2 范数。浮点 `passed` 沿用项目五指标阈值，逐元素超差使用 `atol=1e-5, rtol=1e-3`。INT8 的 `error_ratio` 是不同编码的元素占比，`max_abs` 是最大编码差，按完全一致判定。板端执行完成和数值是否过阈值分别记录。
