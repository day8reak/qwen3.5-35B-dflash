# L0 逐步三端对比变量表

适用于 `decoder_compare.sh l0-steps`，使用四层 T64 prefill 的实际输入比较 CPU、310P 和 CPro。L0 是第一层 linear attention。它向下一层传递 `block_out`，向后续推理调用保存 `new_history` 和 `recurrent_cache`。四层模型中的 RoPE 位于 L3 full attention。

下表给出比较时的逻辑形状，省略大小为 1 的 batch 维。T64 表示 64 个物理位置，当前 prompt 有 19 个有效 token。常规激活按位置 0–18 计算指标。三个后端分别从各自的上游结果继续计算，因此某个观测点的误差包含上游已经产生的差异。

## 入口与四路投影 · entry-projections

同一投影有三个观测点。`*_a8` 是 DynamicQuant 对该投影**输入**生成的 INT8 激活，`*_scale` 是每个 token 的浮点尺度，无后缀变量是使用这些激活和 W4 权重得到的投影输出。CPU 的对称量化参考对每行计算 `scale = max(abs(x)) / 127`，零行的 scale 为 0。下表中的尺度都属于激活。

| 报告变量 | 直接上游与含义 | T64 逻辑形状 |
|---|---|---|
| `block_entry` | L0 收到的 hidden，同时送入 input RMSNorm 和第一条残差。CPU、310P 使用实测的层入口作为共同起点，CPro 从四层包的 embedding 和 scale 算到这里。 | 64×2048 |
| `hidden` | `input_layernorm(block_entry)`，QKV、z、a、b 四个投影共同读取它。 | 64×2048 |
| `qkv_a8`、`qkv_scale`、`qkv` | `in_proj_qkv(hidden)` 的输入编码、输入尺度、混合 QKV 投影结果。输出沿通道分为 Q 2048、K 2048、V 4096。 | 64×2048、64、64×8192 |
| `z_a8`、`z_scale`、`z` | `in_proj_z(hidden)` 的输入编码、输入尺度、门控投影。后面的 `gate_silu` 从这里取值。 | 64×2048、64、64×4096 |
| `a_a8`、`a_scale`、`a` | `in_proj_a(hidden)` 的输入编码、输入尺度、衰减分支的原始值。每个 token 每个 value head 一个值。 | 64×2048、64、64×32 |
| `b_a8`、`b_scale`、`b` | `in_proj_b(hidden)` 的输入编码、输入尺度、更新分支的原始值。后面取 sigmoid 得到 `beta`。 | 64×2048、64、64×32 |

## 卷积、历史与 Q/K/V · convolution

| 报告变量 | 直接上游与含义 | T64 逻辑形状 |
|---|---|---|
| `window` | 沿时间拼接旧卷积历史和本轮 `qkv`，取末 67 个位置作为宽度 4 的因果卷积输入。开头 3 个是历史位置，当前报告的有效 token 从窗口下标 3 开始取。 | 8192 通道 × 67 位置 |
| `new_history` | 上述拼接结果的末 4 个位置，对应本轮 **卷积前** `qkv` 的物理位置 60–63。它对应四层报告中的 `conv_state_0`。 | 8192×4，整份缓存 |
| `conv` | 每个通道独立对 `window` 做因果卷积；观测点是卷积 SiLU 的输入。 | 8192×64 |
| `conv_silu` | `SiLU(conv)`，当前 CPro 图用 FP16 SiluPatch。Q/K/V 从这一结果拆分。 | 8192×64 |
| `query` | 取卷积激活的 Q 通道，按 head 整理并展开到 32 个 value head，作为 GDR 的查询。 | 64×32×128 |
| `key` | 取卷积激活的 K 通道，按同样的 head 顺序展开，作为 GDR 的键。 | 64×32×128 |
| `value` | 取卷积激活的 V 通道，整理为 32 个 value head，作为写入递推状态的值。 | 64×32×128 |

## 衰减、更新与递推 · gdr

| 报告变量 | 直接上游与含义 | T64 逻辑形状 |
|---|---|---|
| `biased_a` | `a.float() + dt_bias` 的加法结果。CPro 从该结果转换到 SoftplusPatch 的 FP16 输入端口。 | 64×32 |
| `softplus` | `softplus(biased_a)`，得到正的步长。CPro 在 SoftplusPatch 输出端取值。 | 64×32 |
| `g` | `-exp(A_log) × softplus`，送入 GDR 的对数衰减系数。 | 64×32 |
| `beta` | `sigmoid(b)`，控制每个 token 对递推状态的更新强度。 | 64×32 |
| `core` | GDR 读取 `query/key/value/g/beta`、初始状态和有效长度后产生的逐 token 输出，供后面的 RMSNorm 使用。 | 64×32×128 |
| `final_state` | 同一次 GDR 的第二个输出，表示处理完 19 个有效 token 时的递推状态。 | 32×128×128，整份状态 |
| `recurrent_cache` | 从 `final_state` 写回、供下次调用读取的缓存值。CPU 参考在写回时舍入到 FP16。它对应四层报告中的 `recurrent_state_0`。 | 32×128×128，整份缓存 |

## 门控、输出投影与第一条残差 · attention-output

| 报告变量 | 直接上游与含义 | T64 逻辑形状 |
|---|---|---|
| `rms` | `core` 在每个 head 的 128 维上做 RMSNorm，再乘该层的 gamma。它是门控乘法的一侧。 | 64×32×128 |
| `gate_silu` | 对 `z` 做 SiLU，构成门控乘法的另一侧。当前 CPro 的 SiluPatch 输入、输出均为 FP16，使用 64×4096 视图。 | 64×32×128 |
| `product` | `rms × gate_silu` 的逐元素乘积。当前 CPro 在两个分支汇合处做 FP32 乘法。 | 64×32×128 |
| `gated` | 将 `product` 转回 FP16，并合并 32 个 head，形成 `out_proj` 的浮点输入。 | 64×4096 |
| `out_a8`、`out_scale`、`out` | `out_proj(gated)` 的输入编码、输入尺度、GDN 最终输出。查 `out_scale[t]` 时，从同一后端的 `gated[t, :]` 求行绝对最大值再除以 127。 | 64×4096、64、64×2048 |
| `residual` | 第一条残差相加，`block_entry + out`。 | 64×2048 |
| `post_norm` | `post_attention_layernorm(residual)`，进入 MoE 的 hidden。共享专家和路由器都读取它。 | 64×2048 |

## 共享专家与路由器 · shared-router

| 报告变量 | 直接上游与含义 | T64 逻辑形状 |
|---|---|---|
| `shared_gate_proj_a8`、`shared_gate_proj_scale`、`shared_gate_proj` | 共享专家从 `post_norm` 计算 gate 投影。依次是输入编码、输入尺度和 512 维 gate 输出。 | 64×2048、64、64×512 |
| `shared_up_a8`、`shared_up_scale`、`shared_up` | 共享专家从 `post_norm` 计算平行的 up 投影。依次是输入编码、输入尺度和 512 维 up 输出。 | 64×2048、64、64×512 |
| `shared_silu` | `SiLU(shared_gate_proj)`。 | 64×512 |
| `shared_product` | `shared_silu × shared_up`，是共享专家 down 投影的输入。 | 64×512 |
| `shared_down_a8`、`shared_down_scale`、`shared_down` | 对 `shared_product` 量化并做 down 投影，得到共享专家的 2048 维输出。 | 64×512、64、64×2048 |
| `shared_gate_logit` | `shared_expert_gate(post_norm)`，每个 token 一个标量，决定整个共享专家输出占多大权重。 | 64×1 |
| `shared_weighted` | `shared_down × sigmoid(shared_gate_logit)`，共享专家对 MoE 的最终贡献。 | 64×2048 |
| `router_logits_a8`、`router_logits_scale`、`router_logits` | 路由器从 `post_norm` 计算每个 token 对全部 256 个路由专家的打分。依次是输入编码、输入尺度和专家分数。 | 64×2048、64、64×256 |
| `expert_ids` | 每个 token 实际选中的 Top‑K 专家编号。三端比较时展开成 256 列的 0/1 集合。报告还给出 CPro 与 310P 集合的 Jaccard，即交集大小除以并集大小。 | 原始 64×K；比较 19×256 |

## 路由专家、汇总与层输出 · experts-output

| 报告变量 | 直接上游与含义 | 比较单位与宽度 |
|---|---|---|
| `route_map` | 路由分发产生的行索引。脚本用它把分组矩阵乘的物理行还原到对应的 token 和专家。这个辅助变量不单列精度矩阵。 | 每个 token–专家分配一项 |
| `route_weights` | 实际送入路由汇总算子的 Top‑K 权重，按专家 ID 对齐，未选中的专家填 0。 | 原始 64×K；比较 19×256 |
| `expert_gate_up` | 选中专家的第一跳分组 W4 矩阵乘，把 gate 和 up 拼在一起输出，各占 512 维。 | 每个共同的 token–专家对 × 1024 |
| `expert_product` | 对该专家计算 `SiLU(gate) × up`。310P 在融合 SwiGLU 的乘积输出处取值。 | 每个共同的 token–专家对 × 512 |
| `expert_down` | 选中专家对 `expert_product` 做第二跳 W4 down 投影，得到该专家的输出。 | 每个共同的 token–专家对 × 2048 |
| `routed` | 同一个 token 下，把各个 `expert_down` 乘对应的 `route_weights` 后求和。 | 每 token × 2048 |
| `moe_out` | `shared_weighted + routed`，共享专家与路由专家合并后的 MoE 输出。 | 每 token × 2048 |
| `block_out` | 第二条残差相加，`residual + moe_out`。这是 L0 的最终 hidden，也是 L1 收到的输入。 | 每 token × 2048 |

## 单测 L0 的 input_layernorm

已有 L0 捕获文件时，在其所在服务器运行以下命令，可把单算子 AIR、实际输入和导出脚本放进同一目录。

```bash
case_dir="${AI_RUN_DIR:?}/target/tests/4-prefill-t64/board/L00-L03/l0-input-rmsnorm"
python targets/cpro/scripts/export_l0_rmsnorm_air.py --case-dir "$case_dir"
```

默认生成目录为 `$case_dir/single-op/`，也可以用 `--output-dir` 指定。目录里有 `export.air`、`x.bin`、`gamma.bin`、`export_l0_rmsnorm_air.py` 和脚本使用的 GE 描述文件 `torchair-ge-ir.pb`。Python 需要 NumPy 和 protobuf。两个 Data 节点按 x、gamma 的顺序输入，NetOutput 取唯一计算节点的第 0 个输出。默认 x 和 y 均为 FP16 `[1,64,2048]`，gamma 为 FP16 `[2048]`。脚本取 310P 实测 x 和原 CPro AIR 的有效 gamma，按指定端口类型写入两个 bin。epsilon 优先取 `--epsilon`，其次取捕获值，缺省时用 `1e-6`。

在生成目录中可用已有输入导出采用相同数值接口的自定义算子。以下命令适用于注册输入名为 `x/gamma`、单输出名为 `y`、浮点属性名为 `epsilon` 的实现。将环境变量 `RMS_OP_TYPE` 设为 DDK 中注册的 GE OpType。

```bash
cd "$case_dir/single-op"
python export_l0_rmsnorm_air.py --op-type "${RMS_OP_TYPE:?}" --output-ports y --output custom.air
```

`--input-ports` 可指定注册的两个输入名，`--output-ports y rstd` 可声明第二个 FP32 输出。AIR 对外只暴露归一化结果。若基线打印的 epsilon 与 `1e-6` 不同，给上面的命令加上同值的 `--epsilon`。epsilon 属性名可用 `--epsilon-attr` 指定，若注册接口无此属性，传 `--epsilon-attr ''`。只需生成内置模型时可直接运行 `python export_l0_rmsnorm_air.py --output export.air`。

在 CPro OMC 生成服务器的仓库根目录执行以下命令，从该用例的 `single-op/export.air` 生成同目录的 `export.omc`。OMG 从 AIR 读取输入名和静态形状。编译日志是 `export.compile.log`，上板输入仍是同目录的 `x.bin`、`gamma.bin`。

```bash
bash targets/cpro/scripts/decoder_compare.sh l0-input-rmsnorm-omc
```

如果只有上一轮捕获的 `case.json` 和输入文件，默认编译命令会先创建缺少的 `single-op/` 目录并生成 FP16 单算子包。若已在同目录导出 `custom.air`，在命令末尾加 `--rmsnorm-air custom.air`，得到 `custom.omc` 和 `custom.compile.log`。

四层 T64 用例的 310P 实测输入在 `export/fixture/inputs/hidden.bin`，OMC 生成服务器收到的副本在 `board/package/fixture/inputs/hidden.bin`。它是 1 × 64 × 2048 的 FP16 张量，共 262144 字节。L0 逐步采集用 `input_layernorm` 的前钩子记录 `310p_block_entry`，并与这份输入逐元素核对。对应的实测归一化结果是 `values.npz` 中的 `310p_hidden`，来自紧接该层归一化的 GDN 入口。

在当前同时配置 CPro OMC 生成和板端执行环境的服务器运行：

```bash
bash targets/cpro/scripts/decoder_compare.sh l0-input-rmsnorm
```

只需生成 AIR 和输入输出 bin 时，使用 `l0-input-rmsnorm-air`。脚本从原 AIR 的 embedding 乘法下游定位 L0 的输入 RMSNorm，保留该节点的属性和两个端口类型，同时在 `single-op/` 中生成上述可替换的 FP16 单算子包。原 AIR 显式记录 epsilon 时，这条命令将它传给单算子生成脚本。脚本打印各自的输入路径、shape、dtype 和字节数。

产物按用例放在 `board/L00-L03/l0-input-rmsnorm/`。其中 `weight/export.air` 是单算子模型，`input_310p.bin` 和 `output_310p.bin` 是 310P 捕获值，`gamma.bin` 是 CPro 算子的第二个输入。上板结果在 `replay/`，`comparison.json` 分别给出有效 token 和全部 64 行的指标。

## 与矩阵文件对照

| 报告范围 | 取哪些值 |
|---|---|
| `valid19` | 取有效 token 0–18。`window` 跳过开头 3 个历史位置，从窗口下标 3 起取 19 行。有效长度随 YAML prompt 改变。 |
| `full_state` | 对 `new_history`、`final_state`、`recurrent_cache` 分别比较完整张量。`new_history` 是物理尾部，后两个是按有效长度更新的递推状态。 |
| `physical60_63` | 另取 `block_entry`、`qkv`、`out`、`residual`、`post_norm`、`moe_out`、`block_out` 的物理位置 60–63。本次 prompt 下这四行都是 padding。 |
| `valid19_common_M_of_N` | 专家中间量仅在三端都选择了同一专家的同一 token 上比较。M 是共同分配数，N 是有效 token 数乘每 token 的 Top‑K。 |

查看整层误差从哪里放大，可以沿 `block_entry → hidden → out → residual → post_norm → moe_out → block_out` 阅读同一范围的矩阵。若 `moe_out` 出现明显变化，分别看 `shared_weighted`、`expert_ids`、`route_weights` 和 `routed`。切片本身的影响可用 `comparison.json` 中的 `four_layer_cache_control` 检查，它记录两个 L0 缓存与原四层 OMC 返回值的对照。

## 复算共享专家的标量门控

已有 `l0-steps-reference` 和 `l0-steps-same-input` 结果时，先在 CPU/310P golden 生成服务器提取 `layers.0.mlp.shared_expert_gate.weight` 和实测有效行。

```bash
bash targets/cpro/scripts/decoder_compare.sh l0-shared-gate-reference
```

在保留同输入上板结果和四层 AIR 的 CPro 服务器执行离线复算。

```bash
bash targets/cpro/scripts/decoder_compare.sh l0-shared-gate
```

每个 token 的门控分数等于该行 `post_norm` 与 2048 维权重的点积。先对照用 checkpoint 权重复算的 310P 分数和 310P 实测值，再把同一权重作用在 CPro 的 `post_norm` 上。两个预测值之间的差给出了只改变门控输入时的效果。CPro 预测值与板上分数之间的偏差还包含门控实现和中间量读回的影响。两端的点积均按 FP16 输入、FP32 累加计算。

脚本也会沿 AIR 中 sigmoid 的输入回溯到 `post_norm`，记录沿途算子，并比较可识别的 2048 维浮点常量与参考权重。每端的 `shared_down × sigmoid(实测 logit)` 可核对门控读数和加权输出是否相符。最后固定 CPro 的 `shared_down`，代入 310P 实测 logit，观察共享专家输出误差改变多少。逐 token 结果在 `board/L00-L03/l0-steps-same-input/shared-gate.json`。这两步只读已有张量和权重，无需重新导出、编译或上板。

当标量门控的 OMC 输出槽声明为 FP32 时，报告另外从原始 bin 的第 0 字节起按小端 FP16 连续读取有效 token 的分数，并与点积对照。读原始字节可以识别两个相邻 FP16 被当成一个 FP32 的情况。报告同时记录 AIR 生产端和消费端 dtype、OMC 槽位和文件路径。分别把两种解读的分数代入 `shared_down × sigmoid(logit)`，可判断哪种解读与同一切片中实测的共享专家贡献相符。逐 token 的 `observed_cpro` 保留按 OMC 描述读取的结果，`raw_as_fp16` 单列字节解读实验。

已有上述同入口数据时，可在同时配置 CPro OMC 生成和板端执行环境的服务器运行一次门控类型对照。

```bash
bash targets/cpro/scripts/decoder_compare.sh l0-shared-gate-fp16
```

该命令针对保留的旧 FP32 MatMul AIR，编译并执行一个 L0 `shared-router` 切片。它把共享专家标量门控的两个 MatMul 操作数显式转为 FP16，将 FP16 结果数值转换为 FP32 后送入原 sigmoid。对照报告在 `board/L00-L03/l0-shared-gate-fp16/result.json`。新导出的整层已在 PyTorch 中指定相同的类型转换。整层命令见[共享专家门控的 L0 整层回归](../decoder-compare.md#共享专家门控的-l0-整层回归)。在四层新 AIR 上运行常规 `l0-shared-gate` 对照，读取有效 token 的实测 logit 与同入口点积的偏差，以及 `shared_weighted` 对 310P 的相对 L2。
