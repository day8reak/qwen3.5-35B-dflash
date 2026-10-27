# PyTorch 建模、参考与导出

[首页](README.md) · [环境配置](environment-setup.md) · [端到端流程](project-structure.md#端到端如何运行)

PyTorch 路径有原始参考/导出模型，以及新增的融合算子与缓存模型。原始路径可生成 target/draft 双图 AIR；新模型当前用于已有融合实现的适配和状态语义验证，尚未提供自动导出到旧 HIAI ABI 的完整链路。

## 原始模型：先运行参考

按环境页设置变量。tiny 使用随机小模型，不下载完整权重：

```bash
export TORCH_RUN="$(mktemp -d "$AI_RUN_DIR/torch-XXXXXXXX")"
"$MODEL_PYTHON" -m qwen35_moe_dflash.cli demo \
  --sequence-length 8 --prompt-token-ids 1,2,3 --max-new-tokens 4 \
  --output "$TORCH_RUN/reference.json"
```

`model.py` 定义 DFlash draft，`weights.py` 加载锁定的目标模型及草稿；`runtime.py` 执行普通生成与 draft/verify 调度。完整权重与版本见[权重准备](weights.md)。完整模型参考使用：

```bash
"$MODEL_PYTHON" -m qwen35_moe_dflash.cli reference \
  --device cuda:0 --precision fp16 --sequence-length 128 \
  --prompt-token-ids 1,2,3 --max-new-tokens 4 \
  --output "$AI_RUN_DIR/reference-full-$(date +%Y%m%d-%H%M%S).json"
```

`cuda:0` 需有可用 GPU 和足够显存；也可显式选择 CPU，仍需足够内存。FP16 部署参考和上游 BF16 参考是不同精度路径。

## 原始模型：TorchAir → AIR → OMC

在匹配的 Torch/TorchNPU/TorchAir/CANN 环境执行。以下与上面的 PyTorch tiny 配套，不混用 GE 随机权重参考：

```bash
"$MODEL_PYTHON" -m qwen35_moe_dflash.cli export-tiny-air \
  --sequence-length 8 --output "$TORCH_RUN/air"
"$MODEL_PYTHON" -m qwen35_moe_dflash.cli compile-omc \
  --air "$TORCH_RUN/air" --output "$TORCH_RUN/omc"
```

得到成功 OMC 后，按[板端执行](board-guide.md)使用 `$TORCH_RUN/omc` 和 `$TORCH_RUN/reference.json`。TorchAir 负责把 PyTorch 图转为 GE/AIR，目标 DDK 的 OMG 再编译 OMC。

完整量化保留 `quant-plan → quantize → reference → export-air`：`quantize` 读取代表性 `{"token_ids": [[...], ...]}` 校准 JSON，输出 DOPT 仿真检查点；`reference` 和 `export-air` 用 `--quantized-checkpoint` 指向它。仿真检查点不等于压缩部署模型。文档 W4A8/group256/I4ZN 与既有 DOPT 候选不同，原生量化接口与全模型质量尚未闭环。

## 新模型：融合算子与缓存

实现集中在四个文件：

| 文件 | 作用 |
|---|---|
| [fused/model.py](../model/qwen35_moe_dflash/fused/model.py) | FP16 prepared 权重加载、完整 target 前向、层特征和状态输出 |
| [fused/ops.py](../model/qwen35_moe_dflash/fused/ops.py) | 外部算子模块的逻辑接口与模块 ABI 检查（返回张量由 model.py 校验） |
| [fused/session.py](../model/qwen35_moe_dflash/fused/session.py) | 缓存对象、快照、prefill/step、验证事务和接受前缀提交 |
| [fused/__init__.py](../model/qwen35_moe_dflash/fused/__init__.py) | 对外统一导入接口 |

只接受 batch=1、无 padding、连续位置的 token 流。逻辑缓存为 conv/recurrent/KV；读取未量化 FP16 prepared，不能直接加载 W4A8 成品。主 norm 只做一次 `1+w`，gated norm 直接用 `w`。

## 黄区适配层合同

适配模块需声明 `FUSED_OPS_ABI = 1` 和非空 `BACKEND_ID`，提供下列函数。**这是本仓库的逻辑接口，不是推测的GE端口定义。** GE 端口已由用户文档 §4.6 提供；PyTorch 调用层仍需绑定实际安装的模块及其缓存约定。

| 本仓库函数 | 预期黄区实现/边界 |
|---|---|
| `rms_norm(x, gamma, eps)` | `AdnRmsNorm`或确认过的RMSNorm；gamma已经是有效权重 |
| `conv_state_update(projected, history)` | `LinearAttentionConvstateUpdate`；返回规范化历史及卷积输入 |
| `causal_conv1d(padded, weight)` | 实际grouped conv；输出不含SiLU，模型随后调用SiLU |
| `chunk_gated_delta_rule(q,k,v,g,beta,state,effective_length,...)` | `ChunkGatedDeltaRule`；必须返回core和新state |
| `cache_update(old_k,old_v,new_k,new_v)` | `CacheUpdate`或确认过的缓存写入；逻辑接口返回完整ND缓存 |
| `attention(q,k,v,query_start=...,scale=...)` | `AdnFusedInferAttention`；适配层处理因果mask、GQA与物理布局 |
| `moe(hidden,router,gate_up,down,top_k=...)` | 路由、grouped GEMM和合并；不重复计算共享专家 |

逻辑conv历史为 `[1,C,K-1]`；recurrent为 `[1,Hv,Dk,Dv]`；KV为 `[1,Hkv,T,D]`。文档的convstate长度3→4，以及分页/NZ表示需要适配层按真实源码解释。不能直接对不明语义的状态随意截取或reshape。

初始和返回recurrent dtype由构造参数显式选定，默认FP16，错误类型会报错。可为接口核对显式选择FP32，但这不自动修改原生产精度合同。原始输入、权重等非缓存张量是只读参数；算子若就地改缓存，只能修改传入的私有分支。原生异步写入必须在返回前完成，Python tensor clone不能代替无关native stream的DMA同步。

导出还需将这些逻辑操作绑定为真实GE节点、记录多输出与状态类型，才可接入OMG/HIAI。当前PyTorch接口中的校验、Python状态对象和逻辑ND缓存不是可直接导出的HIAI wire ABI。

## 使用独立路径

显式加载已经完成真实绑定的适配模块；`YELLOW_OPS_MODULE`应是后续提供的实际Python模块名，不是已经随本仓库安装的包：

```python
import os
from qwen35_moe_dflash.fused import (
    ModuleFusedOps, load_prepared_target, FusedTargetSession,
)

ops = ModuleFusedOps.from_name(os.environ["YELLOW_OPS_MODULE"])
model = load_prepared_target(
    os.path.join(os.environ["GE_PAIR_RUN"], "prepared"), ops=ops,
)
session = FusedTargetSession(model)
prefill = session.prefill([1, 2, 3])
# prefill.features含全部prompt特征，供draft使用。
first_generated_token = session.next_token
session.step(first_generated_token)
```

上例默认CPU权重设备；真实适配层需要什么device，以其实现为准，可通过 `load_prepared_target(..., device=...)` 显式指定。需要做CPU结构检查时，用下面两行替换上例的 `ops = ModuleFusedOps.from_name(...)`，然后重新构造 `model` 和 `session`：

```python
from reference import fused_ops
ops = ModuleFusedOps(fused_ops, allow_reference=True)
```

参考模块需要仓库根目录及 `model/` 在 `PYTHONPATH` 中。该参考在norm、softmax和递推内部使用FP32数学运算，仅用于主机对照，不改变生产算术策略，不应将其结果标记为板端或完整模型精度通过。

## 显式选择文档中的融合算子

`SelectiveFusedOps` 可在一个已明确选择的完整基础后端上覆盖 `gdn`、`conv_state`、`rms` 的任意子集：

```python
import torch
from qwen35_moe_dflash.fused import ModuleFusedOps, SelectiveFusedOps

base = ModuleFusedOps.from_name(os.environ["BASE_OPS_MODULE"])
ops = SelectiveFusedOps(
    base, fused_ops="gdn,conv_state,rms",
    vendor_backend_id=os.environ["VENDOR_BACKEND_ID"],
    gdn_initial_state_dtype=torch.float16, gdn_state_casts="allow",
)
# 再将 ops 传入 load_prepared_target(..., ops=ops)。
```

`BASE_OPS_MODULE` 必须是实际提供全部逻辑操作的适配模块；这里不默认提供生产FP16基础后端。选择 `none` 时不导入TorchNPU；选中角色后要求对应操作已经注册，调用失败直接报错，其余操作始终使用base。GDN/conv按文档限定完整模型尺寸，不接受tiny替代。参考base需显式允许，并保留reference标记。

这层已按文档调用 `torch.ops.npu.npu_chunk_gated_delta_rule`、`torch.ops.npu.adn_rms_norm` 和 `torch.ops.my_ops.linear_attention_convstate_update`；并不证明当前安装包具有这些注册或支持目标芯片。主机mock测试只检查传参与状态边界。

## 提案验证和状态提交

```python
# proposals是draft给出的token列表；不是target确认过的token。
branch = session.verify(proposals)
accepted = branch.max_accepted_prefix
branch.commit(accepted)
# 第一个拒绝位置的target修正token；全接受时是bonus token。
correction_or_bonus = session.next_token
session.step(correction_or_bonus)
```

`verify`会按单token顺序运行target，并保留每个提案之后的完整状态快照；只允许提交连续匹配前缀。拒绝位置及其后计算出的state不会进入已提交状态。`commit(0)`保留原前缀；`abort()`直接丢弃分支。提交后仍需消费修正/bonus token，避免下轮状态落后一个token。`session.last_features`是最后一个已消费token的特征；draft完整上下文特征的积累仍由调用方管理。

本版本验证器是状态正确性实现，尚未连接原draft生成循环，也未实现一次多token设备verify或高效native状态bank。它保存/复制完整逻辑状态，**不能据此宣称DFlash加速**；后续性能实现需要保留这里的接受、回滚和继续解码语义。


验证与限制见[验证范围](validation.md)。实际设备 dtype、布局、同步和状态连续性仍需在最终算子路径验证。
