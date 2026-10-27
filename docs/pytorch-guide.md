# PyTorch 构图、参考与导出

[首页](README.md) · [环境配置](environment-setup.md) · [GE 路径](ge-guide.md)

先按[环境配置](environment-setup.md)激活 `custom_pypto` 并加载项目配置，以下使用当前环境的 `python3`。以下完整模型使用 40 层 target、6 层 DFlash、hidden 2048、词表 248320；`sequence-length` 只控制输入长度。

## 1. 生成完整模型结构

使用 meta 参数，生成完整模块结构、层级 FX 图和可读代码，无需下载权重；产物不能推理，也不是 AIR。

```bash
export TORCH_FULL_RUN="$(mktemp -d "$AI_RUN_DIR/torch-full-XXXXXXXX")"
python3 -m qwen35_moe_dflash.cli build-full-pytorch \
  --sequence-length 8 --output "$TORCH_FULL_RUN/structure"
```

## 2. 使用真实权重运行参考

按[权重准备](weights.md)设置完整 target/draft 目录。下面使用 FP16 和 CUDA，需要足够显存；改为 `--device cpu` 时需要足够主机内存。

```bash
python3 -m qwen35_moe_dflash.cli reference \
  --device cuda:0 --precision fp16 --sequence-length 8 \
  --prompt-token-ids 1,2,3 --max-new-tokens 4 \
  --output "$TORCH_FULL_RUN/reference.json"
```

## 3. PyTorch → AIR → OMC

在匹配的 TorchNPU/TorchAir/CANN 环境导出，再使用目标 DDK 编译。`--float-baseline` 是完整未量化 FP16 转换诊断，不是 32GB 板端部署方案。

```bash
python3 -m qwen35_moe_dflash.cli export-air \
  --float-baseline --sequence-length 8 --output "$TORCH_FULL_RUN/air"
python3 -m qwen35_moe_dflash.cli compile-omc \
  --air "$TORCH_FULL_RUN/air" --output "$TORCH_FULL_RUN/omc"
```

成功后按[板端指南](board-guide.md)使用该 OMC 与同源参考；这里导出的是原始双图 ABI，新缓存模型使用下一节的独立接口。

## 4. 融合算子与缓存模型

先按 [GE 指南](ge-guide.md#2-准备完整真实权重)生成完整 FP16 prepared，并设置实际适配模块 `BASE_OPS_MODULE`、设备 `MODEL_DEVICE` 和供应商标识 `VENDOR_BACKEND_ID`。`FUSED_OPS=none` 使用基础后端，或选 `gdn,conv_state,rms` 的任意组合；选中的算子必须已安装注册，失败不会自动回退。

```bash
export FUSED_OPS='gdn,conv_state,rms'
python3 - <<'PY'
import os
import torch
from qwen35_moe_dflash.fused import (
    ModuleFusedOps, SelectiveFusedOps, load_prepared_target, FusedTargetSession,
)

base = ModuleFusedOps.from_name(os.environ["BASE_OPS_MODULE"])
ops = SelectiveFusedOps(
    base,
    fused_ops=os.environ["FUSED_OPS"],
    vendor_backend_id=os.environ.get("VENDOR_BACKEND_ID"),
    gdn_initial_state_dtype=torch.float16,
    gdn_state_casts="allow",
)
model = load_prepared_target(
    os.path.join(os.environ["GE_FULL_PAIR_RUN"], "prepared"),
    ops=ops, device=os.environ["MODEL_DEVICE"],
)
session = FusedTargetSession(model)
prefill = session.prefill([1, 2, 3])
print("first token:", session.next_token)
session.step(session.next_token)
print("next token:", session.next_token)
PY
```

此路径支持 batch=1、连续有效 token 和显式 conv/recurrent/KV 状态；真实后端、专用 AIR 导出及 native DFlash 调度仍待接通，不能直接交给旧双图 runner。

## 接口参考

基础适配模块须声明 `FUSED_OPS_ABI=1`、非空 `BACKEND_ID`，提供以下逻辑操作；具体签名见 [fused/ops.py](../model/qwen35_moe_dflash/fused/ops.py)。

| 函数 | 作用 |
|---|---|
| `rms_norm` | 使用有效 gamma；主 norm 为 `1+w`，gated norm 为 `w` |
| `conv_state_update`、`causal_conv1d` | 更新历史并卷积；SiLU 由模型执行 |
| `chunk_gated_delta_rule` | 返回 core 和新 recurrent state |
| `cache_update`、`attention` | 追加完整逻辑 ND KV，并执行因果 GQA |
| `moe` | 路由专家计算；共享专家由模型另算 |

逻辑缓存为 conv `[1,C,K-1]`、recurrent `[1,Hv,Dk,Dv]`、KV `[1,Hkv,T,D]`。适配层负责物理布局和异步同步；GDN 输出 FP32 state，示例显式允许转回默认 FP16 缓存。

验证提案使用 `branch=session.verify(proposals)`、`branch.commit(branch.max_accepted_prefix)`，随后 `session.step(session.next_token)` 消费修正或 bonus token；`branch.abort()` 丢弃分支。当前实现逐 token 验证并复制状态，完整 draft 特征上下文由调用方维护。

量化路径保留 `quant-plan → quantize → reference → export-air`；DOPT 仿真检查点通过 `--quantized-checkpoint` 传入，不等于文档 W4A8/group256/I4ZN 压缩部署模型。验证范围见 [validation.md](validation.md)。

## tiny 回归

仅检查原始小模型链路；TorchAir 和目标 DDK 仍需按阶段配置。

```bash
export TORCH_TINY_RUN="$(mktemp -d "$AI_RUN_DIR/torch-tiny-XXXXXXXX")"
python3 -m qwen35_moe_dflash.cli demo \
  --sequence-length 8 --prompt-token-ids 1,2,3 --max-new-tokens 4 \
  --output "$TORCH_TINY_RUN/reference.json"
python3 -m qwen35_moe_dflash.cli export-tiny-air \
  --sequence-length 8 --output "$TORCH_TINY_RUN/air"
python3 -m qwen35_moe_dflash.cli compile-omc \
  --air "$TORCH_TINY_RUN/air" --output "$TORCH_TINY_RUN/omc"
```
