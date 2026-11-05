# StableArgMax：通用 Top1

有限连续ND FP16，静态rank1–8、正维度、总元素数≤INT32_MAX；沿最后维归约，返回INT32索引，并列取最小索引。TopK、其他轴、FP32和310P执行适配未实现。

**当前Qwen配置 `[1,15,248320] → [1,15]` 已在kirinx90完成七组、70次精确稳定的上板验证，单NPU子图；同步API均值约1.232ms。** 基线保留在 `draft/tests/stable-arg-max-ge-config`。17×8193尾部四组、40次也已通过，API均值0.515ms。当前进入[完整Draft接入](../../../../docs/current-status.md)，无需重建allops或重复单算子验证；下面是首次构建流程。

## DDK服务器：构建与GE单算子测试

不需要Torch/TorchAir。准备Python＋NumPy、已有allops工程、GE C++ SDK和所选CPro DDK。以下变量沿用现有环境：`REPO_DIR`为模型仓库，`ALLOPS_DIR`为多算子工程，`AI_RUN_DIR`为运行产物根目录，`CANN_ROOT`为GE SDK，`CPRO_DDK_ROOT`/`CPRO_DDK_ID`为安装算子的DDK。

只复制三个文件：`op_host/stable_arg_max.cpp`、`op_host/stable_arg_max_tiling.h`、`op_kernel/stable_arg_max.cpp`。复制工具保留其他算子和CMake；`--replace`备份被替换的自有文件。报告目录须为新目录。

```bash
export OP_DIR="$REPO_DIR/targets/cpro/ops/draft-top1"
python3 "$OP_DIR/scripts/prepare_allops.py" \
  --allops-root "$ALLOPS_DIR" \
  --output "$AI_RUN_DIR/draft/tests/stable-arg-max-copy" --replace

(cd "$ALLOPS_DIR" && bash ./build.sh)

python3 "$OP_DIR/scripts/stable_arg_max_ge.py" \
  --inspect-installation --ddk-root "$CPRO_DDK_ROOT"
```

本项目allops使用ascend310b的INI生成包装，host同时注册该键和kirinx90；OMG目标保持kirinx90。预检需为 `STATIC_PATH_RESOLVED`，不能仅依赖build.sh成功字样。不要把kernel CPP改名成PY，包装应由厂商生成。

首次生成模型时执行：

```bash
python3 "$OP_DIR/scripts/stable_arg_max_ge.py" \
  --ge-root "$CANN_ROOT" \
  --ddk-root "$CPRO_DDK_ROOT" --ddk-id "$CPRO_DDK_ID" \
  --tbe-diagnosis "$AI_RUN_DIR/common/tbe-diagnosis" \
  --output "$AI_RUN_DIR/draft/tests/stable-arg-max-ge-config"
```

此入口构建Data→StableArgMax→输出，默认只编译一份Qwen OMC供七组输入复用。TBE参数使用本项目已保留的诊断；进度在stderr，最终JSON及result.json保存状态。`COMPILED`表示编译和图层面归属检查通过，后续使用同一OMC上板。

- `--shape 17 8193`：另一静态shape的尾部检查，需要单独编译；不与当前Qwen产物混用。
- `--reuse-export <旧目录>`：仅修复编译环境时核验并复用AIR，输出写入新目录；不能复用已失效的OMC。
- `--inspect-compilation <目录>`：只读已有失败日志，无需重新编译。

Standalone探针IO限16MiB；现有板端runner支持输入rank2–4。rank1标量输出及rank5–8需要runner适配，不能从可导出推定可上板。所有模型都是静态shape。

## 权重服务器：Torch调用与导出

该服务器有Torch/TorchAir。将本目录python加入模块搜索路径，导出前注册converter：

```bash
export PYTHONPATH="$REPO_DIR/targets/cpro/ops/draft-top1/python${PYTHONPATH:+:$PYTHONPATH}"
```

```python
import torch
import torchair
from stable_arg_max import register_ge
register_ge(torchair)
tokens = torch.ops.custom_ops.stable_arg_max.default(logits)
```

| 层 | 名称 |
|---|---|
| Torch schema / Meta | custom_ops::stable_arg_max |
| GE / host注册 | StableArgMax，输入x、输出indices |
| kernel入口 | stable_arg_max |

`scripts/export_probe.py`保留TorchAir导出；DDK服务器使用上面的GE入口。完整Draft接入、PrivateUse1执行和310P adapter尚未验证。

[合同](docs/CONTRACT.md) · [环境](docs/ENVIRONMENT.md) · [验证](docs/VALIDATION.md) · [官方接口来源](docs/OFFICIAL_SOURCES.md)
