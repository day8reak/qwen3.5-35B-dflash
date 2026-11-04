# Qwen35DraftTop1

固定边界：连续 FP16[1,15,248320] → INT32[1,15]，有限 logits、完整词表、同值取最小索引。

当前为合同与主机可执行参考阶段。NPU kernel、host tiling、GE 注册和设备验证均待完成；不能构建或接入 Draft。原内置 Bitcast 路线已因所选配置不匹配结束。

先在所选 DDK 主机一次收集实际接入接口（不加载 vendor Python，不编译，不连接设备）：

```bash
python3 "$REPO_DIR/targets/cpro/operators/draft-top1/scripts/inspect_ddk.py" \
  --ddk-root "$CPRO_DDK_ROOT" \
  --compile-environment "$AI_RUN_DIR/draft/tests/top1-last-builtin/top1_bits_payload/omc/omg-environment.json" \
  --output "$AI_RUN_DIR/draft/tests/custom-top1-ddk"
```

返回 `ddk-context.tar.gz`，内含带 SHA256 的源码、注册、构建与接口片段；`summary.json` 只是收集摘要，不含接入所需全部内容。只遍历所选 DDK 和显式 `--source-root`；外部链接记录为缺失，预算不足记录 PARTIAL。SDK 不修改，输出目录不覆盖。

主机验证（需 C++17、NumPy、PyTorch）：

```bash
python3 "$REPO_DIR/targets/cpro/operators/draft-top1/scripts/validate_host.py" \
  --output "$AI_RUN_DIR/draft/tests/custom-top1-host"
```

该命令生成七组完整尺寸 fixture，对照 NumPy/PyTorch ArgMax 和独立 C++ binary16 整数排序参考。70 份合成输出文件仅用于比较器测试，不是70次设备执行。后续设备必须绑定 OMC、输入、运行器、环境哈希，并另做3次预热和10次真实调用。

详见 [合同](docs/CONTRACT.md)、[环境缺口](docs/ENVIRONMENT.md)、[验证](docs/VALIDATION.md)、[计划](docs/PLAN.md)。
