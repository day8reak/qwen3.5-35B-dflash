# 当前更新与执行命令

[首页](README.md) · [环境配置](environment-setup.md) · [验证记录](validation.md)

更新日期：2026-10-27。当前主线是完整40层target、6层DFlash和可选融合算子。

## 用户现在执行什么

**一条命令上板执行已编译的GDN prefill/decode。** 脚本自动构建当前runner、上传OMC及运行库、执行、回收并比较；复用已有OMC。

沿用[OHOS/HIAI/HDC环境](environment-setup.md#5-ohos--hiai--hdc)和刚才的 `FUSED_PROBE_RUN`。更新代码后执行：

```bash
cd "$REPO_DIR"
git pull --ff-only
"$MODEL_PYTHON" targets/cpro/scripts/run_fused_probes.py
```

`FUSED_PROBE_RUN` 是编译产物目录，也可用 `--probe-run 路径` 指定。脚本自动在 `$AI_RUN_DIR` 下新建结果目录、查询运行库版本和固件标识，结束时打印报告及失败日志。运行库默认从本次runner构建记录自动收集；已有板端私有库时可用 `BOARD_HIAI_LIB_DIR` 复用。

回传脚本打印的 `comparison/report.txt`；失败也保留本次板端目录、服务器日志和已回收结果。数值误差暂作诊断，NPU执行位置仍待验证。

## 当前状态

| 项目 | 状态 |
|---|---|
| PyTorch完整结构 | 全尺寸meta模型及层级FX图；不加载权重 |
| GE完整结构 | target/draft完整节点、连边、C++和recipe；融合可选 |
| 融合算子探针 | GDN prefill/decode已真实编译通过，名称适配已生效；上板待验证。conv/RMS接口问题待处理 |
| 真实权重AIR导出 | 旧双图支持完整FP16；新GE target支持显式缓存 |
| 状态执行 | 主机fake prefill→decode接续通过；真实板端待验证 |
| 尚未完成 | W4A8映射、常驻缓存与native DFlash调度、完整模型设备精度和性能 |

## 已确认与剩余问题

| 项目 | 结论 |
|---|---|
| GDN两例 | 用户回传构图、AIR、OMC均通过；每例名称适配 `APPLIED=1`、拒绝0 |
| 编译目标 | 始终为 `kirinx90`；`Ascend310B1` 是DDK初始化后返回的内部名称，`Ascend310P3` 是原工程平台 |
| GDN状态 | 安装版初态/末态为FP16；本次runner修正了末态只接受FP32的旧校验 |
| conv / RMS | 状态宽度4与文档3不一致 / 当前维推注册缺失，分别待处理 |
| 尚未验证 | 真实设备加载、执行、输出数值和NPU执行位置；完整模型与连续状态验证另行进行 |

## 完整模型构图（按需）

需要完整模型结构时选择下面一条路径；`sequence-length=8`只限定输入长度，模型层数、专家和词表均为完整尺寸。

**PyTorch：** 生成完整meta模型和层级FX结构，不需要检查点或TorchAir。

```bash
"$MODEL_PYTHON" -m qwen35_moe_dflash.cli build-full-pytorch \
  --sequence-length 8 --output "$AI_RUN_DIR/torch-full-$(date +%Y%m%d-%H%M%S)"
```

**GE：** 生成完整target/draft构图源码，默认不使用融合算子，不需要检查点或GE库。

```bash
"$MODEL_PYTHON" "$REPO_DIR/targets/cpro/scripts/direct_model_air.py" build-full-structure \
  --sequence-length 8 --expert-dispatch selected --fused-ops none \
  --output "$AI_RUN_DIR/ge-full-$(date +%Y%m%d-%H%M%S)"
```

这些是未绑定真实权重的结构产物。实际导出AIR：按 [PyTorch](pytorch-guide.md)或 [GE](ge-guide.md)指南继续；无需重跑旧探针。

## 文档融合接口

上面的 `GE_CUSTOM_BINDINGS` 可复用于GE融合构图。绑定来自用户文档，不代表kernel已安装或已在目标芯片验证；融合构图命令见[GE指南](ge-guide.md)。

完整W4A8 + FP16 DFlash先按24–28GiB可用RAM规划；这不是实测峰值，也不适用于当前FP16权重导出，见[内存预算](environment-setup.md#内存预算文档-w4a8--dflash)。
