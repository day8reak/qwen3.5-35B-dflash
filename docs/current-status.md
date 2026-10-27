# 当前更新与待执行命令

[首页](README.md) · [环境配置](environment-setup.md) · [项目结构](project-structure.md)

更新时间：2026-10-27。当前主线是已有融合算子与状态缓存的接入。

## 当前结果

| 项目 | 状态 |
|---|---|
| 原 tiny target/draft 双图 | 用户回传编译、加载、执行成功；普通/DFlash 各10次一致，但与主机参考有 token 偏差 |
| 旧41案例 GDN 修复 | 全部编译/加载/执行成功，数值32通过、9失败；不再要求重跑 |
| 独立 PyTorch 模型 | 模型、缓存事务及可选gdn/conv_state/rms适配已实现；真实基础后端与kernel待验证 |
| 独立 GE 状态图 | 融合可选；源码/AIR导出、独立OMC编译、step打包入口已实现 |
| 新状态执行 | 正式runner增加 `--stateful-step`；prefill/decode状态文件接续，主机fake验证，真实板端未跑 |
| 文档 §4.6 | 已保存并转为三个算子的具体端口绑定；FP16/FP32 初态需显式选择 |
| 尚未完成 | 黄区 kernel 安装与目标支持验证、常驻状态bank与native DFlash调度、量化部署、完整模型质量及性能 |

41案例提供了有效结论：所测卷积补零/窗口、beta/decay布局、weighted value及scan恢复；小幅度Q/K的MatMul归约仍丢值，原生归约对照通过。采用原生归一化后，首层最终输出最大误差为 `0.000793457`、相对L2约 `0.005915`；分阶段的 `normalized_key` 仍失败。不能把最终输出通过当成全部阶段或融合内核已通过。运行位置仍未独立确认。

## 用户现在执行什么

先按[环境配置](environment-setup.md)设置 `REPO_DIR`、`MODEL_PYTHON`、`AI_RUN_DIR`，沿用现有 DDK 标识。同步本次源码后，可以生成新接口文件；本步只读取文档并写 JSON，无需权重、重编模型或上板：

```bash
export GE_BINDING_RUN="$(mktemp -d "$AI_RUN_DIR/ge-bindings-XXXXXXXX")"
export GE_CUSTOM_BINDINGS="$GE_BINDING_RUN/bindings.json"
"$MODEL_PYTHON" "$REPO_DIR/targets/cpro/scripts/direct_model_air.py" bindings-from-document \
  --target "${CPRO_TARGET:-kirinx90}" \
  --ddk-id "${CPRO_DDK_ID:?填写拟使用的DDK版本标识}" \
  --gdn-initial-state-dtype float16 \
  --output "$GE_CUSTOM_BINDINGS"
```

产物为 `bindings.json` 和 `bindings.metadata.json`。这里选择文档 Cpro 调用的 FP16 初态，输出 state 为 FP32；选择 FP16 缓存时还需在导出中显式允许状态 Cast。JSON 中 target/DDK 名称只是声明，不能证明安装完成。

这个绑定限制文档实际模型的32头×128维、8192卷积通道，不能套到此前的 tiny 形状。已有 tiny prepared 可按 [GE 指南](ge-guide.md#状态图与可选融合)生成 `--fused-ops none` 的状态图源码，检查新 ABI；此操作是可选源码检查。

**当前不需要再执行旧探针或重复既有 benchmark。** 新状态图有独立 `compile-stateful` / `prepare-stateful-step` 和runner单步模式，具体命令见[GE指南](ge-guide.md#状态图编译和输入打包)。旧双图 `run_hdc.py` 不接新ABI。具备匹配资源后集中验证实际融合路径；当前未量化FP16全模型不能直接作为32GB部署方案。

## 本次整理

- GE 命令统一到 `direct_model_air.py`；原 `direct_fused_model_air.py` 的命令改为 `plan-stateful` / `export-stateful`，绑定命令不变。
- PyTorch 新模型的权重加载并入 `fused/model.py`，缓存与会话并入 `fused/session.py`；包级调用接口保留。
- GE 文档绑定并入 `ge_fused/bindings.py`，recipe 读写并入 `ge_fused/recipe.py`。
- 有结论的独立探针代码、测试与历史报告保存在仓库外归档，Git 历史亦保留原版本；现行文档只保留操作流程和结论。

## 本周结构交付重点

并行推进融合接口、GE编译与状态打包、正式runner状态执行；共同校验同一份状态ABI。当前仍缺常驻模型的高效状态bank、native draft/verify接续和真正W4A8权重映射。主机结构通过后集中做一次设备联调，停止扩展已收敛的算子探针。

文档W4A8 + FP16 DFlash的内存预算约为20 GiB常驻权重，模型进程先按24–28 GiB可用空间规划，加载副本和工作区峰值待测；详见[环境与内存](environment-setup.md#内存预算文档-w4a8--dflash)。
