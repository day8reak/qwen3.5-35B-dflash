# Target V2 加载顺序的小型实验

首次常驻请求的 prefill 已完成，随后新 runner 在 `InitWeights` 返回 `3=INVALID_PARAM`，尚未调用 manager `Init`，draft 未加载。既有完整板端成功结果使用 `/data/local/tmp/tools_sysdbg/model_run_tool_internal --modelmanager_itf=V2` 加载 target；它的内部实现尚未取得。

用户随后提供的 `/home/w00949577/model_run_tool/src/model_runner.cpp` 使用 `AiModelMngerClient::Load/Process`，相邻测试脚本是 4B 的 72 输入／66 输出模型。这份源码不能用来推断当前 35B 工具的 V2 权重加载顺序。旧接口和 V2 的存在，也不能证明两者初始化契约相同。

用户的 v1 实验已经观察到 `Init(options, builtModel, listener)` 和随后 `InitWeights(weight_directory)` 都返回0。失败发生在关闭阶段：先 `DeInit` 后 `DeinitWeights`，后者返回3。总加载／关闭墙钟10568ms，没有 Run 或 draft 调用，任务已退出并释放锁，见[设备观察](../specs/resident-v2-target-init-observed.json)。

当前 v2 保留已成功的初始化顺序，把关闭改成 `DeinitWeights → DeInit`。这项关闭修复还需要实机验证。完整 resident 入口也已同步采用同一实现并写入显式策略；v2通过后即可直接运行[新常驻请求](current-status.md)，无需再等代码修改。

## 在 DDK 主机运行

复用首次 resident 任务的 prepared 和已存在的板端 target 缓存。命令只构建／发送当前 native runner 及其运行依赖；缓存缺失或哈希不符即停止，不重新上传大模型或权重。不使用 embedding 表、prompt、prefill 或 draft，不调用模型 `Run`，不创建输入／输出 tensor buffer。

```bash
git -C "$REPO_DIR" pull --ff-only

python3 "$REPO_DIR/targets/cpro/scripts/run_target_load_probe.py" \
  --prepared "$AI_RUN_DIR/dflash-board-resident8-run/prepared" \
  --output "$AI_RUN_DIR/target-resident-load-probe-v2" \
  --apply
```

HDC 中断时，保留原目录，通过同一 attempt 回收；不会再次初始化模型：

```bash
python3 "$REPO_DIR/targets/cpro/scripts/run_target_load_probe.py" \
  --recover "$AI_RUN_DIR/target-resident-load-probe-v2" \
  --output "$AI_RUN_DIR/target-resident-load-probe-v2-recovery"
```

每个输出目录必须是新目录。共享锁仍按实际进程退出和完成回执释放，未知执行状态保留锁。新请求写入 `cleanup_order=weights-before-manager`；历史v1恢复仍核对自己的旧参数／回执，不把旧关闭失败重新解释为成功。

## 如何解释结果

实验固定执行一次 `RestoreFromFile → SetName → CreateModelManager → Init → InitWeights`，随后清理。失败后不换顺序重试、不回退旧接口。报告保留实际调用阶段、精确返回码、加载／关闭墙钟及清理结果；Init／InitWeights未调用或异常无返回时，其状态值为 `null`。

- `Init` 失败：保留其返回码，`InitWeights` 应显示未调用；继续排查模型恢复、运行选项及原工具的加载方式。
- `Init` 成功、`InitWeights` 失败：检查权重目录与该 API 的具体契约，不能仅凭错误码断言缺文件或内存不足。
- 两项成功并完成清理：只证明该 target 的一次加载／关闭实验成功。下一步直接运行已修复的 resident runner，验证 embedded draft V2、双模型同时驻留、多轮执行和旧基线 token。

关闭采用 `DeinitWeights → DeInit`，报告分别保留尝试与状态；该关闭顺序属于本次待验范围。仅 `InitWeights` 成功后才调用 `DeinitWeights`；即使它失败，仍继续对已尝试 Init 的 manager 调用 `DeInit`。最初失败和清理错误分别保留，不重试同一清理 API。成功加载但清理失败仍报告失败，不计作常驻成功。

本实验不产生 token，不证明推理、精度、加速或两模型常驻；它用于缩短加载故障的定位周期。

## 官方资料与当前接口的区别

2026-10-30 检索到华为官方[模型管家 V2 接口索引](https://developer.huawei.com/consumer/cn/doc/hiai-References/model-manager-class-0000001139115907)，公开索引包含 `CreateModelManager`、`Init`、`Run`、`DeInit` 等，未列出本次使用的 `InitWeights`／`DeinitWeights`。本次检索尚未找到覆盖这两个外部权重 API 的官方调用顺序示例；这不等于断言所有官方资料都没有。

官方 [Android HiAI Foundation Codelab](https://developer.huawei.com/consumer/en/codelab/HiAIFoundation/index.html?cardName=HiAIFoundation&lang=en) 演示的是 `AiModelMngerClient`；[HarmonyOS 模型集成示例](https://developer.huawei.com/consumer/en/doc/harmonyos-guides-V14/hiaifoundation-integration-model-V14) 使用 `OH_NNCompilation_*`／`OH_NNExecutor_*`。它们可作为生命周期设计参考，不能替代当前所选 DDK 的 `IModelManager` 外部权重契约。当前顺序修复的证据仍是用户提供的所选头文件及这次实机返回码。

## 主机验证

v1的主机测试保存在[历史验证记录](../specs/target-resident-load-probe-validation.json)。v2关闭修复及完整常驻策略的主机检查见[当前验证记录](../specs/resident-v2-lifecycle-repair-validation.json)。底层 HIAI／HDC 是明确的 host fake，未在此处连接用户设备；主机通过不代替设备初始化／关闭或双模型运行结果。
