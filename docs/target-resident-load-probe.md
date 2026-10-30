# Target V2 加载顺序的小型实验

首次常驻请求的 prefill 已完成，随后新 runner 在 `InitWeights` 返回 `3=INVALID_PARAM`，尚未调用 manager `Init`，draft 未加载。既有完整板端成功结果使用 `/data/local/tmp/tools_sysdbg/model_run_tool_internal --modelmanager_itf=V2` 加载 target；它的内部实现尚未取得。

用户随后提供的 `/home/w00949577/model_run_tool/src/model_runner.cpp` 使用 `AiModelMngerClient::Load/Process`，相邻测试脚本是 4B 的 72 输入／66 输出模型。这份源码不能用来推断当前 35B 工具的 V2 权重加载顺序。旧接口和 V2 的存在，也不能证明两者初始化契约相同。

当前假设：先用 `Init(options, builtModel, listener)` 将模型绑定给 manager，再调用 `InitWeights(weight_directory)`。这是基于当前失败点设计的一次实验，尚未被设备结果证实；原完整 resident 路线的默认顺序保持不变。

## 在 DDK 主机运行

复用首次 resident 任务的 prepared 和已存在的板端 target 缓存。命令只构建／发送当前 native runner 及其运行依赖；缓存缺失或哈希不符即停止，不重新上传大模型或权重。不使用 embedding 表、prompt、prefill 或 draft，不调用模型 `Run`，不创建输入／输出 tensor buffer。

```bash
git -C "$REPO_DIR" pull --ff-only

python3 "$REPO_DIR/targets/cpro/scripts/run_target_load_probe.py" \
  --prepared "$AI_RUN_DIR/dflash-board-resident8-run/prepared" \
  --output "$AI_RUN_DIR/target-resident-load-probe-v1" \
  --apply
```

HDC 中断时，保留原目录，通过同一 attempt 回收；不会再次初始化模型：

```bash
python3 "$REPO_DIR/targets/cpro/scripts/run_target_load_probe.py" \
  --recover "$AI_RUN_DIR/target-resident-load-probe-v1" \
  --output "$AI_RUN_DIR/target-resident-load-probe-v1-recovery"
```

每个输出目录必须是新目录。共享锁仍按实际进程退出和完成回执释放，未知执行状态保留锁。

## 如何解释结果

实验固定执行一次 `RestoreFromFile → SetName → CreateModelManager → Init → InitWeights`，随后清理。失败后不换顺序重试、不回退旧接口。报告保留实际调用阶段、精确返回码、加载／关闭墙钟及清理结果；未调用或异常无返回的状态值是 `null`。

- `Init` 失败：保留其返回码，`InitWeights` 应显示未调用；继续排查模型恢复、运行选项及原工具的加载方式。
- `Init` 成功、`InitWeights` 失败：检查权重目录与该 API 的具体契约，不能仅凭错误码断言缺文件或内存不足。
- 两项成功并完成清理：只证明该 target 的一次加载／关闭实验成功。下一步才将已观察到的顺序接入 resident runner，验证 embedded draft V2、双模型同时驻留、多轮执行和旧基线 token。

关闭仍采用现有 adapter 的 `DeInit → DeinitWeights`，报告分别保留尝试与状态；该关闭顺序也属于本次实验范围。尝试过 manager `Init` 后会调用 `DeInit`，仅 `InitWeights` 成功后才调用 `DeinitWeights`。成功加载但清理失败同样报告失败，不计作常驻成功。

本实验不产生 token，不证明推理、精度、加速或两模型常驻；它用于缩短加载故障的定位周期。

## 主机验证

2026-10-30：6项 native CTest 和15项主机测试通过。覆盖实际 adapter 的加载／部分失败／清理、原正式顺序回归、缓存缺失时不上传、启动响应丢失后只回收、未启动时释放自己所持锁，以及真实测试 runner 生成的成功和失败 JSON 被主机校验器接收。底层 HIAI／HDC 是明确的 host fake，未在此处连接用户设备。源码及测试文件哈希见[验证记录](../specs/target-resident-load-probe-validation.json)。
