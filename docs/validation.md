# 验证范围与保留结论

[首页](README.md) · [当前更新](current-status.md) · [项目结构](project-structure.md)

本页记录当前实现能证明什么。安装、编译、加载、执行、数值一致、实际执行位置和性能是不同检查；主机测试不作为设备通过记录。

## 状态执行与融合接口接入

2026-10-27，本轮新增了可选 PyTorch 融合适配、GE 状态图独立编译及输入打包、正式 runner 的 `--stateful-step` 模式。

| 范围 | 主机验证结果 |
|---|---|
| PyTorch 融合选择、模型及缓存事务 | 108 项和27项 subtests 通过；其中一项测试夹具修正后单独复测 |
| GE 编译与来源、ABI 校验 | 133项回归及4项补充检查通过 |
| 状态输入打包与连续调用校验 | 35项通过 |
| C++ runner | CTest 5/5通过 |
| Python → native prefill → 状态回传 → native decode | 通过；past_length 为 `0→3→4`，conv、FP32 recurrent、K/V 四组状态字节与 SHA256 精确接续 |

跨模块验证只替换外部 GE/OMG 运行产物和 HIAI 实现为明确标记的测试替身，保留真实的图构造、产物校验、打包和 native 执行流程。它证明主机接口与状态传递接通，**不证明真实融合 kernel、OMC 编译或板端精度通过**。常驻状态管理及 native DFlash 调度仍未接通。当前环境也未取得可用的目标资源清单，真实设备联调待完成。

命令见 [GE 指南](ge-guide.md#状态图编译和输入打包)和[板端单步执行](board-guide.md#6-新状态图的单步执行)，证据定位见 [specs/verification.json](../specs/verification.json) 的 `latest_integration`。

## 本次代码整理的验证

2026-10-27，模型解释器下的定向回归：

| 范围 | 结果 |
|---|---|
| PyTorch 新模型、算子接口、缓存及验证事务 | 89 tests + 27 subtests 通过 |
| GE 融合绑定、状态图、生成器与原导出回归 | 154 tests 通过 |
| HIAI/HDC 生产路径与依赖回归 | 150 tests 通过 |
| C++ 主机 runner 构建与 CTest | 构建成功，5/5 通过 |

剩余1843项 Python 测试收集通过，无导入错误；88个本地文档链接、29段Shell和3段Python示例通过静态检查。PyTorch/GE tiny参考均为PASS，文档接口生成与非融合状态图源码生成也已实际通过。

移动代码的定义与入口保留情况另做 AST 对照；整理验证没有执行真实 GE 库、OMG、融合 kernel 或板端程序。新代码的 host/reference/fake 测试覆盖接口和行为，不证明黄区 kernel 已安装或新 ABI 已在真实设备接通。

需修改模型或接口时，可在[环境配置](environment-setup.md)后执行对应回归；仅使用项目不必重跑：

```bash
export TEST_RUN="$(mktemp -d "$AI_RUN_DIR/tests-XXXXXXXX")"
"$MODEL_PYTHON" -m pytest "$REPO_DIR/tests/test_fused_model.py" \
  "$REPO_DIR/tests/test_fused_session.py" "$REPO_DIR/tests/test_ge_fused_target.py" \
  -p no:cacheprovider --basetemp "$TEST_RUN/tmp" --junitxml "$TEST_RUN/results.xml"
```

## 用户回传的设备结果

旧 tiny 双图已回传加载与生成成功。普通/DFlash 在其测试中一致，但尚未与主机参考完全对齐。41个旧 GDN repair 用例全部编译/加载/运行，数值32通过、9失败；这些是用户粘贴报告，未在本次获取原始设备产物重新校验。

已收敛的结论：

- 所测 rank-3 padding/window 及 beta/decay 布局修复后，对照恢复；weighted value 和 scan 对照通过。
- 小幅度 Q/K 的 MatMul 归约丢值，原生 Sum 对照保留16/16个非零结果；正确 sum 输入下的加 epsilon/Rsqrt 对照通过。
- 原生归一化首层最终输出通过，max_abs=`0.000793457`、relative_L2=`0.005915196552830379`；分阶段 `normalized_key` 仍失败，max_abs=`0.00390625`。
- 部分 MatMul/BMM 边界用例仍丢小值。最终 BMM 对照虽然过原阈值，仍只有92/128个非零结果，relative_L2约3.81%；不能只看 PASSED，也不能据此把所有算子概括为同一种下溢机制。

这些结果结束了相应探针的反复定位，未改变原有数值阈值。它们不覆盖最终融合 GDN、完整模型、缓存连续生成或 DFlash 加速，执行位置也未独立确认。

## 后续验收

围绕实际采用的融合路径，集中验证算子接口、同权重目标输出、prefill/decode 状态连续性、部分接受回滚，以及普通/DFlash token 与停止原因一致性。最终性能比较必须匹配权重、量化、缓存策略、提示、生成长度和测量条件。完整模型质量与至少2倍加速目标均未验收。

## 历史归档

本次删去的探针源码、用例、旧文档和逐次验证 JSON，已与整理前的未提交更改一起归档到仓库外。归档有逐文件 SHA256 清单，包含原 HEAD `4a043641fb5872b5af24c39b86b4b86e70b77014`；已提交版本也可从 Git 历史找回。

归档文件为 `before-reorganization/source.tar.gz`，SHA256：

```text
4269cccd3b3a57f04a15e8b520fb81f248a77a441593fa8a3b2d4ba0651071c3
```

本地运行记录和归档定位见 [specs/verification.json](../specs/verification.json)。源码仓库只保留当前结论与必要生产回归。
