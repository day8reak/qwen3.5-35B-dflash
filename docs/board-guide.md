# 板端执行与参考比较

[返回首页](README.md) · [环境配置](environment-setup.md) · [当前进展](current-status.md)

第1–5节用于**已有 ABI v2 的 target/draft 双图 OMC**。PyTorch → AIR 和直接 GE → AIR
都可走此流程，但必须使用各自同源的参考。新 stateful GE 图带显式缓存输入/输出，
使用[第6节的单步模式](#6-新状态图的单步执行)，尚未接入下述旧双图 `run_hdc.py` 流程。

## 1. 选择已有 OMC 与同源参考

先按[环境配置](environment-setup.md)设置 `REPO_DIR`、`MODEL_PYTHON`、`PYTHONPATH`、
`AI_RUN_DIR`、OHOS/HIAI 资源以及 `SN`、`BIP`、`HDC`。保留已可用的设备连接和工具链。
`BIP` 是 HDC server 的 `IP:端口`。

将下列两个路径替换为已有产物；每次执行用新的结果目录：

```bash
export OMC_DIR='/实际路径/已编译的双图OMC目录'
export REFERENCE_JSON='/实际路径/同源reference.json'
: "${AI_RUN_DIR:?请先设置本次仓库外输出目录}"
export BOARD_JOB="$(mktemp -d "$AI_RUN_DIR/board-XXXXXXXX")"
test -f "$OMC_DIR/manifest.json"
test -f "$REFERENCE_JSON"
"${HDC:-hdc}" -s "$BIP" list targets
```

OMC 清单须为 `qwen35-cpro-omc-pair`、`COMPILED`、ABI `2`，包含 target 和 draft，
精度与特征类型为 `torch.float16`。参考的 `source`、`precision`、`quantization`、
`sequence_length` 必须与 OMC 清单相同；脚本会在上传前检查这些字段及模型载荷哈希。

| OMC 来源 | 参考来源 |
|---|---|
| 旧 PyTorch 双图导出 | 按 [PyTorch 指南](pytorch-guide.md)使用相同 target/draft 权重、精度、量化和序列配置生成的参考 |
| 旧直接 GE tiny 双图 | 按 [GE 指南](ge-guide.md)用该次 `prepared` 执行 `direct_model_air.py reference-tiny` 得到的参考 |

不能用另一次 `demo` 生成的随机 tiny 权重替代原参考，也不能手改参考身份字段来绕过检查。
`reference-tiny` 仅适用于其支持的 tiny prepared；完整模型参考需走对应的完整权重流程。

## 2. 构建正式 HIAI runner

```bash
"$MODEL_PYTHON" "$REPO_DIR/targets/cpro/scripts/build_hiai.py" \
  --output "$BOARD_JOB/hiai-build"
cat "$BOARD_JOB/hiai-build/result.json"
export HIAI_RUNNER="$BOARD_JOB/hiai-build/build/qwen35_dflash_hiai"
```

应得到 `LINKED` 和 HarmonyOS ARM64 程序。需要指定最低兼容 API 时，可显式增加
`--compatible-sdk-version 正整数`；不指定则使用所选 SDK 默认值。
`--headers-only` 只验证头文件编译，不能用于上板。

已有匹配 runner 可直接将 `HIAI_RUNNER` 指向它，跳过构建。保留相邻的
`command.json`、`result.json` 以及其引用的库和 SDK 路径；运行脚本据此校验并收集同套
HIAI/C++ 库，无需另行猜测库目录。资源路径来自环境或 `QWEN35_RESOURCES`。

## 3. 只验证双图执行

需要首次检查该产物能否加载、执行和传递 target 特征时运行：

```bash
"$MODEL_PYTHON" "$REPO_DIR/targets/cpro/scripts/run_hdc.py" \
  --omc "$OMC_DIR" --runner "$HIAI_RUNNER" \
  --reference "$REFERENCE_JSON" \
  --output "$BOARD_JOB/smoke" --execution-only
cat "$BOARD_JOB/smoke/result.json"
```

脚本创建独立板端目录，上传并校验载荷，检查运行库能力，再让 target/draft 各执行一次。
该模式仍核对参考身份与 prompt，但不要求参考状态为 PASS，也不比较生成质量或计时收益。
`EXECUTION_COMPLETE` 表示本次双图执行与回传检查完成；精度、性能和实际算子运行位置未由此确认。
已有同一产物成功执行证据时，不必重复这一步。

## 4. 完整生成比较

严格比较要求参考状态为 `PASS`，并满足
`prompt 长度 + max_new_tokens − 1 ≤ OMC sequence_length`。prompt 与生成上限直接读取参考 JSON。
去掉 `--execution-only`，使用新的输出目录：

```bash
"$MODEL_PYTHON" "$REPO_DIR/targets/cpro/scripts/run_hdc.py" \
  --omc "$OMC_DIR" --runner "$HIAI_RUNNER" \
  --reference "$REFERENCE_JSON" \
  --output "$BOARD_JOB/benchmark"
cat "$BOARD_JOB/benchmark/result.json"
cat "$BOARD_JOB/benchmark/comparison.json"
```

该流程检查普通生成与 DFlash 一致性，并为两种模式各做 3 次预热、10 次测量，
回收结果后自动调用项目 `compare`。通过要求包括 OMC/权重身份、prompt、生成上限、
两种模式每次测量的 token 序列及停止原因与参考一致。

若已经有完整 benchmark，可只在主机重新比较，无需再上板；输出文件必须尚不存在：

```bash
"$MODEL_PYTHON" -m qwen35_moe_dflash.cli compare \
  --omc "$OMC_DIR" \
  --report "$BOARD_JOB/benchmark/benchmark.json" \
  --reference "$REFERENCE_JSON" \
  --output "$BOARD_JOB/comparison-recheck.json"
```

这里的 `--report` 必须是完整生成测量报告，不能使用 execution-only 的 smoke 报告。
比较失败会保存 `FAIL` 结果并以非零退出码退出；查看 `diagnostics` 可区分重复运行不一致、
ordinary/DFlash 互不一致，以及两者都稳定但偏离参考。

## 5. 产物、清理和结果边界

| 文件 | 内容 |
|---|---|
| `result.json` | 流程状态、身份、失败阶段和板端清理结果 |
| `runtime-probe.json` | HIAI 运行库版本与能力查询；不是模型执行位置证明 |
| `benchmark.json` | smoke 双图输出，或 ordinary/DFlash 的完整测量 |
| `comparison.json` | 完整模式的参考比较；smoke 不生成 |
| `benchmark.stdout` / `.stderr` / `.exitcode` | 板端推理日志与实际退出码 |
| `calls/`、`failure.json` | HDC 调用记录；失败时的原因 |

默认成功完成回传及相应检查后删除本次板端目录，失败时保留；服务器结果始终保留。
需要保留部署内容时加 `--board-cleanup none`；`--board-cleanup omc` 仅在成功后删除清单中的
OMC 文件，保留独立权重及其它文件。下次执行仍需新目录。传到板端的是程序、OMC 及附属载荷、
清单和运行库；prepared、AIR、Python、CANN、DDK、编译 SDK 留在服务器。

`PASS` 的范围是这批输入和产物的 token/停止原因比较，不能代替逐层数值检查。
tiny 结果也不代表完整 35B 模型质量或加速收益；NPU 能力查询、编译分区和 Process 成功
均不单独证明实际 NPU 执行位置。新融合与缓存路径的接入状态统一查看[当前进展](current-status.md)。

## 6. 新状态图的单步执行

新状态图使用同一份正式runner的独立 `--stateful-step` 模式；构建方法同第2节，旧runner需重新构建。主机先按[GE指南](ge-guide.md#状态图编译和输入打包)生成完整step bundle。自动 `run_hdc.py` 当前仍只接旧双图，不接这个bundle。

将完整bundle、runner及匹配运行库部署到板端，保持bundle相对路径；然后在板端已配置对应进程库搜索路径的shell执行：

```bash
"$BOARD_RUNNER" --stateful-step "$BOARD_STEP_DIR/step.tsv" \
  --output "$BOARD_RESULT_DIR" \
  --expected-runtime-version "$HIAI_RUNTIME_VERSION" \
  --ddk-id "$CPRO_DDK_ID" --firmware-id "$FIRMWARE_ID"
```

上述变量是**板端**runner路径、step目录、新结果目录、实际运行库版本与固件身份，不是服务器路径。输出 `result.json` 及全部 `output_*.bin`，回收完整结果目录后，才能准备下一T位置的decode。该模式检查模型/ABI/输入哈希、shape和dtype，保留FP32 recurrent状态；与旧target/draft参数互斥。

`EXECUTED`只表示一次target执行及输出写出，不表示数值验收或DFlash性能通过。当前主机fake已覆盖接口；真实编译和板端结果仍待获得。单步进程用于接通状态路径，每次启动会重新加载模型，不能用它估计最终常驻解码性能。
