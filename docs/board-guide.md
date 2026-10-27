# 板端执行与参考比较

[首页](README.md) · [环境配置](environment-setup.md) · [GE 指南](ge-guide.md)

新缓存图使用 runner 的 `--stateful-step`；原始 ABI v2 target/draft 双图使用 `run_hdc.py`。完整 FP16 模型约 65 GiB 权重，以下执行接口不代表它已能部署到 32GB 板端。

## 1. 构建 HIAI runner

先配置 OHOS/HIAI/HDC 环境；已有匹配的新 runner 可跳过构建，保留相邻构建记录及其引用的库路径。

```bash
export BOARD_JOB="$(mktemp -d "$AI_RUN_DIR/board-XXXXXXXX")"
"$MODEL_PYTHON" "$REPO_DIR/targets/cpro/scripts/build_hiai.py" \
  --output "$BOARD_JOB/hiai-build"
export HIAI_RUNNER="$BOARD_JOB/hiai-build/build/qwen35_dflash_hiai"
cat "$BOARD_JOB/hiai-build/result.json"
"$HDC" -s "$BIP" list targets
```

构建结果应为 `LINKED`；`--headers-only` 产物不能上板，需要指定最低 SDK API 时增加 `--compatible-sdk-version 正整数`。

<a id="6-新状态图的单步执行"></a>

## 2. 新状态图的单步执行

按 [GE 指南](ge-guide.md#状态图编译和输入打包)准备 step bundle，将完整 bundle、runner 和匹配运行库部署到板端并保持相对路径。当前 `run_hdc.py` 不负责此新 ABI 的部署；在已配置运行库路径的**板端 shell**中设置实际路径和身份后执行：

```bash
"$BOARD_RUNNER" --stateful-step "$BOARD_STEP_DIR/step.tsv" \
  --output "$BOARD_RESULT_DIR" \
  --expected-runtime-version "$HIAI_RUNTIME_VERSION" \
  --ddk-id "$CPRO_DDK_ID" --firmware-id "$FIRMWARE_ID"
```

`BOARD_RUNNER`、`BOARD_STEP_DIR`、`BOARD_RESULT_DIR` 均为板端路径，结果目录必须是新目录；运行库版本和固件标识需填实际值。回收 `result.json` 及全部 `output_*.bin` 后，按 GE 指南准备下一 T 位置的 decode；`EXECUTED` 只表示单步执行完成，真实数值及 DFlash 性能仍待验证。

## 3. 选择原始双图与同源参考

以下步骤适用于清单为 `qwen35-cpro-omc-pair`、`COMPILED`、ABI 2 的 target/draft OMC；参考必须匹配权重、精度、量化、序列长度和 prompt，脚本会检查。

```bash
export OMC_DIR='/实际路径/已编译的双图OMC目录'
export REFERENCE_JSON='/实际路径/同源reference.json'
test -f "$OMC_DIR/manifest.json"
test -f "$REFERENCE_JSON"
```

## 4. 双图执行检查

`--execution-only` 让 target/draft 各执行一次并检查回传，不做生成质量或速度验收；已有相同产物的成功记录可跳过。

```bash
"$MODEL_PYTHON" "$REPO_DIR/targets/cpro/scripts/run_hdc.py" \
  --omc "$OMC_DIR" --runner "$HIAI_RUNNER" --reference "$REFERENCE_JSON" \
  --output "$BOARD_JOB/smoke" --execution-only
cat "$BOARD_JOB/smoke/result.json"
```

## 5. 完整生成比较

参考状态须为 `PASS`，并满足 `prompt长度 + max_new_tokens − 1 ≤ sequence_length`。该命令对 ordinary/DFlash 各做 3 次预热、10 次测量，并比较全部 token 与停止原因。

```bash
"$MODEL_PYTHON" "$REPO_DIR/targets/cpro/scripts/run_hdc.py" \
  --omc "$OMC_DIR" --runner "$HIAI_RUNNER" --reference "$REFERENCE_JSON" \
  --output "$BOARD_JOB/benchmark"
cat "$BOARD_JOB/benchmark/result.json"
cat "$BOARD_JOB/benchmark/comparison.json"
```

已有完整 benchmark 时，可只重新比较，无需上板；`--report` 不能指向 execution-only 报告。

```bash
"$MODEL_PYTHON" -m qwen35_moe_dflash.cli compare \
  --omc "$OMC_DIR" --report "$BOARD_JOB/benchmark/benchmark.json" \
  --reference "$REFERENCE_JSON" --output "$BOARD_JOB/comparison-recheck.json"
```

## 结果与保留策略

| 产物 | 内容 |
|---|---|
| `result.json` | 状态、身份、失败阶段及清理结果 |
| `benchmark.json` / `comparison.json` | 双图输出或完整生成测量 / 参考比较；smoke 不生成 comparison |
| `benchmark.stdout` / `.stderr` / `.exitcode` | 板端日志和实际退出码 |
| `calls/` / `failure.json` | HDC 调用记录 / 失败原因 |

默认成功回传并完成检查后清理本次板端目录，失败保留；加 `--board-cleanup none` 可保留全部内容，服务器结果始终保留。编译分区、能力查询和 Process 成功均不能单独证明实际 NPU 运行位置；验证范围见 [validation.md](validation.md)。

## 融合算子调用探针

使用 `targets/cpro/scripts/run_fused_probes.py --all-ops` 一次完成五类算子的缺项编译、runner构建、上传、执行和回收。执行命令集中在[当前更新](current-status.md#用户现在执行什么)。已有 `FUSED_PROBE_RUN` 中编译成功的OMC直接复用，缺项另建目录；编译失败仍继续其余用例。省略 `--all-ops` 则只运行已有编译产物。

`--all-ops --probe-run 上次批量目录 --retry-cases attention_prefill,attention_decode` 只重编指定缺项，并复用该批次各组已编译OMC。未重试的失败仍保留；已有conv/RMS编译失败时，自动读取DDK注册配置到 `inspection/`，不执行安装或改写DDK。

默认新建独立板端目录，按原上板脚本的规则收集构建对应的运行库、排除DDK HCL库对，并加入 `/vendor/lib64/chipsetsdk` 搜索路径。`BOARD_HIAI_LIB_DIR` 可显式复用已有私有库目录。运行库版本和固件标识自动查询；传输文件校验SHA256，执行失败仍尝试回收。

全部模式的 `report.txt` 汇总9例；`probes/` 为缺项编译产物，`board-*/` 分别保留原批次/新批次的上板记录。各组 `results/` 为输出和日志，`comparison/report.txt` 为详细诊断。runner在同次批量运行中只构建一次。成功和失败均保留板端目录；数值验收及NPU执行位置另行验证。

## tiny 回归

复用同一执行命令，只替换 OMC 和参考：PyTorch tiny 使用同次 `demo/export-tiny-air`；GE tiny 使用同次 prepared 的 `reference-tiny`。两种随机模型的参考不能混用，tiny 通过也不代表完整模型通过。
