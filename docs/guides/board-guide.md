# 板端执行与参考比较

> 开发指南：本页涉及其他导出／ABI路径。当前原40层模型的文字＋预读运行入口在[这里](../reference/board-text-input-output.md)。


[首页](../README.md) · [环境配置](environment-setup.md) · [GE 指南](ge-guide.md)

新缓存图使用 runner 的 `--stateful-step`；原始 ABI v2 target/draft 双图使用 `run_hdc.py`。完整 FP16 模型约 65 GiB 权重，以下执行接口不代表它已能部署到 32GB 板端。

## 独立 SiluPatch 的分段 prefill

用 `targets/cpro/scripts/split_prefill_t64.sh` 运行 W4、S=64、有效 token 为 19 的同一条输入。每个模式都执行完整 decoder，包含 attention、MoE 和最终残差。每处 SiLU 使用 FP16 输入、FP32 输出的独立 NPU OMC。

| `--mode` | 层输入与读出 | 对比的状态 |
| --- | --- | --- |
| `linear` | 从原 CPro embedding 输入跑 L0，再接原模型 norm/head | L0 conv_state 和 recurrent_state |
| `full` | 从 310P 实测 L2 输出跑 L3，再接原模型 norm/head | L3 更新后的 K 和 V |
| `40` | 从 embedding 跑 L0 至 L39，逐层传 CPro 的实际 hidden，在 L39 后接 norm/head | 全部 30 个 linear 层的两种状态和 10 个 full 层的 K、V |

78 的 `air` 命令采集或复用 310P 参考、导出 AIR，并用原 W4 权重计算单独的 CPU 参考。53 负责编译、上板和对比。两台主机拉取同一版代码，以下示例的 `MODE` 在两端取相同值。

```bash
# 78
MODE=full
bash targets/cpro/scripts/split_prefill_t64.sh air --mode "$MODE"

# 53
MODE=full
bash targets/cpro/scripts/split_prefill_t64.sh board --mode "$MODE"
```

`--run DIR` 指定该模式的运行目录。当前默认目录分别是 `$AI_RUN_DIR/target/tests` 下的 `l0-split-silupatch-t64-valid19-02`、`prefill-split-full-t64-valid19-02` 和 `prefill-split-40-t64-valid19-02`。三个模式均使用 CPro W4 投影，导出时检查每个量化 matmul 的 weight scale 和 per-token scale 都是 FP16，并将端口记录在各层的 `air/manifest.json` 的 `precision` 字段里。linear 层返回物理序列末四行的卷积状态，按有效长度更新 GDR 状态。full 层在 FP32 计算 RoPE，再将 Q/K 转回 attention 的输入类型。各层在 FP16 计算残差，在分段接口以 FP32 传递 decoder hidden。

首次导出使用 `ORIGINAL4_RUN` 的成功原版基线，默认 `decoder4-original-t64-01`。可以用 `air --capture DIR` 复用旧实验同一份 310P 参考。切换到修正版时先导出，再从第一段上板。后续在同版模型内使用恢复参数。40 层模式在 310P 捕获记录的两个模型段交界处保留上段 CPro 的前 19 行并将剩余 45 行清零。

**恢复上板。** 在 53 用 `--recovery` 接着最近一次运行，或显式指定旧 replay。脚本将已完成段的 OMC、输入和输出与回执核对，在新 replay 中使用校验通过的输出，每段成功后写入 `chain.json`。有意重跑某段时加 `--from-stage`，其前面的输入必须能从已完成的前缀恢复。

```bash
bash targets/cpro/scripts/split_prefill_t64.sh board --mode linear \
  --recovery --from-stage decoder_hidden
bash targets/cpro/scripts/split_prefill_t64.sh board --mode 40 --recovery
```

**复现 L0 分段。** 三个模式的 `board` 默认使用 L0 跑通的切法。MoE 尾部的三条 W4 投影独立执行，attention 后的残差独立读回，接着计算其余 MoE 和最终残差。linear 层有四个独立 SiluPatch。full 层的两个 SiluPatch 分别用于共享专家和路由专家。每次执行会打印当前 AIR 的 OMC 段数。独立 L0 包含最后的 norm/head，40 层连续执行时只在 L39 接 norm/head。

在 53 用已有 AIR 和编译缓存从第一段重新上板，例如：

```bash
bash targets/cpro/scripts/split_prefill_t64.sh board --mode linear --run "$R"
```

每次从头执行都会新建 replay。各段使用 CPro 实际返回的张量，`chain.json` 的 `stages` 列出本次完成的段名和 OMC 哈希。默认切法的节点和数据依赖见 `plan-w4-residual.json`，W4 激活和投影使用内嵌权重，编译产物在 `omc/decoder_w4_*-embedded/`。复查先前只拆 W4 投影的对照实验时，可显式传 `--split-decoder-projections`。

若旧 replay 的三个 W4 投影已完成，OMG 在 `decoder_hidden_after_w4` 为 `Cast` 生成任务时失败，下列命令会从 `decoder_attention_residual` 继续，再将其板端输出送入 `decoder_hidden_from_residual`。

```bash
bash targets/cpro/scripts/split_prefill_t64.sh board --mode linear \
  --run "$R" --recovery --from-stage decoder_hidden_after_w4 --split-decoder-residual
```

40 层显式起点的写法如 `--from-stage L03/out_proj`，可用段名见各层的计划文件。full 和 40 会先逐层保存板端输出，再自动对比。若拉取 CPU 参考时中断，补齐参考后单独运行 `compare` 即可。linear 的报告是 `replay-*/comparison.json`，full 和 40 的汇总是 `replay-*/prefill.json`，各层明细是 `Lxx/comparison.json`。同目录的 `matrices.csv` 对每个指标给出 CPU、310P、CPro 的 3×3 矩阵，行是 actual，列是 reference。精度通过条件沿用 CPro 对 310P，logits 另要求 top1 一致。

| 输出 | 比较范围 |
| --- | --- |
| out_proj、decoder hidden | 前 19 行合并为 valid19 |
| norm/head 返回的 logits | 位置 18 的一行全词表，另记三方 top1 |
| conv_state、recurrent_state | 本次 prefill 后的完整状态。卷积历史取 T64 的末四行，递归累计 19 个有效 token |
| key_cache、value_cache | 分别比较 19 个有效位置，另校验 64 个物理写入位置以外的页 |

CPU 在 40 层模式逐层传自己的 hidden，在 310P 的分段边界同步清零 padding。对 19 个位置之外的 CPU 行，attention 分支输出按零处理，残差和 MoE 仍传播。这一约定影响后续层的物理卷积历史。单测 full 模式的 CPU 和 CPro 都以同一份 L2 实测输出为入口。

已有修正版板端输出时，在 53 运行 `bash targets/cpro/scripts/split_prefill_t64.sh compare --mode "$MODE"`，可加 `--replay DIR`。显式指定失败的 replay 时，只对已返回的输出生成矩阵，并标记未完成。

如果 53 拉取 `cpu/` 时收到 `ENOENT`，可在 78 单独补 CPU 参考。脚本按 AIR 中记录的 SHA-256 从本次及同级实验目录查找原始 310P capture。如果 capture 在其他位置，加 `--capture /实际路径/310p`。以下示例在 53 复用最近一次执行完成的 L0 replay。

```bash
# 78
R="${AI_RUN_DIR:?}/target/tests/l0-split-silupatch-t64-valid19-02"
bash targets/cpro/scripts/split_prefill_t64.sh cpu --mode linear --run "$R"

# 53
R="${AI_RUN_DIR:?}/target/tests/l0-split-silupatch-t64-valid19-02"
bash targets/cpro/scripts/split_prefill_t64.sh compare --mode linear --run "$R"
```

若 V2 在同一段反复加载失败，在 53 读取已有尝试的设备日志。以下示例读取一次失败的 `decoder_hidden`，在命令末尾追加其他失败段的目录即可对比。脚本核对 OMC、外置权重和有序输入的哈希，使用每次调用的设备时间和 client_tid 从同一份 HiLog 中提取记录。命令只读取板端文件，诊断结果写到第一段回执旁的 `load-diagnostics-*`。

```bash
R="${L0_SPLIT_RUN:-${AI_RUN_DIR:?}/target/tests/l0-split-silupatch-t64-valid19-01}"
bash targets/cpro/scripts/split_prefill_t64.sh diagnose \
  "$R/replay-d3iori8t/decoder_hidden"
```

## 1. 构建 HIAI runner

先按[环境配置](environment-setup.md)激活 `custom_pypto` 并配置 OHOS/HIAI/HDC；以下 `python3` 使用该环境。已有匹配的新 runner 可跳过构建，保留相邻构建记录及其引用的库路径。

```bash
export BOARD_JOB="$(mktemp -d "$AI_RUN_DIR/board-XXXXXXXX")"
python3 "$REPO_DIR/targets/cpro/scripts/build_hiai.py" \
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
python3 "$REPO_DIR/targets/cpro/scripts/run_hdc.py" \
  --omc "$OMC_DIR" --runner "$HIAI_RUNNER" --reference "$REFERENCE_JSON" \
  --output "$BOARD_JOB/smoke" --execution-only
cat "$BOARD_JOB/smoke/result.json"
```

## 5. 完整生成比较

参考状态须为 `PASS`，并满足 `prompt长度 + max_new_tokens − 1 ≤ sequence_length`。该命令对 ordinary/DFlash 各做 3 次预热、10 次测量，并比较全部 token 与停止原因。

```bash
python3 "$REPO_DIR/targets/cpro/scripts/run_hdc.py" \
  --omc "$OMC_DIR" --runner "$HIAI_RUNNER" --reference "$REFERENCE_JSON" \
  --output "$BOARD_JOB/benchmark"
cat "$BOARD_JOB/benchmark/result.json"
cat "$BOARD_JOB/benchmark/comparison.json"
```

已有完整 benchmark 时，可只重新比较，无需上板；`--report` 不能指向 execution-only 报告。

```bash
python3 -m qwen35_moe_dflash.cli compare \
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

默认成功回传并完成检查后清理本次板端目录，失败保留；加 `--board-cleanup none` 可保留全部内容，服务器结果始终保留。编译分区、能力查询和 Process 成功均不能单独证明实际 NPU 运行位置；验证范围见 [validation.md](../history/validation.md)。

## 融合算子调用探针

先按[环境配置](environment-setup.md)的顺序恢复当前DDK环境、填写剩余固定配置，再执行本节命令。入口会自动构建runner，不需先跑前面的手动runner构建步骤。

当前模型只保留GDN、CacheUpdate、Attention三类自定义算子探针，各prefill/decode，共六例。Conv状态处理和RMS使用基础算子路径。复用已存在的九例批次时：

```bash
python3 "$REPO_DIR/targets/cpro/scripts/run_fused_probes.py" --model-ops \
  --probe-run "$AI_RUN_DIR/fused-all-f6b81ead7ae8"
```

该模式只上传、执行和比较所选六例；旧批次中Conv/RMS即使曾编译成功，也不会执行。旧清单、脚本和失败记录保留。缺少的六例会单独编译，已有OMC复用；无需重新生成绑定或TBE诊断。

**仅首次开始或产物已删除时**，使用 `--model-ops --fresh` 重建六例所需的绑定、TBE诊断和AIR/OMC，然后上板。`--all-ops` 继续作为文档五类九例的可选完整诊断，不能与 `--model-ops` 同时使用。仅重试某个所选缺项，可附加 `--retry-cases`；Conv/RMS不属于模型模式的重试范围。

默认新建独立板端目录，按原上板脚本的规则收集构建对应的运行库、排除DDK HCL库对，并加入 `/vendor/lib64/chipsetsdk` 搜索路径。`BOARD_HIAI_LIB_DIR` 可显式复用已有私有库目录。运行库版本和固件标识自动查询；传输文件校验SHA256，执行失败仍尝试回收。

批量模式的 `report.txt` 按所选范围汇总六例或九例，并显示已有数值诊断；fresh批次另保留 `bindings.json`、`diagnosis/`，`probes/` 保存本次编译产物。`board-*/` 分组保留上板记录，其中 `results/` 为输出和日志，`comparison/report.txt` 为详细诊断。runner在同次批量运行中只构建一次。成功和失败均保留板端目录；脚本不安装或改写DDK，数值验收及NPU执行位置另行验证。

## tiny 回归

复用同一执行命令，只替换 OMC 和参考：PyTorch tiny 使用同次 `demo/export-tiny-air`；GE tiny 使用同次 prepared 的 `reference-tiny`。两种随机模型的参考不能混用，tiny 通过也不代表完整模型通过。
