# 原 CPro 整网的 40 层 prefill 首 token

## 原 CPro 的 40 层整图，使用 SiluPatch 和 SoftplusPatch

78 上沿用 `TARGET_SOURCE`、`CPRO_SOURCE` 和 `AI_RUN_DIR`。在仓库根目录执行：

```bash
git pull --ff-only origin main && bash targets/cpro/scripts/decoder4_t64.sh aligned40-air
```

脚本采集 310P 参考，逐层计算同一提示词的 CPU W4 参考，在私有副本上补丁原 CPro 导出器。原 wrapper 静态导出 L0–L39、最终 norm 和词表 head，S=64，有效长度 19，读出位置 18。导出检查 351 个 W4 matmul、30 个 GDR、140 个 SiluPatch 和 30 个 SoftplusPatch。两种激活都用 FP16 输入输出，在 GDN 门控和衰减乘法前按需恢复 FP32。沿用 W4 scale、RoPE 和 MoE 修复。卷积窗口取拼接历史的末 67 行，下一步卷积状态取末 4 行。改动可在运行目录的 `silu-source-*/source.diff` 查看。

35 上沿用已装两种 FP16 算子的 DDK、SSH 和板端配置，在仓库根目录执行：

```bash
git pull --ff-only origin main && bash targets/cpro/scripts/decoder4_t64.sh aligned40-board
```

默认目录为 `$AI_RUN_DIR/target/tests/prefill40-original-aligned-silu-softplus-t64-valid19-01`。两机可设置同一绝对路径 `ALIGNED40_RUN`。有完整 310P 参考时，78 可将 `ALIGNED40_CAPTURE` 指向含 40 层状态和最终 logits 的采集目录。

35 只拉取 AIR 包，使用外置权重编译一个 OMC。下载前和编译前均显示 AIR 字节数和目标文件系统可用空间。OMG 直接引用拉到本机的 AIR 权重。空间估算按输出权重与输入权重同量级计算，编译临时文件的峰值需在 35 实测。板端单独检查 OMC、权重、输入和输出所需空间。

上板成功后，`replay-*/comparison.json` 包含三端的首 token ID、文本及 CPro 对 310P 的判定。`replay-*/matrices.csv` 是 81 个输出的三端矩阵，包含 token18 的 logits、30 层的完整 conv/recurrent 状态，以及 10 层的 K/V cache。K/V 比较有效 19 行并检查物理写入页之外的缓存。加载或运行失败时，查看同次 replay 下的 `result.json`。

## 既有的单 logits 导出流程


用同一条 19-token 提示词，运行一次静态 T64 prefill。输入为实际的 input_ids、位置、mask 和初始状态。原 CPro wrapper 连续执行 embedding、40 个带残差和 MoE 的 decoder、末端 norm 和词表投影。图输出为位置 18 的一行全词表 logits。主机用原模型 tokenizer 的单 token 映射，显示 CPro 和 310P 的贪心首 token。

## 310P 采集和 AIR 导出

在 AIR 服务器的 Bash 中使用之前的 `REPO_DIR`、`AI_RUN_DIR`、`TARGET_SOURCE`（310P 工程）、`CPRO_SOURCE`（原 CPro 工程）。两个环境都使用该模型的同一份 W4 checkpoint。以下命令从头开始，使用一个新的运行目录。

```bash
(
set -e
conda activate qwen3_5_w4
git -C "$REPO_DIR" pull --ff-only origin main
R="$AI_RUN_DIR/target/tests/prefill40-single-t64-valid19-v1024-qlate-01"
P="$REPO_DIR/targets/cpro/tools/prefill_40_single.py"
python "$P" capture --run "$R" --source-root "${TARGET_SOURCE:?}" \
  --config "${NZ_CONFIG:-config/qwen3_5_moe.yaml}"

conda activate qwen3_5_cpro
python "$P" air --run "$R" --source-root "${CPRO_SOURCE:?}" \
  --config "${TARGET_CONFIG:-config/qwen3_5_moe.yaml}"
du -sh "$R/air"
)
```

原有 40 层 310P 参考目录含完整 `manifest.json`、`logits.bin` 和 `token-text.json` 时，可以跳过 capture，给 air 命令追加 `--reference "$AI_RUN_DIR/target/tests/prefill-40l-t64-valid19-v1024-01/reference"`。导出器从私有源码暂存副本加载原 wrapper、模型、导出器和量化加载器。沿用先前的 CPro 图适配，保留 chunk size 64。脚本在发布前核对 30 个 GDR、10 个 full attention、长度控制边、权重和单 logits 输出，并为 AIR 传输目录设置 0777。

## 拉取、编译和执行

在 OMC 服务器的 Bash 中使用已安装本次算子的 `CPRO_DDK_ROOT`、已有 SSH 传输配置和板端连接。OMG 同时需要 AIR 权重、编译输出权重和临时空间。用上一步的实际 AIR 大小对照本机可用空间。权重文件随一个 OMC 一起部署。整个过程调用一次 HIAI V2。

```bash
(
set -e
conda activate custom_pypto
git -C "$REPO_DIR" pull --ff-only origin main
R="$AI_RUN_DIR/target/tests/prefill40-single-t64-valid19-v1024-qlate-01"
P="$REPO_DIR/targets/cpro/tools/prefill_40_single.py"
df -h "$AI_RUN_DIR"
python "$REPO_DIR/targets/cpro/scripts/pull_artifacts.py" --from air "$R/air"
python "$P" compile --run "$R" --ddk-root "${CPRO_DDK_ROOT:?}" \
  --gdr-source "${GDR_SOURCE:?设置为本次 allops 的 op_kernel/chunk_gated_delta_rule.cpp}" \
  --tbe-diagnosis "${TBE_DIAGNOSIS:-$AI_RUN_DIR/common/tbe-diagnosis}"

export SN="${SN:?设置为目标板序列号}" BIP="${BIP:?设置为 HDC 服务器的地址和端口}"
export HDC="${HDC:-hdc}"
python "$P" board --run "$R"
cat "$R/board/first-token.json"
)
```

编译前后都校验两份 DDK 算子源码和 allops 源码的 SHA-256。确认 allops 文件是预期的 vscale=1024、Q 缩放后移版本。编译日志在 `omc/compile.log`。板端完整 logits 在 `board/logits.bin`，首 token、解码文本、五项精度指标和逐元素超差比例在 `board/first-token.json`。`top1_match` 表示该提示词的首 token ID 与 310P 一致。连续生成多个 token 还需要 decode。
