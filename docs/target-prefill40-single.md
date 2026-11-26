# 原 CPro 整网的 40 层 prefill 首 token

单层、四层、40 层整图和 40 层逐层的 prefill/decode 共用[统一运行入口](decoder-compare.md)。

## 原 CPro 的 40 层整图，使用 SiluPatch 和 SoftplusPatch

配置 `DECODER_MODE=40`、`DECODER_PHASE=prefill`，输入取 310P YAML 的第一条 prompt，物理长度为 64。四个角色的入口和两机组合命令见[运行说明](decoder-compare.md)。

运行目录为 `$AI_RUN_DIR/target/tests/40-prefill-t64`。golden 端生成 `export/reference/` 和各层 CPU 参考，AIR 端写入 `export/air/`，OMC 端编译到 `board/omc/`。上板后查看 `board/comparison.json` 和 `board/matrices.csv`。矩阵覆盖最后有效 token 的 logits、30 个线性层的完整 conv/recurrent 状态和 10 个 full 层的有效 K/V。失败的编译日志在 `board/compile/compile.log`，上板记录在 `board/replay/result.json`。

默认补丁检查 351 个 W4 matmul、30 个 GDR、140 个 FP16 SiluPatch 和 30 个 FP16 SoftplusPatch。默认 Conv2D 使用窗口 `[1,8192,1,67]` 与卷积核 `[8192,1,1,4]`。可选 `--gdn-conv taps` 从相同窗口取四项逐通道乘加，两种实现均把 `[1,8192,64]` 送入 SiluPatch。可用 `CPRO_PATCHED_SOURCE` 指定个人源码目录。

如果日志在 `UpdateArgsForAicore / OpUpdateKernelArgs failed` 后报 `GraphOp Prepare` 失败，HIAI 停在加载模型、初始化 AICore 任务参数的阶段。已有的 Conv2D 和 taps 两种 40 层 prefill 都出现了这一调用链。可在执行服务器直接从这次保存的回执和完整 HiLog 中提取属于该客户端的服务端线程上下文：

```bash
python3 targets/cpro/tools/split_stage_log.py --load-replay "${DECODER_RUN_ROOT:-${AI_RUN_DIR:?}/target/tests}/40-prefill-t64/board/replay"
```

在重复运行覆盖 `board/replay` 前查看这份记录。`SubGraph_0` 是子图名。要定位内部具体算子，需要服务端记录的任务编号、kernel 名或算子名以及对应的编译映射。

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
