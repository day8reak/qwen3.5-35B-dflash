# 板端文字输入输出

入口使用原 Qwen3.5-35B-A3B tokenizer，板端执行：

`UTF-8文字 → 聊天模板 → NFC/Unicode分词/BPE → 原prefill一次 → 常驻decode/draft循环 → 文本解码 → 最终结果`

服务器准备请求时用原版 Hugging Face tokenizer 做一次长度检查，附带预期 ID；板端从文字重新编码并逐项对照后才加载模型。运行期间没有逐token服务器请求，最终文字也是板端解码的，服务器回收后做一次只读对照。板端不依赖 Python、Rust 或额外动态 tokenizer 库。

这是文字接口的实现；原模型数值、接受率、控制语义仍未验收。模型目前可能输出标点或无意义文本，不能通过 tokenizer 修复。

## 1. 准备原 tokenizer（只做一次）

```bash
git -C "$REPO_DIR" pull --ff-only

python3 "$REPO_DIR/targets/cpro/scripts/prepare_board_tokenizer.py" \
  --output "$AI_RUN_DIR/original-tokenizer"
```

该命令只下载三个公开文本配置文件，约12.8MB，不下载模型权重。来源固定为仓库原模型的revision `59d61f3ce65a6d9863b86d2e96597125219dc754`，每个文件校验见 `specs/tokenizer.lock.json`。生成约5.8MB的板端二进制词表。命令输出的 `source_dir` 即下一步 `--tokenizer-dir`。

如果DDK主机不能访问Hugging Face，把同revision的 `tokenizer.json`、`tokenizer_config.json`、`chat_template.jinja` 三个文件放到一个目录，通过 `--source-dir /实际目录` 指定；此时下一步也使用这个实际目录。不要使用4B或其他模型的词表替代。

准备和最终结果对照需要主机当前Python环境可导入 `tokenizers`（本次对照版本0.22.2）。不需要在板端安装它。原有token文件请求和历史恢复无需这个包。

## 2. 运行文字版完整闭环

```bash
python3 "$REPO_DIR/targets/cpro/scripts/run_board_serial_dflash.py" \
  --decode-job "$AI_RUN_DIR/target-pytorch-decode1-compile" \
  --runtime-probe "$AI_RUN_DIR/target-decode1-runtime-io-v2" \
  --state-lineage "$AI_RUN_DIR/target-decode1-state-lineage-v7" \
  --source-torchair-pb2 "$AI_RUN_DIR/target-decode1-row-response/ge_ir_by_protoc_3_19_pb2.py" \
  --draft-bundle "$AI_RUN_DIR/full-draft/residual-fix/bundle" \
  --weight-table "$AI_RUN_DIR/original-embedding-tables/qwen3_5_moe_embdedding_weight.bin" \
  --scale-table "$AI_RUN_DIR/original-embedding-tables/qwen3_5_moe_embdedding_scale.bin" \
  --prefill-feedback "$AI_RUN_DIR/target-prefill64-decode1-feedback-plan-v2" \
  --tokenizer-dir "$AI_RUN_DIR/original-tokenizer/source" \
  --text '你好，请介绍一下自己。' \
  --max-new-tokens 8 --max-proposals-per-round 2 \
  --resident --apply --detach \
  --output "$AI_RUN_DIR/dflash-board-text8-run"
```

原有OMC和模型权重直接复用；自动重编的是包含tokenizer的板端C/C++程序。默认 `dump=false`，中间张量仍在内存中，回收最终文字、token ID、接受率、prefill/decode/draft时延及加载分段计时。新词表和prompt在启动前一次下发，不增加逐token文件I/O或HDC调用。

默认 `--prompt-format chat`：原模板的单条user文本、`add_generation_prompt=true`、`enable_thinking=false`，支持中文及UTF-8文本。多行或包含shell特殊字符时，用 `--text-file /实际路径/prompt.txt` 替换 `--text`。需要原样补全文本时加 `--prompt-format raw`。当前不提供多轮messages、工具调用和图片输入模板。

不再传 `--prompt-input-ids` 或 `--prompt-length`，完整文字的token数自动计算。当前原S64 draft要求 **模板后的输入token数＋max-new-tokens≤64**；超长明确报错，不静默截断。这个限制来自现有模型，增加tokenizer并没有扩大上下文。

保持原固定生成预算和至少两轮draft的执行实验策略，**尚不按EOS提前停止**；`max-new-tokens`目前至少3，默认8。后续产品化应独立实现EOS停止和多轮会话，不把它们混入当前已有执行验收。

## 3. 回收文字和时延

```bash
python3 "$REPO_DIR/targets/cpro/scripts/run_board_serial_dflash.py" \
  --recover "$AI_RUN_DIR/dflash-board-text8-run" \
  --output "$AI_RUN_DIR/dflash-board-text8-recovery"
```

摘要增加：

- `generated_text`：跳过tokenizer标记为special的token后的文字。
- `generated_text_with_special_tokens`：保留特殊token，便于排查。
- `text_io`：实际板端编码／解码、词表和文字摘要、输入token数，以及词表之外的输出ID。
- `text_latency`：板端tokenizer加载＋编码、最终解码的CPU墙钟微秒。

248320维输出头比tokenizer实际248070个条目大。若模型输出无词条的padding ID，显示 `�`，并在 `unmapped_generated_token_ids` 明确列出，不能默默丢掉。截断在一个UTF-8字符中间的输出采用原版的替换字符策略。实际token ID始终保留。

## 实现和验证范围

`board_tokenizer.py`校验锁定原始文件并导出原词表／merge顺序；`tokenizer.cpp`在板端执行NFC、原Unicode正则、byte-level BPE及特殊token处理。Oniguruma6.9.10和utf8proc2.11.0源码及许可证随仓库保留并静态链接，来源和逐文件SHA在`targets/cpro/runtime/third_party/sources.lock.json`；DDK构建不在线下载依赖。

测试使用真实原词表和主机编译的同一C++ tokenizer，对照HF编码、解码和原Jinja模板。C++完整prefill/draft/decode/回收测试使用fake HIAI，只验证接口与调度，不宣称实机推理、语言质量或NPU性能通过。不同Unicode实现版本的边界由每次板端编码与原版admission ID的相等检查覆盖；不相等时在模型加载前失败。

主机复验需要设置 `QWEN35_TOKENIZER_SOURCE_DIR`、`QWEN35_TOKENIZER_TEST_BINARY`，以及混合fixture二进制路径 `QWEN35_RESIDENT_MIXED_TEST_BINARY` 和其 `QWEN35_RESIDENT_MIXED_TEST_BINARY_SHA256`；运行 `pytest tests/test_board_tokenizer.py`。不设置这些显式路径时资源测试跳过，不自动下载或触碰设备。

验证记录见[结构化主机验证](../specs/board-text-host-validation.json)：2670组编码、6066组解码对照无差异；原准备／传输176项回归通过，文字及原生循环16项通过，扩展文字下发4项通过（其中2项与前者传输回归重叠）。实机文字流程待用户运行。
