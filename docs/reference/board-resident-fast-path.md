# 常驻快速模式：关闭诊断与中间张量文件

新建 `--resident` 请求且不加 `--dump`，准备清单将固定 `runtime_profile=compact-memory-v1`。旧 prepared／recovery 不会自动切换模式。原 prefill64、decode1、完整六层 draft S64 OMC 与权重不变，只重编 native runner。

## 已移除的开销

| 位置 | 快速模式行为 |
|---|---|
| 部署已有模型／权重／embedding 缓存 | 复用已拥有的缓存目录与版本标识，检查文件类型、大小；不重新读完整 payload 算 SHA256。新上传文件仍单次核验后发布。 |
| native 启动 | 大模型／表只检查大小，不再完整哈希；小 config／ABI 保留一次绑定。 |
| embedding 查表 | 两张原表各只读 mmap 一次，按 token 复制所需行，取消每行 open／stat／close。 |
| prefill→decode 与连续 decode | 80 份状态在内存中按既有映射移动；没有输入／输出 bin 临时文件、回读、逐文件哈希或临时目录清理。 |
| SDK 输入 | 保留必要 memcpy，关闭源字节 SHA256 和 SDK 输入读回 SHA256。 |
| SDK 缓冲区 | 关闭每次调用和每个输出复制时重复的全体缓冲区两两重叠扫描；加载时仍检查一次，运行后保留轻量尺寸／对象检查。 |
| draft 输出 | 只读回 15 个 top1；不复制、落盘或扫描未参与调度的 draft_hidden。OMC 本身仍计算其既有输出。 |
| 输出诊断 | 不扫描 FP16 feature／hidden 的有限性，不统计输出填充值；logit argmax 与 token 范围检查仍在实际消费时执行。 |
| 逐步日志 | 不写每轮 receipt、输入输出清单、逐 token progress，不构造被丢弃的 round 详情，也不逐输入打印阶段。保留加载阶段、最终完成／失败摘要。 |

`dump=true` 保留原详细检查与文件路径。快速模式不是零拷贝：当前原 ABI 仍需要 SDK 输入拷贝、target logits／features／80 状态读回，以及既有输出缓冲区初始化。它们没有被伪装成可以无条件删除的诊断操作。状态按已提交 token 前进，不执行被拒候选，也没有改变接受／纠正／bonus 调度。

快速模式不检测已缓存 payload 的同长度内容改写；摘要中的模型 SHA 是准备清单声明值，不是本轮新读出的哈希。结果明确记录 `startup_payload_validation=OWNED_CACHE_EXTENTS_NEW_UPLOAD_SHA256`。保持配置、ABI、返回的小摘要与锁归属验证；这些不在逐 token 热路径上。HDC 仍只部署、启动、最终回收，不参与中间决策。

## 计时

- `native_api_latency.prefill/decode/draft`：只围绕同步 `Run`／`Process` 调用计时，单位包含 `total_us`、`mean_ms`。这是 SDK 同步调用墙钟，包含 SDK 内部调度／传输，**不是 NPU kernel-only**。
- `phase_latency.decode/draft`：外层执行器调用墙钟，包含 SDK 输入拷贝、输出初始化／读回，排除模型加载／关闭；快速模式没有哈希或临时文件 I/O。
- `phase_latency.prefill`：一次 prefill 的加载、运行包装、关闭总墙钟。
- `supervisor_timing.total_wall_ms`：整个板端请求，包括启动与三模型加载。

11 token/s 的单 token 平均约 90.91 ms，必须先核对会议数据是否是同模型、同量化、同上下文与普通 decode 的稳定阶段。下次先比较 `native_api_latency.decode.mean_ms`，再看外层包装差额。用户已回传单次本版Run111.64ms（倒数约8.96次／秒），但draft加载202.958秒，整轮230.735秒。见[首次实机结果](../history/board-resident-fast-first-completion.md)。它不是同条件配对基准；接受率为零也不能证明DFlash算法加速。

## 在 DDK 主机执行

使用新输出目录，不覆盖之前成功请求，不带 `--dump`：

```bash
git -C "$REPO_DIR" pull --ff-only

python3 "$REPO_DIR/targets/cpro/scripts/run_board_serial_dflash.py" \
  --decode-job "$AI_RUN_DIR/target-pytorch-decode1-compile" \
  --runtime-probe "$AI_RUN_DIR/target-decode1-runtime-io-v2" \
  --state-lineage "$AI_RUN_DIR/target-decode1-state-lineage-v7" \
  --source-torchair-pb2 "$AI_RUN_DIR/target-decode1-row-response/ge_ir_by_protoc_3_19_pb2.py" \
  --draft-bundle "$AI_RUN_DIR/full-draft/residual-fix/bundle" \
  --weight-table "$AI_RUN_DIR/original-embedding-tables/qwen3_5_moe_embdedding_weight.bin" \
  --scale-table "$AI_RUN_DIR/original-embedding-tables/qwen3_5_moe_embdedding_scale.bin" \
  --prompt-input-ids /home/w00949577/run_35B/chunksize_64_data/input_ids.bin \
  --prefill-feedback "$AI_RUN_DIR/target-prefill64-decode1-feedback-plan-v2" \
  --prompt-length 8 --max-new-tokens 8 --max-proposals-per-round 2 \
  --resident --apply --detach \
  --output "$AI_RUN_DIR/dflash-board-resident-fast8-run"
```

同一请求完成后回收：

```bash
python3 "$REPO_DIR/targets/cpro/scripts/run_board_serial_dflash.py" \
  --recover "$AI_RUN_DIR/dflash-board-resident-fast8-run" \
  --output "$AI_RUN_DIR/dflash-board-resident-fast8-recovery"
```

检查最终 token、接受率、三个 `native_api_latency`、`phase_latency`、总墙钟，以及两个 resident 模型仍各加载／关闭一次。首次快速模式已由用户执行，结果见[实机记录](../history/board-resident-fast-first-completion.md)；不要重复覆盖这两个已完成目录。主机 fake SDK 验证覆盖真实 C++ 调度、80 状态映射、输出一致性、失败关闭和同 attempt 回收，不证明模型精度或实机性能。

主线仍需解决先前新旧运行第 2 个 token 的差异；本次按用户优先级先清除运行框架开销，没有把该数值问题标记为已解决。

主机验证已完成：192 个单元场景、13 个真实 C++／fake SDK 混合循环场景，以及 18 组 native 回归。最终快速路径的 6 个成功／失败场景再次通过。测试覆盖状态映射、只生成小摘要、旧请求恢复和失败关闭；详细哈希与范围见[验证记录](../../specs/board-resident-fast-host-validation.json)。
