# 当前阶段：把已测输入打包收益接入完整 Draft

2026-11-05：特征投影前缀对照已完成，计入CPU打包和SetInput后，每次净省 **4.93–5.55ms**。三组输入各十次输出与设备baseline逐字节一致，所有调用有限且重复稳定。对CPU参考的原始数值门槛仍为FAILED，两种前缀误差相同。用户已决定暂缓进一步精度排查，优先推进性能。

## 已测收益

| source case／valid | baseline均值 ms | packed均值 ms | 净省 ms |
|---|---:|---:|---:|
| 2／2 | 15.9736 | 10.9998 | 4.9738 |
| 16／16 | 16.0192 | 10.4716 | 5.5476 |
| 17／14 | 16.0485 | 11.1210 | 4.9275 |

三组等权均值省5.1496ms，约32.16%的前缀耗时。每组候选最慢调用仍快于baseline最快调用。计时为端侧预分配CPU打包＋SetInput＋同步Execute墙钟，不含加载和快照；这是独立前缀的结果，完整Draft和端到端收益尚未测得。[完整结果](results.md)。

## 完整候选

新增 `--cpu-feature-pack`，与 `--rope-fp32-table` 一起使用。原FP16逻辑输入 `[1,16,16384]` 在端侧CPU做字节重排，模型物理输入改为 `[1,1024,16,16]`。保留原Conv权重、属性、输出布局及全部14个模型输出。源export不改，新增独立候选。

fixture仍保存原行序字节；accuracy plan版本3明确要求打包，原版本1／2保持原行为。候选只允许绑定原RoPE baseline的回放，逐项核对原输入、golden、权重和语义槽位。生产resident流程尚未切换。vendor profiling工具暂拒绝直接使用这类逻辑字节fixture，避免错误喂入物理ABI。

主机验证覆盖真实TorchAir schema的合成多输入图、用户回传的CPU Data描述、原Data.index保留、共享布局拒绝、编译前处理、物理槽位重排、原golden保持、native字节打包、isolated／feedback回写和离线baseline对比。真实DDK／OMC和设备结果待服务器运行。

## 服务器步骤

沿用现有运行目录、DDK和环境变量。先编译一份完整候选：

```bash
cd "$REPO_DIR"
git pull --ff-only origin main
TOP1_DIR="$AI_RUN_DIR/draft/tests/top1-integration"
ROPE_DIR="$TOP1_DIR/rope-fp32"
FULL_PACK_DIR="$TOP1_DIR/feature-pack"

python3 "$REPO_DIR/targets/cpro/scripts/compile_incremental_draft.py" \
  --air-export "$TOP1_DIR/export" --rope-fp32-table --cpu-feature-pack \
  --ddk-root "$CPRO_DDK_ROOT" --ddk-id "$CPRO_DDK_ID" \
  --tbe-diagnosis "$AI_RUN_DIR/common/tbe-diagnosis" \
  --output "$FULL_PACK_DIR/compiled"
```

仅在返回 `PYTORCH_DRAFT_OMC_COMPILED` 后继续。若空间预检失败，未发生权重快照或OMG调用；释放已完成的重复副本后，用同一目录加 `--replace-failed` 重试。其他编译失败保留日志检查。

编译成功后，现有清理工具也覆盖这个固定候选目录，只删除已经嵌入OMC的外部权重重复副本，保留ABI绑定所需的源快照、prepared AIR、外部OMC描述和报告：

```bash
python3 "$REPO_DIR/targets/cpro/tools/clean_draft_probes.py" \
  --root "$AI_RUN_DIR" --config "$AI_RUN_DIR/runtime/runtime.json" \
  --integration-sidecars --apply

python3 "$REPO_DIR/targets/cpro/scripts/build_hiai.py" \
  --output "$AI_RUN_DIR/runtime/native" --replace
```

先用新runner重放现有RoPE baseline，使两边都记录输入准备耗时；不重编baseline、不生成新golden：

```bash
python3 "$REPO_DIR/targets/cpro/tools/validate_incremental_draft.py" \
  --replay "$ROPE_DIR/kv-reproduction/result" \
  --config "$AI_RUN_DIR/runtime/runtime.json" \
  --output "$FULL_PACK_DIR/baseline-run" --output-initialization untouched --apply
```

启动确认后回收；若尚未完成，继续回收同一目录，不重新启动：

```bash
python3 "$REPO_DIR/targets/cpro/tools/validate_incremental_draft.py" \
  --recover "$FULL_PACK_DIR/baseline-run" --output "$FULL_PACK_DIR/baseline-result"
```

baseline必须完成且锁已释放。数值状态预计仍为 `NUMERICAL_FAILED`，这与执行失败不同。随后运行完整packed候选。当前共享目录布局把原FP16准备目录保存在 `$AI_RUN_DIR/draft/source/prepared`，先给 `DRAFT_SOURCE` 赋值。验证器会核对其中 `prepared.json` 的哈希与原回放绑定一致；不要重新生成权重或参考数据。若使用其他目录布局，改为之前精度验证 `--source` 使用的目录。

`${DRAFT_SOURCE:?...}` 只检查变量，不会赋值；`git pull` 也不会设置当前shell的变量。下面将文件检查和验证命令放在同一个条件中，避免缺失源目录后仍继续启动。

```bash
DRAFT_SOURCE="$AI_RUN_DIR/draft/source/prepared"
if [ -f "$DRAFT_SOURCE/prepared.json" ]; then
  python3 "$REPO_DIR/targets/cpro/tools/validate_incremental_draft.py" \
    --replay "$FULL_PACK_DIR/baseline-result" \
    --compiled "$FULL_PACK_DIR/compiled" --source "$DRAFT_SOURCE" \
    --air-export "$TOP1_DIR/export" --ddk-root "$CPRO_DDK_ROOT" \
    --config "$AI_RUN_DIR/runtime/runtime.json" \
    --output "$FULL_PACK_DIR/run" --output-initialization untouched --apply
else
  printf '未找到原FP16准备文件：%s\n' "$DRAFT_SOURCE/prepared.json" >&2
fi
```

启动确认后回收，再做纯本地对比：

```bash
python3 "$REPO_DIR/targets/cpro/tools/validate_incremental_draft.py" \
  --recover "$FULL_PACK_DIR/run" --output "$FULL_PACK_DIR/result"

python3 "$REPO_DIR/targets/cpro/tools/validate_incremental_draft.py" \
  --compare-feature-pack "$FULL_PACK_DIR/result"
```

离线对比检查两种模式全部有效Top1／hidden以及完整KV bank，并报告 `baseline_equivalence`、原CPU参考失败数、KV误差、重复稳定性和 `prepared_timings`。完整调用的准备耗时采用分段求和：CPU打包＋SetInput／ReuseInput＋同步API，排除文件读取、诊断KV读回、快照、commit及加载；与前缀连续墙钟的计时口径有区别。

完整对照仍沿用原19例×2模式的诊断调度，没有额外预热十次，也不包含target、embedding查表和整轮调度。因此先据此判断是否值得接入resident，不能直接宣布端到端加速。

## 后续排序

| 顺序 | 方向 | 判断标准与成本 |
|---|---|---|
| 2 | 8.788ms的特征投影 | 复用约64MiB投影权重做少量前缀对照；保持完整点积和原舍入边界，先确认选用的表达／布局是否更快 |
| 3 | 19.807ms完整词表head | 覆盖每轮proposal、耗时最大，但逻辑FP16权重约1.017GB；先查实际布局和实现，不能因热点大就直接重写自定义内核 |
| 暂缓 | 零新增KV的专用proposal | 当前调度把最后一批追加与proposal合并，正常多轮生成很少满足valid=0；独立快路径收益覆盖有限 |
| 暂缓 | 继续改StableArgMax | 已测仅0.416ms，优先级低 |

当前head输出logits约7.45MB，远小于完整权重；单独融合Top1减少中间写回，不能据此推断能解决19.8ms。暂不以缩词表、量化或减少层数作为默认路线。没有匹配的X90实测能力画像，不给出带宽极限或硬件无优化空间结论。

零新增KV原型已保存在本次本地实验目录，未发布、未编译、未接入生产，不占服务器运行盘。若以后采用KV-only catchup＋proposal拆分，必须比较两次调用总耗时、缓存交接和模型常驻成本；测试中6/19例为零追加不代表实际请求频率。现阶段无需执行该原型的编译命令，保留当前已完成profiling和基线OMC即可。
