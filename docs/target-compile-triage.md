# 剩余 38 项 OMC：对照原 Target 定位编译失败

更新：2026-11-10。`environment-recheck-001` 已有 120 项编译成功并回传数值摘要；剩余普通 Norm 20、ConvState 12、Conv1d 6。当前先读取这一批失败日志和实际 AIR，按共同失败原因归并。已成功的 OMC、上板输出和原始 golden 直接保留。

## 原 Target 给出的基线

本次实际阅读了此前保留的 master Target 源码。`models/modeling_qwen3_5_moe_hiai_cpro.py` 的 SHA256 为 `bd716718818b6b152ece5a2e345a5b674f091f53ba82059c86cd37060511f07d`，`export_glm_model_qwen3_5_moe.py` 为 `7ef1710a301b88a8f9f8201f95b97b62d22ac80f43a1c5bfffe710e1b39d21f6`。这是保留版本，尚不能证明与 35 服务器当前源码相同；本批 `export.json.source_files` 可用于核对真正参与导出的文件。私有源码不复制到仓库。

| 失败组 | 原 Target 实际调用 | 小图需要核对的差异 |
|---|---|---|
| 普通 Norm，20 项 | `npu_rms_norm(hidden_states, weight + 1.0, eps)[0]`，随后转回输入 dtype | 原图调用和权重语义保持一致。已有小图修正是在导出前按捕获权重 dtype 计算 gamma，避免在编译时计算 Add；检查本次 AIR 是否真的包含此修正 |
| Conv1d，6 项 | `F.conv1d(padded, weight, groups=conv_dim)`；`padded` 由 ConvState 的第二输出直接产生 | 独立小图的 padded 是捕获的三维 Data。旧 AIR 的 Conv2D 输入三维、filter 四维；已有修正用显式升维／降维表达相同卷积，需检查新的 AIR 结构 |
| ConvState，12 项 | `linear_attention_convstate_update(mixed_qkv, conv_state, linear_update_tensor)` 返回两项；第一项接 `conv_state.copy_`，第二项接卷积 | 单输出探针只接出选定输出，状态是普通 Data。先核对当前实现注册和格式支持，再判断是否需要保留原双输出、消费者及状态写回的连续子图 |

普通 Norm 的实现以实际 forward 为准。原导出脚本虽然注册了 `adn_rms_norm → AdnRmsNorm`，普通 Norm 调用的是 `npu_rms_norm`；不能据此直接换算子。已成功的 gated Norm 使用 FP32 输入和直接 gamma，也不能替代普通 Norm 的验证。

原 ConvState fake 实现返回 `[1,8192,4]` 和 `[1,8192,T+3]`，converter 输出端口为 `y1/y2`。历史失败日志中，DDK 将第二输出推成 `[1,8192,1,T+3]`。这使“原图直接连接卷积”和“小图以三维 Data 为卷积输入”成为值得核对的差异，但当前没有新日志证明它就是这 12 项的失败原因。也不能只修改 golden 的 shape 来消除差异。

## 原成功命令应如何借鉴

用户提供的四层 T64 成功命令使用 `source set_ascendc_env.sh`，随后调用外层 `tools/tools_omg/omg`，参数为 `--framework 1 --limit_partition_size=false --save_weights_as_external_data=true`。40 层命令还带有原图输入名、RefData shape，以及 `--dynamic_dims="64,64,64,64,64;1,1,1,1,1"`。

| 对照项 | 核对方法 |
|---|---|
| wrapper 与直接 master/omg | wrapper 除选择二进制外还配置 TBE／AscendCPP 库、`CCE_LIB_DIR`、`KIRIN_CONF_JSON_PATH` 和工具路径。与失败任务保留的 `omg-environment.json` 比较，不能仅凭二进制路径不同认定根因 |
| 编译选项 | 当前小图显式使用 `--target=omc --platform=kirinx90`、内嵌权重和 `use_origin_format=true`；原成功命令省略目标／平台并使用外置权重。先以实际日志确认目标与实现选择，不再轮流盲试开关 |
| shape 和状态 | 原示例为 T64，40 层命令含 T64/T1 档位及 RefData；本批历史 Prefill 是静态 T256。不能把原图输入名／dynamic_dims 直接复制到已经固化 reshape 的小图 |
| 导出与转换阶段 | 原脚本的 `torchair.dynamo_export(dynamic=False, ...)` 与 OMG 的动态档位属于不同阶段；分别检查 AIR 和编译命令 |
| 上板入口 | 用户提供的是 `run_model_prefill.sh -m qwen3_5_35B/qwen3_5_35B.omc`。本地没有该脚本正文，不能据此猜测具体输入顺序或 ABI；本次编译定位尚不需要上板 |

35 是当前 DDK／OMG 服务器；310P golden 仍来自此前权重服务器。147 和历史命令中的安装路径只作为来源记录。

## 现在执行一次只读检查

沿用 35 上本批的 `REPO_DIR`、`RECHECK`、`OMC_BATCH` 和修正后的 `CPRO_DDK_ROOT`：

```bash
: "${REPO_DIR:?}" "${RECHECK:?}" "${OMC_BATCH:?}" "${CPRO_DDK_ROOT:?}"
git -C "$REPO_DIR" pull --ff-only origin main
export COMPILE_INSPECT="$RECHECK/compile-inspect-001"

python "$REPO_DIR/targets/cpro/tools/target_layer_probe.py" inspect-compile \
  --batch "$OMC_BATCH" --failed \
  --ddk-root "$CPRO_DDK_ROOT" --compiler-entry \
  --output "$COMPILE_INSPECT"

cat "$COMPILE_INSPECT/summary.txt"
```

输出目录若已存在，换一个编号。`--failed` 自动选择本批全部 38 个 FAILED case，只读取其现有证据；整个检查不执行导出、OMG 或设备命令。`--compiler-entry` 只读取入口脚本和环境记录，不 source 脚本。

先回传 `summary.txt` 即可。它列出每个失败边界的导出修正标记、AIR 算子类型、退出阶段与关键错误摘要；错误检索覆盖日志前部，避免只看末尾退出信息。完整 `report.json` 保留源码身份、图描述、环境、错误上下文和 IR 清单。日志扫描有 64 MiB 上限；摘要是定位线索，不自动认定根因。检查结果为 PARTIAL 时先看缺失字段，它不表示又进行了一次失败编译。

## 根据检查结果只重试必要范围

本轮 `manifest_sha256=9f433d0092c93f5c156a5c6ab59d6d603dcef338b8132dc28b1e0f77fcf1b1e2`、`source_batch=../../../air-batch`，与之前 AIR 批次相同。更新 Python 代码再执行 `compile` 不会更新既有 AIR。先查以下证据：

| 组别 | 应查的记录 | 下一步 |
|---|---|---|
| Norm | `FOLD_ZERO_CENTERED_GAMMA`；`export_graph_check`；RmsNorm 的 gamma 是否为常量、是否仍有 Add | 若尚未应用修正，复用已有 capture 重导；若已应用，按本轮实际失败节点继续定位，不重复旧修复 |
| Conv1d | `EXPLICIT_CONV2D_SINGLETON_HEIGHT`；Conv2D 输入和 filter 是否为四维、单例高度是否明确 | 若仍是旧三维输入，重导此类；若结构已经正确，再检查 dtype、groups 和所选实现的实际拒绝原因 |
| ConvState | 当前 `InferShape`、`CheckSupport`、kernel store／format config；两个输出端口的实际描述 | 先判断是否仍是实现选择问题。只有这部分已核对、且失败指向独立图边界时，才补保留原连接关系的最小连续子图 |

旧批次若没有 `export_adjustments`，这只能说明没有记录；仍以 AIR 节点与结构检查为准。历史 `inputTensorPtr is NULL`、Conv2D rank mismatch、ConvState implementation/format config 错误是检索线索，当前只有 exit 1 时不能直接沿用这些根因。

需要重导时，已有 [`export --kinds norm conv`](target-layer-accuracy.md#仅重导重编普通-norm-和-conv1d) 可以仅生成这 26 项，复用原 capture，不重跑 310P 四层推理。先编译同一失败类别的代表项：普通 hidden Norm、Q/K Norm 的不同输入 rank、Conv 的 Prefill/Decode；代表项越过原失败点后再补齐该组。若新日志显示不同错误则拆组，不把同名算子默认归为同一原因。

当前主机已通过 18 项检查，覆盖 FAILED 自动选择、已有元数据和哈希约束、只读行为、导出修正记录及日志前部错误提取。这里只验证诊断工具；尚无本轮 38 项的新 DDK 成功证据。数值异常与已完成项继续见[状态表](target-accuracy-status.md)。
