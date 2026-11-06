# 当前阶段：以约60ms的packed Draft为基线继续优化

2026-11-05：完整模型对照的488个区域逐字节一致，覆盖isolated／feedback原两序列的有效Top1、hidden和完整KV。38次调用按次数加权，准备＋API从66.4573ms降到60.7651ms，减少5.6922ms（8.565%）。用户已选择这份packed模型作为后续性能基线，继续暂缓精度排查；原数值状态仍为FAILED。[完整结果](results.md)。

| 模式／阶段 | baseline ms | packed ms | 净省 ms |
|---|---:|---:|---:|
| isolated catchup，11次 | 66.6092 | 60.9815 | 5.6276 |
| isolated proposal，8次 | 66.3970 | 61.2258 | 5.1713 |
| feedback catchup，11次 | 66.4160 | 60.1885 | 6.2275 |
| feedback proposal，8次 | 66.3655 | 60.7998 | 5.5657 |

历史基线OMC为 `draft/tests/top1-integration/feature-pack/compiled/model.omc`，SHA256 `1105ce436ed4dd889ce4c26d0a5eeb620382b51ec5c98d8b810e3875af37600b`；用户现已授权删除，包括其测试结果，清理后须从权重重建。计时包含端侧CPU打包及输入设置／复用和同步API，排除文件I/O、诊断回读、快照、commit、加载；不是整轮端到端或预热后十次重复的基准。常驻入口仍为旧token-ID接口，不能仅改OMC路径就使用host embedding＋packed候选。

## 2×2投影实验结束，保留当前基线

2026-11-06用户回传：六组执行／回收成功、锁已释放，输出有限且重复稳定。每组预热3次、记录10次，30对设备输出均不满足逐字节等价；原CPU参考门槛也仍失败。

| source case | 当前4×4准备＋API ms | 2×2准备＋API ms | 净省 ms |
|---|---:|---:|---:|
| 2 | 10.5075 | 10.8774 | −0.3699 |
| 16 | 10.8752 | 10.9697 | −0.0945 |
| 17 | 10.9199 | 10.9672 | −0.0473 |

三组平均10.7675→10.9381ms，观测总耗时增加0.1706ms。API平均省0.1761ms，CPU打包平均增加0.3623ms；测量范围重叠，不据此声称统计显著变慢，但没有值得整图集成的净收益。2×2对CPU参考的L2略低，不能把逐字节差异说成精度恶化，也不据此推翻当前不集成的决定。停止这条路线，不重跑或编译完整2×2 Draft。[详细结果](results.md)。

## 下一步：全部静态矩阵权重的离线布局

2026-11-06回传已确认：源AIR的head权重是FP16 `[248320,2048]`；编译输入已经删除权重Transpose，以 `transpose_x2=true` 使用。报告PARTIAL只表示没有保留的编译IR，不能据此推断最终内核的权重布局。

用户要求统一检查全部权重。按当前packed源码，矩阵权重用途如下；下面是本轮离线打包覆盖范围；编译器导入与板端验证尚待执行。

| 用途 | 数量 | 原权重shape（输出×输入） | 布局检查范围 |
|---|---:|---|---|
| Q投影 | 6 | [4096,2048] | MatMul右矩阵 |
| 合并K/V投影 | 6 | [2048,2048] | 保留K后V的拼接顺序 |
| O投影 | 6 | [2048,4096] | MatMul右矩阵 |
| 合并gate/up投影 | 6 | [12288,2048] | 保留gate后up的拼接顺序 |
| down投影 | 6 | [2048,6144] | MatMul右矩阵 |
| 全词表head | 1 | [248320,2048] | MatMul右矩阵 |
| feature投影 | 1 | [2048,16384] | 当前Conv2D按[2048,1024,4,4]使用，单独核对卷积权重格式 |

共31个常量权重MatMul，FP16权重合计1642MiB；Conv另64MiB。26个归一化向量和host embedding行查表不套用矩阵分块格式。QKᵀ、PV的两个操作数来自本次运行，没有静态权重可离线打包。head检查器增加了同一文件路径的全部FileConstant别名及视图消费者审计，范围仅为当前源图，不推断其它模型是否共享该文件。

用户明确的Zn表示“大块行优先、小块列优先”，符合[华为分形格式说明](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/programug/Ascendcopdevg/docs/en/guide/technical_appendix/concepts_and_terms/neural_networks_and_operators/data_layout_format.md)。对逻辑右矩阵 `B=W.T`，匹配的16×16分块下，`Zn(B[K,N])` 与 `Nz(W[N,K])` 都按 `[K1,N1,N0,K0]` 存放，逐字相同；本地4组索引检查通过。这是分块数学关系，不是X90外部导入能力证明，不能直接修改格式标签或把普通ND转置当作Zn打包。

用户进一步确认CPro CCE的mmad约定为L0A的A使用ZZ、L0B的B使用ZN、L0C的C使用NZ。指令级右矩阵格式以此为依据；当前尚需核对所选DDK 100.520.020.010的FP16 FileConstant导入、GM到L0B搬运、逻辑／物理shape、padding及transpose语义，并确认编译器是否已经预打包。公开[PyPTO MatMul文档](https://pypto.gitcode.com/api/tensor_api/operation/pypto-matmul.html)列出KirinX90不支持TILEOP_NZ输入；这是PyPTO接口限制，不能据此断言OMG内置MatMul同样不支持，也不否定mmad的内部ZN约定。先前的ND head原型没有发布。本轮改为统一实现31个静态MatMul的位保持布局转换，并先运行下面的格式／transpose矩阵；尚未确定可用参数，也未声称加速。

### 当前先清空全部Draft测试

用户最新授权删除 `draft/tests` 的全部内容，包括60ms基线、AIR副本、所有测试输入输出和历史结果，只保留测试目录外的源权重和代码。60ms是历史观测值；首轮服务器清理在删除前失败，尚无空间释放结果。

```bash
(
set -eu
: "${REPO_DIR:?}" "${AI_RUN_DIR:?}"
cd "$REPO_DIR"
git pull --ff-only origin main
python3 targets/cpro/tools/clean_draft_probes.py \
  --root "$AI_RUN_DIR" --all-tests --apply
df -h "$AI_RUN_DIR"
)
```

工具优先检查 `draft/source/prepared`，否则在运行根目录内查找测试目录外的 `prepared.json`；核验完整模型拓扑、必需FP16权重集合、shape、大小和SHA256后再删除。无需预设 `DRAFT_SOURCE`。源权重在运行目录外时可用 `--source` 指定其prepared目录；如果源本身还依赖tests，工具停止，须先将源移到测试目录外。实际占用进程或持有的写锁会阻止删除；旧精度失败或历史结果缺失不阻止本次明确授权的清理。不归档旧结果，不访问板端，不改runtime配置或其它角色目录。返回 `free_bytes_after` 是实测空闲量。

清空后原来的replay及依赖旧输入的探针命令不能直接运行。本轮布局探针使用确定性合成权重与已安装TorchAir的schema，不依赖旧tests、旧AIR或DRAFT_SOURCE；整图阶段再从保留的prepared权重重建。清理入口此前9项主机检查通过，包含源权重保护、锁、软链接、不完整删除后重试。

2026-11-06服务器首次清理在删除前扫描时遇到 `stable-arg-max-ge-tbe/.../.ipc-*/listener-*` 条目而停止。修复后的 `--all-tests` 允许已关闭的Unix socket，读取本地 `/proc/net/unix` 检查是否仍被绑定，并在删除前再次检查；不连接socket，不放宽其它清理模式。新增9项本地回归检查通过，覆盖已关闭／活跃socket、删除前重新绑定、源权重损坏、锁及重复执行。拉取更新后重跑上方同一条命令；尚未收到修复后的服务器结果。

### 本轮：一次确定格式与transpose参数

2026-11-06本地实现，服务器编译／板端执行均未运行。整套为6种设置×9组尺寸，共54项；每个可编译且在NPU任务中的候选预热3次、保留10次输出。当前机器没有选定的CPro DDK／板端连接，由用户在原服务器执行。

| 变体 | FileConstant导入格式 | 逻辑右输入 | transpose_x2 | 原FP16字节变换 |
|---|---|---|---|---|
| nd_t | ND | W[N,K] | true | ND基线 |
| nd_n | ND | W.T[K,N] | false | 普通离线转置 |
| nz_t | FRACTAL_NZ | W[N,K] | true | [K1,N1,N0,K0] |
| nz_n | FRACTAL_NZ | W.T[K,N] | false | [N1,K1,K0,N0] |
| zn_t | FRACTAL_ZN | W[N,K] | true | [N1,K1,K0,N0] |
| zn_n | FRACTAL_ZN | W.T[K,N] | false | [K1,N1,N0,K0] |

全部验证同一个 `X @ W.T`。NZ使用GE格式编号29；ZN以字面格式名试验，不猜测整数编号，不替换成卷积FRACTAL_Z或FRACTAL_ZN_RNN。当前尝试的导入合同是FileConstant的shape和消费端shape均声明物理尺寸，origin_shape声明逻辑尺寸。这些设置需要设备验证；编译拒绝只否定该接口表达，不能否定mmad内部ZN格式。

九组W shape／M：`48×80 / 95`、`80×48 / 1`、`48×2048 / 15`、`48×4096 / 16`、`48×6144 / 32`、`4096×32 / 15`、`2048×48 / 16`、`12288×32 / 32`、`248320×16 / 15`。覆盖非方阵、完整K长度、各算子族N宽度、尾块、对齐块和词表末尾。大K与大N分轴测试，不称为真实head的完整尺寸验证；48×80一组用basis rows逐项暴露每个权重，其它组增加分块边界脉冲、全1、交错符号、零行和混合点积。整数参考限制在FP16可精确表达范围，逐字节比较全部输出。

编译失败的组合保留日志，剩余可编译组合一次上板。结果中的 `offline_weight_layouts` 分开列出编译、导入就绪、数值、重复稳定性及API均值；`passing_variants` 必须覆盖全部九组。探针计时只用于发现异常，不用于推算完整Draft收益。

首轮在上面清理完成后运行，后续回收时不要重复清理：

```bash
(
set -eu
: "${REPO_DIR:?}" "${AI_RUN_DIR:?}" "${CPRO_DDK_ROOT:?}" "${CPRO_DDK_ID:?}"
cd "$REPO_DIR"
python3 targets/cpro/tools/probe_draft_offline_weights.py \
  --ddk-root "$CPRO_DDK_ROOT" --ddk-id "$CPRO_DDK_ID" \
  --tbe-diagnosis "$AI_RUN_DIR/common/tbe-diagnosis" \
  --output "$AI_RUN_DIR/draft/tests/offline-weights/compiled"
)
```

若TorchAir schema自动发现存在歧义，使用已有安装中的 `ge_ir_by_protoc_*_pb2.py` 作为 `--torchair-pb2`；工具只读取其字面descriptor，不执行该模块。首次构建失败可在同一目录加 `--replace-failed`，不创建时间戳目录。约59.43MiB合成权重覆盖全部54项；最大全部成功时10次输出约482.93MiB。预检查同时预留编译工作区和回收归档／解包空间，当前下界约3.9GiB；实际编译scratch仍可能更多。

编译报告COMPILED或PARTIAL且普通ND基线九组都通过时：

```bash
(
set -eu
cd "${REPO_DIR:?}"
python3 targets/cpro/tools/validate_draft_vector.py \
  --compiled "${AI_RUN_DIR:?}/draft/tests/offline-weights/compiled" \
  --config "$AI_RUN_DIR/runtime/runtime.json" --ddk-root "${CPRO_DDK_ROOT:?}" \
  --output "$AI_RUN_DIR/draft/tests/offline-weights/run" --apply
)
```

启动后回收同一个请求；若返回PENDING，继续使用这一条回收命令，不重新 `--apply`：

```bash
(
set -eu
cd "${REPO_DIR:?}"
python3 targets/cpro/tools/validate_draft_vector.py \
  --recover "${AI_RUN_DIR:?}/draft/tests/offline-weights/run" \
  --output "$AI_RUN_DIR/draft/tests/offline-weights/result"
)
```

整图编译接口已接入 `--offline-weights <变体> --offline-weight-canary <上述result目录>`。它重读原始设备输出，要求选中变体和nd_t在九组全部通过，才允许对31个常量MatMul统一打包；原权重、融合K/V和gate/up的逻辑顺序不变。每份打包文件都验证逆转换SHA256，验证入口再次核对实际编译引用。需额外1642MiB打包载体并在编译前计入空间下界；Conv、归一化向量、host embedding继续采用原接口。参数未选定前不开始完整模型编译。

### 离线权重布局的收益判断

当前head约19.96ms，权重Transpose已经折入MatMul属性。离线转置没有可直接扣除的现有Transpose时间；只有改变消费内核的数据访问或实际转换成本才可能加速，编译器也可能把两种输入排成相同布局。

以head约20ms、整图约60ms，假设其它开销不变：head快10%／25%／50%分别对应约省2／5／10ms。这是条件估算，不是收益预测。单head净省约2ms可作为实验选择目标；其它矩阵按各自实测收益累计，不把全部MatMul耗时当作可省时间。全词表、完整点积和原FP16边界均需保持，最终比较设备输出及含必要准备的实测耗时。编译器可能已离线完成相同排布，此时用户预打包可能没有推理收益。
