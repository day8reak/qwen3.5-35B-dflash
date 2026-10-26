# Target / Draft 直接 GE 构图 → AIR

新增入口为 `targets/cpro/scripts/direct_model_air.py`。target 和 draft 的图定义分别在 `ge_target.py`、`ge_draft.py`；通过主机 GE C++ 接口保存 AIR，生成端不导入 PyTorch、TorchNPU、TorchAir，也不初始化 NPU。原有 PyTorch 模型、参考、`export-air` 与 `export-tiny-air` 全部保留。

2026-10-22 用户反馈 tiny target/draft 双图 AIR→OMC 编译成功。2026-10-26 的后续证据确认：旧 target OMC 在 OMG 主图分区时因 NPUCL 超过9个分区回退 CPU，板端恢复被 CPUCL 的 FP16 MatMul 检查拒绝。用户基于 `78d94ef` 重新导出后，新 target 的主图窗口 L91094–L127615 仍以分区超限回退结束。后续六个 MatMul 归约小图已全部通过 Load、Process 和数值检查；整图采用该归约后仍超限；九项布局探针已回传；row Mul整图修正后ReshapeFusion通过、19个Mul CPU-only候选消失，但主图仍分区超限。[剩余CPU候选批次](cpu-gap-probes.md)的后续完整回传见下。

剩余43项探针现已回传：42项编译并完成板端数值比较，1项CastT编译失败；附加HiLog回收失败不需要重跑42项。此前按[整图GDN候选命令](hdc-board-run.md#当前将已验证的-gdn-gather-复制接入整图)复用prepared，显式叠加`--gdn-indexed-copy gather`。该开关仅改写已测FP16形状的三角复制与outer Sub，不改递推算术或权重。每图生成哈希绑定的`gdn-copy-lowering.json`，编译校验其mode/counts与manifest一致。默认`native`保留旧图，未测形状继续原实现。

最新 `ge-draft-rope-20261026-194345` 已完成双图编译、runner构建和板端 `EXECUTION_COMPLETE`。target/draft 均 Load 并各 Process 一次，target特征传入draft，必要输出及smoke报告检查通过。该记录来自用户控制台，原始报告和哈希未在本工作区独立读取；精度、NPU运行位置及性能未验证。证据见 `specs/draft-rope-execution-user-report.json`。

此前 target 的静态RoPE七节点原子折叠解决编译/加载阻塞；draft `SubGraph_2/v29/Mul` 的类型不匹配改为有界K=1 BatchMatMul后，实际tiny图已完成上述执行。仍保留INT32→FP16舍入、动态mask位置、双图ABI和外部权重；不同形状和域外合同不借此次结果视作板端通过。源码与主机验证见 `specs/ge-draft-rope-frequency-verification.json`。

下一步按[现有OMC生成比较入口](hdc-board-run.md#当前读取-gdn-整图加载失败证据)，复用 `GE_DRAFT_ROPE_RUN` 的 OMC/runner 与匹配PASS参考，写入全新 `GE_DRAFT_ROPE_CHECK_RUN`。无需重新导出、编译、构建、prepare或跑旧探针。完整权重质量和最终DFlash至少2倍收益仍是后续门槛。

此前TopK、ArgMax、Cast、归约及公共类型校验的修复见`specs/ge-dtype-contract-verification.json`；其合法构图记录保持不变，是独立的合同检查。公开GE `Cast`源类型来自输入descriptor，不猜测增加`src_type`或直接修改OMC中的Cast属性。

第1–4节保留首次生成流程和历史失败修复；当前现场无需重新 prepare 或生成参考。算子核查见 [兼容性记录](ge-operator-compatibility.md)。tiny 编译成功不表示完整35B权重、量化、内存或性能已经通过。

## 实现范围

target 覆盖 40 层完整结构定义：门控全注意力、部分 RoPE、chunk-64 GDN 和卷积、动态 TopK 路由、全部专家与共享专家、最终归一化、八层特征拼接及完整词表 Top1。draft 覆盖六层结构：目标特征投影、共享 embedding/head、动态 anchor、滑窗因果注意力与最后一层双向块注意力、RMSNorm、RoPE、SwiGLU 和 15-token 输出。

保持现有固定 S、batch 1、纯文本、前缀重算、HIAI ABI v2。静态图计算所有专家，不是稀疏 MoE 性能实现。Python 浮点标量采用 GE 属性保留原值，避免提前转 FP16 常量造成额外舍入；所有浮点张量仍为 FP16。原始 BF16/FP32 模型的质量对比是另一道未完成门槛。

**本次直接 GE 路径先提供 FP16 正确性基线，尚未映射 DOPT 的压缩整数权重算子。** `prepare` 生成的 `.bin` 是 FP16，不是 INT4。完整 35B 只能显式使用 `--float-baseline` 做服务器转换诊断，不能作为 32GB 板机部署包。原有量化代码保留；后续需要补充直接 GE 的原生量化算子映射，再验证压缩存储、质量和峰值内存。

## 1. 服务器：先准备结构小模型

在之前成功生成 Add AIR 的环境执行。保留已有 `CANN_ROOT`、`AI_RUN_DIR`；输出目录使用新的名称。这里的 NumPy 随机权重与旧 `cli demo` 即使种子相同也不同，后续参考必须使用本页的 `reference-tiny`。

```bash
cd "$REPO_DIR"
git pull --ff-only
export PYTHONDONTWRITEBYTECODE=1
export GE_MODEL_TOOL="$REPO_DIR/targets/cpro/scripts/direct_model_air.py"
export GE_PAIR_RUN="$AI_RUN_DIR/ge-pair-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$GE_PAIR_RUN"

python3 "$GE_MODEL_TOOL" prepare --tiny --sequence-length 8 \
  --output "$GE_PAIR_RUN/prepared"
```

准备只需要 Python 和 NumPy。成功状态为 `PREPARED`，产生 target/draft 的 GE 图描述、常量、C++ 生成器源码和按哈希锁定的 FP16 权重，不会把图描述文件冒充 AIR。生成器使用紧凑的二进制图描述，完整 target 的数十万节点不会展开成巨型 C++ 源文件。

tiny 为 target 4 层、draft 2 层、hidden 16、词表 64、block 4；它保留 GDN/MoE/两类注意力结构，没有可读文本质量意义。

## 2. 服务器：生成两张 AIR

```bash
export GE_AIR_DIR="$GE_PAIR_RUN/air"
python3 "$GE_MODEL_TOOL" export \
  --prepared "$GE_PAIR_RUN/prepared" \
  --ge-root "$CANN_ROOT" \
  --output "$GE_AIR_DIR"
```

使用与你已跑通的小图相同的 CANN GE 头文件和真实库。脚本编译并运行两个主机生成器；每张图保存前和加载后都检查实际计算图节点。成功时 `air/manifest.json` 为 `EXPORTED`，包含 `target/target.air` 和 `draft/draft.air`。失败保留阶段、`build.log` / `export.log` 和 `failure.json`，重试 `export` 只需换新输出目录，可复用 `prepared`。

如果旧版报 `BuildGraph: tensor_desc is null`、随后报输入不包含 Data，原因是生成器缺少 Data 算子工厂注册。修复版补齐该注册，并在构图前检查工厂原型。已有 `prepared` 的权重和图配置可直接复用；导出时会重新生成当前版本的 C++ 与图描述。保留旧失败目录，执行：

如果旧修复版报 `Data factory input x status=4294967295`，且日志已显示 `Register creator v2 for Data`，说明工厂已注册，失败来自额外检查误用了 `TryGetInputDesc`。GE 9.1 的该接口要求形状维度数大于零，不能用于检查工厂原型是否声明了端口。当前版本已移除这项误判，保留工厂及实际计算图检查；同样按下面命令更新并复用 prepared。

```bash
cd "$REPO_DIR"
git pull --ff-only
export GE_AIR_DIR="$GE_PAIR_RUN/air-datafix-$(date +%Y%m%d-%H%M%S)"
python3 "$GE_MODEL_TOOL" export \
  --prepared "$GE_PAIR_RUN/prepared" --ge-root "$CANN_ROOT" \
  --output "$GE_AIR_DIR"
```

新日志先显示 `Checking GE Data factory...`，随后进入 `SetInputs` 和计算图检查；只有两张图均保存、回读成功，最终才输出 `EXPORTED`。失败报告同时保留前几个错误和日志尾部，完整日志仍在 `export.log`。实际修复效果以服务器重试结果为准。

默认 tiny 内嵌权重。需要单独测试内部 OMG 对外部权重的支持时，在新的准备目录加 `--external-weights`，再导出、转换。完整模型自动使用 `FileConstant`；该方案依据 [TorchAir 官方外部权重实现](https://github.com/Ascend/torchair/blob/e1118e8b0fc1b95190ab53b732c0872989550e20/python/torchair/_utils/export_utils.py)，但本入口不依赖其 Python 包。AIR 中含绝对权重路径，转换完成前不要移动或清理 AIR bundle。

## 3. 服务器：使用内部 OMG 转 OMC

切回已经 source 内部 AscendC 配置脚本的终端，恢复同一 `REPO_DIR`、`GE_MODEL_TOOL`、`GE_PAIR_RUN`、`GE_AIR_DIR` 和 `AI_RUN_DIR`。`GE_AIR_DIR` 必须指向刚刚成功导出的目录。不重复拼接脚本已有环境变量。下面的 DDK 版本是你此前反馈的版本，若安装包已更新，请填写实际版本；脚本不会自动检测该版本号。

```bash
export CPRO_DDK_ROOT="$(realpath "${DDK_PATH:?先加载内部 DDK 环境}")"
export CPRO_DDK_ID=100.520.020.010
export GE_OMC_DIR="$GE_PAIR_RUN/omc"

python3 "$GE_MODEL_TOOL" compile \
  --air "$GE_AIR_DIR" \
  --ddk-root "$CPRO_DDK_ROOT" --ddk-id "$CPRO_DDK_ID" \
  --output "$GE_OMC_DIR" --reject-cpu-fallback
```

沿用现有 `compile_air`：`--framework=1 --target=omc --platform=kirinx90 --weight_data_type=FP16`，两个 OMG 子进程各有独立的 run 内临时目录和日志。`--reject-cpu-fallback` 是本项目包装器的检查参数，不是厂商 OMG 参数，行为见第7节。成功为 `COMPILED`，设备状态仍是 `PENDING`。这一步可能暴露内部 DDK 的算子、属性或布局差异，应按具体 `omg.log` 修图。

### 内部 OMG 兼容性修正与重试

**以下为历史失败及重建说明。当时的双图编译成功不代表板端加载成功；当前按页首链接确认实际运行文件，再检查可用的编译诊断，保留原 prepared、专家模式及已导出的图。**

此前失败日志（draft 进程 158383）已越过整数 Gather 的验证，最终在兼容性检查拒绝 `v12/Maximum`。它执行上下文位置的 `max(cumsum(context_valid)-1, 0)`，输入/输出是 INT32。前面的描述回显、ImportWarning 和缺少内置 kernel 库提示不是这条日志明确报告的最终失败原因。

这次只改 draft 的位置截断：对 `x=cumsum(context_valid)-1` 判断 `x<0`，再用同形 Select 选择零或原整数。`x` 的范围是 `[-1,S-2]`，图中没有浮点转换、权重修改或近似，输出节点仍叫 `v12`。模型构图限制 `S<=INT32_MAX`；原 PyTorch 路径、HIAI ABI 和 FP16 算子保持不变。

蓝区 CPUCL Maximum 只允许浮点数据，Less 和 Select 支持 INT32。但进一步审查发现 CPUCL Less 的整数实现先计算差值再比较，任意 INT32 极值可能溢出。因此没有推广成通用整数 max/min 替代：本模型的 `0-x` 范围是 `[2-S,1]`，不会溢出，转换成 FP32 后符号也保持不变。当前 target 没有 Maximum/Minimum，draft 仅这一处。蓝区 INT32 Select 支持来自 CPUCL，不能据此承诺 NPU 放置。target 进展仍依据用户日志和脚本顺序推断，未收到独立产物哈希。

**这次修改 draft 的 AIR 图，旧 prepared 可复用，但必须重新 export：**

```bash
cd "$REPO_DIR"
git pull --ff-only
export GE_AIR_DIR="$GE_PAIR_RUN/air-clamp-$(date +%Y%m%d-%H%M%S)"
export GE_OMC_DIR="$GE_PAIR_RUN/omc-clamp-$(date +%Y%m%d-%H%M%S)"

python3 "$GE_MODEL_TOOL" export \
  --prepared "$GE_PAIR_RUN/prepared" --ge-root "$CANN_ROOT" \
  --output "$GE_AIR_DIR"

python3 "$GE_MODEL_TOOL" compile \
  --air "$GE_AIR_DIR" \
  --ddk-root "$CPRO_DDK_ROOT" --ddk-id "$CPRO_DDK_ID" \
  --output "$GE_OMC_DIR"
```

只有 export 输出 `EXPORTED` 后才继续 compile。双图命令会重新生成配套产物，原先成功的 target OMC 与失败日志保留在旧目录。若仍是 draft 失败，提取本次日志：

```bash
tail -n 100 "$GE_OMC_DIR/draft/omg.log"
```

### 此前整数 Gather 兼容处理

此前日志（draft 进程 158166）明确报 `v4/GatherElements` 的 `Input[0] DataType INT32 is wrong`。这是本次失败的直接原因，前面的输入描述回显和 ImportWarning 不能代替该错误。编译已经进入 draft 阶段，按现有脚本执行顺序，target 已完成非空 OMC 输出检查与原输入哈希复核；这仍属于用户日志证据，尚无独立产物哈希及板端结果。

`v4` 用于取得最后一个有效 token。现在将 INT32 `input_ids[1,S]` reshape 成 `[S]`，用同样的 INT32 `anchor_position[1,1]` 在轴0执行 GatherV2，输出仍是 INT32 `[1,1]`，节点名仍是 `v4`。HIAI 输入合同要求 batch1、非空右填充前缀，保证 anchor 位于 `[0,S-1]`。全程不把 token ID 转成 FP16 或 FP32；其余模型浮点计算仍为 FP16。

已检查 tiny/full 的 target/draft：原先只有 draft 的这一处整数 GatherElements。蓝区实际 GatherElements 验证只接受浮点数据，而 GatherV2D 前端与 CPUCL 的 Gather 支持 INT32 数据和 INT32 索引，CPUCL 有直接 `DoGather<int>` 路径。这不证明内部 X90 会将该节点放到 NPU；最终后端分配与精度仍以内部编译和板端验证为准。导出器现在提前拒绝其他非 FP16 数据的 GatherElements，避免把已知不兼容的节点继续写进 AIR。

### 此前 BatchMatMul 兼容处理

此前日志（进程 157847）明确显示 `Tbe inner init successfully` 和 `Parallel compilation init success`，此前 socket 路径问题已通过用户复验。失败仍停在 `v1115/BatchMatMul` 支持检查；这次不能继续将它归因于 TE 初始化失败。缺少 `libai_npucore_ascendc_kernel.so` 的提示之后，DDK 明确继续使用 JIT；现有证据不足以判定该库必须补齐。

本轮把 target/draft 中批次前缀相同的四维、五维矩阵乘统一成三维 BatchMatMul，再恢复输出形状。例如 `v1115` 的 `[1,2,1,64,64] × [1,2,1,64,8]` 变为 `[2,64,64] × [2,64,8]`，结果恢复成 `[1,2,1,64,8]`；新原生节点名为 `__ge/v1115/batch_output`。只合并 batch 维，M/K/N 和每个矩阵的对应关系不变，使用受检查的 INT32 shape。前缀不一致会明确拒绝，不能将交叉广播误写成一对一批次。现有三维卷积 BMM 和二维投影保留，不逐批展开：完整模型卷积分支每层最多8192个批次，逐批拆分会造成大量节点增长。

蓝区库的实际支持检查还拒绝 `adj_x1=true`。完整模型 S128 有30个这样的 GDN 状态更新节点；现在在三维左输入上显式执行 `Transpose(0,2,1)`，再设置 `adj_x1=false`，右侧转置语义保留。没有增加浮点计算或 FP32 张量。**蓝区检查器没有给出五维不支持的直接证据，所以批次规范化仍是待内部 OMG 验证的兼容处理，不能宣称已确定 v1115 的唯一根因。**

日志中的 `Ascend310B1` 也不足以判断目标配错：蓝区 Kirin 工具链存在平台名映射、Ascend310B kernel magic 和 ESL 标识。内部 X90 的具体映射不可直接检查，继续使用原来的 `--platform=kirinx90`，不手工改成其他芯片。

以下为此前 BatchMatMul 更新的历史命令；当前现场使用第7节命令。

```bash
cd "$REPO_DIR"
git pull --ff-only
export GE_AIR_DIR="$GE_PAIR_RUN/air-bmm-$(date +%Y%m%d-%H%M%S)"
export GE_OMC_DIR="$GE_PAIR_RUN/omc-bmm-$(date +%Y%m%d-%H%M%S)"

python3 "$GE_MODEL_TOOL" export \
  --prepared "$GE_PAIR_RUN/prepared" --ge-root "$CANN_ROOT" \
  --output "$GE_AIR_DIR"

python3 "$GE_MODEL_TOOL" compile \
  --air "$GE_AIR_DIR" \
  --ddk-root "$CPRO_DDK_ROOT" --ddk-id "$CPRO_DDK_ID" \
  --output "$GE_OMC_DIR"
```

只有 export 输出 `EXPORTED` 才继续 compile，原目录与日志保留。真实 X90 内核支持、HIAI 输出和耗时仍需服务器与板端验证。

### 此前 IPC 与图兼容性修正

此前日志（进程 157707）首错是 `OSError: AF_UNIX path too long`，随后 `multi_process_check`、TE 初始化失败，最后才报 `v1115/BatchMatMul` 不支持。旧脚本将 OMG 的 `TMPDIR` 设为完整 run 路径；[CPython 的 multiprocessing](https://github.com/python/cpython/blob/3.10/Lib/multiprocessing/connection.py) 还会在它下面追加 `pymp-*`、`listener-*`，长路径可复现同样的 socket 错误。

现在 OMG 子进程的 `TMPDIR/TMP/TEMP` 使用短的 `/proc/<父进程PID>/fd/<FD>/.ipc-<随机后缀>` 别名。实际文件仍位于本次 run 的独立临时目录，父进程在编译期间保持目录描述符有效；每次调用使用不同后缀，防止旧进程因描述符复用而误访问新调用的目录。父 shell 不变。每图 `omg-environment.json` 记录短路径、物理目录和有效期。此方案依赖 Linux procfs 访问；厂商若自行将别名解析成真实长路径，仍须进一步适配，不能据主机测试宣称内部 TE 已通过。

源码核对 `v1115` 为 FP16 `[1,2,1,64,64] × [1,2,1,64,8] → [1,2,1,64,8]`，发射 `BatchMatMulV2`，无转置或 batch 广播不匹配。TE 初始化失败可能导致后续算子注册不可用，因此先修复 IPC 再判定 X90 内核支持；此修改不改变模型算子或精度。

下面保留的是此前仅修复 IPC 时的 compile 重试方式；**本轮 BatchMatMul 更新请使用上方 export + compile 命令**。

```bash
cd "$REPO_DIR"
git pull --ff-only
export GE_OMC_DIR="$GE_PAIR_RUN/omc-ipc-$(date +%Y%m%d-%H%M%S)"

python3 "$GE_MODEL_TOOL" compile \
  --air "$GE_AIR_DIR" \
  --ddk-root "$CPRO_DDK_ROOT" --ddk-id "$CPRO_DDK_ID" \
  --output "$GE_OMC_DIR"
```

原 AIR 和输出日志保留。若 TE 初始化正常后仍报 BatchMatMul 不支持，再使用新目录的日志确定内部 X90 对该 FP16 五维形状的限制；不要将两次进程的日志混在一起判断。

此前日志（进程 157562）显示 `v974` 的形状常量从 `1 0 2 0 1 0 55 0` 被 OMG 改成 `1 0 2 0 1 0 -1 0`，随后推导为 `[1,2,1,2147483647]`；相邻 `v975` 也有相同改写。源图实际形状为 `[1,2,1,55]`，每个维度存储为 INT64，没有大整数。该字节模式符合只改低32位、未作64位符号扩展的情况，内部实现仍不可访问。现在只将 **Reshape 的 shape 辅助张量**改为官方支持的 INT32；维度写入前必须在非负 INT32 范围内。数据、权重和浮点计算精度不变，其他轴/切片参数仍使用原类型，不新增 Squeeze/ExpandDims。

同一日志将 Slice 类型限制明确定位在 `TCH ConstantFolding` 阶段。导出端新增有界静态 FP16 Slice/Reshape 预计算，直接复制原字节，tiny 的32个专家静态切片全部提前完成；每个源和输出≤64KiB，累计读取及结果≤8MiB。大型专家权重仍跳过，不能据 tiny 结果宣称完整35B的常量折叠已兼容。现有动态 Slice 不转成 FP32；原 prepared 权重文件和哈希保持不变。

`te_fusion.log_util` 缺少 `get_py_exception_dict` 指向加载包与本地库的接口不匹配。OMG 子进程现在优先使用所选 DDK 的 `tools/tools_ascendc/package/python` 和 `ddk/tbe/ops`，清除已明确声明的外部 CANN 根目录所贡献的 Python/库/可执行路径，保留内部扩展和模拟器路径。父 shell 及 AIR 生成环境不变；不修改厂商包中的 Python 函数。每图输出 `omg-environment.json` 供本地核对包来源；它记录候选路径，不代表已经验证嵌入式 Python 实际加载结果。如果内部 DDK 自身存在版本错配，仍需要同版本包。

转换失败会优先保留 `dump_graph/*fail*.irpb` 等图转储至 `target/diagnostics/` 或 `draft/diagnostics/`，总载荷上限16MiB，超限记录在该目录的 manifest 中；临时运行目录仍清理。完整 `omg.log` 始终保留。

此前日志（进程 157297）明确报 `v92/Cumsum` 缺少 infershape 函数。名称符合官方 GE schema，并非大小写错误。生成器现在分开处理两种用途：GDN 的非正 FP16 衰减前缀使用 `Slice → ReduceSum → ConcatD`，每个前缀仅输出一次 FP16；draft 的整数位置前缀使用整数移位相加。FP16 路径只允许模型标记的非正衰减域和长度 ≤256，当前 GDN 长度为64；不使用逐步 FP16 Add，也不承诺对任意正负混合输入的前缀和等价。主机比较保持原有 FP16 容差，真实 ReduceSum 内核数值仍需板端验证。

日志同时报 CPUCL 输入2字节、预期4字节。附近 `__ge/v3735/broadcast_2` 对应 FP16 标量零扩展到 `(8,16)`；日志没有给出失败节点，不能断言全部缓冲区错误都源于它。已将小型静态广播直接存为 Const，按源字节和目标形状复用，保留正负零、Inf 和 NaN 载荷。每个扩展 ≤1MiB，累计扩展载荷 ≤8MiB，超限保留原 BroadcastTo；只处理内联常量，不读取完整 checkpoint。常量按实际使用生成，避免留下不连通节点。

此前日志（进程 157178）报 `v52/SoftplusV2` 缺少 infershape 函数的修正仍保留。当前模型固定使用 FP16、`beta=1`、`threshold=20`，生成器改发原生 `Softplus`（x/y，无额外属性）；[官方 GE 文档](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/programug/graphdevg/atlasag_25_0009.html)有该算子的构图用法。穷举全部 63,488 个有限 FP16 输入，理想函数的阈值版本与非阈值版本舍入到 FP16 后逐位一致。其他 beta/threshold 设置会明确拒绝，不擅自改变公式；真实内部 kernel 数值仍需上板比较。

同一日志还报 CPUCL 输入 32 字节、预期 64 字节，且 `fp16Flag:0`。这与小型 FP16 常量算术进入 FP32 执行路径相符，但日志未给出该失败节点，暂不把具体归属当作定论。生成器现在只提前计算小常量的 `scalar_add/scalar_mul/exp/neg`，保留每个原始节点的 FP16 输出舍入，避免把归一化偏移和 A_log 衰减参数计算交给这条 CPUCL 路径。原始 prepared 权重及其哈希不变；不展开专家矩阵、切片或 reshape，不加载完整 35B。单张量上限 64 KiB、累计缓存/读取预算 8 MiB。

此前的 Select 和 RMSNorm 修正仍保留；这次 RMSNorm 的常量 Power 会进一步变成直接存储的 FP16 结果：

2026-10-22 用户的 tiny target 日志中，`v7/Select` 报 `x1Shape.GetDimNum() != x2Shape.GetDimNum()`；源码对应四维条件和两个标量分支。现已在 target/draft 的每个 Select 前显式广播三个输入到相同形状，保留真正的选择操作及 `-inf`/NaN 语义。

同一日志的 `v23/Add` 对应 RMSNorm 的 `1 + weight`，CPUCL 报输入类型不一致。保存并回读的 AIR 描述中二者均为 FP16，现有日志不能确定 OMG 内部发生了什么类型变化；源码将此处改为单输入 Power 的 `shift=1`。全部有限 FP16 输入的本地检查与原加法逐位一致，真实转换仍须重试。节点编号会随重新构图变化，应以新日志为准。

若尚未重新导出过上述 Reshape、静态常量和 Cumsum 修正，须先更新 AIR，不能只重跑旧 AIR 的 `compile`。可复用旧 `prepared`，不用重新下载或生成权重；保留原目录并运行：

```bash
cd "$REPO_DIR"
git pull --ff-only
export GE_AIR_DIR="$GE_PAIR_RUN/air-compat-$(date +%Y%m%d-%H%M%S)"
export GE_OMC_DIR="$GE_PAIR_RUN/omc-compat-$(date +%Y%m%d-%H%M%S)"

python3 "$GE_MODEL_TOOL" export \
  --prepared "$GE_PAIR_RUN/prepared" --ge-root "$CANN_ROOT" \
  --output "$GE_AIR_DIR"

python3 "$GE_MODEL_TOOL" compile \
  --air "$GE_AIR_DIR" \
  --ddk-root "$CPRO_DDK_ROOT" --ddk-id "$CPRO_DDK_ID" \
  --output "$GE_OMC_DIR"
```

只有 export 输出 `EXPORTED` 后再执行 compile。新版本的转换失败会直接显示 graph、阶段、日志路径、首错和尾部，并在 `failure.json` 保留退出码或超时值；draft 失败时不会发布成功的双图 manifest。摘要识别内部 OMG 的 `E/AI_FMK` 等格式，不将 W/I 级别的 `failed to load` 插件提示列作首错；也会跳过本次已确认的 shape 回显及明确返回 SUCCESS 的空配置消息，保留真正的 CPUCL 缓冲区错误。完整日志保留所有消息；尾部不应用上述过滤。

若仍失败，提供终端摘要，或提取完整日志的尾部；target 通过而 draft 失败时将下列路径中的 target 改为 draft：

```bash
tail -n 100 "$GE_OMC_DIR/target/omg.log"
```

## 4. 服务器参考与板端运行

完整、无需替换旧指南变量的命令见 [直接 GE tiny OMC 上板指南](direct-ge-board.md)。下方保留原路径映射供已有操作续接。

在已有 PyTorch/Transformers 的 CPU 参考环境执行。此命令才需要 PyTorch，会沿用项目的 CPU 后端隔离；它读取本次 prepared 的实际权重，而非重新生成另一份随机模型。切换环境后再设置下面的解释器与项目路径；后续复用板端指南时保持这些变量。

```bash
export MODEL_PYTHON="$(command -v python3)"
export PYTHONPATH="$REPO_DIR/model:$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
python3 "$GE_MODEL_TOOL" reference-tiny \
  --prepared "$GE_PAIR_RUN/prepared" \
  --prompt-token-ids 1,2,3 --max-new-tokens 4 \
  --output "$GE_PAIR_RUN/tiny-reference.json"
python3 - <<'PY'
import json, os
from pathlib import Path
run = Path(os.environ["GE_PAIR_RUN"])
with (run / "tiny-prompt.json").open("x") as out:
    json.dump({"prompt_token_ids": [1, 2, 3]}, out)
PY
```

参考应为 `PASS`。双图输出与现有 HIAI runner 一致：target 为 INT32 `[1,8]` 和 FP16 `[1,8,32]`，draft 为 INT32 `[1,3]`。板端编译、传输、运行及回传比较沿用 [tiny 验证指南第 2–5 节](tiny-validation.md#2-服务器编译程序并传文件)，将路径对应如下；不要使用那个页面第 1 节重新生成旧随机权重。

| 旧指南路径 | 本次路径 |
|---|---|
| `$AI_RUN_DIR/tiny-omc` | `$GE_OMC_DIR` |
| `$AI_RUN_DIR/tiny-reference.json` | `$GE_PAIR_RUN/tiny-reference.json` |
| `$AI_RUN_DIR/tiny-prompt.json` | `$GE_PAIR_RUN/tiny-prompt.json` |

传板端的是整个 OMC bundle、交叉编译的 `qwen35_dflash_hiai` 和板端缺少的匹配运行库。复用旧指南时，把服务器 `$GE_OMC_DIR/` 的内容放到板端 `$BOARD_ROOT/models/tiny-omc/`，保持板端命令中的目录名一致。AIR、prepared 权重、图描述、C++ 生成器、原始 safetensors 和 Python 环境都留在服务器。参考与 OMC 的 `source` 包含所有 FP16 权重内容的聚合哈希，可拒绝混用不同随机权重的报告。

## 5. 完整 35B：先查结构，再做服务器诊断

无需加载权重即可核对完整结构和空间估算：

```bash
python3 "$GE_MODEL_TOOL" plan --full --sequence-length 8 \
  --output "$GE_PAIR_RUN/full-plan"
```

完整 target 的 693 个参数定义已经与锁定模型结构/检查点索引核对。此项不代表实际权重加载或量化完成。完整 target/draft 的唯一 FP16 权重约 70.09 GB；原始下载之外还要为准备权重、AIR 和 OMC 留出空间。准备阶段按块读取 [safetensors 格式](https://github.com/safetensors/safetensors#format)，转换 BF16→FP16，无需把整个模型载入主机内存；转换/编译工具自身仍可能需要大量 RAM。

只在需要排查完整模型转换且服务器资源充足时执行：

```bash
python3 "$GE_MODEL_TOOL" prepare --full --float-baseline --sequence-length 8 \
  --target-weights "$QWEN35_TARGET_WEIGHTS" \
  --draft-weights "$QWEN35_DRAFT_WEIGHTS" \
  --output "$GE_PAIR_RUN/full-prepared"
python3 "$GE_MODEL_TOOL" export \
  --prepared "$GE_PAIR_RUN/full-prepared" --ge-root "$CANN_ROOT" \
  --output "$GE_PAIR_RUN/full-air"
```

不要将这份未量化 FP16 全模型传到 32GB 板机。完成 tiny 双图 AIR/OMC/HIAI 验证后，下一步是原生量化映射、全权重数值验证和内存实测；原 PyTorch/TorchAir 路径随时可继续联调。

## 6. 按 Cpro 参考结构调整

用户提供了一个已在同一 Kirin/Cpro 板上运行、但精度仍有问题的工程结构摘要。摘要的 40 层、30/10 混合注意力、2048 隐藏维、256 专家和 top-8 与本仓库锁定配置一致。该摘要作为结构参考；数值对照继续使用原始 checkpoint 语义及本仓库已有的 FP16 参考，不能把另一工程“能运行”视为精度通过。

本次调整直接 GE 图中的两部分：

| 部分 | 改动 | 精度与验证边界 |
|---|---|---|
| GDN chunk64 | `(1,H,C,64,D)` 去掉 batch=1 单例轴成为 `(H,C,64,D)`；三处常量三角掩码改为 Slice/Concat 补零 | 保留 FP16 边界、64 宽归约和跨 chunk 状态更新次序；移除这部分 rank5 张量及 BOOL Select。静态切片会增加节点，不能据此声称提速 |
| MoE selected 模式 | 仅收集每个 token 的 top-k 专家，执行 W13 → SiLU×up → W2，再合并共享专家 | 保留原 softmax/top-k/权重归一化；按专家 ID 升序逐路 FP16 累加，不能改成 top-k 分数顺序或一次 ReduceSum。使用现有 Gather/BatchMatMul，不冒充原生 GMM |
| 归一化 | 保持现有正确语义 | loader 不折叠 gamma；普通 RMSNorm forward 加一次 `1`，GDN gated norm 和 DFlash norm 直接用权重。没有复制摘要中的重复 `+1` |

`selected` 为显式候选，`static` 继续作默认对照。输入输出 ABI、全部词表、目标特征层、DFlash 调度、原始权重及 FP16 精度不变。没有在本轮引入 W4A8、INT4/I4ZN 打包、PagedAttention 或增量缓存；这些改动分别需要实际算子端口/属性/状态合约及独立精度验证。这里的 FP16 结构等价检查也不等于 BF16/FP32 原模型的语言质量验证。

每个 token 收集的两组专家权重合计为 `K × 3 × H × I × 2` 字节，35B 配置下为 48 MiB。这只是逻辑张量大小；整个静态图的调度、活跃期和峰值工作区未测，不能当作内存上界。全量专家权重仍然存在，FP16 完整模型的存储问题也未解决。候选支持最多 2048 个专家，以保证用 FP16 排序专家 ID 时不丢失整数精度。

以下是后续可独立开展的 selected 结构实验，不是当前上板步骤；当前仍保留原 prepared 的专家模式。开展此实验时，在新目录生成 tiny 候选及独立参考，再导出编译，沿用已有 `AI_RUN_DIR`、`CANN_ROOT`、`CPRO_DDK_ROOT` 和 `CPRO_DDK_ID`：

```bash
git -C "$REPO_DIR" pull --ff-only
export GE_MODEL_TOOL="$REPO_DIR/targets/cpro/scripts/direct_model_air.py"
export GE_STRUCTURE_RUN="$AI_RUN_DIR/ge-structure-$(date +%Y%m%d-%H%M%S)"

python3 "$GE_MODEL_TOOL" prepare --tiny --sequence-length 8 \
  --expert-dispatch selected --output "$GE_STRUCTURE_RUN/prepared"
python3 "$GE_MODEL_TOOL" reference-tiny \
  --prepared "$GE_STRUCTURE_RUN/prepared" \
  --prompt-token-ids 1,2,3 --max-new-tokens 4 \
  --output "$GE_STRUCTURE_RUN/reference.json"
python3 "$GE_MODEL_TOOL" export \
  --prepared "$GE_STRUCTURE_RUN/prepared" --ge-root "$CANN_ROOT" \
  --output "$GE_STRUCTURE_RUN/air"
python3 "$GE_MODEL_TOOL" compile \
  --air "$GE_STRUCTURE_RUN/air" \
  --ddk-root "$CPRO_DDK_ROOT" --ddk-id "$CPRO_DDK_ID" \
  --output "$GE_STRUCTURE_RUN/omc"
```

每一步成功后再执行下一步。plan、prepared、AIR 和 OMC 的 manifest 都记录 `expert_dispatch`，export 按 prepared 的选择重建图；老 prepared 缺少该字段时沿用 static。全量旧算子复测仍可用 `probe_hdc_runtime.py --replay-all`，但它复用旧 OMC，不能代替本节新结构的 OMG/板端验证。正式双图上板入口使用探针已验证的候选部署；当前采用已通过小图的GDN复制候选，并进入[重导出与只执行模式](hdc-board-run.md#当前双图采用可选-matmul-归约)，不切换到 selected。

可选的主机 GDN 诊断入口为 `targets/cpro/scripts/diagnose_gdn_precision.py`，输出逐 chunk FP16/FP32 输出与状态误差。无输入时仅跑小型合成样例，不能确认真实模型的精度根因。当前优先上板执行，此诊断不作为前置步骤，也不改变部署 FP16。

<a id="7-当前现场复用原权重重新导出并检查分区回退"></a>
<a id="7-当前现场先读取现有编译产物定位候选切断点"></a>

<a id="7-当前现场验证-matmul-归约候选"></a>

## 7. 当前现场：GDN 复制整图候选与探针历史


### 当前入口：检查已有 OMC 的 CPU 编译子图

用户已回传GDN7项及其余43项：四个GDN Gather候选通过，其余42个已编译案例完成数值比较。整图GDN候选的target Load随后失败；CPU子图解码已定位到静态RoPE BMM之后的insert_cast_4。后续完整静态RoPE折叠已通过双图编译和target Load，draft动态频率BMM候选也已通过tiny双图执行，当前按[现有OMC生成比较入口](hdc-board-run.md#当前读取-gdn-整图加载失败证据)验证生成结果。以下保留GDN候选设计及历史命令。

<a id="gdn-indexed-copy-probes"></a>

### 历史 GDN 子集说明（已包含在 remaining-cpu 批次）

用户更新后重跑 `ge-row-chain-20261026-144144`，ReshapeFusion 已通过（fused 18 pairs），此前19个row Mul不再列为CPU-only计算候选；完整支持表15,747项，无重复或冲突，排除8,683项边界及7项角色未确定后，已识别CPU-only计算从117降至98。主图L150570仍超过9个NPUCL分区而回退；draft编译、新正式runner构建和上板未执行。支持候选数量不等于分区数量，也不是运行位置证明。`ge-row-chain-20261026-143421`那次相同早期错误已由用户确认漏了git pull，不作为807c063的有效复测。

当前重点是6个GDN Concat、3个outer Sub；另有80个MoE bool/int路由节点、4个TopK和5个其他候选。源码核对v2624来自Exp后的三角掩码，v2879来自严格三角掩码；原“64独立Data输入”Concat探针没有覆盖动态Slice前缀、常量零尾和Exp前后两个mask。源码共有9个64路mask，不把这些名字或数量直接等同于编译后的6个候选。

本轮新增7项隔离对照，生产target/draft及其prepared/参考不变。掩码候选将FP16矩阵展平后追加一个正零，使用静态INT32 Gather(axis0)选取原元素或该零，再恢复shape；不乘浮点mask。Sub候选用两次静态Gather分别将左右输入复制到同形，再做一次原FP16减法。拷贝步骤保留元素位模式，真实内核的行为和新shape支持仍须编译/上板验证。

| 对照 | 案例 | 数值门槛 |
|---|---|---|
| 普通三角mask | `gdn_mask_lower_slice_concat` / `gdn_mask_lower_gather` | 原Slice/Concat参考，零容差 |
| 严格三角mask（不保留对角） | `gdn_mask_strict_slice_concat` / `gdn_mask_strict_gather` | 原Slice/Concat参考，零容差 |
| mask→Exp→mask真实上下文 | `gdn_mask_exp_chain_slice_concat` / `gdn_mask_exp_chain_gather` | 原两个mask和一次FP16 Exp，原FP16门槛 |
| outer Sub | `sub_outer_gather` | 与已测`sub_outer_rank4`完全相同的输入、参考、门槛 |

全部矩阵输出为`[2,1,64,64]`。mask输入含8192个不同且非零的有限FP16值，两个head数据不同；Gather索引最大8192，单mask索引32KiB。Sub输入仍为`[2,1,64,1]`和`[2,1,1,64]`，展开只复制数据。新helper限正静态rank4和输出最多1,048,576元素，不引入浮点索引或权重转换。

下方保留7项子集的独立命令，仅用于需要单独复测时。当前已有完整回传，请使用页首链接中的整图命令，不必执行下方命令：

```bash
git -C "$REPO_DIR" pull --ff-only
export GE_GDN_COPY_RUN="$AI_RUN_DIR/gdn-copy-$(date +%Y%m%d-%H%M%S)"
export GE_GDN_COPY_BOARD_RUN="$AI_RUN_DIR/gdn-copy-board-$(date +%Y%m%d-%H%M%S)"

python3 "$REPO_DIR/targets/cpro/scripts/probe_hdc_ops.py" \
  --compile-only \
  --case gdn_mask_lower_slice_concat --case gdn_mask_lower_gather \
  --case gdn_mask_strict_slice_concat --case gdn_mask_strict_gather \
  --case gdn_mask_exp_chain_slice_concat --case gdn_mask_exp_chain_gather \
  --case sub_outer_gather \
  --output "$GE_GDN_COPY_RUN" &&
python3 "$REPO_DIR/targets/cpro/scripts/probe_hdc_runtime.py" \
  --probe-run "$GE_GDN_COPY_RUN" --native-from "$OP_PROBE_RUN" \
  --refresh-runner --replay-all --output "$GE_GDN_COPY_BOARD_RUN"

cat "$GE_GDN_COPY_RUN/report.txt"
if [ -f "$GE_GDN_COPY_BOARD_RUN/runtime-comparison.txt" ]; then
  cat "$GE_GDN_COPY_BOARD_RUN/runtime-comparison.txt"
fi
```

保留`GE_ROW_CHAIN_RUN`及其inspect报告。根据新case的编译主图分区、Load、Process和原参考数值结果再决定整图适配；如果控制和候选都只在CPU，不能因数值通过就晋级。即使探针通过，仍须验证完整图的分区和执行，当前精度后置、最终DFlash至少2倍目标保持。

<a id="当前下一步可选-row-mul-整图适配"></a>

### 历史：row Mul 整图适配与Reshape修正

以下是已完成的复测过程；当前采用页首链接中的GDN整图候选，无需再次运行这组旧整图命令。

历史用户结果 `ge-row-mul-20261026-121239` 已导出target/draft，归约替换为421/10，row Mul替换为19/0；target的OMG在`ReshapeFusion`删除视图后报`graph_bypasser.cpp ByPassNode: PreCheck false`，退出1。该次未到主图分区检查，不能沿用上一轮“超过9分区”的失败分类；draft编译、runner构建和上板均未运行。更早的kernel库路径报错尚未证实是这次终止原因。

本地实际配方显示，新增Mul右输入连接到原升维reshape，恢复后的Mul输出又连接到原reshape和BMM辅助reshape。当前`compact`候选在导出端只合并涉及这些视图的连续、同dtype、同元素数的完整静态Reshape，保持最外视图名与输出形状，保留仍被其他分支或图输出引用的父视图，不跨越Transpose或算术。该精确改写在用户后续重跑中已通过ReshapeFusion；没有证明厂商pass内部的故障原因，原默认`native`不改变。

此前`inspect-compile`要求已有compiled graph记录，而这种早期失败的`graphs=[]`，因此误拒绝诊断。更新后可直接读取旧留档；只在failed_graph、stage、log_path和AIR身份匹配时接受，并明确日志哈希是本次诊断采集、没有历史编译哈希。无可信支持表时报告缺失/不完整，不把常量折叠子图当成主图分区：

```bash
git -C "$REPO_DIR" pull --ff-only
export GE_EARLY_INSPECT_RUN="$AI_RUN_DIR/ge-early-inspect-$(date +%Y%m%d-%H%M%S)"
python3 "$GE_MODEL_TOOL" inspect-compile \
  --air "$GE_ROW_MUL_RUN/air" --omc "$GE_ROW_MUL_RUN/omc" \
  --graph target --output "$GE_EARLY_INSPECT_RUN"
```

用户回传 `layout-probe-20261026-113524` / `layout-board-20261026-113529`：9项均通过 Load、Process 和原数值比较。两种 row Mul 原形为 NPU=0、CPU=1，compact 均为 NPU=1、CPU=0，因此只将 row Mul compact 推进整图候选。outer Sub compact 仍为 NPU=2、CPU=1；Concat64 原形、tree8、tree2 都是 NPU=1、CPU=0，不能将整图中的Concat拒绝归因于64路扇入，也没有依据用树替换原形。Sub和Concat保持原实现。

以上是用户粘贴报告，未独立获取并核对原始产物哈希；此前建议 revision 为 `37ba5d3`，实际运行 revision 未报告。编译分区与实际执行位置分开记录，9项的 runtime placement 仍为 `UNVERIFIED`。

| 案例 | 编译主图 NPU/CPU | Load / Process / 数值 |
|---|---|---|
| `mul_rows16_rank4`、`mul_rows128_rank4` | 0 / 1 | 全部 PASSED |
| `mul_rows16_compact`、`mul_rows128_compact` | 1 / 0 | 全部 PASSED |
| `sub_outer_rank4` | 0 / 1 | 全部 PASSED |
| `sub_outer_compact` | 2 / 1 | 全部 PASSED |
| `concat64_rank4`、`concat64_tree8`、`concat64_tree2` | 1 / 0 | 全部 PASSED |

复用 `$GE_PAIR_RUN/prepared` 和 `$GE_BOARD_RUN/reference.json`，显式叠加 `--fp16-reductions matmul --fp16-row-mul compact`，导出到新目录。两个选项默认均为 `native`；不重新 prepare、不改专家模式、不换库。构建沿用原 resources/环境中的 SDK、头文件和链接资源；若此前显式指定 resources 或兼容 SDK 版本，继续使用同一配置。

```bash
git -C "$REPO_DIR" pull --ff-only
export GE_MODEL_TOOL="$REPO_DIR/targets/cpro/scripts/direct_model_air.py"
export GE_ROW_CHAIN_RUN="$AI_RUN_DIR/ge-row-chain-$(date +%Y%m%d-%H%M%S)"
export HIAI_ROW_CHAIN_BUILD_DIR="$GE_ROW_CHAIN_RUN/hiai-build"
export HDC_ROW_CHAIN_RUN_DIR="$GE_ROW_CHAIN_RUN/board"

python3 "$GE_MODEL_TOOL" export \
  --prepared "$GE_PAIR_RUN/prepared" --ge-root "$CANN_ROOT" \
  --fp16-reductions matmul --fp16-row-mul compact \
  --output "$GE_ROW_CHAIN_RUN/air" &&
python3 "$GE_MODEL_TOOL" compile \
  --air "$GE_ROW_CHAIN_RUN/air" \
  --ddk-root "$CPRO_DDK_ROOT" --ddk-id "$CPRO_DDK_ID" \
  --output "$GE_ROW_CHAIN_RUN/omc" --reject-cpu-fallback &&
python3 "$REPO_DIR/targets/cpro/scripts/build_hiai.py" \
  --output "$HIAI_ROW_CHAIN_BUILD_DIR" &&
python3 "$REPO_DIR/targets/cpro/scripts/run_hdc.py" \
  --omc "$GE_ROW_CHAIN_RUN/omc" \
  --runner "$HIAI_ROW_CHAIN_BUILD_DIR/build/qwen35_dflash_hiai" \
  --reference "$GE_BOARD_RUN/reference.json" \
  --output "$HDC_ROW_CHAIN_RUN_DIR" --execution-only
```

四步用 `&&` 串联；导出或编译失败就停止正式 runner 构建和上板。新的 `qwen35_dflash_hiai` 包含现有 ND 补维兼容；旧独立 probe 或旧正式二进制不能代替该次构建。旧 prepared、OMC、runner、探针与板端结果均保留。

row Mul compact 只将已验证的两种 FP16 广播形状压成二维，保持行顺序、原操作数及一次 FP16 Mul，再恢复原输出形状。当前 tiny/static 源码预期 target 替换19项（rows16为10项、rows128为9项）、draft为0项；用户已确认上一轮实际导出计数为19/0，链处理后的新重跑通过ReshapeFusion，19项Mul退出CPU-only列表；主图仍分区超限，完整执行待验证。每图新增 `row-mul-lowering.json`，同时保留 `reduction-lowering.json`；反馈 `mode` 和 `counts` 即可：

```bash
python3 - <<'PYCOUNTS'
import json
import os
from pathlib import Path

for name in ("target", "draft"):
    folder = Path(os.environ["GE_ROW_CHAIN_RUN"]) / "air" / name
    for filename in ("reduction-lowering.json", "row-mul-lowering.json"):
        report = json.loads((folder / filename).read_text())
        print(name, filename, report["mode"], report["counts"])
        if "reshape_chains" in report:
            print("  reshape_chains", report["reshape_chains"]["counts"])
PYCOUNTS
# 仅上板已尝试并生成对应文件时读取；编译失败不会进入该阶段。
for file in result.json benchmark.stderr; do
  if [ -f "$HDC_ROW_CHAIN_RUN_DIR/$file" ]; then
    cat "$HDC_ROW_CHAIN_RUN_DIR/$file"
  fi
done
```

历史首次导出在OMG图优化阶段失败；链处理后的重跑通过ReshapeFusion，但主图仍超过9分区，draft编译和上板未运行。整数归约、路由、TopK及GDN掩码等候选仍待处理。编译通过后 `--execution-only` 强制 target/draft 各执行一次并回传输出；精度和性能为 `NOT_EVALUATED`。完整精度后续验证，最终目标仍为同板同目标权重下 DFlash 至少提速2倍。

若新编译失败或仍回退，保留本次 AIR/OMC、日志和 `failure.json`，对同一产物运行只读诊断，不直接重复编译：

```bash
export GE_ROW_CHAIN_INSPECT_RUN="$AI_RUN_DIR/ge-row-chain-inspect-$(date +%Y%m%d-%H%M%S)"
python3 "$GE_MODEL_TOOL" inspect-compile \
  --air "$GE_ROW_CHAIN_RUN/air" --omc "$GE_ROW_CHAIN_RUN/omc" \
  --graph target --output "$GE_ROW_CHAIN_INSPECT_RUN"
cat "$GE_ROW_CHAIN_INSPECT_RUN/report.txt"
```

上例对应 target 失败；draft 失败时改 `--graph draft` 并使用新诊断目录。候选支持与构造时边不等于最终分区或设备执行位置。

<a id="当前下一步广播与拼接布局探针"></a>

### 历史：已完成的广播与拼接布局探针

`ge-reduce-inspect-20261026-111501` 已取得完整主图支持表，15,675项无冲突，15,668项与recipe精确匹配；排除8,647个边界项后，CPU-only计算候选为117（此前538），另7项角色未确定。target确实替换421个FP16归约（含192个扫描前缀），保留16个整数归约；draft替换10个、保留1个。新CPU计算列表不再出现FP16 Reduce，整图仍因分区数超过9回退。

优先隔离两侧均为NPU候选的19个Mul、3个Sub和6个64路Concat。当时保持整图产物并运行以下9项探针；当前结果和选定适配已记录在本节开头，无需重复这批探针。两种Mul形状在本地tiny源码中匹配19个节点，但回传仅列每组前两例，不将其视为19个精确名称均已核对。

| 探针组 | 原形 | 候选 | 数量 |
|---|---|---|---|
| `mul_rows16` | `[1,8,2,8] * [1,8,2,1]` | `[16,8] * [16,1]` 后恢复 | 2 |
| `mul_rows128` | `[1,2,64,8] * [1,2,64,1]` | `[128,8] * [128,1]` 后恢复 | 2 |
| `sub_outer` | `[2,1,64,1] - [2,1,1,64]` | `[2,64,1] - [2,1,64]` 后恢复 | 2 |
| `concat64` | 64个 `[2,1,1,64]`，axis=2 | 相邻8路或2路逐层拼接，保持顺序 | 3 |

Mul/Sub仅改变Reshape，保留原始操作顺序和一次FP16出口；不引入额外算术或新dtype。Concat只复制数据，三图均使用64个独立动态输入，每个元素使用不同的有限FP16位模式，避免常量折叠、整段Slice→Concat恒等消除及重复数据掩盖错序。三图IO载荷各32KiB、65个IO均已在回传中通过加载与执行；OMG是否将树重新合并尚未独立核验。

```bash
git -C "$REPO_DIR" pull --ff-only
export GE_LAYOUT_PROBE_RUN="$AI_RUN_DIR/layout-probe-$(date +%Y%m%d-%H%M%S)"
export GE_LAYOUT_BOARD_RUN="$AI_RUN_DIR/layout-board-$(date +%Y%m%d-%H%M%S)"

python3 "$REPO_DIR/targets/cpro/scripts/probe_hdc_ops.py" \
  --compile-only \
  --case mul_rows16_rank4 --case mul_rows16_compact \
  --case mul_rows128_rank4 --case mul_rows128_compact \
  --case sub_outer_rank4 --case sub_outer_compact \
  --case concat64_rank4 --case concat64_tree8 --case concat64_tree2 \
  --output "$GE_LAYOUT_PROBE_RUN" &&
python3 "$REPO_DIR/targets/cpro/scripts/probe_hdc_runtime.py" \
  --probe-run "$GE_LAYOUT_PROBE_RUN" --native-from "$OP_PROBE_RUN" \
  --refresh-runner --replay-all --output "$GE_LAYOUT_BOARD_RUN"

cat "$GE_LAYOUT_PROBE_RUN/report.txt"
cat "$GE_LAYOUT_BOARD_RUN/runtime-comparison.txt"
```

以上历史batch只包含这9项；复用原`OP_PROBE_RUN`的native/库身份，刷新小探针一次。原形和候选使用同一输入/原参考：Mul/Sub沿用FP16容差，Concat零容差；不更改运行库、整图或数值门槛。报告分别判断编译分区、Load、Process和数值结果，原形控制落CPU不表示候选失败；设备位置仍需独立证据。64路Concat每图需上传64个小输入文件，传输较慢不等于执行变慢。本次未进行性能测量。

其他CPU-only候选仍有MoE布尔/整数路由、TopK及图边界；CPU-only节点数不是分区数，即使上述探针通过也不能保证整图≤9分区。最终目标仍是精度达标后DFlash至少提速2倍，当前先打通完整执行链路。

### 历史：已执行的 MatMul 归约整图流程



用户回传 `reduce-mm-replay-20261026-104025` 的六项结果已全部 Load、Process、数值 PASS；每个主图编译分区为 NPU=1、CPU=0，实际执行位置仍为 `UNVERIFIED`。此前五项 IO `NOT_RUN` 已通过受限 ND 前补1适配后的复测，不再要求重复小图。完整双图执行、全图精度和2倍加速仍未验证。

当前 `export --fp16-reductions matmul` 为显式候选，默认 `native` 保留旧归约。候选只替换符合条件的 FP16 单轴归约；整数、非 FP16 及不符合条件的归约仍走 native。保留归约前已经舍入的 FP16 数据，Mean 以可精确表示的 `1/N` 权重收缩，避免先形成 FP16 总和再缩放；MatMul 的内部累加和舍入可能与原 Reduce 不同，小图通过不能替代全模型数值验证。

以下为已执行的整图流程，当前无需重复。复用 `$GE_PAIR_RUN/prepared` 和 `$GE_BOARD_RUN/reference.json`，不重新 prepare、不改专家模式、不更换库。沿用原 `CANN_ROOT`、DDK、resources/环境中的 OHOS SDK、HIAI 头文件和链接配置，以及 `SN`、`BIP`。每次使用新目录：

```bash
git -C "$REPO_DIR" pull --ff-only
export GE_MODEL_TOOL="$REPO_DIR/targets/cpro/scripts/direct_model_air.py"
export GE_REDUCE_MODEL_RUN="$AI_RUN_DIR/ge-reduce-model-$(date +%Y%m%d-%H%M%S)"
export HIAI_REDUCE_BUILD_DIR="$GE_REDUCE_MODEL_RUN/hiai-build"
export HDC_REDUCE_RUN_DIR="$GE_REDUCE_MODEL_RUN/board"

python3 "$GE_MODEL_TOOL" export \
  --prepared "$GE_PAIR_RUN/prepared" --ge-root "$CANN_ROOT" \
  --fp16-reductions matmul --output "$GE_REDUCE_MODEL_RUN/air" &&
python3 "$GE_MODEL_TOOL" compile \
  --air "$GE_REDUCE_MODEL_RUN/air" \
  --ddk-root "$CPRO_DDK_ROOT" --ddk-id "$CPRO_DDK_ID" \
  --output "$GE_REDUCE_MODEL_RUN/omc" --reject-cpu-fallback &&
python3 "$REPO_DIR/targets/cpro/scripts/build_hiai.py" \
  --output "$HIAI_REDUCE_BUILD_DIR" &&
python3 "$REPO_DIR/targets/cpro/scripts/run_hdc.py" \
  --omc "$GE_REDUCE_MODEL_RUN/omc" \
  --runner "$HIAI_REDUCE_BUILD_DIR/build/qwen35_dflash_hiai" \
  --reference "$GE_BOARD_RUN/reference.json" \
  --output "$HDC_REDUCE_RUN_DIR" --execution-only
```

四步以 `&&` 串联，导出或编译失败就停止后续构建及上板。新的正式 runner 含 `97dd374` 的受限 ND rank3 `[a,b,c] → [1,a,b,c]` 兼容，原 ND 尾补1继续允许；rank2 token 合同、宽度推导、字节检查和数值门槛不变。之前刷新的是独立 probe，不能拿旧正式二进制代替这次构建。若原 `build_hiai.py` 使用显式 resources 文件或兼容 SDK 版本，继续沿用同一配置，不临时换 SDK/库。

导出分别写出构造记录 `air/target/reduction-lowering.json`、`air/draft/reduction-lowering.json`，其中 `mode` 记录选择，`counts` 汇总 `total`、`replaced`、`retained`、`fp16_replaced`、`integer_retained` 和 `scan_prefix_replaced`。当前 tiny/static 预期 target/draft 分别替换421/10个 FP16 归约、保留16/1个整数归约；实际以该次导出报告为准，不将统计预期当作编译或执行证明。可直接反馈：

```bash
python3 - <<'PYCOUNTS'
import json
import os
from pathlib import Path

for name in ("target", "draft"):
    path = Path(os.environ["GE_REDUCE_MODEL_RUN"]) / "air" / name / "reduction-lowering.json"
    report = json.loads(path.read_text())
    print(name, report["mode"], report["counts"])
PYCOUNTS
cat "$HDC_REDUCE_RUN_DIR/result.json"
cat "$HDC_REDUCE_RUN_DIR/benchmark.stderr"
```

`--reject-cpu-fallback` 只检查主图或归属未知的 NPUCL 分区超限回退，不会证明所有算子进入 NPU。小图通过和归约替换都不保证完整图解除9分区上限。编译成功后，`--execution-only` 强制 target/draft 各执行一次，使用原参考的身份、prompt 与配置，不比较精度、不测 benchmark；成功为 `EXECUTION_COMPLETE`，精度/性能为 `NOT_EVALUATED`。当前优先打通执行，最终仍要求完整模型精度达标后 DFlash 至少提速2倍。

### 新图仍发生分区回退时

以下保留上次MatMul归约图的诊断命令；本轮row Mul新图应使用本节开头的`GE_ROW_CHAIN_INSPECT_RUN`命令。保留新目录的 AIR、recipe、OMC、OMG 日志和 `failure.json`，对这一次产物运行只读诊断；不要把旧 `GE_REEXPORT_RUN` 的日志用于解释新图，也不要直接重复编译：

```bash
export GE_REDUCE_INSPECT_RUN="$AI_RUN_DIR/ge-reduce-inspect-$(date +%Y%m%d-%H%M%S)"
python3 "$GE_MODEL_TOOL" inspect-compile \
  --air "$GE_REDUCE_MODEL_RUN/air" --omc "$GE_REDUCE_MODEL_RUN/omc" \
  --graph target --output "$GE_REDUCE_INSPECT_RUN"
cat "$GE_REDUCE_INSPECT_RUN/report.txt"
```

上例对应 target 失败；若失败的是 draft，把 `--graph` 改为 `draft` 并使用新的诊断输出目录。诊断按产物哈希关联构图与日志，不访问板端、不读取权重载荷。候选支持集及构造时边不等于最终分区或实际运行位置。

### 已通过的六项 MatMul 归约探针

候选将归约轴移到末轴、展平为二维，与常量右权重 `[1,N]` 做 `transpose_b=True` 的 MatMul，再还原输出维度。Sum 使用全1权重；本轮 Mean 长度8/16的 `1/N` 在 FP16 中可精确表示，不先生成 FP16 的总和再缩放，避免额外的求和中间值溢出。普通浮点输入、输出仍为 FP16；模型归约之前已经舍入的平方或其他乘积不可因此被提升精度或重新计算。

MatMul 和 Reduce 的累加顺序、舍入仍可能不同；六项按原容差通过，不代表 FP16 位级等价。本轮全图适配只在显式 `--fp16-reductions matmul` 下生效，默认 native 不变。

| case | 输入与等价归约 | 覆盖目的 |
|---|---|---|
| `reduce_mean_rms_matmul` | `[1,8,16]`，末轴 Mean，保留维度 | RMSNorm 宽度16 |
| `reduce_mean_head_matmul` | `[1,8,2,8]`，末轴 Mean，保留维度 | head norm 宽度8及 rank4 |
| `reduce_sum_scan_matmul` | `[2,1,64]`，末轴 Sum，保留维度 | GDN scan 长度64 |
| `reduce_sum_gdn_matmul_k64` | `[2,1,64,64]`，axis=2 Sum，去掉归约维 | GDN 非末轴归约 |
| `reduce_sum_gdn_matmul_k7` | `[2,1,7,7]`，axis=2 Sum，去掉归约维 | 非对齐短归约 |
| `reduce_sum_gdn_matmul_k1` | `[2,1,1,1]`，axis=2 Sum，去掉归约维 | 长度1边界 |

当时的小图复测命令如下，仅作历史复现。当前六项已通过，继续页首链接中的GDN整图流程；如果需要复现，保留原 `GE_REDUCE_MM_RUN`、原 `OP_PROBE_RUN`，仅新建结果目录：

```bash
git -C "$REPO_DIR" pull --ff-only
export GE_REDUCE_MM_REPLAY_RUN="$AI_RUN_DIR/reduce-mm-replay-$(date +%Y%m%d-%H%M%S)"
python3 "$REPO_DIR/targets/cpro/scripts/probe_hdc_runtime.py" \
  --probe-run "$GE_REDUCE_MM_RUN" --native-from "$OP_PROBE_RUN" \
  --refresh-runner --replay-all --output "$GE_REDUCE_MM_REPLAY_RUN"
cat "$GE_REDUCE_MM_REPLAY_RUN/runtime-comparison.txt"
```

两类来源用途固定：

- `--probe-run` 提供本轮模型、输入、参考和各自哈希，必须是没有 native 产物的 `COMPILE_COMPLETE` 批次，且 `board_requested=false`。
- `--native-from` 只提供原完整 native 构建身份、SDK、头文件、链接库和记录过的库集合。两批的规范化 `ddk_root`、`ddk_id`、`omg_sha256` 必须一致；原链接输入也必须来自该 DDK。工具校验两份来源及冻结库，身份不匹配或 native 记录残缺时在 HDC 前拒绝，不把它当作允许换库的入口。

刷新仅构建当前 `qwen35_hiai_op_probe`，不导出 AIR、不调用 OMG，不改两份源目录、旧 OMC 或系统库。上板仅运行 `candidate_without_ddk_hcl`：排除两个 DDK HCL 库并使用已验证的 vendor 搜索路径。旧53项或原47项不会随本轮候选一起重跑。默认回传校验后清理本次板端目录；需要保留现场时加 `--keep-board`。

已通过的结果保存在 `GE_REDUCE_MM_REPLAY_RUN/runtime-comparison.txt` 和该目录的 `results.tar.gz`，与原编译 `report.txt` 一起保留。编译失败的案例保持失败，不会被 replay 变成成功；每个已编译案例分别记录 Load、Process、数值状态和原编译分区。即使小图通过，运行位置仍为 `UNVERIFIED`，直到有独立执行证据；它也不能证明完整 target/draft 通过或分区数已下降。当前先打通执行，完整模型精度验证后置，最终仍要求精度达标后 DFlash 至少提速2倍。

已声明 blue profile 中另有 `RmsNorm` 的公开 schema、FP16 ND 配置及预编译 kernel 键，属于后续候选线索。本地 `kirin9030/next-6.0.1.0` 不能代替用户 `kirinx90/ddk_0806` 的可用性验证；本轮没有新增该探针，也没有将 GDN scan/三角归约视作可直接替换的 RMSNorm。

### 首次生成六项 MatMul 候选

当前六项已完成编译、执行及数值检查，以下仅供首次生成或历史复现。当前现场不要运行或重设 `GE_REDUCE_MM_RUN`；执行页首链接中的GDN整图流程。首次生成需要 NumPy、`CANN_ROOT`、`CPRO_DDK_ROOT`、`CPRO_DDK_ID` 和主机编译环境；`--compile-only` 不构建 runner，也不访问板端：

```bash
git -C "$REPO_DIR" pull --ff-only
export GE_REDUCE_MM_RUN="$AI_RUN_DIR/reduce-mm-$(date +%Y%m%d-%H%M%S)"
python3 "$REPO_DIR/targets/cpro/scripts/probe_hdc_ops.py" \
  --compile-only \
  --case reduce_mean_rms_matmul --case reduce_mean_head_matmul \
  --case reduce_sum_scan_matmul --case reduce_sum_gdn_matmul_k64 \
  --case reduce_sum_gdn_matmul_k7 --case reduce_sum_gdn_matmul_k1 \
  --output "$GE_REDUCE_MM_RUN"
cat "$GE_REDUCE_MM_RUN/report.txt"
```

首次编译完成后可使用上文历史小图的 `--native-from` 命令；当前现场已有通过结果，无需重复。

### 历史参考：只读编译诊断

本次诊断已完成，以下命令保留用于未来新失败产物，当前无需重跑。`GE_REEXPORT_RUN` 指向对应失败目录，`GE_INSPECT_RUN` 必须是新目录：

```bash
export GE_MODEL_TOOL="$REPO_DIR/targets/cpro/scripts/direct_model_air.py"
export GE_INSPECT_RUN="$AI_RUN_DIR/ge-inspect-$(date +%Y%m%d-%H%M%S)"
python3 "$GE_MODEL_TOOL" inspect-compile \
  --air "$GE_REEXPORT_RUN/air" \
  --omc "$GE_REEXPORT_RUN/omc" \
  --graph target --output "$GE_INSPECT_RUN"
```

命令只需 Python 标准库，验证 AIR manifest、编译 `failure.json` 或成功 manifest、recipe 的绑定；已有编译图记录必须通过历史 OMG 日志哈希检查。旧版早期失败没有编译图记录时，严格匹配失败图、阶段和日志路径，并标明日志哈希为本次诊断采集，不能冒充历史核验。命令不读取权重载荷、不访问板端。`report.txt` 给出候选分组，`report.json` 保留明细及原构图生产消费关系；旧 prepared 的 recipe 不能替代该次 AIR 的 recipe。主图、常量折叠、子图及未知范围分开，没有主图支持窗口时不关联常量图中的候选；GE 改名节点不按前缀猜测关联。构造时 shape/dtype 和候选跨设备边都不是最终分区或实际执行证据。

<a id="六项归约编译对照"></a>

### 历史参考：六项 rank/keep_dims 归约对照

已取得的结果如下，无需重复提交相同统计。全部为 FP16 数据、INT32 axes：

| case | 输入与归约 | 编译主图分区 |
|---|---|---|
| `reduce_mean_rms_rank3` / `reduce_mean_rms_rank2` | `[1,8,16]` / `[8,16]`，末轴 Mean，保留维度 | 两项均 NPU=0、CPU=1 |
| `reduce_sum_scan_rank3` / `reduce_sum_scan_rank2` | `[2,1,64]` / `[2,64]`，末轴 Sum，保留维度，非正输入 | 两项均 NPU=0、CPU=1 |
| `reduce_sum_gdn_dropdims` | `[2,1,64,64]`，axis=2 Sum，直接去维 | NPU=0、CPU=1 |
| `reduce_sum_gdn_keepdims` | 同输入及归约轴，保留维后 Reshape 为 `[2,1,64]` | NPU=1、CPU=1；ReduceSum 位置未确定 |

以下仅保留当时的复现命令；当前执行页首链接中的GDN整图流程：

```bash
export GE_REDUCE_PROBE_RUN="$AI_RUN_DIR/reduce-probe-$(date +%Y%m%d-%H%M%S)"
python3 "$REPO_DIR/targets/cpro/scripts/probe_hdc_ops.py" \
  --compile-only \
  --case reduce_mean_rms_rank3 --case reduce_mean_rms_rank2 \
  --case reduce_sum_scan_rank3 --case reduce_sum_scan_rank2 \
  --case reduce_sum_gdn_dropdims --case reduce_sum_gdn_keepdims \
  --output "$GE_REDUCE_PROBE_RUN"
```

`v18/sum_1` 的 `elementary_lib` 拒绝仍只有 `check elem support fail`，尚无具体 format 原因；这些结果不能反推 V0/V1。`COMPILED` 与 NPU 分区、板端执行和数值检查是不同的检查阶段。

### 明确修复后的重导出与上板流程

row Mul周围的连续Reshape合并已通过用户复测，19个Mul退出CPU-only候选列表；主图仍超过9个NPUCL分区。保留现有整图及探针产物，按页首链接导出GDN候选新图；[remaining-cpu批次](cpu-gap-probes.md)已有完整回传。完整双图执行尚未打通，详细报告、清理行为和后续benchmark入口见[HDC指南](hdc-board-run.md#当前双图采用可选-matmul-归约)。
