# Target / Draft 直接 GE 构图 → AIR

新增入口为 `targets/cpro/scripts/direct_model_air.py`。target 和 draft 的图定义分别在 `ge_target.py`、`ge_draft.py`；通过主机 GE C++ 接口保存 AIR，生成端不导入 PyTorch、TorchNPU、TorchAir，也不初始化 NPU。原有 PyTorch 模型、参考、`export-air` 与 `export-tiny-air` 全部保留。

2026-10-22 用户在位置截断修复后反馈 **tiny target/draft 双图 AIR→OMC 编译成功**，记录为USER_REPORTED_PASS；尚未取得独立产物哈希和HIAI板端结果。下一步按 [直接 GE tiny OMC 上板指南](direct-ge-board.md) 生成匹配参考、交叉编译HIAI程序并上板。成功的OMC和prepared直接复用，不必重导。

下文保留首次生成流程和历史失败修复；算子核查见 [兼容性记录](ge-operator-compatibility.md)。tiny编译成功不表示完整35B权重、量化、内存或性能已经通过。

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
  --output "$GE_OMC_DIR"
```

沿用现有 `compile_air`：`--framework=1 --target=omc --platform=kirinx90 --weight_data_type=FP16`，两个 OMG 子进程各有独立的 run 内临时目录和日志。成功为 `COMPILED`，设备状态仍是 `PENDING`。这一步可能暴露内部 DDK 的算子、属性或布局差异，应按具体 `omg.log` 修图。

### 内部 OMG 兼容性修正与重试

**以下为历史失败及重建说明。用户已反馈tiny双图编译成功；已有成功产物请直接进入[上板步骤](direct-ge-board.md)。**

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

以下为此前 BatchMatMul 更新的命令；本轮请使用上方 `air-clamp` / `omc-clamp` 命令。

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

先在新目录生成 tiny 结构候选及独立参考，再导出编译。沿用已有 `AI_RUN_DIR`、`CANN_ROOT`、`CPRO_DDK_ROOT` 和 `CPRO_DDK_ID`：

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

每一步成功后再执行下一步。plan、prepared、AIR 和 OMC 的 manifest 都记录 `expert_dispatch`，export 按 prepared 的选择重建图；老 prepared 缺少该字段时沿用 static。全量旧算子复测仍可用 `probe_hdc_runtime.py --replay-all`，但它复用旧 OMC，不能代替本节新结构的 OMG/板端验证。正式双图上板入口现使用探针已验证的候选部署；先按 [只执行模式](hdc-board-run.md#当前优先复用-omc-打通双图执行) 复用旧 OMC 打通链路，再验证本节结构候选。

可选的主机 GDN 诊断入口为 `targets/cpro/scripts/diagnose_gdn_precision.py`，输出逐 chunk FP16/FP32 输出与状态误差。无输入时仅跑小型合成样例，不能确认真实模型的精度根因。当前优先上板执行，此诊断不作为前置步骤，也不改变部署 FP16。
