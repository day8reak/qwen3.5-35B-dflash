# 直接构图生成 AIR：服务器最小验证

当前优先验证 **GE C++ 构图 → AIR → 内部 OMG → OMC**。这条入口不加载 PyTorch、TorchNPU 或 TorchAir；先用一个 FP16 加法图确认主机保存与内部转换接口，再推进 Qwen/DFlash 的算子映射。完整模型现有的 `export-tiny-air` / `export-air` 仍使用 TorchAir，尚未迁移到直接 GE 构图。

CANN 9.1 的 [`Graph::SaveToFile`](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendgraphapi/atlasgeapi_07_0017.html) 明确保存 AIR，头文件为 `graph/graph.h`，库为 `libgraph.so`。[官方构图教程](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/devaids/aoe/aoeep_16_065.html) 将保存操作放在 `aclgrphBuildInitialize` 之前。本样例只构图和保存，不调用 GE 会话、ACL 初始化或 NPU 执行接口。

**当前验证范围见 [validation.md](validation.md)：服务器真实 `libgraph.so` 链接与 AIR 保存、内部 OMG 转换和 X90 执行需要在用户环境验证。** Python 编排测试不能证明这些步骤通过。

## 图与产物

图为 `y = x + bias`，输入、常量、输出均为 ND、FP16、`[1,4]`：

| 数据 | 值 |
|---|---|
| 输入 `x` | `[0, 1, 2, 3]` |
| 内嵌常量 `bias` | `[1, 2, 3, 4]` |
| 预期 `y` | `[1, 3, 5, 7]` |

所有数值均可用 FP16 精确表示。这里声明的是内置 Data/Const/Add 的图原型，不是自定义 AscendC 算子，没有新内核需要安装。

样例有独立的 `ge-air-smoke` 身份，不是 Qwen target/draft 双图，也不是 DFlash 模型。它不能直接交给现有 `compile-omc`、`board-command` 或 `qwen35_dflash_hiai` 的双模型推理入口。

## 1. 服务器：生成 AIR

在自己的 CANN 9.1 环境执行。`CANN_ROOT` 填包含 `set_env.sh` 的实际版本目录；它与安装器的 `--install-path` 前缀不是同一个概念。不要把内部端侧 DDK 的 `libgraph_host.so` 改名成 `libgraph.so`。

```bash
cd /your/repos/qwen3.5-35B-dflash
git pull --ff-only
export REPO_DIR="$PWD"
export AI_RUN_DIR="${AI_RUN_DIR:-/your/run/storage/ge-air-001}"
mkdir -p "$AI_RUN_DIR"
export PYTHONDONTWRITEBYTECODE=1

export CANN_ROOT=/your/own/ascend/cann-9.1.0
source "$CANN_ROOT/set_env.sh"
cd "$AI_RUN_DIR"
python3 "$REPO_DIR/targets/cpro/scripts/direct_air_smoke.py" export \
  --ge-root "$CANN_ROOT" --output "$AI_RUN_DIR/ge-air"
```

此脚本只使用 Python 标准库编排 C++ 编译和执行，不要求卸载或导入已有 TorchNPU。主 CLI `preflight` 的 `air_export_ready` 检查的是 TorchAir，不作为这个独立入口的准入条件。需要主机 C++ 编译器，以及同一套 CANN 的 GE 头文件、`libgraph.so` 和其实际依赖（包括该版本提供的 `libgraph_base.so`）。也可通过 `--resources` 指定私有资源清单中的 `ge_root`；不要将自己的路径写进仓库。缺少原生依赖时仍会失败，编译和运行日志用于定位，不能靠 Python 后端隔离修复。

编译默认使用 `--cxx11-abi 0`；该值必须匹配实际 GE 库的 C++ ABI。如果你们 SDK 的编译约定为新 ABI，则显式传 `--cxx11-abi 1`。链接错误先看 `build.log` 的具体符号与库路径，不要用端侧 `libgraph_host.so` 或桩库替换真实主机库。

成功后检查 `ge-air/manifest.json` 的状态应为 `AIR_EXPORTED`。AIR、二进制测试输入和参考输出、生成器程序、命令、日志及哈希都保留在该目录；输出拒绝覆盖，失败重试换新目录。

### 已遇到的 `x failed, graph status=4294967295`

`94c053e` 的样例在保存前执行 `VerifyGraph`，其中 `FindOpByName("x")` 查询失败。这发生在查找节点阶段，不是 NPU 执行失败，也不能据此判断缺少 HCCL 或需要 GE/ACL 设备初始化。

GE 的 [`AddOp`](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendgraphapi/atlasgeapi_07_0014.html) 注册表与 `SetInputs` 创建的计算图不同；`FindOpByName` 查询前者。保存后 `LoadFromFile` 恢复计算图，也不自动恢复这个注册表。修复后的保存前和回读后验证统一使用 [`GetAllNodes`](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendgraphapi/atlasgeapi_07_0027.html) 与 GNode 接口，保留节点类型、FP16 描述、输入索引和常量内容检查。

更新代码后使用新输出目录重试，保留已有失败日志：

```bash
cd "$REPO_DIR"
git pull --ff-only
export GE_AIR_RETRY="$AI_RUN_DIR/ge-air-$(date +%Y%m%d-%H%M%S)"
python3 "$REPO_DIR/targets/cpro/scripts/direct_air_smoke.py" export \
  --ge-root "$CANN_ROOT" --output "$GE_AIR_RETRY"
```

沿用上文自己的 CANN 环境。构建或导出子进程失败时，新脚本会显示阶段、日志路径和受限长度的日志尾部；完整日志与 `failure.json` 仍留在本次目录。`FileExistsError` 表示输出路径已被使用，另选目录即可，不需要删除旧产物。

只有新目录的 manifest 为 `AIR_EXPORTED` 时再转换；下面 `compile` 的 `--air-bundle` 要改成实际成功的目录（上述重试为 `$GE_AIR_RETRY`）。

## 2. 服务器：内部 OMG 转 OMC

切回已经加载内部 DDK AscendC 环境脚本的终端，恢复相同的 `REPO_DIR` 和 `AI_RUN_DIR`。无需重复拼接其工具路径，也不需要把 CANN 根目录赋给 `CPRO_DDK_ROOT`。

```bash
export REPO_DIR=/your/repos/qwen3.5-35B-dflash
export AI_RUN_DIR=/your/run/storage/ge-air-001
export PYTHONDONTWRITEBYTECODE=1
export CPRO_DDK_ROOT="$(realpath "${DDK_PATH:?先加载内部 DDK 环境}")"
export CPRO_DDK_ID=100.520.020.010
cd "$AI_RUN_DIR"
python3 "$REPO_DIR/targets/cpro/scripts/direct_air_smoke.py" compile \
  --air-bundle "$AI_RUN_DIR/ge-air" \
  --ddk-root "$CPRO_DDK_ROOT" --ddk-id "$CPRO_DDK_ID" \
  --output "$AI_RUN_DIR/ge-omc"
```

实际转换仍是 `OMG --framework=1 --target=omc --platform=kirinx90 --use_origin_format=true --weight_data_type=FP16`。脚本在 run 内临时目录使用短相对路径，保留真实 OMG 日志，并检查输入哈希和非空 OMC。成功转换只标记 `COMPILED`，设备验证仍为 `PENDING`。

AIR 解析失败、算子不支持或形状错误时，应根据内部 OMG 日志调整构图与描述；不能把文件改后缀、改成别的 framework 值或生成一个空 OMC 当作成功。

## 3. 板端与下一步

本次先验证服务器生成与转换。当前 `qwen35_dflash_hiai` 要求 target 和 draft 两个模型的固定 ABI，不能加载上述单图；单图的板端加载/执行尚未接入。AIR 留在服务器，后续应传 OMC、对应单图 HIAI 程序及测试输入，再回传结果与参考比较。现有双图小模型的完整板端步骤仍见 [tiny-validation.md](tiny-validation.md)，但其 AIR 导出环境要求与本样例不同。

先保留本次 manifest、编译日志、运行日志和 OMG 日志。小图通过后仍需实现完整 Qwen/DFlash 的注意力、GDN、MoE、权重加载、量化算子与双图 ABI，重新做数值比较。直接保存 AIR 不会自动完成离线量化，也不会解决 35B 的内存预算。
