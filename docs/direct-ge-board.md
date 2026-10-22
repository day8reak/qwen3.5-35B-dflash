# 直接 GE tiny OMC：服务器准备与 HIAI 上板

2026-10-22 用户反馈 target/draft 双图 AIR→OMC 编译成功。当前为 **USER_REPORTED_PASS**；尚未取得独立产物哈希或板端结果。使用已经成功的 OMC 和同一份 prepared，继续下面步骤；无需重新 prepare/export/compile。

这是 hidden16、词表64的小模型联调。它验证接口和普通/DFlash token 一致性，不代表完整35B量化、质量、内存或性能已经通过。

## 1. 服务器：锁定成功目录，生成匹配参考

保留已有 `REPO_DIR`、`GE_PAIR_RUN`、`AI_RUN_DIR`。`GE_OMC_DIR` 必须与成功那次 compile 的 `--output` 完全一致：最近贴出的命令使用 `$GE_PAIR_RUN/omc`；如果实际使用了 `omc-clamp-*`，将下面赋值换成那个完整路径，不重新创建目录。

在已安装 PyTorch/Transformers 的模型 Python 环境执行：

```bash
cd "$REPO_DIR"
git pull --ff-only
export PYTHONDONTWRITEBYTECODE=1
export MODEL_PYTHON="$(command -v python3)"
export PYTHONPATH="$REPO_DIR/model:$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export GE_MODEL_TOOL="$REPO_DIR/targets/cpro/scripts/direct_model_air.py"
export GE_OMC_DIR="$GE_PAIR_RUN/omc"
export GE_BOARD_RUN="$GE_PAIR_RUN/board-$(date +%Y%m%d-%H%M%S)"
mkdir "$GE_BOARD_RUN"

"$MODEL_PYTHON" - <<'PY'
import json, os
from pathlib import Path
from qwen35_moe_dflash.artifacts import checked_file
root = Path(os.environ["GE_OMC_DIR"])
m = json.loads((root / "manifest.json").read_text())
assert m["status"] == "COMPILED" and m["artifact_kind"] == "qwen35-cpro-omc-pair"
assert m["sequence_length"] == 8 and m["vocab_size"] == 64
assert {g["name"] for g in m["graphs"]} == {"target", "draft"}
for g in m["graphs"]:
    checked_file(root, {**g, "path": g["omc"]})
    print(g["name"], g["omc"], g["sha256"])
PY

"$MODEL_PYTHON" "$GE_MODEL_TOOL" reference-tiny \
  --prepared "$GE_PAIR_RUN/prepared" \
  --prompt-token-ids 1,2,3 --max-new-tokens 4 \
  --output "$GE_BOARD_RUN/reference.json"

"$MODEL_PYTHON" - <<'PY'
import json, os
from pathlib import Path
run = Path(os.environ["GE_BOARD_RUN"])
reference = json.loads((run / "reference.json").read_text())
omc = json.loads((Path(os.environ["GE_OMC_DIR"]) / "manifest.json").read_text())
assert reference["status"] == "PASS"
assert reference["source"] == omc["source"], "prepared and OMC weights differ"
with (run / "prompt.json").open("x") as out:
    json.dump({"prompt_token_ids": reference["prompt_token_ids"]}, out)
PY
```

逐条成功再继续。参考直接读取这批 prepared 的权重，不能用旧 `cli demo` 重新生成另一组随机权重。

## 2. 服务器：交叉编译正式程序

已有变量可直接复用。若未配置，按实际内部路径填写：

```bash
export OHOS_NATIVE_SDK=/your/ohos/native
export HIAI_INCLUDE_DIR=/your/internal/hiai/include
export HIAI_LIBRARIES=/your/internal/ohos-arm64/libhiai.so

"$MODEL_PYTHON" "$REPO_DIR/targets/cpro/scripts/build_hiai.py" \
  --output "$GE_BOARD_RUN/hiai-build"
cat "$GE_BOARD_RUN/hiai-build/result.json"
```

`HIAI_INCLUDE_DIR` 中须有 `HiAiModelManagerService.h`；多个链接库用冒号分隔，具体库名以内部DDK为准。若设置了 `QWEN35_RESOURCES`，其中非空的 `hiai_libraries` 优先于环境变量；应同步更新该资源文件。使用 HarmonyOS ARM64 库，AscendC 脚本中的服务器 x86_64 工具库不适用。需要指定系统兼容版本时使用脚本的 `--compatible-sdk-version`（默认21）。结果应为 `LINKED`，程序为 `hiai-build/build/qwen35_dflash_hiai`；`--headers-only` 和主机fake程序不能替代正式链接。

## 3. 传到板端的文件

使用你们现有传输通道。选一个新的、允许执行并访问HIAI的部署目录，下文以 `/data/local/tmp/qwen35-tiny-001` 为例。

| 服务器来源 | 板端相对部署目录 |
|---|---|
| `$GE_BOARD_RUN/hiai-build/build/qwen35_dflash_hiai` | `bin/qwen35_dflash_hiai` |
| `$GE_OMC_DIR/` 的目录内容，含manifest与全部附属文件 | `models/tiny-omc/` |
| 板端未提供的匹配运行库及其依赖 | `lib/` |

用 `readelf -d` 查看正式程序依赖，补齐板端缺失的同平台运行库（包括所需C++运行库）。AIR、prepared、原始权重、Python、CANN、OMG和编译SDK留在服务器。OMC内部是否包含CPU子图以实际编译/运行结果为准，不能仅凭编译成功认定全部算子都跑NPU。

## 4. 板端：探测运行库

```sh
export BOARD_ROOT=/data/local/tmp/qwen35-tiny-001
mkdir -p "$BOARD_ROOT/lib"
chmod +x "$BOARD_ROOT/bin/qwen35_dflash_hiai"
export LD_LIBRARY_PATH="$BOARD_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
set -C
"$BOARD_ROOT/bin/qwen35_dflash_hiai" --probe \
  > "$BOARD_ROOT/runtime-probe.json" 2> "$BOARD_ROOT/probe.stderr.log"
```

退出码应为0，JSON中的 `npu_supported` 应为true。把 `runtime-probe.json` 取回服务器，存为 `$GE_BOARD_RUN/runtime-probe.json`，并记录实际固件版本。probe检查运行库/NPU能力，尚未加载这两张OMC。

## 5. 服务器生成运行脚本，板端执行

服务器上设置与板端一致的路径，以及实际固件版本：

```bash
export BOARD_ROOT=/data/local/tmp/qwen35-tiny-001
export FIRMWARE_ID='填写实际固件版本'
RUNTIME_VERSION=$("$MODEL_PYTHON" -c \
  'import json,sys; p=json.load(open(sys.argv[1])); assert p["npu_supported"] is True; print(p["runtime_version"])' \
  "$GE_BOARD_RUN/runtime-probe.json")

"$MODEL_PYTHON" -m qwen35_moe_dflash.cli board-command \
  --omc "$GE_OMC_DIR" --prompt "$GE_BOARD_RUN/prompt.json" \
  --board-model-dir "$BOARD_ROOT/models/tiny-omc" \
  --runner "$BOARD_ROOT/bin/qwen35_dflash_hiai" \
  --board-output "$BOARD_ROOT/benchmark.json" \
  --runtime-version "$RUNTIME_VERSION" --firmware-id "$FIRMWARE_ID" \
  --mode benchmark --max-new-tokens 4 --output "$GE_BOARD_RUN/command.json"

"$MODEL_PYTHON" - <<'PY'
import json, os
from pathlib import Path
run = Path(os.environ["GE_BOARD_RUN"])
command = json.loads((run / "command.json").read_text())["shell"]
with (run / "run-tiny.sh").open("x") as out:
    out.write("#!/bin/sh\nset -eu\n" + command + "\n")
PY
```

将 `$GE_BOARD_RUN/run-tiny.sh` 传到板端 `$BOARD_ROOT/run-tiny.sh`。在第4节的板端shell中执行；新开shell先恢复 `BOARD_ROOT`、`LD_LIBRARY_PATH` 和 `set -C`：

```sh
sh "$BOARD_ROOT/run-tiny.sh" \
  > "$BOARD_ROOT/benchmark.stdout.log" 2> "$BOARD_ROOT/benchmark.stderr.log"
```

benchmark先检查普通/DFlash一致性，再各做3次预热与10次测量。程序按manifest自动使用tiny接口宽度和模型哈希，不手工填写35B尺寸。把 `benchmark.json` 取回服务器，存为 `$GE_BOARD_RUN/benchmark.json`，同时保留stdout/stderr；失败先查看stderr，不继续比较。

## 6. 服务器：与参考比较

```bash
"$MODEL_PYTHON" -m qwen35_moe_dflash.cli compare \
  --report "$GE_BOARD_RUN/benchmark.json" \
  --reference "$GE_BOARD_RUN/reference.json" \
  --omc "$GE_OMC_DIR" --output "$GE_BOARD_RUN/comparison.json"
```

`PASS` 表示本次报告的普通/DFlash token与停止原因匹配参考，并通过权重、OMC等身份检查；不等于逐层特征数值或完整35B模型验证。所有产物拒绝覆盖；重试使用新的板端部署目录及新的报告/脚本路径，保留已生成OMC和失败证据。
