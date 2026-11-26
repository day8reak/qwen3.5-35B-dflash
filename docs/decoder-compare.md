# 完整 decoder 的三端精度对比

78 采集 YAML 输入对应的 310P 参考、计算 CPU golden 并导出 CPro AIR。35 编译 OMC、在板上运行，把 CPU、310P、CPro 的结果写成 3×3 矩阵。两机使用相同版本的仓库。

## 一次配置

在每台主机的仓库根目录执行一次，然后编辑该机的 `decoder.env`。已有配置时直接编辑已有文件。

```bash
mkdir -p ~/.config/qwen35-cpro
cp -n targets/cpro/scripts/decoder.env.example ~/.config/qwen35-cpro/decoder.env
```

| 位置 | 配置 |
|---|---|
| 两机 | `AI_RUN_DIR` 是该机可写的运行根目录。设置相同的 `DECODER_MODE`、`DECODER_PHASE`。命令行的 `--mode`、`--phase` 可覆盖本次选择。 |
| 78 | `TARGET_SOURCE` 是含 `inference_segment.py` 的 310P 工程，`CPRO_SOURCE` 是原始 CPro 工程。两份配置 `NZ_CONFIG`、`TARGET_CONFIG` 默认各自指向对应工程的 `config/qwen3_5_moe.yaml`。使用同一份 YAML 时可将两项设为同一绝对路径。 |
| 35 | 加载 DDK 环境，设置 `CPRO_DDK_ROOT`、HDC 服务地址 `BIP` 和设备序列号 `SN`。DDK 需要原模型算子及 FP16 输入输出的 SiluPatch、SoftplusPatch。运行配置默认是 `$AI_RUN_DIR/runtime/runtime.json`。 |
| 35 获取 AIR | 配置 SSH 的 `air` 端点，首次可执行 `python targets/cpro/scripts/pull_artifacts.py --init AIR主机名 OMC主机名`。如果两机目录不同，设置 `DECODER_REMOTE_RUN_ROOT` 指向 **78 的 case 父目录**。共享同一绝对路径时直接读取本地导出。 |

入口也会读取已导出的环境变量。可用 `DECODER_ENV=/绝对路径/decoder.env` 指定配置文件。示例文件列出了三个 conda 环境的默认名以及可覆盖的工具路径；环境名设为 `current` 会用当前 Python。[详细环境配置](environment-setup.md)

### 输入 prompt

编辑 78 上 `NZ_CONFIG` 的 `prompts`。每次对比取第一条，例如

```yaml
prompts:
  - 世界上最高的山是哪座山？
```

原 310P 驱动用自己的 chat template 和 tokenizer 生成输入。令实际 token 数为 N，本流程支持一次 T64 prefill，即格式化后 1–64 个 token。Prefill 比较前 N 行，logits 取位置 N−1。Decode 从这次 prefill 的实际预测 token 和层状态开始，比较位置 N 的一步输出。更换 prompt 后重新运行 78 和 35 的命令即可。

### 自己维护 CPro 源码

在 78 的配置中设置 `CPRO_PATCHED_SOURCE=/绝对路径/cpro-edit`，选择运行结果目录外的位置。第一次执行 `prepare` 时会从 `CPRO_SOURCE` 生成补丁后的 Python 源码、模型文件和 `source.diff`。如果想先 review 或修改，可以单独执行

```bash
bash targets/cpro/scripts/decoder_compare.sh source
```

之后可直接编辑该目录里的 `models/modeling_qwen3_5_moe_hiai_cpro.py`、wrapper、`utils.py` 或 `export_glm_model_qwen3_5_moe.py`。当 `CPRO_PATCHED_SOURCE` 已包含导出器和模型源码时，所有模式都从那里加载代码，重跑会保留手改内容。也可以直接指向自己维护好的兼容源码目录；其导出接口、自定义算子注册、输出形状和 dtype 需与该模式一致。权重仍从配置所指的位置加载。若只提供现成的源码目录，可省略 `CPRO_SOURCE`，在 `TARGET_CONFIG` 指定配置文件，并将相对资源路径按该目录设置或改为绝对路径。

希望采用更新版的自动补丁时，给 `CPRO_PATCHED_SOURCE` 选一个新目录并执行 `source`，再对比合并自己的修改。未设置该变量时，每次 `prepare` 都从原 CPro 生成本次的 `export/source/`。

## 每次运行

78

```bash
bash targets/cpro/scripts/decoder_compare.sh prepare
```

35

```bash
bash targets/cpro/scripts/decoder_compare.sh board
```

| `DECODER_MODE` | 静态 OMC 数 | 范围 | Prefill 目录名 |
|---|---:|---|---|
| `linear` | 1 | L0 完整 linear decoder 和最终 norm/head | `1-linear-prefill-t64` |
| `full` | 1 | 从实测 L3 入口运行完整 full decoder 和最终 norm/head | `1-full-prefill-t64` |
| `4` | 1 | L0–L3 连续执行，再接最终 norm/head | `4-prefill-t64` |
| `40` | 1 | L0–L39 连续执行，再接最终 norm/head | `40-prefill-t64` |
| `layerwise` | 40 | 每层一个完整 decoder OMC，层间传递板端 hidden，L39 接最终 norm/head | `40x1-prefill-t64` |

Decode 将表中目录名的 `prefill-t64` 换成 `decode-t1`。默认父目录是 `$AI_RUN_DIR/target/tests`，可用 `DECODER_RUN_ROOT` 指定 `AI_RUN_DIR` 下的其它父目录。可先用 `bash targets/cpro/scripts/decoder_compare.sh plan` 查看当前配置对应的目录和 OMC 数。

## 覆盖和结果

78 的普通 `prepare` 清空同一 case，重新写入 `export/`。35 的普通 `board` 清空该 case 的 `board/`，使用最新的导出做完整对比。共享目录时，78 重跑会同时清掉此 case 旧的板端结果。并行启动同一 case 的本地命令会排队。若选中的目录已经有其它用途的文件，脚本会提示选择其它父目录。

- `board/matrices.csv` 给出三端的 3×3 矩阵，行是 actual，列是 reference。`board/comparison.json` 包含 prompt、logits 位置、各端预测 token 和未达阈值的张量。逐层模式还写 `board/Lxx/matrices.csv`。
- 线性层比较完整 conv/recurrent 状态，full 层比较本次有效的 K/V，并检查物理写入区以外的缓存。小范围的 logits 是在所选层的末端接原模型 head 得到的诊断值。
- `40 + prefill` 沿用原 CPro wrapper 和量化 embedding 入口。其它组合从实测 310P 入口 hidden 和各层状态出发，三端各自向后计算。默认补丁包括 W4 scale、RoPE、MoE、GDN 四维卷积、FP16 SiluPatch 和 SoftplusPatch。[整网 prefill 说明](target-prefill40-single.md)
- 只有逐层上板中断需要续跑时，在同一份导出上执行 `bash targets/cpro/scripts/decoder_compare.sh board --mode layerwise --recovery`。它核对模型、设备、输入和已保存的输出，接着执行缺失层。

### 板端传输和空间回收

上板日志会在命令前标明当前层和操作，包括 OMG 编译、发送 OMC 和外置权重、发送输入、HIAI V2 执行、回收输出和清理。统一入口每张 OMC 用一次 `hdc file send` 发送整包，一次 `hdc file recv` 回收整个输出目录。需另行读取运行时槽序的分段诊断会先部署模型，再按探测到的槽序一次发送输入。收到的原始文件保存在本次服务器 replay 下的 `outputs/`。

运行成功且输出文件清单、SHA-256、槽位、字节数全部核对后，脚本先保存 `result.json`，再删除本次 `/data/local/tmp/qwen35_step_*` 目录。回执中的 `board_cleanup=REMOVED` 表示已确认清除。如果执行或回收失败，从回执的 `remote` 字段找到保留的板端现场。
