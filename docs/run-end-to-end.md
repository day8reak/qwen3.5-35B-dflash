# 完整流程执行

板端自主完成文字编码→Prefill一次→常驻Draft/Decode循环→文字解码。服务器只准备、下发和回收。普通运行只检查文件大小和张量ABI，不再依赖历史测试的绝对路径、哈希或时间戳；默认关闭dump，开启Draft预读。

## 一次性导入配置、整理已有目录

先结束并回收正在运行的板端任务。以下从已完成的文字闭环导入模型、词表、ABI和80状态映射；只需做一次，不重编模型。若已生成`runtime.json`，跳过导入命令。

```bash
git -C "$REPO_DIR" pull --ff-only

python3 "$REPO_DIR/targets/cpro/scripts/export_runtime_config.py" \
  --prepared "$AI_RUN_DIR/dflash-board-text-preread8-run/prepared" \
  --output "$AI_RUN_DIR/runtime.json"

# 预览移动、删除和保留项
python3 "$REPO_DIR/targets/cpro/scripts/organize_run_directory.py" \
  --root "$AI_RUN_DIR" --clean-completed

# 实际移动并清掉已确认结束的旧测试
python3 "$REPO_DIR/targets/cpro/scripts/organize_run_directory.py" \
  --root "$AI_RUN_DIR" --clean-completed --apply
```

整理后按`prefill/`、`decode/`、`draft/`、`common/`存放依赖，`work/`放当前测试，`results/`保留简短结果及profiling CSV。实际同盘移动，不复制大权重；配置路径自动更新。用途不明或退出状态不明的目录保留，见命令输出的`keep`。中断后重新执行同一整理命令即可继续。

`runtime.json`使用相对路径，可以编辑模型、词表位置和默认参数；`runtime-assets.json`保存必要ABI与状态映射。模型/权重内容替换后，须更换对应`cache_key`，避免复用旧板端缓存。它是部署版本标识，不表示内容已校验。生成新标识：

```bash
python3 -c 'import secrets; print(secrets.token_hex(32))'
```

## 输入文字并启动

同一测试始终复用下列固定目录。前一次已确认结束时，脚本自动保存最新简要结果、清掉旧临时目录并重新创建；尚未确认退出时先回收。当前模板后的prompt＋生成预算须≤64，固定生成指定数量，不按EOS提前停止。

```bash
export RUN_DIR="$AI_RUN_DIR/work/dflash-text"
export RECOVERY_DIR="$AI_RUN_DIR/work/dflash-text-result"
export PROMPT='你好，请介绍一下自己。'

python3 "$REPO_DIR/targets/cpro/scripts/run_board_serial_dflash.py" \
  --config "$AI_RUN_DIR/runtime.json" \
  --text "$PROMPT" \
  --max-new-tokens 8 --max-proposals-per-round 2 \
  --apply --detach --output "$RUN_DIR"
```

## 回收结果

结果包含生成文字、token ID、接受率、Prefill/Decode/Draft时延和加载耗时。回收不重跑模型；断线后也使用原请求路径。

```bash
export RUN_DIR="$AI_RUN_DIR/work/dflash-text"
export RECOVERY_DIR="$AI_RUN_DIR/work/dflash-text-result"

python3 "$REPO_DIR/targets/cpro/scripts/run_board_serial_dflash.py" \
  --recover "$RUN_DIR" --output "$RECOVERY_DIR"
```

需要清掉已结束的测试目录时，重复执行整理命令即可；正常复测直接重用启动命令，不再创建`v1/v2/v3`。
