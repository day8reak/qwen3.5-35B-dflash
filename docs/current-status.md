# 当前阶段：暂缓精度排查，清理运行目录

2026-11-05：用户接受当前约百分之一量级的残余误差，暂缓进一步MatMul／精度排查；先释放服务器运行盘，再继续性能工作。原数值验收仍为FAILED，不修改golden或容差。当前hidden相对L2为1.1563%～1.7719%，新增KV最差0.6084%，两模式各83/86个Top1匹配。[完整结果](results.md)。

当前RoPE候选OMC SHA为 `b43ae9cfcd50999277829a21adc24099170d4a7c4856b70729385b59d362a828`。同OMC的原38次回放及追加20次调用均未复现176／177重复行，追加KV逐位稳定；记录为“本轮未复现”，未认定根因已修复。无需再执行55点探针、KV复现或新的精度支持检查。

## DDK服务器清理

沿用现有 `REPO_DIR` 和 `AI_RUN_DIR`。只访问服务器文件，不连接板端，不重编模型。运行配置默认读取 `$AI_RUN_DIR/runtime/runtime.json`；若实际配置在其他位置，用 `--config` 指定。

```bash
git -C "$REPO_DIR" pull --ff-only

for mode in --compiler-scratch --integration-sidecars --layer0-probes; do
  python3 "$REPO_DIR/targets/cpro/tools/clean_draft_probes.py" \
    --root "$AI_RUN_DIR" "$mode" --apply
done

df -h "$AI_RUN_DIR"
du -h -d 2 "$AI_RUN_DIR/draft/tests/top1-integration" | sort -h
```

三个范围分别为：

- 旧 `pytorch-omc`、`pytorch-optimized-omc` 的已完成编译临时副本，保留打包模型和报告。
- 当前 `top1-integration/compiled`、`rope-fp32/compiled` 中已写入打包OMC的重复 `.weight` 文件。核对打包模型、编译清单和文件内容后才删除；保留compiler AIR、输入权重快照和外置OMC描述，后续ABI绑定仍需读取它们。
- 已完成的 `hidden-layer0/{compiled,run,result}`。保留 `retained-summary.json` 和探针目录根部的分析JSON；已清过则跳过。

当前模型、准备权重、AIR导出、运行器、完整精度回放输入／参考／快照及整行重复证据保留。工具对活动进程、锁、运行配置引用或归属不明项报告保留；不会清理整个 `draft/tests`。`PARTIAL` 时查看 `retained` 原因，其他独立范围仍可清理。实际释放空间以 `df` 和 `free_bytes_after` 为准，不能把逻辑文件大小当作实际释放量。

新增重复权重清理完成12项主机检查，包括预览／删除／重复执行、中断续清、内容及清单冲突、符号链接、运行配置引用、锁与模拟活动进程保护、ABI／源文件保留和旧模式回归。尚未在用户服务器执行；回传三份清理摘要及磁盘输出即可继续判断是否需要清理其他目录。
