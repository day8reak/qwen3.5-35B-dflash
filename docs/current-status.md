# 当前阶段：归档约60ms Draft，转向Target精度

更新：2026-11-06。用户决定暂停离线权重转置及其它Draft性能实验，保留最新packed基线。模型结构已按当前PyTorch图更新，优化与各轮时延见[优化策略](optimization-strategy.md)。

本次保留 `offline-weights/full/baseline/model.omc`，1791018585字节，SHA256 `bc693db0bc3833d532f6b2158b3f1f19939311beb6941ff43c8c05b9aa4631dd`。最新38次调用加权准备＋API为60.4067ms；nd_n候选73.6574ms，不采用。两模型原数值门槛仍FAILED，488个对照区域中只有isolated第0例hidden不同；其余487个区域相等。详细限制见[结果](results.md)。

## 用户现在执行什么

以下仅在Draft归档／清理尚未完成时，由用户在已有产物的DDK服务器执行；新窗口的Target工作从[精度交接](target-accuracy-handoff.md)开始。该命令不导出、不编译、不启动设备：先从现有编译文件导入一次ABI绑定、保存紧凑对照摘要，将AIR同盘移动到`draft/air`，将自包含OMC及必要元数据发布到`draft/omc`，确认能从公共目录重新绑定后，再清空 **整个 `draft/tests` 的全部内容**。

```bash
(
set -eu
cd "${REPO_DIR:?}"
git pull --ff-only origin main
: "${AI_RUN_DIR:?}" "${CPRO_DDK_ROOT:?}"
full="$AI_RUN_DIR/draft/tests/offline-weights/full"
python3 targets/cpro/scripts/archive_incremental_draft.py \
  --root "$AI_RUN_DIR" \
  --compiled "$full/baseline" --air-export "$full/export" \
  --source "$AI_RUN_DIR/draft/source/prepared" \
  --comparison "$full/nd_n-result" --ddk-root "$CPRO_DDK_ROOT" --apply
python3 targets/cpro/tools/clean_draft_probes.py \
  --root "$AI_RUN_DIR" --source "$AI_RUN_DIR/draft/source/prepared" \
  --all-tests --apply
df -h "$AI_RUN_DIR"
)
```

归档应返回`ARCHIVED`或`ALREADY_ARCHIVED`，清理返回`CLEANED`或`ALREADY_CLEAN`。归档失败时`set -e`会阻止清理。已完成后重复执行不需要原tests输入；中途移动失败按同一命令恢复，不另建版本目录。服务器物理迁移与释放空间以该命令回执为准，本地没有访问服务器或设备。

最终保留：

```text
draft/source/prepared/  原始准备权重
draft/air/              当前export.air及引用权重
draft/omc/              当前model.omc、pytorch-abi.json、compiler.air、manifest.json
draft/archive.json      基线身份、已有数值状态及本轮对照摘要
draft/tests/            空目录
```

权重、仓库代码与摘要都不留在tests里。旧探针、nd_n候选、编译副本、输入／golden、快照和历史结果全部随tests删除；不把它们整体搬进archive。若旧runtime配置仍引用被替换的公共模型，工具仅保留该真实依赖于`.omc-previous`／`.air-previous`并调整原配置引用。当前packed模型的host embedding接口尚未接入旧token-ID常驻循环，不把归档当作运行接口切换。

## 后续工作

正常Draft生成入口已改为公共`draft/air`、`draft/omc`，不再把可复用模型发布在tests中；从零配方见[完整流程](run-end-to-end.md)。本次直接保留已有模型，不重建。

下一阶段在新窗口处理原40层量化Target精度，优先抽取实际自定义／融合／量化算子，按Prefill64／Decode1的真实shape和原权重、状态测试，再定位串联与整模差异。两份NZ／CPro原建模源码已在本地找到并重新核验；代码位置、静态差异、历史GDN输出覆盖异常和完整下一步见[Target精度交接](target-accuracy-handoff.md)。本轮只完成交接，未启动新探针；不继续Draft格式实验、不调整验收阈值。
