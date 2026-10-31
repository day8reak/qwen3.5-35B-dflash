# 当前执行计划

<a id="用户现在执行什么"></a>

**当前结论：原板端闭环已跑通；新 V1 常驻版本在 prefill 的 SDK Load 阶段失败，尚未开始推理。不要重复运行同一 path-only 候选。**

用户回传 `dflash-board-resident-v1-file8-run`／`dflash-board-resident-v1-file8-recovery`，运行库 `800.631.120.010`，进程63072。HiLog显示 `DDK pipe is HCL`，随后 `CreateLocalBuffer` 拒绝空buffer／零长度，V1 Load返回1。只设置模型路径的描述符在OMC恢复、权重解析及Process之前被拒绝。全部推理调用计数0，token为空，已知退出后锁已释放。`libffrt_acc.so` warning不是日志直接报出的失败点。证据来自用户粘贴，未由助手重新连接设备核验，见[本次记录](validation.md#resident-v1-hcl-file-load-failure)。

**下一步是确认所选SDK可用的V1外置权重入口。** 4B工具的 `SetModelPath → Load` 支持开展此前实验，但当前35B已经实机失败。官方旧buffer示例没有建立外置权重合同；不能添加builder、交换setter顺序或猜测环境开关后再试。已观察到path／buffer setter互斥，18.8GB旁置权重也不能塞入现有UINT32长度的单模型buffer；此长度限制本身不证明所有V1扩展均不可用。

用户主机此前确认存在的工具头文件目录是 `/home/w00949577/model_run_tool/inc/compatible/`，不能假定DDK也有相同子目录。工具头文件、runner选中头文件和实际运行库分别记录。用户随后贴出的工具头文件有SetModelPath／SetModelBuffer／Load，未显示显式外置权重入口；它不等同于runner选中头文件的完整身份。当前不重新运行模型或load-only探针。已向用户提出路线选择：坚持全V1并确认厂商外置权重合同，或允许prefill／decode暂用V2、embedded draft保留V1继续推进常驻。后者仍需解决已记录的V2清理报错和双模型常驻，尚未获准或验证；不静默切回V2。

**目标仍是按用户要求，将 prefill、decode 和 draft 统一为模型管家 V1。** V2 在两次独立加载实验中均 Init／InitWeights成功，但显式DeinitWeights分别返回3和1；原因未定，见[历史观察](../specs/resident-v2-target-cleanup-observed.json)。用户随后提供了官方V2弃用说明，并明确要求全部改V1。此前让用户继续运行V2闭环的指令已经撤回；旧attempt和结果保留用于恢复及对照。

**P0 原 prefill 端侧完整执行已完成。** 旧版用户回传 prefill1／decode8／draft7，token为 `[4,12,10,8,0,4,32,2]`，接受0／7，总墙钟305.788秒。它证明原板端执行闭环，不证明新V1常驻、精度或加速，见[旧成功基线](board-prefill-first-completion.md)。

历史候选191项Python测试及13组native CTest通过，其中9项使用实际C++框架和fake SDK；实际HCL拒绝文件描述符，说明替身测试没有验证SDK兼容性。原记录见[候选验证](../specs/resident-v1-file-buffer-candidate.json)。本轮仅改失败摘要与日志采集，132项transport及26项日志采集测试通过，无设备执行或模型编译，见[本轮记录](../specs/resident-v1-hcl-load-failure.json)。

## V1 候选设计与接口约束（尚未实机完成）

一次部署和启动后，板端同一执行线程完成：

1. 创建独立V1 prefill实例，Load一次，Process一次，UnLoadModel后释放。
2. 创建decode和draft两个独立V1实例，各Load一次。
3. 在两个常驻实例间交替Process，完成候选、decode1 verify、接受／纠正和80状态反馈。
4. 关闭两个实例，回传最终token、接受率、分阶段时延及内存摘要。

模型管家不跨线程共享。同一实例只调用一次Load，失败后不重新Load；不使用异步接口或任务队列。prefill释放后才加载decode/draft；不并行加载卸载。每个实例只加载一个模型，循环期间保留两个实例。

三个既有OMC、权重、六层draft S64及15候选不变；只编译native runner。循环内的embedding查表、状态和特征传递、接受判断均在板端，没有逐token服务器交互。

新请求明确绑定 `target_lifecycle_policy=legacy-v1-file-target-buffer-draft-v1`；session摘要为 `model_load_api=legacy-v1-mixed`，prefill／target为 `legacy-file`，draft为 `legacy-buffer`。FILE_ONLY检查path精确且buffer为空、长度0；这些检查与SDK加载成功是不同结论。旧请求恢复仍使用其原配置。

## 当前失败诊断与历史请求

`2fa9d7d` 的FILE_ONLY候选已实机失败，启动命令撤下，不再重复运行。`dflash-board-resident-v1-file8-run`／`dflash-board-resident-v1-file8-recovery` 均保留；旧707d825／50184b4记录也不覆盖。

摘要现在带出 `prefill_lifecycle` 和 `prefill_failure_stage`，不再把未加载的decode／draft会话正常 `CLOSED` 当作失败原因。底层加载路线未改变。

新增 `collect_board_serial_failure_logs.py`：用于未来compact请求失败后，只读收集同一已完成attempt的stdout／stderr末尾各64KiB，可选PID过滤的HiLog快照；不启动模型、不改锁、不读模型或权重。HiLog可能轮转或PID复用，不视为严格的同attempt证据。本次已有足够HiLog，**无需再次采集、恢复或重跑**。

未来另一请求需要时使用以下模板，替换目录名，output必须新建：

```bash
python3 "$REPO_DIR/targets/cpro/scripts/collect_board_serial_failure_logs.py" \
  --recovery "$AI_RUN_DIR/已完成的失败恢复目录" \
  --output "$AI_RUN_DIR/新的诊断目录" \
  --hilog
```

## 后续新加载方案的验收目标

- `execution_mode=resident-v1`，prefill1、decode8和至少两次完整六层draft；每次draft输出15个候选，最多考虑前K=2，首个不匹配就纠正。
- decode/draft各一次Load、多次Process，全部V1调用；prefill也是V1且仅执行一次，没有原工具V2子进程。
- 对照旧同模型、同输入、同控制的token `[4,12,10,8,0,4,32,2]`。差异先按API／状态迁移回归定位，不因P0允许乱码就忽略。
- 接受率和prefill／decode／draft时延，另列加载与关闭耗时。时延包含所声明的复制和读回，不称纯NPU推理时间。内存统计为进程RSS/HWM和系统MemAvailable，不是独立NPU显存。
- 失败保留准确阶段、Load／Process／Unload返回码和已经完成的token/计数。回收stage=complete不等于模型成功；清理错误不被吞掉。

默认dump=false，只回收token与小型摘要。需要逐状态证据时在新请求加dump并使用新output；旧请求恢复模式不变。新V1调用失败时停止并保留证据，不自动切回V2，也不重新编译模型或改动原权重。

## 保留的模型及阶段边界

prefill物理长度64、逻辑prompt长度8，沿用 `PREFILL64_PADDED_SHORT_PROMPT_EXPERIMENTAL` 的控制和输入。前8行原始embedding/scale后补零，logits选第7行。前缀和特征不裁剪，P+G不超过原draft容量64。输入仍是token ID文件；板端分词和转文字尚未包含。

P0已完成，P5常驻工程按用户要求提前推进；P1/P2精度、P3历史一致性、P4增量draft、P6批量verify、P7性能、P8交付继续见[总计划](dflash-delivery-plan.md)。原始全网精度、有效接受率、稳定性和配对加速仍未验收。
