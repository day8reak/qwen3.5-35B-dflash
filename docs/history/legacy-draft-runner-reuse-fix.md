# 旧host循环第二次draft的runner复用修复

> 历史记录：保留当时的观察、失败和候选命令。当前操作请从[文档首页](../README.md)或[当前状态](../current-status.md)进入；不要直接重复本页旧请求。


2026-10-30，用户回传的`dflash-serial8-run`在10次target、1次draft后失败：第二次draft已经复用首次构建的runner，但依赖库发现仍查找第二次调用目录下不存在的`execution/native/result.json`和`command.json`。这是host编排代码的路径错误；现有回传结果不能算完整DFlash通过。

修复保留首次真实LINKED构建目录及binary/result/command身份。后续调用沿用该目录发现依赖库，并绑定首次选定的库路径、SHA和文件身份；不复制旧构建记录冒充第二次构建。

主机39项针对性检查通过，覆盖无`BOARD_HIAI_LIB_DIR`时的连续两次调用、第二次不生成native目录、构建记录及库变动拒绝。记录位于工作区run `20261030T105923Z-14268-375427/out/draft-runner-reuse.xml`；未执行设备。

用户已选择直接运行新版`run_board_serial_dflash.py`。本修复仅保留旧诊断入口的可用性，不要求重跑旧的一小时任务，也未实现从旧失败目录继续推理。保留旧目录，新的端侧运行使用单独输出目录，命令见[当前入口](../current-status.md)。
