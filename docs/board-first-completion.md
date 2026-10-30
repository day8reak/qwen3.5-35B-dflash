# 首次端侧自主多轮执行完成

2026-10-30，用户在DDK主机回传`run_board_serial_dflash.py --recover`的成功结果。本记录依据用户提供的终端输出；助手未连接设备，也没有在本地重新读取该机器的完整文件。

- 原请求：`$AI_RUN_DIR/dflash-board8-run`。
- 回收目录：`$AI_RUN_DIR/dflash-board8-recovery`。
- 实际attempt：`/data/local/tmp/qwen35_board_serial_1056ef4d658548d5ba16ee7c505a34d7`。
- 状态：`BOARD_SERIAL_DFLASH_EXECUTED_UNVALIDATED`，`EXECUTED_OUTPUTS_RETRIEVED`。
- target调用16次，draft调用7次；considered=7、accepted=0、corrections=7、bonuses=0。
- 锁：`NO_LOCK_HELD_OR_RELEASED_AFTER_KNOWN_EXIT`。
- 模型编译未运行；精度未评估。

该入口在板端完成prompt输入、embedding查表、draft、decode1判断与状态反馈。恢复命令只回收同一次运行，不重新推理；返回成功前，旧完整dump校验路径检查调用receipt、输出槽位和状态链。按此用户回传结果，P0执行里程碑已完成；不是旧host编排循环的结果。

此次自然发生的是全拒绝／target纠正分支，没有实机接受或bonus分支覆盖。8个生成token的具体值未包含在用户贴回的简短输出中；原`results/loop.json`保留其内容。未提供配对耗时、数值reference或物理断连时间记录，不由此宣称精度、加速、模型常驻或专门的断连实验通过。

保留该成功目录及其原始源码／模型／配置身份。后续dump开关、常驻等代码变化需要各自验证；此历史结果不能自动验收修改后的二进制。当前用户新增优先级是默认精简回收、通过`--dump`保留详细调试数据。
