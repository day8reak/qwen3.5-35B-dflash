# Qwen3.5-35B-A3B · DFlash

原40层量化Target与完整六层Draft，已在端侧完成文字输入、Prefill、常驻草稿／验证循环和文字输出。当前归档约60ms的完整packed Draft基线，暂停其性能实验，转向Target精度；完整模型精度与配对端到端加速尚未通过。

- [模型结构](model-architecture.md)
- [优化策略与各项时延收益](optimization-strategy.md)
- [端到端执行流程](end-to-end-flow.md)
- [当前结果：时延与接受率](results.md)
- [环境配置](environment-setup.md)
- [完整流程执行](run-end-to-end.md)
- [当前阶段：归档Draft、清空测试、转向Target精度](current-status.md)
