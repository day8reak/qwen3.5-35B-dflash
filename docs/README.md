# Qwen3.5-35B-A3B · DFlash

原40层量化Target与完整六层Draft，已在端侧完成文字输入、Prefill、常驻草稿／验证循环和文字输出。当前优先Draft增量KV、15／7／3／1候选档位、独立精度与性能；Target精度就绪后再开发批量verify。

- [模型结构](model-architecture.md)
- [端到端执行流程](end-to-end-flow.md)
- [当前结果：时延与接受率](results.md)
- [环境配置](environment-setup.md)
- [完整流程执行](run-end-to-end.md)
- [当前阶段测试命令](current-status.md)
