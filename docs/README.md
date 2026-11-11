# Qwen3.5-35B-A3B · DFlash

原40层量化Target与完整六层Draft，已在端侧完成文字输入、Prefill、常驻草稿／验证循环和文字输出。当前归档约60ms的完整packed Draft基线，暂停其性能实验，转向Target精度；完整模型精度与配对端到端加速尚未通过。

- [模型结构](model-architecture.md) · [Target 交互图 HTML](assets/target-architecture-interactive.html) · [静态 SVG](assets/target-architecture-accuracy.svg)
- [优化策略与各项时延收益](optimization-strategy.md)
- [端到端执行流程](end-to-end-flow.md)
- [当前结果：时延与接受率](results.md)
- [环境配置](environment-setup.md)
- [完整流程执行](run-end-to-end.md)
- [当前阶段：Target 四层整图精度定位](current-status.md)
- [OMC 编译／精度状态表与四层 dump 流程](target-accuracy-status.md)
- [整图算子族与精度风险 CSV](../specs/target-operator-accuracy-risk.csv)
- [四层 dump 批量比较脚本与所需信息](target-dump-compare.md)
- [环境修正后：批量重编 OMC、上板、回收与比较命令](target-batch-rerun.md)
