# 端到端执行流程

服务器只准备请求、部署和回收。请求启动后，文字编码、embedding查表、模型调用、接受判定、状态更新和文字解码均在板端完成。

```mermaid
flowchart TD
    A[服务器：文字与配置，一次下发] --> B[板端Tokenizer：编码并检查长度]
    B --> C[原表查Embedding，Prefill运行一次]
    C --> D[保存80状态与特征，关闭Prefill]
    D --> E[加载常驻Decode与Draft，Draft顺序预读]
    E --> F[用Decode消费首个已提交token]
    F --> G[Draft生成15个候选]
    G --> H[Target逐个验证，当前每轮最多2个]
    H --> I[提交接受前缀与纠正或Bonus，推进Decode和状态]
    I --> J{达到生成预算?}
    J -- 否 --> G
    J -- 是 --> K[板端Tokenizer解码，关闭模型]
    K --> L[服务器一次回收文字、Token和时延]
```

1. 原tokenizer在服务器本地做长度准入，板端重新从文字编码并核对后才加载模型。
2. 短prompt补齐到物理64行，原Prefill执行一次，产生首token、80状态和8层特征。
3. Prefill关闭后，Decode与Draft各加载一次，在同一板端进程内交替运行。
4. Draft提案与Target greedy结果逐个比较；首次不匹配使用Target纠正token，全接受时生成Bonus，继续下一轮。当前是decode1串行验证，尚未使用批量verify。
5. 默认关闭dump：中间张量在内存流转，不逐token传输或写临时张量文件。仅启动时预读Draft OMC，预读时间计入加载耗时。

当前支持单条user聊天模板或raw文本、固定生成预算；尚未接入EOS早停和多轮会话。Token输入闭环已有实机结果；文字版组合已通过主机测试，待本轮板端验证。
