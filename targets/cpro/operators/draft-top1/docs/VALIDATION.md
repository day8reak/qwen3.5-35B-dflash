# Validation

2026-11-04 host run 20261104T075532Z-9847-483675：35项检查通过。C++17以-Wall -Wextra -Werror构建主机参考；七组完整尺寸fixture与NumPy/PyTorch直接ArgMax输出逐字节一致。覆盖索引0/511/512/65535/65536/248319、跨块并列、全相等、正负零、正负子正规数、有限极值、相邻FP16、固定随机数及全部有限binary16位模式。

比较器要求完整七组输入及10份输出，核对源码和fixture哈希，并重新计算直接ArgMax。缺失、短文件、额外填充、错误索引、修改golden或同时伪造golden哈希、删减suite均拒绝。输入NaN/±Inf、shape/dtype/stride错误拒绝。C++对非有限或错误长度在输出打开前拒绝。

DDK收集器用synthetic SDK验证：只读、完整源码收集、内部链接循环去重、外部链接不读取、输出不能位于SDK或其祖先、已有输出不覆盖、旧环境指向另一DDK拒绝、预算不足记录PARTIAL。

证据位于该run的evidence/host-final/host-validation.json，含源码、CPU二进制、fixture manifest哈希。70个复制的主机输出文件仅测试比较器；真实NPU调用0次。build入口未具备适配时明确拒绝；kernel build/注册/独立复制构建/OMC落点/设备数值/时延/整Draft全部PENDING。

设备阶段另需对真实新OMC确认CPUCL=0、计算均归属NPU；每例3次预热+10次测量，取回60字节INT32。比较文件本身不能证明执行设备、调用次数、来源或计时，须由运行记录绑定。尾部写入与workspace安全还需canary/诊断或选定API契约验证。
