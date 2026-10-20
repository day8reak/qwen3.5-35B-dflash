# 经验审计（2026-10-20）

本工程由 `deliver-model-end-to-end` 管理；下表是首次实现的历史记录。当前因量化精度要求更新，完整模型阶段停留在 contract。以下结论均限定证据范围；没有把未运行的 X90 行为写入通用技能。

| 观察/最小复现 | 证据与范围 | 决定和下一验证 |
|---|---|---|
| 4B 中已有 DFlash draft math 和 native greedy scheduler | `specs/reference-provenance.json` 保留原始文件 SHA256；旧工程记录有 CPU/mock 范围限制 | 复用源代码并审计 35B 配置；不继承 310P 或 4B 的板端/性能结论 |
| 用户把 NNRT 更正为 HIAI，明确 AIR→OMG→OMC | 用户本会话；蓝区头文件与 OMG help | 正式源码移除 NNRT；保留 HIAI 适配和 AIR pipeline，旧原型仅在本 run 历史日志中 |
| 去掉签名查询串的华为下载链接返回 403 | 早期失败记录、官方 preparations.md 中完整签名链接、官方 SHA256 | 下载脚本保留签名链接并核验哈希；作用域为本次两个公开 6.1.1.0 包 |
| 解压的 OMG binary 未带执行权限；vendor wrapper 包含在 DDK 树创建 glibc symlink 的逻辑 | 实际文件权限、wrapper 源码、真实 `--help` | 只修改 run 内解压副本；转换直接运行 binary 并显式设置隔离环境，不修改 shared profile |
| eager MoE 只遍历样例命中专家，直接 trace 会冻结 expert coverage | 上游专家 forward；跨 seed/input 路由测试 | ONNX 数学验证使用动态路由索引；AIR 基线使用固定形状全部专家 + 动态 mask。后者成本高，尚无性能晋升 |
| Transformers 输出采集进入 lock/ContextVar，Dynamo fullgraph 断图 | `log/dynamo-preflight.log`（实际失败）；`log/dynamo-explicit-loop.log`（修复后完整捕获） | 目标导出使用显式层循环，保留相同 mask/position/feature 顺序；与未包装 upstream 比较，完整权重仍待验证 |
| 文本 MRoPE 的原地 strided slice 在早期 tiny ONNX 导出失败 | 旧失败日志与 `tests/test_onnx.py` | 三轴文本位置相同，改为等价频率计算；不扩展到 multimodal 假设 |
| HIAI 公共兼容 API 有 ND dims、同步 Process、NPU 检查和路径加载，size buffer 为 uint32 | 蓝区真实头文件；OHOS ARM64 对象文件编译；fault double 测试 | 用路径加载大模型，检查 tensor byte overflow；不宣称 >4GB OMC 和 BF16 已被运行库支持 |
| HIAI 公开 API 不暴露模型 IO dtype | 真实 `GetModelIOTensorDim` 签名 | ABI dtype 由 AIR/OMC hash manifest 锁定，加载端检查 ND/shape/bytes；内部接口若不同先反馈编译或加载日志 |
| 双方目前没有 TorchAir；本机无完整权重和板机权限 | dependency preflight、用户回复 | 加入导出前依赖检查与 `export-tiny-air`；未用 fake payload 或 ONNX 冒充真实 AIR/OMC |

新增可执行检查：MoE 路由/显式导出循环跨输入与 BF16 测试；真实 Dynamo fullgraph 捕获；AIR 外部权重篡改与 OMG 失败清理；HIAI 初始化/兼容性/部分加载失败/张量分配/运行失败及单次推理测试。结果见 validation。

未修改通用 skill：这些观察与新芯片/模型的目标集成尚缺完整证据，当前沉淀为模型局部实现和回归检查，避免推广成未经验证的平台规律。首次实现没有委派 custom operator，也没有执行量化；恢复后的量化工作见下节。

拒绝的晋升：toy/ONNX/Dynamo PASS 不能替代 AIR；fake HIAI PASS 不能替代设备；对象文件编译不能替代内部库链接；缺少画像不能自动解释为 hardware-reference opt-out；未量测不能宣称 DFlash 加速。

剩余假设：内部 HIAI 与蓝区 API 是否一致、X90 BF16/GE 算子支持、FileConstant 大权重转换及加载、实际峰值内存、真机 token 一致性和收益。下一步由用户内网先补 TorchAir 并跑 tiny AIR→OMC，再提供日志用于有针对性的修改。

## 恢复后经验审计

当前 run：`20261020T092412Z-4528-622cba`。用户解除暂停，并明确允许权重与激活量化、其余 FP16。

| 观察/最小复现 | 证据与范围 | 处理及下一验证 |
|---|---|---|
| `.half()` 后上游仍显式执行 FP32 GDN、norm、router、RoPE | `tests/test_precision.py`，真实 FX tensor dtype metadata | 普通图执行 FP16 lowering 并重新捕获审计；这是数值改变，保留 upstream reference；全权重质量待验证 |
| 改特征 dtype 必须同步 HIAI 字节数和 artifact 身份 | ABI v2、真实头文件编译、fake HIAI 首末 FP16 元素复制检查 | 拒绝旧 FP32 bundle；整数 token/mask 仍为4字节 |
| MoE 专家是 Parameter bank，普通 Linear visitor 会漏量化 | 全模型 meta 统计、专家改写前后跨输入比较 | 暴露全部10,240专家/20,480矩阵；不得仅统计被样例命中的专家 |
| DOPT 公开插件可实际校准小 MoE，但 Cython QLinear 不能直接 Dynamo capture | `log/dopt-model-probe.log`、`log/dopt-fx-probe.log` | 主机参考以不改变 vendor math 的 opaque bridge 调用；AIR 只允许原生 packed quant op，不能回退导出模拟层 |
| A16 输入量化的 FP16 缩放可溢出，产生 NaN | `log/dopt-frozen-compare-v2.log`、首次完整测试中 quant reference 失败 | 输入量化器内部改为 FP32、输出回 FP16；用独立高精度饱和量化 oracle 和真实 DOPT 回归；板端仍待验证 |
| DOPT 暴露 get_air_model，但缺 NpuQLinear/torch_npu 扩展 | `log/dopt-linear-probe.log`、`out/preflight.json` | 导出前检查特定扩展，避免先分配35B再报错；没有宣称当前环境具备 AIR 量化能力 |
| 只比较 checkpoint SHA 不能区分 BF16、FP16 和不同校准 | `tests/test_cli.py` 的 precision、quantization、ABI 负例 | token/stop比较同时绑定源、校准、precision、ABI、OMC哈希；普通/DFlash一致不能证明量化质量 |
| OMG 短相对路径修复后实际负例进入 AIR parser | 原 run `log/omg-air-short-path-smoke.log` | 路径问题已定位；当前编排回归覆盖成功/失败清理；仍无真实有效AIR转换PASS |

本轮加强的是模型局部可执行检查。未修改通用技能：DOPT二进制、CPU模拟溢出和X90导出能力没有足够跨版本/板端证据，不推广成普遍规则。没有委派独立 NPU operator。

拒绝晋升：真实 DOPT tiny校准也不能证明35B质量；主机opaque桥接不能作为AIR；存在 get_air_model 不等于已支持X90整个模型；32GiB预算不能替代峰值测量。余下假设是内部量化op布局与API、全模型转换覆盖、数据校准质量、峰值内存、量化参考与设备数值一致性及实际性能。
