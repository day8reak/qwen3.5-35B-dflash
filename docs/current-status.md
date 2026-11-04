# 当前阶段测试命令

目标为C16／cap192／K15常驻预热后平均≤50ms。最新完整Draft的model inference为103.720ms，算子合计73.656ms，算术差额30.064ms；差额尚未归因。完整Draft仍为NUMERICAL_FAILED，host embedding准备未计入API，原数值门槛保持。

## 当前进展：Top1探针未找到可用NPU路径

用户已完成三个小图的服务器编译，设备未执行：

| case | 结果 | 直接证据 |
|---|---|---|
| top1_argmax_fp16 | 编译通过，CPUCL1，无NPU | CPU子图输入FP32、输出INT32 |
| top1_with_value_fp16 | 编译失败 | argmax_value的ArgMaxWithValue infershape函数查找失败 |
| top1_tiled_fp16 | 编译失败 | Reshape到1×15×485×512成功，随后local_max同样找不到ArgMaxWithValue infershape函数 |

两种候选遇到相同的编译入口问题，不能据此判断词表或512分块过大。libai_npucore_ascendc_kernel.so路径提示、TBE内部Ascend310B1字符串未被确定为本次原因，不更换kirinx90目标或修改共享DDK。当前三种探针无需上板，也不重复编译。

## 只读所选DDK中的归约注册

同步本次源码后，沿用现有REPO_DIR、AI_RUN_DIR和CPRO_DDK_ROOT执行：

```bash
python3 "$REPO_DIR/targets/cpro/tools/probe_draft_vector.py" \
  --inspect-top1 "$AI_RUN_DIR/draft/tests/top1-probes" \
  --ddk-root "$CPRO_DDK_ROOT"
```

回贴输出JSON即可。工具优先读本轮omg-environment.json中的KIRIN_CONF_JSON_PATH，核对保存的DDK根，并查ArgMax／ArgMin／TopK／ReduceMax／ReduceMin注册和相关源码文件的有限片段。支持配置目录或单个JSON文件，输出所查路径、覆盖限制和匹配记录；不导入vendor模块、不写文件、不重新编译、不访问设备。

本地17项合成文件与CLI检查通过。注册中出现某个名字仍不代表已编译到NPU；未找到记录也不证明硬件不支持。根据所选DDK实际记录，再选择内置归约组合或其他集成路径，避免继续用公开GE定义猜本版本实现。原AddCustom探针仍暂停。完整记录见[结果](results.md)。
