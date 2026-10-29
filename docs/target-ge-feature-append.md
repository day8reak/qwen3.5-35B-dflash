# B：保留原图并追加8层真实特征

原AIR已经严格解码：9520节点、89输入、80 RefData、81输出、1153 FileConstant。用户回传了全部40层input norm的精确权重路径，以及`FileConstant → Add(1) → RmsNorm`连接；FileConstant属性和Add输入明确声明gamma宽2048，而多个内部输出shape仍为空。

新入口不填充或改写原描述符。它以精确FileConstant路径绑定全部40个input norm，要求scalar-one的实际小数据SHA、规范端口、全图有序DAG、40个norm均进入原logits、原embedding/scale真实数据流和残差连接。8个输出对应原post-decoder层`[1,6,11,16,22,27,32,37]`，取下一层input norm的源hidden；不使用层名相似度、arg序号规律或占位数据。

## 当前执行命令

当前DDK机沿用成功draft导出的主机GE/CANN环境。源目录应为上传的原始5py+YAML，不能指向A已经改过wrapper的派生源码：

```bash
git -C "$REPO_DIR" pull --ff-only
python3 "$REPO_DIR/targets/cpro/scripts/append_original_ge_features.py" \
  --air /home/l00886453/qwen3_5_moe/glm_export_air_cpro_40block_55_quant/export.air \
  --torchair-pb2 /root/anaconda3/envs/custom_pypto/lib/python3.11/site-packages/torch_npu/dynamo/torchair/_ge_concrete_graph/ge_ir_by_protoc_3_19_pb2.py \
  --source-root /home/w00949577/qwen3.5-35B-moe \
  --gamma-template './glm_export_air_cpro_40block_55_quant/model_model_layers_{layer}_input_layernorm_weight' \
  --sequence-length 55 --ge-root "$CANN_ROOT" \
  --output "$AI_RUN_DIR/target-ge/features"
```

若当前机没有原源码，可把`--source-root ...`替换为`--source-receipt "$AI_RUN_DIR/target-pytorch-transfer/source-receipt.json"`；该小文件来自A准备阶段，代码审计只选择其中原source_files，不把派生wrapper当原source。

本次仍使用原AIR的静态55量序；没有把输入或控制值改成64/1。OMG动态档和运行状态连续性后续单独验证。

## 实际构建和检查

已实测：GE加载后的Graph没有可供`FindOpByName`使用的Operator表，直接改该空Operator不会改原图；不能仅因头文件有DynamicInputRegister就认为追加成功。因此新入口使用以下有界步骤：

1. 用实际GE生成独立分支模板：8 Data → 8 Cast(FP32→FP16) → ConcatD(axis2)。独立模板的Data和NetOutput不移入原图。
2. 通过显式schema字段编号，将9个GE节点移入原protobuf，仅连接已验证tap，并给原NetOutput追加第82路。原其它计算节点的编码payload逐字节保持，新node ID避开原ID和node-index范围。
3. 独立验证原Model/Graph字段、原节点顺序、全部1153 FileConstant、89输入及81原输出完全保持；只允许已观察的GE新增字段和值。
4. 对完整追加图实际执行GE Load/Save，并严格比较全部decoded metadata（仅忽略protobuf map排序），再复核原图保持。任一检查失败都不发布正常`air/export.air`。

原AIR、原源码/receipt、所选pb2、GE头/库/编译器、生成器与验证器身份均绑定并做末尾复核。源变化或工具链变化会使结果失败。所有产物在新任务目录；不覆盖原AIR，不读取、复制或改写外置权重，也不调用OMG/HDC。

成功`FEATURE_AIR_EXPORTED`时，结果在`$AI_RUN_DIR/target-ge/features/air/export.air`，末路FP16`[1,55,16384]`。原权重引用仍是原55前缀，payload尚未在该输出目录暂存；不能直接把它当完整编译输入。当前优先编译A，B不同时启动第二次大编译。

本地主机已对声明CANN上的真实分支生成、schema解码，以及含89输入/80 RefData/1153 FileConstant的synthetic完整接口图进行全流程Load/Save验证。它证明构图机制和保真检查能工作；当前用户9520节点原图的特征追加结果、完整模型编译、板端精度、多轮状态和速度仍待回传。source-to-original-AIR导出revision缺少原receipt，保持UNVERIFIED。
