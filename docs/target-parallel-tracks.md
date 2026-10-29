# 两条完整 target 路线并行

用户于2026-10-29要求两条方案同时推进，防止直接GE target出现问题。两条保留独立产物、来源和设备结果，共用完整`residual-fix` draft；真实8层features、完整40层原量化target、248320词表及最终验证要求相同。

| | A：原PyTorch/TorchAir | B：直接GE |
| --- | --- | --- |
| 结构依据 | 用户原Cpro wrapper/model实现 | 同一原Cpro实现及实际AIR算子/权重连接 |
| 构建位置 | 原权重机器，已有PyTorch/TorchAir | 当前服务器的原DDK；大scratch/output需共享盘或大盘 |
| 权重 | 原HF非expert参数、原quant数据、原embedding表 | 优先从原AIR引用绑定原外置payload；需精确路径/dtype/shape/layout/extent证据 |
| 接口 | 原89输入/80state，保留81输出并追加真实features | 对齐原paged/NZ/RefData接口，不套现有functional target |
| 当前入口 | `prepare_target_pytorch.py` | `prepare_original_ge_target.py` |
| 独立目录 | `$AI_RUN_DIR/target-pytorch`（原机） | `$AI_RUN_DIR/target-ge`（当前服务器） |

## A：在原权重机器准备

本轮原机已在`qwen3_5_cpro`环境回传`SOURCE_READY`，目录为`/home/w00949577/qwen35-target-pytorch-run/target-pytorch`，实际导出尚未运行。保持该环境，原机导出空间可用时直接执行下条；无需重复准备同名目录：

```bash
bash /home/w00949577/qwen35-target-pytorch-run/target-pytorch/export.sh
```

此次会生成完整新AIR和外置sidecar；源码准备阶段未复制大检查点，不表示完整导出不需要空间。新AIR预期路径为`target-pytorch/target-pytorch-source/source/glm_export_air_cpro_40block_64_quant/export.air`，以实际导出结果为准。

原机器仅有PyTorch/TorchAir，完整大检查点留在原位置。先把本仓库最新代码同步到原机，令原机`REPO_DIR`指向该仓库（不是原模型工程），并激活当初导出AIR的环境。然后进入原PyTorch模型工程根目录。`AI_RUN_DIR`应放原机有足够导出空间的独立目录；下面默认原工程旁，空间不足时先换成大盘路径。执行：

```bash
export ORIGINAL_TARGET_SOURCE="$PWD"
export AI_RUN_DIR="$(dirname "$PWD")/qwen35-target-pytorch-run"

python3 "$REPO_DIR/targets/cpro/scripts/prepare_target_pytorch.py" \
  --source-root "$ORIGINAL_TARGET_SOURCE" \
  --config "$ORIGINAL_TARGET_SOURCE/config/qwen3_5_moe.yaml" \
  --output "$AI_RUN_DIR/target-pytorch" \
  --sequence-length 64
```

原源码`utils.py`的专家scale装配实际调用`.npu()`和`npu_trans_quant_param`。激活原导出环境后，先检查原机是否具备该转换所需NPU；这是原源码的依赖，不代表所有TorchAir导出都需要设备：

```bash
python3 -c 'import torch, torch_npu; print("npu_available:", torch.npu.is_available())'
```

`SOURCE_READY`只说明资源头/小源码就绪，不检查这一厂商依赖。若上面为False或导入失败，先保留结果，不加载完整模型；应核对原导出环境/原转换资源，不能把失败解释成模型精度问题。

用户已回传原机该检查为`True`，NPU可见性已确认；继续保持同一原导出环境，运行上方资源预检。该结果不代替TorchAir/原自定义算子兼容性或实际导出结果。

返回`SOURCE_READY`且原导出依赖与输出空间可用后，可直接在原机执行，无需再等待一次确认：

```bash
bash "$AI_RUN_DIR/target-pytorch/export.sh"
```

返回`SOURCE_BLOCKED`时保留错误报告，不运行导出脚本。若任务目录已经存在，保留旧证据，用新输出目录重试。HF、quant检查点和两张原embedding表继续使用原机路径；它们不搬到当前10GB服务器。

默认只核对配置、safetensors头与文件尺寸，并准备小型派生源码；不加载权重payload、不运行模型或导出。当前prefill64命令显式使用`--sequence-length 64`，TorchAir仍为静态导出；OMG配置64/1不自动证明图能正确decode，两个档位及状态连续性需实际验证。原loader缺HF参数仅打印Missing，故该入口必须提前硬性拒绝缺参数/错shape/错dtype；不能让随机初始化参数混入“完整模型”。`SOURCE_READY`也不是权重内容或设备精度通过。

源模型的计算表达式原样保留，包括量化、norm和状态布局；只新增8层features输出。原机空间与资源核对后才执行顶层`target-pytorch/export.sh`。生成新AIR及原sidecar的核对清单后，再安排当前DDK编译；当前10GB服务器不接收整套HF/quant检查点。

## B：当前服务器先绑定真实图

本轮详细报告已成功严格解码`RAW_MODELDEF`：9520节点、9 Data+80 RefData输入、81输出；GDN/convstate各30、attention10、CacheUpdate20、MoE routing/finalize各40、GMM80。原文件SHA保持。81个输出均报告描述符差异，终端展示的前6个只有shape差异：NetOutput有完整维度，producer显示`[]`，dtype/layout相同。完整81项仍保留在服务器JSON；不能只据前6项宣布全部同类。

空shape当前不能自动当scalar或任意维度通配，也不能据`origin_shape_initialized=false`宣称未推导。B契约仅在完整81个实际端口都严格满足shape-only条件时细分为`PENDING_PRODUCER_SHAPE_INFERENCE`，ABI仍未通过；tap分析只使用完整内部描述符，不把NetOutput维度填回producer。新版诊断保留shape字段和dim字段是否实际出现，不据序列化推语义。

**当前不重复解析AIR**。先从已有JSON提取精确权重路径和前几个norm/Add连接，供绑定8层真实features；只读小报告，不访问外置权重：

```bash
python3 - "$AI_RUN_DIR/target-ge-source-air-ge-details.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
g = next(c['graph'] for c in r['candidates'] if isinstance(c.get('graph'), dict))
cs = g['file_constants']['examples']
selected = [c for c in cs if 'input_layernorm' in str(c.get('file_path', ''))]
print('input_norm_constants:', json.dumps(selected or cs[:8], ensure_ascii=False))
print('first_norms:', json.dumps([n for n in g['operations'] if n['type'] == 'RmsNorm'][:4], ensure_ascii=False))
print('first_adds:', json.dumps([n for n in g['operations'] if n['type'] == 'Add'][:4], ensure_ascii=False))
PY
```

路径中的`input_layernorm`只筛选要回传的候选，不构成语义绑定。未改图GE加载/保存检查可与此独立进行，命令和范围见下文；其主机GE根沿用已成功draft导出的`CANN_ROOT`，不是`DDK_PATH`。

此处只需已上传的5py+YAML和实际图元数据，不读取其四个权重路径。B入口可直接重新哈希这6个小源文件；不要求在当前服务器运行A的派生源码准备器：

当前服务器已找到真实TorchAir schema，不需要额外搬文件：protoc3_13与3_19的descriptor均为7381字节、SHA256 `5a1eac27746a7bb5109e819ce6dd7813a3ffcaab04bd0ab2ca3c2cdf1e4084ea`。选择更小的3_19生成文件；版本shim`ge_ir_pb2.py`不用于解析。执行：

```bash
python3 "$REPO_DIR/targets/cpro/scripts/inspect_original_target_air.py" \
  --air /home/l00886453/qwen3_5_moe/glm_export_air_cpro_40block_55_quant/export.air \
  --torchair-pb2 /root/anaconda3/envs/custom_pypto/lib/python3.11/site-packages/torch_npu/dynamo/torchair/_ge_concrete_graph/ge_ir_by_protoc_3_19_pb2.py \
  --full --output "$AI_RUN_DIR/target-ge-source-air-ge-details.json"
```

这是已执行的显式schema选择，仍只解析AST数据，不导入TorchAir/NPU；本轮结果以上方详细报告为准。其他环境可使用`--find-torchair-schema`重新定位实际生成文件。

最新使用真实GE schema的回传已到达`NetOutput differs from its explicit producer descriptor`，说明RAW_MODELDEF的protobuf解码已进入后续边界检查；旧检查器将两阶段混在一起，笼统打印OPAQUE。新版分别记录`protobuf_decode`与`metadata_inspection`，保留NetOutput每个不匹配端口两侧shape/dtype/layout及origin/storage元数据；ABI差异仍不放行。重读使用新的`target-ge-source-air-ge-details.json`，前两份报告保留。

用户已回传原AIR：6361049字节、SHA256 `57934fc4177929a20c978eadbb121741602cefbfb212027e440fce67b08d32a6`，无IMOD头。当前HIAI schema解raw GraphDef报unknown field7，解raw ModelDef报AttrDef.list_list_float wire mismatch，状态`OPAQUE`。这是schema不兼容观察，不能据此判坏AIR或放宽检查器。新增入口只从显式选中的pb2 AST取descriptor bytes，不执行文件；未知字段/错类型、shim、多个descriptor、外部依赖仍拒绝。默认HIAI检查不变。旧`target-ge-source-air.json`保留，匹配原schema的新报告写到`target-ge-source-air-ge.json`。

完整图元数据只写入指定JSON，终端打印小摘要；不读外置权重、不改AIR、不调用OMG/HDC。所选schema不兼容或原89/81边界不同必须如实报告，不能据此猜出8层hidden。

直接GE路线需要保留原`AdnFusedInferAttention`的paged/NZ block32接口、CacheUpdate、GDN/convstate、MoE routing/GMM/finalize和80 RefData，并逐项固定FP16/FP32转换。仓库现有`adapt_original_target.py --stage ge`是一条functional状态改写，其ND attention和gather专家路径不能直接视作原Cpro等价实现。

完整图现在为少量Const标量记录声明dtype/shape、最多8字节的数据长度及SHA，不输出tensor值。已审查源代码的`weight+1`路径仅在FP32/FP16标量的字节SHA精确证明为1.0、gamma和Add链唯一且每段描述符一致时，允许作为结构候选；不信报告中的one标记，也不穿过其他变换。

8层tap仅能从实际图的精确权重引用与有序连接证明。可用下一层input norm的数据输入对应上一层post-decoder hidden，但必须验证gamma来源、norm数据端口、dtype/shape和40层锚点；不靠Add名字或节点序号猜。原source的norm表达式和loader赋值都需保留，不能把看似多余的`+1`自动修正掉。

基础绑定命令如下；若已存在同名输出，保留旧报告并使用新任务目录：

```bash
python3 "$REPO_DIR/targets/cpro/scripts/prepare_original_ge_target.py" \
  --air-report "$AI_RUN_DIR/target-ge-source-air-ge-details.json" \
  --source-root /home/w00949577/qwen3.5-35B-moe \
  --output "$AI_RUN_DIR/target-ge"
```

`--gamma-template`需从实际AIR的FileConstant路径核对后提供，不能直接把猜测的层名模板当作来源证据。

拿到实际图后，`prepare_original_ge_target.py`产出明确的原接口GE契约和绑定状态；证据不足时保留`PENDING_GRAPH_BINDING`，不生成随机权重或冒充可执行AIR。真实GE构建/编译/设备结果必须另行出现后才能推进。

### B的无改图GE往返基础检查

新增`probe_original_air_roundtrip.py`，用于新增features之前验证GE加载/保存能否保留原图。它只把有界AIR复制到独立目录，不放权重文件；FileConstant引用必须能安全留在该目录内，绝对路径、父目录逃逸或未知路径在调用GE前阻断。所选GE头文件、库、编译器和schema身份进入报告。

在GE/CANN环境实际可用、引用可隔离时，可独立执行：

```bash
python3 "$REPO_DIR/targets/cpro/scripts/probe_original_air_roundtrip.py" \
  --air /home/l00886453/qwen3_5_moe/glm_export_air_cpro_40block_55_quant/export.air \
  --torchair-pb2 /root/anaconda3/envs/custom_pypto/lib/python3.11/site-packages/torch_npu/dynamo/torchair/_ge_concrete_graph/ge_ir_by_protoc_3_19_pb2.py \
  --ge-root "$CANN_ROOT" --output "$AI_RUN_DIR/target-ge-roundtrip" --trace-files
```

该步骤调用GE LoadFromFile/SaveToFile，生成新的AIR副本；不新增features、不运行模型。它比较全部decoded protobuf字段，仅归一map条目顺序；不把二进制序列化顺序变化当计算变化。记录文件syscall供独立审查，不能从未提供权重文件推断opaque库绝无任何访问。

真实两层小CANN AIR已经Load/Save成功并保留decoded metadata，27个FileConstant引用未出现在保留的223行文件syscall记录中；原AIR SHA/stat保留。这不是当前完整target、特征扩展或设备精度通过。当前B仍优先取齐完整target的边界差异报告，A可同时在原机预检/导出。

## 共同验证与空间

源码准备、导出与接口工作并行；大编译和板端验证按实际磁盘/内存预算排队，板端每次载入一个target和完整draft。两条分别保留源码/图/权重引用、compiler及编译选项、89输入与state文件、实际输出和native结果。比较时采用同输入、同一有效长度和状态基点，检查完整词表logits/Top1、真实8层features和80状态；随后各自验证64→1及多轮DFlash。结构或编译通过不能代替设备精度；一条通过也不能替另一条通过。

执行服务器只需按token查原INT8 embedding及FP32逐词scale，两表约485.95MiB；完整HF与quant目录留原机。新AIR复用源sidecar前须逐项校验，旧OMC编译权重不能直接替换新OMC。原机只有导出环境，当前DDK的大输入快照、scratch/output需要已核实容量的共享挂载/大盘。
