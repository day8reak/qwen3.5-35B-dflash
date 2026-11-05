# 当前阶段：批量定位 hidden 偏差

2026-11-05：完整Draft精度仍未通过，性能优化暂缓。首次query小图的单输出通过，但五输出版本中的FP16检查点失败；两图末端不一致，暂不能把它当作模型首错。FP32探针输出在编译后还经过FP16→FP32转换，不能据此证明内部FP32计算。已有实测见[结果页](results.md)。

本轮一次覆盖**完整第0层**：特征投影、各RMSNorm、Q/K/V投影、RoPE、QK/Softmax/PV、输出投影、两段残差及MLP。每个检查点单独输出，避免成对FP16/FP32输出互相干扰；使用保留的case 0／2／17（空上下文、部分上下文、满缓存），每组3次预热＋10次比较。同一检查点只编译、上传一次，三组输入共用OMC。

## DDK服务器执行

沿用已配置的 `REPO_DIR`、`AI_RUN_DIR`、`CPRO_DDK_ROOT`、`CPRO_DDK_ID` 和支持untouched的运行器。需要原 `export`、`compiled/compilation`、`prepared`、`output-init/result`；不需要Torch/TorchAir，也不重编整Draft或算子。

```bash
git -C "$REPO_DIR" pull --ff-only
export TOP1_DIR="$AI_RUN_DIR/draft/tests/top1-integration"
export LAYER_DIR="$TOP1_DIR/hidden-layer0"

# 只分析上次已回收的文件，补查零值与FP32转回FP16的对应关系。
python3 "$REPO_DIR/targets/cpro/tools/validate_draft_vector.py" \
  --analyze "$TOP1_DIR/hidden-first-query/result" \
  > "$TOP1_DIR/hidden-first-query/diagnosis.json"

# 所有检查点先完成源图/参考预检，再顺序编译。
python3 "$REPO_DIR/targets/cpro/tools/probe_draft_hidden.py" \
  --suite layer0 --air-export "$TOP1_DIR/export" \
  --source "$AI_RUN_DIR/draft/source/prepared" \
  --result "$TOP1_DIR/output-init/result" \
  --ddk-root "$CPRO_DDK_ROOT" --ddk-id "$CPRO_DDK_ID" \
  --tbe-diagnosis "$AI_RUN_DIR/common/tbe-diagnosis" \
  --output "$LAYER_DIR/compiled"
```

编译摘要的`coverage`列出实际检查点数和缺口。单点编译失败不阻止其余点；`PARTIAL`且有已编译检查点时仍可执行现有覆盖。脚本检查主机及板端空间；这批前缀包含原始FP16权重，大小高于前一轮两个query小图。使用固定新目录`hidden-layer0`，保留原query结果。

```bash
python3 "$REPO_DIR/targets/cpro/tools/validate_draft_vector.py" \
  --compiled "$LAYER_DIR/compiled" --ddk-root "$CPRO_DDK_ROOT" \
  --config "$AI_RUN_DIR/runtime/runtime.json" \
  --output "$LAYER_DIR/run" --apply

# 启动后回收同一请求；尚未完成时再次recover，不重复apply。
python3 "$REPO_DIR/targets/cpro/tools/validate_draft_vector.py" \
  --recover "$LAYER_DIR/run" --output "$LAYER_DIR/result"

python3 "$REPO_DIR/targets/cpro/tools/validate_draft_vector.py" \
  --analyze "$LAYER_DIR/result" > "$LAYER_DIR/diagnosis.json"

python3 - "$LAYER_DIR/diagnosis.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print(json.dumps({k: d[k] for k in (
    'numerical_validation', 'coverage_complete', 'coverage',
    'repeat_bitwise_stable', 'source_cases', 'independent_cast_pairs')}, ensure_ascii=False))
PY
```

贴最后的诊断摘要即可。完整文件保留每点的源码边、权重、误差、零值、十次稳定性和编译归属；缺失点不会写成整批通过。NumPy按源AIR计算，另列原参考gamma舍入方式作为诊断对照；原模型golden和0.01/0.01容差不变。独立前缀会影响融合，定位结果仍需回到整Draft确认。本批已完成主机检查，真实DDK编译与板端验证待执行。
