# 当前阶段：批量定位 hidden 偏差

2026-11-05：完整Draft精度仍未通过，性能优化暂缓。首次query小图的单输出通过，但五输出版本中的FP16检查点失败；两图末端不一致，暂不能把它当作模型首错。FP32探针输出在编译后还经过FP16→FP32转换，不能据此证明内部FP32计算。已有实测见[结果页](results.md)。

本轮一次覆盖**完整第0层**：特征投影、各RMSNorm、Q/K/V投影、RoPE、QK/Softmax/PV、输出投影、两段残差及MLP。每个检查点单独输出，避免成对FP16/FP32输出互相干扰；使用保留的case 0／2／17（空上下文、部分上下文、满缓存），每组3次预热＋10次比较。同一检查点只编译、上传一次，三组输入共用OMC。

## DDK服务器执行

沿用已配置的 `REPO_DIR`、`AI_RUN_DIR`、`CPRO_DDK_ROOT`、`CPRO_DDK_ID` 和支持untouched的运行器。保留 `top1-integration`、`draft/source/prepared`、`runtime`、`common`；探针继续使用原AIR、权重、编译报告和untouched精度夹具，不需要Torch/TorchAir。

三组源图参考计算已完成，当前受服务器空间阻塞（仅585MB可用）。先清理旧 `pytorch-omc`／`pytorch-optimized-omc` 中已打包的编译副本；脚本验证模型和编译完成状态，保留OMC、报告、诊断和profiling CSV，不访问设备。去掉 `--apply` 可只查看计划。

```bash
git -C "$REPO_DIR" pull --ff-only
export TOP1_DIR="$AI_RUN_DIR/draft/tests/top1-integration"
export LAYER_DIR="$TOP1_DIR/hidden-layer0"

python3 "$REPO_DIR/targets/cpro/tools/clean_draft_probes.py" \
  --root "$AI_RUN_DIR" --compiler-scratch --apply
df -h "$AI_RUN_DIR"

# 上次空间预检未创建compiled目录，仍用同一路径。
python3 "$REPO_DIR/targets/cpro/tools/probe_draft_hidden.py" \
  --suite layer0 --air-export "$TOP1_DIR/export" \
  --source "$AI_RUN_DIR/draft/source/prepared" \
  --result "$TOP1_DIR/output-init/result" \
  --ddk-root "$CPRO_DDK_ROOT" --ddk-id "$CPRO_DDK_ID" \
  --tbe-diagnosis "$AI_RUN_DIR/common/tbe-diagnosis" \
  --output "$LAYER_DIR/compiled"
```

新版本只建立一份本轮只读权重快照，每点成功打包后释放编译副本，保留独立OMC、输入、参考值和日志。启动时打印实际空间预算；部署tar及回收输出另需空间。已知退出的失败保留现场并继续；退出不明或剩余空间不足会停止，并在`coverage`标明缺口。原检查点和三组输入覆盖不变。

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
