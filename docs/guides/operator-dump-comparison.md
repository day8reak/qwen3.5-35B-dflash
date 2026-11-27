# 算子 dump 三端对比

在保留本次 dump 和两端参考的主机上执行。对已有单层 linear prefill 用例，`--case` 会读本次 AIR 找到变量的生产者和输出槽，用本次板端回执绑定整层输出，并按有效 token、完整状态或共同 token/expert 对齐三端张量。

```bash
python3 targets/cpro/tools/compare_operator_dump.py --case "$case_dir"
```

`board/L00/dump-comparison/comparison.json` 记录每个点的源图边、文件、dtype、字节偏移、比较范围、3×3 指标和覆盖缺口。`cosine.txt`、`relative_l2.txt` 和 `error_ratio.txt` 便于逐项阅读。`matrices.csv` 另含 KL、RMSE 和平均相对误差。矩阵行是 actual、列是 reference。整数编码的 error_ratio 统计数值不等的比例。

终端的首个大偏差按观测点顺序和 CPro 对 310P 判断，默认 cosine ≤ 0.99、relative L2 ≥ 0.05 或指标不可计算即超限。可通过 `--min-cosine` 和 `--max-rel-l2` 调整。CPU 对两端的数据保留在矩阵内。对并行分支，此顺序给出首个超限观测点，算子自身的误差需再核对其实际输入。

解析器按已观察到的 CPro 文件名提取节点、尾部输出槽、dtype 和形状，采用文件声明的 dtype。L0 配置还会用同一次的输入和公共输出检查能否逐元素复现某种编码。内容相同的重复输出合并。若文件多出 32 字节且偏移未确定，分别比较偏移 0 和 32。多种读法给出不同判定时标为 `AMBIGUOUS`；全都超限时标为 `LARGE_ERROR_ALL_CANDIDATES`，各张候选矩阵保留。已从工具格式确认偏移时，可传 `--padding-offset 0` 或 `--padding-offset 32`。文件缺失、形状不符等记在各点的 `issues`，首个超限点以前的缺口列在 `earlier_unresolved`。脚本本身不运行模型。

## 其他层或模型

同一工具可读取任意层的 CPU/310P 数值 NPZ 和同次板端的 `dump-inventory.json`。写一份 JSON 映射，路径相对映射文件目录。`points` 按希望检查的前向顺序排列，`edge` 使用该模型实际 AIR/OMC 算子名和输出槽。下列名字仅作示例。

```json
{
  "inventory": "replay/dump-inventory.json",
  "dump_dir": "replay/dump",
  "cpu": "references.npz",
  "310p": "references.npz",
  "valid_rows": 19,
  "points": [
    {"stage": "layer7", "name": "input_norm", "edge": "RmsnormPatch_12:0", "cpu": "cpu_l7_norm", "310p": "310p_l7_norm", "shape": [1,64,2048], "token_axis": 1},
    {"stage": "layer7", "name": "quant_scale", "edge": "DynamicQuant_42:1", "cpu": "cpu_l7_scale", "310p": "310p_l7_scale", "shape": [64], "token_axis": 0}
  ]
}
```

```bash
python3 targets/cpro/tools/compare_operator_dump.py --mapping mapping.json --output dump-comparison
```

CPU 和 310P 的 NPZ 也可以分开提供。某个点省略 `cpu` 或 `310p` 时默认读取 `name` 同名数组。`token_axis` 和 `valid_rows` 可用于任意序列长度；省略 `token_axis` 时对比整个张量，适用于缓存或单步输出。三端逻辑轴确实相同但分组方式不同时可显式指定 `shape`；板端存储轴顺序已知时先用 `cpro_permute` 声明转置。按参考分数试选轴顺序会掩盖真实误差。

参考张量已经按 `节点名:输出槽` 命名时，可省略整个 `points` 列表，改填 `"air": "weight/export.air"`。脚本在 AIR 中查找具有任一端参考的输出，按图依赖排序并对比完整张量。若两个后端共用一个 NPZ 且数组名如 `cpu_Add_2:0`、`310p_Add_2:0`，再填 `"cpu_prefix": "cpu_"` 和 `"310p_prefix": "310p_"`。需要只看有效 token 时使用显式的逐点范围。
