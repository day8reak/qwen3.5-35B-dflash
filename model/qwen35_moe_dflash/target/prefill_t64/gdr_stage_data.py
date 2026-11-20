"""Logical interfaces of the three unscaled 64/19 prefill GDR modules."""
TOKEN_OPERANDS = ('query', 'key', 'value', 'g', 'beta')
GDR = (*TOKEN_OPERANDS, 'initial_state', 'effective_length', 'core', 'final_state')
NORM = ('rms_input', 'gamma', 'rms_output', 'gate', 'silu_input', 'silu_output', 'mul_lhs', 'mul_rhs', 'mul_output', 'gated_return')
STATIC = ('initial_state', 'effective_length', 'final_state', 'gamma')
GROUPS = {'gdr': GDR, 'norm': NORM}
BASIS = 'SEGMENT_310P_MEASURED_GDR_RMS_SILU_64_VALID19'


def shapes(name, t=64):
    if name in ('initial_state', 'final_state'):
        return [[1, 32, 128, 128]]
    if name == 'gamma':
        return [[128], [1, 128], [1, 1, 128], [1, 1, 1, 128]]
    if name == 'effective_length':
        return [[1]]
    if name in ('g', 'beta'):
        return [[1, t, 32], [t, 32]]
    if name == 'golden_hidden':
        return [[1, t, 2048]]
    return [[1, t, 32, 128], [1, t, 4096], [t, 32, 128], [t*32, 128], [1, t*32, 128], [t, 4096]]


def logical(value, name, t=64):
    views = shapes(name, t)
    if list(value.shape) not in views:
        raise ValueError(f'{name}: unexpected {value.shape}, expected one of {views}')
    return value.reshape(views[0])


def case_id(layer, group):
    return f'l{layer:02}_c00_{group}_prefill_stages_o0'


# Names describe the value at the consumer boundary, including casts in the source graph.
DESCRIPTIONS = {
    'golden_hidden': ('GDN 整层输出', '门控结果经 out_proj 后的整层输出，在 decoder 残差加法之前'),
    'query': ('GDR 输入', '当前 token 的 Q，送入 chunk GDR 的读出向量'),
    'key': ('GDR 输入', '当前 token 的 K，参与状态预测与更新'),
    'value': ('GDR 输入', '当前 token 的 V，参与计算状态更新的残差'),
    'g': ('GDR 输入', '对数衰减 g，递推按 exp(g) 衰减上一状态'),
    'beta': ('GDR 输入', 'delta 更新系数 beta，控制本步残差写入量'),
    'initial_state': ('GDR 输入', '进入本次 prefill 前的 recurrent state，本例初始为零'),
    'effective_length': ('GDR 输入', '有效长度 INT16 [19]，此 chunk 的物理行数为 64'),
    'core': ('GDR 输出', '逐 token 的 GDR 读出，作为后续归一化的数据来源'),
    'final_state': ('GDR 输出', '完成有效 token 0–18 后的 recurrent state'),
    'rms_input': ('RMSNorm 输入', '实际送入 RMSNorm 的 core，包含该后端在此处执行的类型转换'),
    'gamma': ('RMSNorm 输入', '通道权重 gamma，即作用于归一化结果的乘数'),
    'rms_output': ('RMSNorm 输出', '乘过 gamma 的归一化激活，即 forward 使用的第 0 个结果'),
    'gate': ('SiLU 输入来源', 'z 投影得到的原始 gate，在 SiLU 输入转换之前'),
    'silu_input': ('SiLU 输入', '实际送入 SiLU 的 gate，对应源码 gate.to(torch.float32)'),
    'silu_output': ('SiLU 输出', '门值 SiLU(gate)，供后续逐元素乘法使用'),
    'mul_lhs': ('门控乘法输入', '本次乘法实际读取的归一化激活 hidden_states'),
    'mul_rhs': ('门控乘法输入', '本次乘法实际读取的 SiLU 门值'),
    'mul_output': ('门控乘法输出', 'hidden_states × SiLU(gate) 的结果，在返回值转换之前'),
    'gated_return': ('门控模块返回', '乘积转回源输入 dtype 后传往 out_proj 的激活'),
}
