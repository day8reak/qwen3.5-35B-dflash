"""Source-level boundaries for a T64 L0 GDN replay. Views stay in their consuming slice."""

PROJECTIONS = {'qkv': 8192, 'z': 4096, 'a': 32, 'b': 32, 'out': 2048}
ENTRY = ('hidden', 'history', 'state', 'length')
WHOLE_OUTPUTS = ('out', 'new_history', 'recurrent_cache')
STATE = {'history', 'state', 'new_history', 'final_state', 'recurrent_cache'}
STAGES = {}
DESCRIPTION = {
    'hidden': 'linear_attn.forward(hidden_states)：input_layernorm 后的 GDN 输入',
    'history': 'past_key_values[0]：进入 L0 的卷积历史',
    'state': 'past_key_values[1]：进入 L0 的递推状态',
    'length': 'effective_length：本次更新 19 个有效 token',
    'window': 'joined 的末 67 个位置：F.conv2d 的实际卷积窗口，含前三个历史位置',
    'new_history': 'new_conv_state：T64 物理位置 60–63 的 QKV',
    'conv': 'conv_out：逐通道因果卷积输出，进入第一个 SiLU',
    'conv_silu': 'qkv_conv：SiLU(conv_out)，按通道优先布局读取',
    'query': 'query：从卷积激活拆出 Q，按 value head 展开后送入 GDR',
    'key': 'key：从卷积激活拆出 K，按 value head 展开后送入 GDR',
    'value': 'value：从卷积激活拆出 V，按 head 整理后送入 GDR',
    'biased_a': 'a.float() + dt_bias：softplus 的输入',
    'softplus': 'F.softplus(a.float() + dt_bias)：正的步长',
    'g': 'g = -exp(A_log) * softplus(...)：GDR 的对数衰减输入',
    'beta': 'beta = sigmoid(b)：GDR 的更新系数',
    'core': 'core_attn_out：chunk GDR 的逐 token 输出 0',
    'final_state': 'last_recurrent_state：chunk GDR 的状态输出 1，已处理 19 个 token',
    'recurrent_cache': 'past_key_values[1]：模块返回的 FP16 recurrent cache，供下一次调用使用',
    'rms': 'norm.forward 的 hidden_states：core 按 head 做 RMSNorm 并乘 gamma 的结果',
    'gate_silu': 'norm.forward 的 F.silu(gate.float())：z 门控激活',
    'product': 'norm.forward 的 hidden_states * SiLU(gate)：转回 FP16 前的乘积',
    'gated': 'out_proj 的输入：门控乘积转回 FP16 并合并 heads',
}

def projection(p):
    name = 'out_proj' if p == 'out' else 'in_proj_' + p
    var = {'qkv': 'mixed_qkv', 'z': 'z', 'a': 'a', 'b': 'b', 'out': 'output'}[p]
    activation = 'gated' if p == 'out' else 'hidden'
    codes, scale = p + '_a8', p + '_scale'
    STAGES[name + '_quant'] = ((activation,), (codes, scale))
    STAGES[name] = ((codes, scale), (p,))
    DESCRIPTION[codes] = f'{name} 的 x_quant：DynamicQuant 的 INT8 激活'
    DESCRIPTION[scale] = f'{name} 的 pertoken_scale：DynamicQuant 的逐 token 浮点尺度（matmul 转型前）'
    DESCRIPTION[p] = f'{var} = self.{name}(...)：W4 matmul 输出'

for _proj in ('qkv', 'z', 'a', 'b'):
    projection(_proj)
STAGES.update({
    'conv_history': (('qkv', 'history'), ('window', 'new_history')),
    'depthwise_conv': (('window',), ('conv',)),
    'conv_silu': (('conv',), ('conv_silu',)),
    **{k: (('conv_silu',), (k,)) for k in ('query', 'key', 'value')},
    'decay_bias': (('a',), ('biased_a',)),
    'decay_softplus': (('biased_a',), ('softplus',)),
    'decay_g': (('softplus',), ('g',)),
    'beta_sigmoid': (('b',), ('beta',)),
    'chunk_gdr': (('query', 'key', 'value', 'g', 'beta', 'state', 'length'), ('core', 'final_state')),
    'gated_rmsnorm': (('core',), ('rms',)),
    'gate_silu': (('z',), ('gate_silu',)),
    'gated_mul': (('rms', 'gate_silu'), ('product', 'gated')),
})
projection('out')

# Canonical external shapes. Each AIR cut records its own original view separately.
SHAPES = dict(hidden=(1,64,2048), history=(1,8192,4), state=(1,32,128,128), length=(1,),
              window=(1,8192,1,67), new_history=(1,8192,4), conv=(1,8192,64), conv_silu=(1,8192,64),
              final_state=(1,32,128,128), recurrent_cache=(1,32,128,128), gated=(64,4096))
for _name in ('query', 'key', 'value', 'core', 'rms', 'gate_silu', 'product'):
    SHAPES[_name] = (1,64,32,128)
for _name in ('biased_a', 'softplus', 'g', 'beta'):
    SHAPES[_name] = (1,64,32)
for _p, _width in PROJECTIONS.items():
    SHAPES[_p] = (64,_width)
    SHAPES[_p + '_a8'] = (64,4096 if _p == 'out' else 2048)
    SHAPES[_p + '_scale'] = (64,)

def check_view(name, shape):
    """Match semantic axes; allow known contiguous views and singleton axes only."""
    dims = tuple(d for d in shape if d != 1)
    expected = tuple(d for d in SHAPES[name] if d != 1)
    allowed = {expected}
    if name in ('query', 'key', 'value', 'core', 'rms', 'gate_silu', 'product', 'z', 'gated'):
        allowed.update(((64, 32, 128), (2048, 128), (64, 4096)))
    if dims not in allowed:
        raise ValueError(f'{name}: got {tuple(shape)}, expected axes {sorted(allowed)}')


def canonical(name, array):
    import numpy as np
    a = np.asarray(array)
    check_view(name, a.shape)
    return np.ascontiguousarray(a.reshape(SHAPES[name]))


def compare_view(name, array):
    a = canonical(name, array)
    if name in STATE:
        return a
    if name == 'window':
        # Three history positions precede the 19 valid tokens.
        return a[0, :, 0, 3:22].T
    if name in ('conv', 'conv_silu'):
        return a[0, :, :19].T
    return a.reshape(64, -1)[:19]
