"""CPU math for one fresh W4 T64 Prefill through four decoder layers and the LM head."""
import torch
from torch.nn import functional as F

from ..layer_probe.qlinear_math import dynamic_quant, group_matmul
from ..original_quant import OriginalQuantWeights
from ..layer_probe.target_weights import read_tensors
from .gdr_math import compute


def linear(x, weight, scale):
    shape = x.shape
    rows = x.reshape(-1, shape[-1]).contiguous().numpy()
    a8, token_scale, _ = dynamic_quant(rows)
    _, y = group_matmul(a8, token_scale, weight, scale)
    return torch.from_numpy(y).reshape(*shape[:-1], weight.shape[0])


def norm(x, module, *, gated=False):
    value = x.float()
    weight = module.weight.detach().float()
    gamma = weight if gated else 1.0 + weight
    value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + module.variance_epsilon) * gamma
    return value if gated else value.to(x.dtype)


def verify_floating_weights(model, checkpoint):
    """The source loader can skip missing weights. Verify every floating checkpoint parameter the CPU uses."""
    names = ['norm.weight']
    for i, layer in enumerate(model.model.layers):
        tail = ['input_layernorm.weight', 'post_attention_layernorm.weight', 'mlp.shared_expert_gate.weight']
        if layer.block_type == 'linear_attention':
            tail.extend(['linear_attn.conv1d.weight', 'linear_attn.norm.weight', 'linear_attn.A_log', 'linear_attn.dt_bias'])
            if layer.linear_attn.conv1d.bias is not None:
                tail.append('linear_attn.conv1d.bias')
        else:
            tail.extend(['self_attn.q_norm.weight', 'self_attn.k_norm.weight'])
        names.extend(f'layers.{i}.{suffix}' for suffix in tail)
    prefixes = ['model.language_model.', 'model.', '']
    tensors = read_tensors(checkpoint, {prefix + name for name in names for prefix in prefixes})
    sources = {}
    for name in names:
        matches = [prefix + name for prefix in prefixes if prefix + name in tensors]
        if len(matches) != 1:
            raise ValueError(f'Expected a unique checkpoint tensor for {name}, found {matches}')
        param = model.model.get_parameter(name).detach()
        if not torch.equal(param, tensors[matches[0]].to(param.dtype)):
            raise ValueError(f'Loaded floating parameter differs from checkpoint: {name}')
        sources[name] = matches[0]
    return sources


class CpuPrefill:
    def __init__(self, wrapper, quant_directory):
        self.model = wrapper.model
        self.weights = OriginalQuantWeights(quant_directory)

    def project(self, x, name, module):
        weight, scale = self.weights.logical('target.' + name + '.weight', tuple(module.weight.shape))
        return linear(x, weight, scale)

    def gdn(self, x, layer, prefix, pair, *, stages=None):
        def project(name):
            y = self.project(x, prefix + '.' + name, getattr(layer, name))
            if stages is not None:
                stages[name] = y
            return y

        qkv = project('in_proj_qkv').transpose(1, 2)
        history, state = pair
        joined = torch.cat((history, qkv), dim=2).to(layer.conv1d.weight.dtype)
        next_history = joined[:, :, -history.size(2):].to(history.dtype).contiguous()
        conv = F.conv1d(joined, layer.conv1d.weight, layer.conv1d.bias, groups=qkv.size(1))
        conv = conv[:, :, -x.size(1):]
        mixed = F.silu(conv).to(qkv.dtype).transpose(1, 2)
        q, k, v = mixed.split((layer.key_dim, layer.key_dim, layer.value_dim), dim=-1)
        repeat = layer.num_v_heads // layer.num_k_heads
        q, k = [t.reshape(1, x.size(1), layer.num_k_heads, layer.head_k_dim)
                .repeat_interleave(repeat, dim=2) for t in (q, k)]
        v = v.reshape(1, x.size(1), layer.num_v_heads, layer.head_v_dim)
        g = -layer.A_log.float().exp() * F.softplus(project('in_proj_a').float() + layer.dt_bias.float())
        beta = project('in_proj_b').sigmoid()
        core, final = compute(*(t.contiguous().numpy() for t in (q, k, v, g, beta, state)))
        core = torch.from_numpy(core)
        z = project('in_proj_z').reshape_as(core)
        gated = (norm(core, layer.norm, gated=True) * F.silu(z.float())).to(core.dtype)
        y = self.project(gated.flatten(2), prefix + '.out_proj', layer.out_proj)
        if stages is not None:
            stages.update(conv=conv.transpose(1, 2), query=q, key=k, value=v, g=g, beta=beta,
                          core=core, out_proj_input=gated.flatten(2), hidden=y)
        return y, (next_history, torch.from_numpy(final).to(state.dtype))

    def attention(self, x, layer, prefix, pair, args, *, stages=None):
        heads, kv, d = layer.num_heads, layer.num_key_value_heads, layer.head_dim
        b, t = x.shape[:2]

        def project(name):
            return self.project(x, prefix + '.' + name, getattr(layer, name))

        q, gate = project('q_proj').reshape(b, t, heads, 2*d).chunk(2, dim=-1)
        k = norm(project('k_proj').reshape(b, t, kv, d), layer.k_norm)
        q = norm(q, layer.q_norm)
        v = project('v_proj').reshape(b, t, kv, d)
        rope = layer.rotary_emb
        angles = args['position_ids'].float().unsqueeze(-1) * rope.inv_freq.float()
        angles = torch.cat((angles, angles), dim=-1).unsqueeze(2)
        cos = (angles.cos() * rope.attention_scaling).to(x.dtype)
        sin = (angles.sin() * rope.attention_scaling).to(x.dtype)

        def rotate(tensor):
            a, rest = tensor.split((angles.size(-1), d - angles.size(-1)), dim=-1)
            first, second = a.chunk(2, dim=-1)
            return torch.cat((a*cos + torch.cat((-second, first), dim=-1)*sin, rest), dim=-1)

        q, k = rotate(q), rotate(k)
        positions = args['new_kv_cache_pos'].long()
        pages = layer.block_table[0, positions // layer.block_size].long()
        offsets = positions % layer.block_size
        states = []
        for dense, before in zip((k, v), pair):
            after = before.clone()
            after[pages, :, offsets, :] = dense.reshape(t, kv*d//16, 16).to(after.dtype)
            states.append(after)
        q = q.transpose(1, 2).double()
        length = args.get('kv_length', t)
        if 'kv_length' in args:
            # Decode attends through the current position in the updated paged caches.
            logical = torch.arange(length)
            pages = layer.block_table[0, logical // layer.block_size].long()
            k, v = [cache[pages, :, logical % layer.block_size, :].reshape(b, length, kv, d) for cache in states]
        k, v = [a.transpose(1, 2).repeat_interleave(heads//kv, dim=1).double() for a in (k, v)]
        logits = (q @ k.transpose(-1, -2)) * layer.scaling
        prob = torch.softmax(logits + args['attention_mask'][..., :length].double(), dim=-1)
        core = (prob @ v).to(torch.float16).transpose(1, 2)
        y = (core * gate.sigmoid()).reshape(b, t, heads*d)
        if stages is not None:
            stages.update(query=q.to(x.dtype), core=core.transpose(1, 2), proj_input=y)
        return self.project(y, prefix + '.o_proj', layer.o_proj), states

    def routes(self, x, layer, prefix):
        logits = self.project(x.flatten(0, 1), prefix + '.gate', layer.gate)
        probs, ids = logits.float().softmax(dim=-1).topk(layer.top_k, dim=-1)
        if layer.norm_topk_prob:
            probs = probs / probs.sum(dim=-1, keepdim=True)
        return probs.to(torch.float16), ids

    def moe(self, x, layer, prefix, *, routes=None):
        flat = x.flatten(0, 1)
        shared = layer.shared_expert
        shared_prefix = prefix + '.shared_expert.'
        gate, up = [self.project(flat, shared_prefix + name, getattr(shared, name))
                    for name in ('gate_proj', 'up_proj')]
        shared_out = self.project(F.silu(gate)*up, shared_prefix + 'down_proj', shared.down_proj)
        shared_gate = F.linear(flat.float(), layer.shared_expert_gate.weight.float()).to(flat.dtype)
        shared_out = shared_out * shared_gate.sigmoid()
        probs, ids = self.routes(x, layer, prefix) if routes is None else routes
        n, h = layer.num_experts, x.size(-1)
        middle = self.model.config.moe_intermediate_size
        w13, s13 = self.weights.logical('target.' + prefix + '.experts.gate_up_proj', (n, middle*2, h))
        w2, s2 = self.weights.logical('target.' + prefix + '.experts.down_proj', (n, h, middle))
        routed = torch.empty((*ids.shape, h), dtype=torch.float16)
        for expert in ids.unique().tolist():
            rows, slots = torch.where(ids == expert)
            gate, up = linear(flat[rows], w13[expert], s13[expert, ..., None]).chunk(2, dim=-1)
            act = (F.silu(gate.float()) * up.float()).to(torch.float16)
            routed[rows, slots] = linear(act, w2[expert], s2[expert, ..., None])
        output = (routed.float() * probs.float().unsqueeze(-1)).sum(dim=1).to(torch.float16)
        return (output + shared_out).reshape_as(x)

    @torch.inference_mode()
    def run(self, args):
        layers = self.model.model.layers
        if [l.block_type for l in layers] != ['linear_attention']*3 + ['full_attention']:
            raise ValueError('Expected three GDN layers and one full attention layer')
        t = 64
        positions = torch.arange(t)
        mask = args['attention_mask']
        triangular = torch.arange(mask.size(-1))[None, :] <= positions[:, None]
        if (not torch.equal(args['position_ids'], positions[None])
                or not torch.equal(args['new_kv_cache_pos'], positions)
                or args['output_pos'].tolist() != [t-1] or args['effective_length'].tolist() != [t]
                or args['allQLen'] != [t] or mask.shape != (1, 1, t, self.model.config.kv_cache_max_len)
                or not torch.equal(mask[0, 0], torch.where(triangular, 0., -torch.inf))
                or any(torch.count_nonzero(v) for pair in args['past_key_values'] for v in pair)):
            raise ValueError('CPU reference requires the fresh T64 Prefill input contract')
        x = (args['inputs_embeds'] * args['embed_scale']).to(torch.float16)
        states, boundaries = [], []
        for i, layer in enumerate(layers):
            print(f'[target-prefill] CPU layer {i}', flush=True)
            prefix = f'layers.{i}.'
            pre = norm(x, layer.input_layernorm)
            pair = args['past_key_values'][i]
            if i < 3:
                mix, new_pair = self.gdn(pre, layer.linear_attn, prefix + 'linear_attn', pair)
            else:
                mix, new_pair = self.attention(pre, layer.self_attn, prefix + 'self_attn', pair, args)
            x = x + mix
            x = x + self.moe(norm(x, layer.post_attention_layernorm), layer.mlp, prefix + 'mlp')
            states.extend(new_pair)
            boundaries.append(x)
        head_input = norm(x[:, -1:, :].to(torch.float16), self.model.model.norm)
        logits = self.project(head_input, 'lm_head', self.model.lm_head).float()
        return (logits, *states), boundaries
