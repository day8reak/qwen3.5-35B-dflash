"""Target CPro attention call with Draft-owned masks and temporary NZ storage.

The GE signature follows the original Target exporter. No kernel is implemented
here: OMG and the selected HIAI runtime must still support the resulting node.
"""
import torch
from torch import nn


class TargetAttention(nn.Module):
    def __init__(self, config, capacity, block_rows, kernel):
        super().__init__()
        if config.head_dim % 16:
            raise ValueError('Target CPro NZ attention requires head_dim divisible by 16')
        self.heads, self.kv_heads, self.dim = config.num_attention_heads, config.num_key_value_heads, config.head_dim
        self.rows, self.block_size = block_rows, 32
        self.length = capacity + block_rows
        self.blocks = (self.length + self.block_size - 1) // self.block_size
        self.extent = self.blocks * self.block_size
        self.kernel = kernel
        self.register_buffer('block_table', torch.arange(self.blocks, dtype=torch.int32).reshape(1, -1))
        # These are length ports, not Gather indices. Target uses INT64 here.
        self.register_buffer('query_lengths', torch.tensor([block_rows], dtype=torch.int64))
        self.register_buffer('kv_padding', torch.zeros(1, self.kv_heads, self.extent - self.length, self.dim, dtype=torch.float16))
        self.register_buffer('mask_padding', torch.full((1, 1, block_rows, self.extent - self.length), -float('inf'), dtype=torch.float16))

    def pack_kv(self, tensor):
        # ND [1,H,S,D] -> [blocks,H*D/16,32,16], matching Target CacheUpdate's
        # physical layout. This is private attention storage, never committed KV.
        if self.extent != self.length:
            tensor = torch.cat((tensor, self.kv_padding), dim=2)
        return tensor.transpose(1, 2).reshape(self.blocks, self.block_size, self.kv_heads * self.dim // 16, 16).transpose(1, 2).contiguous()

    def forward(self, query, key, value, visible):
        query_nz = query.reshape(1, self.heads, self.rows, self.dim // 16, 16).transpose(2, 3).contiguous().reshape(1, self.heads * self.dim // 16, self.rows, 16)
        mask = torch.where(visible, torch.zeros_like(visible, dtype=torch.float16),
                           torch.full_like(visible, -float('inf'), dtype=torch.float16))
        if self.extent != self.length:
            mask = torch.cat((mask, self.mask_padding), dim=-1)
        output = self.kernel(query=query_nz, key=[self.pack_kv(key)], value=[self.pack_kv(value)],
            pse_shift=self.query_lengths, atten_mask=mask,
            actual_seq_lengths_q=[self.rows], actual_seq_lengths_kv=[self.extent],
            block_table=self.block_table, num_heads=self.heads, num_key_value_heads=self.kv_heads,
            block_size=self.block_size, input_layout='BNSD', scale_value=self.dim ** -0.5, inner_precise=2)
        return output.reshape(1, self.heads, self.dim // 16, self.rows, 16).transpose(2, 3).contiguous().reshape(1, self.heads, self.rows, self.dim)

    def contract(self):
        return dict(policy='target-cpro-fused-attention-v1', op_type='AdnFusedInferAttention',
            num_heads=self.heads, num_key_value_heads=self.kv_heads, head_dim=self.dim,
            block_size=self.block_size, input_layout='BNSD', inner_precise=2, scale_value=self.dim ** -0.5,
            query_shape=[1, self.heads * self.dim // 16, self.rows, 16],
            kv_shape=[self.blocks, self.kv_heads * self.dim // 16, self.block_size, 16],
            mask_shape=[1, 1, self.rows, self.extent], block_table_shape=[1, self.blocks],
            logical_kv_rows=self.length, padded_kv_rows=self.extent,
            mask='DRAFT_VISIBILITY_WITH_INACTIVE_QUERY_ANCHOR_WITNESS_AND_MASKED_PADDING',
            cache='TEMPORARY_NZ_ONLY; PUBLIC_ND_COMMITTED_KV_UNCHANGED',
            precision='FUSED_INTERNAL_ROUNDING_UNVERIFIED_REQUIRES_DEVICE_COMPARISON')


_META_LIBRARY = None
_REGISTERED = False


def register_target_attention(torchair):
    """Register the Target's converter for this export process only."""
    global _META_LIBRARY, _REGISTERED
    try:
        op = torch.ops.npu.adn_fused_infer_attention.default
    except AttributeError as error:
        raise RuntimeError('Target CPro export needs torch_npu.adn_fused_infer_attention; use the original Target export environment') from error
    if _REGISTERED:
        return op
    if not torch._C._dispatch_has_kernel_for_dispatch_key('npu::adn_fused_infer_attention', 'Meta'):
        _META_LIBRARY = torch.library.Library('npu', 'IMPL', 'Meta')
        def meta(query, key, value, *args, **kwargs):
            return torch.empty_like(query)
        _META_LIBRARY.impl('adn_fused_infer_attention', meta)

    @torchair.register_fx_node_ge_converter(op)
    def convert(query, key, value, pse_shift=None, atten_mask=None,
                actual_seq_lengths_q=None, actual_seq_lengths_kv=None,
                dequant_scale1=None, quant_scale1=None, dequant_scale2=None,
                quant_scale2=None, quant_offset2=None, antiquant_scale=None,
                antiquant_offset=None, block_table=None, kv_padding_size=None,
                all_seq_lengths_q=None, actual_seq_lengths_q_back=None,
                num_heads=1, scale_value=1.0, input_layout='BSH',
                num_key_value_heads=0, block_size=0, inner_precise=1, meta_outputs=None):
        from torchair._ge_concrete_graph.utils import dtype_promote
        from torchair.ge import DataType
        actual_seq_lengths_q = dtype_promote(actual_seq_lengths_q, target_dtype=DataType.DT_INT64)
        actual_seq_lengths_kv = dtype_promote(actual_seq_lengths_kv, target_dtype=DataType.DT_INT64)
        # The original CPro exporter forwards the Python pse_shift argument to
        # GE all_seq_lengths_q. It is NOT an attention bias tensor.
        return torchair.ge.custom_op('AdnFusedInferAttention', inputs={
            'query': query, 'key': key, 'value': value, 'pse_shift': None, 'atten_mask': atten_mask,
            'actual_seq_lengths_q': actual_seq_lengths_q, 'actual_seq_lengths_kv': actual_seq_lengths_kv,
            'dequant_scale1': dequant_scale1, 'quant_scale1': quant_scale1, 'dequant_scale2': dequant_scale2,
            'quant_scale2': quant_scale2, 'quant_offset2': quant_offset2, 'antiquant_scale': antiquant_scale,
            'antiquant_offset': antiquant_offset, 'block_table': block_table, 'kv_padding_size': kv_padding_size,
            'all_seq_lengths_q': pse_shift, 'actual_seq_lengths_q_back': actual_seq_lengths_q_back},
            attrs={'num_heads': torchair.ge.attr.Int(num_heads), 'scale_value': torchair.ge.attr.Float(scale_value),
                'input_layout': torchair.ge.attr.Str(input_layout), 'num_key_value_heads': torchair.ge.attr.Int(num_key_value_heads),
                'block_size': torchair.ge.attr.Int(block_size), 'inner_precise': torchair.ge.attr.Int(inner_precise)},
            outputs=['attention_out'])
    _REGISTERED = True
    return op
