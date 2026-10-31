"""Native Jittor QueST tokenizer; no Torch execution or imports.

Architecture follows the pinned QueST implementation and its FSQ dependency.
Numerical validation status is recorded separately in reports/, not inferred
from this file merely existing. See THIRD_PARTY_NOTICES.md for attribution.
"""
import math

import jittor as jt
from jittor import nn
import numpy as np

from .config import TokenizerConfig


def round_ties_even(value):
    """Match Torch round at exact halves, including negative half-integers."""
    lower = value.floor()
    fraction = value - lower
    increment = (fraction > 0.5) | ((fraction == 0.5) & ((lower.int32() % 2) != 0))
    return lower + increment.float32()


def round_ste(value):
    # 前向取整，反向保留对value的梯度；取整规则与PyTorch一致。
    return value + (round_ties_even(value) - value).stop_grad()


class FSQ(jt.Module):
    def __init__(self, dim, levels):
        self.dim, self.codebook_dim, self.codebook_size = dim, len(levels), math.prod(levels)
        self._levels = jt.array(np.asarray(levels, dtype=np.int32)).stop_grad()
        self._basis = jt.array(np.cumprod([1] + list(levels[:-1])).astype(np.int32)).stop_grad()
        self._levels.persistent = self._basis.persistent = False
        self.project_in = nn.Linear(dim, len(levels)) if dim != len(levels) else nn.Identity()
        self.project_out = nn.Linear(len(levels), dim) if dim != len(levels) else nn.Identity()

    # 用tanh将每一维限制到对应档位的范围，偶数档位需要半格偏移。
    def bound(self, value, eps=1e-3):
        half = (self._levels - 1).float32() * ((1 + eps) / 2)
        offset = jt.where(self._levels % 2 == 0, jt.ones_like(half) * 0.5, jt.zeros_like(half))
        ratio = offset / half
        shift = 0.5 * ((1 + ratio) / (1 - ratio)).log()
        return (value + shift).tanh() * half - offset

    # 先限幅、再取整，最后缩放为解码器使用的量化数值。
    def quantize(self, value):
        return round_ste(self.bound(value)) / (self._levels // 2).float32()

    # 按各维的档位数计算混合进制编号；[8,5,5,5]共有1000种组合。
    def codes_to_indices(self, codes):
        half = (self._levels // 2).float32()
        digits = codes * half + half
        return (digits * self._basis).sum(dim=-1).int32()

    def _indices_to_codes(self, indices):
        digits = (indices.unsqueeze(-1) // self._basis) % self._levels
        half = self._levels // 2
        return (digits - half).float32() / half.float32()

    def indices_to_codes(self, indices):
        return self.project_out(self._indices_to_codes(indices))

    def execute(self, value):
        # 将256维动作特征投影到4维，并分别量化到固定档位。
        codes = self.quantize(self.project_in(value))
        # 同时返回解码器使用的特征，以及策略训练使用的token编号。
        return self.project_out(codes), self.codes_to_indices(codes)


class PositionalEncoding1D(jt.Module):
    def __init__(self, channels):
        self.channels = math.ceil(channels / 2) * 2
        frequencies = 1.0 / (10000 ** (np.arange(0, self.channels, 2, dtype=np.float32) / self.channels))
        self.inv_freq = jt.array(frequencies.astype(np.float32)).stop_grad()

    def execute(self, tensor):
        batch, length, channels = tensor.shape
        phase = jt.arange(length).float32().unsqueeze(1) * self.inv_freq.unsqueeze(0)
        embedding = jt.stack((phase.sin(), phase.cos()), dim=-1).reshape((length, self.channels))
        return embedding[:, :channels].unsqueeze(0).repeat(batch, 1, 1)


class Summer(jt.Module):
    def __init__(self, penc):
        self.penc = penc

    def execute(self, value):
        return value + self.penc(value)


class LayerNorm(jt.Module):
    def __init__(self, dim, eps=1e-5):
        self.weight, self.bias = jt.ones(dim), jt.zeros(dim)
        self.eps = eps

    def execute(self, value):
        mean = value.mean(dim=-1, keepdims=True)
        centered = value - mean
        variance = (centered * centered).mean(dim=-1, keepdims=True)
        return centered / (variance + self.eps).sqrt() * self.weight + self.bias


class GroupNorm(jt.Module):
    def __init__(self, groups, channels, eps=1e-5):
        self.groups, self.channels, self.eps = groups, channels, eps
        self.weight, self.bias = jt.ones(channels), jt.zeros(channels)

    def execute(self, value):
        shape = value.shape
        grouped = value.reshape((shape[0], self.groups, -1))
        centered = grouped - grouped.mean(dim=-1, keepdims=True)
        normalized = centered / ((centered * centered).mean(dim=-1, keepdims=True) + self.eps).sqrt()
        affine_shape = (1, self.channels) + (1,) * (len(shape) - 2)
        return normalized.reshape(shape) * self.weight.reshape(affine_shape) + self.bias.reshape(affine_shape)


# 上三角设为负无穷，softmax后未来位置的权重为0。
def causal_mask(length):
    mask = np.where(np.triu(np.ones((length, length), dtype=bool), 1), -np.inf, 0).astype(np.float32)
    return jt.array(mask).stop_grad()


class MultiheadAttention(jt.Module):
    def __init__(self, dim, heads, dropout):
        self.dim, self.heads, self.head_dim = dim, heads, dim // heads
        self.in_proj_weight = jt.init.xavier_uniform((3 * dim, dim))
        self.in_proj_bias = jt.zeros(3 * dim)
        self.out_proj = nn.Linear(dim, dim)
        self.attention_dropout = nn.Dropout(dropout)

    def execute(self, query, key, value, mask=None):
        d = self.dim
        # 分别计算Q、K、V。自注意力时，三者来自同一段动作特征。
        q = nn.linear(query, self.in_proj_weight[:d], self.in_proj_bias[:d])
        k = nn.linear(key, self.in_proj_weight[d:2*d], self.in_proj_bias[d:2*d])
        v = nn.linear(value, self.in_proj_weight[2*d:], self.in_proj_bias[2*d:])
        # 拆分多头：[B,L,256]变为[B,4,L,64]，各个头并行计算。
        batch, length = q.shape[:2]
        q = q.reshape((batch, length, self.heads, self.head_dim)).transpose(0, 2, 1, 3)
        k = k.reshape((batch, k.shape[1], self.heads, self.head_dim)).transpose(0, 2, 1, 3)
        v = v.reshape((batch, v.shape[1], self.heads, self.head_dim)).transpose(0, 2, 1, 3)
        # 计算缩放点积；scores的最后两维表示查询位置和被关注位置。
        scores = jt.matmul(q * (self.head_dim ** -0.5), k.transpose(0, 1, 3, 2))
        # 加上因果掩码，禁止当前位置读取未来的信息。
        if mask is not None:
            scores = scores + mask
        # 沿被关注的位置做softmax，再对V加权求和。
        probabilities = self.attention_dropout(nn.softmax(scores, dim=-1))
        # 合并各个头，还原[B,L,256]，最后通过输出投影。
        output = jt.matmul(probabilities, v).transpose(0, 2, 1, 3).reshape((batch, length, d))
        return self.out_proj(output)


class EncoderLayer(jt.Module):
    def __init__(self, dim, heads, dropout):
        self.self_attn = MultiheadAttention(dim, heads, dropout)
        self.linear1, self.linear2 = nn.Linear(dim, 4 * dim), nn.Linear(4 * dim, dim)
        self.norm1, self.norm2 = LayerNorm(dim), LayerNorm(dim)
        self.dropout, self.dropout1, self.dropout2 = nn.Dropout(dropout), nn.Dropout(dropout), nn.Dropout(dropout)

    def execute(self, value, mask=None):
        normalized = self.norm1(value)
        value = value + self.dropout1(self.self_attn(normalized, normalized, normalized, mask))
        value = value + self.dropout2(self.linear2(self.dropout(nn.gelu(self.linear1(self.norm2(value))))))
        return value


class DecoderLayer(jt.Module):
    def __init__(self, dim, heads, dropout):
        self.self_attn, self.multihead_attn = MultiheadAttention(dim, heads, dropout), MultiheadAttention(dim, heads, dropout)
        self.linear1, self.linear2 = nn.Linear(dim, 4 * dim), nn.Linear(4 * dim, dim)
        self.norm1, self.norm2, self.norm3 = LayerNorm(dim), LayerNorm(dim), LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.dropout1, self.dropout2, self.dropout3 = nn.Dropout(dropout), nn.Dropout(dropout), nn.Dropout(dropout)

    def execute(self, value, memory, mask=None):
        normalized = self.norm1(value)
        value = value + self.dropout1(self.self_attn(normalized, normalized, normalized, mask))
        value = value + self.dropout2(self.multihead_attn(self.norm2(value), memory, memory))
        value = value + self.dropout3(self.linear2(self.dropout(nn.gelu(self.linear1(self.norm3(value))))))
        return value


class Encoder(jt.Module):
    def __init__(self, dim, heads, layers, dropout):
        self.layers = nn.ModuleList([EncoderLayer(dim, heads, dropout) for _ in range(layers)])

    def execute(self, value, mask=None):
        for layer in self.layers:
            value = layer(value, mask)
        return value


class Decoder(jt.Module):
    def __init__(self, dim, heads, layers, dropout):
        self.layers = nn.ModuleList([DecoderLayer(dim, heads, dropout) for _ in range(layers)])

    def execute(self, value, memory, mask=None):
        for layer in self.layers:
            value = layer(value, memory, mask)
        return value


class CausalConv1d(jt.Module):
    def __init__(self, dim, kernel, stride):
        self.padding, self.kernel, self.stride = kernel - 1, kernel, stride
        self.conv = nn.Conv1d(dim, dim, kernel, padding=self.padding, stride=stride)

    def execute(self, value):
        value = self.conv(value)
        trim = (2 * self.padding - self.kernel) // self.stride + 1
        return value[:, :, :-trim] if trim > 0 else value


class ExpandTime(jt.Module):
    def execute(self, value):
        return value.unsqueeze(2)


class CollapseTime(jt.Module):
    def execute(self, value):
        return value.squeeze(2)


class Mish(jt.Module):
    """Same formula as Torch; Jittor 1.3.11 nn.Mish calls missing jt.softplus."""
    def execute(self, value):
        return value * nn.softplus(value).tanh()


class Conv1dBlock(jt.Module):
    def __init__(self, dim, kernel, stride, causal):
        convolution = CausalConv1d(dim, kernel, stride) if causal else nn.Conv1d(dim, dim, kernel, stride=stride, padding=kernel//2)
        self.block = nn.Sequential(convolution, ExpandTime(), GroupNorm(8, dim), CollapseTime(), Mish())

    def execute(self, value):
        return self.block(value)


class ResidualTemporalBlock(jt.Module):
    def __init__(self, dim, factor, causal):
        count = int(math.log2(factor))
        kernels, strides = [5] + [3] * count, [2] * count + [1]
        if count == 0:
            kernels, strides = [3, 2], [1, 1]
        self.blocks = nn.ModuleList([Conv1dBlock(dim, k, s, causal) for k, s in zip(kernels, strides)])

    def execute(self, value):
        value = value.transpose(0, 2, 1)
        for block in self.blocks:
            value = block(value)
        return value.transpose(0, 2, 1)


class SkillVAE(jt.Module):
    def __init__(self, config=None):
        self.config = config or TokenizerConfig()
        c = self.config
        self.encoder_dim, self.decoder_dim = c.encoder_dim, c.decoder_dim
        self.skill_block_size, self.vq_type, self.fsq_level = c.skill_block_size, "fsq", list(c.levels)
        self.vq = FSQ(c.encoder_dim, c.levels)
        self.action_proj, self.action_head = nn.Linear(c.action_dim, c.encoder_dim), nn.Linear(c.decoder_dim, c.action_dim)
        self.conv_block = ResidualTemporalBlock(c.encoder_dim, c.downsample_factor, c.use_causal_encoder)
        self.encoder = Encoder(c.encoder_dim, c.encoder_heads, c.encoder_layers, c.attn_pdrop)
        self.decoder = Decoder(c.decoder_dim, c.decoder_heads, c.decoder_layers, c.attn_pdrop)
        self.add_positional_emb = Summer(PositionalEncoding1D(c.encoder_dim))
        self.fixed_positional_emb = PositionalEncoding1D(c.decoder_dim)

    # 动作先投影、时间下采样，再经因果Transformer编码；32步压缩为8个位置。
    def encode(self, actions):
        value = self.add_positional_emb(self.conv_block(self.action_proj(actions)))
        mask = causal_mask(value.shape[1]) if self.config.use_causal_encoder else None
        return self.encoder(value, mask)

    # 用32个带位置信息的查询，从量化特征中重建32步动作。
    def decode(self, codes):
        queries = self.fixed_positional_emb(jt.zeros((codes.shape[0], self.skill_block_size, self.decoder_dim)))
        mask = causal_mask(queries.shape[1]) if self.config.use_causal_decoder else None
        return self.action_head(self.decoder(queries, codes, mask))

    def execute(self, actions):
        codes, indices = self.vq(self.encode(actions))
        utilization = len(np.unique(indices.numpy())) / self.vq.codebook_size
        per_sample = np.mean([len(np.unique(item)) for item in indices.numpy()]) / indices.shape[1]
        return self.decode(codes), jt.array(utilization), jt.array(per_sample), jt.zeros(1), codes

    def get_indices(self, actions):
        return self.vq(self.encode(actions))[1]

    def decode_actions(self, indices):
        return self.decode(self.vq.indices_to_codes(indices))

    def forward_details(self, actions):
        projected = self.action_proj(actions)
        convolution = self.conv_block(projected)
        position = self.add_positional_emb(convolution)
        encoded = self.encoder(position, causal_mask(position.shape[1]) if self.config.use_causal_encoder else None)
        fsq_projected = self.vq.project_in(encoded)
        quantized = self.vq.quantize(fsq_projected)
        indices = self.vq.codes_to_indices(quantized)
        codes = self.vq.project_out(quantized)
        queries = self.fixed_positional_emb(jt.zeros((actions.shape[0], self.skill_block_size, self.decoder_dim)))
        decoded = self.decoder(queries, codes, causal_mask(queries.shape[1]) if self.config.use_causal_decoder else None)
        prediction = self.action_head(decoded)
        return {"input": actions, "action_projection": projected, "convolution": convolution,
                "encoder": encoded, "fsq_project_in": fsq_projected, "codes": codes,
                "indices": indices, "decoder": decoded, "prediction": prediction}


def strict_load_npz(model, path):
    state = model.state_dict()
    with np.load(path, allow_pickle=False) as weights:
        if set(weights.files) != set(state):
            raise ValueError({"missing": sorted(set(state) - set(weights.files)),
                              "unexpected": sorted(set(weights.files) - set(state))})
        for name, parameter in state.items():
            value = weights[name]
            if name.endswith("num_batches_tracked") and value.shape == () and list(parameter.shape) == [1]:
                # Jittor builds may represent scalars as one-element Vars.
                value = value.reshape(1)
            if list(value.shape) != list(parameter.shape):
                raise ValueError(f"Shape mismatch for {name}: {value.shape} vs {parameter.shape}")
            parameter.assign(jt.array(value))


def eval_with_parameter_gradients(model):
    """Jittor eval stops parameter gradients; Torch eval does not. Match Torch for alignment."""
    model.eval()
    for name, parameter in model.named_parameters():
        if not getattr(parameter, "repro_inactive", False) and not name.endswith(("inv_freq", "pos_x", "pos_y", "running_mean", "running_var", "num_batches_tracked")):
            parameter.start_grad()
    return model
