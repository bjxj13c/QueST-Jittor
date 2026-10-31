"""Torch reference-compatible QueST FSQ tokenizer.

Architecture adapted from pairlab/QueST skill_vae.py at
cc25a835a00cf2d91df69704bdd757904d13c061 (MIT, Atharva Mete).
FSQ adapted from vector-quantize-pytorch 1.14.24 (MIT, Phil Wang).
Sinusoidal encoding follows positional-encodings 6.0.3.
See THIRD_PARTY_NOTICES.md and licenses/ for original license texts.
"""
import math

import torch
from torch import nn

from .config import TokenizerConfig


def exact_gelu(value):
    """Preserve exact GELU and bypass Torch 2.2's fused eval Encoder path.

    Its CUDA fused path differed from the ordinary path enough to change one
    frozen-corpus FSQ index. A public custom activation callable selects the
    ordinary layer path without changing GELU, weights, masks or quantization.
    """
    return torch.nn.functional.gelu(value, approximate="none")


def round_ste(value):
    return value + (value.round() - value).detach()


class FSQ(nn.Module):
    def __init__(self, dim, levels):
        super().__init__()
        self.dim = dim
        self.codebook_dim = len(levels)
        self.codebook_size = math.prod(levels)
        self.register_buffer("_levels", torch.tensor(levels, dtype=torch.int32), persistent=False)
        self.register_buffer("_basis", torch.cumprod(torch.tensor([1] + list(levels[:-1])), dim=0,
                                                    dtype=torch.int32), persistent=False)
        self.project_in = nn.Linear(dim, len(levels)) if dim != len(levels) else nn.Identity()
        self.project_out = nn.Linear(len(levels), dim) if dim != len(levels) else nn.Identity()

    def bound(self, value, eps=1e-3):
        half = (self._levels - 1) * (1 + eps) / 2
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
        shift = (offset / half).atanh()
        return (value + shift).tanh() * half - offset

    # PyTorch参考量化流程：限幅、直通取整、缩放，与Jittor逐项核对。
    def quantize(self, value):
        return round_ste(self.bound(value)) / (self._levels // 2)

    def codes_to_indices(self, codes):
        digits = codes * (self._levels // 2) + self._levels // 2
        return (digits * self._basis).sum(dim=-1).to(torch.int32)

    def _indices_to_codes(self, indices):
        digits = (indices[..., None] // self._basis) % self._levels
        return (digits - self._levels // 2) / (self._levels // 2)

    def indices_to_codes(self, indices):
        return self.project_out(self._indices_to_codes(indices))

    def forward(self, value):
        original_dtype = value.dtype
        projected = self.project_in(value)
        if projected.dtype not in (torch.float32, torch.float64):
            projected = projected.float()
        # 量化后的特征用于动作重建，离散编号用于训练条件策略。
        codes = self.quantize(projected)
        indices = self.codes_to_indices(codes)
        return self.project_out(codes.to(original_dtype)), indices


class PositionalEncoding1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = math.ceil(channels / 2) * 2
        self.register_buffer("inv_freq", 1.0 / (10000 ** (
            torch.arange(0, self.channels, 2).float() / self.channels)))

    def forward(self, tensor):
        batch, length, channels = tensor.shape
        positions = torch.arange(length, device=tensor.device, dtype=self.inv_freq.dtype)
        sin_input = torch.einsum("i,j->ij", positions, self.inv_freq)
        embedding = torch.stack((sin_input.sin(), sin_input.cos()), dim=-1).flatten(-2, -1)
        embedding = embedding.to(tensor.dtype)
        return embedding[None, :, :channels].repeat(batch, 1, 1)


class Summer(nn.Module):
    def __init__(self, penc):
        super().__init__()
        self.penc = penc

    def forward(self, value):
        return value + self.penc(value)


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.padding = kernel_size - 1
        self.kernel_size, self.stride = kernel_size, stride
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=self.padding, stride=stride)

    def forward(self, value):
        value = self.conv(value)
        trim = (2 * self.padding - self.kernel_size) // self.stride + 1
        return value[:, :, :-trim] if trim > 0 else value


class ExpandTime(nn.Module):
    def forward(self, value):
        return value.unsqueeze(2)


class CollapseTime(nn.Module):
    def forward(self, value):
        return value.squeeze(2)


class Conv1dBlock(nn.Module):
    def __init__(self, dim, kernel_size, stride, causal=True):
        super().__init__()
        convolution = CausalConv1d(dim, dim, kernel_size, stride) if causal else nn.Conv1d(
            dim, dim, kernel_size, padding=kernel_size // 2, stride=stride)
        self.block = nn.Sequential(convolution, ExpandTime(), nn.GroupNorm(8, dim),
                                   CollapseTime(), nn.Mish())

    def forward(self, value):
        return self.block(value)


class ResidualTemporalBlock(nn.Module):
    """Retains upstream naming; default upstream configuration has residual=False."""
    def __init__(self, dim, factor, causal=True):
        super().__init__()
        count = int(math.log2(factor))
        strides, kernels = [2] * count + [1], [5] + [3] * count
        if count == 0:
            kernels, strides = [3, 2], [1, 1]
        self.blocks = nn.ModuleList([Conv1dBlock(dim, k, s, causal) for k, s in zip(kernels, strides)])

    def forward(self, value):
        value = value.transpose(1, 2)
        for block in self.blocks:
            value = block(value)
        return value.transpose(1, 2)


class SkillVAE(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or TokenizerConfig()
        c = self.config
        self.encoder_dim, self.decoder_dim = c.encoder_dim, c.decoder_dim
        self.skill_block_size = c.skill_block_size
        self.use_causal_encoder, self.use_causal_decoder = c.use_causal_encoder, c.use_causal_decoder
        self.vq_type, self.fsq_level = "fsq", list(c.levels)
        self.vq = FSQ(c.encoder_dim, c.levels)
        self.action_proj = nn.Linear(c.action_dim, c.encoder_dim)
        self.action_head = nn.Linear(c.decoder_dim, c.action_dim)
        self.conv_block = ResidualTemporalBlock(c.encoder_dim, c.downsample_factor, c.use_causal_encoder)
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            c.encoder_dim, c.encoder_heads, 4 * c.encoder_dim, dropout=c.attn_pdrop,
            activation=exact_gelu, batch_first=True, norm_first=True), c.encoder_layers,
            enable_nested_tensor=False)
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(
            c.decoder_dim, c.decoder_heads, 4 * c.decoder_dim, dropout=c.attn_pdrop,
            activation=exact_gelu, batch_first=True, norm_first=True), c.decoder_layers)
        self.add_positional_emb = Summer(PositionalEncoding1D(c.encoder_dim))
        self.fixed_positional_emb = PositionalEncoding1D(c.decoder_dim)

    def encode(self, actions, obs_emb=None):
        value = self.conv_block(self.action_proj(actions))
        length = value.shape[1]
        if obs_emb is not None:
            value = torch.cat((obs_emb, value), dim=1)
        value = self.add_positional_emb(value)
        if self.use_causal_encoder:
            mask = nn.Transformer.generate_square_subsequent_mask(value.shape[1], device=value.device)
            value = self.encoder(value, mask=mask, is_causal=True)
        else:
            value = self.encoder(value)
        return value[:, -length:]

    def quantize(self, value):
        codes, indices = self.vq(value)
        utilization = value.new_tensor(torch.unique(indices).numel() / self.vq.codebook_size)
        # Upstream constructs this diagnostic through NumPy mean, yielding float64.
        per_sample = value.new_tensor(sum(torch.unique(i).numel() for i in indices)
                                      / (indices.shape[0] * indices.shape[1]), dtype=torch.float64)
        return codes, indices, utilization, per_sample, value.new_zeros(1)

    def decode(self, codes, obs_emb=None):
        queries = self.fixed_positional_emb(codes.new_zeros(
            (codes.shape[0], self.skill_block_size, self.decoder_dim)))
        memory = torch.cat((obs_emb, codes), dim=1) if obs_emb is not None else codes
        if self.use_causal_decoder:
            mask = nn.Transformer.generate_square_subsequent_mask(queries.shape[1], device=queries.device)
            queries = self.decoder(queries, memory, tgt_mask=mask, tgt_is_causal=True)
        else:
            queries = self.decoder(queries, memory)
        return self.action_head(queries)

    def forward(self, actions, obs_emb=None):
        codes, _, utilization, per_sample, auxiliary = self.quantize(self.encode(actions, obs_emb))
        return self.decode(codes, obs_emb), utilization, per_sample, auxiliary, codes

    def get_indices(self, actions, obs_emb=None):
        return self.vq(self.encode(actions, obs_emb))[1]

    def decode_actions(self, indices):
        return self.decode(self.vq.indices_to_codes(indices))
