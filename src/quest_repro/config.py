from dataclasses import asdict, dataclass


@dataclass
class TokenizerConfig:
    action_dim: int = 4
    encoder_dim: int = 256
    decoder_dim: int = 256
    skill_block_size: int = 32
    downsample_factor: int = 4
    attn_pdrop: float = 0.1
    use_causal_encoder: bool = True
    use_causal_decoder: bool = True
    encoder_heads: int = 4
    encoder_layers: int = 2
    decoder_heads: int = 4
    decoder_layers: int = 4
    levels: tuple = (8, 5, 5, 5)

    def __post_init__(self):
        self.levels = tuple(self.levels)
        if self.encoder_dim != self.decoder_dim:
            raise ValueError("This reference architecture requires equal encoder/decoder dimensions")
        if self.downsample_factor <= 0 or self.downsample_factor & (self.downsample_factor - 1):
            raise ValueError("downsample_factor must be a positive power of two")
        if self.skill_block_size % self.downsample_factor:
            raise ValueError("chunk length must be divisible by downsample factor")
        if self.encoder_dim % 8 or self.encoder_dim % self.encoder_heads or self.decoder_dim % self.decoder_heads:
            raise ValueError("dimensions must be divisible by GroupNorm groups and attention heads")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def tiny(cls, dropout=0.0):
        """Reduced CPU diagnostic architecture, never presented as the paper configuration."""
        return cls(encoder_dim=64, decoder_dim=64, encoder_layers=1,
                   decoder_layers=1, attn_pdrop=dropout)


@dataclass
class PolicyConfig:
    embed_dim: int = 384
    image_embed_dim: int = 256
    lowdim_embed_dim: int = 128
    layers: int = 6
    heads: int = 6
    dropout: float = 0.1
    vocab_size: int = 1000
    block_size: int = 8
    tasks: int = 1
    image_size: int = 128

    def __post_init__(self):
        if self.embed_dim % self.heads or self.image_embed_dim % 2:
            raise ValueError("Invalid attention or spatial-keypoint dimensions")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def tiny(cls, dropout=0.0):
        return cls(embed_dim=64, image_embed_dim=64, lowdim_embed_dim=32,
                   layers=1, heads=4, dropout=dropout)
