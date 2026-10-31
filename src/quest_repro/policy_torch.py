"""Single-task QueST stage-one policy, adapted from the pinned MIT source.

No text/language generalization is claimed. Default widths/layers follow QueST;
reduced configurations are explicitly diagnostics. Augmentation is deliberately
disabled for numeric comparisons and the first small-data baseline.
"""
import torch
from torch import nn
from torch.nn import functional as F
import torchvision

from .tokenizer_torch import PositionalEncoding1D, Summer, exact_gelu
from .config import PolicyConfig


class SpatialSoftmax(nn.Module):
    def __init__(self, in_c, in_h, in_w, num_kp):
        super().__init__()
        self._spatial_conv = nn.Conv2d(in_c, num_kp, 1)
        x, y = torch.meshgrid(torch.linspace(-1, 1, in_w),
                              torch.linspace(-1, 1, in_h), indexing="ij")
        # Preserve official coordinate ordering (even though axis names are unusual).
        self.register_buffer("pos_x", x.reshape(1, -1))
        self.register_buffer("pos_y", y.reshape(1, -1))
        self._in_c, self._in_h, self._in_w, self._num_kp = in_c, in_h, in_w, num_kp

    def forward(self, value):
        if tuple(value.shape[1:]) != (self._in_c, self._in_h, self._in_w):
            raise ValueError("Image feature size differs from spatial projection configuration")
        if self._num_kp != self._in_c:
            value = self._spatial_conv(value)
        probabilities = value.reshape(-1, self._in_h * self._in_w).softmax(-1)
        x = (probabilities * self.pos_x).sum(1, keepdim=True).view(-1, self._num_kp)
        y = (probabilities * self.pos_y).sum(1, keepdim=True).view(-1, self._num_kp)
        return torch.cat((x, y), dim=1)


class SpatialProjection(nn.Module):
    def __init__(self, shape, out_dim):
        super().__init__()
        self.spatial_softmax = SpatialSoftmax(*shape, num_kp=out_dim // 2)
        self.projection = nn.Linear(out_dim, out_dim)

    def forward(self, value):
        return self.projection(self.spatial_softmax(value))


class ResnetEncoder(nn.Module):
    """Official partial ResNet18: stem, layer1/layer2, spatial softmax.

    No pretrained weights, ImageNet normalization, layer3/layer4 or global pool.
    """
    def __init__(self, image_size=128, output_size=256):
        super().__init__()
        layers = list(torchvision.models.resnet18(weights=None).children())[:-4]
        self.resnet18_base = nn.Sequential(*layers[:4])
        self.block_1, self.block_2 = layers[4][0], layers[4][1]
        self.block_3, self.block_4 = layers[5][0], layers[5][1]
        self.normalizer = nn.Identity()
        # Match the official constructor's dummy forward, including BN buffers.
        with torch.no_grad():
            value = self.features(torch.zeros(1, 3, image_size, image_size))
        self.projection_layer = SpatialProjection(value.shape[1:], output_size)
        self.out_channels = output_size

    def features(self, value):
        return self.block_4(self.block_3(self.block_2(self.block_1(self.resnet18_base(value)))))

    def forward(self, value):
        return self.projection_layer(self.features(self.normalizer(value)))


class MLPProj(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(input_size, output_size))

    def forward(self, value):
        return self.projection(value)


class SkillGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config, self.start_token, self.block_size = config, config.vocab_size, config.block_size
        self.tok_emb = nn.Embedding(config.vocab_size + 1, config.embed_dim)
        self.add_positional_emb = Summer(PositionalEncoding1D(config.embed_dim))
        self.decoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            config.embed_dim, config.heads, 4 * config.embed_dim, config.dropout,
            activation=exact_gelu, batch_first=True, norm_first=True),
            num_layers=config.layers, enable_nested_tensor=False)
        self.head = nn.Linear(config.embed_dim, config.vocab_size)
        self.drop = nn.Dropout(config.dropout)
        self.lnf = nn.LayerNorm(config.embed_dim)

    def forward(self, indices, context):
        value = self.add_positional_emb(self.tok_emb(indices))
        value = self.drop(torch.cat((context, value), dim=1))
        mask = nn.Transformer.generate_square_subsequent_mask(value.size(1), device=value.device)
        value = self.decoder(value, mask=mask, is_causal=True)
        return self.head(self.lnf(value[:, context.shape[1]:]))

    @torch.no_grad()
    def generate(self, context, top_k=1, temperature=1.0, generator=None):
        if not 1 <= top_k <= self.config.vocab_size or temperature <= 0:
            raise ValueError("Invalid top-k or temperature")
        indices = torch.full((len(context), 1), self.start_token, dtype=torch.long, device=context.device)
        for _ in range(self.block_size):
            logits = self(indices, context)[:, -1] / temperature
            values, labels = logits.topk(top_k, dim=-1)
            sampled = torch.multinomial(values.softmax(-1), 1, generator=generator)
            indices = torch.cat((indices, labels.gather(-1, sampled)), dim=1)
        return indices[:, 1:]

    @torch.no_grad()
    def generate_sampled(self,context,top_k=5,temperature=1.0,rng=None):
        import numpy as np
        from .sampling import sample_top_k
        if rng is None:raise ValueError('Provide explicit NumPy RNG')
        indices=torch.full((len(context),1),self.start_token,dtype=torch.long,device=context.device)
        for _ in range(self.block_size):
            logits=self(indices,context)[:,-1].cpu().numpy()
            next_id=sample_top_k(logits,top_k,rng,temperature)
            indices=torch.cat((indices,torch.from_numpy(next_id[:,None].astype(np.int64)).to(context.device)),dim=1)
        return indices[:,1:]


class QueSTPolicy(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = c = config or PolicyConfig()
        self.image_encoders = nn.ModuleDict({"corner_rgb": ResnetEncoder(c.image_size, c.image_embed_dim)})
        self.lowdim_encoders = nn.ModuleDict({"robot_states": MLPProj(8, c.lowdim_embed_dim)})
        self.obs_proj = nn.Linear(c.image_embed_dim + c.lowdim_embed_dim, c.embed_dim)
        self.task_encoder = nn.Embedding(c.tasks, c.embed_dim)
        self.policy_prior = SkillGPT(c)

    def get_context(self, images, robot_states, task_id):
        if images.dtype != torch.uint8:
            raise TypeError("Policy requires raw uint8 RGB: normalize exactly once inside the policy")
        if robot_states.shape[-1] != 8:
            raise ValueError("Policy receives 8-D robot state; privileged 39-D state is forbidden")
        batch, time = images.shape[:2]
        visual = self.image_encoders["corner_rgb"](
            images.float().div(255).reshape(batch * time, *images.shape[2:])).reshape(batch, time, -1)
        state = self.lowdim_encoders["robot_states"](robot_states.float())
        observation = self.obs_proj(torch.cat((visual, state), dim=-1))
        return torch.cat((self.task_encoder(task_id).unsqueeze(1), observation), dim=1)

    def forward(self, images, robot_states, task_id, targets):
        context = self.get_context(images, robot_states, task_id)
        start = torch.full((len(targets), 1), self.config.vocab_size, dtype=torch.long, device=targets.device)
        inputs = torch.cat((start, targets[:, :-1]), dim=1)
        logits = self.policy_prior(inputs, context)
        return logits, F.cross_entropy(logits.reshape(-1, self.config.vocab_size), targets.reshape(-1))

    @torch.no_grad()
    def predict_actions(self, images, robot_states, task_id, tokenizer, top_k=1, temperature=1.0):
        self.eval()
        tokenizer.eval()
        context = self.get_context(images, robot_states, task_id)
        indices = self.policy_prior.generate(context, top_k, temperature)
        return tokenizer.decode_actions(indices), indices
