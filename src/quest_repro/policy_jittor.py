"""Native Jittor single-task QueST policy port; runtime validation pending.

No Torch imports. A Torch fixture provides only static NPZ weights/reference
arrays. See reports for measured status before describing numerical alignment.
"""
import jittor as jt
from jittor import nn
import numpy as np

from .config import PolicyConfig
from .tokenizer_jittor import Encoder, LayerNorm, PositionalEncoding1D, Summer, causal_mask


class NamedModules(jt.Module):
    def __init__(self, modules):
        for name, module in modules.items():
            setattr(self, name, module)

    def __getitem__(self, key):
        return getattr(self, key)


class BatchNorm2d(jt.Module):
    """Torch BN convention: biased forward variance, unbiased running variance.

    Jittor's default BN updates running_var with the biased estimate. Preserve
    the official Torch convention explicitly for cross-framework checkpoints.
    """
    def __init__(self, channels, momentum=0.1, eps=1e-5):
        self.weight, self.bias = jt.ones(channels), jt.zeros(channels)
        self.running_mean, self.running_var = jt.zeros(channels).stop_grad(), jt.ones(channels).stop_grad()
        self.num_batches_tracked = jt.array(np.asarray(0, dtype=np.int64)).stop_grad()
        self.momentum, self.eps = momentum, eps

    def execute(self, value):
        shape = (1, value.shape[1], 1, 1)
        if self.is_training():
            count = value.shape[0] * value.shape[2] * value.shape[3]
            if count <= 1:
                raise ValueError("BatchNorm training requires more than one value per channel")
            mean = value.mean(dims=[0, 2, 3])
            variance = ((value - mean.reshape(shape)) ** 2).mean(dims=[0, 2, 3])
            self.running_mean.update(self.running_mean * (1 - self.momentum) + mean.detach() * self.momentum)
            self.running_var.update(self.running_var * (1 - self.momentum) +
                                    variance.detach() * (count / (count - 1)) * self.momentum)
            self.num_batches_tracked.update(self.num_batches_tracked + 1)
        else:
            mean, variance = self.running_mean, self.running_var
        normalized = (value - mean.reshape(shape)) / (variance.reshape(shape) + self.eps).sqrt()
        return normalized * self.weight.reshape(shape) + self.bias.reshape(shape)


class BasicBlock(jt.Module):
    def __init__(self, input_channels, channels, stride=1):
        self.conv1 = nn.Conv2d(input_channels, channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = BatchNorm2d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = BatchNorm2d(channels)
        self.downsample = (nn.Sequential(nn.Conv2d(input_channels, channels, 1, stride=stride, bias=False),
                                        BatchNorm2d(channels))
                           if stride != 1 or input_channels != channels else None)

    def execute(self, value):
        residual = self.downsample(value) if self.downsample is not None else value
        return self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(value))))) + residual)


class SpatialSoftmax(jt.Module):
    def __init__(self, channels, height, width, keypoints):
        self._spatial_conv = nn.Conv2d(channels, keypoints, 1)
        x, y = np.meshgrid(np.linspace(-1, 1, width, dtype=np.float32),
                           np.linspace(-1, 1, height, dtype=np.float32), indexing="ij")
        self.pos_x = jt.array(x.reshape(1, -1)).stop_grad()
        self.pos_y = jt.array(y.reshape(1, -1)).stop_grad()
        self.channels, self.height, self.width, self.keypoints = channels, height, width, keypoints
        if channels == keypoints:
            # The official forward bypasses this registered convolution.
            # Torch gives these parameters grad=None and AdamW skips them.
            for parameter in self._spatial_conv.parameters():
                parameter.repro_inactive = True
                parameter.stop_grad()

    def execute(self, value):
        if tuple(value.shape[1:]) != (self.channels, self.height, self.width):
            raise ValueError("Image feature shape mismatch")
        if self.channels != self.keypoints:
            value = self._spatial_conv(value)
        probability = nn.softmax(value.reshape((-1, self.height * self.width)), dim=-1)
        x = (probability * self.pos_x).sum(dim=1).reshape((-1, self.keypoints))
        y = (probability * self.pos_y).sum(dim=1).reshape((-1, self.keypoints))
        return jt.concat((x, y), dim=1)


class SpatialProjection(jt.Module):
    def __init__(self, shape, output_dim):
        self.spatial_softmax = SpatialSoftmax(*shape, output_dim // 2)
        self.projection = nn.Linear(output_dim, output_dim)

    def execute(self, value):
        return self.projection(self.spatial_softmax(value))


class ResnetEncoder(jt.Module):
    def __init__(self, image_size=128, output_dim=256):
        if image_size != 128:
            raise ValueError("This port currently supports the pinned 128x128 observation contract")
        self.resnet18_base = nn.Sequential(nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(3, stride=2, padding=1))
        self.block_1, self.block_2 = BasicBlock(64, 64), BasicBlock(64, 64)
        self.block_3, self.block_4 = BasicBlock(64, 128, 2), BasicBlock(128, 128)
        self.projection_layer = SpatialProjection((128, 16, 16), output_dim)

    def execute(self, value):
        value = self.block_4(self.block_3(self.block_2(self.block_1(self.resnet18_base(value)))))
        return self.projection_layer(value)


class MLPProj(jt.Module):
    def __init__(self, input_size, output_size):
        self.projection = nn.Sequential(nn.Linear(input_size, output_size))

    def execute(self, value):
        return self.projection(value)


class SkillGPT(jt.Module):
    def __init__(self, config):
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size + 1, config.embed_dim)
        self.add_positional_emb = Summer(PositionalEncoding1D(config.embed_dim))
        self.decoder = Encoder(config.embed_dim, config.heads, config.layers, config.dropout)
        self.head, self.lnf = nn.Linear(config.embed_dim, config.vocab_size), LayerNorm(config.embed_dim)
        self.drop = nn.Dropout(config.dropout)

    def execute(self, indices, context):
        value = self.add_positional_emb(self.tok_emb(indices))
        value = self.drop(jt.concat((context, value), dim=1))
        value = self.decoder(value, causal_mask(value.shape[1]))
        return self.head(self.lnf(value[:, context.shape[1]:]))

    def generate_greedy(self, context):
        indices = jt.full((len(context), 1), self.config.vocab_size, dtype="int32")
        for _ in range(self.config.block_size):
            # Jittor argmax returns (indices, values), unlike Torch.
            next_token, _ = self(indices, context)[:, -1].argmax(dim=-1)
            indices = jt.concat((indices, next_token.unsqueeze(-1)), dim=1)
        return indices[:, 1:]

    def generate_sampled(self,context,top_k=5,temperature=1.0,rng=None):
        from .sampling import sample_top_k
        if top_k==1:return self.generate_greedy(context)
        if rng is None:raise ValueError('Provide an explicit NumPy RNG for repeatable sampling')
        indices=jt.full((len(context),1),self.config.vocab_size,dtype='int32')
        for _ in range(self.config.block_size):
            logits=self(indices,context)[:,-1].numpy()
            token=sample_top_k(logits,top_k,rng,temperature)
            indices=jt.concat((indices,jt.array(token).unsqueeze(-1)),dim=1)
        return indices[:,1:]


class QueSTPolicy(jt.Module):
    def __init__(self, config=None):
        self.config = c = config or PolicyConfig()
        self.image_encoders = NamedModules({"corner_rgb": ResnetEncoder(c.image_size, c.image_embed_dim)})
        self.lowdim_encoders = NamedModules({"robot_states": MLPProj(8, c.lowdim_embed_dim)})
        self.obs_proj = nn.Linear(c.image_embed_dim + c.lowdim_embed_dim, c.embed_dim)
        self.task_encoder = nn.Embedding(c.tasks, c.embed_dim)
        self.policy_prior = SkillGPT(c)

    def get_context(self, images, robot_states, task_id):
        if str(images.dtype) != "uint8" or robot_states.shape[-1] != 8:
            raise ValueError("Expected raw uint8 RGB and 8-D robot state")
        batch, time = images.shape[:2]
        visual = self.image_encoders["corner_rgb"]((images.float32() / 255).reshape((-1, *images.shape[2:])))
        state = self.lowdim_encoders["robot_states"](robot_states.float32())
        observation = self.obs_proj(jt.concat((visual.reshape((batch, time, -1)), state), dim=-1))
        return jt.concat((self.task_encoder(task_id).unsqueeze(1), observation), dim=1)

    def execute(self, images, robot_states, task_id, targets):
        context = self.get_context(images, robot_states, task_id)
        start = jt.full((len(targets), 1), self.config.vocab_size, dtype="int32")
        indices = jt.concat((start, targets[:, :-1]), dim=1)
        logits = self.policy_prior(indices, context)
        loss = nn.cross_entropy_loss(logits.reshape((-1, self.config.vocab_size)), targets.reshape((-1,)))
        return logits, loss
