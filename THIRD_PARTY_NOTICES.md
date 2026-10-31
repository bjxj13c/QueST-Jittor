# Third-party sources

This reproduction adapts architecture and numerical definitions from the following sources. Original license texts are retained in `licenses/`. Reference checkouts and wheel audit copies are stored separately in ignored directories.

1. **QueST** — Atharva Mete, MIT license. Repository: https://github.com/pairlab/QueST. Pinned revision: `cc25a835a00cf2d91df69704bdd757904d13c061`. Adapted references: `quest/algos/quest_modules/skill_vae.py`, `skill_gpt.py`, `quest/algos/utils/rgb_modules.py`, `mlp_proj.py`, `quest/algos/base.py` and their configuration/observation contracts. Independent reimplementations and source comparison tests are identified in reports.
2. **vector-quantize-pytorch 1.14.24** — Phil Wang, MIT license. Repository: https://github.com/lucidrains/vector-quantize-pytorch. `finite_scalar_quantization.py` supplies FSQ bounding, straight-through rounding, mixed-radix indexing and projection definitions. The PyPI wheel digest and URL are recorded under `provenance/dependencies/`.
3. **positional-encodings 6.0.3** — original author/license from the retained distribution. Repository: https://github.com/tatp22/multidim-positional-encoding. The sinusoidal frequency definition and interleaved sine/cosine order follow `torch_encodings.py`.
4. **torchvision 0.17.0** — BSD-3-Clause. Repository: https://github.com/pytorch/vision. The Torch visual encoder uses its ResNet18; the native Jittor port follows the same BasicBlock/stem structure. Original license retained in `licenses/torchvision-BSD.txt`.

FSQ paper: *Finite Scalar Quantization: VQ-VAE Made Simple*, https://arxiv.org/abs/2309.15505. This is a supporting quantization reference, not the selected reproduction paper.

No author endorsement or full benchmark equivalence is implied by an implementation-level numerical comparison.
