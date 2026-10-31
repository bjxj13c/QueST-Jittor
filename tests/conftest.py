from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(4)
