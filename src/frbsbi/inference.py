"""Checkpoint inference on observable arrays; posterior validation is separate.

Host parameters are returned as physical median and NATURAL LOG width.
The checkpoint must come from this trusted repository/run.
"""
from pathlib import Path
import torch
from .model import PosteriorFlow,ModelConfig


def load_checkpoint(path):
    payload = torch.load(Path(path),map_location="cpu",weights_only=True)
    model = PosteriorFlow(ModelConfig(**payload["model_config"]))
    model.load_state_dict(payload["model_state"],strict=True)
    model.eval()
    return model,payload
