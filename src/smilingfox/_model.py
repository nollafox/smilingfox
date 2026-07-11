"""Device selection and model construction."""

from __future__ import annotations

from typing import Mapping

import timm
import torch
import torch.nn as nn

from ._bundle import ModelBundle


def pick_device(device: str | torch.device | None = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(bundle: ModelBundle, device: torch.device) -> nn.Module:
    model = timm.create_model(f"hf_hub:{bundle.model_repo}", pretrained=False)
    if not hasattr(model, "reset_classifier"):
        raise RuntimeError(f"timm model '{bundle.model_repo}' does not support reset_classifier().")
    model.reset_classifier(num_classes=len(bundle.labels))

    state_dict = bundle.checkpoint.get("state_dict")
    if not isinstance(state_dict, Mapping):
        raise ValueError("Checkpoint does not contain a state_dict mapping.")
    model.load_state_dict(state_dict, strict=True)

    model.to(device)
    model.eval()
    return model
