from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from PIL import Image

LABELS = ["a", "b", "c"]
THRESHOLDS = [0.5, 0.5, 0.5]
# sigmoid(LOGITS) == (0.9, 0.6, 0.1)
LOGITS = [math.log(0.9 / 0.1), math.log(0.6 / 0.4), math.log(0.1 / 0.9)]
INPUT_SIZE = (3, 8, 8)


class DummyBackbone(nn.Module):
    """Stand-in for a timm model: flatten -> linear, with a resettable classifier head.

    With its weights zeroed and a fixed bias, its output is independent of the
    input image, which lets tests assert on exact tag/score values without
    depending on real model weights or network access.
    """

    def __init__(self, in_features: int = math.prod(INPUT_SIZE), num_classes: int = len(LABELS)) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.in_features = in_features
        self.fc = nn.Linear(in_features, num_classes)

    def reset_classifier(self, num_classes: int) -> None:
        self.fc = nn.Linear(self.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.flatten(x))


def fixed_logit_state_dict() -> dict[str, torch.Tensor]:
    model = DummyBackbone()
    with torch.no_grad():
        model.fc.weight.zero_()
        model.fc.bias.copy_(torch.tensor(LOGITS, dtype=torch.float32))
    return model.state_dict()


@pytest.fixture
def bundle_dir(tmp_path: Path) -> Path:
    """A minimal, valid model bundle directory with a tiny deterministic checkpoint."""
    (tmp_path / "labels.json").write_text(json.dumps(LABELS))
    (tmp_path / "best_thresholds.json").write_text(json.dumps({"labels": LABELS, "thresholds": THRESHOLDS}))
    (tmp_path / "args.json").write_text(json.dumps({"model_repo": "test/dummy"}))
    torch.save(
        {
            "state_dict": fixed_logit_state_dict(),
            "data_config": {
                "input_size": INPUT_SIZE,
                "mean": (0.0, 0.0, 0.0),
                "std": (1.0, 1.0, 1.0),
                "crop_pct": 1.0,
                "interpolation": "bilinear",
            },
        },
        tmp_path / "best.pt",
    )
    return tmp_path


@pytest.fixture(autouse=True)
def patch_timm_create_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test gets a tiny DummyBackbone instead of downloading a real timm/HF model."""
    monkeypatch.setattr("smilingfox._model.timm.create_model", lambda *a, **k: DummyBackbone())


@pytest.fixture
def sample_image_path(tmp_path: Path) -> Path:
    path = tmp_path / "sample.jpg"
    Image.new("RGB", (16, 16), color=(120, 90, 200)).save(path)
    return path
