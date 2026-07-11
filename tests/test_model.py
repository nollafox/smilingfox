from __future__ import annotations

import dataclasses

import pytest
import torch
import torch.nn as nn

from smilingfox._bundle import load_bundle
from smilingfox._model import build_model, pick_device


class TestPickDevice:
    def test_explicit_device_string_is_respected(self) -> None:
        assert pick_device("cpu") == torch.device("cpu")

    def test_prefers_cuda_when_available(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        assert pick_device(None) == torch.device("cuda")

    def test_falls_back_to_mps_when_no_cuda(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
        assert pick_device(None) == torch.device("mps")

    def test_falls_back_to_cpu_when_nothing_else_available(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
        assert pick_device(None) == torch.device("cpu")


class TestBuildModel:
    def test_happy_path_loads_state_dict_and_moves_to_eval(self, bundle_dir) -> None:
        bundle = load_bundle(bundle_dir)
        model = build_model(bundle, torch.device("cpu"))
        assert isinstance(model, nn.Module)
        assert model.training is False
        assert model.fc.out_features == len(bundle.labels)

    def test_raises_when_model_lacks_reset_classifier(
        self, bundle_dir, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("smilingfox._model.timm.create_model", lambda *a, **k: nn.Linear(1, 1))
        bundle = load_bundle(bundle_dir)
        with pytest.raises(RuntimeError, match="reset_classifier"):
            build_model(bundle, torch.device("cpu"))

    def test_raises_when_checkpoint_missing_state_dict(self, bundle_dir) -> None:
        bundle = dataclasses.replace(load_bundle(bundle_dir), checkpoint={})
        with pytest.raises(ValueError, match="state_dict"):
            build_model(bundle, torch.device("cpu"))
