"""Locating and loading a SmilingFox model bundle."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

DEFAULT_HF_MODEL_ID = "nollafox/smilingfox"
DEFAULT_THRESHOLD = 0.35
_CHECKPOINT_NAME = "best.pt"
_REQUIRED_FILES = (_CHECKPOINT_NAME, "best_thresholds.json", "labels.json", "args.json")


@dataclass(frozen=True)
class ModelBundle:
    model_dir: Path
    checkpoint: Mapping[str, Any]
    labels: tuple[str, ...]
    thresholds: torch.Tensor
    model_repo: str
    data_config: Mapping[str, Any]


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _is_bundle_dir(path: Path) -> bool:
    return path.is_dir() and all((path / name).is_file() for name in _REQUIRED_FILES)


def _download_bundle(repo_id: str) -> Path:
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            allow_patterns=["*.pt", "*.json", "config.json"],
        )
    )


def resolve_model_dir(source: str | Path | None) -> Path:
    """Resolve *source* to a local directory containing a full model bundle.

    *source* may be a local directory, a Hugging Face repo id ("org/name"),
    or None, in which case the ``SMILINGFOX_MODEL`` env var is used if set,
    otherwise the published `nollafox/smilingfox` repo is downloaded (and
    cached) via huggingface_hub.
    """
    source = source or os.environ.get("SMILINGFOX_MODEL") or DEFAULT_HF_MODEL_ID
    path = Path(source).expanduser()

    if path.exists():
        if not _is_bundle_dir(path):
            raise FileNotFoundError(
                f"'{path}' does not contain a full model bundle (expected {', '.join(_REQUIRED_FILES)})."
            )
        return path.resolve()

    return _download_bundle(str(source)).resolve()


def _load_thresholds(model_dir: Path, labels: Sequence[str], default_threshold: float) -> torch.Tensor:
    threshold_path = model_dir / "best_thresholds.json"
    fallback = torch.full((len(labels),), float(default_threshold), dtype=torch.float32)
    if not threshold_path.is_file():
        return fallback

    payload = _read_json(threshold_path)
    raw_thresholds = payload.get("thresholds")
    threshold_labels = payload.get("labels")
    if not isinstance(raw_thresholds, list):
        return fallback

    if threshold_labels == list(labels):
        return torch.tensor([float(value) for value in raw_thresholds], dtype=torch.float32)

    if isinstance(threshold_labels, list):
        by_label = {str(label): float(value) for label, value in zip(threshold_labels, raw_thresholds)}
        return torch.tensor([by_label.get(label, default_threshold) for label in labels], dtype=torch.float32)

    return fallback


def load_bundle(source: str | Path | None = None, *, default_threshold: float = DEFAULT_THRESHOLD) -> ModelBundle:
    model_dir = resolve_model_dir(source)

    checkpoint = torch.load(model_dir / _CHECKPOINT_NAME, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"Checkpoint {model_dir / _CHECKPOINT_NAME} did not contain a mapping.")

    labels = tuple(str(item) for item in _read_json(model_dir / "labels.json"))
    thresholds = _load_thresholds(model_dir, labels, default_threshold)

    args_payload = _read_json(model_dir / "args.json")
    model_repo = args_payload.get("model_repo") if isinstance(args_payload, Mapping) else None
    if not model_repo:
        raise ValueError(
            f"{model_dir / 'args.json'} does not specify 'model_repo'; "
            "cannot determine which base architecture to build without guessing."
        )
    model_repo = str(model_repo)

    data_config = checkpoint.get("data_config")
    if not isinstance(data_config, Mapping):
        data_config = {}

    return ModelBundle(
        model_dir=model_dir,
        checkpoint=checkpoint,
        labels=labels,
        thresholds=thresholds,
        model_repo=model_repo,
        data_config=dict(data_config),
    )
