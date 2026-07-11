"""The public SmilingFox tagging API."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Union

import torch
from PIL import Image
from timm.data import create_transform, resolve_model_data_config

from ._bundle import ModelBundle, load_bundle
from ._model import build_model, pick_device

ImageInput = Union[str, Path, Image.Image]


@dataclass(frozen=True)
class Prediction:
    """The tags SmilingFox selected for one image, with their confidence scores."""

    tags: tuple[str, ...]
    scores: dict[str, float]

    def __iter__(self):
        return iter(self.tags)

    def __len__(self) -> int:
        return len(self.tags)

    def __contains__(self, tag: object) -> bool:
        return tag in self.tags


class Tagger:
    """A loaded SmilingFox model, ready to tag images."""

    def __init__(self, bundle: ModelBundle, device: torch.device) -> None:
        self.bundle = bundle
        self.device = device
        self.model = build_model(bundle, device)
        data_config = dict(bundle.data_config) if bundle.data_config else resolve_model_data_config(self.model)
        self.transform = create_transform(**data_config, is_training=False)

    @classmethod
    def from_pretrained(cls, model: str | Path | None = None, *, device: str | None = None) -> "Tagger":
        """Load a tagger from a local bundle directory or a Hugging Face repo id.

        Defaults to downloading (and caching) the published `nollafox/smilingfox` repo.
        """
        return cls(load_bundle(model), pick_device(device))

    @property
    def labels(self) -> tuple[str, ...]:
        return self.bundle.labels

    def predict(self, image: ImageInput, *, required_confidence: float = 0.0) -> Prediction:
        """Tag a single image."""
        return self.predict_batch([image], required_confidence=required_confidence)[0]

    @torch.no_grad()
    def predict_batch(
        self,
        images: Sequence[ImageInput],
        *,
        required_confidence: float = 0.0,
        batch_size: int = 8,
    ) -> list[Prediction]:
        """Tag a batch of images, at most `batch_size` at a time."""
        if not 0.0 <= required_confidence <= 1.0:
            raise ValueError("required_confidence must be between 0.0 and 1.0")

        thresholds = torch.maximum(
            self.bundle.thresholds, torch.full_like(self.bundle.thresholds, float(required_confidence))
        )

        predictions: list[Prediction] = []
        for start in range(0, len(images), batch_size):
            batch = images[start : start + batch_size]
            tensor = torch.stack([self._load_image(image) for image in batch]).to(self.device)

            if self.device.type == "cuda":
                with torch.autocast("cuda"):
                    logits = self.model(tensor)
            else:
                logits = self.model(tensor)
            probabilities = torch.sigmoid(logits.float()).cpu()

            predictions.extend(self._decode(probabilities[i], thresholds) for i in range(len(batch)))
        return predictions

    def _load_image(self, image: ImageInput) -> torch.Tensor:
        if isinstance(image, Image.Image):
            return self.transform(image.convert("RGB"))
        with Image.open(image) as handle:
            return self.transform(handle.convert("RGB"))

    def _decode(self, probabilities: torch.Tensor, thresholds: torch.Tensor) -> Prediction:
        indices = torch.nonzero(probabilities >= thresholds, as_tuple=False).flatten().tolist()
        indices.sort(key=lambda i: float(probabilities[i]), reverse=True)
        tags = tuple(self.bundle.labels[i] for i in indices)
        scores = {self.bundle.labels[i]: round(float(probabilities[i]), 6) for i in indices}
        return Prediction(tags=tags, scores=scores)


_default_tagger: Tagger | None = None


def tag(image: ImageInput, *, required_confidence: float = 0.0) -> Prediction:
    """Tag a single image using a lazily-loaded, cached default Tagger.

    Convenient for one-off scripts; for tagging many images, construct a
    `Tagger` once with `Tagger.from_pretrained()` and reuse it.
    """
    global _default_tagger
    if _default_tagger is None:
        _default_tagger = Tagger.from_pretrained()
    return _default_tagger.predict(image, required_confidence=required_confidence)
