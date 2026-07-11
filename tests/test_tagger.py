from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from smilingfox import Prediction, Tagger

from conftest import LABELS


class TestPrediction:
    def test_iter_yields_tags(self) -> None:
        prediction = Prediction(tags=("a", "b"), scores={"a": 0.9, "b": 0.6})
        assert list(prediction) == ["a", "b"]

    def test_len_matches_tag_count(self) -> None:
        assert len(Prediction(tags=("a", "b"), scores={})) == 2

    def test_contains_checks_tags(self) -> None:
        prediction = Prediction(tags=("a",), scores={"a": 0.9})
        assert "a" in prediction
        assert "z" not in prediction


@pytest.fixture
def tagger(bundle_dir: Path) -> Tagger:
    return Tagger.from_pretrained(bundle_dir, device="cpu")


class TestTaggerPredict:
    def test_predict_from_path(self, tagger: Tagger, sample_image_path: Path) -> None:
        prediction = tagger.predict(sample_image_path)
        assert prediction.tags == ("a", "b")
        assert prediction.scores == {"a": 0.9, "b": 0.6}

    def test_predict_from_str_path(self, tagger: Tagger, sample_image_path: Path) -> None:
        prediction = tagger.predict(str(sample_image_path))
        assert prediction.tags == ("a", "b")

    def test_predict_from_pil_image(self, tagger: Tagger) -> None:
        image = Image.new("RGB", (16, 16), color=(1, 2, 3))
        prediction = tagger.predict(image)
        assert prediction.tags == ("a", "b")

    def test_required_confidence_raises_effective_threshold(self, tagger: Tagger, sample_image_path: Path) -> None:
        prediction = tagger.predict(sample_image_path, required_confidence=0.7)
        assert prediction.tags == ("a",)

    def test_predictions_are_sorted_by_descending_score(self, tagger: Tagger, sample_image_path: Path) -> None:
        prediction = tagger.predict(sample_image_path, required_confidence=0.0)
        scores = [prediction.scores[tag] for tag in prediction.tags]
        assert scores == sorted(scores, reverse=True)

    def test_out_of_range_required_confidence_raises(self, tagger: Tagger, sample_image_path: Path) -> None:
        with pytest.raises(ValueError, match="required_confidence"):
            tagger.predict(sample_image_path, required_confidence=1.5)

    def test_labels_property_exposes_bundle_labels(self, tagger: Tagger) -> None:
        assert tagger.labels == tuple(LABELS)


class TestTaggerPredictBatch:
    def test_batch_matches_single_predictions_and_preserves_order(
        self, tagger: Tagger, tmp_path: Path
    ) -> None:
        paths = []
        for index, color in enumerate([(1, 2, 3), (4, 5, 6), (7, 8, 9)]):
            path = tmp_path / f"img_{index}.jpg"
            Image.new("RGB", (16, 16), color=color).save(path)
            paths.append(path)

        results = tagger.predict_batch(paths, batch_size=2)

        assert len(results) == 3
        assert all(result.tags == ("a", "b") for result in results)

    def test_empty_batch_returns_empty_list(self, tagger: Tagger) -> None:
        assert tagger.predict_batch([]) == []
