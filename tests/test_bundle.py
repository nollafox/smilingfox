from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from smilingfox import _bundle
from smilingfox._bundle import (
    DEFAULT_HF_MODEL_ID,
    _is_bundle_dir,
    _load_thresholds,
    load_bundle,
    resolve_model_dir,
)

from conftest import LABELS, THRESHOLDS


class TestIsBundleDir:
    def test_true_for_complete_bundle(self, bundle_dir: Path) -> None:
        assert _is_bundle_dir(bundle_dir) is True

    def test_false_when_a_required_file_is_missing(self, bundle_dir: Path) -> None:
        (bundle_dir / "labels.json").unlink()
        assert _is_bundle_dir(bundle_dir) is False

    def test_false_for_nonexistent_path(self, tmp_path: Path) -> None:
        assert _is_bundle_dir(tmp_path / "does-not-exist") is False


class TestResolveModelDir:
    def test_returns_resolved_local_bundle(self, bundle_dir: Path) -> None:
        assert resolve_model_dir(str(bundle_dir)) == bundle_dir.resolve()

    def test_expands_user_in_local_path(self, bundle_dir: Path) -> None:
        assert resolve_model_dir(bundle_dir) == bundle_dir.resolve()

    def test_raises_for_incomplete_local_dir(self, tmp_path: Path) -> None:
        incomplete = tmp_path / "incomplete"
        incomplete.mkdir()
        (incomplete / "labels.json").write_text("[]")
        with pytest.raises(FileNotFoundError, match="does not contain a full model bundle"):
            resolve_model_dir(incomplete)

    def test_downloads_default_repo_when_source_is_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: dict[str, str] = {}

        def fake_download(repo_id: str) -> Path:
            seen["repo_id"] = repo_id
            return tmp_path

        monkeypatch.delenv("SMILINGFOX_MODEL", raising=False)
        monkeypatch.setattr(_bundle, "_download_bundle", fake_download)

        assert resolve_model_dir(None) == tmp_path.resolve()
        assert seen["repo_id"] == DEFAULT_HF_MODEL_ID

    def test_downloads_explicit_repo_id_when_not_a_local_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: dict[str, str] = {}

        def fake_download(repo_id: str) -> Path:
            seen["repo_id"] = repo_id
            return tmp_path

        monkeypatch.setattr(_bundle, "_download_bundle", fake_download)

        resolve_model_dir("someorg/some-model")
        assert seen["repo_id"] == "someorg/some-model"

    def test_env_var_used_when_source_is_none(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SMILINGFOX_MODEL", "org/env-model")
        seen: dict[str, str] = {}

        def fake_download(repo_id: str) -> Path:
            seen["repo_id"] = repo_id
            return tmp_path

        monkeypatch.setattr(_bundle, "_download_bundle", fake_download)

        resolve_model_dir(None)
        assert seen["repo_id"] == "org/env-model"


class TestLoadThresholds:
    def test_missing_file_returns_default_for_every_label(self, tmp_path: Path) -> None:
        thresholds = _load_thresholds(tmp_path, LABELS, default_threshold=0.42)
        assert torch.allclose(thresholds, torch.full((len(LABELS),), 0.42))

    def test_matching_label_order_uses_thresholds_directly(self, bundle_dir: Path) -> None:
        thresholds = _load_thresholds(bundle_dir, LABELS, default_threshold=0.35)
        assert thresholds.tolist() == pytest.approx(THRESHOLDS)

    def test_mismatched_label_order_is_remapped_by_label(self, tmp_path: Path) -> None:
        (tmp_path / "best_thresholds.json").write_text(
            json.dumps({"labels": ["c", "a", "b"], "thresholds": [0.9, 0.1, 0.2]})
        )
        thresholds = _load_thresholds(tmp_path, ["a", "b", "c"], default_threshold=0.35)
        assert thresholds.tolist() == pytest.approx([0.1, 0.2, 0.9])

    def test_label_missing_from_threshold_file_falls_back_to_default(self, tmp_path: Path) -> None:
        (tmp_path / "best_thresholds.json").write_text(json.dumps({"labels": ["a"], "thresholds": [0.1]}))
        thresholds = _load_thresholds(tmp_path, ["a", "b"], default_threshold=0.35)
        assert thresholds.tolist() == pytest.approx([0.1, 0.35])

    def test_non_list_thresholds_falls_back_to_default(self, tmp_path: Path) -> None:
        (tmp_path / "best_thresholds.json").write_text(json.dumps({"labels": LABELS, "thresholds": "nope"}))
        thresholds = _load_thresholds(tmp_path, LABELS, default_threshold=0.35)
        assert torch.allclose(thresholds, torch.full((len(LABELS),), 0.35))


class TestLoadBundle:
    def test_happy_path(self, bundle_dir: Path) -> None:
        bundle = load_bundle(bundle_dir)
        assert bundle.model_dir == bundle_dir.resolve()
        assert bundle.labels == tuple(LABELS)
        assert bundle.thresholds.tolist() == pytest.approx(THRESHOLDS)
        assert bundle.model_repo == "test/dummy"
        assert "input_size" in bundle.data_config

    def test_raises_when_checkpoint_is_not_a_mapping(self, bundle_dir: Path) -> None:
        torch.save([1, 2, 3], bundle_dir / "best.pt")
        with pytest.raises(ValueError, match="did not contain a mapping"):
            load_bundle(bundle_dir)

    def test_raises_rather_than_guessing_when_args_json_missing(self, bundle_dir: Path) -> None:
        (bundle_dir / "args.json").unlink()
        with pytest.raises(FileNotFoundError, match="does not contain a full model bundle"):
            load_bundle(bundle_dir)

    def test_raises_rather_than_guessing_when_model_repo_missing(self, bundle_dir: Path) -> None:
        (bundle_dir / "args.json").write_text(json.dumps({}))
        with pytest.raises(ValueError, match="model_repo"):
            load_bundle(bundle_dir)
