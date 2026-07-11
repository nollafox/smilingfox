from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner
from PIL import Image

from smilingfox.cli import main


class TestTopLevel:
    def test_help(self) -> None:
        result = CliRunner().invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "tag" in result.output
        assert "info" in result.output

    def test_version(self) -> None:
        result = CliRunner().invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "smilingfox" in result.output.lower()


class TestTagCommand:
    def test_tag_single_image_to_stdout(self, bundle_dir: Path, sample_image_path: Path) -> None:
        result = CliRunner().invoke(
            main, ["tag", str(sample_image_path), "--model", str(bundle_dir), "--device", "cpu", "--stdout"]
        )
        assert result.exit_code == 0, result.output
        assert json.loads(result.stdout) == {"tags": ["a", "b"]}

    def test_tag_single_image_writes_sidecar_by_default(self, bundle_dir: Path, sample_image_path: Path) -> None:
        result = CliRunner().invoke(
            main, ["tag", str(sample_image_path), "--model", str(bundle_dir), "--device", "cpu", "--quiet"]
        )
        assert result.exit_code == 0, result.output

        sidecar = sample_image_path.with_suffix(".json")
        assert sidecar.is_file()
        assert json.loads(sidecar.read_text()) == {"tags": ["a", "b"]}

    def test_tag_directory_writes_one_sidecar_per_image(self, bundle_dir: Path, tmp_path: Path) -> None:
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        image_paths = []
        for index in range(3):
            path = images_dir / f"img_{index}.png"
            Image.new("RGB", (16, 16), color=(index, index, index)).save(path)
            image_paths.append(path)

        result = CliRunner().invoke(
            main, ["tag", str(images_dir), "--model", str(bundle_dir), "--device", "cpu", "--quiet"]
        )
        assert result.exit_code == 0, result.output

        for path in image_paths:
            assert json.loads(path.with_suffix(".json").read_text()) == {"tags": ["a", "b"]}

    def test_tag_directory_to_stdout_keys_results_by_path(self, bundle_dir: Path, tmp_path: Path) -> None:
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        image_path = images_dir / "only.jpg"
        Image.new("RGB", (16, 16), color=(9, 9, 9)).save(image_path)

        result = CliRunner().invoke(
            main, ["tag", str(images_dir), "--model", str(bundle_dir), "--device", "cpu", "--stdout"]
        )
        assert result.exit_code == 0, result.output
        payload = json.loads(result.stdout)
        assert payload == {str(image_path): {"tags": ["a", "b"]}}

    def test_confidence_option_raises_effective_threshold(self, bundle_dir: Path, sample_image_path: Path) -> None:
        result = CliRunner().invoke(
            main,
            [
                "tag",
                str(sample_image_path),
                "--model",
                str(bundle_dir),
                "--device",
                "cpu",
                "--stdout",
                "--confidence",
                "0.7",
            ],
        )
        assert result.exit_code == 0, result.output
        assert json.loads(result.stdout) == {"tags": ["a"]}

    def test_no_images_found_exits_cleanly(self, bundle_dir: Path, tmp_path: Path) -> None:
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        result = CliRunner().invoke(main, ["tag", str(empty_dir), "--model", str(bundle_dir)])
        assert result.exit_code == 0
        assert "No images found" in result.output

    def test_unrecognized_file_extension_errors(self, bundle_dir: Path, tmp_path: Path) -> None:
        text_file = tmp_path / "notes.txt"
        text_file.write_text("hello")
        result = CliRunner().invoke(main, ["tag", str(text_file), "--model", str(bundle_dir)])
        assert result.exit_code != 0
        assert "not a recognized image file" in result.output

    def test_nonexistent_path_errors(self, bundle_dir: Path, tmp_path: Path) -> None:
        result = CliRunner().invoke(main, ["tag", str(tmp_path / "missing.jpg"), "--model", str(bundle_dir)])
        assert result.exit_code != 0

    def test_invalid_model_bundle_errors(self, tmp_path: Path, sample_image_path: Path) -> None:
        empty_dir = tmp_path / "not-a-bundle"
        empty_dir.mkdir()
        result = CliRunner().invoke(main, ["tag", str(sample_image_path), "--model", str(empty_dir)])
        assert result.exit_code != 0
        assert "does not contain a full model bundle" in result.output


class TestInfoCommand:
    def test_info_reports_bundle_details(self, bundle_dir: Path) -> None:
        result = CliRunner().invoke(main, ["info", "--model", str(bundle_dir), "--device", "cpu"])
        assert result.exit_code == 0, result.output
        assert "test/dummy" in result.output
        assert "3" in result.output  # label count
        assert "cpu" in result.output
