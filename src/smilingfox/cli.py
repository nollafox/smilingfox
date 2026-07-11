"""Command-line interface for the SmilingFox tagger."""

from __future__ import annotations

import json
from pathlib import Path

import click
from rich.console import Console
from rich.progress import track
from rich.table import Table

from .tagger import Prediction, Tagger

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"})

console = Console()  # `info`'s table: this command's actual output, so it stays on stdout.

# `tag`'s progress/status chatter goes to stderr, so `tag --stdout`'s JSON stays
# clean and pipeable on stdout.
status_console = Console(stderr=True)


def _iter_images(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise click.ClickException(
                f"'{path}' is not a recognized image file ({', '.join(sorted(IMAGE_EXTENSIONS))})."
            )
        return [path]
    return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def _sidecar_path(image_path: Path) -> Path:
    return image_path.with_suffix(".json")


def _load_tagger(model_source: str | None, device: str | None) -> Tagger:
    try:
        return Tagger.from_pretrained(model_source, device=device)
    except (OSError, RuntimeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc


@click.group()
@click.version_option(package_name="smilingfox", prog_name="smilingfox")
def main() -> None:
    """SmilingFox: an E621 image tagger."""


@main.command()
@click.argument("path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "-c",
    "--confidence",
    "required_confidence",
    type=float,
    default=0.0,
    show_default=True,
    help="Minimum confidence required for a tag, on top of its calibrated threshold.",
)
@click.option(
    "-m",
    "--model",
    "model_source",
    default=None,
    help="Local bundle directory or Hugging Face repo id. Defaults to the published nollafox/smilingfox repo.",
)
@click.option("--device", default=None, help="Torch device to run on (cpu, cuda, mps). Auto-detected by default.")
@click.option("--stdout", "to_stdout", is_flag=True, help="Print results as JSON instead of writing .json sidecar files.")
@click.option("--quiet", is_flag=True, help="Suppress progress output.")
def tag(
    path: Path,
    required_confidence: float,
    model_source: str | None,
    device: str | None,
    to_stdout: bool,
    quiet: bool,
) -> None:
    """Tag an image, or every image in a directory (recursively)."""
    images = _iter_images(path)
    if not images:
        status_console.print(f"No images found under {path}")
        return

    if not quiet:
        status_console.print(f"Loading model ([bold]{model_source or 'nollafox/smilingfox'}[/bold])...")
    tagger = _load_tagger(model_source, device)

    results: dict[Path, Prediction] = {}
    for image_path in track(images, description="tagging", console=status_console, disable=quiet):
        results[image_path] = tagger.predict(image_path, required_confidence=required_confidence)

    if to_stdout:
        if path.is_file():
            prediction = results[images[0]]
            click.echo(json.dumps({"tags": list(prediction.tags)}, indent=2))
        else:
            payload = {str(p): {"tags": list(pred.tags)} for p, pred in results.items()}
            click.echo(json.dumps(payload, indent=2))
        return

    for image_path, prediction in results.items():
        _sidecar_path(image_path).write_text(
            json.dumps({"tags": list(prediction.tags)}, indent=2) + "\n", encoding="utf-8"
        )
    if not quiet:
        status_console.print(f"Wrote {len(results)} sidecar file{'s' if len(results) != 1 else ''}.")


@main.command()
@click.option(
    "-m",
    "--model",
    "model_source",
    default=None,
    help="Local bundle directory or Hugging Face repo id. Defaults to the published nollafox/smilingfox repo.",
)
@click.option("--device", default=None, help="Torch device to run on (cpu, cuda, mps). Auto-detected by default.")
def info(model_source: str | None, device: str | None) -> None:
    """Show details about the loaded model bundle."""
    tagger = _load_tagger(model_source, device)

    table = Table(title="SmilingFox", header_style="bold magenta")
    table.add_column("field")
    table.add_column("value")
    table.add_row("model dir", str(tagger.bundle.model_dir))
    table.add_row("base architecture", tagger.bundle.model_repo)
    table.add_row("labels", f"{len(tagger.labels):,}")
    table.add_row("device", str(tagger.device))
    console.print(table)
