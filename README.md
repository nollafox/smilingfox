# smilingfox

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-1E90FF" alt="Python 3.10+">
  <a href="https://github.com/nollafox/smilingfox/actions/workflows/test.yml">
    <img src="https://img.shields.io/github/actions/workflow/status/nollafox/smilingfox/test.yml?branch=main&label=tests&color=2E8B57" alt="Test status">
  </a>
  <img src="https://img.shields.io/github/license/nollafox/smilingfox?color=4169E1" alt="License: MIT">
  <a href="https://huggingface.co/nollafox/smilingfox">
    <img src="https://img.shields.io/badge/model-%F0%9F%A4%97%20nollafox%2Fsmilingfox-8A2BE2" alt="Model on Hugging Face">
  </a>
  <img src="https://img.shields.io/badge/author-Nolla%20Fox-2E8B57" alt="Author: Nolla Fox">
</p>

<p align="center">
  <a href="#install">Install</a> •
  <a href="#python-api">Python API</a> •
  <a href="#cli">CLI</a> •
  <a href="#development">Development</a> •
  <a href="#model-details">Model details</a>
</p>

Python API and CLI for the [SmilingFox E621 tagger](https://huggingface.co/nollafox/smilingfox) — a 32,617-label multi-label image tagger with calibrated per-label thresholds instead of one global confidence cutoff.

## Install

Not on PyPI yet, so install from a checkout of this repo.

If you just want the `smilingfox` command, use [pipx](https://pipx.pypa.io) — it keeps torch/timm and everything else out of your system Python:

```bash
git clone git@github.com:nollafox/smilingfox.git
cd smilingfox
pipx install .
smilingfox --help
```

No pipx? `brew install pipx` on macOS, or `python3 -m pip install --user pipx` anywhere, then run `pipx ensurepath` once and restart your shell. `pip install --user .` also works without pipx — you just lose the isolation.

Working on the package itself, use Poetry instead:

```bash
poetry install
poetry run smilingfox --help
```

## Python API

```python
from smilingfox import Tagger

tagger = Tagger.from_pretrained()  # downloads and caches nollafox/smilingfox
prediction = tagger.predict("photo.jpg")

prediction.tags    # ("anthro", "mammal", "fur", ...)
prediction.scores  # {"anthro": 0.91, "mammal": 0.88, ...}
```

Batching:

```python
predictions = tagger.predict_batch(["a.jpg", "b.jpg", "c.jpg"], batch_size=8)
```

For a one-off script you don't need to construct a `Tagger` at all — `smilingfox.tag()` lazily loads and caches a default one on first use:

```python
import smilingfox

smilingfox.tag("photo.jpg").tags
```

`from_pretrained` takes either a Hugging Face repo id or a local bundle directory. `predict`/`predict_batch` take a `required_confidence` that's applied as a floor on top of each label's own calibrated threshold:

```python
tagger = Tagger.from_pretrained("/path/to/local/bundle", device="cuda")
tagger.predict("photo.jpg", required_confidence=0.5)
```

## CLI

```bash
smilingfox tag ./photos                  # one .json sidecar per image
smilingfox tag photo.jpg --stdout        # print JSON instead of writing sidecars
smilingfox tag ./photos --confidence 0.5
smilingfox tag ./photos --device cuda --model /path/to/bundle
smilingfox info                          # what model is loaded, and from where
```

`smilingfox tag --help` / `smilingfox info --help` for the rest of the options.

## Development

```bash
poetry install
poetry run pytest
```

## Model details

See the [model card](https://huggingface.co/nollafox/smilingfox) for architecture, training setup, and how the thresholds were calibrated.
