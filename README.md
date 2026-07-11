# smilingfox

<p align="center">
  <img src="https://github.com/nollafox/smilingfox/raw/main/docs/banner.png" alt="smilingfox banner" style="border-radius: 16px; box-shadow: 0 8px 32px rgba(0, 0, 0, 0.12); max-width: 100%; height: auto;">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-1E90FF" alt="Python 3.10+">
  <a href="https://pypi.org/project/smilingfox/">
    <img src="https://img.shields.io/pypi/v/smilingfox?color=4169E1" alt="PyPI version">
  </a>
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
  <a href="#how-thresholds-work">Thresholds</a> •
  <a href="#development">Development</a> •
  <a href="#model-details">Model details</a>
</p>

Most taggers hand you one confidence slider and leave you to figure out the rest. **smilingfox** wraps the [SmilingFox E621 tagger](https://huggingface.co/nollafox/smilingfox) with something more deliberate: all 32,617 labels carry their own calibrated decision threshold, so a rare, easily-confused tag isn't held to the same bar as `anthro` or `fur`. Point it at an image or a folder and get tags back — as Python objects or `.json` sidecars, whichever fits the job.

## Install

```bash
pipx install smilingfox
```

[pipx](https://pipx.pypa.io) keeps torch/timm and everything else out of your system Python. No pipx? `brew install pipx` on macOS, or `python3 -m pip install --user pipx` anywhere, then run `pipx ensurepath` once and restart your shell. Plain `pip install smilingfox` also works — you just lose the isolation.

Confirm it's on your `PATH`:

```bash
smilingfox --help
```

### From source

For development on the package itself, or to run an unreleased version:

```bash
git clone git@github.com:nollafox/smilingfox.git
cd smilingfox
poetry install
poetry run smilingfox --help
```

`pipx install .` / `pip install --user .` from the checkout also put the `smilingfox` command on `PATH` without going through Poetry.

## Python API

```python
from smilingfox import Tagger

tagger = Tagger.from_pretrained()  # downloads and caches nollafox/smilingfox
prediction = tagger.predict("photo.jpg")

prediction.tags    # ("anthro", "mammal", "fur", ...)
prediction.scores  # {"anthro": 0.91, "mammal": 0.88, ...}
```

`Tagger` only builds the model once, so reuse it for batches instead of calling `predict` in a loop:

```python
predictions = tagger.predict_batch(["a.jpg", "b.jpg", "c.jpg"], batch_size=8)
```

For a one-off script you don't need to construct a `Tagger` at all — `smilingfox.tag()` lazily loads and caches a default one the first time you call it:

```python
import smilingfox

smilingfox.tag("photo.jpg").tags
```

`from_pretrained` takes either a Hugging Face repo id or a local bundle directory, so the same code works against a fine-tune or an offline copy:

```python
tagger = Tagger.from_pretrained("/path/to/local/bundle", device="cuda")
```

Both `predict` and `predict_batch` also take `required_confidence`, applied as a floor on top of each label's own calibrated threshold — raise it when you want fewer, more confident tags:

```python
tagger.predict("photo.jpg", required_confidence=0.5)
```

## CLI

```bash
$ smilingfox tag ./photos                  # one .json sidecar per image
$ smilingfox tag photo.jpg --stdout        # print JSON instead of writing sidecars
$ smilingfox tag ./photos --confidence 0.5
$ smilingfox tag ./photos --device cuda --model /path/to/bundle
```

`--stdout` prints plain JSON, so it pipes straight into `jq` or a script. Progress and status messages go to stderr, not stdout, so they never end up mixed into the output:

```bash
$ smilingfox tag photo.jpg --stdout
{
  "tags": [
    "anthro",
    "mammal",
    "fur",
    "solo"
  ]
}
```

`smilingfox info` shows what's actually loaded — the fastest way to check that a `--model` override took effect before you tag a thousand images against the wrong bundle:

```bash
$ smilingfox info
                                 SmilingFox
┏━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ field             ┃ value                                                 ┃
┡━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ model dir         │ ~/.cache/huggingface/hub/models--nollafox--smilingfox │
│ base architecture │ SmilingWolf/wd-swinv2-tagger-v3                       │
│ labels            │ 32,617                                                │
│ device            │ mps                                                   │
└───────────────────┴───────────────────────────────────────────────────────┘
```

`smilingfox tag --help` / `smilingfox info --help` for the rest of the options.

## How Thresholds Work

A single global cutoff treats every label the same, but not every label is equally hard to call. The 0.35 default is really a fallback: most of the 32,617 labels don't have enough training examples to earn their own confident cutoff, so they inherit it. The labels with enough support get their own threshold instead — picked by grid search to maximize that label's F1, then shrunk back toward the default in proportion to how little data backed it up. A well-supported, easy-to-separate label might settle near 0.24; a label the model tends to over-fire on might sit closer to 0.77.

`--confidence` / `required_confidence` doesn't replace that calibration, it sets a floor underneath it. Pass `0.5` and every label's effective threshold becomes `max(calibrated_threshold, 0.5)` — results get tighter without losing the higher bar already set on labels that needed one.

See the [model card](https://huggingface.co/nollafox/smilingfox) for the full calibration policy and the threshold distribution across all 32,617 labels.

## Development

```bash
$ poetry install
$ poetry run smilingfox --help
$ poetry run pytest
$ poetry run python -m compileall -q src/smilingfox
```

## Model details

See the [model card](https://huggingface.co/nollafox/smilingfox) for the base architecture and training configuration.

<br>

***

<p align="center">
  <strong>smilingfox</strong> — a calibrated multi-label tagger for e621 images.
</p>

<p align="center">
  Crafted with ❤️ by <strong>Nolla Fox</strong>
</p>
