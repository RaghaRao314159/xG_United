# xG United: frame-level ball possession for More than a Hack 2027

This repository turns the provided BAR-ATH match data into our submission file,
`submission/xG United.csv`: one label per frame of the tactical video (`H`, `A`, `D` or `X`).
It includes the model inference code, fitted weights, and one command that runs the whole
pipeline. Training code is also included; the physics model's original external-data conversion
script is not available (see its README).

The labels come from an ensemble of three models built by the team. Their supervised training sources are
open data from other matches (IDSSE and SkillCorner). The tabular and physics models use the same
27 source matches, with different preprocessing; the BiLSTM uses an IDSSE subset. Unlabelled
BAR-ATH tracking is used for domain adaptation and physical time-series fitting.

| model | what it gives | folder |
|---|---|---|
| **Tabular models** (Ragha): 8 team members (LightGBM and Kumo Tabular) and 2 dead-ball models | P(home \| ball alive) per 0.2 s block; P(ball alive) per 0.48 s window | [`models/tabular/`](models/tabular/) |
| **Two-head BiLSTM** (Tiya) | P(home), P(away), P(dead) per frame | [`models/bilstm/`](models/bilstm/) |
| **Physics model v9d** (Narmee): rules, kinematics and ARIMA filters | a label per frame, used as a second opinion on dead ball | [`models/physics/`](models/physics/) |

The ensemble (`main_pipeline/ensemble.py`) adds the models' evidence in logit space per 0.2 s block,
then decodes each half with a 3-state Viterbi (home, away, dead) with switching costs and
minimum durations. Its settings, with the reason for each, are in that file's `Settings`, and the
recipe is in its docstring.

## Layout

```
xG_United/
├── main_pipeline/             `python -m main_pipeline`: runs every stage, checks the weights, ensembles
├── models/                    one folder per model: its code, training code, inference stage and weights
│   ├── tabular/               Ragha's tabular models
│   │   ├── src/               feature, table and training code, shared by training and inference
│   │   ├── inference/         the pipeline stages: feature tables, then the models' predictions
│   │   ├── training/          wcap.py: saves every model a training script fits
│   │   └── weights/           LightGBM boosters, Kumo contexts and column lists
│   ├── bilstm/                Tiya's BiLSTM
│   │   ├── inference/         her network code and the pipeline stage
│   │   ├── training/          data preparation, training, decoder selection
│   │   └── weights/           the trained network
│   └── physics/               Narmee's physics model
│       ├── physics_rules/     the rules (a Python package), including the cross-match fit
│       ├── inference/         the pipeline stage
│       └── weights/           its learned numbers
├── challenge/                 the challenge's games.py, data loaders and submission validator
│                              (MIT, Metrica Sports; see challenge/LICENSE)
├── weights.lock.json          SHA-256 of every weight file
├── pyproject.toml / uv.lock   dependencies and full dependency lock; Python 3.11.13
├── requirements.txt           pip-compatible export of the runtime lock
├── requirements-training.txt  pip-compatible export including training dependencies
├── training_data.lock.json    external match IDs and immutable source revisions
├── reproduction.json          recorded GPU setup and exact submission checksums
├── verification/              clean RunPod run log, environment and checksum evidence
├── xG United.md / .pdf         technical document for the submission form
└── LICENSE                    MIT
```

Each model's README describes the model, its training data, how to train it and how the
pipeline runs it.

## Run it

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) (0.13.6 or newer).
Run these commands from this repository's root on the reference NVIDIA A40 machine:

```bash
uv sync --locked

# Put the provided match files in data/, or use --data /path/to/data:
# data/tactical/BAR-ATH_tactical_{atd.txt,atd.xml,homography.csv,smart_tagging.xml}
# data/broadcast/BAR-ATH_broadcast_{atd.txt,atd.xml}
uv run --locked python -m main_pipeline --device cuda --threads 8 --verify-reference

# Validate the exact upload file and check its recorded SHA-256:
uv run --locked python challenge/evaluation/validate_submission.py "submission/xG United.csv"
uv run --locked python -m main_pipeline.verify_submission
```

The command writes **`submission/xG United.csv`**, matching the registered team name including
its space. Quote this filename in shell commands. The competition does not prescribe replacing
spaces with underscores. The GitHub repository can keep the name `xG_United`.

The technical document is [xG United.md](<xG United.md>) and [xG United.pdf](<xG United.pdf>).
Upload the CSV and one version of the document to the competition form.

`uv sync --locked --group training` adds training dependencies. The two requirements files are
exports from `uv.lock` for pip users; `requirements-training.txt` includes the runtime packages.
To regenerate them after deliberately changing the project dependencies:

```bash
uv export --locked --no-default-groups --no-hashes --no-header --no-annotate \
  --format requirements-txt --output-file requirements.txt
uv export --locked --group training --no-default-groups --no-hashes --no-header --no-annotate \
  --format requirements-txt --output-file requirements-training.txt
```

The lock preserves the recorded direct inference versions and pins transitive dependencies.
It was generated during packaging, not recovered from the original machine. Historical
`scikit-learn` and `lxml` training versions were not recorded; the training group now pins
1.9.1 and 6.1.3 respectively as a reconstruction environment, without retraining the bundled weights.


The first run downloads one file that is not stored here: NVIDIA's public Kumo Tabular network
(`nvidia/Kumo-Tabular`, revision `v1.0.1`, 855 MB, too large for git). It is pinned by SHA-256
like every other weight file. No account or token is needed.

Options: `--verify-reference` checks the submitted CSV checksum; `--data DIR` (default `data/`), `--device auto|cuda|cpu` (default `auto`: CUDA if
available), `--threads N` (CPU threads, default 8), `--stages ...` to run only some stages.

## What happens

`python -m main_pipeline` runs six stages in order, each in its own process. Each stage reads the
previous stages' outputs from `work/`.

| stage | code | does | writes |
|---|---|---|---|
| `check_weights` | `main_pipeline/check_weights.py` | checks every weight file against `weights.lock.json`; downloads the Kumo network | `models/tabular/weights/kumo_checkpoint/` |
| `tabular_features` | `models/tabular/inference/tabular_features.py` | the tabular models' feature tables from the tactical tracking | `work/tabular/features/` |
| `tabular_models` | `models/tabular/inference/tabular_models.py` | the 8 team members and 2 dead-ball models | `work/tabular/probabilities.npz` |
| `bilstm_model` | `models/bilstm/inference/bilstm_model.py` | the BiLSTM, on the plain and on the home/away-exchanged inputs | `work/bilstm/probabilities.npz` |
| `physics_model` | `models/physics/inference/physics_model.py` | the physics model v9d | `work/physics/v9d_labels.npy` |
| `ensemble` | `main_pipeline/ensemble.py` | combines everything, decodes, writes the file and runs the challenge's validator | `submission/xG United.csv` |

## Reproducibility

**Verified on 18 November 2026 (Singapore time): a clean RunPod A40 run regenerated the submitted
CSV byte for byte in 369 seconds (6 minutes 9 seconds).** Only source, bundled weights and the six
original input assets were uploaded. The reference CSV and prediction caches were not uploaded.
The run rebuilt every stage, passed the official format validator and passed `--verify-reference`.
The downloaded result also passed a local byte-for-byte comparison with the original submission.

| Setting | Verified environment |
|---|---|
| GPU | One NVIDIA A40, 46,068 MiB reported by `nvidia-smi` |
| NVIDIA driver | 595.112.07 |
| OS | Ubuntu 24.04.3 LTS, Linux x86_64, glibc 2.39 |
| Base container | `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` |
| Python / uv | 3.11.13 / 0.13.6 |
| Project PyTorch / CUDA runtime | 2.15.0+cu132 / 13.2 |
| cuDNN package | 9.26.0.51 |
| CPU allocation / inference threads | 9 vCPU / 8 threads |
| Neural precision | CUDA float16 autocast |
| Inference batching | Kumo: 8,192 query rows; BiLSTM: 64 windows |

`uv sync --locked` creates the project environment independently of the base image's Python and
PyTorch. `.python-version` now pins the Python patch version observed in the successful run.

The exact output checksums are:

    md5  6b266c5aadc2885a8b3051de0db15386  submission/xG United.csv
    sha256  ac2aba453442471bde7e2e34f74e36b58062701b5be6523b22f9b432f4edf6f6

The [verification report](verification/report.json), [captured environment](verification/environment.json)
and [run log](verification/run.log) preserve the checks and measured setup. The report also links
SHA-256 manifests for the six input assets and tested source/configuration files.

The earlier recorded A40 run used driver 595.91.07; its OS/container and Python patch version were
not recorded. That historical setup is retained separately in `reproduction.json`. The new run
confirms exact reproduction with driver 595.112.07 and the locked environment above.

On other hardware, byte equality is checked rather than assumed:

- Two parts are neural networks: the Kumo members of the tabular models and the BiLSTM. Their
  probabilities depend slightly on the hardware (around the third decimal place), because
  different chips add up floating-point numbers in different orders. Near a decision boundary
  that can flip a label.
- Keep the complete software environment fixed for all stages, including numerical libraries
  used by LightGBM and the physics model. Cross-platform equality is not guaranteed.
- On a GPU the networks run in float16 autocast, as the models were built; on the CPU in float32.
  The CPU works but takes hours, almost all of it in the Kumo members.

Every run rebuilds its features from the provided files. Caches go to `work/` and
`models/tabular/runs/`, never to the data folder.

## Data

External dataset identities and pinned download revisions are in `training_data.lock.json`.
The tabular and BiLSTM downloaders use those revisions; SkillCorner downloads only the twenty
listed matches. See the model READMEs for their different preprocessing and label conventions.

The provided competition match assets are not in this repository: they are licensed for the competition
only. `.gitignore` excludes `data/`, `work/`, the tabular code's scratch folder and other submission
outputs. It allows only the final algorithm-generated predictions, `submission/xG United.csv`.
That file contains frame indices and possession labels, without tracking coordinates or video.
Including it in the repository does not replace uploading the CSV to the competition form.

## Licence

MIT (see `LICENSE`). The files in `challenge/` are the challenge organisers' own, under their MIT
licence (`challenge/LICENSE`). The Kumo Tabular network is NVIDIA's and is downloaded under
NVIDIA's terms; it is not redistributed here.
