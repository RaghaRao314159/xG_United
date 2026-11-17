"""Locations, constants and the process environment shared by every stage.

`python -m main_pipeline` runs each stage in its own process and passes the locations below through
environment variables (see `Paths.environment`). A single stage can therefore also be run on its
own from the repository root, e.g. `python -m models.bilstm.inference.bilstm_model` or
`python -m main_pipeline.ensemble`, once the stages before it have written their outputs.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Each model lives in models/<name>/ with its code, its training code and its weights.
MODELS = REPO / "models"
TABULAR = MODELS / "tabular"                     # Ragha: LightGBM + Kumo Tabular team and dead-ball models
TABULAR_CODE = TABULAR / "src"                   # his feature and table code (used for training and inference)
TABULAR_WEIGHTS = TABULAR / "weights"
KUMO_CHECKPOINT = TABULAR_WEIGHTS / "kumo_checkpoint"   # Hugging Face cache for NVIDIA's Kumo network (downloaded)
BILSTM = MODELS / "bilstm"                       # Tiya: two-head BiLSTM
BILSTM_CODE = BILSTM / "inference"
BILSTM_WEIGHTS = BILSTM / "weights" / "model_weights.pt"
PHYSICS = MODELS / "physics"                     # Narmee: rule-based physics model v9d
PHYSICS_CODE = PHYSICS / "physics_rules"         # a Python package; its learned numbers are in physics/weights/

# The challenge's own files (MIT licence, Metrica Sports), which the models' code imports:
# games.py (match constants and data paths), starter/ (data loaders), evaluation/ (the validator).
CHALLENGE = REPO / "challenge"
IMPORT_PATHS = (CHALLENGE, CHALLENGE / "starter", CHALLENGE / "evaluation")

TEAM_NAME = "xG United"           # exact registered team name, including the space

N_FRAMES = 199_319                # one submission row per frame of the BAR-ATH tactical video
H, A, D, X = 0, 1, 2, 3           # label codes used everywhere: home, away, dead ball, ungraded
LETTERS = "HADX"
DEFAULT_THREADS = 8               # fixed so that results do not depend on the machine's core count


@dataclass(frozen=True)
class Paths:
    """Where the match data, each stage's outputs and the submission live."""

    data: Path      # the provided match data: tactical/ and broadcast/ as the challenge ships them
    work: Path      # intermediate outputs of every stage
    out: Path       # the submission file

    @classmethod
    def from_env(cls) -> "Paths":
        """The locations chosen by `python -m main_pipeline` (or the defaults inside the repository)."""
        def get(name: str, default: str) -> Path:
            return Path(os.environ.get(name, REPO / default)).resolve()
        return cls(data=get("XG_DATA", "data"), work=get("XG_WORK", "work"),
                   out=get("XG_OUT", "submission"))

    # -- stage outputs ----------------------------------------------------------------------------
    @property
    def tabular_features(self) -> Path:
        return self.work / "tabular" / "features"

    @property
    def tabular_outputs(self) -> Path:
        return self.work / "tabular" / "probabilities.npz"

    @property
    def bilstm_outputs(self) -> Path:
        return self.work / "bilstm" / "probabilities.npz"

    @property
    def physics_outputs(self) -> Path:
        return self.work / "physics" / "v9d_labels.npy"

    @property
    def submission(self) -> Path:
        return self.out / f"{TEAM_NAME}.csv"

    # -- process environment ----------------------------------------------------------------------
    def environment(self, threads: int, device: str, offline: bool = True) -> dict[str, str]:
        """Environment for one stage process.

        Besides our own locations it sets what the models' code and the numeric libraries read:
        the import path to the challenge's files, the match-data folder for the challenge's
        `games.py`, the Hugging Face cache holding the Kumo checkpoint, a fixed hash seed and a
        fixed number of threads.
        """
        env = dict(os.environ)
        env.update({
            "XG_DATA": str(self.data), "XG_WORK": str(self.work), "XG_OUT": str(self.out),
            "XG_THREADS": str(threads), "XG_DEVICE": device,
            "PYTHONPATH": os.pathsep.join([str(p) for p in IMPORT_PATHS]
                                          + [p for p in [os.environ.get("PYTHONPATH")] if p]),
            "MTAH_DATA_DIR": str(self.data),
            "HF_HOME": str(KUMO_CHECKPOINT),
            "PYTHONHASHSEED": "0",
            "OMP_NUM_THREADS": str(threads), "MKL_NUM_THREADS": str(threads),
            "OPENBLAS_NUM_THREADS": str(threads),
        })
        if offline:
            env["HF_HUB_OFFLINE"] = "1"   # inference never downloads anything
        return env


def torch_device():
    """The device for the neural parts (Kumo and the BiLSTM), chosen by `--device`.

    `auto` (the default) uses a CUDA GPU when one is available, as the models were built, and
    the CPU otherwise. The CPU threads are fixed by `--threads` either way.
    """
    import torch
    torch.set_num_threads(int(os.environ.get("XG_THREADS", DEFAULT_THREADS)))
    choice = os.environ.get("XG_DEVICE", "auto")
    if choice == "auto":
        choice = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(choice)
