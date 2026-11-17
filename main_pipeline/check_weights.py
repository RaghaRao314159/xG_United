"""Stage 1: check every model file against `weights.lock.json`, and fetch NVIDIA's Kumo network.

Every weight file the team trained is stored in this repository, under `models/<model>/weights/`:

    models/tabular/weights/team_idsse_rule_R2/            team model, IDSSE rule-R2 labels + SkillCorner
    models/tabular/weights/team_idsse_rule_E/             team model, IDSSE rule-E labels + SkillCorner
    models/tabular/weights/dead_ball_idsse_skillcorner/   dead-ball model, IDSSE + SkillCorner
    models/tabular/weights/dead_ball_idsse_carried_ball/  dead-ball model, IDSSE + carried-ball copies
    models/bilstm/weights/model_weights.pt                Tiya's network
    models/physics/weights/cross_match_arima_learned.json the numbers Narmee's v9d learned

The one exception is the pretrained network behind the Kumo members: NVIDIA's public
`nvidia/Kumo-Tabular` checkpoint (855 MB, too large for a git repository). It is downloaded once,
at a pinned revision, into the Hugging Face cache `models/tabular/weights/kumo_checkpoint/`, where
`sdm` looks for it. Every file is checked by SHA-256 and a mismatch stops the run, so a run either
uses exactly the weights the submission was made with or does not run at all.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .config import KUMO_CHECKPOINT, REPO

LOCK_FILE = REPO / "weights.lock.json"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path: Path, expected: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{path}: missing (see weights.lock.json)")
    actual = sha256_of(path)
    if actual != expected:
        raise RuntimeError(f"{path}: SHA-256 {actual} differs from the lock ({expected})")
    return path


@dataclass(frozen=True)
class Download:
    """One file of a public Hugging Face repository, at a pinned revision."""

    repo: str
    revision: str
    file: str
    sha256: str

    def fetch(self) -> Path:
        """Download into the Kumo cache (unless it is already there) and verify."""
        from huggingface_hub import hf_hub_download
        local = hf_hub_download(self.repo, self.file, revision=self.revision,
                                cache_dir=KUMO_CHECKPOINT / "hub")
        return verify(Path(local), self.sha256)


def main() -> None:
    lock = json.loads(LOCK_FILE.read_text())
    for entry in lock["files"]:
        verify(REPO / entry["path"], entry["sha256"])
        print(f"  ok  {entry['path']}")
    for entry in lock["downloads"]:
        Download(**entry).fetch()
        print(f"  ok  {entry['repo']}/{entry['file']} @ {entry['revision']}")
    print(f"weights: {len(lock['files']) + len(lock['downloads'])} files verified against {LOCK_FILE.name}")


if __name__ == "__main__":
    main()
