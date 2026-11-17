"""Stage 5: Narmee's physics model v9d (`pro_v9d_ultimate_arimax_physics`), run with his own code.

v9d is rule-based: tracking physics, ARIMA filters fitted on this match's own tracking, Smart
Tagging and the broadcast camera's tracking. Its only learned numbers come from his cross-match
fit on the 27 open IDSSE / SkillCorner matches: models/physics/weights/cross_match_arima_learned.json.

His code (models/physics/physics_rules) caches parsed tracking in `<data folder>/cache`. To start
from the provided files every time without writing into the data folder, this stage hands it a
private data folder of links to the provided files, rebuilt on every run.

Output: work/physics/v9d_labels.npy, one label code per frame (H 0, A 1, D 2, X 3).
"""
from __future__ import annotations

import importlib
import os
import shutil
import sys
from pathlib import Path

import numpy as np

from main_pipeline.config import LETTERS, PHYSICS_CODE, Paths

INPUTS = {   # what v9d reads from the provided data
    "tactical": ("BAR-ATH_tactical_atd.txt", "BAR-ATH_tactical_atd.xml",
                 "BAR-ATH_tactical_smart_tagging.xml"),
    "broadcast": ("BAR-ATH_broadcast_atd.txt", "BAR-ATH_broadcast_atd.xml"),
}


class PhysicsModel:
    """Runs v9d on BAR-ATH from a private, freshly linked copy of the data folder."""

    def __init__(self, data: Path, work: Path) -> None:
        self.data = data
        self.private_data = work / "physics" / "data"

    def labels(self) -> np.ndarray:
        self._link_inputs()
        os.environ["MTAH_DATA_DIR"] = str(self.private_data)   # read by games.py on import
        sys.path.insert(0, str(PHYSICS_CODE.parent))
        package = PHYSICS_CODE.name
        load_match = importlib.import_module(f"{package}.data").load_match
        compute_features = importlib.import_module(f"{package}.features").compute
        tag_reliability = importlib.import_module(f"{package}.methods").tag_reliability
        v9d = importlib.import_module(f"{package}.v9").pro_v9d_ultimate_arimax_physics

        match = load_match("BAR-ATH", verbose=False)
        frames = compute_features(match)
        letters = v9d(match, frames, use_tags=True, reliability=tag_reliability(match, frames))
        return np.array([LETTERS.index(letter) for letter in letters], dtype=np.int8)

    def _link_inputs(self) -> None:
        """His code silently skips the broadcast camera when its files are absent, which would
        change the labels, so every input must exist."""
        shutil.rmtree(self.private_data, ignore_errors=True)
        for folder, names in INPUTS.items():
            (self.private_data / folder).mkdir(parents=True)
            for name in names:
                source = self.data / folder / name
                if not source.is_file():
                    raise FileNotFoundError(f"v9d needs {source}")
                (self.private_data / folder / name).symlink_to(source)


def main() -> None:
    paths = Paths.from_env()
    codes = PhysicsModel(paths.data, paths.work).labels()
    paths.physics_outputs.parent.mkdir(parents=True, exist_ok=True)
    np.save(paths.physics_outputs, codes)
    print(f"physics model v9d: written to {paths.physics_outputs}")


if __name__ == "__main__":
    main()
