"""Stage 2: build the inputs of Ragha's tabular models from the provided tactical tracking.

Ragha's feature code (models/tabular/src, the code that built the training tables) is used
unchanged, in three steps:

1. `pipeline.py features`: per-frame tracking features of BAR-ATH;
2. `lab/exp_t/build_bar.py`: the tracking in the layout the models' feature tables read;
3. the model tables, exported here: 0.2 s team blocks (column sets `b` and `ab`), 12-frame
   dead-ball windows, and the kick-off mask with the half boundaries for the decoder.

Steps 1 and 2 keep their intermediate files in models/tabular/runs/, that code's own scratch
folder; it is deleted first so every run starts from the provided data. Step 3 writes
work/tabular/features/.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from main_pipeline.config import TABULAR, TABULAR_CODE, Paths

SCRATCH = TABULAR / "runs"         # where the tabular code keeps its intermediate files
CONDITION = "d85suc"               # the original code's name for the training data; BAR-ATH's tables do not depend on it


def run_original_script(script: Path, *args: str) -> None:
    subprocess.run([sys.executable, str(script), *args], check=True, cwd=TABULAR)


def to_json(spec: dict) -> dict:
    """Mirror specs hold tuples; JSON stores them as lists (tabular_models turns them back)."""
    return {str(k): list(v) if isinstance(v, tuple) else v for k, v in spec.items()}


class FeatureTables:
    """Writes the three model tables, built by the same code that built the training tables."""

    def __init__(self) -> None:
        sys.path.insert(0, str(TABULAR_CODE / "lab" / "exp_k"))
        import bar_k    # the BAR-ATH entry point of the original code; sets up its import paths
        self.code = bar_k

    def write(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        self._dead_ball_windows(out_dir)
        self._team_blocks(out_dir)
        self._decoder_inputs(out_dir)

    def _dead_ball_windows(self, out_dir: Path) -> None:
        """12-frame windows with every column the dead-ball models read."""
        code = self.code
        (x, _, meta), names, spec = code.xfer.palive_table(code.TARGET, CONDITION, {})
        starts = np.array([start for _, _, start, _ in code.pipeline.windows(meta)], dtype=np.int64)
        assert len(starts) == len(x)
        np.savez_compressed(out_dir / "dead_ball_windows.npz", x=x.astype(np.float32), starts=starts)
        (out_dir / "dead_ball_windows.json").write_text(json.dumps({"names": list(names), "spec": to_json(spec)}))

    def _team_blocks(self, out_dir: Path) -> None:
        """0.2 s blocks with the team models' columns. The team tables are built with the
        original home/away mirror and one-second windows, so those are set first."""
        code = self.code
        code.tb.swap = code.xfer.ORIGINAL_SWAP
        code.pipeline.WINDOW = code.tb.WINDOW = code.pipeline.FPS
        for column_set in ("b", "ab"):
            x, info, (a_names, b_names, spec) = code.kt.table(code.TARGET, CONDITION, column_set)
            np.savez_compressed(out_dir / f"team_blocks_{column_set}.npz", x=x.astype(np.float32),
                                **{"info_" + k: v for k, v in info.items()})
            (out_dir / f"team_blocks_{column_set}.json").write_text(json.dumps(
                {"a_names": list(a_names), "b_names": list(b_names), "spec": to_json(spec)}))

    def _decoder_inputs(self, out_dir: Path) -> None:
        """Half boundaries and the kick-off rule: frames before each half's kick-off and the
        detected kick-off set-ups are forced to dead ball."""
        code = self.code
        meta, frames = code.pipeline.load_dataset(code.TARGET)
        setups, _, _, _ = code.kick.kick_mask(code.kick.load_raw(code.TARGET), meta, frames,
                                              players_only=True)
        before_kickoff = np.zeros(meta["n_frames"], dtype=bool)
        kickoffs = code.dball.kickoff_frames()
        for half, first, last in meta["halves"]:
            before_kickoff[first:min(kickoffs[half], last + 1)] = True
        np.savez_compressed(out_dir / "decoder_inputs.npz", forced_dead=setups | before_kickoff,
                            halves=np.array(meta["halves"], dtype=np.int64))


def main() -> None:
    paths = Paths.from_env()
    shutil.rmtree(SCRATCH, ignore_errors=True)
    run_original_script(TABULAR_CODE / "pipeline.py", "features")
    run_original_script(TABULAR_CODE / "lab" / "exp_t" / "build_bar.py")
    FeatureTables().write(paths.tabular_features)
    print(f"tabular features: written to {paths.tabular_features}")


if __name__ == "__main__":
    main()
