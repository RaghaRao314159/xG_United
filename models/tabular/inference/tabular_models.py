"""Stage 3: Ragha's tabular models.

Two kinds of model. Each answers a row and its home/away-mirrored copy and averages the two, so
neither team is favoured for being listed first.

* **Team models**: P(home has the ball | ball alive) for each 0.2 s block. Eight members: Kumo
  and LightGBM, each on column sets `b` and `ab`, each trained with two labellings of the open
  IDSSE matches (rules R2 and E) plus the open SkillCorner matches.
* **Dead-ball models**: P(ball alive) for each 12-frame window. Two models, each a 50/50 logit
  blend of a LightGBM part and a Kumo part: one trained on IDSSE + SkillCorner, one on IDSSE
  plus copies of it with a realistically carried ball during stoppages.

Output: work/tabular/probabilities.npz, read by the ensemble stage.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from main_pipeline.config import TABULAR_CODE, TABULAR_WEIGHTS, Paths
from .kumo import KumoMember

TEAM_MODELS = {"R2": "team_idsse_rule_R2", "E": "team_idsse_rule_E"}     # label rule -> weights folder
TEAM_MEMBERS = ("kumo_b", "kumo_ab", "lightgbm_b", "lightgbm_ab")        # <family>_<column set>
DEAD_BALL_MODELS = ("dead_ball_idsse_skillcorner", "dead_ball_idsse_carried_ball")


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def as_tuples(spec: dict) -> dict:
    return {k: tuple(v) if isinstance(v, list) else v for k, v in spec.items()}


def read_json(path: Path):
    return json.loads(path.read_text())


class OriginalCode:
    """Ragha's original code keeps two settings in module globals: the window length and the
    function that mirrors a table's columns between home and away. The dead-ball tables use
    12-frame windows, the team tables one-second windows and the original mirror, so the right
    pair must be set before each model's mirror is built."""

    def __init__(self) -> None:
        sys.path.insert(0, str(TABULAR_CODE / "lab" / "exp_k"))
        import bar_k    # importing it sets the 12-frame dead-ball windows
        self.code = bar_k
        self.tb = bar_k.tb   # table helpers: label codes H/A/D and the NaN fill value
        self.dead_ball_windows = (bar_k.pipeline.WINDOW, bar_k.tb.WINDOW)

    def dead_ball_mirror(self, names: list[str], spec: dict):
        """Mirror for dead-ball rows: exchanges the home and away columns."""
        code = self.code
        code.pipeline.WINDOW, code.tb.WINDOW = self.dead_ball_windows
        code.tb.swap = swap = code.palive.subset_swap(code.run_tb.make_swap(as_tuples(spec)), names)(names)
        return lambda x: swap(x, names)

    def team_mirror(self, columns: dict):
        """Mirror for team rows of one column set."""
        code = self.code
        code.tb.swap = code.xfer.ORIGINAL_SWAP
        code.pipeline.WINDOW = code.tb.WINDOW = code.pipeline.FPS
        return code.kt.u_fit.make_swap(columns["a_names"], columns["b_names"], as_tuples(columns["spec"]))


class DeadBallModel:
    """P(ball alive) per window: a LightGBM part and a Kumo part, blended 50/50 in logit space.

    Both parts are 3-class (home, away, dead). Each answers a window and its mirrored copy, in
    which home and away are exchanged, and P(alive) = 1 - mean P(dead)."""

    def __init__(self, folder: Path, original: OriginalCode) -> None:
        self.folder, self.original = folder, original

    def predict(self, x: np.ndarray, names: list[str], spec: dict) -> np.ndarray:
        assert read_json(self.folder / "features.json") == names, \
            f"{self.folder.name}: the feature table's columns differ from the training columns"
        mirror = self.original.dead_ball_mirror(names, spec)
        lightgbm_part, kumo_part = self._lightgbm(x, mirror), self._kumo(x, mirror)
        return sigmoid(0.5 * logit(lightgbm_part) + 0.5 * logit(kumo_part))

    def _p_alive(self, p: np.ndarray, p_mirrored: np.ndarray) -> np.ndarray:
        """P(alive) from the answers for the rows and for their mirrors (home and away swapped back)."""
        tb = self.original.tb
        mean = 0.5 * (p + p_mirrored[:, [tb.A, tb.H, tb.D]])
        return 1.0 - mean[:, tb.D]

    def _lightgbm(self, x: np.ndarray, mirror) -> np.ndarray:
        import lightgbm
        booster = lightgbm.Booster(model_file=str(self.folder / "lightgbm.txt"))
        classes = read_json(self.folder / "lightgbm.meta.json")["classes"]
        rows = np.nan_to_num(np.vstack([x, mirror(x)]), nan=self.original.tb.FILL)  # NaN filled as in training
        proba = np.zeros((len(rows), 3))
        proba[:, np.asarray(classes, dtype=int)] = booster.predict(rows)
        return self._p_alive(proba[:len(x)], proba[len(x):])

    def _kumo(self, x: np.ndarray, mirror) -> np.ndarray:
        member = KumoMember(self.folder / "kumo.context.npz", self.folder / "kumo.columns.npy", n_classes=3)
        p, p_mirrored = member.predict_twice(x, mirror(x))
        normalise = lambda q: q / q.sum(axis=1, keepdims=True)    # noqa: E731
        return self._p_alive(normalise(p), normalise(p_mirrored))


class TeamMember:
    """P(home | alive) per block from one member: 0.5 * (p(rows) + 1 - p(mirrored rows))."""

    def __init__(self, folder: Path, name: str) -> None:
        self.folder, self.name = folder, name

    def predict(self, x: np.ndarray, mirror) -> np.ndarray:
        if self.name.startswith("lightgbm"):
            import lightgbm
            booster = lightgbm.Booster(model_file=str(self.folder / f"{self.name}.txt"))
            p = booster.predict(np.vstack([x, mirror(x)]))
            return 0.5 * (p[:len(x)] + 1.0 - p[len(x):])
        member = KumoMember(self.folder / f"{self.name}.context.npz",
                            self.folder / f"{self.name}.columns.npy", n_classes=2)
        p, p_mirrored = member.predict_twice(x, mirror(x))
        return 0.5 * (p[:, 1] + 1.0 - p_mirrored[:, 1])


class TabularModels:
    """Runs every model on the feature tables and collects their outputs."""

    def __init__(self, paths: Paths) -> None:
        self.weights = TABULAR_WEIGHTS
        self.features = paths.tabular_features
        self.original = OriginalCode()

    def run(self) -> dict[str, np.ndarray]:
        outputs = self._dead_ball()
        outputs.update(self._team())
        layout = np.load(self.features / "team_blocks_b.npz")
        for key in ("start", "end", "half"):                     # the 0.2 s block layout
            outputs[f"block_{key}"] = layout["info_" + key]
        return outputs

    def _dead_ball(self) -> dict[str, np.ndarray]:
        windows = np.load(self.features / "dead_ball_windows.npz")
        columns = read_json(self.features / "dead_ball_windows.json")
        outputs = {"window_start": windows["starts"]}
        for name in DEAD_BALL_MODELS:
            model = DeadBallModel(self.weights / name, self.original)
            outputs[f"p_alive_{name}"] = model.predict(windows["x"], columns["names"], columns["spec"])
            print(f"  {name} done", flush=True)
        return outputs

    def _team(self) -> dict[str, np.ndarray]:
        outputs = {}
        for column_set in ("b", "ab"):
            x = np.load(self.features / f"team_blocks_{column_set}.npz")["x"]
            columns = read_json(self.features / f"team_blocks_{column_set}.json")
            mirror = self.original.team_mirror(columns)
            for rule, folder_name in TEAM_MODELS.items():
                folder = self.weights / folder_name
                trained = read_json(folder / f"features_{column_set}.json")["columns"]
                assert trained == columns["a_names"] + columns["b_names"], \
                    f"{folder_name}: the {column_set} table's columns differ from the training columns"
                for name in (m for m in TEAM_MEMBERS if m.endswith("_" + column_set)):
                    outputs[f"team_{rule}_{name}"] = TeamMember(folder, name).predict(x, mirror)
                    print(f"  {folder_name} {name} done", flush=True)
        return outputs


def main() -> None:
    paths = Paths.from_env()
    outputs = TabularModels(paths).run()
    paths.tabular_outputs.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(paths.tabular_outputs, **outputs)
    print(f"tabular models: written to {paths.tabular_outputs}")


if __name__ == "__main__":
    main()
