"""Stage 4: Tiya's two-head BiLSTM.

Her network (models/bilstm/inference/run_inference.py, unchanged) reads 30 "schema-3"
tracking columns per frame and gives P(home), P(away), P(dead) per frame. Her handoff did not
include the step that builds those columns from the tracking, so `Schema3Features` rebuilds it.
That rebuild is checked the only way that counts: fed to her own code, it reproduces her frozen
BAR-ATH label files on every frame.

The network is run twice: on her inputs as they are, and with the home and away columns
exchanged (then home and away exchanged back in the answer). The ensemble uses the exchanged run
for team evidence and her own decoder on the plain run as a dead-ball opinion.

Output: work/bilstm/probabilities.npz with `raw` and `swapped`, each (frames, 3) =
P(H), P(A), P(D).
"""
from __future__ import annotations

import importlib.util
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from main_pipeline.config import BILSTM_CODE, BILSTM_WEIGHTS, N_FRAMES, Paths, torch_device

FPS = 25.0
PITCH_M = (105.0, 68.0)    # her inputs are in metres on a 105 x 68 pitch (tracking is 0-1)
BATCH_WINDOWS = 64
# Her column order is 6 ball columns, 3 home distances, 3 away distances, 9 home summaries and
# 9 away summaries; this permutation exchanges home and away.
HOME_AWAY_SWAP = np.r_[0:6, 9:12, 6:9, 21:30, 12:21]


def load_her_code():
    """Tiya's run_inference.py as a module (network class, batching and decoder)."""
    spec = importlib.util.spec_from_file_location("tiya_run_inference", BILSTM_CODE / "run_inference.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Schema3Features:
    """Her 30 input columns per frame, built from the tactical tracking (EPTS text + metadata)."""

    def __init__(self, data: Path) -> None:
        self.raw_file = data / "tactical" / "BAR-ATH_tactical_atd.txt"
        self.meta_file = data / "tactical" / "BAR-ATH_tactical_atd.xml"

    def build(self) -> np.ndarray:
        """(frames, 30) float32 in her column order, NaN where unknown."""
        ball, players = self._read_tracking()
        team = self._team_of_slots()
        scale = np.array(PITCH_M)
        b, p = ball.astype(np.float64) * scale, players.astype(np.float64) * scale
        v = self._backward_difference(b)
        columns = [b[:, 0], b[:, 1], v[:, 0], v[:, 1], np.hypot(v[:, 0], v[:, 1]),
                   (~np.isnan(b[:, 0])).astype(np.float64)]
        with np.errstate(invalid="ignore"):
            distance = np.sqrt(((p - b[:, None, :]) ** 2).sum(-1))
        for side in (0, 1):                       # 3 nearest home, then away players to the ball
            nearest = np.sort(distance[:, team == side], axis=1)    # NaN sorts last
            columns += [nearest[:, 0], nearest[:, 1], nearest[:, 2]]
        for side in (0, 1):                       # team summaries: home, then away
            columns += self._team_summary(p[:, team == side])
        return np.column_stack(columns).astype(np.float32)

    def _read_tracking(self) -> tuple[np.ndarray, np.ndarray]:
        """Ball (frames, 2) and players (frames, slots, 2) on the 0-1 scale, NaN where untracked.
        Each line is `frame_id:slot;slot;...:ball`, a slot `x,y,speed` or empty / NaN."""
        ball = np.full((N_FRAMES, 2), np.nan, dtype=np.float32)
        players = None
        with open(self.raw_file) as handle:
            for line in handle:
                parts = line.rstrip("\r\n").split(":")
                if len(parts) != 3:
                    continue
                row = int(parts[0]) - 1               # frame ids count from 1
                slots = parts[1].split(";")
                if players is None:
                    players = np.full((N_FRAMES, len(slots), 2), np.nan, dtype=np.float32)
                for slot, group in enumerate(slots):
                    if group and group[0] != "N":
                        fields = group.split(",")
                        players[row, slot] = (float(fields[0]), float(fields[1]))
                if parts[2] and parts[2][0] != "N":
                    fields = parts[2].split(",")
                    ball[row] = (float(fields[0]), float(fields[1]))
        return ball, players

    def _team_of_slots(self) -> np.ndarray:
        """0 for a home track slot, 1 for away; the first team in the metadata is the home side."""
        root = ET.parse(self.meta_file).getroot()
        home_id = next(root.iter("Team")).get("id")
        return np.array([0 if p.get("teamId") == home_id else 1 for p in root.iter("Player")],
                        dtype=np.int8)

    @classmethod
    def _team_summary(cls, q: np.ndarray) -> list[np.ndarray]:
        """Tracked players, centroid x/y, spread x/y (population std), centroid velocity x/y,
        and the median and 75th percentile of the players' speeds."""
        count = (~np.isnan(q[..., 0])).sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            centroid = np.nansum(q, axis=1) / count[:, None]
            spread = np.sqrt(np.nansum((q - centroid[:, None, :]) ** 2, axis=1) / count[:, None])
        velocity = cls._backward_difference(centroid)
        speed = np.linalg.norm(cls._backward_difference(q), axis=2)
        return [count.astype(np.float64), centroid[:, 0], centroid[:, 1], spread[:, 0], spread[:, 1],
                velocity[:, 0], velocity[:, 1], cls._row_quantile(speed, 0.5), cls._row_quantile(speed, 0.75)]

    @staticmethod
    def _backward_difference(f: np.ndarray) -> np.ndarray:
        """Per-second change from the previous frame (NaN on the first frame)."""
        out = np.full(f.shape, np.nan)
        out[1:] = (f[1:] - f[:-1]) * FPS
        return out

    @staticmethod
    def _row_quantile(a: np.ndarray, q: float) -> np.ndarray:
        """`np.nanquantile(a, q, axis=1)` with linear interpolation, NaN for all-NaN rows."""
        ordered = np.sort(a, axis=1)                          # NaN sorts last
        n = (~np.isnan(a)).sum(1)
        position = np.maximum(n - 1, 0) * q
        low = np.floor(position).astype(int)
        high = np.minimum(low + 1, np.maximum(n - 1, 0))
        rows = np.arange(len(a))
        value = ordered[rows, low] * (1 - (position - low)) + ordered[rows, high] * (position - low)
        return np.where(n > 0, value, np.nan)


class TiyaNetwork:
    """Her trained network, on a CUDA GPU (default) or the CPU, over the whole match as one
    sequence."""

    def __init__(self, weights_file: Path) -> None:
        import torch
        self.device = torch_device()
        self.code = load_her_code()
        checkpoint = torch.load(weights_file, map_location=self.device, weights_only=True)
        self.model = self.code.TwoHeadLSTM(hidden=checkpoint["config"]["hidden"]).to(self.device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.mean = np.asarray(checkpoint["mean"], dtype=np.float32)
        self.scale = np.asarray(checkpoint["scale"], dtype=np.float32)
        self.spans = [(0, N_FRAMES)]      # her inputs ran the whole file as one span

    def probabilities(self, features: np.ndarray) -> np.ndarray:
        """(frames, 3): P(H), P(A), P(D)."""
        return self.code.predict(self.model, features, self.spans, self.mean, self.scale,
                                 self.device, BATCH_WINDOWS)


def main() -> None:
    paths = Paths.from_env()
    features = Schema3Features(paths.data).build()
    network = TiyaNetwork(BILSTM_WEIGHTS)
    raw = network.probabilities(features)
    swapped = network.probabilities(features[:, HOME_AWAY_SWAP])[:, [1, 0, 2]]
    paths.bilstm_outputs.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(paths.bilstm_outputs, raw=raw.astype(np.float32), swapped=swapped.astype(np.float32))
    print(f"bilstm: written to {paths.bilstm_outputs}")


if __name__ == "__main__":
    main()
