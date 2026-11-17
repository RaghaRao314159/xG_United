"""BAR-ATH raw_tb.npz in Track B's layout, read from the tactical ATD exactly as pipeline.cmd_features
reads it: ball (n,2) and players (n,K,2) on the 0-1 scale, players NaN where pipeline.on_screen
says the slot is not in the tactical image, team (K,) 0 home / 1 away, pitch (2,) metres, ball_z
all NaN (the ATD ball's 3rd channel is unusable and the models never read it). Asserts alignment
with runs/BAR-ATH/meta.json and features.npz. Prints aggregates only.

    MTAH_ROOT=/root/mtah python build_bar.py
"""
import os as _os
ROOT = _os.environ.get("MTAH_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))))
LAB = _os.environ.get("MTAH_LAB") or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
LABCACHE = _os.path.join(ROOT, "runs", "lab-cache")

import os
import sys
import time

import numpy as np

sys.path.insert(0, ROOT + "/src")  # xG United: folder renamed from jev_track
import pipeline  # noqa: E402
import baseline  # noqa: E402
import load_homography  # noqa: E402
from games import GAME, GAMES, require, source_frames  # noqa: E402


def main():
    started = time.time()
    meta_path, raw_path = require(["atd_meta", "atd_raw"])
    n = source_frames("tactical")
    ball, players, _ = pipeline.parse_atd(raw_path, n)
    team = np.asarray(baseline.team_of_each_track(meta_path), dtype=np.int8)
    assert players.shape[1] == len(team)
    matrices, _, valid = load_homography.load()
    shown = pipeline.on_screen(players, matrices, valid)
    meta, f = pipeline.load_dataset(GAME)
    assert meta["n_frames"] == n == len(ball) == len(players)
    assert (f["seen"] == ~np.isnan(ball[:, 0])).all(), "ball differs from features.npz"
    assert (f["shown_count"] == shown.sum(axis=1)).all(), "on-screen mask differs"
    p = players.astype(np.float32).copy()
    p[~shown] = np.nan
    out = {"ball": ball.astype(np.float32), "ball_z": np.full(n, np.nan, dtype=np.float32),
           "players": p, "team": team,
           "pitch": np.asarray(GAMES[GAME]["pitch"], dtype=np.float32)}
    np.savez_compressed(os.path.join(pipeline.dataset_dir(GAME), "raw_tb.npz"), **out)

    graded = np.zeros(n, dtype=bool)
    for _, first, last in meta["halves"]:
        graded[first:last + 1] = True
    home_shown = shown[:, team == 0].sum(axis=1)
    away_shown = shown[:, team == 1].sum(axis=1)
    print("raw_tb: {} frames, {} slots ({} home / {} away), pitch {}".format(
        n, len(team), int((team == 0).sum()), int((team == 1).sum()), out["pitch"].tolist()))
    print("  graded frames {}, ball seen on {:.1f}%".format(int(graded.sum()),
                                                         100 * f["seen"][graded].mean()))
    print("  median players on screen per graded frame: {:.0f} (home {:.0f}, away {:.0f}); "
          "share of graded frames with >= 18 shown {:.1%}".format(
              np.median(shown[graded].sum(axis=1)), np.median(home_shown[graded]),
              np.median(away_shown[graded]), np.mean(shown[graded].sum(axis=1) >= 18)))
    print("  {:.0f} s".format(time.time() - started))


if __name__ == "__main__":
    main()
