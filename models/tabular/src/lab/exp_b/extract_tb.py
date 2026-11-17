"""Raw IDSSE positions for Track B, frame-aligned with the existing runs/idsse-<m>/ datasets.

For each match: fetch the two XMLs (kloppy, KLOPPY_CACHE_DIR), read them exactly as
jev_track/idsse.py does (same unusable-frame handling, layout and SEPARATOR), add the ball height,
check the alignment against the dataset already built (n_frames, halves, seen, d_home, labels) and
write

    runs/idsse-<m>/raw_tb.npz      ball (n,2) 0-1, ball_z (n,), players (n,K,2) 0-1, team (K,),
                                   pitch (2,) metres
    runs/idsse-<m>-d85/raw_tb.npz  the same with the ball hidden exactly where the d85 copy hides it

then delete the cached XMLs. One match per process call keeps kloppy's ~7 GB peak to one match.
Prints aggregates only.

    MTAH_ROOT=/root/mtah KLOPPY_CACHE_DIR=/root/kloppy_cache python extract_tb.py J03WMX
"""
import os as _os
ROOT = _os.environ.get("MTAH_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))))
LAB = _os.environ.get("MTAH_LAB") or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
LABCACHE = _os.path.join(ROOT, "runs", "lab-cache")

import gc
import glob
import json
import os
import shutil
import sys
import time

import numpy as np

ROOT = os.environ.get("MTAH_ROOT", ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))  # xG United: folder renamed from jev_track
import idsse  # noqa: E402
import pipeline  # noqa: E402
import possession_metrics as pm  # noqa: E402

CONDS = ("d85",)


def ball_height(dataset, blocks, unusable, n_rows):
    z = np.full(n_rows, np.nan, dtype=np.float32)
    for frame in dataset.records:
        first_row, first_number, _ = blocks[frame.period.id]
        row = first_row + frame.frame_id - first_number
        c = frame.ball_coordinates
        bad = unusable.get((frame.period.id, frame.frame_id), ())
        if c is not None and "X" not in bad and "Y" not in bad:
            value = getattr(c, "z", None)
            if value is not None:
                z[row] = value
    return z


def main(match):
    started = time.time()
    meta_x, raw_x = idsse.fetch(match)
    unusable = idsse.unusable_ball_frames(raw_x)
    dataset = idsse.load(meta_x, raw_x, None)
    data = idsse.tracking_arrays(dataset, unusable)
    n_rows = len(data["alive"])
    z = ball_height(dataset, data["blocks"], unusable, n_rows)
    del dataset
    gc.collect()

    folder = pipeline.dataset_dir("idsse-" + match)
    with open(os.path.join(folder, "meta.json")) as handle:
        meta = json.load(handle)
    assert n_rows == meta["n_frames"], (n_rows, meta["n_frames"])
    halves = [[period, first, first + n - 1]
              for period, (first, _, n) in sorted(data["blocks"].items())]
    assert halves == meta["halves"], (halves, meta["halves"])
    assert list(data["pitch"]) == list(meta["pitch"])
    with np.load(os.path.join(folder, "features.npz")) as old:
        old = {k: old[k] for k in ("seen", "d_home", "nearest")}
    ball, players, team, pitch = data["ball"], data["players"], data["team"], data["pitch"]
    seen = ~np.isnan(ball[:, 0])
    assert (seen == old["seen"]).all(), "seen differs"
    shown = ~np.isnan(players[..., 0])
    feats = pipeline.frame_features(ball, players, pipeline.position_speed(players, pitch), team,
                                    shown, pitch)
    assert np.allclose(feats["d_home"], old["d_home"], equal_nan=True), "d_home differs"
    assert (feats["nearest"] == old["nearest"]).all(), "nearest differs"
    labels = pm.label_codes(idsse.labels_of(data["alive"], data["owner"]))
    gt = pm.label_codes(pm.read_label_csv(os.path.join(folder, "gt.csv")))
    assert (labels == gt).all(), "labels differ"

    np.savez_compressed(os.path.join(folder, "raw_tb.npz"), ball=ball, ball_z=z, players=players,
                        team=team, pitch=np.array(pitch, dtype=np.float32))
    for cond in CONDS:
        other = pipeline.dataset_dir("idsse-{}-{}".format(match, cond))
        with np.load(os.path.join(other, "features.npz")) as d:
            seen_c = d["seen"]
        assert len(seen_c) == n_rows and not (seen_c & ~seen).any()
        hidden = seen & ~seen_c
        ball_c, z_c = ball.copy(), z.copy()
        ball_c[hidden] = np.nan
        z_c[hidden] = np.nan
        np.savez_compressed(os.path.join(other, "raw_tb.npz"), ball=ball_c, ball_z=z_c,
                            players=players, team=team, pitch=np.array(pitch, dtype=np.float32))
        print("  {} {}: {:.1%} of ball frames hidden".format(match, cond, hidden.sum() / seen.sum()))
    zs = z[seen & ~np.isnan(z)]
    print("{}: {} frames, {} player slots, ball z known on {:.1%} of seen frames (median {:.2f}, "
          "p99 {:.2f}), all alignment checks passed, {:.0f} s".format(
              match, n_rows, players.shape[1], len(zs) / max(1, seen.sum()),
              np.median(zs) if len(zs) else float("nan"),
              np.percentile(zs, 99) if len(zs) else float("nan"), time.time() - started),
          flush=True)
    cache = os.environ.get("KLOPPY_CACHE_DIR")
    if cache:
        for path in glob.glob(os.path.join(cache, "*")):
            shutil.rmtree(path) if os.path.isdir(path) else os.remove(path)


if __name__ == "__main__":
    main(sys.argv[1])
