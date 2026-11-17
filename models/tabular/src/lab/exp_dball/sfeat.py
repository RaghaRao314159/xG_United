"""Stoppage columns for P(alive) that do not need the ball to go missing (hand-off idea 2).

The IDSSE-trained P(alive) leans on "ball missing = stoppage" (exp_sc/palive.py's blackout geometry,
gap, seen). pfeat.py adds player-only context (speeds, walking shares, set-piece geometry over
2-30 s). What neither has is the *event clock* of a stoppage with the ball in view: how long ago the
ball crossed a line, how long ago it last moved fast, how long it has been slow, and how soon it
will move fast again (the restart kick). Per frame, from raw_tb.npz (metres, m/s):

  ball     beyond a line by > OUT_M; seen moving > FAST; seen and slower than SLOW; seen and
           still (< STILL); signed distance to the nearest line (negative outside)
  players  median player speed above CROWD_FAST (collective running)
  clocks   seconds since / until the last / next frame of each event inside the half (capped at
           CAP); the run length so far / still to come of a visible slow ball (capped at RUNCAP)

Per 12-frame window (palive.py's rows) the mean over the window's frames, plus the share of frames
with the ball still and beyond a line, and the window's maximum ball speed. Every column is
team-agnostic (swap rule "same"). Cached per dataset as sf_cols_w<WINDOW>.npz. Prints nothing.
"""
import os as _os
ROOT = _os.environ.get("MTAH_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))))
LAB = _os.environ.get("MTAH_LAB") or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
LABCACHE = _os.path.join(ROOT, "runs", "lab-cache")

import os
import sys
import warnings

import numpy as np

sys.path.insert(0, LAB + "/exp")
from common import pipeline, tb  # noqa: E402

FPS = pipeline.FPS
LAG = 2
OUT_M = 0.5          # m beyond a touch / goal line
FAST = 8.0           # m/s: a pass, shot or clearance
SLOW = 3.0           # m/s: a carried, rolled or placed ball
STILL = 0.5          # m/s
CROWD_FAST = 2.0     # m/s median player speed (stoppages ~1.0, open play ~1.7)
CAP = 60.0           # s
RUNCAP = 120.0       # s
VERSION = 1


def deriv(x, k):
    out = np.full(x.shape, np.nan, dtype=np.float32)
    out[k:-k] = (x[2 * k:] - x[:-2 * k]) / (2.0 * k / FPS)
    return out


def since(mask, halves, n, cap):
    """Seconds since the last True at or before each frame inside its half (cap when none)."""
    out = np.full(n, cap, dtype=np.float32)
    for _, first, last in halves:
        m = mask[first:last + 1]
        idx = np.arange(len(m))
        pos = np.maximum.accumulate(np.where(m, idx, -1))
        out[first:last + 1] = np.where(pos >= 0, np.minimum((idx - pos) / FPS, cap), cap)
    return out


def until(mask, halves, n, cap):
    """Seconds until the next True at or after each frame inside its half (cap when none)."""
    out = np.full(n, cap, dtype=np.float32)
    for _, first, last in halves:
        m = mask[first:last + 1][::-1]
        idx = np.arange(len(m))
        pos = np.maximum.accumulate(np.where(m, idx, -1))
        val = np.where(pos >= 0, np.minimum((idx - pos) / FPS, cap), cap)
        out[first:last + 1] = val[::-1]
    return out


def frame_stats(raw, meta):
    length, width = [float(v) for v in raw["pitch"]]
    scale = np.array([length, width], dtype=np.float32)
    b = raw["ball"].astype(np.float32) * scale
    p = raw["players"].astype(np.float32) * scale
    n = len(b)
    halves = meta["halves"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        seen = ~np.isnan(b[:, 0])
        bspeed = np.linalg.norm(deriv(b, LAG), axis=1)
        known = seen & ~np.isnan(bspeed)
        x, y = b[:, 0], b[:, 1]
        line = np.minimum(np.minimum(x, length - x), np.minimum(y, width - y))
        out_ = seen & (line < -OUT_M)
        fast = known & (bspeed > FAST)
        slow = known & (bspeed < SLOW)
        still = known & (bspeed < STILL)
        pspeed = np.linalg.norm(deriv(p, LAG), axis=2)
        crowd = np.nanmedian(pspeed, axis=1)
        cfast = ~np.isnan(crowd) & (crowd > CROWD_FAST)
    out = {
        "sf_since_out": since(out_, halves, n, CAP),
        "sf_until_out": until(out_, halves, n, CAP),
        "sf_since_fast": since(fast, halves, n, CAP),
        "sf_until_fast": until(fast, halves, n, CAP),
        "sf_since_cfast": since(cfast, halves, n, CAP),
        "sf_until_cfast": until(cfast, halves, n, CAP),
        # a visible slow ball: how long it has been slow, how long it will stay slow (0 otherwise)
        "sf_slow_back": np.where(slow, since(~slow, halves, n, RUNCAP), 0.0).astype(np.float32),
        "sf_slow_fwd": np.where(slow, until(~slow, halves, n, RUNCAP), 0.0).astype(np.float32),
        "sf_line": np.where(seen, line, np.nan).astype(np.float32),
        "sf_out": np.where(seen, out_, np.nan).astype(np.float32),
        "sf_still": np.where(known, still, np.nan).astype(np.float32),
        "sf_bspeed": np.where(known, bspeed, np.nan).astype(np.float32),
    }
    return out


def columns(dataset):
    """({name: per-window column}, swap spec) for one dataset, cached beside raw_tb.npz."""
    folder = pipeline.dataset_dir(dataset)
    cache = os.path.join(folder, "sf_cols_w{}_v{}.npz".format(tb.WINDOW, VERSION))
    meta, _ = pipeline.load_dataset(dataset)
    if os.path.exists(cache):
        with np.load(cache) as d:
            cols = {k: d[k] for k in d.files}
    else:
        with np.load(os.path.join(folder, "raw_tb.npz")) as d:
            raw = {k: d[k] for k in d.files}
        cols = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            for name, values in frame_stats(raw, meta).items():
                vals, _ = tb.by_second(values, meta, np.nan)
                if name == "sf_bspeed":
                    cols["sf_bspeed_max"] = np.nanmax(vals, axis=1).astype(np.float32)
                cols[name] = np.nanmean(vals, axis=1).astype(np.float32)
        np.savez_compressed(cache, **cols)
    return cols, {name: ("same",) for name in cols}
