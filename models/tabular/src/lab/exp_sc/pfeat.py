"""Idea #2: player-only context columns for P(alive). Nothing here reads the ball, so they mean the
same whether a provider loses the ball in stoppages (d85su, SkillCorner) or keeps it in view through
them (d50su, cleansu), and on BAR-ATH whichever it does. Lang et al. (Sci Rep 2022) reached 92%
frame accuracy on unseen Bundesliga matches from player positions alone.

Per frame, from raw_tb.npz's players (NaN where not shown):
  per team (_h / _a)  mean speed, share walking (< 1.5 m/s), share running (> 4 m/s), share
                      standing (< 0.5 m/s), mean |acceleration|, centroid speed (over 0.5 s),
                      length and width of the shown players, count shown
  everyone            the same kinematics, speed spread, players within 1.5 m of a touchline,
                      within 2 m of a goal line, the most inside one penalty area / one goal area,
                      the distance between the team centroids, and how far the most isolated
                      touchline player stands from anyone else (a throw-in taker)
Per 12-frame window the mean; the kinematic and set-piece stats also as past and future means
over 2, 5, 10, 20 and 30 s and the changes over 5 s either side. Cached per dataset as
pf_cols_w<WINDOW>.npz. Prints nothing.
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
import ctx  # noqa: E402

FPS = pipeline.FPS
LAG = 2
WALK, RUN, STILL = 1.5, 4.0, 0.5     # m/s
TOUCH_M, GOAL_M = 1.5, 2.0           # m from the line
BOX_X, BOX_Y = 16.5, 20.16           # penalty area depth, half width
SIX_X, SIX_Y = 5.5, 9.16
TEAM_STATS = ("spd", "walk", "run", "still", "acc", "cspd", "len", "wid", "n")
CONTEXT = ("spd", "walk", "run", "still", "acc", "sstd", "box", "touch", "gline")
WIDTHS = (2, 5, 10, 20, 30)          # s


def deriv(x, k):
    out = np.full(x.shape, np.nan, dtype=np.float32)
    out[k:-k] = (x[2 * k:] - x[:-2 * k]) / (2.0 * k / FPS)
    return out


def share(mask, valid):
    n = valid.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, (mask & valid).sum(axis=1) / np.maximum(n, 1), np.nan)


def kinematics(p, prefix, out):
    valid = ~np.isnan(p[..., 0])
    v = deriv(p, LAG)
    speed = np.linalg.norm(v, axis=2)
    acc = np.linalg.norm(deriv(v, LAG), axis=2)
    ok = valid & ~np.isnan(speed)
    out[prefix.format("spd")] = np.nanmean(np.where(ok, speed, np.nan), axis=1)
    out[prefix.format("walk")] = share(speed < WALK, ok)
    out[prefix.format("run")] = share(speed > RUN, ok)
    out[prefix.format("still")] = share(speed < STILL, ok)
    out[prefix.format("acc")] = np.nanmean(np.where(ok & ~np.isnan(acc), acc, np.nan), axis=1)
    return speed, ok


def frame_stats(raw):
    length, width = [float(v) for v in raw["pitch"]]
    p = raw["players"].astype(np.float32) * np.array([length, width], dtype=np.float32)
    team = raw["team"]
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        speed, ok = kinematics(p, "pf_{}", out)
        out["pf_sstd"] = np.nanstd(np.where(ok, speed, np.nan), axis=1)
        cents = []
        for side, tag in ((0, "h"), (1, "a")):
            q = p[:, team == side]
            kinematics(q, "pf_{}_" + tag, out)
            c = np.nanmean(q, axis=1)
            cents.append(c)
            out["pf_cspd_" + tag] = np.linalg.norm(deriv(c, 6), axis=1)
            out["pf_len_" + tag] = np.nanmax(q[..., 0], axis=1) - np.nanmin(q[..., 0], axis=1)
            out["pf_wid_" + tag] = np.nanmax(q[..., 1], axis=1) - np.nanmin(q[..., 1], axis=1)
            out["pf_n_" + tag] = (~np.isnan(q[..., 0])).sum(axis=1).astype(np.float32)
        out["pf_cgap"] = np.linalg.norm(cents[0] - cents[1], axis=1)
        x, y = p[..., 0], p[..., 1]
        near_touch = (y < TOUCH_M) | (y > width - TOUCH_M)
        out["pf_touch"] = near_touch.sum(axis=1).astype(np.float32)
        out["pf_gline"] = ((x < GOAL_M) | (x > length - GOAL_M)).sum(axis=1).astype(np.float32)
        mid = np.abs(y - width / 2.0)
        left, right = x < BOX_X, x > length - BOX_X
        out["pf_box"] = np.maximum((left & (mid < BOX_Y)).sum(axis=1),
                                   (right & (mid < BOX_Y)).sum(axis=1)).astype(np.float32)
        out["pf_six"] = np.maximum(((x < SIX_X) & (mid < SIX_Y)).sum(axis=1),
                                   ((x > length - SIX_X) & (mid < SIX_Y)).sum(axis=1)).astype(np.float32)
        iso = np.full(len(p), np.nan, dtype=np.float32)
        rows = np.flatnonzero(near_touch.any(axis=1))
        for i in rows:
            here = p[i][~np.isnan(p[i, :, 0])]
            if len(here) < 2:
                continue
            d = np.linalg.norm(here[:, None, :] - here[None, :, :], axis=2)
            np.fill_diagonal(d, np.inf)
            at = (here[:, 1] < TOUCH_M) | (here[:, 1] > width - TOUCH_M)
            iso[i] = d[at].min(axis=1).max()
        out["pf_touch_iso"] = iso
    return out


def columns(dataset):
    """({name: per-window column}, swap spec) for one dataset, cached beside raw_tb.npz."""
    folder = pipeline.dataset_dir(dataset)
    cache = os.path.join(folder, "pf_cols_w{}.npz".format(tb.WINDOW))
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
            for name, values in frame_stats(raw).items():
                vals, _ = tb.by_second(values, meta, np.nan)
                cols[name] = np.nanmean(vals, axis=1).astype(np.float32)
        half = tb.half_of_windows(meta)
        for stat in CONTEXT:
            now = cols["pf_" + stat]
            for w in WIDTHS:
                k = max(1, int(round(w * FPS / float(tb.WINDOW))))
                cols["pf_{}_p{}".format(stat, w)] = ctx.window_mean(now, half, -k, -1).astype(np.float32)
                cols["pf_{}_f{}".format(stat, w)] = ctx.window_mean(now, half, 1, k).astype(np.float32)
            cols["pf_{}_dp5".format(stat)] = now - cols["pf_{}_p5".format(stat)]
            cols["pf_{}_df5".format(stat)] = cols["pf_{}_f5".format(stat)] - now
        np.savez_compressed(cache, **cols)
    spec = {}
    for name in cols:
        if name.endswith(("_h", "_a")):
            spec[name] = ("pair", name[:-1] + ("a" if name.endswith("_h") else "h"))
        else:
            spec[name] = ("same",)
    return cols, spec
