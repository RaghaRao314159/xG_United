"""Track B per-second columns from raw positions (runs/idsse-<m>[-<cond>]/raw_tb.npz).

Per frame, in metres and m/s (NaN where the ball is hidden or a player is off the pitch; the
IDSSE frozen dead ball, |v| == 0 exactly, counts as unknown rather than as a cue):
  ball      velocity vector, acceleration magnitude, heading change over +-3 frames, height,
            position and velocity along the home attacking direction
  touches   local maxima of acceleration above TOUCH_ACC with a player within TOUCH_DIST; the
            touching team, the last and the next touch team and the time since / until them
  per team  2nd-nearest distance, players within 3/5/10 m, nearest player's approach speed,
            closest player to the ball's line of travel (the receiver lane), players goal-side
            of the ball, centroid progression from its own goal, centroid speed toward the
            opponent goal, spread along / across, mean player speed
Per one-second window (pipeline.windows order) these become medians / means / maxima, plus a few
of them at +-1, 2, 3, 5 s. Every team-specific column comes as a home/away pair; columns along
the home attacking direction negate or flip under a home/away swap. columns() returns the arrays
and a swap spec {name: ("pair", other) | ("neg",) | ("inv",) | ("same",)}.
Prints aggregates only.
"""
import os as _os
ROOT = _os.environ.get("MTAH_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))))
LAB = _os.environ.get("MTAH_LAB") or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
LABCACHE = _os.path.join(ROOT, "runs", "lab-cache")

import os

import sys

import numpy as np

sys.path.insert(0, LAB + "/exp")  # common.py, when imported on its own

from common import H, A, pipeline, tb

FPS = pipeline.FPS
LAG = 2
TURN_LAG = 3
TOUCH_ACC = float(os.environ.get("TB_TOUCH_ACC", 30.0))   # m/s^2
TOUCH_DIST = 2.0         # m
FLIGHT_SPEED = 3.0       # m/s: the ball counts as travelling, for the receiver lane
LANE_MAX = 40.0          # m ahead of the ball
TIME_CAP = 30.0          # s
CONTEXT_OFFSETS = (-5, -3, -2, -1, 1, 2, 3, 5)
CONTEXT_STATS = ("touch", "lane", "cnt5", "goalside", "cx", "cv", "approach")


def deriv(x, k):
    out = np.full(x.shape, np.nan, dtype=np.float32)
    out[k:-k] = (x[2 * k:] - x[:-2 * k]) / (2.0 * k / FPS)
    return out


def shift_frames(x, k):
    out = np.full(x.shape, np.nan, dtype=np.float32)
    if k > 0:
        out[:-k] = x[k:]
    elif k < 0:
        out[-k:] = x[:k]
    else:
        out[:] = x
    return out


def attack_dir(players, team, meta, n, length):
    """+1 where home attacks toward x = length, -1 the other way, NaN outside the halves."""
    out = np.full(n, np.nan, dtype=np.float32)
    for _, first, last in meta["halves"]:
        xs = players[first:last + 1, :, 0]
        home_x = np.nanmean(xs[:, team == 0])
        away_x = np.nanmean(xs[:, team == 1])
        out[first:last + 1] = 1.0 if home_x < away_x else -1.0
    return out


def fill_state(event_team, event_mask, meta, n, backward):
    """(team code +1 home / -1 away / NaN, seconds since or until) of the last or next event,
    within each half."""
    team = np.full(n, np.nan, dtype=np.float32)
    since = np.full(n, np.nan, dtype=np.float32)
    for _, first, last in meta["halves"]:
        idx = np.arange(first, last + 1)
        if backward:
            idx = idx[::-1]
        current, at = np.nan, None
        for t in idx:
            if event_mask[t]:
                current, at = event_team[t], t
            team[t] = current
            if at is not None:
                since[t] = min(abs(t - at) / FPS, TIME_CAP)
    return team, since


def frame_features(raw, meta):
    length, width = [float(v) for v in raw["pitch"]]
    scale = np.array([length, width], dtype=np.float32)
    b = raw["ball"].astype(np.float32) * scale
    p = raw["players"].astype(np.float32) * scale
    team = raw["team"]
    n = len(b)
    home, away = team == 0, team == 1
    seen = ~np.isnan(b[:, 0])

    v = deriv(b, LAG)
    spd = np.linalg.norm(v, axis=1)
    frozen = spd == 0
    v[frozen] = np.nan
    spd[frozen] = np.nan
    acc = np.linalg.norm(deriv(v, LAG), axis=1)
    v1, v2 = shift_frames(v, -TURN_LAG), shift_frames(v, TURN_LAG)
    n1, n2 = np.linalg.norm(v1, axis=1), np.linalg.norm(v2, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = (v1 * v2).sum(axis=1) / (n1 * n2)
        turn = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    turn[~((n1 > 2) & (n2 > 2))] = np.nan

    rel = p - b[:, None, :]
    dist = np.linalg.norm(rel, axis=2)
    pv = deriv(p, LAG)
    pspeed = np.linalg.norm(pv, axis=2)
    ddist = deriv(dist, LAG)

    out = {"ball_speed": spd, "acc": acc, "turn": turn, "z": raw["ball_z"].astype(np.float32)}
    direction = attack_dir(p, team, meta, n, length)
    out["ball_adv"] = np.where(direction > 0, b[:, 0] / length, 1 - b[:, 0] / length)
    out["v_adv"] = v[:, 0] * direction

    with np.errstate(invalid="ignore"):
        u = v / spd[:, None]
        along = (rel * u[:, None, :]).sum(axis=2)
        perp = np.linalg.norm(rel - along[..., None] * u[:, None, :], axis=2)
        lane_ok = (along > 0) & (along < LANE_MAX) & (spd[:, None] > FLIGHT_SPEED)

    filled = np.where(np.isnan(dist), np.inf, dist)
    nearest_any = filled.argmin(axis=1)
    near_d = filled[np.arange(n), nearest_any]
    for side, mask in (("h", home), ("a", away)):
        ds = np.sort(np.where(np.isnan(dist[:, mask]), np.inf, dist[:, mask]), axis=1)
        d2 = ds[:, 1].copy()
        d2[~np.isfinite(d2) | ~seen] = np.nan
        out["d2_" + side] = d2
        for r in (3, 5, 10):
            c = (ds < r).sum(axis=1).astype(np.float32)
            c[~seen] = np.nan
            out["cnt{}_{}".format(r, side)] = c
        sub = np.where(np.isnan(dist[:, mask]), np.inf, dist[:, mask])
        idx = sub.argmin(axis=1)
        rows = np.arange(n)
        appr = -ddist[:, mask][rows, idx]
        appr[~seen] = np.nan
        out["approach_" + side] = appr
        lane = np.where(lane_ok[:, mask], perp[:, mask], np.inf).min(axis=1)
        lane[~np.isfinite(lane)] = np.nan
        out["lane_" + side] = lane
        xs = p[:, mask, 0]
        sign = direction if side == "h" else -direction
        own_goal_side = np.where(sign[:, None] > 0, xs < b[:, 0:1], xs > b[:, 0:1])
        gs = own_goal_side.sum(axis=1).astype(np.float32)
        gs[~seen] = np.nan
        out["goalside_" + side] = gs
        with np.errstate(invalid="ignore"):
            cx_m = np.nanmean(xs, axis=1)
        prog = np.where(sign > 0, cx_m / length, 1 - cx_m / length)
        out["cx_" + side] = prog.astype(np.float32)
        out["cv_" + side] = deriv((prog * length).astype(np.float32), 12)
        with np.errstate(invalid="ignore"):
            out["sx_" + side] = np.nanstd(xs, axis=1)
            out["sy_" + side] = np.nanstd(p[:, mask, 1], axis=1)
            out["pspeed_" + side] = np.nanmean(pspeed[:, mask], axis=1)

    peak = acc >= np.fmax(shift_frames(acc, -1), shift_frames(acc, 1))
    touch = seen & np.isfinite(acc) & (acc > TOUCH_ACC) & peak & (near_d < TOUCH_DIST)
    touch_team = np.where(team[nearest_any] == 0, 1.0, -1.0).astype(np.float32)
    out["touch_h"] = (touch & (touch_team > 0)).astype(np.float32)
    out["touch_a"] = (touch & (touch_team < 0)).astype(np.float32)
    out["last_team"], out["last_since"] = fill_state(touch_team, touch, meta, n, backward=False)
    out["next_team"], out["next_until"] = fill_state(touch_team, touch, meta, n, backward=True)
    return out, touch


def nanmean_rows(values):
    with np.errstate(invalid="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return np.nanmean(values, axis=1)


def nanmax_rows(values):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmax(values, axis=1)


PAIRED = ("d2", "cnt3", "cnt5", "cnt10", "approach", "lane", "goalside", "cx", "cv", "sx", "sy",
          "pspeed", "touch")


def columns(dataset):
    """({name: per-window column}, swap spec) for one dataset, cached beside raw_tb.npz."""
    folder = pipeline.dataset_dir(dataset)
    cache = os.path.join(folder, "tb_cols_{:g}_w{}.npz".format(TOUCH_ACC, tb.WINDOW))
    meta, _ = pipeline.load_dataset(dataset)
    if os.path.exists(cache):
        with np.load(cache, allow_pickle=False) as d:
            cols = {k: d[k] for k in d.files}
    else:
        with np.load(os.path.join(folder, "raw_tb.npz")) as d:
            raw = {k: d[k] for k in d.files}
        f, _ = frame_features(raw, meta)
        cols = {}
        ends = np.array([stop - 1 for _, _, _, stop in pipeline.windows(meta)])
        starts = np.array([start for _, _, start, _ in pipeline.windows(meta)])
        for key, how in (("ball_speed", "med"), ("acc", "max"), ("turn", "max"), ("z", "max"),
                         ("ball_adv", "med"), ("v_adv", "med")):
            vals, _ = tb.by_second(f[key], meta, np.nan)
            cols["tb_" + key] = (tb.nanmedian_rows(vals) if how == "med" else nanmax_rows(vals))
        for stat in PAIRED:
            for side in ("h", "a"):
                vals, _ = tb.by_second(f["{}_{}".format(stat, side)], meta, np.nan)
                if stat == "touch":
                    col = np.nansum(vals, axis=1)
                elif stat == "lane":
                    col = tb.nanmedian_rows(vals)
                else:
                    col = nanmean_rows(vals)
                cols["tb_{}_{}".format(stat, side)] = col
        cols["tb_last_team"] = f["last_team"][ends]
        cols["tb_last_since"] = f["last_since"][ends]
        cols["tb_next_team"] = f["next_team"][starts]
        cols["tb_next_until"] = f["next_until"][starts]
        cols = {k: np.asarray(v, dtype=np.float32) for k, v in cols.items()}
        np.savez_compressed(cache, **cols)

    half = tb.half_of_windows(meta)
    out = dict(cols)
    for stat in CONTEXT_STATS:
        for offset in CONTEXT_OFFSETS:
            for side in ("h", "a"):
                out["tb_{}_{}@{:+d}".format(stat, side, offset)] = tb.shifted(
                    cols["tb_{}_{}".format(stat, side)], half, offset)
    for key in ("tb_last_team", "tb_next_team"):
        for offset in CONTEXT_OFFSETS:
            out["{}@{:+d}".format(key, offset)] = tb.shifted(cols[key], half, offset)
    return out, spec_of(out)


def spec_of(cols):
    spec = {}
    for name in cols:
        base, _, offset = name.partition("@")
        if base.endswith(("_h", "_a")) and base[:-2].split("tb_", 1)[-1] in PAIRED:
            other = base[:-1] + ("a" if base.endswith("_h") else "h")
            spec[name] = ("pair", other + ("@" + offset if offset else ""))
        elif base in ("tb_v_adv", "tb_last_team", "tb_next_team"):
            spec[name] = ("neg",)
        elif base == "tb_ball_adv":
            spec[name] = ("inv",)
        else:
            spec[name] = ("same",)
    return spec


def touch_stats(dataset):
    """Aggregate check of the touch detector against the truth: touches per alive minute and how
    often the touching team is the team in possession a moment later."""
    meta, _ = pipeline.load_dataset(dataset)
    with np.load(os.path.join(pipeline.dataset_dir(dataset), "raw_tb.npz")) as d:
        raw = {k: d[k] for k in d.files}
    f, touch = frame_features(raw, meta)
    gt = tb.gt_codes(dataset)
    alive = (gt == H) | (gt == A)
    team = np.where(f["touch_h"] > 0, H, A)
    later = np.clip(np.flatnonzero(touch) + 5, 0, len(gt) - 1)
    agree = (gt[later] == team[touch])
    ok = alive[later]
    acc = f["acc"][alive & np.isfinite(f["acc"])]
    return {"touches_per_alive_min": touch.sum() / (alive.sum() / FPS / 60.0),
            "touch_team_matches_possession_0.2s_later": float(agree[ok].mean()),
            "acc_p50_p90_p99": np.percentile(acc, [50, 90, 99]).round(1).tolist()}
