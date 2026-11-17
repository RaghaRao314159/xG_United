"""A realistic visible ball during IDSSE stoppages: carried, passed and placed by the real players.

Why: the IDSSE-trained P(alive) recalls 0.98 of dead windows with the ball in view on IDSSE (d85suc, and
even d50suc / cleansuc), but 0.61 on SkillCorner's real broadcast ball (in-domain SkillCorner: 0.85).
ball_stats.py shows why. DFL freezes the dead ball, and unfreeze.py replaces the frozen ball with a
slow random drift around the spot where play stopped, away from everyone (nearest player median 3.9 m,
a player within 1 m on 9% of frames, "carried" on 6%). A real dead ball is picked up, carried to the
restart, thrown or kicked to the taker and placed (SkillCorner: nearest player 2.0 m, within 1 m on 28%,
carried on 25%, median speed 1.3 m/s vs 0.4). So the model learns "slow ball far from players = dead"
and calls a carried or placed ball alive.

Generator (physical, parameters set by hand, not fitted to SkillCorner): for every dead run of the
clean truth, starting from the position where play stopped (b0, with its last live velocity) and ending
at the real restart position (b1, the first live frame after the run):
  roll      the ball keeps its last velocity and slows (time constant 0.5 s) for 0.3-1.5 s
  holders   1-3 players in turn (40/40/20%): the nearest player to the ball first, a random player within
            25 m next, the taker last (the player nearest the restart spot at the restart). The ball reaches
            each by a pass (6-15 m/s) when he is more than 3 m away, else a 0.5 s pick-up, and is then
            carried 0.3-0.7 m from him along his real track; a non-last holder may instead leave it lying
            still (25%)
  place     half the runs end with the ball moved to b1 and left still there for 0.5-4 s (a set piece);
            the others blend into b1 over the last 0.4 s (a throw-in from the hands)
  noise     3 cm of white tracking noise per axis
Live frames keep cleanu's ball. Then, exactly as the other conditions are built:
  crsu    nothing hidden, smoothed (sigma 6), features.npz recomputed
  r85su   the ball hidden on d85su's frames (85% of the hidden time in stoppages), then smoothed
plus the controlled-truth copies crsuc / r85suc (conds.link_controlled).

    MTAH_ROOT=/root/mtah python carry.py
Prints aggregates only.
"""
import os as _os
ROOT = _os.environ.get("MTAH_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))))
LAB = _os.environ.get("MTAH_LAB") or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

import os
import shutil
import sys

import numpy as np

sys.path.insert(0, LAB + "/exp")
sys.path.insert(0, LAB + "/exp_t")
sys.path.insert(0, LAB + "/exp_u")
sys.path.insert(0, LAB + "/exp_dball")
from common import pipeline, tb  # noqa: E402
import smooth  # noqa: E402  (Track T)
import unfreeze  # noqa: E402
import conds  # noqa: E402

FPS = pipeline.FPS
DT = 1.0 / FPS
SIGMA = 6.0
SEED = 20261107
JITTER = 0.03


def nearest(pf, point):
    d = np.linalg.norm(pf - point, axis=1)
    if not np.isfinite(d).any():
        return None
    return int(np.nanargmin(d))


def simulate(b0, v0, b1, P, rng):
    """(T, 2) metres. P: (T + 1, players, 2) metres over the run and the restart frame."""
    T = len(P) - 1
    out = np.empty((T, 2))
    if T < 10:
        return b0 + (b1 - b0) * (np.arange(1, T + 1) / (T + 1.0))[:, None]
    x = b0.astype(np.float64).copy()
    i = 0
    v = np.where(np.isfinite(v0), v0, 0.0).astype(np.float64)
    sp = np.linalg.norm(v)
    if sp > 20.0:
        v *= 20.0 / sp
    for _ in range(min(int(rng.uniform(0.3, 1.5) * FPS), T // 4)):
        v *= np.exp(-DT / 0.5)
        x = x + v * DT
        out[i] = x
        i += 1
    n_place = int(rng.uniform(0.5, 4.0) * FPS) if rng.uniform() < 0.5 else 0
    n_place = min(n_place, (T - i) // 3)
    n_mid = T - i - n_place
    taker = nearest(P[T], b1)
    if taker is None:
        taker = nearest(P[T - 1], b1)
    h = int(rng.choice([1, 2, 3], p=[0.4, 0.4, 0.2]))
    lens = np.floor(rng.dirichlet(np.ones(h)) * n_mid).astype(int)
    lens[-1] = n_mid - lens[:-1].sum()
    for j, n in enumerate(lens):
        if n <= 0:
            continue
        last = j == h - 1
        if last:
            who = taker
        elif j == 0:
            who = nearest(P[i], x)
        else:
            d = np.linalg.norm(P[i] - x, axis=1)
            near = np.flatnonzero(np.isfinite(d) & (d < 25.0))
            who = int(rng.choice(near)) if len(near) else nearest(P[i], x)
        still = (not last) and rng.uniform() < 0.25
        stop = i + n
        if who is None or still:
            out[i:stop] = x
            i = stop
            continue
        ang = rng.uniform(0, 2 * np.pi)
        off_len = rng.uniform(0.3, 0.7)
        # transfer: a pass when the holder is far, a pick-up when near
        here = P[i, who]
        if np.isfinite(here).all():
            dist = np.linalg.norm(here - x)
            n_t = int(dist / rng.uniform(6.0, 15.0) * FPS) + 1 if dist > 3.0 else int(0.5 * FPS)
            n_t = max(1, min(n_t, n))
            arrive = P[min(i + n_t, T), who]
            if not np.isfinite(arrive).all():
                arrive = here
            target = arrive + off_len * np.array([np.cos(ang), np.sin(ang)])
            for q in range(1, n_t + 1):
                out[i] = x + (target - x) * (q / float(n_t))
                i += 1
            x = out[i - 1].copy()
        while i < stop:
            hp = P[i, who]
            ang += rng.normal(0.0, 0.05)
            if np.isfinite(hp).all():
                x = hp + off_len * np.array([np.cos(ang), np.sin(ang)])
            out[i] = x
            i += 1
    if n_place > 0:
        n_move = min(int(0.5 * FPS), n_place)
        for q in range(1, n_move + 1):
            out[i] = x + (b1 - x) * (q / float(n_move))
            i += 1
        out[i:T] = b1
    else:
        n_b = min(10, T)
        ramp = (np.arange(1, n_b + 1) / float(n_b))[:, None]
        out[T - n_b:T] = out[T - n_b:T] + (b1 - out[T - 1]) * ramp
    return out + rng.normal(0.0, JITTER, out.shape)


def realistic_ball(m):
    """cleanu's ball (normalised) with every dead run replaced; returns it with the raw dict."""
    clean_ds, src = tb.dataset_name(m, "clean"), tb.dataset_name(m, "cleanu")
    raw = smooth.load_raw(src)
    frozen = smooth.load_raw(clean_ds)["ball"]          # IDSSE's own ball: the frozen dead position
    meta, _ = pipeline.load_dataset(src)
    gt = tb.gt_codes(clean_ds)
    scale = np.array([float(v) for v in raw["pitch"]])
    b = raw["ball"].astype(np.float64) * scale
    bf = frozen.astype(np.float64) * scale
    P = raw["players"].astype(np.float64) * scale
    rng = np.random.default_rng([SEED, tb.MATCHES.index(m)])
    n_runs = 0
    for _, first, last in meta["halves"]:
        dead = np.zeros(len(gt), dtype=bool)
        dead[first:last + 1] = gt[first:last + 1] == tb.D
        starts, stops = tb.run_bounds(dead)
        for s, e in zip(starts, stops):
            b0 = bf[s] if np.isfinite(bf[s]).all() else b[s]
            if not np.isfinite(b0).all():
                continue
            v0 = (bf[s - 1] - bf[s - 3]) / (2 * DT) if s - 3 >= first else np.zeros(2)
            b1 = None
            for f in range(e, min(e + 2 * FPS, last + 1)):
                if np.isfinite(b[f]).all():
                    b1 = b[f]
                    break
            end = min(e, last)
            if b1 is None:
                b1 = b0
            sim = simulate(b0, v0, b1, P[s:end + 1] if end + 1 - s == e - s + 1 else
                           np.concatenate([P[s:e], P[e - 1:e]]), rng)
            b[s:e] = sim
            n_runs += 1
    return (b / scale).astype(np.float32), raw, meta, n_runs


def write(dst, src_ds, meta, raw, ball):
    g = unfreeze.recompute(ball, raw)
    folder = pipeline.dataset_dir(dst)
    os.makedirs(os.path.join(folder, "results"), exist_ok=True)
    pipeline.write_dataset(dst, dict(meta, name=dst), g)
    shutil.copyfile(os.path.join(pipeline.dataset_dir(src_ds), "gt.csv"), os.path.join(folder, "gt.csv"))
    np.savez_compressed(os.path.join(folder, "raw_tb.npz"), ball=ball,
                        ball_z=np.full(len(ball), np.nan, dtype=np.float32),
                        players=raw["players"], team=raw["team"], pitch=raw["pitch"])
    return g


def main():
    matches = tb.built_matches()
    for m in matches:
        ball, raw, meta, n_runs = realistic_ball(m)
        crsu = smooth.smooth_ball(ball.copy(), SIGMA).astype(np.float32)
        write(tb.dataset_name(m, "crsu"), tb.dataset_name(m, "cleanu"), meta, raw, crsu)
        _, f85 = pipeline.load_dataset(tb.dataset_name(m, "d85su"))
        hidden = ~np.asarray(f85["seen"], dtype=bool)
        b85 = ball.copy()
        b85[hidden] = np.nan
        r85 = smooth.smooth_ball(b85, SIGMA).astype(np.float32)
        g = write(tb.dataset_name(m, "r85su"), tb.dataset_name(m, "cleanu"), meta, raw, r85)
        assert (np.asarray(g["seen"]) == np.asarray(f85["seen"])).all(), "r85su hiding differs from d85su"
        print("  {}: {} dead runs re-simulated; crsu and r85su written".format(m, n_runs), flush=True)
    conds.link_controlled(("crsu", "r85su"), matches)
    print("done", flush=True)


if __name__ == "__main__":
    main()
