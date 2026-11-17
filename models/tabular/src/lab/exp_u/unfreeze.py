"""Track U: copies of the IDSSE datasets with the frozen ball replaced by a moving one.

DFL freezes the ball while it is dead (every visible dead frame repeats the previous position
exactly); BAR-ATH's visible ball never stands still. Every run of exactly repeated ball positions
(label-free: live repeats, 1.3% of live frames, are treated the same) gets a synthetic trajectory
instead: an Ornstein-Uhlenbeck velocity (speed scale drawn per run, log-uniform SPEED_LO-SPEED_HI
m/s, time constant TAU s), integrated and bridged so that it starts and ends at the frozen
position, plus JITTER of white tracking noise, rounded to IDSSE's 1 cm grid. Ball height is
dropped everywhere (BAR-ATH has none). features.npz is recomputed exactly as pipeline.frame_features
does (verified to reproduce the originals), the d85 copy hides the ball on exactly the d85 frames.

    runs/idsse-<m>-cleanu/   raw_tb.npz features.npz meta.json gt.csv
    runs/idsse-<m>-d85u/     the same with the d85 hiding

    MTAH_ROOT=/root/mtah python unfreeze.py --calibrate      # jitter vs the live slow ball
    MTAH_ROOT=/root/mtah python unfreeze.py                  # write the copies
Prints aggregates only.
"""
import os as _os
ROOT = _os.environ.get("MTAH_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))))
LAB = _os.environ.get("MTAH_LAB") or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
LABCACHE = _os.path.join(ROOT, "runs", "lab-cache")

import argparse
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, LAB + "/exp")
from common import A, D, H, pipeline, tb  # noqa: E402

FPS = pipeline.FPS
DT = 1.0 / FPS
SPEED_LO, SPEED_HI = 0.3, 2.0      # m/s, per run
TAU = 1.0                          # s, velocity memory
JITTER = float(os.environ.get("U_JITTER", 0.0075))   # m, scale of the tracking noise per axis
NOISE_DF = float(os.environ.get("U_DF", 0))          # Student-t degrees of freedom, 0 = normal
GRID = 0.01                        # m: IDSSE positions are whole centimetres
SEED = 20261030
GEN = os.environ.get("U_GEN", "v2")   # v1: one OU drift per run; v2: regime-switching segments
SEG_MEAN = 1.5                     # s, mean segment length (v2)
STILL_SHARE = 0.35                 # share of segments that are near-still (v2)
CONDS = ("d85",)


def load_raw(dataset):
    with np.load(os.path.join(pipeline.dataset_dir(dataset), "raw_tb.npz")) as d:
        return {k: d[k] for k in d.files}


def recompute(ball, raw):
    players, team, pitch = raw["players"], raw["team"], raw["pitch"]
    shown = ~np.isnan(players[..., 0])
    return pipeline.frame_features(ball, players, pipeline.position_speed(players, pitch), team,
                                   shown, pitch)


def frozen_runs(b, seen):
    rep = np.zeros(len(b), dtype=bool)
    rep[1:] = seen[1:] & seen[:-1] & (b[1:, 0] == b[:-1, 0]) & (b[1:, 1] == b[:-1, 1])
    frozen = rep.copy()
    frozen[:-1] |= rep[1:]
    edges = np.diff(np.concatenate(([0], frozen.astype(np.int8), [0])))
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


def velocities_v2(length, rng):
    """OU velocities whose speed scale and memory change from segment to segment: near-still,
    carried and rolled stretches, as a dead ball is placed, picked up and moved."""
    v = np.empty((length, 2))
    cur = np.zeros(2)
    t = 0
    while t < length:
        seg = max(5, int(rng.exponential(SEG_MEAN) * FPS))
        if rng.uniform() < STILL_SHARE:
            speed = rng.uniform(0.02, 0.15)
        else:
            speed = np.exp(rng.uniform(np.log(0.3), np.log(2.5)))
        tau = rng.uniform(0.15, 0.8)
        a = np.exp(-DT / tau)
        sigma = speed / np.sqrt(2.0)
        stop = min(length, t + seg)
        noise = rng.normal(0.0, sigma * np.sqrt(1 - a * a), (stop - t, 2))
        for i in range(stop - t):
            cur = a * cur + noise[i]
            v[t + i] = cur
        t = stop
    return v


def synthetic(length, rng, jitter):
    """(length, 2) metre offsets: bridged drift plus jitter."""
    if GEN == "v2":
        v = velocities_v2(length, rng)
        x = np.cumsum(v * DT, axis=0)
        x -= x[0]
        if length > 1:
            x -= np.linspace(0.0, 1.0, length)[:, None] * x[-1]
        if NOISE_DF > 0:
            return x + jitter * rng.standard_t(NOISE_DF, (length, 2))
        return x + rng.normal(0.0, jitter, (length, 2))
    speed = np.exp(rng.uniform(np.log(SPEED_LO), np.log(SPEED_HI)))
    sigma = speed / np.sqrt(2.0)
    a = np.exp(-DT / TAU)
    v = np.empty((length, 2))
    v[0] = rng.normal(0.0, sigma, 2)
    noise = rng.normal(0.0, sigma * np.sqrt(1 - a * a), (length, 2))
    for t in range(1, length):
        v[t] = a * v[t - 1] + noise[t]
    x = np.cumsum(v * DT, axis=0)
    x -= x[0]
    if length > 1:
        x -= np.linspace(0.0, 1.0, length)[:, None] * x[-1]
    if NOISE_DF > 0:
        return x + jitter * rng.standard_t(NOISE_DF, (length, 2))
    return x + rng.normal(0.0, jitter, (length, 2))


def unfreeze(raw, rng, jitter):
    pitch = raw["pitch"].astype(np.float64)
    b = raw["ball"].astype(np.float64) * pitch
    seen = ~np.isnan(b[:, 0])
    starts, stops = frozen_runs(b, seen)
    out = b.copy()
    for s, e in zip(starts, stops):
        out[s:e] = np.round((b[s] + synthetic(e - s, rng, jitter)) / GRID) * GRID
    changed = np.zeros(len(b), dtype=bool)
    for s, e in zip(starts, stops):
        changed[s:e] = True
    return (out / pitch).astype(np.float32), changed


def bend_cm(ball_norm, pitch):
    b = ball_norm.astype(np.float64) * pitch
    out = np.full(len(b), np.nan)
    out[1:-1] = np.linalg.norm(b[2:] - 2 * b[1:-1] + b[:-2], axis=1) * 100.0
    return out


def calibrate(matches):
    q = lambda x: np.percentile(x, [10, 50, 90]).round(2).tolist()  # noqa: E731
    for jitter in [float(j) for j in os.environ.get("U_GRID", "0.0,0.005,0.0075,0.01,0.0125").split(",")]:
        dead_b, live_b = [], []
        for m in matches[:3]:
            ds = tb.dataset_name(m, "clean")
            raw = load_raw(ds)
            meta, f = pipeline.load_dataset(ds)
            gt = tb.gt_codes(ds)
            ball, changed = unfreeze(raw, np.random.default_rng(SEED), jitter)
            pitch = raw["pitch"].astype(np.float64)
            bend = bend_cm(ball, pitch)
            g = recompute(ball, raw)
            speed = np.asarray(g["ball_speed"], dtype=np.float64)
            ok = np.isfinite(bend) & np.isfinite(speed)
            dead_b.append(bend[ok & (gt == D) & changed])
            live_b.append(bend[ok & ((gt == H) | (gt == A)) & ~changed & (speed < 2.0)])
        print("jitter {:.4f} m: bend cm/frame^2 p10/50/90 synthetic dead {} vs live slow {}".format(
            jitter, q(np.concatenate(dead_b)), q(np.concatenate(live_b))), flush=True)


def write(dataset, source, raw, ball, feats):
    folder = pipeline.dataset_dir(dataset)
    os.makedirs(folder, exist_ok=True)
    for name in ("meta.json", "gt.csv"):
        shutil.copyfile(os.path.join(pipeline.dataset_dir(source), name),
                        os.path.join(folder, name))
    np.savez_compressed(os.path.join(folder, "features.npz"), **feats)
    np.savez_compressed(os.path.join(folder, "raw_tb.npz"), ball=ball,
                        ball_z=np.full(len(ball), np.nan, dtype=np.float32),
                        players=raw["players"], team=raw["team"], pitch=raw["pitch"])
    for name in os.listdir(folder):            # stale column caches from an earlier build
        if name.startswith("tb_cols_"):
            os.remove(os.path.join(folder, name))


def build(matches, jitter):
    for k, m in enumerate(matches):
        src = tb.dataset_name(m, "clean")
        raw = load_raw(src)
        meta, f = pipeline.load_dataset(src)
        gt = tb.gt_codes(src)
        ball, changed = unfreeze(raw, np.random.default_rng(SEED + k), jitter)
        g = recompute(ball, raw)
        assert (g["seen"] == f["seen"]).all() and (g["gap"] == f["gap"]).all()
        write(src.replace("idsse-" + m, "idsse-{}-cleanu".format(m)), src, raw, ball, g)
        seen = f["seen"]
        dead, live = (gt == D) & seen, ((gt == H) | (gt == A)) & seen
        z_before = lambda s, mask: float(np.mean(np.asarray(s)[mask] == 0))  # noqa: E731
        print("{} cleanu: changed {:.1%} of dead / {:.2%} of live visible frames; ball speed "
              "exactly 0 on dead {:.1%} -> {:.2%}, live {:.2%} -> {:.2%}; median dead speed "
              "{:.2f} m/s".format(m, changed[dead].mean(), changed[live].mean(),
                                  z_before(f["ball_speed"], dead), z_before(g["ball_speed"], dead),
                                  z_before(f["ball_speed"], live), z_before(g["ball_speed"], live),
                                  float(np.nanmedian(np.asarray(g["ball_speed"])[dead]))),
              flush=True)
        for cond in CONDS:
            other = tb.dataset_name(m, cond)
            _, fc = pipeline.load_dataset(other)
            hidden = seen & ~fc["seen"]
            ball_c = ball.copy()
            ball_c[hidden] = np.nan
            gc = recompute(ball_c, raw)
            assert (gc["seen"] == fc["seen"]).all() and (gc["gap"] == fc["gap"]).all()
            write("idsse-{}-{}u".format(m, cond), other, raw, ball_c, gc)
        with open(os.path.join(pipeline.dataset_dir("idsse-{}-cleanu".format(m)),
                               "unfreeze.json"), "w") as handle:
            json.dump({"speed": [SPEED_LO, SPEED_HI], "tau": TAU, "jitter": jitter, "df": NOISE_DF, "gen": GEN,
                       "grid": GRID, "seed": SEED + k}, handle)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibrate", action="store_true")
    args = parser.parse_args()
    matches = tb.built_matches()
    if args.calibrate:
        calibrate(matches)
    else:
        build(matches, JITTER)
        print("done", flush=True)


if __name__ == "__main__":
    main()
