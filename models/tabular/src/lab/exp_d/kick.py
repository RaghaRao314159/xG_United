"""Track D: the kick-off rule. "Before a kick-off both teams stand inside their own halves with the
ball at the centre spot; the ball is dead until the kick-off is taken" (the challenge rules, as
context_v3 words them). A setup episode is a run of context_v3.kickoff_setup frames (both teams
wholly in their own halves, >= 14 players visible, ball at the centre spot or untracked), runs
closer than GAP_MERGE merged, at least MIN_SETUP long. The kick is the first frame from KICK_LEAD
before the episode's end to TAKE_WINDOW after it where the tracked ball moves faster than
KICK_SPEED within CENTRE_TAKE of the centre spot; without one, the episode's end. Every frame from
the episode's start to the kick is forced dead (rule K1). Rule K2, the half start: play cannot be
live before a half's kick-off, so every frame from the half's first frame to the half's first
centre kick (tracked ball faster than KICK_SPEED within CENTRE_TAKE of the centre spot, within
HALF_START of the half) is forced dead. `players_only` drops the ball condition from K1's setup
(IDSSE freezes its dead ball off the centre spot, so the ball test cannot be validated there).
No labels are used to build the masks.

    MTAH_ROOT=/root/mtah python kick.py idsse        # precision and spellF1 effect on IDSSE
    MTAH_ROOT=/root/mtah python kick.py bar          # BAR-ATH aggregates of the forced frames

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

sys.path.insert(0, LAB + "/exp")
sys.path.insert(0, ROOT + "/src")  # xG United: folder renamed from jev_track
from common import pipeline, tb  # noqa: E402
import context_v3 as c3  # noqa: E402

pm = tb.pm
FPS = pipeline.FPS
GAP_MERGE = 2 * FPS
MIN_SETUP = FPS
KICK_LEAD = 2 * FPS
TAKE_WINDOW = 3 * FPS
KICK_SPEED = 2.0       # m/s
CENTRE_TAKE = 3.0      # metres
HALF_START = 5 * 60 * FPS


def kick_mask(raw, meta, f, players_only=False):
    """(K1 forced, K2 forced, [(half, start, kick, setup frames, kicked by the ball)], {half: K2
    kick frame or None}) for one dataset."""
    pitch = np.asarray(raw["pitch"], dtype=np.float64)
    ball_m = raw["ball"].astype(np.float64) * pitch
    players_m = raw["players"].astype(np.float64) * pitch
    shown = ~np.isnan(players_m[..., 0])
    team = np.asarray(raw["team"])
    setup, centre = c3.kickoff_setup(players_m, team, shown, ball_m, pitch)
    seen = f["seen"].astype(bool)
    ok = setup if players_only else setup & (centre | ~seen)
    with np.errstate(invalid="ignore"):
        near = np.linalg.norm(ball_m - pitch / 2.0, axis=1) < CENTRE_TAKE
        moving = seen & (np.nan_to_num(f["ball_speed"], nan=0.0) > KICK_SPEED)
    kicked = near & moving
    k1 = np.zeros(len(seen), dtype=bool)
    k2 = np.zeros(len(seen), dtype=bool)
    episodes, half_kick = [], {}
    for half, first, last in meta["halves"]:
        n = last + 1 - first
        hits = np.flatnonzero(kicked[first:first + min(n, HALF_START)])
        half_kick[half] = int(hits[0]) if len(hits) else None
        if len(hits):
            k2[first:first + int(hits[0])] = True
        starts, stops = tb.run_bounds(ok[first:last + 1])
        merged = []
        for a, b in zip(starts.tolist(), stops.tolist()):
            if merged and a - merged[-1][1] < GAP_MERGE:
                merged[-1][1] = b
            else:
                merged.append([a, b])
        for a, b in merged:
            frames = int(ok[first + a:first + b].sum())
            if frames < MIN_SETUP:
                continue
            lo, hi = max(a, b - KICK_LEAD), min(n, b + TAKE_WINDOW)
            hits = np.flatnonzero(kicked[first + lo:first + hi])
            kick = lo + int(hits[0]) if len(hits) else b
            k1[first + a:first + kick] = True
            episodes.append((half, first + a, first + kick, frames, bool(len(hits))))
    return k1, k2, episodes, half_kick


def load_raw(dataset):
    with np.load(os.path.join(pipeline.dataset_dir(dataset), "raw_tb.npz")) as d:
        return {k: d[k] for k in d.files}


def pooled_spell(gts, preds):
    sep = np.full(tb.SEPARATOR, pm.X, dtype=np.int8)
    gt = np.concatenate([p for g in gts for p in (g, sep)][:-1])
    pred = np.concatenate([p for q in preds for p in (q, sep)][:-1])
    pred = np.where(gt == pm.X, pm.X, pred)
    s = pm.spell_scores(gt, pred, pm.valid_runs(gt))
    return s["mean_f1"], s["n_pred"], s["n_gt"]


def idsse():
    for cond in ("clean", "d85s"):
        for players_only in (True, False):
            hit = {"k1": 0, "k2": 0}
            tot = {"k1": 0, "k2": 0}
            n_ep = n_kicked = 0
            kick_s = []
            gts, before, after = [], [], []
            for m in tb.built_matches():
                dataset = tb.dataset_name(m, cond)
                meta, f = pipeline.load_dataset(dataset)
                gt = tb.gt_codes(dataset).astype(np.int8)
                k1, k2, episodes, half_kick = kick_mask(load_raw(dataset), meta, f, players_only)
                graded = gt != pm.X
                for key, mask in (("k1", k1), ("k2", k2)):
                    hit[key] += int((mask & graded & (gt == pm.D)).sum())
                    tot[key] += int((mask & graded).sum())
                n_ep += len(episodes)
                n_kicked += sum(e[4] for e in episodes)
                kick_s += [v / FPS for v in half_kick.values() if v is not None]
                pred = pm.label_codes(pm.read_label_csv(os.path.join(
                    pipeline.dataset_dir(dataset), "results", "exp_ab_b_tbh_p1.csv"))).astype(np.int8)
                gts.append(gt)
                before.append(pred)
                after.append(np.where(k1 & graded, pm.D, pred).astype(np.int8))  # K2 cannot apply: IDSSE halves start at the kick-off
            f0, p0, n_gt = pooled_spell(gts, before)
            f1, p1, _ = pooled_spell(gts, after)
            print("{} {}: K1 {} episodes ({} kicks seen), {} forced frames ({:.1f} s/match), "
                  "precision {:.3f}; K2 kick found in {}/14 halves, median {:.1f} s / max {:.1f} s "
                  "into the half, {} forced frames, precision {:.3f}; exp_ab_b_tbh_p1 spellF1 "
                  "{:.4f} -> {:.4f} (poss {} -> {}, true {})".format(
                      cond, "players-only setup" if players_only else "setup + ball", n_ep,
                      n_kicked, tot["k1"], tot["k1"] / FPS / 7.0, hit["k1"] / max(tot["k1"], 1),
                      len(kick_s), np.median(kick_s) if kick_s else -1,
                      max(kick_s) if kick_s else -1, tot["k2"], hit["k2"] / max(tot["k2"], 1),
                      f0, f1, p0, p1, n_gt), flush=True)


def bar():
    meta, f = pipeline.load_dataset("BAR-ATH")
    graded = np.zeros(meta["n_frames"], dtype=bool)
    for _, first, last in meta["halves"]:
        graded[first:last + 1] = True
    for players_only in (False, True):
        k1, k2, episodes, half_kick = kick_mask(load_raw("BAR-ATH"), meta, f, players_only)
        print("BAR-ATH {}: K1 {} episodes, {} forced graded frames ({:.1%}); K2 kick at {} s "
              "into each half, {} forced frames; 2nd-half pre-kick-off forced: K1 {:.1%}, K2 {:.1%}"
              .format("players-only setup" if players_only else "setup + ball", len(episodes),
                      int((k1 & graded).sum()), np.mean(k1[graded]),
                      {h: (None if v is None else round(v / FPS, 1)) for h, v in half_kick.items()},
                      int((k2 & graded).sum()), np.mean(k1[110808:113742]),
                      np.mean(k2[110808:113742])))
        for half, start, kick, frames, by_ball in episodes:
            first = [fr for h, fr, _ in meta["halves"] if h == half][0]
            print("  half {}: K1 forces {:.1f} s from {:.0f} s into the half, {} setup frames, "
                  "kick {}".format(half, (kick - start) / FPS, (start - first) / FPS, frames,
                                   "seen" if by_ball else "= setup end"))


if __name__ == "__main__":
    {"idsse": idsse, "bar": bar}[sys.argv[1]]()
