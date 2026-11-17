"""Track A: a 5 Hz team model P(H | alive) from frame-level control and touch features, stacked.

Every half is cut into 0.2 s blocks (5 frames). A block's row holds
    block stats       ball seen share, gap, median ball speed (exact zeros dropped: IDSSE freezes
                      the dead ball), median distance of each team's nearest player, minimum
                      distance, share of frames each team "controls" the ball (its nearest player
                      within r m: r = 1 and 1.5 with the other team 1 m further away, and 2.5),
                      touches per team (ball speed changing by > 2 / 4 m/s over 4 frames with a
                      player within 2 m), the nearest-player share
    event timing      seconds since each team's last control / touch and until its next, and which
                      team had the last and has the next one
    hysteresis teams  control by r m that only flips after k frames of the other team, run forward
                      and backward in time
    window counts     control frames and touches per team over 0.5-4 s before and after
    context           a few block stats at +-1, 2, 3, 5 blocks
    the second's row  tabular.rows() of the second the block lies in (78 columns)
A LightGBM team model is trained on alive blocks of the training matches (both orientations),
then a second pass sees the first pass's out-of-fold P(H | alive) at offsets up to 20 s; first-
pass answers for training match t of fold m come from the model trained without m and t.

    ta5.py --cond clean      writes cache/<cond>_<match>.npz: rows' meta, p1, p2
Prints aggregates only.
"""
import os as _os
ROOT = _os.environ.get("MTAH_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))))
LAB = _os.environ.get("MTAH_LAB") or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
LABCACHE = _os.path.join(ROOT, "runs", "lab-cache")

import argparse
import itertools
import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, LAB + "/exp")
from common import A, D, H, pipeline, tb  # noqa: E402

pm = tb.pm
X = pm.X
FPS = pipeline.FPS
BLOCK = 5
CAP = 30.0
CACHE = LABCACHE + "/exp_a"
CTX_OFFSETS = (-5, -3, -2, -1, 1, 2, 3, 5)
CTX_STATS = ("b_d_h", "b_d_a", "b_c1_h", "b_c1_a", "b_seen")
CTL_WINDOWS = (12, 25, 50, 100)        # frames
TOUCH_WINDOWS = (25, 50)
STACK_OFFSETS = (1, 2, 3, 5, 8, 12, 25, 50, 100)
STACK_MEANS = (5, 15, 50)
LGBM = dict(tb.LGBM, n_jobs=4)


def hysteresis(team, k):
    out = np.full(len(team), -1, dtype=np.int8)
    cur, run_team, run = -1, -1, 0
    for i, t in enumerate(team.tolist()):
        if t >= 0:
            if t == run_team:
                run += 1
            else:
                run_team, run = t, 1
            if cur < 0 or (t != cur and run >= k):
                cur = t
        out[i] = cur
    return out


def since_last(event):
    """Frames since the last True at or before each frame (inf if none)."""
    idx = np.where(event, np.arange(len(event)), -1)
    last = np.maximum.accumulate(idx)
    return np.where(last >= 0, np.arange(len(event)) - last, np.inf)


def until_next(event):
    n = len(event)
    idx = np.where(event, np.arange(n), n + 10 ** 9)
    nxt = np.minimum.accumulate(idx[::-1])[::-1]
    return np.where(nxt < n, nxt - np.arange(n), np.inf)


def blocks_of(values, n_blocks, fill):
    pad = np.full(n_blocks * BLOCK, fill, dtype=np.float64)
    pad[:len(values)] = values
    return pad.reshape(n_blocks, BLOCK)


def window_count(event, starts, ends, lo, hi):
    """Events in [starts+lo, starts) when lo < 0, in [ends, ends+hi) when hi > 0."""
    c = np.concatenate([[0], np.cumsum(event)])
    n = len(event)
    if lo < 0:
        a, b = np.clip(starts + lo, 0, n), np.clip(starts, 0, n)
    else:
        a, b = np.clip(ends, 0, n), np.clip(ends + hi, 0, n)
    return (c[b] - c[a]).astype(np.float64)


def team_code(value):
    return np.where(value == H, 1.0, np.where(value == A, 0.0, 0.5))


def half_rows(f, first, last):
    """{name: column} over the blocks of one half, plus the blocks' first frames."""
    s = slice(first, last + 1)
    n = last + 1 - first
    nb = -(-n // BLOCK)
    starts = np.arange(nb) * BLOCK
    ends = np.minimum(starts + BLOCK, n)
    seen = f["seen"][s]
    dh = f["d_home"][s].astype(np.float64)
    da = f["d_away"][s].astype(np.float64)
    dh[~seen] = np.nan
    da[~seen] = np.nan
    near = np.where(seen, f["nearest"][s], -1).astype(np.int8)
    bs = f["ball_speed"][s].astype(np.float64)
    bs[(~seen) | (bs == 0)] = np.nan
    with np.errstate(invalid="ignore"):
        dmin, dmax = np.fmin(dh, da), np.fmax(dh, da)
        change = np.zeros(n)
        change[2:-2] = np.abs(bs[4:] - bs[:-4])
        change = np.nan_to_num(change, nan=0.0)
        ev = {}
        for tag, r in (("c1", 1.0), ("c15", 1.5)):
            base = seen & (dmin < r) & ((dmax - dmin) > 1.0)
            ev[tag + "_h"], ev[tag + "_a"] = base & (near == H), base & (near == A)
        loose = seen & (dmin < 2.5)
        ev["c25_h"], ev["c25_a"] = loose & (near == H), loose & (near == A)
        for tag, t in (("t2", 2.0), ("t4", 4.0)):
            base = seen & (dmin < 2.0) & (change > t)
            ev[tag + "_h"], ev[tag + "_a"] = base & (near == H), base & (near == A)

    out = {}
    seen_b = blocks_of(seen, nb, 0)
    out["b_seen"] = seen_b.mean(axis=1)
    out["b_gap"] = np.minimum(f["gap"][s][ends - 1] / float(FPS), 60.0)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        out["b_speed"] = np.nanmedian(blocks_of(bs, nb, np.nan), axis=1)
        out["b_d_h"] = np.nanmedian(blocks_of(dh, nb, np.nan), axis=1)
        out["b_d_a"] = np.nanmedian(blocks_of(da, nb, np.nan), axis=1)
        out["b_dmin"] = np.nanmin(blocks_of(dmin, nb, np.nan), axis=1)
        known = blocks_of(near >= 0, nb, 0).sum(axis=1)
        out["f_near"] = np.where(known > 0, blocks_of(near == H, nb, 0).sum(axis=1)
                                 / np.maximum(known, 1), np.nan)
    for tag in ("c1", "c15", "c25"):
        for side in ("_h", "_a"):
            out["b_" + tag + side] = blocks_of(ev[tag + side], nb, 0).mean(axis=1)
    for tag in ("t2", "t4"):
        for side in ("_h", "_a"):
            out["b_" + tag + side] = blocks_of(ev[tag + side], nb, 0).sum(axis=1)

    for tag in ("c1", "c15", "t4"):
        sl = {side: since_last(ev[tag + side])[ends - 1] for side in ("_h", "_a")}
        un = {side: until_next(ev[tag + side])[starts] for side in ("_h", "_a")}
        for side in ("_h", "_a"):
            out["since_" + tag + side] = np.minimum(sl[side] / FPS, CAP)
            out["until_" + tag + side] = np.minimum(un[side] / FPS, CAP)
        out["f_last_" + tag] = np.where(sl["_h"] < sl["_a"], 1.0,
                                        np.where(sl["_a"] < sl["_h"], 0.0, 0.5))
        out["f_next_" + tag] = np.where(un["_h"] < un["_a"], 1.0,
                                        np.where(un["_a"] < un["_h"], 0.0, 0.5))

    mid = np.minimum(starts + BLOCK // 2, n - 1)
    for r, k in ((1.0, 3), (1.5, 6)):
        team = np.where(seen & (dmin < r), near, -1).astype(np.int8)
        fwd = hysteresis(team, k)
        rev = hysteresis(team[::-1], k)[::-1]
        tag = "{}_{}".format(int(r * 10), k)
        out["f_hystf" + tag] = team_code(fwd[mid])
        out["f_hystr" + tag] = team_code(rev[mid])

    for w in CTL_WINDOWS:
        for side in ("_h", "_a"):
            out["w_c1_p{}{}".format(w, side)] = window_count(ev["c1" + side], starts, ends, -w, 0)
            out["w_c1_f{}{}".format(w, side)] = window_count(ev["c1" + side], starts, ends, 0, w)
    for w in TOUCH_WINDOWS:
        for side in ("_h", "_a"):
            out["w_t4_p{}{}".format(w, side)] = window_count(ev["t4" + side], starts, ends, -w, 0)
            out["w_t4_f{}{}".format(w, side)] = window_count(ev["t4" + side], starts, ends, 0, w)

    for off in CTX_OFFSETS:
        for stat in CTX_STATS:
            col = np.full(nb, np.nan)
            src = np.arange(nb) + off
            ok = (src >= 0) & (src < nb)
            col[ok] = out[stat][src[ok]]
            out["o{:+d}_{}".format(off, stat)] = col
    return out, first + starts, first + ends


def match_table(match, cond):
    """(matrix, names, block meta) of one match: 5 Hz columns, then the second's tabular row."""
    dataset = tb.dataset_name(match, cond)
    meta, f = pipeline.load_dataset(dataset)
    gt = tb.gt_codes(dataset).astype(np.int8)
    tab = tb.rows(f, meta)
    wins = pipeline.windows(meta)
    win_index = {start: i for i, (_, _, start, _) in enumerate(wins)}
    cols, b_start, b_end, b_half, b_win = None, [], [], [], []
    for half, first, last in meta["halves"]:
        out, starts, ends = half_rows(f, first, last)
        cols = {k: [v] for k, v in out.items()} if cols is None else \
            {k: cols[k] + [out[k]] for k in cols}
        b_start.append(starts)
        b_end.append(ends)
        b_half.append(np.full(len(starts), half))
        b_win.append(np.array([win_index[first + ((s - first) // FPS) * FPS] for s in starts]))
    names5 = list(cols)
    m5 = np.column_stack([np.concatenate(cols[k]) for k in names5])
    b_start, b_end = np.concatenate(b_start), np.concatenate(b_end)
    b_win = np.concatenate(b_win)
    matrix = np.hstack([m5, tab[b_win]]).astype(np.float32)
    names = names5 + ["s_" + n for n in tb.feature_names()]
    counts = np.zeros((len(b_start), 3))
    for k, code in enumerate((H, A, D)):
        hit = (gt == code).astype(np.int64)
        c = np.concatenate([[0], np.cumsum(hit)])
        counts[:, k] = c[b_end] - c[b_start]
    return matrix, names, {"start": b_start, "end": b_end, "half": np.concatenate(b_half),
                           "win": b_win, "counts": counts, "n_frames": meta["n_frames"]}


def make_swap(names):
    tab_names = tb.feature_names()
    tab_idx = [i for i, n in enumerate(names) if n.startswith("s_")]
    index = {n: i for i, n in enumerate(names)}
    pairs, flips = [], []
    for i, n in enumerate(names):
        if n.startswith("s_"):
            continue
        if n.endswith("_h"):
            pairs.append((i, index[n[:-2] + "_a"]))
        elif n.endswith("_a"):
            assert n[:-2] + "_h" in index
        elif n.startswith("f_") or "_f_" in n:
            flips.append(i)

    def swap(m):
        out = m.copy()
        for i, j in pairs:
            out[:, i], out[:, j] = m[:, j], m[:, i]
        for i in flips:
            out[:, i] = 1.0 - m[:, i]
        out[:, tab_idx] = tb.swap(m[:, tab_idx], tab_names)
        return out
    return swap


def team_target(counts):
    alive = counts[:, 0] + counts[:, 1]
    total = counts.sum(axis=1)
    keep = (alive > 0) & (alive >= 0.5 * total) & (counts[:, 0] != counts[:, 1])
    return keep, (counts[:, 0] > counts[:, 1]).astype(np.int8)


def fit_team(tables, train, tests, swap):
    """P(H | alive) for every block of each test match, trained on the train matches."""
    import lightgbm
    xs, ys = [], []
    for m in train:
        x, counts = tables[m][0], tables[m][2]["counts"]
        keep, y = team_target(counts)
        xs.append(x[keep])
        ys.append(y[keep])
    x = np.vstack(xs)
    y = np.concatenate(ys)
    both_x = np.vstack([x, swap(x)])
    both_y = np.concatenate([y, 1 - y])
    clf = lightgbm.LGBMClassifier(random_state=tb.SEED, **LGBM).fit(both_x, both_y)
    out = {}
    for m in tests:
        t = tables[m][0]
        p = clf.predict_proba(np.vstack([t, swap(t)]))[:, 1]
        out[m] = 0.5 * (p[:len(t)] + 1.0 - p[len(t):])
    return out


def stack_columns(p, half):
    cols = {"f_p1_o0": p}
    for off in STACK_OFFSETS:
        for sign in (-1, 1):
            col = np.full(len(p), np.nan)
            src = np.arange(len(p)) + sign * off
            ok = (src >= 0) & (src < len(p))
            ok[ok] = half[src[ok]] == half[ok]
            col[ok] = p[src[ok]]
            cols["f_p1_o{:+d}".format(sign * off)] = col
    for w in STACK_MEANS:
        for tag, lo, hi in (("p", -w, -1), ("f", 1, w)):
            cols["f_p1_m{}{}".format(tag, w)] = window_mean(p, half, lo, hi)
    return cols


def window_mean(values, half, lo, hi):
    out = np.full(len(values), np.nan)
    for h in np.unique(half):
        idx = np.flatnonzero(half == h)
        v = values[idx]
        s = np.concatenate([[0.0], np.cumsum(v)])
        t = np.arange(len(idx))
        a = np.clip(t + lo, 0, len(idx))
        b = np.clip(t + hi + 1, 0, len(idx))
        n = b - a
        out[idx] = np.where(n > 0, (s[b] - s[a]) / np.maximum(n, 1), np.nan)
    return out


def team_acc(p, counts):
    keep, y = team_target(counts)
    return float(np.mean((p[keep] > 0.5) == (y[keep] == 1)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cond", default="clean")
    parser.add_argument("--matches", type=int, default=0, help="debug: only this many matches")
    parser.add_argument("--tag", default="", help="cache name suffix for a parameter variant")
    parser.add_argument("--trees", type=int, default=LGBM["n_estimators"])
    parser.add_argument("--leaves", type=int, default=LGBM["num_leaves"])
    args = parser.parse_args()
    LGBM.update(n_estimators=args.trees, num_leaves=args.leaves)
    tag = "_" + args.tag if args.tag else ""
    os.makedirs(CACHE, exist_ok=True)
    matches = tb.built_matches()
    if args.matches:
        matches = matches[:args.matches]
    started = time.time()
    tables = {}
    for m in matches:
        x, names, info = match_table(m, args.cond)
        tables[m] = (x, names, info)
    swap = make_swap(names)
    probe = tables[matches[0]][0][:2000]
    ok = np.allclose(np.nan_to_num(swap(swap(probe)), nan=-9), np.nan_to_num(probe, nan=-9))
    print("lgbm trees {} leaves {}".format(LGBM["n_estimators"], LGBM["num_leaves"]))
    print("{}: {} matches, {} columns, {} blocks in all, swap involution {}, {:.0f} s".format(
        args.cond, len(matches), len(names), sum(len(t[0]) for t in tables.values()), ok,
        time.time() - started), flush=True)
    assert ok

    started = time.time()
    first = {}
    for pair in itertools.combinations(matches, 2):
        train = [m for m in matches if m not in pair]
        got = fit_team(tables, train, list(pair), swap)
        for m in pair:
            first[frozenset(pair), m] = got[m]
    print("pass 1: {} pair fits, {:.0f} s".format(len(first) // 2, time.time() - started),
          flush=True)

    stack_names = None
    for held in matches:
        started = time.time()
        train = [m for m in matches if m != held]
        p1 = fit_team(tables, train, [held], swap)[held]
        stacked = {}
        for m in matches:
            p = p1 if m == held else first[frozenset((held, m)), m]
            cols = stack_columns(p, tables[m][2]["half"])
            stack_names = list(cols)
            extra = np.column_stack([cols[k] for k in stack_names]).astype(np.float32)
            stacked[m] = (np.hstack([tables[m][0], extra]), None, tables[m][2])
        swap2 = make_swap(names + stack_names)
        p2 = fit_team(stacked, train, [held], swap2)[held]
        info = tables[held][2]
        np.savez_compressed(os.path.join(CACHE, "{}{}_{}.npz".format(args.cond, tag, held)),
                            p1=p1, p2=p2, **{k: v for k, v in info.items() if k != "n_frames"},
                            n_frames=info["n_frames"])
        print("  {} held out: team acc|alive blocks pass 1 {:.3f}, pass 2 {:.3f}, {:.0f} s".format(
            held, team_acc(p1, info["counts"]), team_acc(p2, info["counts"]),
            time.time() - started), flush=True)
    print("done")


if __name__ == "__main__":
    main()
