"""Track AB: Track A's block team model P(H | alive) with Track B's raw-position columns added.

Variants (columns of each block row):
    ab        Track A's 0.2 s block columns (control / touch / timing / hysteresis / counts /
              context + the second's 78-column tabular row) + Track B's columns at 5-frame
              windows (ball vector, touches, last / next touch team, pressure counts, team shape)
              with B's context at +-1, 2, 3, 5 blocks plus +-10, 25 blocks (2 s, 5 s)
    b         the tabular row + Track B's 5-frame columns, no Track A block columns
    ab_half   ab at 12-frame blocks (0.48 s), B's columns at 12-frame windows
Ball height (tb_z) is left out: BAR-ATH has no usable ball height.
LOMO over the 7 IDSSE matches, both orientations, one pass (pass 1); --stack adds Track A's
out-of-fold second pass (pair fits, so no row ever sees a model trained on its own match).
Writes cache/<cond>_<variant>_<held>.npz with the blocks' meta and p1 [, p2].
Prints aggregates only.

    MTAH_ROOT=/root/mtah python ab_fit.py --cond clean --variant ab --trees 800 --leaves 63
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

import numpy as np

sys.path.insert(0, LAB + "/exp")
sys.path.insert(0, LAB + "/exp_a")
sys.path.insert(0, LAB + "/exp_b")
from common import pipeline, tb  # noqa: E402
import ta5  # noqa: E402
import feat_tb  # noqa: E402

CACHE = os.environ.get("U_CACHE", LABCACHE + "/exp_u")
EXTRA_OFFSETS = (-25, -10, 10, 25)
DROP_B = ("tb_z",)


def b_columns(dataset, window):
    """Track B's per-window columns at `window` frames, with extra longer context, and their
    swap spec. pipeline / tabular WINDOW are set only for the call."""
    saved = pipeline.WINDOW, tb.WINDOW
    pipeline.WINDOW = tb.WINDOW = window
    try:
        cols, _ = feat_tb.columns(dataset)
        meta, _ = pipeline.load_dataset(dataset)
        half = tb.half_of_windows(meta)
        starts = np.array([s for _, _, s, _ in pipeline.windows(meta)])
    finally:
        pipeline.WINDOW, tb.WINDOW = saved
    out = {k: v for k, v in cols.items() if k.partition("@")[0] not in DROP_B}
    for stat in feat_tb.CONTEXT_STATS:
        for offset in EXTRA_OFFSETS:
            for side in ("h", "a"):
                out["tb_{}_{}@{:+d}".format(stat, side, offset)] = tb.shifted(
                    cols["tb_{}_{}".format(stat, side)], half, offset)
    for key in ("tb_last_team", "tb_next_team"):
        for offset in EXTRA_OFFSETS:
            out["{}@{:+d}".format(key, offset)] = tb.shifted(cols[key], half, offset)
    return out, feat_tb.spec_of(out), starts


def make_swap(a_names, b_names, b_spec):
    a_swap = ta5.make_swap(a_names)
    na = len(a_names)
    b_index = {n: i for i, n in enumerate(b_names)}
    moves = []
    for i, n in enumerate(b_names):
        kind = b_spec[n]
        if kind[0] == "pair":
            moves.append(("pair", na + i, na + b_index[kind[1]]))
        elif kind[0] in ("neg", "inv"):
            moves.append((kind[0], na + i, None))

    def swap(m):
        out = m.copy()
        out[:, :na] = a_swap(m[:, :na])
        for kind, i, j in moves:
            if kind == "pair":
                out[:, i] = m[:, j]
            elif kind == "neg":
                out[:, i] = -m[:, i]
            else:
                out[:, i] = 1.0 - m[:, i]
        return out
    return swap


def match_table(match, cond, variant):
    dataset = tb.dataset_name(match, cond)
    block = 12 if variant == "ab_half" else 5
    ta5.BLOCK = block
    x_a, a_names, info = ta5.match_table(match, cond)
    bcols, spec, starts = b_columns(dataset, block)
    assert len(starts) == len(info["start"]) and (starts == info["start"]).all(), "block align"
    b_names = list(bcols)
    x_b = np.column_stack([bcols[n] for n in b_names]).astype(np.float32)
    if variant == "b":
        keep = [i for i, n in enumerate(a_names) if n.startswith("s_")]
        x_a, a_names = x_a[:, keep], [a_names[i] for i in keep]
    return np.hstack([x_a, x_b]), a_names, b_names, spec, info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cond", default="clean")
    parser.add_argument("--variant", default="ab", choices=("ab", "b", "ab_half"))
    parser.add_argument("--trees", type=int, default=800)
    parser.add_argument("--leaves", type=int, default=63)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--stack", action="store_true")
    args = parser.parse_args()
    ta5.LGBM.update(n_estimators=args.trees, num_leaves=args.leaves, n_jobs=args.jobs)
    os.makedirs(CACHE, exist_ok=True)
    matches = tb.built_matches()
    started = time.time()
    tables = {}
    for m in matches:
        x, a_names, b_names, spec, info = match_table(m, args.cond, args.variant)
        tables[m] = (x, None, info)
    names = a_names + b_names
    swap = make_swap(a_names, b_names, spec)
    probe = tables[matches[0]][0][:3000]
    ok = np.allclose(np.nan_to_num(swap(swap(probe)), nan=-9), np.nan_to_num(probe, nan=-9),
                     atol=1e-5)
    print("{} / {}: {} columns ({} A-side, {} B), {} blocks, swap involution {}, trees {} leaves "
          "{}, {:.0f} s".format(args.cond, args.variant, len(names), len(a_names), len(b_names),
                                sum(len(t[0]) for t in tables.values()), ok, args.trees,
                                args.leaves, time.time() - started), flush=True)
    assert ok

    first = {}
    if args.stack:
        started = time.time()
        for pair in itertools.combinations(matches, 2):
            train = [m for m in matches if m not in pair]
            got = ta5.fit_team(tables, train, list(pair), swap)
            for m in pair:
                first[frozenset(pair), m] = got[m]
        print("pass 1 pair fits: {:.0f} s".format(time.time() - started), flush=True)

    for held in matches:
        started = time.time()
        train = [m for m in matches if m != held]
        p1 = ta5.fit_team(tables, train, [held], swap)[held]
        extra = {}
        if args.stack:
            stacked = {}
            stack_names = None
            for m in matches:
                p = p1 if m == held else first[frozenset((held, m)), m]
                cols = ta5.stack_columns(p, tables[m][2]["half"])
                stack_names = list(cols)
                more = np.column_stack([cols[k] for k in stack_names]).astype(np.float32)
                stacked[m] = (np.hstack([tables[m][0], more]), None, tables[m][2])
            assert all(n.startswith("f_p1_") for n in stack_names)
            na = len(names)

            def swap2(mtx):
                out = mtx.copy()
                out[:, :na] = swap(mtx[:, :na])
                out[:, na:] = 1.0 - mtx[:, na:]   # P(H | alive) context flips
                return out
            extra["p2"] = ta5.fit_team(stacked, train, [held], swap2)[held]
        info = tables[held][2]
        np.savez_compressed(os.path.join(CACHE, "{}_{}_{}.npz".format(args.cond, args.variant,
                                                                      held)),
                            p1=p1, **extra, **{k: v for k, v in info.items()})
        print("  {} held out: team acc|alive blocks p1 {:.3f}{}, {:.0f} s".format(
            held, ta5.team_acc(p1, info["counts"]),
            ", p2 {:.3f}".format(ta5.team_acc(extra["p2"], info["counts"])) if extra else "",
            time.time() - started), flush=True)
    print("done")


if __name__ == "__main__":
    main()
