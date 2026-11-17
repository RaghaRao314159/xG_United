"""Does LightGBM gain from more time-series context than tabular.py's +-5 s of per-second stats?

    exp_ctx_wide     base 78 columns + blackout geometry (seconds back/forward to the ball, the
                     blackout's length) + past/future means over 15 s and 30 s + d_home/d_away
                     for the +-5 s context seconds
    exp_ctx_noblack  wide without the blackout geometry: on degraded IDSSE blackouts are placed
                     in stoppages by construction, so a gain that needs them is partly circular
    exp_ctx_stack    wide + a second pass that sees the first pass's out-of-fold P(D) and
                     P(H | alive) at offsets up to +-30 s ("auto-context" stacking)

All 3-class, both orientations, LOMO over the IDSSE matches, decoded at switch cost 3.0 and 0.1,
pooled against exp_3class (two_stage.py). Out-of-fold first-pass answers for a training match t
of fold m come from the model trained without m and t, so no row ever sees its own label.
Prints aggregates only.

    .venv/bin/python ctx.py --cond d85
"""
import os as _os
ROOT = _os.environ.get("MTAH_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))))
LAB = _os.environ.get("MTAH_LAB") or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
LABCACHE = _os.path.join(ROOT, "runs", "lab-cache")

import argparse
import itertools
import sys
import time

import numpy as np

from common import D, H, decode_all, paired, pipeline, pooled, print_pooled, tb

WIDTHS = (15, 30)
MEAN_STATS = ("seen", "crowd", "home", "dmin", "speed")
BLACK_CAP = 120.0
STACK_OFFSETS = (1, 2, 3, 5, 8, 12, 20, 30)
BLACK = ("black_back", "black_fwd", "black_len")
ORIGINAL_SWAP = tb.swap


def window_mean(values, half, lo, hi):
    """Mean of the non-NaN values in [t+lo, t+hi] within t's half, NaN when there are none."""
    out = np.full(len(values), np.nan)
    for h in np.unique(half):
        idx = np.flatnonzero(half == h)
        v = values[idx]
        ok = ~np.isnan(v)
        s = np.concatenate([[0.0], np.cumsum(np.where(ok, v, 0.0))])
        c = np.concatenate([[0], np.cumsum(ok)])
        t = np.arange(len(idx))
        a = np.clip(t + lo, 0, len(idx))
        b = np.clip(t + hi + 1, 0, len(idx))
        n = c[b] - c[a]
        with np.errstate(invalid="ignore", divide="ignore"):
            out[idx] = np.where(n > 0, (s[b] - s[a]) / np.maximum(n, 1), np.nan)
    return out


def blackout(seen, half):
    """Seconds back to the last second with the ball, forward to the next, and their sum; 0 on
    seconds that show the ball."""
    back = np.zeros(len(seen))
    fwd = np.zeros(len(seen))
    for h in np.unique(half):
        idx = np.flatnonzero(half == h)
        lost = seen[idx] == 0
        run = 0
        for i, k in enumerate(idx):
            run = run + 1 if lost[i] else 0
            back[k] = run
        run = 0
        for i, k in reversed(list(enumerate(idx))):
            run = run + 1 if lost[i] else 0
            fwd[k] = run
    cap = lambda x: np.minimum(x, BLACK_CAP)  # noqa: E731
    return {"black_back": cap(back), "black_fwd": cap(fwd),
            "black_len": cap(np.where(back > 0, back + fwd - 1, 0))}


def extra(match, cond):
    """{name: column} of the wide additions for one match."""
    meta, f = pipeline.load_dataset(tb.dataset_name(match, cond))
    stats = tb.second_stats(f, meta, tb.crowd_scale_of(f, meta))
    half = tb.half_of_windows(meta)
    out = dict(blackout(stats["seen"], half))
    for width in WIDTHS:
        for stat in MEAN_STATS:
            out["{}_p{}".format(stat, width)] = window_mean(stats[stat], half, -width, -1)
            out["{}_f{}".format(stat, width)] = window_mean(stats[stat], half, 1, width)
    for offset in range(-tb.CONTEXT, tb.CONTEXT + 1):
        if offset:
            out["dh{:+d}".format(offset)] = tb.shifted(stats["d_home"], half, offset)
            out["da{:+d}".format(offset)] = tb.shifted(stats["d_away"], half, offset)
    return out, half


def stack_columns(proba, half):
    """{name: column}: first-pass P(D) and P(H | alive) at the stack offsets, and their
    past/future means."""
    alive = proba[:, H] + proba[:, 1]
    p_d = proba[:, D]
    p_h = np.where(alive > tb.PROB_EPS, proba[:, H] / np.maximum(alive, tb.PROB_EPS), 0.5)
    out = {"sD0": p_d, "sH0": p_h}
    for offset in STACK_OFFSETS:
        for sign in (-1, 1):
            out["sD{:+d}".format(sign * offset)] = tb.shifted(p_d, half, sign * offset)
            out["sH{:+d}".format(sign * offset)] = tb.shifted(p_h, half, sign * offset)
    for width in WIDTHS:
        out["sD_p{}".format(width)] = window_mean(p_d, half, -width, -1)
        out["sD_f{}".format(width)] = window_mean(p_d, half, 1, width)
        out["sH_p{}".format(width)] = window_mean(p_h, half, -width, -1)
        out["sH_f{}".format(width)] = window_mean(p_h, half, 1, width)
    return out


def swap(matrix, names):
    """tabular.swap, plus the new columns: home shares and P(H | alive) flip, distances pair up."""
    out = ORIGINAL_SWAP(matrix, names)
    index = {n: i for i, n in enumerate(names)}
    for i, name in enumerate(names):
        if name.startswith(("home_", "sH")):
            out[:, i] = 1.0 - matrix[:, i]
        elif name.startswith("dh"):
            j = index["da" + name[2:]]
            out[:, i], out[:, j] = matrix[:, j], matrix[:, i]
    return out


tb.swap = swap  # symmetric_proba looks it up in tabular's namespace


def proba_of(train, test, tables, names):
    x = np.vstack([tables[m][0] for m in train])
    y = np.concatenate([tables[m][1] for m in train])
    proba, _ = tb.symmetric_proba("lgbm", x, y, tables[test][0], names, tb.SEED)
    return proba


def with_columns(table, cols):
    x, y, meta = table
    names = list(cols)
    return np.column_stack([x] + [cols[n] for n in names]).astype(np.float32), y, meta, names


def report(name, proba, y):
    graded = y >= 0
    pred = proba[graded].argmax(axis=1)
    return "{}: acc {:.3f} alive/dead acc {:.3f} P(D) {:.3f} vs {:.3f}".format(
        name, np.mean(pred == y[graded]), np.mean((pred == D) == (y[graded] == D)),
        proba[graded, D].mean(), np.mean(y[graded] == D))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cond", default="d85")
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    matches = tb.built_matches()
    base_names, cache = tb.feature_names(), {}
    wide, noblack, halves = {}, {}, {}
    for m in matches:
        cols, halves[m] = extra(m, args.cond)
        x, y, meta, extra_names = with_columns(tb.table_of(m, args.cond, cache), cols)
        wide[m] = (x, y, meta)
        keep = [i for i, n in enumerate(base_names + extra_names) if n not in BLACK]
        noblack[m] = (x[:, keep], y, meta)
    wide_names = base_names + extra_names
    noblack_names = [n for n in wide_names if n not in BLACK]
    print("columns: base {}, wide {}, noblack {}".format(
        len(base_names), len(wide_names), len(noblack_names)), flush=True)

    # first pass: the wide model without each pair of matches, and without each single match
    started = time.time()
    first = {}
    for pair in itertools.combinations(matches, 2):
        train = [m for m in matches if m not in pair]
        x = np.vstack([wide[m][0] for m in train])
        y = np.concatenate([wide[m][1] for m in train])
        both = np.vstack([wide[m][0] for m in pair])
        proba, _ = tb.symmetric_proba("lgbm", x, y, both, wide_names, tb.SEED)
        n = len(wide[pair[0]][0])
        first[frozenset(pair), pair[0]] = proba[:n]
        first[frozenset(pair), pair[1]] = proba[n:]
    print("first pass, {} pair fits: {:.0f} s".format(
        len(list(itertools.combinations(matches, 2))), time.time() - started), flush=True)

    for held in matches:
        started = time.time()
        train = [m for m in matches if m != held]
        dataset = tb.dataset_name(held, args.cond)
        meta, y = wide[held][2], wide[held][1]
        line = [held]

        p_wide = proba_of(train, held, wide, wide_names)
        p_noblack = proba_of(train, held, noblack, noblack_names)

        stack = {}
        for m in matches:
            p = p_wide if m == held else first[frozenset((held, m)), m]
            x, yy, mm, stack_names = with_columns(wide[m], stack_columns(p, halves[m]))
            stack[m] = (x, yy, mm)
        p_stack = proba_of(train, held, stack, wide_names + stack_names)

        for name, proba in (("exp_ctx_wide", p_wide), ("exp_ctx_noblack", p_noblack),
                            ("exp_ctx_stack", p_stack)):
            tb.write_answers(dataset, name, meta, proba)
            decode_all(dataset, name)
            line.append(report(name, proba, y))
        line.append("{:.0f} s".format(time.time() - started))
        print("\n  ".join(line), flush=True)

    sys.stdout.flush()
    folder = pooled(args.cond, matches, ("exp_3class", "exp_2stage", "exp_ctx", "baseline"),
                    args.bootstrap)
    print_pooled(folder)
    paired(folder)


if __name__ == "__main__":
    main()
