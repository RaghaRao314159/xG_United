"""Idea #2: P(alive) variants, run like u_half.py (Track B's half-wide 3-class LightGBM at 12-frame
rows, leave-one-match-out, both orientations) on any condition and match set:
  pf      u_half's columns + pfeat.py's player-only context columns
  blind   only columns that never read the ball: the crowd speed stats, Track B's team shape
          (cx, cv, sx, sy, pspeed) and pfeat.py's columns
u_half.py's own answers (exp_tb_half_wide) are the reference. Each column's home/away swap rule is
read off the full table's swap once, so any subset of columns swaps correctly. Writes
answers_exp_sc_pa_<variant>.jsonl in each dataset dir. Prints aggregates only.

    MTAH_ROOT=/root/mtah MTAH_LGBM_JOBS=4 python palive.py --cond d85su --variants pf blind
"""
import os as _os
ROOT = _os.environ.get("MTAH_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))))
LAB = _os.environ.get("MTAH_LAB") or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
LABCACHE = _os.path.join(ROOT, "runs", "lab-cache")

import argparse
import sys
import time

sys.path.insert(0, LAB + "/exp")
sys.path.insert(0, ROOT + "/src")  # xG United: folder renamed from jev_track
sys.path.insert(0, LAB + "/exp_b")
sys.path.insert(0, LAB + "/exp_sc")
import pipeline  # noqa: E402
import tabular  # noqa: E402

HALF = 12
pipeline.WINDOW = HALF
tabular.WINDOW = HALF

import numpy as np  # noqa: E402

import ctx  # noqa: E402
import feat_tb  # noqa: E402
import pfeat  # noqa: E402
import run_tb  # noqa: E402
from common import tb  # noqa: E402

BALL_FREE_B = ("tb_cx_", "tb_cv_", "tb_sx_", "tb_sy_", "tb_pspeed_")


def ball_free(name):
    return (name.startswith("pf_") or name.startswith(BALL_FREE_B)
            or name.partition("_")[0] in ("crowd",) or name.startswith("crowd"))


def subset_swap(full_swap, names):
    """{column: (kind, partner)} read off the full swap with a probe of distinct values."""
    rng = np.random.default_rng(0)
    probe = rng.uniform(0.1, 0.9, size=(4, len(names))) + np.arange(len(names))[None, :]
    out = full_swap(probe, names)
    rules = {}
    for i, n in enumerate(names):
        col = out[:, i]
        if np.allclose(col, probe[:, i]):
            rules[n] = ("same", None)
        elif np.allclose(col, 1.0 - probe[:, i]):
            rules[n] = ("inv", None)
        elif np.allclose(col, -probe[:, i]):
            rules[n] = ("neg", None)
        else:
            j = int(np.argmin(np.abs(probe - col[:, None]).sum(axis=0)))
            assert np.allclose(probe[:, j], col), "no swap rule for " + n
            rules[n] = ("pair", names[j])

    def make(sub):
        index = {n: i for i, n in enumerate(sub)}
        for n in sub:
            kind, partner = rules[n]
            assert kind != "pair" or partner in index, "{} needs {}".format(n, partner)

        def swap(matrix, cols):
            assert list(cols) == list(sub)
            res = matrix.copy()
            for i, n in enumerate(sub):
                kind, partner = rules[n]
                if kind == "pair":
                    res[:, i] = matrix[:, index[partner]]
                elif kind == "inv":
                    res[:, i] = 1.0 - matrix[:, i]
                elif kind == "neg":
                    res[:, i] = -matrix[:, i]
            return res
        return swap
    return make


def tables_of(cond):
    matches = tb.built_matches()
    base_names, cache = tb.feature_names(), {}
    wide, spec, names = {}, {}, None
    for m in matches:
        dataset = tb.dataset_name(m, cond)
        cols, _ = ctx.extra(m, cond)
        x, y, meta, extra_names = ctx.with_columns(tb.table_of(m, cond, cache), cols)
        tcols, tspec = feat_tb.columns(dataset)
        pcols, pspec = pfeat.columns(dataset)
        assert len(next(iter(pcols.values()))) == len(x), "pfeat windows"
        spec.update(tspec)
        spec.update(pspec)
        t_names, p_names = list(tcols), list(pcols)
        x = np.column_stack([x] + [tcols[n] for n in t_names] + [pcols[n] for n in p_names])
        wide[m] = (x.astype(np.float32), y, meta)
        names = base_names + extra_names + t_names + p_names
    return matches, wide, names, run_tb.make_swap(spec)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cond", nargs="+", required=True)
    parser.add_argument("--variants", nargs="+", default=["pf", "blind"], choices=("pf", "blind"))
    parser.add_argument("--precompute", action="store_true", help="only cache pfeat's columns")
    args = parser.parse_args()
    if args.precompute:
        for cond in args.cond:
            for m in tb.built_matches():
                started = time.time()
                cols, _ = pfeat.columns(tb.dataset_name(m, cond))
                print("  {} {}: {} pf columns, {:.0f} s".format(m, cond, len(cols),
                                                                 time.time() - started), flush=True)
        return
    for cond in args.cond:
        started = time.time()
        matches, wide, names, full_swap = tables_of(cond)
        make = subset_swap(full_swap, names)
        for variant in args.variants:
            keep = [i for i, n in enumerate(names) if variant == "pf" or ball_free(n)]
            sub = [names[i] for i in keep]
            tables = {m: (wide[m][0][:, keep], wide[m][1], wide[m][2]) for m in matches}
            tb.swap = make(sub)
            probe = tables[matches[0]][0][:2000]
            assert np.allclose(np.nan_to_num(tb.swap(tb.swap(probe, sub), sub), nan=-9),
                               np.nan_to_num(probe, nan=-9), atol=1e-5), "swap involution"
            print("{} / {}: {} columns ({} pf_), {} matches".format(
                cond, variant, len(sub), sum(n.startswith("pf_") for n in sub), len(matches)),
                flush=True)
            name = "exp_sc_pa_" + variant
            for held in matches:
                train = [m for m in matches if m != held]
                proba = ctx.proba_of(train, held, tables, sub)
                tb.write_answers(tb.dataset_name(held, cond), name, tables[held][2], proba)
                print("  " + held + " " + ctx.report(name, proba, tables[held][1]), flush=True)
        print("{} done in {:.0f} s".format(cond, time.time() - started), flush=True)


if __name__ == "__main__":
    main()
