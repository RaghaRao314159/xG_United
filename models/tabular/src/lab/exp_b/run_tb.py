"""Track B models: ctx.py's wide and stacked LightGBM with the raw-position columns added.

    exp_tb_wide    ctx wide (121 columns) + feat_tb columns, one pass
    exp_tb_stack   exp_tb_wide + out-of-fold first-pass P(D), P(H | alive) context (ctx.py's
                   stacking, first pass = exp_tb_wide)
LOMO over the 7 IDSSE matches, decoded at cost 3.0 and 0.1, pooled into runs/exp-tb-pooled-<cond>
with exp_ctx_stack / exp_ctx_wide as references, paired bootstrap, top LightGBM gains.
Prints aggregates only.

    MTAH_ROOT=/root/mtah MTAH_LGBM_JOBS=4 python run_tb.py --cond clean d85
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

sys.path.insert(0, LAB + "/exp")
import common  # noqa: E402
import ctx  # noqa: E402  (sets tabular.swap to ctx.swap)
import feat_tb  # noqa: E402
from common import decode_all, paired, print_pooled, tb  # noqa: E402

CTX_SWAP = ctx.swap


def make_swap(spec):
    def swap(matrix, names):
        out = CTX_SWAP(matrix, names)
        index = {n: i for i, n in enumerate(names)}
        for i, name in enumerate(names):
            kind = spec.get(name)
            if kind is None or kind[0] == "same":
                continue
            if kind[0] == "pair":
                out[:, i] = matrix[:, index[kind[1]]]
            elif kind[0] == "neg":
                out[:, i] = -matrix[:, i]
            elif kind[0] == "inv":
                out[:, i] = 1.0 - matrix[:, i]
        return out
    return swap


def importance(tables, names, top=25):
    import lightgbm
    x = np.vstack([t[0] for t in tables.values()])
    y = np.concatenate([t[1] for t in tables.values()])
    keep = y >= 0
    x2 = np.vstack([x[keep], tb.swap(x[keep], names)])
    y2 = np.concatenate([y[keep], tb.swap_labels(y[keep])])
    clf = lightgbm.LGBMClassifier(random_state=tb.SEED, importance_type="gain", **tb.LGBM)
    clf.fit(np.nan_to_num(x2, nan=tb.FILL), y2)
    gain = clf.booster_.feature_importance(importance_type="gain")
    share = gain / gain.sum()
    order = np.argsort(-share)
    tb_share = sum(share[i] for i, n in enumerate(names) if n.startswith("tb_"))
    print("  gain share of all tb_ columns: {:.1%}".format(tb_share))
    print("  top {} by gain: ".format(top) + ", ".join(
        "{} {:.1%}".format(names[i], share[i]) for i in order[:top]), flush=True)


def run(cond, bootstrap):
    started = time.time()
    matches = tb.built_matches()
    base_names, cache = tb.feature_names(), {}
    wide, halves, spec = {}, {}, None
    for m in matches:
        cols, halves[m] = ctx.extra(m, cond)
        x, y, meta, extra_names = ctx.with_columns(tb.table_of(m, cond, cache), cols)
        tcols, spec = feat_tb.columns(tb.dataset_name(m, cond))
        tb_names = list(tcols)
        x = np.column_stack([x] + [tcols[n] for n in tb_names]).astype(np.float32)
        wide[m] = (x, y, meta)
    names = base_names + extra_names + tb_names
    tb.swap = make_swap(spec)
    probe = wide[matches[0]][0][:3000]
    twice = tb.swap(tb.swap(probe, names), names)
    assert np.allclose(np.nan_to_num(twice, nan=-9), np.nan_to_num(probe, nan=-9), atol=1e-5), "swap"
    print("{}: {} columns ({} tb_), swap involution ok, built in {:.0f} s".format(
        cond, len(names), len(tb_names), time.time() - started), flush=True)
    importance(wide, names)

    first = {}
    for pair in itertools.combinations(matches, 2):
        train = [m for m in matches if m not in pair]
        x = np.vstack([wide[m][0] for m in train])
        y = np.concatenate([wide[m][1] for m in train])
        both = np.vstack([wide[m][0] for m in pair])
        proba, _ = tb.symmetric_proba("lgbm", x, y, both, names, tb.SEED)
        n = len(wide[pair[0]][0])
        first[frozenset(pair), pair[0]] = proba[:n]
        first[frozenset(pair), pair[1]] = proba[n:]
    print("  first pass: {:.0f} s".format(time.time() - started), flush=True)

    for held in matches:
        train = [m for m in matches if m != held]
        dataset = tb.dataset_name(held, cond)
        meta, y = wide[held][2], wide[held][1]
        p_wide = ctx.proba_of(train, held, wide, names)
        stack = {}
        for m in matches:
            p = p_wide if m == held else first[frozenset((held, m)), m]
            x, yy, mm, stack_names = ctx.with_columns(wide[m], ctx.stack_columns(p, halves[m]))
            stack[m] = (x, yy, mm)
        p_stack = ctx.proba_of(train, held, stack, names + stack_names)
        line = [held]
        for name, proba in (("exp_tb_wide", p_wide), ("exp_tb_stack", p_stack)):
            tb.write_answers(dataset, name, meta, proba)
            decode_all(dataset, name)
            line.append(ctx.report(name, proba, y))
        print("  " + "\n    ".join(line), flush=True)

    common.POOL = "exp-tb-pooled-{}"
    folder = common.pooled(cond, matches, ("exp_ctx_stack", "exp_ctx_wide", "exp_tb_", "baseline"),
                           bootstrap)
    sys.stdout.flush()
    print_pooled(folder)
    sys.stdout.flush()
    paired(folder)
    print("{} done in {:.0f} s".format(cond, time.time() - started), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cond", nargs="+", default=["clean", "d85"])
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    for cond in args.cond:
        run(cond, args.bootstrap)


if __name__ == "__main__":
    main()
