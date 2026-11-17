"""P(alive) variants for the dead-ball hand-off: palive.py's 3-class rows (variant pf: u_half's columns,
Track B, pfeat.py's player context; optionally + sfeat.py's stoppage clocks), trained on one or more
IDSSE conditions at once (the same matches, each condition a separate copy of its rows), with
LightGBM (tabular.symmetric_proba, as palive.py) or Kumo Tabular (kt_pa.symmetric, as kt_pa.py).
Labels are the controlled-possession truth shared by every <cond>c copy (conds.py); only P(D)
matters downstream.

  LOMO   for each IDSSE match: fit on the other six (all --train conditions), answer the held-out
         match in every --evals condition -> runs/idsse-<m>-<eval>/answers_exp_sc_pa_<name>.jsonl
  --xfer one fit on all seven, answers for the 20 SkillCorner matches (clean)
         -> answers_xf_pa_<name>.jsonl (what kt_xdec.py / kt_own.py read)
  --bar  the same fit, answers for BAR-ATH -> runs/BAR-ATH/answers_exp_k_all_pa_<name>.jsonl
         (what bar_k3.py decode --palive-from <name> reads)
Nothing is ever fitted on SkillCorner or BAR-ATH rows.

    MTAH_ROOT=/root/mtah python dpa.py --name T1 --train d85suc d50suc --feats pf --model lgbm \
        --evals d85suc d50suc cleansuc --xfer --bar
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
import sys
import time

sys.path.insert(0, LAB + "/exp_sc")
import palive  # noqa: E402  (12-frame windows, set before tabular is used)
import xfer  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, LAB + "/exp_t")
sys.path.insert(0, LAB + "/exp_k")
sys.path.insert(0, LAB + "/exp_dball")
import bar  # noqa: E402  (BAR-ATH as one more dataset name)
import sfeat  # noqa: E402
from palive import ctx, tb  # noqa: E402

pipeline = xfer.pipeline
TARGET = bar.TARGET
IDSSE = xfer.IDSSE


def sc_matches():
    with open(os.path.join(pipeline.RUNS, "sc-stats.json")) as h:
        return list(json.load(h)["matches"])


def table(m, cond, cache, feats):
    (x, y, meta), names, spec = xfer.palive_table(m, cond, cache)
    if "sf" in feats:
        cols, sspec = sfeat.columns(tb.dataset_name(m, cond))
        sn = sorted(cols)
        assert len(cols[sn[0]]) == len(x), "sfeat windows"
        x = np.column_stack([x] + [cols[n] for n in sn]).astype(np.float32)
        names = names + sn
        spec = dict(spec, **sspec)
    if "noblk" in feats:     # without the blackout geometry and the missing-ball clocks
        keep = [i for i, n in enumerate(names) if not n.startswith(("black_", "gap"))]
        x, names = x[:, keep], [names[i] for i in keep]
    return (x, y, meta), names, spec


class Fitter:
    def __init__(self, model, names, args):
        self.model, self.names, self.args = model, names, args
        self.kumo = None
        if model == "kumo":
            import kt_pa
            self.kt_pa = kt_pa
            self.kumo = kt_pa.Kumo3("large")

    def __call__(self, x, y, tests):
        """{key: (n, 3) P(H, A, D)} for each test matrix, one fit on (x, y)."""
        if self.model == "lgbm":
            keys = list(tests)
            stacked = np.vstack([tests[k] for k in keys])
            proba, _ = tb.symmetric_proba("lgbm", x, y, stacked, self.names, tb.SEED)
            bounds = np.cumsum([0] + [len(tests[k]) for k in keys])
            return {k: proba[bounds[i]:bounds[i + 1]] for i, k in enumerate(keys)}
        ns = argparse.Namespace(k=self.args.k, ctx=self.args.ctx, est=self.args.est, seed=0)
        got, _ = self.kt_pa.symmetric(self.kumo, x, y, tests, self.names, ns)
        return got


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--train", nargs="+", required=True, help="IDSSE conditions to fit on")
    parser.add_argument("--evals", nargs="+", default=["d85suc", "d50suc", "cleansuc"])
    parser.add_argument("--feats", nargs="+", default=["pf"], choices=("pf", "sf", "noblk"))
    parser.add_argument("--model", default="lgbm", choices=("lgbm", "kumo"))
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--ctx", type=int, default=50000)
    parser.add_argument("--est", type=int, default=4)
    parser.add_argument("--no-lomo", action="store_true")
    parser.add_argument("--precompute", action="store_true", help="only build the column caches")
    parser.add_argument("--xfer", action="store_true")
    parser.add_argument("--bar", action="store_true")
    args = parser.parse_args()
    started = time.time()
    matches = [m for m in IDSSE if m in tb.built_matches()]
    conds = list(dict.fromkeys(args.train + args.evals))
    cache, tables, spec, names = {}, {}, {}, None
    for m in matches:
        for c in conds:
            tables[m, c], n2, s = table(m, c, cache, args.feats)
            assert names is None or n2 == names, "column layout differs"
            names = n2
            spec.update(s)
        for c in conds[1:]:
            assert (tables[m, c][1] == tables[m, conds[0]][1]).all(), (m, c, "labels differ")
    targets = sc_matches() if args.xfer else []
    for m in targets:
        tables[m, "clean"], n2, s = table(m, "clean", cache, args.feats)
        assert n2 == names
        spec.update(s)
    if args.bar:
        tables[TARGET, "bar"], n2, s = table(TARGET, "d85suc", cache, args.feats)
        assert n2 == names
        spec.update(s)
    tb.swap = palive.subset_swap(xfer.run_tb.make_swap(spec), names)(names)
    print("{}: {} on {} ({}), {} columns ({} pf_, {} sf_), tables {:.0f} s".format(
        args.name, args.model, "+".join(args.train), "+".join(args.feats), len(names),
        sum(n.startswith("pf_") for n in names), sum(n.startswith("sf_") for n in names),
        time.time() - started), flush=True)
    if args.precompute:      # column caches written once, before parallel runs read them
        return
    fit = Fitter(args.model, names, args)
    out_name = "exp_sc_pa_" + args.name

    if not args.no_lomo:
        pooled = {c: ([], []) for c in args.evals}
        for held in matches:
            t0 = time.time()
            train = [m for m in matches if m != held]
            x = np.vstack([tables[m, c][0] for m in train for c in args.train])
            y = np.concatenate([tables[m, c][1] for m in train for c in args.train])
            got = fit(x, y, {c: tables[held, c][0] for c in args.evals})
            parts = []
            for c in args.evals:
                tb.write_answers(tb.dataset_name(held, c), out_name, tables[held, c][2], got[c])
                pooled[c][0].append(got[c])
                pooled[c][1].append(tables[held, c][1])
                parts.append("{} {}".format(c, ctx.report("", got[c], tables[held, c][1])[2:]))
            print("  {} ({:.0f} s): {}".format(held, time.time() - t0, " | ".join(parts)), flush=True)
        for c in args.evals:
            print("LOMO {} pooled {}".format(c, ctx.report(out_name, np.vstack(pooled[c][0]),
                                                           np.concatenate(pooled[c][1]))), flush=True)

    if args.xfer or args.bar:
        t0 = time.time()
        x = np.vstack([tables[m, c][0] for m in matches for c in args.train])
        y = np.concatenate([tables[m, c][1] for m in matches for c in args.train])
        tests = {m: tables[m, "clean"][0] for m in targets}
        if args.bar:
            tests[TARGET] = tables[TARGET, "bar"][0]
        got = fit(x, y, tests)
        if targets:
            for m in targets:
                tb.write_answers(tb.dataset_name(m, "clean"), "xf_pa_" + args.name, tables[m, "clean"][2], got[m])
            print("xfer -> SkillCorner {}".format(ctx.report(
                args.name, np.vstack([got[m] for m in targets]),
                np.concatenate([tables[m, "clean"][1] for m in targets]))), flush=True)
        if args.bar:
            tb.write_answers(TARGET, "exp_k_all_pa_" + args.name, tables[TARGET, "bar"][2], got[TARGET])
            print("BAR-ATH: mean P(D) {:.3f}, share of windows with P(D) > 0.5 {:.3f}".format(
                got[TARGET][:, tb.D].mean(), np.mean(got[TARGET][:, tb.D] > 0.5)), flush=True)
        print("full fit {:.0f} s".format(time.time() - t0), flush=True)
    print("done in {:.0f} s".format(time.time() - started), flush=True)


if __name__ == "__main__":
    main()
