"""Track K: Kumo Tabular as the 3-class P(H, A, D) model behind P(alive), on palive.py's 12-frame rows
(variant pf: u_half's columns + Track B + pfeat.py's player context), in place of LightGBM -- inference
only. Symmetric like tabular.symmetric_proba: the context mixes both orientations (swapped rows get
H<->A labels) and each query row is answered in both orientations and averaged.

    lomo: leave-one-match-out over tb.built_matches() of --cond; writes answers_exp_k_pa_<tag>.jsonl
    xfer: context from the 7 IDSSE matches of --cond, answers for every SkillCorner match (clean) as
          answers_xf_pa_<tag>.jsonl (what kt_xdec.py --palive <tag> reads)
Columns: top-k by gain of a quick 3-class LightGBM on the same training rows. Prints aggregates only.

    MTAH_ROOT=/root/mtah python kt_pa.py --cond d85su --tag k100 --k 100 --ctx 50000 --est 4
"""
import os as _os
LAB = _os.environ.get("MTAH_LAB") or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

import argparse
import json
import os
import sys
import time

sys.path.insert(0, LAB + "/exp_sc")
import palive  # noqa: E402  (12-frame windows, set before tabular is used)
import xfer  # noqa: E402

import numpy as np  # noqa: E402

from palive import ctx, tb  # noqa: E402

H, A, D = tb.H, tb.A, tb.D


def select_columns(x, y, k, seed):
    if k <= 0 or k >= x.shape[1]:
        return np.arange(x.shape[1])
    import lightgbm
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x), size=min(len(x), 120000), replace=False)
    clf = lightgbm.LGBMClassifier(n_estimators=200, learning_rate=0.1, num_leaves=31,
                                  colsample_bytree=0.5, subsample=0.8, subsample_freq=1,
                                  importance_type="gain", n_jobs=8, verbose=-1,
                                  random_state=seed).fit(x[idx], y[idx])
    return np.sort(np.argsort(-clf.feature_importances_)[:k])


class Kumo3:
    def __init__(self, size):
        import torch
        from sdm import Stype
        from sdm.models import KumoTabular
        self.torch, self.stype = torch, Stype
        self.dev = torch.device("cuda")
        self.model = KumoTabular(task="classification", size=size, device=self.dev)

    def fit(self, x, y, est, seed):
        torch = self.torch
        x = np.where(np.isfinite(x), x, np.nan).astype(np.float32)
        torch.manual_seed(seed)
        with torch.autocast("cuda", dtype=torch.float16):
            self.model.fit(torch.from_numpy(x).to(self.dev),
                           torch.from_numpy(y.astype(np.int64)[:, None]).to(self.dev),
                           num_estimators=est)

    def predict(self, x, batch=8192):
        """(n, 3) probabilities in code order H, A, D."""
        torch = self.torch
        x = np.where(np.isfinite(x), x, np.nan).astype(np.float32)
        out = []
        with torch.autocast("cuda", dtype=torch.float16):
            for i in range(0, len(x), batch):
                t = self.model.predict(torch.from_numpy(x[i:i + batch]).to(self.dev))
                names = list(t.columns[self.stype.numerical])
                num = t.numerical.float().cpu().numpy()
                res = np.zeros((len(num), 3))
                for code in (H, A, D):
                    if str(code) in names:
                        res[:, code] = num[:, names.index(str(code))]
                out.append(res)
        p = np.concatenate(out)
        return p / p.sum(axis=1, keepdims=True)


def symmetric(kumo, x, y, test, sub, args):
    keep = y >= 0
    both_x = np.vstack([x[keep], tb.swap(x[keep], sub)])
    both_y = np.concatenate([y[keep], tb.swap_labels(y[keep])])
    cols = select_columns(both_x, both_y, args.k, args.seed)
    rng = np.random.default_rng(args.seed)
    pick = np.sort(rng.choice(len(both_y), size=min(args.ctx, len(both_y)), replace=False))
    kumo.fit(both_x[pick][:, cols], both_y[pick], args.est, args.seed)
    out = {}
    for m, xt in test.items():
        p = kumo.predict(xt[:, cols])
        q = kumo.predict(tb.swap(xt, sub)[:, cols])
        out[m] = 0.5 * (p + q[:, [A, H, D]])
    kumo.model.clear()
    return out, len(cols)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="lomo", choices=("lomo", "xfer"))
    parser.add_argument("--cond", default="d85su")
    parser.add_argument("--size", default="large")
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--ctx", type=int, default=50000)
    parser.add_argument("--est", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    started = time.time()
    kumo = Kumo3(args.size)
    if args.mode == "lomo":
        matches, wide, names, full_swap = palive.tables_of(args.cond)
        tb.swap = palive.subset_swap(full_swap, names)(names)
        for held in matches:
            train = [m for m in matches if m != held]
            x = np.vstack([wide[m][0] for m in train])
            y = np.concatenate([wide[m][1] for m in train])
            got, nc = symmetric(kumo, x, y, {held: wide[held][0]}, names, args)
            tb.write_answers(tb.dataset_name(held, args.cond), "exp_k_pa_" + args.tag,
                             wide[held][2], got[held])
            print("  {} {} (cols {})".format(held, ctx.report(args.tag, got[held], wide[held][1]), nc),
                  flush=True)
    else:
        cache, tables, spec, names = {}, {}, {}, None
        targets = xfer.json.load(open(os.path.join(xfer.pipeline.RUNS, "sc-stats.json")))["matches"]
        for m in xfer.IDSSE:
            tables[m], names, s = xfer.palive_table(m, args.cond, cache)
            spec.update(s)
        for m in targets:
            tables[m], n2, s = xfer.palive_table(m, xfer.TARGET_COND, cache)
            assert n2 == names
            spec.update(s)
        tb.swap = palive.subset_swap(xfer.run_tb.make_swap(spec), names)(names)
        x = np.vstack([tables[m][0] for m in xfer.IDSSE])
        y = np.concatenate([tables[m][1] for m in xfer.IDSSE])
        got, nc = symmetric(kumo, x, y, {m: tables[m][0] for m in targets}, names, args)
        for m in targets:
            tb.write_answers(tb.dataset_name(m, xfer.TARGET_COND), "xf_pa_" + args.tag,
                             tables[m][2], got[m])
        allp = np.vstack([got[m] for m in targets])
        ally = np.concatenate([tables[m][1] for m in targets])
        print("xfer -> SkillCorner: " + ctx.report(args.tag, allp, ally) + " (cols {})".format(nc),
              flush=True)
    print("done in {:.0f} s".format(time.time() - started), flush=True)


if __name__ == "__main__":
    main()
