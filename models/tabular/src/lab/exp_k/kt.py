"""Track K: NVIDIA Kumo Tabular (in-context learning tabular foundation model, sdm 0.2.1) as the team
model P(H | alive) in place of u_fit.py's LightGBM -- inference only, no weights trained.

Rows are u_fit.py's 0.2 s block tables (variant b or ab). For each held-out match (lomo) or each
target match (xfer: IDSSE -> SkillCorner), the context is a sample of the training matches' alive
blocks (ta5.team_target), drawn from both orientations (home/away swapped, label flipped), and every
query block is predicted in both orientations and averaged, exactly like ta5.fit_team. Columns: the
top-k by gain of a quick LightGBM fitted on the same training rows (never the query match), or all.
Writes U_CACHE/<cond>_<tag>_<match>.npz (p1 + the block meta), so ev.py --variant <tag> decodes it.

    MTAH_ROOT=/root/mtah python kt.py --cond d85suc --variant b --k 100 --ctx 50000 --tag kt_b
    MTAH_ROOT=/root/mtah python kt.py --mode xfer --cond d85suc --variant b --tag kt_b   (-> SC clean)
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
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, LAB + "/exp")
sys.path.insert(0, LAB + "/exp_a")
sys.path.insert(0, LAB + "/exp_u")
from common import pipeline, tb  # noqa: E402
import ta5  # noqa: E402
import u_fit  # noqa: E402

TABLES = os.environ.get("K_TABLES", LABCACHE + "/exp_k/tables")
OUT = u_fit.CACHE
IDSSE = ("J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9")
INFO_KEYS = ("start", "end", "half", "win", "counts", "n_frames")


def sc_matches():
    with open(os.path.join(pipeline.RUNS, "sc-stats.json")) as h:
        return list(json.load(h)["matches"])


def table(match, cond, variant):
    """(x, info) with names/spec in a sidecar; built once by u_fit.match_table, then cached."""
    os.makedirs(TABLES, exist_ok=True)
    path = os.path.join(TABLES, "{}_{}_{}.npz".format(cond, variant, match))
    side = os.path.join(TABLES, "{}_{}.names.pkl".format(variant, cond))
    if not os.path.exists(path):
        x, a_names, b_names, spec, info = u_fit.match_table(match, cond, variant)
        np.savez(path, x=x, **{k: info[k] for k in INFO_KEYS})
        with open(side, "wb") as h:
            pickle.dump((a_names, b_names, spec), h)
    z = np.load(path)
    with open(side, "rb") as h:
        a_names, b_names, spec = pickle.load(h)
    return z["x"], {k: z[k] for k in INFO_KEYS}, (a_names, b_names, spec)


def training_pool(tables, train, swap):
    xs, ys, ms = [], [], []
    for i, m in enumerate(train):
        x, info = tables[m]
        keep, y = ta5.team_target(info["counts"])
        xs.append(x[keep])
        ys.append(y[keep])
        ms.append(np.full(keep.sum(), i))
    x = np.vstack(xs)
    y = np.concatenate(ys).astype(np.int64)
    m = np.concatenate(ms)
    return np.vstack([x, swap(x)]), np.concatenate([y, 1 - y]), np.concatenate([m, m])


def select_columns(x, y, k, seed):
    """Top-k columns by LightGBM gain on (a sample of) the training rows."""
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


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def bag_contexts(y, groups, n, how, bags, seed):
    """`bags` context samples of n rows; with bags > 1 and random sampling they are disjoint
    slices of one permutation (the whole pool is seen once when bags * n >= pool)."""
    if bags == 1 or how != "random":
        return [sample_context(y, groups, n, how, seed + b) for b in range(bags)]
    perm = np.random.default_rng(seed).permutation(len(y))
    out = []
    for b in range(bags):
        part = perm[(b * n) % len(y):][:n]
        if len(part) < n:
            part = np.concatenate([part, perm[:n - len(part)]])
        out.append(np.sort(part))
    return out


def knn_predict(kumo, x, y, cols, queries, n, clusters, est, seed, mix=0.25):
    """Local in-context learning: the queries (rows of every test matrix, both orientations) are
    k-means clustered on the standardised selected columns; each cluster's context is the union of its
    rows' nearest training rows (GPU distances), topped up with a random share `mix` of the pool so
    the context never collapses onto one class. Returns P(class 1) per query row."""
    import torch
    from sklearn.cluster import MiniBatchKMeans
    xs = x[:, cols]
    mu = np.nanmean(xs, axis=0)
    sd = np.nanstd(xs, axis=0) + 1e-6
    norm = lambda a: np.nan_to_num((a - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    pool = torch.from_numpy(norm(xs)).cuda()
    q = np.vstack(queries)
    qn = norm(q[:, cols])
    rng = np.random.default_rng(seed)
    km = MiniBatchKMeans(n_clusters=clusters, random_state=seed, n_init=3, batch_size=4096)
    lab = km.fit_predict(qn)
    out = np.empty(len(q))
    n_rand = int(mix * n)
    for g in range(clusters):
        rows = np.flatnonzero(lab == g)
        if len(rows) == 0:
            continue
        per = max(1, (n - n_rand) // len(rows))
        votes = np.zeros(len(pool), dtype=np.float32)
        for i in range(0, len(rows), 2048):
            qq = torch.from_numpy(qn[rows[i:i + 2048]]).cuda()
            d = torch.cdist(qq, pool)
            k = min(per * 4, len(pool))
            nn = d.topk(k, largest=False).indices[:, :per].reshape(-1).cpu().numpy()
            np.add.at(votes, nn, 1.0)
        near = np.flatnonzero(votes > 0)
        if len(near) > n - n_rand:
            near = near[np.argsort(-votes[near])[:n - n_rand]]
        rest = np.setdiff1d(np.arange(len(pool)), near)
        extra = rng.choice(rest, size=min(n - len(near), len(rest)), replace=False)
        pick = np.sort(np.concatenate([near, extra]))
        kumo.fit(xs[pick], y[pick], est, seed + g)
        out[rows] = kumo.predict(q[rows][:, cols])
        kumo.model.clear()
    return out


def sample_context(y, groups, n, how, seed):
    rng = np.random.default_rng(seed)
    if n >= len(y):
        return np.arange(len(y))
    if how == "random":
        return np.sort(rng.choice(len(y), size=n, replace=False))
    if how == "balanced":            # equal rows per training match, then per class within it
        out = []
        ug = np.unique(groups)
        per = n // (2 * len(ug))
        for g in ug:
            for c in (0, 1):
                pool = np.flatnonzero((groups == g) & (y == c))
                out.append(rng.choice(pool, size=min(per, len(pool)), replace=False))
        return np.sort(np.concatenate(out))
    raise ValueError(how)


class Kumo:
    def __init__(self, size, device="cuda"):
        import torch
        from sdm import Stype
        from sdm.models import KumoTabular
        self.torch, self.stype = torch, Stype
        self.device = torch.device(device)
        self.model = KumoTabular(task="classification", size=size, device=self.device)

    def fit(self, x, y, est, seed):
        torch = self.torch
        x = np.where(np.isfinite(x), x, np.nan).astype(np.float32)
        torch.manual_seed(seed)
        with torch.autocast("cuda", dtype=torch.float16):
            self.model.fit(torch.from_numpy(x).to(self.device),
                           torch.from_numpy(y[:, None]).to(self.device), num_estimators=est)

    def predict(self, x, batch=8192):
        torch = self.torch
        x = np.where(np.isfinite(x), x, np.nan).astype(np.float32)
        out = []
        with torch.autocast("cuda", dtype=torch.float16):
            for i in range(0, len(x), batch):
                t = self.model.predict(torch.from_numpy(x[i:i + batch]).to(self.device))
                j = list(t.columns[self.stype.numerical]).index("1")
                out.append(t.numerical[:, j].float().cpu().numpy())
        return np.concatenate(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="lomo", choices=("lomo", "xfer", "joint"))
    parser.add_argument("--cond", default="d85suc", help="training condition (lomo: also the test)")
    parser.add_argument("--target-cond", default="clean", help="xfer: SkillCorner condition")
    parser.add_argument("--direction", default="idsse2sc", choices=("idsse2sc", "sc2idsse"))
    parser.add_argument("--variant", default="b", choices=("b", "ab"))
    parser.add_argument("--size", default="large", choices=("small", "medium", "large"))
    parser.add_argument("--ctx", type=int, default=50000)
    parser.add_argument("--ctxsel", default="random", choices=("random", "balanced", "knn"))
    parser.add_argument("--clusters", type=int, default=16, help="knn: query clusters (one fit each)")
    parser.add_argument("--k", type=int, default=100, help="columns kept (0 = all)")
    parser.add_argument("--est", type=int, default=4)
    parser.add_argument("--bags", type=int, default=1, help="independent contexts, logits averaged")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--only", nargs="*", default=None, help="lomo: these held-out matches only")
    parser.add_argument("--folds", type=int, default=0, help="joint: K contiguous groups of the held-out matches (0 = one match per fold)")
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    started = time.time()
    os.makedirs(OUT, exist_ok=True)
    if args.mode == "xfer":                # transfer caches never collide with LOMO ones
        prefix = "x" if args.direction == "idsse2sc" else "r"
        args.tag = prefix + args.tag
    if args.mode == "joint":               # joint caches: j<tag>, held-out matches (default SkillCorner)
        args.tag = "j" + args.tag
    cond_of = {}
    if args.mode == "lomo":
        matches = tb.built_matches()
        tables = {m: table(m, args.cond, args.variant) for m in matches}
        folds = [([o for o in matches if o != m], [m]) for m in (args.only or matches)]
        test_cond = args.cond
    elif args.mode == "joint":
        pool = list(IDSSE) + sc_matches()
        tables = {m: table(m, args.cond if m in IDSSE else args.target_cond, args.variant) for m in pool}
        held = list(args.only or sc_matches())
        groups = [list(g) for g in np.array_split(held, args.folds)] if args.folds else [[h] for h in held]
        folds = [([o for o in pool if o not in g], g) for g in groups]
        cond_of = {m: (args.cond if m in IDSSE else args.target_cond) for m in pool}
        test_cond = None
    elif args.direction == "idsse2sc":
        train, targets = list(IDSSE), sc_matches()
        tables = {m: table(m, args.cond, args.variant) for m in train}
        tables.update({m: table(m, args.target_cond, args.variant) for m in targets})
        folds = [(train, targets)]
        test_cond = args.target_cond
    else:                                  # sc2idsse: context from SkillCorner, targets controlled IDSSE
        train, targets = sc_matches(), list(IDSSE)
        tables = {m: table(m, args.target_cond, args.variant) for m in train}
        tables.update({m: table(m, args.cond, args.variant) for m in targets})
        folds = [(train, targets)]
        test_cond = args.cond
    names = next(iter(tables.values()))[2]
    swap = u_fit.make_swap(*names)
    tables = {m: (t[0], t[1]) for m, t in tables.items()}
    ncol = next(iter(tables.values()))[0].shape[1]
    print("{} {} {}: {} matches, {} columns, tables {:.0f} s".format(
        args.mode, args.cond, args.variant, len(tables), ncol, time.time() - started), flush=True)
    kumo = Kumo(args.size)
    accs = []
    for train, tests in folds:
        t0 = time.time()
        x, y, groups = training_pool(tables, train, swap)
        cols = select_columns(x, y, args.k, args.seed)
        bags = ([] if args.ctxsel == "knn" else
                bag_contexts(y, groups, args.ctx, args.ctxsel, args.bags, args.seed))
        swapped = {m: swap(tables[m][0]) for m in tests}
        zs = {m: [] for m in tests}
        fit_s = 0.0
        if args.ctxsel == "knn":
            tb0 = time.time()
            qs = [tables[m][0] for m in tests] + [swapped[m] for m in tests]
            allp = knn_predict(kumo, x, y, cols, qs, args.ctx, args.clusters, args.est, args.seed)
            bounds = np.cumsum([0] + [len(a) for a in qs])
            for i, m in enumerate(tests):
                p = allp[bounds[i]:bounds[i + 1]]
                j = len(tests) + i
                q = allp[bounds[j]:bounds[j + 1]]
                zs[m].append((logit(p), logit(q)))
            fit_s = time.time() - tb0
            bags = [np.arange(args.ctx)]
        for b, pick in enumerate(bags if args.ctxsel != "knn" else []):
            tb0 = time.time()
            kumo.fit(x[pick][:, cols], y[pick], args.est, args.seed + b)
            fit_s += time.time() - tb0
            for m in tests:
                p = kumo.predict(tables[m][0][:, cols])
                q = kumo.predict(swapped[m][:, cols])
                zs[m].append((logit(p), logit(q)))
            kumo.model.clear()
        t1 = time.time()
        for m in tests:
            xt, info = tables[m]
            p = sigmoid(np.mean([a for a, _ in zs[m]], axis=0))
            q = sigmoid(np.mean([b for _, b in zs[m]], axis=0))
            p1 = 0.5 * (p + 1.0 - q)
            pick = bags[0]
            np.savez_compressed(os.path.join(OUT, "{}_{}_{}.npz".format(cond_of.get(m, test_cond), args.tag, m)),
                                p1=p1, **info)
            keep, yt = ta5.team_target(info["counts"])
            acc = float(np.mean((p1[keep] > 0.5) == (yt[keep] == 1)))
            pc = np.clip(p1[keep], 1e-4, 1 - 1e-4)
            ll = float(-np.mean(np.where(yt[keep] == 1, np.log(pc), np.log(1 - pc))))
            accs.append((acc, keep.sum(), ll))
            if args.mode in ("lomo", "joint"):
                print("  {} held out: team acc|alive {:.4f}, logloss {:.4f}, orientation agreement "
                      "(alive) {:.4f}, ctx {} cols {}, fit {:.0f} s, predict {:.0f} s".format(
                          m, acc, ll, np.mean((p[keep] > 0.5) == (q[keep] < 0.5)), len(pick),
                          len(cols), fit_s, t1 - t0 - fit_s), flush=True)
    w = np.array([a[1] for a in accs], float)
    print("{} pooled team acc|alive {:.4f}, logloss {:.4f} over {} matches; {:.0f} s".format(
        args.tag, np.average([a[0] for a in accs], weights=w),
        np.average([a[2] for a in accs], weights=w), len(accs), time.time() - started), flush=True)


if __name__ == "__main__":
    main()
