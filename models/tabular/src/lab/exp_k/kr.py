"""Track K: NVIDIA Kumo Relational (KumoRFM-2-style relational in-context model, sdm 0.2.1) as the team
model P(H | alive) -- inference only. Each 0.2 s block is a task row linked to
    blocks   one row per block: the top-k (LightGBM-gain) columns of u_fit.py's table (as kt.py)
    players  the n nearest players to the ball in that block (to the players' centroid when the ball
             is lost): side (+1 home / -1 away), distance to ball, speed, closing speed, distance rank,
             distance to the nearest opponent -- all orientation-free except side
with players.block_id -> blocks.block_id and 1-hop message passing. Both orientations are used like
kt.py (swapped block = swapped columns, side * -1, label flipped); answers averaged over orientations.
Writes U_CACHE/<cond>_<tag>_<m>.npz like kt.py. Prints aggregates only.

    MTAH_ROOT=/root/mtah python kr.py --cond d85suc --only J03WMX --ctx 10000 --tag kr_smoke
    MTAH_ROOT=/root/mtah python kr.py --mode xfer --cond d85suc --ctx 20000 --tag kr   (-> SkillCorner)
"""
import os as _os
LAB = _os.environ.get("MTAH_LAB") or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, LAB + "/exp_k")
import kt  # noqa: E402

PLAYER_COLS = ("side", "dist", "speed", "closing", "rank", "opp_gap")


def player_rows(match, cond, info, n_near, fps=25.0):
    """(block_index, features) of the n players nearest the ball in each block, from raw_tb.npz."""
    path = os.path.join(kt.pipeline.dataset_dir(kt.tb.dataset_name(match, cond)), "raw_tb.npz")
    z = np.load(path)
    pitch = z["pitch"].astype(np.float32)
    ball = z["ball"] * pitch
    pl = z["players"] * pitch                      # (n, K, 2)
    side = np.where(z["team"] == 0, 1.0, -1.0).astype(np.float32)
    start, end = info["start"], info["end"]
    nb = len(start)
    s0 = np.clip(start, 0, len(ball) - 1)
    s1 = np.clip(end - 1, 0, len(ball) - 1)
    dt = np.maximum(s1 - s0, 1) / fps
    # block means via cumulative sums over frames (NaN-aware)
    def bmean(a):
        ok = np.isfinite(a)
        v = np.where(ok, a, 0.0)
        cs = np.concatenate([np.zeros((1,) + a.shape[1:]), np.cumsum(v, axis=0)])
        cn = np.concatenate([np.zeros((1,) + a.shape[1:]), np.cumsum(ok, axis=0)])
        s = cs[end] - cs[start]
        n = cn[end] - cn[start]
        return np.where(n > 0, s / np.maximum(n, 1), np.nan)
    bpos = bmean(pl)                               # (nb, K, 2)
    bball = bmean(ball)                            # (nb, 2)
    seen = np.isfinite(bball[:, 0])
    centre = np.nanmean(bpos, axis=1)              # players' centroid
    ref = np.where(seen[:, None], bball, centre)
    d = np.linalg.norm(bpos - ref[:, None, :], axis=2)                       # (nb, K)
    p0, p1 = pl[s0], pl[s1]
    speed = np.linalg.norm(p1 - p0, axis=2) / dt[:, None]
    b0 = np.where(np.isfinite(ball[s0]), ball[s0], ref)
    b1 = np.where(np.isfinite(ball[s1]), ball[s1], ref)
    closing = (np.linalg.norm(p0 - b0[:, None], axis=2) - np.linalg.norm(p1 - b1[:, None], axis=2)) / dt[:, None]
    dd = np.where(np.isfinite(d), d, np.inf)
    order = np.argsort(dd, axis=1)[:, :n_near]
    rows_b, feats = [], []
    for r in range(n_near):
        k = order[:, r]
        ok = np.isfinite(dd[np.arange(nb), k])
        idx = np.flatnonzero(ok)
        kk = k[idx]
        # distance from this player to the nearest player of the other side
        pos = bpos[idx, kk]                                                  # (m, 2)
        other = side[None, :] != side[kk][:, None]                           # (m, K)
        gap = np.linalg.norm(bpos[idx] - pos[:, None], axis=2)
        gap = np.where(other & np.isfinite(gap), gap, np.inf).min(axis=1)
        gap = np.where(np.isfinite(gap), gap, np.nan)
        dist = np.where(seen[idx], d[idx, kk], np.nan)
        feats.append(np.column_stack([side[kk], dist, speed[idx, kk], np.where(seen[idx], closing[idx, kk], np.nan),
                                      np.full(len(idx), r, np.float32), gap]))
        rows_b.append(idx)
    return np.concatenate(rows_b), np.vstack(feats).astype(np.float32)


class Relational:
    def __init__(self, device="cuda"):
        import torch
        import pandas as pd
        import sdm
        from sdm.models import KumoRelational
        self.torch, self.pd, self.sdm = torch, pd, sdm
        self.dev = torch.device(device)
        self.model = KumoRelational(task="classification", device=self.dev)

    def _tables(self, ids, xb, block_of_row, prow, names):
        pd, sdm = self.pd, self.sdm
        xb = np.where(np.isfinite(xb), xb, np.nan).astype(np.float32)
        prow = np.where(np.isfinite(prow), prow, np.nan).astype(np.float32)
        bdf = pd.DataFrame(xb, columns=names)
        bdf.insert(0, "block_id", ids)
        pdf = pd.DataFrame(prow, columns=list(PLAYER_COLS))
        pdf.insert(0, "block_id", ids[block_of_row])
        st_b = {c: "numerical" for c in names}
        st_b["block_id"] = "id"
        st_p = {c: "numerical" for c in PLAYER_COLS}
        st_p["block_id"] = "id"
        blocks = sdm.TableTensor.from_pandas(df=bdf, stypes=st_b, device=self.dev)
        players = sdm.TableTensor.from_pandas(df=pdf, stypes=st_p, device=self.dev)
        rel = sdm.RelatedTables(
            tables={"blocks": blocks, "players": players},
            relationships=[{"left_table": "players", "left_columns": "block_id",
                            "right_table": "blocks", "right_columns": "block_id"}],
            task_links=[{"task_columns": "block_id", "table": "blocks", "table_columns": "block_id"}])
        task = sdm.TableTensor.from_pandas(df=pd.DataFrame({"block_id": ids}), stypes={"block_id": "id"},
                                           device=self.dev)
        return task, rel

    def fit(self, ids, xb, y, block_of_row, prow, names, est):
        task, rel = self._tables(ids, xb, block_of_row, prow, names)
        yt = self.sdm.TableTensor.from_pandas(df=self.pd.DataFrame({"y": y.astype(np.int64)}),
                                              stypes={"y": "categorical"}, device=self.dev)
        with self.torch.autocast("cuda", dtype=self.torch.float16):
            self.model.fit(task, yt, rel, num_hops=1, num_estimators=est)

    def predict(self, ids, xb, block_of_row, prow, names):
        task, rel = self._tables(ids, xb, block_of_row, prow, names)
        with self.torch.autocast("cuda", dtype=self.torch.float16):
            out = self.model.predict(task, rel)
        j = list(out.columns[self.sdm.Stype.numerical]).index("1")
        return out.numerical[:, j].float().cpu().numpy()


def subset_rows(block_of_row, prow, pick):
    """Player rows of the picked blocks, re-indexed to positions in `pick`."""
    size = max(int(block_of_row.max()) + 1 if len(block_of_row) else 0, int(np.max(pick)) + 1 if len(pick) else 0, 1)
    pos = np.full(size, -1)
    pos[pick] = np.arange(len(pick))
    keep = pos[block_of_row] >= 0
    return pos[block_of_row[keep]], prow[keep]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="lomo", choices=("lomo", "xfer"))
    parser.add_argument("--cond", default="d85suc")
    parser.add_argument("--target-cond", default="clean")
    parser.add_argument("--variant", default="b")
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--near", type=int, default=10)
    parser.add_argument("--ctx", type=int, default=20000)
    parser.add_argument("--est", type=int, default=1)
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    started = time.time()
    os.makedirs(kt.OUT, exist_ok=True)
    if args.mode == "xfer" and not args.tag.startswith("x"):
        args.tag = "x" + args.tag
    if args.mode == "lomo":
        matches = kt.tb.built_matches()
        conds = {m: args.cond for m in matches}
        folds = [([o for o in matches if o != m], [m]) for m in (args.only or matches)]
    else:
        targets = kt.sc_matches()
        conds = {m: args.cond for m in kt.IDSSE}
        conds.update({m: args.target_cond for m in targets})
        folds = [(list(kt.IDSSE), targets)]
    tables = {m: kt.table(m, c, args.variant) for m, c in conds.items()}
    names = next(iter(tables.values()))[2]
    swap = kt.u_fit.make_swap(*names)
    players = {}
    for m, c in conds.items():
        players[m] = player_rows(m, c, tables[m][1], args.near)
    print("{} {}: {} matches, player rows per block {:.1f}, built in {:.0f} s".format(
        args.mode, args.cond, len(tables), np.mean([len(p[0]) / len(tables[m][1]["start"]) for m, p in players.items()]),
        time.time() - started), flush=True)
    rel = Relational()
    accs = []
    for train, tests in folds:
        t0 = time.time()
        # training pool: alive blocks of the training matches, both orientations
        xs, ys, brs, prs, off = [], [], [], [], 0
        for m in train:
            x, info = tables[m][0], tables[m][1]
            keep, y = kt.ta5.team_target(info["counts"])
            idx = np.flatnonzero(keep)
            br, pr = subset_rows(players[m][0], players[m][1], idx)
            for flip in (False, True):
                xs.append(swap(x[idx]) if flip else x[idx])
                ys.append(1 - y[idx] if flip else y[idx])
                p2 = pr.copy()
                if flip:
                    p2[:, 0] *= -1
                brs.append(br + off)
                prs.append(p2)
                off += len(idx)
        x = np.vstack(xs)
        y = np.concatenate(ys).astype(np.int64)
        br = np.concatenate(brs)
        pr = np.vstack(prs)
        cols = kt.select_columns(x, y, args.k, args.seed)
        cnames = ["c{}".format(i) for i in range(len(cols))]
        pick = np.sort(np.random.default_rng(args.seed).choice(len(y), size=min(args.ctx, len(y)), replace=False))
        cb, cp = subset_rows(br, pr, pick)
        ids = np.arange(len(pick))
        rel.fit(ids, x[pick][:, cols], y[pick], cb, cp, cnames, args.est)
        t1 = time.time()
        for m in tests:
            xt, info = tables[m][0], tables[m][1]
            nb = len(xt)
            out = {}
            for flip in (False, True):
                xx = swap(xt) if flip else xt
                pb, pp = players[m]
                pp = pp.copy()
                if flip:
                    pp[:, 0] *= -1
                res = np.empty(nb)
                for i in range(0, nb, args.batch):
                    sel = np.arange(i, min(nb, i + args.batch))
                    qb, qp = subset_rows(pb, pp, sel)
                    res[sel] = rel.predict(10_000_000 + sel, xx[sel][:, cols], qb, qp, cnames)
                out[flip] = res
            p1 = 0.5 * (out[False] + 1.0 - out[True])
            np.savez_compressed(os.path.join(kt.OUT, "{}_{}_{}.npz".format(conds[m], args.tag, m)), p1=p1, **info)
            keep, yt = kt.ta5.team_target(info["counts"])
            acc = float(np.mean((p1[keep] > 0.5) == (yt[keep] == 1)))
            pc = np.clip(p1[keep], 1e-4, 1 - 1e-4)
            ll = float(-np.mean(np.where(yt[keep] == 1, np.log(pc), np.log(1 - pc))))
            accs.append((acc, keep.sum(), ll))
            if args.mode == "lomo":
                print("  {} held out: team acc|alive {:.4f}, logloss {:.4f}, ctx {}, fit {:.0f} s, predict {:.0f} s".format(
                    m, acc, ll, len(pick), t1 - t0, time.time() - t1), flush=True)
        rel.model.clear()
    w = np.array([a[1] for a in accs], float)
    print("{} pooled team acc|alive {:.4f}, logloss {:.4f} over {} matches; {:.0f} s".format(
        args.tag, np.average([a[0] for a in accs], weights=w), np.average([a[2] for a in accs], weights=w),
        len(accs), time.time() - started), flush=True)


if __name__ == "__main__":
    main()
