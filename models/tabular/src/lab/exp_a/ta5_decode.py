"""Track A: decode ta5.py's team answers, choose switch costs leave-one-match-out, score.

P(alive) is the reference stack's (answers_exp_ctx_stack, per second, repeated over its five
blocks); P(H | alive) is one of
    ref   the reference's own p_home, repeated       (5 Hz decoding alone)
    p1    ta5 first pass                             (control/touch features)
    p2    ta5 second pass (stacked)
    mix   mean logit of p2 and ref
Each is decoded at 5 Hz by a Viterbi with separate costs for H<->A, alive->D and D->alive over a
grid; the costs for each held-out match are the grid point with the best pooled spellF1 on the
other six (leave one match out), and that match is decoded with them. p2 is also averaged per
second and decoded exactly like the reference (1 Hz, cost 0.1, snap) as exp_ta_p2_1hz.
Results go to runs/idsse-*-<cond>/results/exp_ta_*.csv, pooled into runs/exp-ta-pooled-<cond>
with the reference, official scorer and paired bootstrap. Prints aggregates only.
"""
import os as _os
ROOT = _os.environ.get("MTAH_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))))
LAB = _os.environ.get("MTAH_LAB") or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
LABCACHE = _os.path.join(ROOT, "runs", "lab-cache")

import argparse
import itertools
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, LAB + "/exp")
from common import A, D, H, pipeline, tb  # noqa: E402
import diag  # noqa: E402

pm = tb.pm
X = pm.X
CACHE = LABCACHE + "/exp_a"
C_HA = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
C_OUT = (0.5, 1.0, 2.0, 5.0)
C_IN = (0.5, 2.0, 5.0, 10.0, 20.0)
GRID = list(itertools.product(C_HA, C_OUT, C_IN))
IOUS = pm.SPELL_IOUS
FLOOR = pipeline.PROB_FLOOR
REFS = ("exp_ctx_stack_c01", "exp_3class_c01")
POOL_TAG = ""


def cost_matrix(c_ha, c_out, c_in):
    m = np.zeros((3, 3))
    m[H, A] = m[A, H] = c_ha
    m[H, D] = m[A, D] = c_out
    m[D, H] = m[D, A] = c_in
    return m


COSTS = np.stack([cost_matrix(*g) for g in GRID])


def viterbi_grid(le, costs):
    """Best path for every cost matrix at once: (G, n) int8."""
    n, g = len(le), len(costs)
    step = -costs                                   # (G, from, to)
    score = np.repeat(le[:1], g, axis=0)            # (G, 3)
    back = np.empty((n, g, 3), dtype=np.int8)
    for t in range(1, n):
        cand = score[:, :, None] + step
        back[t] = cand.argmax(axis=1)
        score = cand.max(axis=1) + le[t]
    path = np.empty((g, n), dtype=np.int8)
    path[:, -1] = score.argmax(axis=1)
    rows = np.arange(g)
    for t in range(n - 1, 0, -1):
        path[:, t - 1] = back[t, rows, path[:, t]]
    return path


def spells(codes, runs):
    return pm.possession_spells(codes, runs)


def counts(gt_spells, pred_spells):
    """Matched counts per IoU threshold, exactly pm.spell_scores' greedy matching."""
    pairs, j0 = [], 0
    for i, (ga, gb, gk) in enumerate(gt_spells):
        while j0 < len(pred_spells) and pred_spells[j0][1] <= ga:
            j0 += 1
        j = j0
        while j < len(pred_spells) and pred_spells[j][0] < gb:
            pa, pb, pk = pred_spells[j]
            if pk == gk:
                pairs.append(((min(gb, pb) - max(ga, pa)) / float(max(gb, pb) - min(ga, pa)), i, j))
            j += 1
    pairs.sort(key=lambda item: (-item[0], item[1], item[2]))
    out = []
    for thr in IOUS:
        out.append(int(pm.matched_gt_flags(pairs, len(gt_spells), thr).sum()))
    return np.array(out), len(pred_spells), len(gt_spells)


def pooled_f1(rows):
    """rows: list of (tp per IoU, n_pred, n_gt) -> mean F1 over IoUs of the summed counts."""
    tp = np.sum([r[0] for r in rows], axis=0).astype(float)
    n_pred = sum(r[1] for r in rows)
    n_gt = sum(r[2] for r in rows)
    p = tp / max(n_pred, 1)
    r = tp / max(n_gt, 1)
    f = np.where(p + r > 0, 2 * p * r / np.maximum(p + r, 1e-12), 0.0)
    return float(f.mean())


def frame_codes(paths, info):
    """(G, n_frames) labels from block paths; X outside the halves."""
    out = np.full((len(paths), int(info["n_frames"])), X, dtype=np.int8)
    for b, (a, e) in enumerate(zip(info["start"], info["end"])):
        out[:, a:e] = paths[:, b:b + 1]
    return out


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def write(dataset, name, codes):
    letters = np.array(["H", "A", "D", "X"])
    path = os.path.join(pipeline.dataset_dir(dataset), "results", name + ".csv")
    pipeline.baseline.write_submission(path, letters[codes])


def pooled_folder(cond, matches, names, bootstrap):
    folder = pipeline.dataset_dir("exp-ta-pooled-" + cond + POOL_TAG)
    os.makedirs(os.path.join(folder, "results"), exist_ok=True)
    for old in os.listdir(os.path.join(folder, "results")):
        os.remove(os.path.join(folder, "results", old))
    sep = np.full(tb.SEPARATOR, "X")
    gts = []
    for m in matches:
        d = pipeline.dataset_dir(tb.dataset_name(m, cond))
        gts += [pm.read_label_csv(os.path.join(d, "gt.csv"))["label"].values, sep]
    pipeline.baseline.write_submission(os.path.join(folder, "gt.csv"), np.concatenate(gts[:-1]))
    for name in names:
        parts = []
        for m in matches:
            d = pipeline.dataset_dir(tb.dataset_name(m, cond))
            parts += [pm.read_label_csv(os.path.join(d, "results", name + ".csv"))["label"].values,
                      sep]
        pipeline.baseline.write_submission(os.path.join(folder, "results", name + ".csv"),
                                           np.concatenate(parts[:-1]))
    tb.official_score(os.path.join(folder, "gt.csv"), os.path.join(folder, "results"),
                      os.path.join(folder, "score"), bootstrap)
    return folder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cond", default="clean")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    tag = "_" + args.tag if args.tag else ""
    global POOL_TAG
    POOL_TAG = "-" + args.tag if args.tag else ""
    matches = tb.built_matches()
    data = {}
    for m in matches:
        dataset = tb.dataset_name(m, args.cond)
        meta, f = pipeline.load_dataset(dataset)
        gt = tb.gt_codes(dataset).astype(np.int8)
        info = dict(np.load(os.path.join(CACHE, "{}{}_{}.npz".format(args.cond, tag, m))))
        answers = pipeline.load_answers(dataset, "exp_ctx_stack")
        wins = pipeline.windows(meta)
        win_start = np.array([wins[w][2] for w in info["win"]])
        p_in = np.array([answers[s][0] for s in win_start])
        ref = np.array([answers[s][1] for s in win_start])
        data[m] = dict(dataset=dataset, meta=meta, f=f, gt=gt, info=info, p_in=p_in, ref=ref,
                       runs=pm.valid_runs(gt))
    teams = {"ref": lambda d: d["ref"], "p1": lambda d: d["info"]["p1"],
             "p2": lambda d: d["info"]["p2"],
             "mix": lambda d: 1 / (1 + np.exp(-0.5 * (logit(d["info"]["p2"]) + logit(d["ref"]))))}
    if tag:
        teams = {"p1": teams["p1"], "p2": teams["p2"]}

    results = {}
    for tname, team in teams.items():
        per_match = {}
        for m in matches:
            d = data[m]
            p_home = team(d)
            emission = np.column_stack([d["p_in"] * p_home, d["p_in"] * (1 - p_home),
                                        1 - d["p_in"]])
            le = np.log(np.clip(emission, FLOOR, 1.0))
            paths = np.empty((len(GRID), len(le)), dtype=np.int8)
            info = d["info"]
            for h in np.unique(info["half"]):
                idx = np.flatnonzero(info["half"] == h)
                paths[:, idx] = viterbi_grid(le[idx], COSTS)
            codes = frame_codes(paths, info)
            gt_sp = spells(d["gt"], d["runs"])
            per_match[m] = (codes, [counts(gt_sp, spells(np.where(d["gt"] == X, X, c), d["runs"]))
                                    for c in codes])
        insample = [pooled_f1([per_match[m][1][g] for m in matches]) for g in range(len(GRID))]
        best = int(np.argmax(insample))
        chosen, rows = {}, []
        for m in matches:
            others = [o for o in matches if o != m]
            g = int(np.argmax([pooled_f1([per_match[o][1][g] for o in others])
                               for g in range(len(GRID))]))
            chosen[m] = GRID[g]
            rows.append(per_match[m][1][g])
            write(data[m]["dataset"], "exp_ta{}_{}_5hz".format(tag, tname), per_match[m][0][g])
        results[tname] = pooled_f1(rows)
        print("{} {}: LOMO-chosen 5 Hz spellF1 {:.3f} (in-sample best {:.3f} at c_HA, c_out, c_in "
              "= {}); chosen per fold: {}".format(
                  args.cond, tname, results[tname], insample[best], GRID[best],
                  dict(sorted(((str(v), list(chosen.values()).count(v)) for v in chosen.values())))),
              flush=True)

    # p2 averaged per second, decoded like the reference
    for m in matches:
        d = data[m]
        sums, n = {}, {}
        for w, p in zip(d["info"]["win"], d["info"]["p2"]):
            sums[w] = sums.get(w, 0.0) + p
            n[w] = n.get(w, 0) + 1
        wins = pipeline.windows(d["meta"])
        answers = pipeline.load_answers(d["dataset"], "exp_ctx_stack")
        mine = {wins[w][2]: (answers[wins[w][2]][0], sums[w] / n[w]) for w in sums}
        write(d["dataset"], "exp_ta{}_p2_1hz".format(tag), diag.decode(d["meta"], d["f"], mine, 0.1))

    names = sorted({"exp_ta{}_{}_5hz".format(tag, t) for t in teams} | {"exp_ta{}_p2_1hz".format(tag)}
                   | set(REFS) | ({"exp_ta_p2_5hz"} if tag else set()))
    folder = pooled_folder(args.cond, matches, names, args.bootstrap)
    board = tb.scoreboard(folder)
    print("\n| {} | spellF1 [5-95%] | macroF1 | alive/dead F1 | team acc | poss pred/true |".format(
        args.cond))
    print("|---|---|---|---|---|---|")
    for name, row in sorted(board.items(), key=lambda kv: -float(kv[1]["spell_f1_mean"])):
        print("| {} | {:.3f} [{:.3f}, {:.3f}] | {:.3f} | {:.3f} | {:.3f} | {} / {} |".format(
            name, float(row["spell_f1_mean"]), float(row["spell_bootstrap_low"]),
            float(row["spell_bootstrap_high"]), float(row["macro_f1"]),
            float(row["alive_dead_f1"]), float(row["team_acc_both_alive"]), row["poss"],
            row["n_gt_spells"]))
    sys.stdout.flush()
    subprocess.run([sys.executable, ROOT + "/src/paired_bootstrap.py", "--pair",
                    os.path.join(folder, "gt.csv"), os.path.join(folder, "results")], check=False)

    best_name = max((n for n in names if n.startswith("exp_ta_")),
                    key=lambda n: float(board[n]["spell_f1_mean"]))
    gt = pm.label_codes(pm.read_label_csv(os.path.join(folder, "gt.csv"))).astype(np.int8)
    runs = pm.valid_runs(gt)
    for name in (best_name, "exp_ctx_stack_c01"):
        pred = pm.label_codes(pm.read_label_csv(os.path.join(folder, "results", name + ".csv")))
        _, flags, by_len, hit_len, by_start, hit_start, why, _, n_pred = diag.breakdown(
            gt, pred.astype(np.int8), runs)
        print("\n{} at IoU 0.5: recall by length {} | by start {} | missed {}".format(
            name, {k: round(hit_len[k] / by_len[k], 3) for k in sorted(by_len)},
            {k: round(hit_start[k] / v, 3) for k, v in by_start.items()}, dict(why)))


if __name__ == "__main__":
    main()
