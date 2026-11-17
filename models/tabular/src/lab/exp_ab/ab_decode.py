"""Track AB: decode ab_fit.py's block team answers at block resolution, costs chosen LOMO, score.

For every variant (ab, b, ab_half) and every option (P(alive) source, P(H | alive) source):
    P(alive)       ref   exp_ctx_stack's p_in (per second)
                   tbh   exp_tb_half_wide's p_in (Track B, 12-frame windows)
    P(H | alive)   p1 / p2   ab_fit's first / stacked pass
                   mix       mean logit of p1 and exp_tb_half_wide's p_home
                   tbh       exp_tb_half_wide's p_home alone (decoder-only ablation)
the blocks are decoded by a Viterbi with separate H<->A, alive->D and D->alive costs over a grid;
each held-out match uses the grid point with the best pooled spellF1 on the other six. All
P(alive) / P(H) inputs for a held-out match come from models trained without it.
Results go to runs/idsse-*-<cond>/results/exp_ab_<variant>_<palive>_<team>.csv, pooled with the
references into runs/exp-ab-pooled-<cond> (official scorer), paired bootstrap, recall by length.
Prints aggregates only.

    MTAH_ROOT=/root/mtah python ab_decode.py --cond clean --variants ab b ab_half
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
sys.path.insert(0, LAB + "/exp_a")
from common import pipeline, tb  # noqa: E402
import diag  # noqa: E402
import ta5_decode as td  # noqa: E402

pm = tb.pm
X = pm.X
CACHE = LABCACHE + "/exp_ab"
C_HA = (0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 8.0)
C_OUT = (0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0)
C_IN = (2.0, 5.0, 10.0, 20.0, 40.0, 80.0)
GRID = list(itertools.product(C_HA, C_OUT, C_IN))
COSTS = np.stack([td.cost_matrix(*g) for g in GRID])
REFS = ("exp_tb_half_wide_c01", "exp_ta_big_p2_5hz", "exp_ctx_stack_c01")
REFERENCE = "exp_tb_half_wide_c01"
HALF_WINDOW = 12


def lookup(answers, starts, ends, first_of, width, offset):
    """Per block: the answer of the `width`-frame window holding the block's frame start+offset
    (clamped to the block's last frame)."""
    pos = np.minimum(starts + offset, ends - 1)
    keys = first_of + ((pos - first_of) // width) * width
    return np.array([answers[int(k)] for k in keys])


def load(match, cond, variant):
    dataset = tb.dataset_name(match, cond)
    meta, f = pipeline.load_dataset(dataset)
    gt = tb.gt_codes(dataset).astype(np.int8)
    info = dict(np.load(os.path.join(CACHE, "{}_{}_{}.npz".format(cond, variant, match))))
    first_of = np.zeros(len(info["start"]), dtype=np.int64)
    for half, first, _ in meta["halves"]:
        first_of[info["half"] == half] = first
    block = int(np.median(info["end"] - info["start"]))
    ref = pipeline.load_answers(dataset, "exp_ctx_stack")
    tbh = pipeline.load_answers(dataset, "exp_tb_half_wide")
    r = lookup(ref, info["start"], info["end"], first_of, pipeline.WINDOW, 0)
    t = lookup(tbh, info["start"], info["end"], first_of, HALF_WINDOW, block // 2)
    return dict(dataset=dataset, gt=gt, info=info, runs=pm.valid_runs(gt),
                p_in={"ref": r[:, 0], "tbh": t[:, 0]}, p_home_tbh=t[:, 1])


def team_of(d, name):
    if name in ("p1", "p2"):
        return d["info"][name]
    if name == "tbh":
        return d["p_home_tbh"]
    if name == "mix":
        return 1 / (1 + np.exp(-0.5 * (td.logit(d["info"]["p1"]) + td.logit(d["p_home_tbh"]))))
    raise ValueError(name)


def decode_option(data, matches, palive, team, label):
    per_match = {}
    for m in matches:
        d = data[m]
        p_in, p_home = d["p_in"][palive], team_of(d, team)
        emission = np.column_stack([p_in * p_home, p_in * (1 - p_home), 1 - p_in])
        le = np.log(np.clip(emission, td.FLOOR, 1.0))
        info = d["info"]
        paths = np.empty((len(GRID), len(le)), dtype=np.int8)
        for h in np.unique(info["half"]):
            idx = np.flatnonzero(info["half"] == h)
            paths[:, idx] = td.viterbi_grid(le[idx], COSTS)
        codes = td.frame_codes(paths, info)
        gt_sp = td.spells(d["gt"], d["runs"])
        per_match[m] = (codes, [td.counts(gt_sp, td.spells(np.where(d["gt"] == X, X, c),
                                                            d["runs"])) for c in codes])
    insample = [td.pooled_f1([per_match[m][1][g] for m in matches]) for g in range(len(GRID))]
    best = int(np.argmax(insample))
    rows, chosen = [], []
    for m in matches:
        others = [o for o in matches if o != m]
        g = int(np.argmax([td.pooled_f1([per_match[o][1][g] for o in others])
                           for g in range(len(GRID))]))
        chosen.append(GRID[g])
        rows.append(per_match[m][1][g])
        td.write(data[m]["dataset"], label, per_match[m][0][g])
    tally = {}
    for c in chosen:
        tally[str(c)] = tally.get(str(c), 0) + 1
    print("  {}: LOMO spellF1 {:.3f} (in-sample best {:.3f} at c_HA, c_out, c_in = {}); per "
          "fold {}".format(label, td.pooled_f1(rows), insample[best], GRID[best], tally),
          flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cond", default="clean")
    parser.add_argument("--variants", nargs="+", default=["ab", "b", "ab_half"])
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    matches = tb.built_matches()
    names = set(REFS)
    for variant in args.variants:
        if not all(os.path.exists(os.path.join(CACHE, "{}_{}_{}.npz".format(args.cond, variant, m)))
                   for m in matches):
            print("  {} {}: cache incomplete, skipped".format(args.cond, variant), flush=True)
            continue
        data = {m: load(m, args.cond, variant) for m in matches}
        teams = ["p1"] + (["p2"] if "p2" in data[matches[0]]["info"] else [])
        options = [("tbh", t) for t in teams] + [("ref", "p1"), ("tbh", "mix")]
        if variant == "ab":
            options.append(("tbh", "tbh"))
        for palive, team in options:
            label = "exp_ab_{}_{}_{}".format(variant, palive, team)
            decode_option(data, matches, palive, team, label)
            names.add(label)

    td.POOL_TAG = ""
    folder = pipeline.dataset_dir("exp-ab-pooled-" + args.cond)
    os.makedirs(os.path.join(folder, "results"), exist_ok=True)
    for old in os.listdir(os.path.join(folder, "results")):
        os.remove(os.path.join(folder, "results", old))
    sep = np.full(tb.SEPARATOR, "X")
    gts = []
    for m in matches:
        d = pipeline.dataset_dir(tb.dataset_name(m, args.cond))
        gts += [pm.read_label_csv(os.path.join(d, "gt.csv"))["label"].values, sep]
    pipeline.baseline.write_submission(os.path.join(folder, "gt.csv"), np.concatenate(gts[:-1]))
    for name in sorted(names):
        parts = []
        for m in matches:
            d = pipeline.dataset_dir(tb.dataset_name(m, args.cond))
            parts += [pm.read_label_csv(os.path.join(d, "results", name + ".csv"))["label"].values,
                      sep]
        pipeline.baseline.write_submission(os.path.join(folder, "results", name + ".csv"),
                                           np.concatenate(parts[:-1]))
    tb.official_score(os.path.join(folder, "gt.csv"), os.path.join(folder, "results"),
                      os.path.join(folder, "score"), args.bootstrap)
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
    out = subprocess.run([sys.executable, ROOT + "/src/paired_bootstrap.py", "--pair",
                          os.path.join(folder, "gt.csv"), os.path.join(folder, "results")],
                         check=False, capture_output=True, text=True).stdout
    print("\npaired vs {}:".format(REFERENCE))
    for line in out.splitlines():
        parts = line.split()
        if len(parts) > 3 and parts[1] == "-" and parts[2] == REFERENCE:
            print("  " + line)

    best = max((n for n in names if n.startswith("exp_ab_")),
               key=lambda n: float(board[n]["spell_f1_mean"]))
    gt = pm.label_codes(pm.read_label_csv(os.path.join(folder, "gt.csv"))).astype(np.int8)
    runs = pm.valid_runs(gt)
    for name in (best, REFERENCE):
        pred = pm.label_codes(pm.read_label_csv(os.path.join(folder, "results", name + ".csv")))
        _, flags, by_len, hit_len, by_start, hit_start, why, _, n_pred = diag.breakdown(
            gt, pred.astype(np.int8), runs)
        print("\n{} at IoU 0.5: recall by length {} | by start {} | missed {}".format(
            name, {k: round(hit_len[k] / by_len[k], 3) for k in sorted(by_len)},
            {k: round(hit_start[k] / v, 3) for k, v in by_start.items()}, dict(why)))


if __name__ == "__main__":
    main()
