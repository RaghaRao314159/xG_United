"""Track U: decode block team answers (u_fit.py caches) like Track AB, then score against references.

For each cached variant: P(alive) from the half-wide answers in the same dataset dirs (Track B's
model, retrained by u_half.py on the u copies), P(H | alive) = p1 (or p2), decoded by Track AB's
5 Hz Viterbi with separate H<->A / alive->D / D->alive costs, each held-out match using the grid
point that is best on the other six (ab_decode.decode_option). Then everything is pooled into
runs/exp-u-pooled-<cond> with the frozen-data references (same ground truth, asserted), scored
with the official scorer, paired-bootstrapped against REFERENCE, and broken down by possession
length and miss type. Prints aggregates only.

    MTAH_ROOT=/root/mtah python u_decode.py --cond cleanu --variants b
"""
import os as _os
ROOT = _os.environ.get("MTAH_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))))
LAB = _os.environ.get("MTAH_LAB") or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
LABCACHE = _os.path.join(ROOT, "runs", "lab-cache")

import argparse
import os
import re
import subprocess
import sys

import numpy as np

sys.path.insert(0, LAB + "/exp")
sys.path.insert(0, LAB + "/exp_a")
sys.path.insert(0, LAB + "/exp_ab")
from common import D, pipeline, tb  # noqa: E402
import ab_decode as abd  # noqa: E402
import diag  # noqa: E402

pm = tb.pm
X = pm.X
abd.CACHE = os.environ.get("U_CACHE", LABCACHE + "/exp_u")
REFERENCE = "exp_ab_b_tbh_p1"
FROZEN_REFS = ("exp_ab_b_tbh_p1", "exp_tb_half_wide_c01")


def load(match, cond, variant):
    dataset = tb.dataset_name(match, cond)
    meta, _ = pipeline.load_dataset(dataset)
    gt = tb.gt_codes(dataset).astype(np.int8)
    info = dict(np.load(os.path.join(abd.CACHE, "{}_{}_{}.npz".format(cond, variant, match))))
    first_of = np.zeros(len(info["start"]), dtype=np.int64)
    for half, first, _ in meta["halves"]:
        first_of[info["half"] == half] = first
    block = int(np.median(info["end"] - info["start"]))
    tbh = pipeline.load_answers(dataset, "exp_tb_half_wide")
    t = abd.lookup(tbh, info["start"], info["end"], first_of, abd.HALF_WINDOW, block // 2)
    return dict(dataset=dataset, gt=gt, info=info, runs=pm.valid_runs(gt),
                p_in={"tbh": t[:, 0]}, p_home_tbh=t[:, 1])


def pool(base, entries, matches, bootstrap, tag):
    """entries: [(name, cond)]; every name's per-match CSVs pooled with X separators."""
    folder = pipeline.dataset_dir("exp-u-pooled-" + tag)
    os.makedirs(os.path.join(folder, "results"), exist_ok=True)
    for old in os.listdir(os.path.join(folder, "results")):
        os.remove(os.path.join(folder, "results", old))
    sep = np.full(tb.SEPARATOR, "X")
    gts = []
    for m in matches:
        ref_gt = pm.read_label_csv(os.path.join(pipeline.dataset_dir(tb.dataset_name(m, base)),
                                                "gt.csv"))["label"].values
        for _, cond in entries:
            other = pm.read_label_csv(os.path.join(pipeline.dataset_dir(tb.dataset_name(m, cond)),
                                                   "gt.csv"))["label"].values
            assert (other == ref_gt).all(), "ground truth differs between {} and {}".format(base, cond)
        gts += [ref_gt, sep]
    pipeline.baseline.write_submission(os.path.join(folder, "gt.csv"), np.concatenate(gts[:-1]))
    for name, cond in entries:
        parts = []
        for m in matches:
            path = os.path.join(pipeline.dataset_dir(tb.dataset_name(m, cond)), "results",
                                name + ".csv")
            parts += [pm.read_label_csv(path)["label"].values, sep]
        out = name if cond == base else name + "@" + cond
        pipeline.baseline.write_submission(os.path.join(folder, "results", out + ".csv"),
                                           np.concatenate(parts[:-1]))
    tb.official_score(os.path.join(folder, "gt.csv"), os.path.join(folder, "results"),
                      os.path.join(folder, "score"), bootstrap)
    return folder


def report(folder, reference):
    board = tb.scoreboard(folder)
    gt = pm.label_codes(pm.read_label_csv(os.path.join(folder, "gt.csv"))).astype(np.int8)
    graded = gt != X
    runs = pm.valid_runs(gt)
    print("\n| model | spellF1 [5-95%] | macroF1 | alive/dead F1 | team acc | poss pred/true | "
          "D share pred (true {:.3f}) |".format(np.mean(gt[graded] == D)))
    print("|---|---|---|---|---|---|---|")
    preds = {}
    for name, row in sorted(board.items(), key=lambda kv: -float(kv[1]["spell_f1_mean"])):
        pred = pm.label_codes(pm.read_label_csv(os.path.join(folder, "results", name + ".csv")))
        preds[name] = pred.astype(np.int8)
        print("| {} | {:.3f} [{:.3f}, {:.3f}] | {:.3f} | {:.3f} | {:.3f} | {} / {} | {:.3f} |".format(
            name, float(row["spell_f1_mean"]), float(row["spell_bootstrap_low"]),
            float(row["spell_bootstrap_high"]), float(row["macro_f1"]),
            float(row["alive_dead_f1"]), float(row["team_acc_both_alive"]), row["poss"],
            row["n_gt_spells"], np.mean(preds[name][graded] == D)))
    sys.stdout.flush()
    out = subprocess.run([sys.executable, ROOT + "/src/paired_bootstrap.py", "--pair",
                          os.path.join(folder, "gt.csv"), os.path.join(folder, "results")],
                         check=False, capture_output=True, text=True).stdout
    print("\npaired vs {}:".format(reference))
    for line in out.splitlines():
        parts = line.split()
        if len(parts) > 3 and parts[1] == "-" and parts[2] == reference:
            print("  " + line)
    for name in sorted(preds, key=lambda n: -float(board[n]["spell_f1_mean"]))[:3]:
        _, flags, by_len, hit_len, by_start, hit_start, why, _, n_pred = diag.breakdown(
            gt, preds[name], runs)
        print("{}: recall <1/1-2/2-4/4-8 s {}/{}/{}/{} | after other team {:.3f}, after D {:.3f} | "
              "missed {}".format(name, *[round(hit_len[k] / by_len[k], 3) for k in
                                         ("0-1s", "1-2s", "2-4s", "4-8s")],
                                 hit_start["after other team"] / by_start["after other team"],
                                 hit_start["after D"] / by_start["after D"], dict(why)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cond", default="cleanu")
    parser.add_argument("--variants", nargs="+", default=["b"])
    parser.add_argument("--extra", nargs="*", default=[],
                        help="more name@cond entries to pool (already decoded)")
    parser.add_argument("--reference", default=REFERENCE)
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    base = re.sub(r"s?u$", "", args.cond)      # cleanu -> clean, d85u / d85su -> d85
    matches = tb.built_matches()
    entries = [(n, base) for n in FROZEN_REFS] + [("exp_u_tbh_c01", args.cond)]
    for variant in args.variants:
        if not all(os.path.exists(os.path.join(abd.CACHE, "{}_{}_{}.npz".format(args.cond, variant, m)))
                   for m in matches):
            print("  {} {}: cache incomplete, skipped".format(args.cond, variant), flush=True)
            continue
        data = {m: load(m, args.cond, variant) for m in matches}
        teams = ["p1"] + (["p2"] if "p2" in data[matches[0]]["info"] else [])
        for team in teams:
            label = "exp_u_{}_tbh_{}".format(variant, team)
            abd.decode_option(data, matches, "tbh", team, label)
            entries.append((label, args.cond))
    for item in args.extra:
        name, _, cond = item.partition("@")
        entries.append((name, cond or args.cond))
    p_d = []
    for m in matches:
        ds = tb.dataset_name(m, args.cond)
        answers = pipeline.load_answers(ds, "exp_tb_half_wide")
        p_d.append(1 - np.array([v[0] for v in answers.values()]))
    print("{}: mean P(D) of the half-wide answers {:.3f}".format(args.cond, np.concatenate(p_d).mean()))
    folder = pool(base, entries, matches, args.bootstrap, args.cond)
    report(folder, args.reference)


if __name__ == "__main__":
    main()
