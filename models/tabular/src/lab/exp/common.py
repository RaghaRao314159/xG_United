"""Shared bits of the scratchpad LightGBM experiments: imports, decoding at several switch costs,
and pooled scoring restricted to named submissions (other jobs write into the same results dirs).
Prints aggregates only."""
import os as _os
ROOT = _os.environ.get("MTAH_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))))
LAB = _os.environ.get("MTAH_LAB") or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
LABCACHE = _os.path.join(ROOT, "runs", "lab-cache")

import argparse
import os
import subprocess
import sys

import numpy as np

ROOT = os.environ.get("MTAH_ROOT", ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))  # xG United: folder renamed from jev_track
import tabular as tb  # noqa: E402
import pipeline  # noqa: E402

H, A, D = tb.H, tb.A, tb.D
COSTS = (3.0, 0.1)
tb.LGBM["n_jobs"] = int(os.environ.get("MTAH_LGBM_JOBS", 4))  # the same for every variant compared
POOL = "exp-lgbm-pooled-{}"


def cost_name(name, cost):
    return name if cost == 3.0 else "{}_c{}".format(name, str(cost).replace(".", ""))


def decode_all(dataset, name):
    for cost in COSTS:
        pipeline.SWITCH_COST = cost
        pipeline.cmd_decode(argparse.Namespace(dataset=dataset, name=name, vote="mean",
                                               out_name=cost_name(name, cost)))
    pipeline.SWITCH_COST = 3.0


def pooled(cond, matches, keep, bootstrap=1000):
    """tabular.pooled, but only for submissions whose name starts with one of `keep`, into its
    own folder; returns the scoreboard rows."""
    folder = pipeline.dataset_dir(POOL.format(cond))
    os.makedirs(os.path.join(folder, "results"), exist_ok=True)
    for old in os.listdir(os.path.join(folder, "results")):
        os.remove(os.path.join(folder, "results", old))
    sep = np.full(tb.SEPARATOR, "X")
    gts, names = [], None
    for match in matches:
        here_dir = pipeline.dataset_dir(tb.dataset_name(match, cond))
        gts.append(tb.pm.read_label_csv(os.path.join(here_dir, "gt.csv"))["label"].values)
        here = {n[:-4] for n in os.listdir(os.path.join(here_dir, "results"))
                if n.endswith(".csv") and n.startswith(tuple(keep))}
        names = here if names is None else names & here
    pipeline.baseline.write_submission(os.path.join(folder, "gt.csv"), np.concatenate(
        [part for g in gts for part in (g, sep)][:-1]))
    for name in sorted(names):
        parts = []
        for match in matches:
            path = os.path.join(pipeline.dataset_dir(tb.dataset_name(match, cond)), "results",
                                name + ".csv")
            parts += [tb.pm.read_label_csv(path)["label"].values, sep]
        pipeline.baseline.write_submission(os.path.join(folder, "results", name + ".csv"),
                                           np.concatenate(parts[:-1]))
    tb.official_score(os.path.join(folder, "gt.csv"), os.path.join(folder, "results"),
                      os.path.join(folder, "score"), bootstrap)
    return folder


def print_pooled(folder):
    board = tb.scoreboard(folder)
    print("\n| model | spellF1 pooled [5-95%] | macroF1 | alive/dead F1 | team acc | poss "
          "pred/true |\n|---|---|---|---|---|---|")
    for name, row in sorted(board.items(), key=lambda item: -float(item[1]["spell_f1_mean"])):
        print("| {} | {:.3f} [{:.3f}, {:.3f}] | {:.3f} | {:.3f} | {:.3f} | {} / {} |".format(
            name, float(row["spell_f1_mean"]), float(row["spell_bootstrap_low"]),
            float(row["spell_bootstrap_high"]), float(row["macro_f1"]),
            float(row["alive_dead_f1"]), float(row["team_acc_both_alive"]), row["poss"],
            row["n_gt_spells"]))


def paired(folder):
    subprocess.run([sys.executable, os.path.join(ROOT, "src", "paired_bootstrap.py"),
                    "--pair", os.path.join(folder, "gt.csv"), os.path.join(folder, "results")],
                   check=False)
