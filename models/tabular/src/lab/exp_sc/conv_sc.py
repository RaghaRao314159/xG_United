"""IDSSE's DFL truth relabelled to the challenge's controlled-possession convention (Track C's rule
R2: an opposing segment that starts during play reverts to the previous team unless one of its
players stays within 1.5 m of the ball for k s), with k chosen to match SkillCorner's label
aggregates -- 20 matches labelled in the challenge's convention -- instead of the public window's
17 possessions. The criterion is the summed absolute gap in spells per alive-minute (relative) and
in the share of spells under 2 s and under 4 s. Builds runs/idsse-<m>-<cond>c/ via conv.py.

    MTAH_ROOT=/root/mtah python conv_sc.py
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

import numpy as np

sys.path.insert(0, LAB + "/exp")
sys.path.insert(0, LAB + "/exp_c")
from common import pipeline, tb  # noqa: E402
import conv  # noqa: E402

KS = (0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0)


def pooled(codes_by_match):
    lens, n, amin = [], 0, 0.0
    for c in codes_by_match:
        s = conv.stats(c)
        n += s["n"]
        amin += s["alive_min"]
        lens.append(np.array([b - a for a, b, _ in tb.pm.possession_spells(c, tb.pm.valid_runs(c))]))
    lens = np.concatenate(lens) / conv.FPS
    return {"per_min": n / amin, "u2": float(np.mean(lens < 2)), "u4": float(np.mean(lens < 4)),
            "median": float(np.median(lens))}


def line(label, s):
    return "| {} | {:.2f} | {:.0%} | {:.0%} | {:.1f} |".format(label, s["per_min"], s["u2"], s["u4"],
                                                              s["median"])


def main():
    argparse.ArgumentParser().parse_args()
    with open(os.path.join(pipeline.RUNS, "sc-stats.json")) as h:
        sc = json.load(h)["matches"]
    target = pooled([tb.gt_codes(tb.dataset_name(m, "clean")).astype(np.int8) for m in sc])
    truth, ctl = {}, {}
    for m in tb.built_matches():
        ds = tb.dataset_name(m, "clean")
        truth[m] = tb.gt_codes(ds).astype(np.int8)
        _, f = pipeline.load_dataset(ds)
        ctl[m] = conv.control_runs(f)
    print("| labels | spells per alive-min | < 2 s | < 4 s | median s |\n|---|---|---|---|---|")
    print(line("SkillCorner (target)", target))
    print(line("IDSSE DFL", pooled(truth.values())))
    best = None
    for k in KS:
        s = pooled([conv.relabel(truth[m], conv.keeper("R2", None, k, ctl[m])) for m in truth])
        err = (abs(s["per_min"] - target["per_min"]) / target["per_min"] + abs(s["u2"] - target["u2"])
               + abs(s["u4"] - target["u4"]))
        print(line("IDSSE R2 k={:g} (gap {:.3f})".format(k, err), s))
        if best is None or err < best[1]:
            best = (k, err)
    print("k = {:g} chosen".format(best[0]), flush=True)
    conv.cmd_build(argparse.Namespace(rule="R2", T=None, k=best[0], suffix="c"))


if __name__ == "__main__":
    main()
