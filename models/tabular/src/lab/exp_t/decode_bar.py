"""Track T stage 3: decode BAR-ATH with Track A's 5 Hz structured Viterbi. P(alive) per 0.2 s block
from answers_exp_t_tbh_<palive>.jsonl (12-frame windows, looked up at the block's middle frame, as
ab_decode.py does), P(H | alive) from team_bar.py's cache. Costs (c_HA, c_out, c_in) are fixed
from IDSSE d85 LOMO (Track AB: (1, 2, 10) in 4 of 7 folds and in-sample best); _alt is the
other d85 fold choice (1, 0.25, 20). Writes runs/BAR-ATH/results/<name>.csv and prints
ground-truth-free aggregates. Aggregates only.

    MTAH_ROOT=/root/mtah python decode_bar.py --team d85 --palive d85 d70 d95
"""
import os as _os
ROOT = _os.environ.get("MTAH_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))))
LAB = _os.environ.get("MTAH_LAB") or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
LABCACHE = _os.path.join(ROOT, "runs", "lab-cache")

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, LAB + "/exp")
sys.path.insert(0, LAB + "/exp_a")
sys.path.insert(0, LAB + "/exp_ab")
from common import pipeline, tb  # noqa: E402
import bar  # noqa: E402
import ta5_decode as td  # noqa: E402
from ab_decode import lookup  # noqa: E402

pm = tb.pm
FPS = pipeline.FPS
CACHE = LABCACHE + "/exp_t"
HALF_WINDOW = 12
COSTS = {"": (1.0, 2.0, 10.0), "_alt": (1.0, 0.25, 20.0)}


def aggregates(name, codes, meta, f):
    runs = [(first, last + 1) for _, first, last in meta["halves"]]
    graded = np.zeros(len(codes), dtype=bool)
    for a, b in runs:
        graded[a:b] = True
    spells = pm.possession_spells(codes, runs)
    lengths = np.array([b - a for a, b, _ in spells]) / float(FPS)
    alive = graded & ((codes == pm.H) | (codes == pm.A))
    alive_min = alive.sum() / FPS / 60.0
    long_gap = np.zeros(len(codes), dtype=bool)
    for a, b in runs:
        starts, stops = tb.run_bounds(~f["seen"][a:b].astype(bool))
        for s, e in zip(starts, stops):
            if e - s > 10 * FPS:
                long_gap[a + s:a + e] = True
    prekick = codes[110808:113742]
    return ("| {} | {:.1%} | {:.1%} | {:.1%} | {} | {:.1f} | {:.1f} | {:.1f} | {:.0%} | {:.0%} |"
            .format(name, np.mean(codes[graded] == pm.D), np.mean(codes[long_gap] == pm.D),
                    np.mean(prekick == pm.D), len(spells), len(spells) / alive_min,
                    lengths.mean(), np.median(lengths), np.mean(lengths < 2),
                    np.mean(lengths < 4)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--team", default="d85")
    parser.add_argument("--palive", nargs="+", default=["d85"])
    parser.add_argument("--suffix", default="")
    args = parser.parse_args()
    meta, f = pipeline.load_dataset(bar.TARGET)
    info = dict(np.load(os.path.join(CACHE, "bar_b_{}{}.npz".format(args.team, args.suffix))))
    first_of = np.zeros(len(info["start"]), dtype=np.int64)
    for half, first, _ in meta["halves"]:
        first_of[info["half"] == half] = first
    block = int(np.median(info["end"] - info["start"]))
    print("| output | D share | D in >10 s blackouts | D pre-kick-off 2nd half | possessions | per "
          "alive-min | mean s | median s | < 2 s | < 4 s |\n|---|---|---|---|---|---|---|---|---|---|")
    for palive in args.palive:
        answers = pipeline.load_answers(bar.TARGET, "exp_t_tbh_{}{}".format(palive, args.suffix))
        t = lookup(answers, info["start"], info["end"], first_of, HALF_WINDOW, block // 2)
        p_in, p_home = t[:, 0], info["p1"]
        emission = np.column_stack([p_in * p_home, p_in * (1 - p_home), 1 - p_in])
        le = np.log(np.clip(emission, td.FLOOR, 1.0))
        for tag, cost in COSTS.items():
            if tag and palive != "d85":
                continue
            costs = td.cost_matrix(*cost)[None]
            paths = np.empty((1, len(le)), dtype=np.int8)
            for h in np.unique(info["half"]):
                idx = np.flatnonzero(info["half"] == h)
                paths[:, idx] = td.viterbi_grid(le[idx], costs)
            codes = td.frame_codes(paths, info)[0]
            name = "exp_ab_b{}{}{}".format("" if palive == "d85" else "_p" + palive, tag,
                                          args.suffix)
            td.write(bar.TARGET, name, codes)
            print(aggregates(name, codes, meta, f), flush=True)


if __name__ == "__main__":
    main()
