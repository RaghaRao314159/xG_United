"""Paired bootstrap of spellF1 between submissions scored against the same ground truth.

The scorer's --bootstrap interval resamples each submission's possessions on its own, so two
overlapping intervals cannot say whether one submission beats another. Resampling the same true
possessions for every submission at once can: each draw scores all of them on one identical
sample, so possessions every submission gets right (or wrong) cancel out of the difference, and
the spread of that difference answers "is this gap real".

The per-draw score is exactly possession_metrics.bootstrap_spell_f1's: matched-or-not flags per
true possession and IoU threshold, true positives summed over the drawn possessions, the predicted
possession count held fixed.

    python src/paired_bootstrap.py --pair runs/idsse-J03WMX/gt.csv runs/idsse-J03WMX/results \
                                         --pair runs/idsse-J03WN1/gt.csv runs/idsse-J03WN1/results

Several --pair arguments pool matches, as the leave-one-match-out tables do: true positives,
predicted and true possession counts add up across matches, and each draw resamples possessions
within each match so that every draw keeps the real mix of matches. Only submissions present for
every match are compared.

Prints aggregates only.
"""

import argparse
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CHALLENGE_EVALUATION = os.path.abspath(os.path.join(ROOT, "..", "..", "challenge", "evaluation"))
sys.path.insert(0, CHALLENGE_EVALUATION)
import possession_metrics as pm  # noqa: E402
import run_evaluation as scorer  # noqa: E402  (find_gt: 'public', 'private' or a path)

# Params
DRAWS = 10000
SEED = 0
PERCENTILES = (5, 95)  # a 90% interval, the same band the scorer reports


def flags_of(gt, pred, runs):
    """(thresholds x true possessions) matched flags and the predicted possession count."""
    filled = pm.fill_abstentions(pred, runs)
    gt_spells = pm.possession_spells(gt, runs)
    pred_spells = pm.possession_spells(filled, runs)
    overlaps = pm.spell_overlaps(gt_spells, pred_spells)
    flags = np.vstack([pm.matched_gt_flags(overlaps, len(gt_spells), threshold)
                       for threshold in pm.SPELL_IOUS])
    return flags, len(pred_spells)


def scores(flags, n_pred, picks):
    """spellF1 for each row of picks, a (draws x n_gt) array of resampled possession indices."""
    true_positive = flags[:, picks].sum(axis=2)            # thresholds x draws
    false_negative = picks.shape[1] - true_positive
    false_positive = np.maximum(0, n_pred - true_positive)
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = np.where(true_positive > 0,
                      2.0 * true_positive / (2 * true_positive + false_positive + false_negative),
                      0.0)
    return f1.mean(axis=0)


def load_pair(gt_name, results):
    """{submission: (flags, n_pred)} for one ground truth and its results folder."""
    gt = pm.label_codes(pm.read_label_csv(scorer.find_gt(gt_name)))
    runs = pm.valid_runs(gt)
    out = {}
    for path in sorted(glob.glob(os.path.join(results, "*.csv"))):
        table = pm.read_label_csv(path)
        if pm.validate_submission(table, len(gt)):
            continue
        out[os.path.splitext(os.path.basename(path))[0]] = flags_of(gt, pm.label_codes(table), runs)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pair", nargs=2, action="append", required=True, metavar=("GT", "RESULTS"),
                        help="a ground truth ('public', 'private' or a path) and its results "
                             "folder; repeat to pool external matches")
    parser.add_argument("--draws", type=int, default=DRAWS)
    args = parser.parse_args()
    pairs = args.pair

    loaded = [load_pair(gt, results) for gt, results in pairs]
    common = sorted(set.intersection(*(set(d) for d in loaded)))
    names = list(common)
    flags = [np.hstack([d[name][0] for d in loaded]) for name in names]
    counts = [sum(d[name][1] for d in loaded) for name in names]
    sizes = [d[names[0]][0].shape[1] for d in loaded]
    n_gt = sum(sizes)

    full = np.arange(n_gt)[None, :]
    observed = np.array([scores(f, c, full)[0] for f, c in zip(flags, counts)])
    order = np.argsort(-observed)
    rng = np.random.RandomState(SEED)
    offsets = np.cumsum([0] + sizes[:-1])
    picks = np.hstack([offset + rng.randint(0, size, size=(args.draws, size))
                       for offset, size in zip(offsets, sizes)])
    draws = np.vstack([scores(flags[i], counts[i], picks) for i in order])
    names = [names[i] for i in order]
    observed = observed[order]

    print("{} true possessions over {} match(es), {} paired draws, {}-{} percentile intervals".format(
        n_gt, len(pairs), args.draws, *PERCENTILES))
    print("\n{:<20} {:>8} {:>17} {:>6}".format("submission", "spellF1", "unpaired 90%", "poss"))
    for k, name in enumerate(names):
        low, high = np.percentile(draws[k], PERCENTILES)
        print("{:<20} {:>8.3f}   [{:.3f}, {:.3f}] {:>6}".format(
            name, observed[k], low, high, counts[order[k]]))

    print("\npairs: row - column | observed difference | 90% interval of the paired difference | "
          "share of draws row > column, row = column")
    for a in range(len(names)):
        for b in range(len(names)):
            if a == b:
                continue
            diff = draws[a] - draws[b]
            low, high = np.percentile(diff, PERCENTILES)
            verdict = "row better" if low > 0 else "row worse" if high < 0 else "not separable"
            print("{:<18} - {:<18} {:+.3f}  [{:+.3f}, {:+.3f}]  wins {:5.1%}  ties {:5.1%}  {}".format(
                names[a], names[b], observed[a] - observed[b], low, high,
                np.mean(diff > 0), np.mean(diff == 0), verdict))

    print("\nmatrix: row - column, share of draws row > column; ^ / v where the 90% interval "
          "excludes 0")
    print("{:<18}".format("") + "".join("{:>15}".format(name[:14]) for name in names))
    for a in range(len(names)):
        cells = []
        for b in range(len(names)):
            if a == b:
                cells.append("{:>15}".format("--"))
                continue
            diff = draws[a] - draws[b]
            low, high = np.percentile(diff, PERCENTILES)
            mark = "^" if low > 0 else "v" if high < 0 else " "
            cells.append("{:>15}".format("{:+.3f}{} {:3.0f}%".format(
                observed[a] - observed[b], mark, 100 * np.mean(diff > 0))))
        print("{:<18}".format(names[a][:17]) + "".join(cells))


if __name__ == "__main__":
    main()
