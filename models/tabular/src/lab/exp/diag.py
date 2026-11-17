"""Where does spellF1 go? Oracle decodes and an error breakdown of one model's answers.

For each IDSSE match of a condition the model's per-second answers are decoded exactly as
pipeline.cmd_decode does (Viterbi at a switch cost, then snap), but with parts swapped for the
truth:
    model         the model's p_in and p_home
    oracle_dead   true alive/dead per second, the model's team
    oracle_team   the model's alive/dead, true team per second
    oracle_both   both true (per-second truth through the decoder)
    truth_1s      per-second majority of the truth, no decoder, no snap: 1 s resolution ceiling
    truth_1s_snap the same, then snap
    truth_0.2s    5-frame majority of the truth: 0.2 s resolution ceiling
All matches pooled (X between them). Then, for the model at IoU 0.5: recall by true possession
length and boundary type, why missed possessions were missed, and what the spurious ones are.
Prints aggregates only.

    MTAH_ROOT=/root/mtah ../.venv/bin/python diag.py --cond d85 --name exp_ctx_stack
"""
import os as _os
ROOT = _os.environ.get("MTAH_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))))
LAB = _os.environ.get("MTAH_LAB") or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
LABCACHE = _os.path.join(ROOT, "runs", "lab-cache")

import argparse
from collections import Counter

import numpy as np

from common import A, D, H, pipeline, tb

pm = tb.pm
X = pm.X
EPS = 1e-3
BUCKETS = (0, 1, 2, 4, 8, 16, 1e9)


def decode(meta, f, answers, cost):
    codes = np.full(meta["n_frames"], X, dtype=np.int8)
    spans_all = pipeline.windows(meta)
    for half, first, last in meta["halves"]:
        spans = [w for w in spans_all if w[0] == half]
        emission = np.empty((len(spans), 3))
        for row, (_, _, start, _) in enumerate(spans):
            p_in, p_home = answers.get(start, (0.5, 0.5))
            emission[row] = (p_in * p_home, p_in * (1 - p_home), 1 - p_in)
        path = pipeline.viterbi(np.log(np.clip(emission, pipeline.PROB_FLOOR, 1.0)), cost)
        labels = np.repeat(path, [stop - start for _, _, start, stop in spans])
        codes[first:last + 1] = pipeline.snap(labels, f, first, last)
    return codes


def truth_answers(meta, gt, model):
    out = {}
    for _, _, start, stop in pipeline.windows(meta):
        t = gt[start:stop]
        t = t[t != X]
        alive = (t == H) | (t == A)
        p_in = 1 - EPS if len(t) and alive.mean() > 0.5 else EPS
        if alive.any():
            p_home = float(np.clip((t == H).sum() / alive.sum(), EPS, 1 - EPS))
        else:
            p_home = model.get(start, (0.5, 0.5))[1]
        out[start] = (p_in, p_home)
    return out


def block_majority(gt, meta, width):
    out = np.full(len(gt), X, dtype=np.int8)
    for _, first, last in meta["halves"]:
        for start in range(first, last + 1, width):
            stop = min(start + width, last + 1)
            t = gt[start:stop]
            t = t[t != X]
            if len(t):
                out[start:stop] = np.bincount(t, minlength=4)[:3].argmax()
    return out


def bucket(frames):
    seconds = frames / float(pipeline.FPS)
    for lo, hi in zip(BUCKETS[:-1], BUCKETS[1:]):
        if lo <= seconds < hi:
            return "{}-{}s".format(lo, hi if hi < 1e8 else "")
    return "?"


def breakdown(gt, pred, runs, threshold=0.5):
    gt_spells = pm.possession_spells(gt, runs)
    pred_spells = pm.possession_spells(pred, runs)
    overlaps = pm.spell_overlaps(gt_spells, pred_spells)
    flags = pm.matched_gt_flags(overlaps, len(gt_spells), threshold)
    matched_pred = set()
    taken = np.zeros(len(gt_spells), dtype=bool)
    for iou, i, j in overlaps:
        if iou < threshold:
            break
        if taken[i] or j in matched_pred:
            continue
        taken[i] = True
        matched_pred.add(j)

    by_len, by_start = Counter(), Counter()
    hit_len, hit_start = Counter(), Counter()
    why = Counter()
    best = {}
    for iou, i, j in overlaps:
        best.setdefault(i, (iou, j))
    for i, (a, b, k) in enumerate(gt_spells):
        prev = gt[a - 1] if a > 0 else X
        start_kind = "after D" if prev == D else ("after other team" if prev in (H, A) else "half start")
        key = bucket(b - a)
        by_len[key] += 1
        by_start[start_kind] += 1
        if flags[i]:
            hit_len[key] += 1
            hit_start[start_kind] += 1
            continue
        p = pred[a:b]
        same, other, dead = np.mean(p == k), np.mean(p == (A if k == H else H)), np.mean(p == D)
        if other >= max(same, dead):
            why["mostly other team"] += 1
        elif dead >= same:
            why["mostly dead"] += 1
        else:
            iou, j = best.get(i, (0.0, None))
            pa, pb, _ = pred_spells[j]
            if (pb - pa) > (b - a):
                why["same team, absorbed into a longer spell"] += 1
            else:
                why["same team, split / cut short"] += 1

    spur = Counter()
    for j, (a, b, k) in enumerate(pred_spells):
        if j in matched_pred:
            continue
        t = gt[a:b]
        same, other, dead = np.mean(t == k), np.mean(t == (A if k == H else H)), np.mean(t == D)
        kind = ("inside true same-team play" if same >= max(other, dead)
                else "over true other-team play" if other >= dead else "over true dead ball")
        spur["{} | {}".format(bucket(b - a), kind)] += 1
    return gt_spells, flags, by_len, hit_len, by_start, hit_start, why, spur, len(pred_spells)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cond", default="d85")
    parser.add_argument("--name", default="exp_ctx_stack")
    parser.add_argument("--cost", type=float, default=0.1)
    args = parser.parse_args()
    variants = ("model", "oracle_dead", "oracle_team", "oracle_both", "oracle_both_cost0",
                "truth_1s", "truth_1s_snap", "truth_0.2s")
    gts, preds = [], {v: [] for v in variants}
    sep = np.full(tb.SEPARATOR, X, dtype=np.int8)
    for match in tb.built_matches():
        dataset = tb.dataset_name(match, args.cond)
        meta, f = pipeline.load_dataset(dataset)
        gt = tb.gt_codes(dataset).astype(np.int8)
        model = pipeline.load_answers(dataset, args.name)
        truth = truth_answers(meta, gt, model)
        mixed_dead = {s: (truth[s][0], model.get(s, (0.5, 0.5))[1]) for s in truth}
        mixed_team = {s: (model.get(s, (0.5, 0.5))[0], truth[s][1]) for s in truth}
        q1 = block_majority(gt, meta, pipeline.FPS)
        q1_snap = q1.copy()
        for _, first, last in meta["halves"]:
            q1_snap[first:last + 1] = pipeline.snap(q1[first:last + 1].copy(), f, first, last)
        out = {"model": decode(meta, f, model, args.cost),
               "oracle_dead": decode(meta, f, mixed_dead, args.cost),
               "oracle_team": decode(meta, f, mixed_team, args.cost),
               "oracle_both": decode(meta, f, truth, args.cost),
               "oracle_both_cost0": decode(meta, f, truth, 1e-6),
               "truth_1s": q1, "truth_1s_snap": q1_snap,
               "truth_0.2s": block_majority(gt, meta, 5)}
        gts += [gt, sep]
        for v in variants:
            preds[v] += [out[v], sep]
    gt = np.concatenate(gts[:-1])
    runs = pm.valid_runs(gt)
    print("{} / {} answers, switch cost {}".format(args.cond, args.name, args.cost))
    print("| decode | spellF1 | F1@0.3 | F1@0.5 | F1@0.7 | poss pred/true |\n|---|---|---|---|---|---|")
    for v in variants:
        pred = np.concatenate(preds[v][:-1])
        pred = np.where(gt == X, X, pred)
        s = pm.spell_scores(gt, pred, runs)
        print("| {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {} / {} |".format(
            v, s["mean_f1"], s["per_iou"][0.3]["f1"], s["per_iou"][0.5]["f1"],
            s["per_iou"][0.7]["f1"], s["n_pred"], s["n_gt"]))

    pred = np.where(gt == X, X, np.concatenate(preds["model"][:-1]))
    gt_spells, flags, by_len, hit_len, by_start, hit_start, why, spur, n_pred = breakdown(
        gt, pred, runs)
    print("\nmodel at IoU 0.5: {} of {} true possessions matched, {} predicted".format(
        int(flags.sum()), len(gt_spells), n_pred))
    order = ["{}-{}s".format(lo, hi if hi < 1e8 else "") for lo, hi in zip(BUCKETS[:-1], BUCKETS[1:])]
    print("| true length | count | share | recall |\n|---|---|---|---|")
    for key in order:
        if by_len[key]:
            print("| {} | {} | {:.1%} | {:.3f} |".format(key, by_len[key], by_len[key] / len(gt_spells),
                                                     hit_len[key] / by_len[key]))
    print("| starts | count | recall |\n|---|---|---|")
    for key, n in by_start.most_common():
        print("| {} | {} | {:.3f} |".format(key, n, hit_start[key] / n))
    print("| why missed | count |\n|---|---|")
    for key, n in why.most_common():
        print("| {} | {} |".format(key, n))
    print("| spurious (length / where) | count |\n|---|---|")
    for key, n in sorted(spur.items(), key=lambda kv: -kv[1])[:10]:
        print("| {} | {} |".format(key, n))


if __name__ == "__main__":
    main()
