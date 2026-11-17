"""Track C: IDSSE's DFL truth relabelled to the challenge's "controlled possession" convention.

CHALLENGE.md: possession changes only when an opponent gains *controlled* possession; while the ball
is contested or loose it stays with the last team that had it; a touch without control, a
deflection or a won aerial duel without control does not change it; after a restart there is a
clear possessor. DFL's BallPossession does not apply that rule and flips far more often (6.5-11
spells per alive-minute). Every rule here walks the truth's segments in order and reverts an H/A
segment that directly follows the *other* team's segment during play (no D between) to that team,
unless the segment shows control. Segments that start after D (restarts) are never touched; D and
X are never touched.

    R0(T)    keep the segment if it lasts >= T s
    R2(k)    keep it if one of that team's players is within RADIUS m of the ball for >= k s in a
             row inside the segment (clean IDSSE tracking: features.npz d_home / d_away)
    R02(T,k) keep it if it lasts >= T s or shows R2 control

    python conv.py stats                       # IDSSE aggregates per rule
    python conv.py build --rule R2 --k 1.0     # write runs/idsse-<m>-<cond>c/ (gt.csv = relabelled)

R2's k is chosen in exp_sc/conv_sc.py against SkillCorner's label aggregates (k = 0.6).
Prints aggregates only.
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

ROOT = os.environ.get("MTAH_ROOT", ROOT)
sys.path.insert(0, ROOT + "/exp")
from common import A, D, H, pipeline, tb  # noqa: E402

pm = tb.pm
X = pm.X
FPS = pipeline.FPS
RADIUS = 1.5
SOURCES = ("clean", "d85", "d85su")   # every copy shares the clean truth; outputs <cond>c


def stats(codes):
    runs = pm.valid_runs(codes)
    sp = pm.possession_spells(codes, runs)
    lens = np.array([b - a for a, b, _ in sp], dtype=float) / FPS
    alive_min = np.isin(codes, (H, A)).sum() / FPS / 60.0
    return {"n": len(sp), "per_min": len(sp) / alive_min, "u2": float(np.mean(lens < 2)),
            "u4": float(np.mean(lens < 4)), "median": float(np.median(lens)),
            "alive_min": alive_min}


def relabel(codes, keep):
    out = codes.copy()
    for start, stop in pm.valid_runs(codes):
        prev = None
        for a, b, k in pm.segments(codes, [(start, stop)]):
            if k == D:
                prev = None
                continue
            if k not in (H, A):
                continue
            if prev is not None and k != prev and not keep(a, b, k):
                out[a:b] = prev
                continue
            prev = k
    return out


def control_runs(f):
    """Per team: per-frame length (frames) of the current run of 'a player of this team within
    RADIUS m of the ball', so a segment shows k s of control iff max over it >= k * FPS."""
    out = {}
    for team, key in ((H, "d_home"), (A, "d_away")):
        close = np.nan_to_num(f[key], nan=1e9) < RADIUS
        run = np.zeros(len(close), dtype=np.int32)
        c = 0
        for i, v in enumerate(close):
            c = c + 1 if v else 0
            run[i] = c
        out[team] = run
    return out


def keeper(rule, T, k, ctl):
    def keep(a, b, team):
        long_enough = (b - a) >= T * FPS if T is not None else False
        if rule == "R0":
            return long_enough
        has_ctl = ctl[team][a:b].max() >= k * FPS if b > a else False
        if rule == "R2":
            return has_ctl
        return long_enough or has_ctl
    return keep


RULES = ([("R0", T, None) for T in (1.0, 2.0, 3.0, 4.0)]
         + [("R2", None, k) for k in (0.2, 0.4, 0.6, 1.0, 1.5)]
         + [("R02", T, k) for T in (3.0, 4.0) for k in (0.4, 1.0)])


def name_of(rule, T, k):
    return rule + ("" if T is None else "_T{:g}".format(T)) + ("" if k is None else "_k{:g}".format(k))


def cmd_stats(args):
    truth, ctl = {}, {}
    for m in tb.built_matches():
        ds = tb.dataset_name(m, "clean")
        truth[m] = tb.gt_codes(ds).astype(np.int8)
        _, f = pipeline.load_dataset(ds)
        ctl[m] = control_runs(f)
    print("| rule | spells | per alive-min (range over matches) | <2 s | <4 s | median s | DFL spells merged |")
    print("|---|---|---|---|---|---|---|")
    base = {m: stats(truth[m]) for m in truth}
    n0 = sum(b["n"] for b in base.values())

    def row(label, codes_by_match):
        st = {m: stats(c) for m, c in codes_by_match.items()}
        n = sum(x["n"] for x in st.values())
        amin = sum(x["alive_min"] for x in st.values())
        lens = np.concatenate([np.array([b - a for a, b, _ in pm.possession_spells(c, pm.valid_runs(c))])
                               for c in codes_by_match.values()]) / FPS
        per = [x["per_min"] for x in st.values()]
        print("| {} | {} | {:.2f} ({:.1f}-{:.1f}) | {:.0%} | {:.0%} | {:.1f} | {} |".format(
            label, n, n / amin, min(per), max(per), np.mean(lens < 2), np.mean(lens < 4),
            np.median(lens), n0 - n))

    row("DFL (as is)", truth)
    for rule, T, k in RULES:
        row(name_of(rule, T, k), {m: relabel(truth[m], keeper(rule, T, k, ctl[m])) for m in truth})


def cmd_build(args):
    letters = np.array(["H", "A", "D", "X"])
    for m in tb.built_matches():
        src_clean = tb.dataset_name(m, "clean")
        truth = tb.gt_codes(src_clean).astype(np.int8)
        _, f = pipeline.load_dataset(src_clean)
        new = relabel(truth, keeper(args.rule, args.T, args.k, control_runs(f)))
        for cond in SOURCES:
            src = pipeline.dataset_dir(tb.dataset_name(m, cond))
            dst = pipeline.dataset_dir(tb.dataset_name(m, cond + args.suffix))
            if not os.path.isdir(src):
                continue
            assert (tb.gt_codes(tb.dataset_name(m, cond)) == truth).all(), (m, cond)
            os.makedirs(os.path.join(dst, "results"), exist_ok=True)
            for name in os.listdir(src):
                if name in ("gt.csv", "results", "score") or name.startswith("answers_exp_c"):
                    continue
                link = os.path.join(dst, name)
                if not os.path.lexists(link):
                    os.symlink(os.path.join(src, name), link)
            pipeline.baseline.write_submission(os.path.join(dst, "gt.csv"), letters[new])
        s0, s1 = stats(truth), stats(new)
        print("  {}: {} -> {} spells ({:.2f} -> {:.2f}/alive-min), frames relabelled {:.1%}".format(
            m, s0["n"], s1["n"], s0["per_min"], s1["per_min"], np.mean(new != truth)))
    print("rule {} T={} k={}: wrote <cond>{} datasets for {}".format(args.rule, args.T, args.k,
                                                                    args.suffix, SOURCES))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("stats")
    b = sub.add_parser("build")
    b.add_argument("--rule", required=True, choices=("R0", "R2", "R02"))
    b.add_argument("--T", type=float, default=None)
    b.add_argument("--k", type=float, default=None)
    b.add_argument("--suffix", default="c")
    args = p.parse_args()
    {"stats": cmd_stats, "build": cmd_build}[args.cmd](args)


if __name__ == "__main__":
    main()
