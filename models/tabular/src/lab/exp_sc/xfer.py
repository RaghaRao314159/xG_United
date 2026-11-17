"""Transfer test: train on the 7 IDSSE matches of one condition, predict all 20 SkillCorner matches,
decode at the costs IDSSE leave-one-match-out chose -- exactly what we do to BAR-ATH, but scored,
because SkillCorner is labelled. Nothing is tuned on SkillCorner. It is the closest stand-in for
BAR-ATH: another provider and league, a real broadcast ball (missing 41% of graded frames, 90% of it
in stoppages, BAR-ATH 41%) and the challenge's controlled-possession convention.

P(alive): palive.py's tables (base = u_half's columns, pf, blind), one fit on IDSSE per variant.
P(H | alive): u_fit.py's team model (b, or the logit average of --team-variants), one fit on IDSSE. Decoder: Track AB's 5 Hz Viterbi at
COST, and dur.py's minimum lengths at MINS (both IDSSE's choices, passed in). Writes
xf_<variant>[_dur].csv per SkillCorner dataset, pools and scores them. Prints aggregates only.

    MTAH_ROOT=/root/mtah python xfer.py --source d85su --variants base pf blind
    MTAH_ROOT=/root/mtah python xfer.py --direction sc2idsse --source d85suc --variants pf
    ... --source d85suc --variants pf --team-variants b ab --cost 2 0.1 20 --mins 5 1   (final recipe)
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
import time

sys.path.insert(0, LAB + "/exp_sc")
import palive  # noqa: E402  (sets the 12-frame windows first)

import numpy as np  # noqa: E402

sys.path.insert(0, LAB + "/exp_a")
sys.path.insert(0, LAB + "/exp_ab")
sys.path.insert(0, LAB + "/exp_u")
import ctx  # noqa: E402
import feat_tb  # noqa: E402
import pfeat  # noqa: E402
import run_tb  # noqa: E402
import ta5  # noqa: E402
import ta5_decode as td  # noqa: E402
import u_decode as ud  # noqa: E402
import u_fit  # noqa: E402
import dur  # noqa: E402
from common import pipeline, tb  # noqa: E402

ORIGINAL_SWAP = tb.swap   # ctx.swap; ta5's team swap reads tb.swap, so restore it after P(alive)
IDSSE = ("J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9")
TARGET_COND = "clean"


def palive_table(m, cond, cache):
    dataset = tb.dataset_name(m, cond)
    cols, _ = ctx.extra(m, cond)
    x, y, meta, extra_names = ctx.with_columns(tb.table_of(m, cond, cache), cols)
    tcols, tspec = feat_tb.columns(dataset)
    pcols, pspec = pfeat.columns(dataset)
    t_names, p_names = list(tcols), list(pcols)
    x = np.column_stack([x] + [tcols[n] for n in t_names] + [pcols[n] for n in p_names])
    names = tb.feature_names() + extra_names + t_names + p_names
    return (x.astype(np.float32), y, meta), names, dict(tspec, **pspec)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="d85su", help="IDSSE condition (either direction)")
    parser.add_argument("--direction", default="idsse2sc", choices=("idsse2sc", "sc2idsse"))
    parser.add_argument("--variants", nargs="+", default=["base", "pf", "blind"])
    parser.add_argument("--cost", nargs=3, type=float, default=[1.0, 0.1, 20.0])
    parser.add_argument("--mins", nargs=2, type=int, default=None, help="dur.py blocks (alive, dead)")
    parser.add_argument("--team-variants", nargs="+", default=["b"], choices=("b", "ab"),
                        help="u_fit variants whose P(H | alive) are averaged in logit space")
    parser.add_argument("--tag", default="xfer")
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    with open(os.path.join(pipeline.RUNS, "sc-stats.json")) as h:
        sc = json.load(h)["matches"]
    if args.direction == "idsse2sc":
        train, train_cond, targets, tcond = list(IDSSE), args.source, sc, TARGET_COND
    else:
        train, train_cond, targets, tcond = sc, TARGET_COND, list(IDSSE), args.source
    started = time.time()
    cache, tables, spec, names = {}, {}, {}, None
    for m in train:
        tables[m], names, s = palive_table(m, train_cond, cache)
        spec.update(s)
    for m in targets:
        tables[m], n2, s = palive_table(m, tcond, cache)
        assert n2 == names, "column layout differs"
        spec.update(s)
    make = palive.subset_swap(run_tb.make_swap(spec), names)
    print("tables: {} {} -> {} targets, {} columns, {:.0f} s".format(
        len(train), args.direction, len(targets), len(names), time.time() - started), flush=True)

    p_alive = {}
    test_x = np.vstack([tables[m][0] for m in targets])
    bounds = np.cumsum([0] + [len(tables[m][0]) for m in targets])
    for variant in args.variants:
        keep = [i for i, n in enumerate(names) if variant in ("base", "pf") and
                (variant == "pf" or not n.startswith("pf_")) or variant == "blind" and palive.ball_free(n)]
        sub = [names[i] for i in keep]
        tb.swap = make(sub)
        x = np.vstack([tables[m][0][:, keep] for m in train])
        y = np.concatenate([tables[m][1] for m in train])
        proba, _ = tb.symmetric_proba("lgbm", x, y, test_x[:, keep], sub, tb.SEED)
        for k, m in enumerate(targets):
            part = proba[bounds[k]:bounds[k + 1]]
            p_alive[variant, m] = part
            tb.write_answers(tb.dataset_name(m, tcond), "xf_pa_" + variant, tables[m][2], part)
        print("P(alive) {}: {} columns; {} on the targets".format(
            variant, len(sub), ctx.report(variant, proba, np.concatenate([tables[m][1] for m in targets]))),
            flush=True)

    started = time.time()
    ta5.LGBM.update(n_estimators=800, num_leaves=63, n_jobs=int(os.environ.get("MTAH_LGBM_JOBS", 4)))
    tb.swap = ORIGINAL_SWAP
    pipeline.WINDOW = tb.WINDOW = pipeline.FPS   # the team model's tabular row is per second
    logits = {m: [] for m in targets}
    for team_variant in args.team_variants:
        team = {}
        for m, cond in [(m, train_cond) for m in train] + [(m, tcond) for m in targets]:
            x, a_names, b_names, bspec, info = u_fit.match_table(m, cond, team_variant)
            team[m] = (x, None, info)
        swap = u_fit.make_swap(a_names, b_names, bspec)
        for m, p in ta5.fit_team(team, list(train), list(targets), swap).items():
            logits[m].append(td.logit(p))
    p1 = {m: 1.0 / (1.0 + np.exp(-np.mean(logits[m], axis=0))) for m in targets}
    print("team model ({}): fitted on {} matches, {:.0f} s".format(
        "+".join(args.team_variants), len(train), time.time() - started),
          flush=True)

    cost = td.cost_matrix(*args.cost)[None]
    entries = []
    for variant in args.variants:
        label = "xf_" + variant
        for m in targets:
            info = team[m][2]
            dataset = tb.dataset_name(m, tcond)
            meta, _ = pipeline.load_dataset(dataset)
            first_of = np.zeros(len(info["start"]), dtype=np.int64)
            for half, first, _ in meta["halves"]:
                first_of[info["half"] == half] = first
            block = int(np.median(info["end"] - info["start"]))
            t = ud.abd.lookup(pipeline.load_answers(dataset, "xf_pa_" + variant), info["start"],  # 12-frame keys
                              info["end"], first_of, ud.abd.HALF_WINDOW, block // 2)
            p_in, p_home = t[:, 0], p1[m]
            emission = np.column_stack([p_in * p_home, p_in * (1 - p_home), 1 - p_in])
            le = np.log(np.clip(emission, td.FLOOR, 1.0))
            for suffix, fn in [("", lambda e: td.viterbi_grid(e, cost)[0])] + (
                    [("_dur", lambda e: dur.viterbi_min(e, cost, *args.mins)[0])] if args.mins else []):
                path = np.empty(len(le), dtype=np.int8)
                for h in np.unique(info["half"]):
                    idx = np.flatnonzero(info["half"] == h)
                    path[idx] = fn(le[idx])
                td.write(dataset, label + suffix, td.frame_codes(path[None], info)[0])
        entries.append((label, tcond))
        if args.mins:
            entries.append((label + "_dur", tcond))
    saved = tb.MATCHES
    tb.MATCHES = tuple(targets)
    folder = ud.pool(tcond, entries, list(targets), args.bootstrap, args.tag)
    ud.report(folder, entries[0][0])
    tb.MATCHES = saved


if __name__ == "__main__":
    main()
