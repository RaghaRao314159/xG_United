"""BAR-ATH candidates with Kumo Tabular in the recipe: bar_v3.py's pipeline (training on controlled
IDSSE d85suc only, kick-off rules, 5 Hz minimum-length decoder) with the team model and P(alive)
built as logit blends of LightGBM and Kumo members. Every setting (members, weights, costs, minimum
lengths) comes from controlled-IDSSE LOMO and the IDSSE -> SkillCorner transfer test, never from the
public window.

    team members:   lgb_b, lgb_ab (ta5.fit_team, 800 trees / 63 leaves), kt_b, kt_ab (Kumo, 50k
                    context of both orientations, top-100 columns, 4 estimators)
    P(alive):       lgb (palive pf, as bar_v3) or mix50 (lgb + Kumo 3-class, logit 50/50)

    MTAH_ROOT=/root/mtah python bar_k.py --team lgb_b:1 lgb_ab:1 kt_b:1 --palive mix50 \
        --cost 2 0.1 20 --mins 10 10 --name exp_k_mk3_mix
Writes runs/BAR-ATH/results/<name>.csv and answers_<name>_pa.jsonl. Prints aggregates only.
"""
import os as _os
LAB = _os.environ.get("MTAH_LAB") or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

import argparse
import os
import sys
import time

sys.path.insert(0, LAB + "/exp_sc")
import xfer  # noqa: E402  (palive tables at 12-frame windows)

import numpy as np  # noqa: E402

sys.path.insert(0, LAB + "/exp_t")
sys.path.insert(0, LAB + "/exp_d")
sys.path.insert(0, LAB + "/exp_k")
import bar  # noqa: E402  (BAR-ATH dataset names)
import dball  # noqa: E402
import kick  # noqa: E402
import bar_v2  # noqa: E402
from decode_bar import aggregates  # noqa: E402
from xfer import palive, pipeline, run_tb, ta5, tb  # noqa: E402
import ta5_decode as td  # noqa: E402
import kt  # noqa: E402
import kt_pa  # noqa: E402

TARGET = bar.TARGET
COND = os.environ.get("MTAH_IDSSE_COND", "d85suc")   # d85sue = rule E labels (exp_bench/labels.py lab)


def lg(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def palive_answers(kind, name, args):
    cache, tables, spec, names = {}, {}, {}, None
    for m in list(xfer.IDSSE) + [TARGET]:
        tables[m], n2, s = xfer.palive_table(m, COND, cache)
        assert names is None or n2 == names
        names = n2
        spec.update(s)
    tb.swap = palive.subset_swap(run_tb.make_swap(spec), names)(names)
    x = np.vstack([tables[m][0] for m in xfer.IDSSE])
    y = np.concatenate([tables[m][1] for m in xfer.IDSSE])
    lgb, _ = tb.symmetric_proba("lgbm", x, y, tables[TARGET][0], names, tb.SEED)
    proba = lgb
    if kind == "mix50":
        kumo = kt_pa.Kumo3("large")
        a = argparse.Namespace(k=100, ctx=50000, est=4, seed=0)
        got, _ = kt_pa.symmetric(kumo, x, y, {TARGET: tables[TARGET][0]}, names, a)
        kp = got[TARGET]
        del kumo
        # P(alive) and P(H | alive) mixed in logit space, as kt_pamix.py does on the answers
        lgb_in, kp_in = 1 - lgb[:, tb.D], 1 - kp[:, tb.D]
        lgb_h = lgb[:, tb.H] / np.maximum(lgb_in, 1e-9)
        kp_h = kp[:, tb.H] / np.maximum(kp_in, 1e-9)
        p_in = 1 / (1 + np.exp(-(0.5 * lg(lgb_in) + 0.5 * lg(kp_in))))
        p_h = 1 / (1 + np.exp(-(0.5 * lg(lgb_h) + 0.5 * lg(kp_h))))
        proba = np.column_stack([p_in * p_h, p_in * (1 - p_h), 1 - p_in])
        print("  Kumo P(alive): mean P(D) {:.3f} (LightGBM {:.3f})".format(kp[:, tb.D].mean(),
                                                                         lgb[:, tb.D].mean()), flush=True)
    tb.write_answers(TARGET, name, tables[TARGET][2], proba)
    print("P(alive) {}: mean P(D) {:.3f}".format(kind, proba[:, tb.D].mean()), flush=True)


def team_member(member):
    kind, variant = member.split("_", 1)
    tables = {m: kt.table(m, COND, variant) for m in xfer.IDSSE}
    tables[TARGET] = kt.table(TARGET, COND, variant)
    swap = kt.u_fit.make_swap(*tables[TARGET][2])
    if kind == "lgb":
        ta5.LGBM.update(n_estimators=800, num_leaves=63, n_jobs=int(os.environ.get("MTAH_LGBM_JOBS", 8)))
        team = {m: (t[0], None, t[1]) for m, t in tables.items()}
        p = ta5.fit_team(team, list(xfer.IDSSE), [TARGET], swap)[TARGET]
    else:
        t2 = {m: (t[0], t[1]) for m, t in tables.items()}
        x, y, groups = kt.training_pool(t2, list(xfer.IDSSE), swap)
        cols = kt.select_columns(x, y, 100, 0)
        pick = kt.sample_context(y, groups, 50000, "random", 0)
        kumo = kt.Kumo("large")
        kumo.fit(x[pick][:, cols], y[pick], 4, 0)
        xt = t2[TARGET][0]
        pa = kumo.predict(xt[:, cols])
        pb = kumo.predict(swap(xt)[:, cols])
        p = 0.5 * (pa + 1.0 - pb)
        del kumo
    print("  team {}: mean P(H | alive) {:.3f}".format(member, p.mean()), flush=True)
    return p, tables[TARGET][1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--team", nargs="+", required=True, help="member:weight")
    parser.add_argument("--palive", default="lgb", choices=("lgb", "mix50"))
    parser.add_argument("--cost", nargs=3, type=float, required=True)
    parser.add_argument("--mins", nargs=2, type=int, required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    started = time.time()
    pa_name = args.name + "_pa"
    palive_answers(args.palive, pa_name, args)

    tb.swap = xfer.ORIGINAL_SWAP
    pipeline.WINDOW = tb.WINDOW = pipeline.FPS
    parts = [(p.rpartition(":")[0], float(p.rpartition(":")[2])) for p in args.team]
    total = sum(w for _, w in parts)
    z, info = 0.0, None
    for member, w in parts:
        p, info = team_member(member)
        z = z + w / total * lg(p)
    p1 = 1.0 / (1.0 + np.exp(-z))
    info = dict(info, p1=p1)
    print("team blend {}: mean P(H | alive) {:.3f}, {:.0f} s".format(args.team, p1.mean(),
                                                                     time.time() - started), flush=True)

    meta, f = pipeline.load_dataset(TARGET)
    first_of = np.zeros(len(info["start"]), dtype=np.int64)
    for half, first, _ in meta["halves"]:
        first_of[info["half"] == half] = first
    block = int(np.median(info["end"] - info["start"]))
    mid = np.minimum(info["start"] + block // 2, info["end"] - 1)
    p_in = dball.lookup(pipeline.load_answers(TARGET, pa_name), info["start"], info["end"],
                        first_of, 12, block // 2)
    k1, _, _, _ = kick.kick_mask(kick.load_raw(TARGET), meta, f, players_only=True)
    k2 = np.zeros(meta["n_frames"], dtype=bool)
    kicks = dball.kickoff_frames()
    for half, first, last in meta["halves"]:
        k2[first:min(kicks[half], last + 1)] = True
    forced = k1 | k2
    codes = bar_v2.decode(p_in, p1, info, forced[mid], forced, tuple(args.cost), tuple(args.mins))
    td.write(TARGET, args.name, codes)
    print("| output | D share | D in >10 s blackouts | D pre-kick-off 2nd half | possessions | per "
          "alive-min | mean s | median s | < 2 s | < 4 s |\n|---|---|---|---|---|---|---|---|---|---|")
    print(aggregates(args.name, codes, meta, f), flush=True)


if __name__ == "__main__":
    main()
