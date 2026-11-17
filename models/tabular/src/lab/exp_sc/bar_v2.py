"""BAR-ATH candidates from the dead-ball programme (exp_sc), every choice made on IDSSE before any
public-window score:
  exp_sc_ref_k     Track F rebuilt: P(alive) exp_t_tbh_d85su, team cache bar_b_d85su, costs
                   (1, 0.1, 20), kick-off rules (= exp_f_su_k up to the rebuilt team cache)
  exp_sc_dur_k     the same through dur.py's minimum-length decoder at IDSSE d85su's in-sample
                   choice: possessions >= 1 s, no stoppage minimum, costs (0.5, 2, 20)
  exp_sc_pf_k      P(alive) with pfeat.py's player-only context columns, trained on all 7 IDSSE
                   d85su matches (the base variant is refitted alongside and must match
                   exp_t_tbh_d85su), costs (1, 0.1, 20)
  exp_sc_pf_dur_k  pf through the minimum-length decoder as above
On IDSSE d85su: pf -0.000, dur +0.003, pf+dur +0.001 (all n.s.); on no-ball IDSSE pf +0.026; trained
on IDSSE and scored on SkillCorner (the BAR-ATH stand-in) pf +0.002, dur +0.013, pf+dur +0.014.
Writes runs/BAR-ATH/results/exp_sc_*.csv. Aggregates only.

    MTAH_ROOT=/root/mtah MTAH_DATA_DIR=/root/mtah/data python bar_v2.py
"""
import os as _os
ROOT = _os.environ.get("MTAH_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))))
LAB = _os.environ.get("MTAH_LAB") or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
LABCACHE = _os.path.join(ROOT, "runs", "lab-cache")

import sys

sys.path.insert(0, LAB + "/exp_sc")
import xfer  # noqa: E402  (palive tables at 12-frame windows)

import numpy as np  # noqa: E402

sys.path.insert(0, LAB + "/exp_t")
sys.path.insert(0, LAB + "/exp_d")
import bar  # noqa: E402  (BAR-ATH as one more 'match': its own dir, an all-X truth)
import dball  # noqa: E402
import kick  # noqa: E402
from decode_bar import aggregates  # noqa: E402
from xfer import ctx, dur, palive, pipeline, run_tb, tb  # noqa: E402
import ta5_decode as td  # noqa: E402

TARGET = bar.TARGET
SOURCE = "d85su"
CACHE = LABCACHE + "/exp_t/bar_b_d85su.npz"
STD = (1.0, 0.1, 20.0)
DUR_COST, DUR_MINS = (0.5, 2.0, 20.0), (5, 1)


def p_alive_variants():
    cache, tables, spec, names = {}, {}, {}, None
    for m in xfer.IDSSE + (TARGET,):
        tables[m], n2, s = xfer.palive_table(m, SOURCE, cache)
        assert names is None or n2 == names
        names = n2
        spec.update(s)
    make = palive.subset_swap(run_tb.make_swap(spec), names)
    out = {}
    for variant in ("base", "pf"):
        keep = [i for i, n in enumerate(names) if variant == "pf" or not n.startswith("pf_")]
        sub = [names[i] for i in keep]
        tb.swap = make(sub)
        x = np.vstack([tables[m][0][:, keep] for m in xfer.IDSSE])
        y = np.concatenate([tables[m][1] for m in xfer.IDSSE])
        proba, _ = tb.symmetric_proba("lgbm", x, y, tables[TARGET][0][:, keep], sub, tb.SEED)
        name = "exp_sc_pa_{}_{}".format(variant, SOURCE)
        tb.write_answers(TARGET, name, tables[TARGET][2], proba)
        out[variant] = name
        print("{}: {} columns, mean P(D) {:.3f}, P(D) > 0.5 on {:.1%}".format(
            name, len(sub), proba[:, tb.D].mean(), np.mean(proba[:, tb.D] > 0.5)), flush=True)
    tb.swap = xfer.ORIGINAL_SWAP
    return out


def decode(p_in, p_home, info, forced_blocks, forced_frames, cost, mins=None):
    p_in = np.where(forced_blocks, dball.FORCED_P_IN, p_in)
    emission = np.column_stack([p_in * p_home, p_in * (1 - p_home), 1 - p_in])
    le = np.log(np.clip(emission, td.FLOOR, 1.0))
    path = np.empty(len(le), dtype=np.int8)
    costs = td.cost_matrix(*cost)[None]
    for h in np.unique(info["half"]):
        idx = np.flatnonzero(info["half"] == h)
        path[idx] = (dur.viterbi_min(le[idx], costs, *mins) if mins else td.viterbi_grid(le[idx], costs))[0]
    codes = td.frame_codes(path[None], info)[0]
    return np.where(forced_frames & (codes != tb.pm.X), tb.pm.D, codes).astype(np.int8)


def main():
    answers = p_alive_variants()
    meta, f = pipeline.load_dataset(TARGET)
    info = dict(np.load(CACHE))
    first_of = np.zeros(len(info["start"]), dtype=np.int64)
    for half, first, _ in meta["halves"]:
        first_of[info["half"] == half] = first
    block = int(np.median(info["end"] - info["start"]))
    mid = np.minimum(info["start"] + block // 2, info["end"] - 1)
    p = {k: dball.lookup(pipeline.load_answers(TARGET, v), info["start"], info["end"], first_of, 12,
                         block // 2) for k, v in [("ref", "exp_t_tbh_d85su"), ("base", answers["base"]),
                                                  ("pf", answers["pf"])]}
    print("refitted base vs exp_t_tbh_d85su: max |P(alive) difference| {:.4f}".format(
        np.abs(p["ref"] - p["base"]).max()))
    k1, _, episodes, _ = kick.kick_mask(kick.load_raw(TARGET), meta, f, players_only=True)
    k2 = np.zeros(meta["n_frames"], dtype=bool)
    kicks = dball.kickoff_frames()
    for half, first, last in meta["halves"]:
        k2[first:min(kicks[half], last + 1)] = True
    forced = k1 | k2
    variants = [("exp_sc_ref_k", "ref", STD, None), ("exp_sc_dur_k", "ref", DUR_COST, DUR_MINS),
                ("exp_sc_pf_k", "pf", STD, None), ("exp_sc_pf_dur_k", "pf", DUR_COST, DUR_MINS)]
    print("| output | D share | D in >10 s blackouts | D pre-kick-off 2nd half | possessions | per "
          "alive-min | mean s | median s | < 2 s | < 4 s |\n|---|---|---|---|---|---|---|---|---|---|")
    for name, src, cost, mins in variants:
        codes = decode(p[src], info["p1"], info, forced[mid], forced, cost, mins)
        td.write(TARGET, name, codes)
        print(aggregates(name, codes, meta, f), flush=True)


if __name__ == "__main__":
    main()
