"""Track D: dead-ball variants of the BAR-ATH output. The team side is Track T's team model trained
on smoothed d85 (cache bar_b_d85s.npz, P(H | alive) per 0.2 s block); P(alive) is Track T's
d85s answers, optionally logit-blended with OpenJev v3's p_in (mean vote) and the heuristic's;
the kick-off rules force dead blocks: K1 = kick.kick_mask's players-only setup episodes (IDSSE
precision 1.000 clean / 0.985 d85s), K2 = every frame of a half before that half's kick-off as
games.py publishes it ("kickoff" / "kickoff_2nd" anchors: the frame the ball is struck). Decoded
with Track A's 5 Hz Viterbi at Track T's costs (1, 2, 10). The candidate list is fixed here,
before any public-window score; the 45% D-share weight is a label-free calibration to the
LaLiga ball-in-play prior. Writes runs/BAR-ATH/results/exp_d_*.csv. Aggregates only.

    MTAH_ROOT=/root/mtah python dball.py
"""
import os as _os
ROOT = _os.environ.get("MTAH_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))))
LAB = _os.environ.get("MTAH_LAB") or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
LABCACHE = _os.path.join(ROOT, "runs", "lab-cache")

import os
import sys

import numpy as np

sys.path.insert(0, LAB + "/exp")
sys.path.insert(0, LAB + "/exp_a")
sys.path.insert(0, LAB + "/exp_ab")
sys.path.insert(0, LAB + "/exp_t")
sys.path.insert(0, LAB + "/exp_d")
sys.path.insert(0, ROOT)
from common import pipeline, tb  # noqa: E402
import ta5_decode as td  # noqa: E402
from decode_bar import aggregates  # noqa: E402
import kick  # noqa: E402
import games  # noqa: E402

TARGET = "BAR-ATH"
CACHE = LABCACHE + "/exp_t/bar_b_d85s.npz"
COST = (1.0, 2.0, 10.0)
FORCED_P_IN = 1e-4
TARGET_D = 0.45


def lookup(answers, starts, ends, first_of, width, offset):
    pos = np.minimum(starts + offset, ends - 1)
    keys = first_of + ((pos - first_of) // width) * width
    return np.array([answers.get(int(k), (0.5, 0.5))[0] for k in keys])


def kickoff_frames():
    g = games.GAMES[TARGET]
    out = {}
    for half, key in ((1, "kickoff"), (2, "kickoff_2nd")):
        a = g[key]
        out[half] = (a["minute"] * 60 + a["second"]) * games.ANCHOR_FPS + a["frame"]
    return out


def decode(p_in, p_home, info, forced_blocks, forced_frames, cost=COST):
    p_in = np.where(forced_blocks, FORCED_P_IN, p_in)
    emission = np.column_stack([p_in * p_home, p_in * (1 - p_home), 1 - p_in])
    le = np.log(np.clip(emission, td.FLOOR, 1.0))
    paths = np.empty((1, len(le)), dtype=np.int8)
    for h in np.unique(info["half"]):
        idx = np.flatnonzero(info["half"] == h)
        paths[:, idx] = td.viterbi_grid(le[idx], td.cost_matrix(*cost)[None])
    codes = td.frame_codes(paths, info)[0]
    return np.where(forced_frames & (codes != tb.pm.X), tb.pm.D, codes).astype(np.int8)


def d_share(codes, graded):
    return float(np.mean(codes[graded] == tb.pm.D))


def mix(ps, weights):
    z = sum(w * td.logit(p) for p, w in zip(ps, weights))
    return 1.0 / (1.0 + np.exp(-z))


def main():
    meta, f = pipeline.load_dataset(TARGET)
    info = dict(np.load(CACHE))
    first_of = np.zeros(len(info["start"]), dtype=np.int64)
    for half, first, _ in meta["halves"]:
        first_of[info["half"] == half] = first
    block = int(np.median(info["end"] - info["start"]))
    mid = np.minimum(info["start"] + block // 2, info["end"] - 1)
    ours = lookup(pipeline.load_answers(TARGET, "exp_t_tbh_d85s"), info["start"], info["end"],
                  first_of, 12, block // 2)
    oj = lookup(pipeline.load_answers(TARGET, "openjev_v3", vote="mean"), info["start"],
                info["end"], first_of, pipeline.WINDOW, block // 2)
    heur = lookup(pipeline.load_answers(TARGET, "exp_heur_alone"), info["start"], info["end"],
                  first_of, pipeline.WINDOW, block // 2)
    p_home = info["p1"]
    graded = np.zeros(meta["n_frames"], dtype=bool)
    for _, first, last in meta["halves"]:
        graded[first:last + 1] = True

    k1, _, episodes, _ = kick.kick_mask(kick.load_raw(TARGET), meta, f, players_only=True)
    k2 = np.zeros(meta["n_frames"], dtype=bool)
    kicks = kickoff_frames()
    for half, first, last in meta["halves"]:
        k2[first:min(kicks[half], last + 1)] = True
    forced = k1 | k2
    none = np.zeros(meta["n_frames"], dtype=bool)
    print("K1 {} episodes, {} graded frames; K2 kick-off anchors {} -> {} graded frames forced"
          .format(len(episodes), int((k1 & graded).sum()), kicks, int((k2 & graded).sum())))
    print("mean p_in: ours {:.3f}, OpenJev v3 {:.3f}, heuristic {:.3f}; corr(ours, OpenJev) "
          "{:.2f}".format(ours.mean(), oj.mean(), heur.mean(), np.corrcoef(ours, oj)[0, 1]))

    variants = [("exp_d_base", ours, none), ("exp_d_k", ours, forced),
                ("exp_d_oj50_k", mix([ours, oj], [0.5, 0.5]), forced),
                ("exp_d_heur50_k", mix([ours, heur], [0.5, 0.5]), forced),
                ("exp_d_tri_k", mix([ours, oj, heur], [1 / 3.0] * 3), forced)]
    lo, hi = 0.0, 1.0
    for _ in range(14):
        w = 0.5 * (lo + hi)
        share = d_share(decode(mix([ours, oj], [1 - w, w]), p_home, info, forced[mid],
                               forced), graded)
        lo, hi = (w, hi) if share < TARGET_D else (lo, w)
    w45 = 0.5 * (lo + hi)
    variants.append(("exp_d_oj45_k", mix([ours, oj], [1 - w45, w45]), forced))
    print("OpenJev weight for a {:.0%} D share: {:.3f}".format(TARGET_D, w45))

    print("| output | D share | D in >10 s blackouts | D pre-kick-off 2nd half | possessions | per "
          "alive-min | mean s | median s | < 2 s | < 4 s |\n|---|---|---|---|---|---|---|---|---|---|")
    for name, p_in, mask in variants:
        codes = decode(p_in, p_home, info, mask[mid], mask)
        td.write(TARGET, name, codes)
        print(aggregates(name, codes, meta, f), flush=True)

    print("\ncost sensitivity of exp_d_k (aggregates only, nothing written): c_out, c_in -> D share,"
          " possessions per alive-min")
    for c_out in (0.25, 1.0, 2.0, 4.0):
        cells = []
        for c_in in (5.0, 10.0, 20.0, 40.0):
            codes = decode(ours, p_home, info, forced[mid], forced, (1.0, c_out, c_in))
            alive = graded & ((codes == tb.pm.H) | (codes == tb.pm.A))
            n_spells = len(tb.pm.possession_spells(codes, [(a, b + 1) for _, a, b in meta["halves"]]))
            cells.append("c_in {:g}: {:.1%} / {:.1f}".format(
                c_in, d_share(codes, graded), n_spells / (alive.sum() / 25.0 / 60.0)))
        print("  c_out {:g}: ".format(c_out) + "; ".join(cells), flush=True)


if __name__ == "__main__":
    main()
