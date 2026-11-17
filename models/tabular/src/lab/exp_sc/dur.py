"""Idea #4: a 5 Hz Viterbi that knows how short possessions and stoppages can be. Each of H, A, D is a
chain of sub-states a path must walk through before it may leave, so a possession lasts at least
`alive` blocks and a stoppage at least `dead` blocks (Track V's flipped frames were right but split
possessions into fragments). The first and last runs of a half may be cut short. Minimum lengths
and costs are chosen per held-out match on the other matches, never in-sample, from MINS x GRID.

decode_min(data, matches, label) takes ev.py's per-match dicts and writes <label>.csv per match.
"""
import os as _os
ROOT = _os.environ.get("MTAH_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))))
LAB = _os.environ.get("MTAH_LAB") or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
LABCACHE = _os.path.join(ROOT, "runs", "lab-cache")

import itertools
import sys

import numpy as np

sys.path.insert(0, LAB + "/exp")
sys.path.insert(0, LAB + "/exp_a")
sys.path.insert(0, LAB + "/exp_ab")
from common import A, D, H, tb  # noqa: E402
import ab_decode as abd  # noqa: E402
import ta5_decode as td  # noqa: E402

BLOCKS_PER_S = 5
MINS = [(a, d) for a in (1, 5, 10) for d in (1, 10, 20)]    # blocks: (alive, dead) minimum runs
GRID = list(itertools.product((0.5, 1.0, 2.0), (0.1, 0.5, 2.0), (5.0, 20.0, 80.0)))
COSTS = np.stack([td.cost_matrix(*g) for g in GRID])


def viterbi_min(le, costs, alive, dead):
    """(G, n) macro-state paths with minimum run lengths. le (n, 3), costs (G, 3, 3)."""
    length = {H: alive, A: alive, D: dead}
    offs, s = {}, 0
    for c in (H, A, D):
        offs[c] = s
        s += length[c]
    last = {c: offs[c] + length[c] - 1 for c in (H, A, D)}
    macro = np.empty(s, dtype=np.int8)
    for c in (H, A, D):
        macro[offs[c]:last[c] + 1] = c
    n, g = len(le), len(costs)
    score = np.full((g, s), -np.inf)
    for c in (H, A, D):
        score[:, offs[c]:last[c] + 1] = le[0, c]      # a half may open mid-run
    back = np.empty((n, g, s), dtype=np.int16)
    rows = np.arange(g)
    for t in range(1, n):
        new = np.empty((g, s))
        bp = np.empty((g, s), dtype=np.int16)
        for c in (H, A, D):
            o, e = offs[c], last[c]
            srcs = [last[c2] for c2 in (H, A, D) if c2 != c]
            cand = np.stack([score[:, last[c2]] - costs[:, c2, c] for c2 in (H, A, D) if c2 != c], 1)
            if e == o:
                cand = np.column_stack([cand, score[:, o]])
                srcs.append(o)
            k = cand.argmax(axis=1)
            new[:, o] = cand[rows, k]
            bp[:, o] = np.array(srcs, dtype=np.int16)[k]
            if e > o:
                new[:, o + 1:e + 1] = score[:, o:e]
                bp[:, o + 1:e + 1] = np.arange(o, e, dtype=np.int16)
                stay = score[:, e] > score[:, e - 1]
                new[:, e] = np.where(stay, score[:, e], score[:, e - 1])
                bp[:, e] = np.where(stay, e, e - 1)
            new[:, o:e + 1] += le[t, c]
        score = new
        back[t] = bp
    state = score.argmax(axis=1)
    path = np.empty((g, n), dtype=np.int8)
    path[:, -1] = macro[state]
    for t in range(n - 1, 0, -1):
        state = back[t, rows, state]
        path[:, t - 1] = macro[state]
    return path


def decode_match(d, alive, dead, team="p1"):
    p_in, p_home = d["p_in"]["tbh"], d["info"][team]
    emission = np.column_stack([p_in * p_home, p_in * (1 - p_home), 1 - p_in])
    le = np.log(np.clip(emission, td.FLOOR, 1.0))
    info = d["info"]
    paths = np.empty((len(GRID), len(le)), dtype=np.int8)
    for h in np.unique(info["half"]):
        idx = np.flatnonzero(info["half"] == h)
        paths[:, idx] = viterbi_min(le[idx], COSTS, alive, dead)
    return td.frame_codes(paths, info)


def decode_min(data, matches, label, team="p1"):
    """Every match decoded at every (MINS, GRID) option; each held-out match gets the option best
    on the others. Writes <label>.csv per match and prints the choices."""
    counts = {}
    for m in matches:
        d = data[m]
        gt_sp = td.spells(d["gt"], d["runs"])
        rows = []
        for alive, dead in MINS:
            for c in decode_match(d, alive, dead, team):
                rows.append(td.counts(gt_sp, td.spells(np.where(d["gt"] == tb.pm.X, tb.pm.X, c),
                                                       d["runs"])))
        counts[m] = rows
    options = [(mn, gr) for mn in MINS for gr in GRID]
    insample = [td.pooled_f1([counts[m][k] for m in matches]) for k in range(len(options))]
    chosen, held_rows = [], []
    for m in matches:
        others = [o for o in matches if o != m]
        k = int(np.argmax([td.pooled_f1([counts[o][j] for o in others])
                           for j in range(len(options))]))
        (alive, dead), _ = options[k]
        codes = decode_match(data[m], alive, dead, team)[GRID.index(options[k][1])]
        td.write(data[m]["dataset"], label, codes)
        chosen.append(options[k])
        held_rows.append(counts[m][k])
    tally = {}
    for (mn, gr) in chosen:
        key = "min alive/dead {:.1f}/{:.1f} s".format(mn[0] / BLOCKS_PER_S, mn[1] / BLOCKS_PER_S)
        tally[key] = tally.get(key, 0) + 1
    best = int(np.argmax(insample))
    print("  {}: LOMO spellF1 {:.3f} (in-sample best {:.3f} at mins {} costs {}); per fold {}".format(
        label, td.pooled_f1(held_rows), insample[best], options[best][0], options[best][1], tally),
        flush=True)
