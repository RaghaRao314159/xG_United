"""Benchmark ground truth for the open matches, built to follow CHALLENGE.md's labelling rules.

IDSSE (7 DFL matches). Dead ball is DFL's BallStatus, untouched: like BAR-ATH's it is a provider
flag on corrected tracking ("interrupted by the referee"). Possession starts from DFL's
BallPossession and applies the challenge's "controlled" rule with DFL's own event annotations
(rule E). Walking each half's segments in order, as exp_c/conv.py does, a segment of team T that
directly follows the other team during play (no D between) is a change of possession only if T
shows control inside it:

    control  a completed pass or cross (Play), a ShotAtGoal, a BallClaiming (interception, claim,
             hold), or a TacklingGame that T won with the ball under control (WinnerRole
             withBallControl, or WinnerResult ballClaimed / ballControlRetained / dribbledAround /
             layoff), or lost while holding it (LoserRole withBallControl)
    weak     only an unsuccessful Play: DFL records no body part or first-time flag, so a
             misplaced pass after a controlled reception and a hoofed first-time clearance look
             the same
    touch    OtherBallAction (clearances and other touches), BallDeflection, or a TacklingGame won
             without control (ballcontactSucceeded / fouled, WinnerRole withoutBallControl): the
             rule's "simple touch", "deflection" and "aerial duel won without subsequent control"
    neither  DFL logged no on-ball event of T inside the segment

Control keeps the segment; touch-only reverts it to the previous team ("contested or loose:
possession stays with the last team"); weak and neither fall back to Track C's tracking rule R2 (a
T player within 1.5 m of the ball for >= 0.6 s, k as matched to SkillCorner). Segments after a
restart are never touched. Events are placed by dfl_events.py (one offset per half from tracking
alone) and then snapped to the nearest frame within SNAP s where the acting team is within 2 m of
the ball. Nothing here reads the BAR-ATH ground truth.

SkillCorner (20 A-League matches). Phases of play, exactly as exp_sc/sc_build.py labels them:
H/A inside a phase by team_in_possession_id (SkillCorner changes phase only on a controlled
player possession, the challenge's convention), D inside a period outside every phase, X outside
the periods; 10 fps resampled to 25 fps by nearest sample, 25 X rows between periods. Only
match.json and phases_of_play.csv are read.

    python labels.py stats                   # rule E class counts and label aggregates (aggregates only)
    python labels.py build                   # runs/bench-build/labels/{E,R2,DFL}/ + manifest.json
    python labels.py lab --cond clean d85su  # runs/idsse-<m>-<cond>e/ for jev_track/lab scripts

Data: IDSSE (DFL), CC BY 4.0, Bassek et al. 2025; SkillCorner open data, MIT. Prints aggregates only.
"""
import os as _os
ROOT = _os.environ.get("MTAH_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))))
LAB = _os.environ.get("MTAH_LAB") or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

import argparse
import csv
import json
import os
import sys
import urllib.request

import numpy as np

sys.path.insert(0, LAB + "/exp")
sys.path.insert(0, LAB + "/exp_c")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import pipeline, tb  # noqa: E402
import conv  # noqa: E402  (relabel walk, control_runs, R2 keeper)
import dfl_events as dfl  # noqa: E402

pm = tb.pm
H, A, D, X = pm.H, pm.A, pm.D, pm.X
FPS = pipeline.FPS
SEP = tb.SEPARATOR
R2_K = 0.6
SNAP = 0.6                 # s: dfl_events' p90 residual after the per-half offset
SLACK = 3                  # frames an event may sit outside its segment after snapping
LETTERS = np.array(["H", "A", "D", "X"])
BENCH = os.path.join(pipeline.RUNS, "bench-build")   # build workspace; runs/bench = the evaluation folder
from models.training_sources import SKILLCORNER_MATCHES, SKILLCORNER_REVISION
SC_IDS = SKILLCORNER_MATCHES
SC_RAW = "https://raw.githubusercontent.com/SkillCorner/opendata/" + SKILLCORNER_REVISION + "/data/matches/{0}/{0}_{1}"
SC_CACHE = os.path.join(pipeline.RUNS, "bench-cache", "skillcorner", SKILLCORNER_REVISION)
CONTROL_WIN = {"ballClaimed", "ballControlRetained", "dribbledAround", "layoff"}


# -- IDSSE: rule E -----------------------------------------------------------------------------

def evidence(events):
    """[(row, team, 'control' | 'weak' | 'touch')] from synced events."""
    out = []
    for e in events:
        if e["row"] is None:
            continue
        a, k = e["attrs"], e["kind"]
        if k == "Play":
            out.append((e["row"], a.get("Team", -1),
                        "weak" if a.get("Evaluation") == "unsuccessful" else "control"))
        elif k in ("ShotAtGoal", "BallClaiming"):
            out.append((e["row"], a.get("Team", -1), "control"))
        elif k in ("OtherBallAction", "BallDeflection"):
            out.append((e["row"], a.get("Team", -1), "touch"))
        elif k == "TacklingGame":
            won_ctl = (a.get("WinnerRole") == "withBallControl"
                       or a.get("WinnerResult") in CONTROL_WIN)
            out.append((e["row"], a.get("WinnerTeam", -1), "control" if won_ctl else "touch"))
            if a.get("LoserRole") == "withBallControl":
                out.append((e["row"], a.get("LoserTeam", -1), "control"))
    return [x for x in out if x[1] in (0, 1)]


def snap(rows, teams, f, meta):
    """Each evidence row moved to the nearest frame within SNAP s where its team is within
    dfl.HIT_RADIUS m of the ball (clean tracking), else left where the offset put it."""
    near = np.nan_to_num(dfl.team_distance(f), nan=1e9) < dfl.HIT_RADIUS
    w = int(round(SNAP * FPS))
    n = len(near)
    out = []
    for r, t in zip(rows, teams):
        lo, hi = max(0, r - w), min(n - 1, r + w)
        idx = np.flatnonzero(near[lo:hi + 1, t])
        out.append(int(lo + idx[np.argmin(np.abs(lo + idx - r))]) if len(idx) else int(r))
    return np.array(out, dtype=np.int64)


def rule_e(match, return_classes=False):
    """(DFL codes, rule-E codes, R2 codes[, class counts]) for one match."""
    name = tb.dataset_name(match, "clean")
    meta, f = pipeline.load_dataset(name)
    truth = tb.gt_codes(name).astype(np.int8)
    events, _ = dfl.synced(match, meta, f)
    ev = evidence(events)
    rows = snap([r for r, _, _ in ev], [t for _, t, _ in ev], f, meta)
    teams = np.array([t for _, t, _ in ev])
    kinds = np.array([k for _, _, k in ev])
    ctl = conv.control_runs(f)
    r2 = conv.keeper("R2", None, R2_K, ctl)
    classes = {"control": [], "touch": [], "weak_kept": [], "weak_reverted": [],
               "neither_kept": [], "neither_reverted": []}
    team_code = {H: 0, A: 1}

    def keep(a, b, team):
        t = team_code[team]
        inside = (teams == t) & (rows >= a - SLACK) & (rows < b + SLACK)
        L = (b - a) / FPS
        if np.any(inside & (kinds == "control")):
            classes["control"].append(L)
            return True
        weak = bool(np.any(inside & (kinds == "weak")))
        if not weak and np.any(inside & (kinds == "touch")):
            classes["touch"].append(L)
            return False
        kept = r2(a, b, team)
        classes[("weak" if weak else "neither") + ("_kept" if kept else "_reverted")].append(L)
        return kept

    e_codes = conv.relabel(truth, keep)
    r2_codes = conv.relabel(truth, r2)
    if return_classes:
        return truth, e_codes, r2_codes, classes
    return truth, e_codes, r2_codes


# -- SkillCorner: phases -----------------------------------------------------------------------

def sc_fetch(mid):
    folder = os.path.join(SC_CACHE, mid)
    os.makedirs(folder, exist_ok=True)
    for kind in ("match.json", "phases_of_play.csv"):
        path = os.path.join(folder, kind)
        if not os.path.exists(path) or os.path.getsize(path) < 100:
            urllib.request.urlretrieve(SC_RAW.format(mid, kind), path + ".part")
            os.replace(path + ".part", path)
    return folder


def sc_codes(mid):
    """(25 fps codes, halves, home, away) -- sc_build.read_source's labels, resample's layout."""
    folder = sc_fetch(mid)
    with open(os.path.join(folder, "match.json")) as h:
        m = json.load(h)
    home = m["home_team"]["id"]
    periods = sorted((p["period"], p["start_frame"], p["end_frame"]) for p in m["match_periods"])
    n = max(e for _, _, e in periods) + 1
    label = np.full(n, X, dtype=np.int8)
    for _, s, e in periods:
        label[s:e + 1] = D
    with open(os.path.join(folder, "phases_of_play.csv")) as h:
        for row in csv.DictReader(h):
            if not row["team_in_possession_id"]:
                continue
            s, e = int(row["frame_start"]), int(row["frame_end"])
            label[s:e + 1] = H if int(float(row["team_in_possession_id"])) == home else A
    inside = np.zeros(n, dtype=bool)
    for _, s, e in periods:
        inside[s:e + 1] = True
    label[~inside] = X
    parts, halves, row = [], [], 0
    for period, s, e in periods:
        if halves:
            parts.append(np.full(SEP, X, dtype=np.int8))
            row += SEP
        j = np.arange(int((e - s) * FPS // 10) + 1)          # sc_build.grid
        a = s + (2 * j) // 5
        w = ((2 * j) % 5) / 5.0
        near = np.minimum(a + (w > 0.5), e)
        parts.append(label[near])
        halves.append([period, row, row + len(j) - 1])
        row += len(j)
    names = (m["home_team"]["short_name"], m["away_team"]["short_name"])
    return np.concatenate(parts), halves, names


# -- aggregates --------------------------------------------------------------------------------

def handover_stats(codes_list):
    """Pooled: possessions per alive-minute, share of possessions < 2 s / < 4 s, and of
    team-after-team hand-overs < 1.5 s; D share."""
    n = 0
    alive = graded = dead = 0
    lens, ho = [], []
    for c in codes_list:
        runs = pm.valid_runs(c)
        sp = pm.possession_spells(c, runs)
        n += len(sp)
        lens += [(b - a) / FPS for a, b, _ in sp]
        g = c != X
        graded += g.sum()
        dead += (c == D).sum()
        alive += np.isin(c, (H, A)).sum()
        for start, stop in runs:
            segs = pm.segments(c, [(start, stop)])
            for i in range(1, len(segs)):
                if segs[i][2] in (H, A) and segs[i - 1][2] in (H, A) and segs[i][2] != segs[i - 1][2]:
                    ho.append((segs[i][1] - segs[i][0]) / FPS)
    lens, ho = np.array(lens), np.array(ho)
    return dict(n=n, per_min=n / (alive / FPS / 60.0), u2=np.mean(lens < 2), u4=np.mean(lens < 4),
                median=np.median(lens), ho_u15=np.mean(ho < 1.5) if len(ho) else np.nan,
                D=dead / graded)


def line(label, s, per_match=None):
    rng = " ({:.1f}-{:.1f})".format(min(per_match), max(per_match)) if per_match else ""
    return ("| {} | {} | {:.2f}{} | {:.0%} | {:.0%} | {:.1f} | {:.0%} | {:.1%} |".format(
        label, s["n"], s["per_min"], rng, s["u2"], s["u4"], s["median"], s["ho_u15"], s["D"]))


HEADER = ("| labels | possessions | per alive-min (match range) | < 2 s | < 4 s | median s | "
          "hand-overs < 1.5 s | D share |\n|---|---|---|---|---|---|---|---|")


def agreement(ref, other):
    r = pm.evaluate(ref, other, bootstrap_n=0)
    return r["spell_f1_mean"], r["macro_f1"], r["accuracy"]


def stitch(arrays):
    out = []
    for i, a in enumerate(arrays):
        if i:
            out.append(np.full(SEP, X, dtype=np.int8))
        out.append(a)
    return np.concatenate(out)


def cmd_stats(args):
    sets = {"DFL": [], "E": [], "R2": []}
    cls = {}
    for m in dfl.MATCHES:
        t, e, r2, c = rule_e(m, return_classes=True)
        sets["DFL"].append(t); sets["E"].append(e); sets["R2"].append(r2)
        for k, v in c.items():
            cls.setdefault(k, []).extend(v)
    tot = sum(len(v) for v in cls.values())
    print("rule E, opposing segments during play ({} in 7 matches):".format(tot))
    for k, v in cls.items():
        v = np.array(v)
        print("  {:<17} {:5d} ({:4.0%}), length p50 {:.1f} s, p90 {:.1f} s".format(
            k, len(v), len(v) / tot, np.median(v) if len(v) else 0, np.percentile(v, 90) if len(v) else 0))
    sc = [sc_codes(i)[0] for i in SC_IDS]
    print(HEADER)
    print(line("SkillCorner phases (20)", handover_stats(sc), [handover_stats([c])["per_min"] for c in sc]))
    for k in ("E", "R2", "DFL"):
        print(line("IDSSE " + k + " (7)", handover_stats(sets[k]),
                   [handover_stats([c])["per_min"] for c in sets[k]]))
    for a, b in (("E", "R2"), ("E", "DFL"), ("R2", "DFL")):
        sf, mf, acc = agreement(stitch(sets[a]), stitch(sets[b]))
        print("IDSSE {} scored against {} as truth: spellF1 {:.3f} macroF1 {:.3f} accuracy {:.3f}; "
              "frames differing {:.1%}".format(b, a, sf, mf, acc, np.mean(stitch(sets[a]) != stitch(sets[b]))))


def write_codes(path, codes):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pipeline.baseline.write_submission(path, LETTERS[codes])


def cmd_build(args):
    manifest = {"fps": FPS, "separator_rows": SEP, "label_sets": {
        "E": "primary: IDSSE rule E (DFL events confirm control) + SkillCorner phases",
        "R2": "alternate: IDSSE rule R2 k=0.6 (tracking proximity) + SkillCorner phases",
        "DFL": "reference: IDSSE raw DFL BallPossession + SkillCorner phases"},
        "matches": []}
    for m in dfl.MATCHES:
        t, e, r2 = rule_e(m)
        meta, _ = pipeline.load_dataset(tb.dataset_name(m, "clean"))
        game = "IDSSE-" + m
        for tag, c in (("E", e), ("R2", r2), ("DFL", t)):
            write_codes(os.path.join(BENCH, "labels", tag, game + "_possession_gt.csv"), c)
        manifest["matches"].append({"game": game, "source": "IDSSE (DFL), CC BY 4.0",
                                    "n_rows": len(t), "halves": meta["halves"],
                                    "home": meta["home"], "away": meta["away"]})
        print("  {}: {} rows".format(game, len(t)), flush=True)
    for i in SC_IDS:
        c, halves, names = sc_codes(i)
        game = "SC-" + i
        for tag in ("E", "R2", "DFL"):
            write_codes(os.path.join(BENCH, "labels", tag, game + "_possession_gt.csv"), c)
        manifest["matches"].append({"game": game, "source": "SkillCorner open data, MIT",
                                    "n_rows": len(c), "halves": halves,
                                    "home": names[0], "away": names[1]})
    with open(os.path.join(BENCH, "labels", "manifest.json"), "w") as h:
        json.dump(manifest, h, indent=1)
    print("wrote {} matches x 3 label sets to {}".format(len(manifest["matches"]),
                                                        os.path.join(BENCH, "labels")))


def cmd_lab(args):
    """runs/idsse-<m>-<cond>e/: rule-E gt.csv beside symlinks to <cond>'s files (conv.cmd_build's
    layout), so every jev_track/lab script can take cond '<cond>e'."""
    for m in dfl.MATCHES:
        t, e, _ = rule_e(m)
        for cond in args.cond:
            src = pipeline.dataset_dir(tb.dataset_name(m, cond))
            if not os.path.isdir(src):
                print("  skip {} {}: not built".format(m, cond))
                continue
            assert (tb.gt_codes(tb.dataset_name(m, cond)) == t).all(), (m, cond)
            dst = pipeline.dataset_dir(tb.dataset_name(m, cond + "e"))
            os.makedirs(os.path.join(dst, "results"), exist_ok=True)
            for name in os.listdir(src):
                if name in ("gt.csv", "results", "score") or name.startswith("answers_exp_"):
                    continue
                link = os.path.join(dst, name)
                if not os.path.lexists(link):
                    os.symlink(os.path.join(src, name), link)
            write_codes(os.path.join(dst, "gt.csv"), e)
        print("  {}: rule E written for {}".format(m, args.cond), flush=True)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("stats")
    sub.add_parser("build")
    lab = sub.add_parser("lab")
    lab.add_argument("--cond", nargs="+", default=["clean", "d85su"])
    args = p.parse_args()
    {"stats": cmd_stats, "build": cmd_build, "lab": cmd_lab}[args.cmd](args)


if __name__ == "__main__":
    main()
