"""SkillCorner open data (MIT, github.com/SkillCorner/opendata) as jev_track datasets: 20 A-League
2024/25 matches of broadcast tracking, built the way jev_track/idsse.py builds IDSSE so every lab
script runs on them unchanged (tabular.dataset_name maps match SC<id> to runs/sc-<id>).

Labels. Phases of play exist only while the ball is in play and name the team in possession,
which changes only when an opponent starts a controlled player possession -- the challenge's
convention, unlike DFL's. So: inside a phase H or A by team_in_possession_id, outside a phase but
inside a period D, outside the periods X. SkillCorner derives "out of play" from its event matching
and a 30 s no-event rule, not from a referee flag, so its dead-ball edges are softer than DFL's.

Tracking. 10 fps, resampled to 25 fps by linear interpolation between two observed samples (NaN if
either is missing); labels and flags come from the nearest sample. As in BAR-ATH's tactical ATD the
ball is missing where SkillCorner did not detect it (its extrapolated position is dropped) and a
player is missing where extrapolated (is_detected false). cut marks frames with no detected player
at all: replays, close-ups and crowd shots, the broadcast's own cutaways. The ball is then
Gaussian-smoothed with the sigma whose seen-frame acceleration p90 / p99 match BAR-ATH's, smooth.py's
label-free criterion (BAR-ATH aggregates only).

    MTAH_ROOT=/root/mtah python sc_build.py            # download to SC_CACHE, build, print
    MTAH_ROOT=/root/mtah python sc_build.py --list     # the MTAH_MATCHES value for the SC set
Writes runs/sc-<id>/ {meta.json, features.npz, raw_tb.npz, gt.csv, extra.npz (cut, group)} and
runs/sc-stats.json (ball-missing aggregates for hard_idsse.py). Prints aggregates only.
"""
import os as _os
ROOT = _os.environ.get("MTAH_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))))
LAB = _os.environ.get("MTAH_LAB") or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
LABCACHE = _os.path.join(ROOT, "runs", "lab-cache")

import argparse
import csv
import json
import os
import sys
import time
import urllib.request

import numpy as np

sys.path.insert(0, LAB + "/exp")
sys.path.insert(0, LAB + "/exp_t")
sys.path.insert(0, LAB + "/exp_u")
from common import pipeline, tb  # noqa: E402
import smooth  # noqa: E402  (Track T: smooth_ball, acc_of)
import unfreeze  # noqa: E402  (recompute: features from a ball and raw positions)

pm = tb.pm
# Use the exact source revision and match list in the submission manifest.
from pathlib import Path as _SourcePath
sys.path.insert(0, str(_SourcePath(__file__).resolve().parents[5]))
from models.training_sources import SKILLCORNER_REVISION, SKILLCORNER_MATCHES, SKILLCORNER_FILES
RAW = "https://raw.githubusercontent.com/SkillCorner/opendata/" + SKILLCORNER_REVISION + "/data/matches/{0}/{0}_{1}"
LFS = "https://media.githubusercontent.com/media/SkillCorner/opendata/" + SKILLCORNER_REVISION + "/data/matches/{0}/{0}_{1}"
FILES = SKILLCORNER_FILES
CACHE_DIR = os.path.join(os.environ.get("SC_CACHE", os.path.expanduser("~/sc_cache")), SKILLCORNER_REVISION)
SIGMAS = (0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0)
FPS = pipeline.FPS


def match_ids():
    """The 20 matches used by the supplied weights; never discover new upstream matches."""
    return list(SKILLCORNER_MATCHES)


def fetch(mid):
    folder = os.path.join(CACHE_DIR, mid)
    os.makedirs(folder, exist_ok=True)
    for kind in FILES:
        path = os.path.join(folder, kind)
        if os.path.exists(path) and os.path.getsize(path) > 1000:
            continue
        url = (LFS if kind.endswith(".jsonl") else RAW).format(mid, kind)
        urllib.request.urlretrieve(url, path + ".part")
        os.replace(path + ".part", path)
    return folder


def read_source(folder):
    """10 fps arrays indexed by SkillCorner frame number."""
    with open(os.path.join(folder, "match.json")) as h:
        m = json.load(h)
    home, away = m["home_team"]["id"], m["away_team"]["id"]
    length, width = float(m["pitch_length"]), float(m["pitch_width"])
    squad = sorted((p for p in m["players"] if p["team_id"] in (home, away)),
                   key=lambda p: (p["team_id"] != home, p["id"]))
    slot = {p["id"]: i for i, p in enumerate(squad)}
    team = np.array([0 if p["team_id"] == home else 1 for p in squad], dtype=np.int8)
    periods = sorted((p["period"], p["start_frame"], p["end_frame"]) for p in m["match_periods"])
    n = max(e for _, _, e in periods) + 1
    ball = np.full((n, 2), np.nan, dtype=np.float32)
    players = np.full((n, len(squad), 2), np.nan, dtype=np.float32)
    cut = np.zeros(n, dtype=bool)
    group = np.full(n, -1, dtype=np.int8)
    unknown = 0
    with open(os.path.join(folder, "tracking_extrapolated.jsonl")) as h:
        for line in h:
            fr = json.loads(line)
            i = fr["frame"]
            if fr["period"] is None or i >= n:
                continue
            b = fr["ball_data"] or {}
            if b.get("is_detected") and b.get("x") is not None:
                ball[i] = ((b["x"] + length / 2) / length, (b["y"] + width / 2) / width)
            detected = 0
            for p in fr["player_data"] or []:
                k = slot.get(p["player_id"])
                if k is None:
                    unknown += 1
                    continue
                if p.get("is_detected") and p["x"] is not None:
                    players[i, k] = ((p["x"] + length / 2) / length, (p["y"] + width / 2) / width)
                    detected += 1
            cut[i] = detected == 0
            g = (fr.get("possession") or {}).get("group")
            group[i] = 0 if g == "home team" else 1 if g == "away team" else -1
    label = np.full(n, "X", dtype="<U1")
    for _, s, e in periods:
        label[s:e + 1] = "D"
    with open(os.path.join(folder, "phases_of_play.csv")) as h:
        for row in csv.DictReader(h):
            if not row["team_in_possession_id"]:
                continue
            s, e = int(row["frame_start"]), int(row["frame_end"])
            label[s:e + 1] = "H" if int(float(row["team_in_possession_id"])) == home else "A"
    inside = np.zeros(n, dtype=bool)
    for _, s, e in periods:
        inside[s:e + 1] = True
    label[~inside] = "X"   # a phase never counts outside its period
    names = (m["home_team"]["short_name"], m["away_team"]["short_name"])
    return dict(ball=ball, players=players, team=team, cut=cut, group=group, label=label,
                periods=periods, pitch=[length, width], names=names, unknown=unknown)


def grid(s, e):
    """(lower sample, weight of the upper one) for 25 fps rows over 10 fps samples s..e."""
    j = np.arange(int((e - s) * FPS // 10) + 1)
    a = s + (2 * j) // 5
    return a, ((2 * j) % 5) / 5.0


def interp(values, a, w):
    b = np.minimum(a + 1, len(values) - 1)
    shape = (-1,) + (1,) * (values.ndim - 1)
    w = w.reshape(shape)
    out = (1 - w) * values[a] + w * values[b]
    return np.where(w == 0, values[a], out).astype(np.float32)


def resample(src):
    """25 fps rows, one block per period, tb.SEPARATOR X rows between periods (as idsse.layout)."""
    parts = {k: [] for k in ("ball", "players", "cut", "group", "label")}
    halves, row = [], 0
    for period, s, e in src["periods"]:
        if halves:
            sep = tb.SEPARATOR
            parts["ball"].append(np.full((sep, 2), np.nan, dtype=np.float32))
            parts["players"].append(np.full((sep,) + src["players"].shape[1:], np.nan,
                                            dtype=np.float32))
            parts["cut"].append(np.zeros(sep, dtype=bool))
            parts["group"].append(np.full(sep, -1, dtype=np.int8))
            parts["label"].append(np.full(sep, "X", dtype="<U1"))
            row += sep
        a, w = grid(s, e)
        near = np.minimum(a + (w > 0.5), e)
        parts["ball"].append(interp(src["ball"], a, w))
        parts["players"].append(interp(src["players"], a, w))
        for k in ("cut", "group", "label"):
            parts[k].append(src[k][near])
        halves.append([period, row, row + len(a) - 1])
        row += len(a)
    return {k: np.concatenate(v) for k, v in parts.items()}, halves


def stats_line(name, out, meta, codes, f):
    graded = codes != pm.X
    dead = graded & (codes == pm.D)
    alive = graded & ~dead
    runs = pm.valid_runs(codes)
    spells = pm.possession_spells(codes, runs)
    lengths = np.array([b - a for a, b, _ in spells]) / float(FPS)
    alive_min = alive.sum() / FPS / 60.0
    seen = f["seen"]
    cut = out["cut"]
    known = alive & (out["group"] >= 0)
    agree = np.mean(out["group"][known] == np.where(codes[known] == pm.H, 0, 1)) if known.any() else np.nan
    shown = (~np.isnan(out["players"][..., 0])).sum(axis=1)
    return ("| {} | {:.1f} | {:.1%} | {} | {:.1f} | {:.1f} | {:.0%} / {:.0%} | {:.0f} | {:.1%} | {:.0%} | "
            "{:.0%} |".format(name, graded.sum() / FPS / 60.0, dead.sum() / graded.sum(), len(spells),
                              len(spells) / alive_min, lengths.mean(), seen[dead].mean(),
                              seen[alive].mean(), np.median(shown[graded & ~cut]), cut[graded].mean(),
                              np.mean(codes[graded & cut] == pm.D), agree))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()
    ids = match_ids()
    if args.list:
        print(",".join("SC" + i for i in ids))
        return
    if args.only and not set(args.only).issubset(ids):
        parser.error("--only must select matches from training_data.lock.json")
    ids = args.only or ids
    target_meta, _ = pipeline.load_dataset(tb.TARGET)
    target = smooth.acc_of(smooth.load_raw(tb.TARGET), target_meta)
    built = {}
    for mid in ids:
        started = time.time()
        src = read_source(fetch(mid))
        out, halves = resample(src)
        name = tb.dataset_name("SC" + mid, "clean")
        meta = {"name": name, "n_frames": len(out["label"]), "pitch": src["pitch"],
                "home": src["names"][0], "away": src["names"][1], "smart_tagging": False,
                "halves": halves}
        print("  {}: {} rows, {} slots, {} unknown-player samples, {:.0f} s".format(
            name, meta["n_frames"], len(src["team"]), src["unknown"], time.time() - started),
            flush=True)
        built[mid] = (name, out, meta, src["team"])

    print("BAR-ATH acc p50/p90/p99 {}".format(target.round(1).tolist()))
    best = None
    for sigma in SIGMAS:
        got = []
        for name, out, meta, team in built.values():
            got.append(smooth.acc_of({"ball": smooth.smooth_ball(out["ball"], sigma),
                                      "pitch": np.array(meta["pitch"])}, meta))
        got = np.mean(got, axis=0)
        err = float(np.abs(np.log(got[1:] / target[1:])).sum())
        print("  sigma {:g}: SkillCorner acc p50/p90/p99 {} | log error (p90 + p99) {:.3f}".format(
            sigma, got.round(1).tolist(), err), flush=True)
        if best is None or err < best[1]:
            best = (sigma, err)
    sigma = best[0]
    print("sigma {:g} chosen".format(sigma))

    print("| match | graded min | D share | possessions | per alive-min | mean s | ball seen dead / "
          "alive | players shown | cut | D when cut | group agrees |\n"
          "|---|---|---|---|---|---|---|---|---|---|---|")
    gaps, hidden, hidden_dead, graded_n = [], 0, 0, 0
    for mid, (name, out, meta, team) in built.items():
        ball = smooth.smooth_ball(out["ball"], sigma).astype(np.float32)
        raw = {"ball": ball, "ball_z": np.full(len(ball), np.nan, dtype=np.float32),
               "players": out["players"], "team": team,
               "pitch": np.array(meta["pitch"], dtype=np.float32)}
        f = unfreeze.recompute(ball, raw)
        folder = pipeline.dataset_dir(name)
        os.makedirs(os.path.join(folder, "results"), exist_ok=True)
        pipeline.write_dataset(name, meta, f)
        np.savez_compressed(os.path.join(folder, "raw_tb.npz"), **raw)
        np.savez_compressed(os.path.join(folder, "extra.npz"), cut=out["cut"], group=out["group"])
        pipeline.baseline.write_submission(os.path.join(folder, "gt.csv"), out["label"])
        codes = tb.gt_codes(name).astype(np.int8)
        print(stats_line(name, out, meta, codes, f), flush=True)
        graded = codes != pm.X
        lost = graded & ~f["seen"]
        hidden += int(lost.sum())
        hidden_dead += int((lost & (codes == pm.D)).sum())
        graded_n += int(graded.sum())
        for _, first, last in meta["halves"]:
            starts, stops = tb.run_bounds(~f["seen"][first:last + 1])
            gaps.extend((stops - starts).tolist())
    stats = {"sigma": sigma, "hidden_share": hidden / graded_n, "dead_share": hidden_dead / hidden,
             "gap_lengths": gaps, "matches": ["SC" + m for m in built]}
    with open(os.path.join(pipeline.RUNS, "sc-stats.json"), "w") as h:
        json.dump(stats, h)
    tb.describe_gaps(gaps, "SkillCorner ball gaps")
    print("SkillCorner: ball missing on {:.1%} of graded frames, {:.1%} of that dead".format(
        stats["hidden_share"], stats["dead_share"]))


if __name__ == "__main__":
    main()
