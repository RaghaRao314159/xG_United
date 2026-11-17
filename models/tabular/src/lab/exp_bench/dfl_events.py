"""DFL event data for the 7 open IDSSE matches: parsing, team mapping and synchronisation to the
tracking rows of runs/idsse-<m>/ (jev_track/idsse.py's layout). Used by labels.py to confirm
"controlled" possession changes with the provider's own annotations.

Synchronisation. DFL's raw events carry a wall-clock EventTime logged live, not a tracking frame.
Each half gets one offset: row = first_row + 25 * (EventTime - kick-off EventTime + offset). The
offset is the one that puts the most single-team on-ball events (passes, shots, ball claims, other
ball actions) at a row where the acting team's nearest player is within HIT_RADIUS m of the ball.
Only clean tracking (features.npz d_home / d_away) is read, never a label, so the sync cannot lean
towards any labelling rule. The residual spread after that offset is reported as the tolerance a
rule should allow.

    python dfl_events.py sync          # offsets and residuals per match and half
Data: IDSSE (DFL), CC BY 4.0, Bassek et al., Scientific Data 12, 195 (2025). Prints aggregates only.
"""
import os as _os
ROOT = _os.environ.get("MTAH_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))))

import argparse
import datetime as dt
import json
import os
import sys
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np

sys.path.insert(0, os.path.join(ROOT, "src"))  # xG United: folder renamed from jev_track
import pipeline  # noqa: E402

FPS = pipeline.FPS
MATCHES = {"J03WMX": "000001", "J03WN1": "000001", "J03WOH": "000002", "J03WOY": "000002",
           "J03WPY": "000002", "J03WQQ": "000002", "J03WR9": "000002"}
# Resolve the submission root even when this script runs from its model folder.
from pathlib import Path as _SourcePath
import sys as _source_sys
_source_sys.path.insert(0, str(_SourcePath(__file__).resolve().parents[5]))
from models.training_sources import IDSSE_MIRROR, IDSSE_REVISION
MIRROR = IDSSE_MIRROR
CACHE = os.path.join(os.environ.get("BENCH_CACHE") or os.path.join(pipeline.RUNS, "bench-cache", "idsse-events"), IDSSE_REVISION)
SECTIONS = {"firstHalf": 1, "secondHalf": 2}
HIT_RADIUS = 2.0           # m: "the acting team is at the ball"
SEARCH = 15.0              # s: offsets searched either side of the kick-off alignment
STEP = 1.0 / FPS
SYNC_KINDS = ("Play", "ShotAtGoal", "BallClaiming", "OtherBallAction")


def _file(kind, match):
    stem = {"events": "DFL_03_02_events_raw", "info": "DFL_02_01_matchinformation"}[kind]
    name = "{}_DFL-COM-{}_DFL-MAT-{}.xml".format(stem, MATCHES[match], match)
    path = os.path.join(CACHE, name)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        os.makedirs(CACHE, exist_ok=True)
        urllib.request.urlretrieve(MIRROR + name, path + ".part")
        os.replace(path + ".part", path)
    return path


def _time(stamp):
    return dt.datetime.fromisoformat(stamp).timestamp()


def team_roles(match):
    """{DFL team id: 0 home / 1 away}."""
    root = ET.parse(_file("info", match)).getroot()
    return {t.get("TeamId"): (0 if t.get("Role") == "home" else 1) for t in root.iter("Team")}


def load(match):
    """Events as dicts: kind (top tag), sub (first child tag or None), t (epoch s), attrs (the
    top element's attributes plus its children's, child keys prefixed 'sub.'), team fields mapped
    to 0/1. Delete markers are dropped (they carry no reference to what they delete)."""
    roles = team_roles(match)
    out = []
    for e in ET.parse(_file("events", match)).getroot():
        if not len(e):
            continue
        top = e[0]
        if top.tag == "Delete":
            continue
        attrs = dict(top.attrib)
        sub = top[0].tag if len(top) else None
        if len(top):
            attrs.update({"sub." + k: v for k, v in top[0].attrib.items()})
            if len(top[0]):
                attrs.update({"subsub." + k: v for k, v in top[0][0].attrib.items()})
                attrs["subsub"] = top[0][0].tag
        for key in ("Team", "WinnerTeam", "LoserTeam", "TeamFouled", "TeamFouler"):
            if key in attrs:
                attrs[key] = roles.get(attrs[key], -1)
        out.append({"kind": top.tag, "sub": sub, "t": _time(e.get("EventTime")), "attrs": attrs,
                    "id": e.get("EventId")})
    out.sort(key=lambda x: x["t"])
    return out


def kickoffs(events):
    """{period: kick-off epoch s, end epoch s} from the KickOff / FinalWhistle events that name
    their GameSection (kick-offs after goals do not)."""
    ko, end = {}, {}
    for e in events:
        sec = SECTIONS.get(e["attrs"].get("GameSection"))
        if sec is None:
            continue
        if e["kind"] == "KickOff" and sec not in ko:
            ko[sec] = e["t"]
        if e["kind"] == "FinalWhistle":
            end[sec] = e["t"]
    return {p: (ko[p], end.get(p, ko[p] + 3 * 3600)) for p in ko}


def actor_team(e):
    a = e["attrs"]
    if e["kind"] == "TacklingGame":
        return a.get("WinnerTeam", -1)
    return a.get("Team", -1)


def assign(events, meta, offsets=None):
    """Adds 'period' and 'row' (float, before rounding) to every event inside a half; offsets
    {period: s} default to 0 (kick-off alignment only)."""
    ko = kickoffs(events)
    halves = {p: (a, b) for p, a, b in meta["halves"]}
    for e in events:
        e["period"], e["row"] = None, None
        for p, (t0, t1) in ko.items():
            if p in halves and t0 - 1.0 <= e["t"] <= t1 + 1.0:
                first, last = halves[p]
                row = first + FPS * (e["t"] - t0 + (offsets or {}).get(p, 0.0))
                e["period"], e["row"] = p, float(row)
    return events


def team_distance(f):
    """(n, 2) nearest distance of each team's players to the ball, m; NaN where unknown."""
    return np.stack([f["d_home"], f["d_away"]], axis=1).astype(np.float32)


def fit_offsets(events, meta, f):
    """{period: offset s} maximising the share of SYNC_KINDS events whose acting team is within
    HIT_RADIUS m of the ball at the event's row; plus per-period diagnostics."""
    dist = team_distance(f)
    hit = np.nan_to_num(dist, nan=1e9) < HIT_RADIUS
    assign(events, meta)
    halves = {p: (a, b) for p, a, b in meta["halves"]}
    grid = np.arange(-SEARCH, SEARCH + 1e-9, STEP)
    out, diag = {}, {}
    for p, (first, last) in halves.items():
        evs = [e for e in events if e["period"] == p and e["kind"] in SYNC_KINDS
               and actor_team(e) in (0, 1)]
        if not evs:
            continue
        base = np.array([e["row"] for e in evs])
        team = np.array([actor_team(e) for e in evs])
        rows = np.rint(base[None, :] + FPS * grid[:, None]).astype(np.int64)
        ok = (rows >= first) & (rows <= last)
        rows = np.clip(rows, first, last)
        score = (hit[rows, team[None, :]] & ok).mean(axis=1)
        best = int(np.argmax(score))
        out[p] = float(grid[best])
        # residual: shift to the nearest hit within +-2 s after the global offset
        r0 = np.rint(base + FPS * grid[best]).astype(np.int64)
        shifts = []
        for r, tm in zip(r0, team):
            lo, hi = max(first, r - 2 * FPS), min(last, r + 2 * FPS)
            idx = np.flatnonzero(hit[lo:hi + 1, tm])
            if len(idx):
                shifts.append((lo + idx[np.argmin(np.abs(lo + idx - r))] - r) / FPS)
        shifts = np.abs(np.array(shifts))
        diag[p] = dict(n=len(evs), hit_at_zero=float(score[np.argmin(np.abs(grid))]),
                       hit_best=float(score[best]), offset=out[p],
                       found=len(shifts) / len(evs), res_p50=float(np.median(shifts)),
                       res_p90=float(np.percentile(shifts, 90)))
    return out, diag


def synced(match, meta=None, f=None):
    """Events of one match with integer 'row' after the fitted offsets; and the diagnostics."""
    name = "idsse-" + match
    if meta is None:
        meta, f = pipeline.load_dataset(name)
    events = load(match)
    offsets, diag = fit_offsets(events, meta, f)
    assign(events, meta, offsets)
    for e in events:
        if e["row"] is not None:
            e["row"] = int(round(e["row"]))
    return events, diag


def cmd_sync(args):
    print("| match | half | events | hit at kick-off alignment | offset s | hit at offset | "
          "nearest hit found | residual p50 / p90 s |\n|---|---|---|---|---|---|---|---|")
    for m in MATCHES:
        _, diag = synced(m)
        for p, d in sorted(diag.items()):
            print("| {} | {} | {n} | {hit_at_zero:.0%} | {offset:+.2f} | {hit_best:.0%} | "
                  "{found:.0%} | {res_p50:.2f} / {res_p90:.2f} |".format(m, p, **d))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("sync")
    args = p.parse_args()
    {"sync": cmd_sync}[args.cmd](args)


if __name__ == "__main__":
    main()
