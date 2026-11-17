"""Build the open IDSSE matches as jev_track datasets, with ground truth for every frame.

A decision model is only worth trusting once it has been scored on matches whose answers are known.
The challenge's own ground truth is private, its public five minutes hold too few possessions to
rank anything, and the competition terms keep the match data away from hosted models, so Jev
cannot be measured on it at all. IDSSE can carry that measurement, because it is the one open
dataset labelled the way the challenge's ground truth is. Every frame of its DFL tracking (TRACAB,
25 fps) carries the provider's BallStatus, 0 when "the match is interrupted by the referee" and 1
otherwise, and its BallPossession, 1 for home and 2 for away. Those map onto the challenge labels
one to one:

    BallStatus 0                        D
    BallStatus 1, BallPossession 1      H
    BallStatus 1, BallPossession 2      A
    BallStatus 1, no BallPossession     X   not graded; counted and printed
    between two periods                 X

Each period is one block in frame order, with SEPARATOR X frames between periods. The scorer reads
every run of non-X frames as a half, and pipeline.position_speed never differences positions across
the break.

Two kloppy details shape this module. kloppy 3.18.0, the version requirements-starter.txt pins,
fetches IDSSE from figshare, which now answers scripts with an empty HTTP 202: kloppy caches the
empty file and fails to parse it. So the Hugging Face mirror (pysport/idsse-data, the files kloppy
3.19 fetches) is named here and handed to sportec.load_tracking. kloppy also fills in what a frame
lacks instead of reporting it: no BallPossession becomes the away team, no BallStatus a dead ball,
no ball X/Y the centre spot. One pass over the raw ball frames finds any such frame first, and it
becomes X, or a missing ball, rather than an invented label.

    python jev_track/idsse.py --match J03WMX [--match J03WN1 ...] [--limit N]
    python jev_track/pipeline.py baseline --dataset idsse-J03WMX
    python evaluation/run_evaluation.py --gt runs/idsse-J03WMX/gt.csv \
        --results runs/idsse-J03WMX/results --out runs/idsse-J03WMX/score

--limit is kloppy's own: the first N frames of the match in period order, applied before the
layout, so a short test reaches the first half only. Each match is a 350-420 MB XML download that
kloppy caches in ~/kloppy_cache (or KLOPPY_CACHE_DIR) and holds in memory while it reads. kloppy
also keeps one Python object per frame and player, about 45 KB a frame as measured on a 10000-frame
slice, so a full match of ~140000 frames peaks near 7 GB. Prints aggregates only.

Data: IDSSE, owned by the Deutsche Fußball Liga (DFL), CC BY 4.0. Bassek, M., Rein, R., Weber, H.
& Memmert, D. An integrated dataset of spatiotemporal and event data in elite soccer. Scientific
Data 12, 195 (2025). https://doi.org/10.1038/s41597-025-04505-y
"""

import argparse
import gc
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import pipeline  # noqa: E402  (datasets and features; it also puts starter/ on the path)
import baseline  # noqa: E402  (write_submission)
import possession_metrics as pm  # noqa: E402  (possession_spells, for the printed summary)

try:
    from kloppy import sportec
    from kloppy.domain import BallState, Ground
    from kloppy.io import open_as_file
    from lxml import etree
except ImportError:
    raise ImportError(
        "kloppy is not installed. The rest of jev_track does without it, but this script needs "
        "it:\n    pip install kloppy==3.18.0")

# Params
# Match id -> DFL competition id: 000001 is the Bundesliga, 000002 the 2. Bundesliga.
MATCHES = {
    "J03WMX": "DFL-COM-000001",  # 1. FC Köln v FC Bayern München
    "J03WN1": "DFL-COM-000001",  # VfL Bochum 1848 v Bayer 04 Leverkusen
    "J03WOH": "DFL-COM-000002",  # Fortuna Düsseldorf v SSV Jahn Regensburg
    "J03WOY": "DFL-COM-000002",  # Fortuna Düsseldorf v F.C. Hansa Rostock
    "J03WPY": "DFL-COM-000002",  # Fortuna Düsseldorf v 1. FC Nürnberg
    "J03WQQ": "DFL-COM-000002",  # Fortuna Düsseldorf v FC St. Pauli
    "J03WR9": "DFL-COM-000002",  # Fortuna Düsseldorf v 1. FC Kaiserslautern
}
# Resolve the submission root even when this script runs from its model folder.
from pathlib import Path as _SourcePath
import sys as _source_sys
_source_sys.path.insert(0, str(_SourcePath(__file__).resolve().parents[3]))
from models.training_sources import IDSSE_MIRROR
MIRROR = IDSSE_MIRROR
FILES = {"meta": "DFL_02_01_matchinformation_{}_DFL-MAT-{}.xml",
         "tracking": "DFL_04_03_positions_raw_observed_{}_DFL-MAT-{}.xml"}
# GameSection -> period id, numbered as kloppy numbers them.
SECTIONS = {"firstHalf": 1, "secondHalf": 2, "firstHalfExtra": 3, "secondHalfExtra": 4}
# The only values kloppy reads correctly. Anything else, absent included, it turns into "away" or
# "dead" without a word, so those frames are left unlabelled instead.
BALL_FLAGS = {"BallStatus": ("0", "1"), "BallPossession": ("1", "2")}
SEPARATOR = pipeline.FPS     # X frames between two periods: one second
# The coordinate check. Normalised ball positions stay on the pitch plus a margin, and over a whole
# match their median sits near (0.5, 0.5). A wrong origin or scale misses both by far more than
# these allow. A --limit slice is held to the range only: the first 10000 frames of J03WOH put the
# median at x 0.89, because a long stoppage keeps the ball near one goal.
BALL_MEDIAN_TOLERANCE = 0.15
BALL_RANGE = (-0.05, 1.05)
BALL_OUTSIDE_MAX = 0.01      # share of ball samples allowed outside BALL_RANGE


# -- reading -----------------------------------------------------------------------------------

def url(match, kind):
    """Mirror URL of one IDSSE file; kind is "meta" or "tracking"."""
    return MIRROR + FILES[kind].format(MATCHES[match], match)


def fetch(match):
    """[meta, tracking] for one match as in-memory byte streams. kloppy downloads each file once
    and serves it from its cache after that. An empty file is refused here, because kloppy caches
    an empty answer as readily as a real one, and every later run would fail on it the same way."""
    streams = []
    for kind in ("meta", "tracking"):
        stream = open_as_file(url(match, kind))
        stream.seek(0, os.SEEK_END)
        if not stream.tell():
            raise IOError("{} came back empty. Delete kloppy's cached copy (~/kloppy_cache, or "
                          "KLOPPY_CACHE_DIR) and run again.".format(url(match, kind)))
        stream.seek(0)
        streams.append(stream)
    return streams


def unusable_ball_frames(raw):
    """{(period, frame number): [ball attributes kloppy would misread]}, read off the raw XML.

    kloppy does not report a missing or unexpected attribute, it fills one in: BallPossession other
    than 1 becomes the away team, BallStatus other than 1 a dead ball, a missing X or Y the centre
    spot. Reading the ball's own frames first lets those frames become X, or a missing ball,
    instead. Streams the file and frees each element as it goes, as kloppy's own reader does.
    """
    out, period = {}, None
    raw.seek(0)
    for event, elem in etree.iterparse(raw, events=("start", "end"), huge_tree=True):
        if event == "start" and elem.tag == "FrameSet":
            ball = elem.get("TeamId") == "BALL"
            period = SECTIONS.get(elem.get("GameSection")) if ball else None
        elif event == "end" and elem.tag == "Frame":
            number = elem.get("N")
            if period is not None and number is not None:
                bad = [name for name, allowed in BALL_FLAGS.items()
                       if elem.get(name) not in allowed]
                bad += [name for name in ("X", "Y") if not elem.get(name)]
                if bad:
                    out[(period, int(number))] = bad
            elem.clear()
        elif event == "end" and elem.tag == "FrameSet":
            elem.clear()
            while elem.getprevious() is not None:
                del elem.getparent()[0]
    raw.seek(0)
    return out


def load(meta, raw, limit):
    """kloppy's reading of the match. only_alive=False keeps the dead-ball frames, which D is made
    of. limit counts frames from the start of the match, so it applies before the layout below
    ever sees them. The coordinate system is named rather than left as None, which means "kloppy's
    global setting": the default of that setting is this same system, 0-1 along each axis from a
    corner. normalise() reads the extent the dataset reports, so any linear system converts."""
    return sportec.load_tracking(meta_data=meta, raw_data=raw, only_alive=False,
                                 limit=limit or None, coordinates="kloppy")


# -- arrays ------------------------------------------------------------------------------------

def layout(spans):
    """{period: (first_row, first_frame_number, n_rows)} and the total row count.

    spans is {period: (first, last)} frame numbers. Periods follow each other in order, SEPARATOR
    rows apart, and each keeps its own numbering inside its block. A frame the file lacks therefore
    keeps an empty row, and rows stay evenly spaced in time, which the speeds depend on.
    """
    blocks, row = {}, 0
    for period in sorted(spans):
        first, last = spans[period]
        if blocks:
            row += SEPARATOR
        blocks[period] = (row, first, last - first + 1)
        row += last - first + 1
    return blocks, row


def normalise(values, dims):
    """Provider units to 0-1, in place: x along the length, y across, from the pitch extent kloppy
    reports for whichever coordinate system it returned."""
    values[..., 0] -= dims.x_dim.min
    values[..., 0] /= dims.x_dim.max - dims.x_dim.min
    values[..., 1] -= dims.y_dim.min
    values[..., 1] /= dims.y_dim.max - dims.y_dim.min
    return values


def tracking_arrays(dataset, unusable):
    """Everything the dataset needs from a kloppy dataset, as numpy arrays.

    Player slots are the two squads, home first. A slot is one person and is NaN while that person
    is off the pitch, and slots nobody ever fills are dropped. alive is 1/0 and owner 0 (home) or
    1 (away), both -1 where the provider says nothing usable.
    """
    if not dataset.records:
        raise ValueError("kloppy read no frames")
    home, away = dataset.metadata.teams
    if home.ground != Ground.HOME:
        raise ValueError("kloppy lists {} first, but not as the home side".format(home.name))
    slot, team = {}, []
    for side, squad in enumerate((home, away)):
        for player in squad.players:
            slot[player.player_id] = len(team)
            team.append(side)

    spans = {}
    for frame in dataset.records:
        first, last = spans.get(frame.period.id, (frame.frame_id, frame.frame_id))
        spans[frame.period.id] = (min(first, frame.frame_id), max(last, frame.frame_id))
    blocks, n_rows = layout(spans)

    ball = np.full((n_rows, 2), np.nan, dtype=np.float32)
    players = np.full((n_rows, len(team), 2), np.nan, dtype=np.float32)
    alive = np.full(n_rows, -1, dtype=np.int8)
    owner = np.full(n_rows, -1, dtype=np.int8)
    framed = np.zeros(n_rows, dtype=bool)
    for frame in dataset.records:
        first_row, first_number, _ = blocks[frame.period.id]
        row = first_row + frame.frame_id - first_number
        bad = unusable.get((frame.period.id, frame.frame_id), ())
        framed[row] = True
        if frame.ball_coordinates is not None and "X" not in bad and "Y" not in bad:
            ball[row] = frame.ball_coordinates.x, frame.ball_coordinates.y
        if frame.ball_state is not None and "BallStatus" not in bad:
            alive[row] = 1 if frame.ball_state == BallState.ALIVE else 0
        if frame.ball_owning_team is not None and "BallPossession" not in bad:
            owner[row] = 0 if frame.ball_owning_team.team_id == home.team_id else 1
        for player, data in frame.players_data.items():
            index = slot.get(player.player_id)
            if index is not None and data.coordinates is not None:
                players[row, index, 0] = data.coordinates.x
                players[row, index, 1] = data.coordinates.y

    dims = dataset.metadata.pitch_dimensions
    keep = ~np.isnan(players[..., 0]).all(axis=0)
    return {"blocks": blocks, "n_read": len(dataset.records), "home": home.name, "away": away.name,
            "pitch": [float(dims.pitch_length), float(dims.pitch_width)],
            "ball": normalise(ball, dims), "players": normalise(players[:, keep], dims),
            "team": np.array(team, dtype=np.int8)[keep],
            "alive": alive, "owner": owner, "framed": framed}


def labels_of(alive, owner):
    """D where the ball is dead, H or A where it is alive and the provider names a team, and X
    everywhere else: between periods, on frames the file lacks, and alive with no team named."""
    labels = np.full(len(alive), "X", dtype="<U1")
    labels[alive == 0] = "D"
    labels[(alive == 1) & (owner == 0)] = "H"
    labels[(alive == 1) & (owner == 1)] = "A"
    return labels


def check_ball(ball, whole_match):
    """(median (x, y), share outside BALL_RANGE) of the normalised ball, raising if the conversion
    is wrong: a wrong origin or scale throws the ball off the pitch, or over a whole match the
    median far off the centre spot, while a right one leaves both well inside the limits."""
    xy = ball[~np.isnan(ball[:, 0])]
    if not len(xy):
        raise ValueError("no ball positions at all")
    median = np.median(xy, axis=0)
    outside = float(np.mean(np.any((xy < BALL_RANGE[0]) | (xy > BALL_RANGE[1]), axis=1)))
    off_centre = bool(np.any(np.abs(median - 0.5) > BALL_MEDIAN_TOLERANCE))
    if outside > BALL_OUTSIDE_MAX or (off_centre and whole_match):
        raise ValueError("coordinate check failed: ball median ({:.2f}, {:.2f}), {:.2f}% outside "
                         "{}".format(median[0], median[1], 100 * outside, BALL_RANGE))
    return median, outside


# -- building ----------------------------------------------------------------------------------

def build(match, limit):
    """Fetch one match, write runs/idsse-<match>/ and print what it holds."""
    times = [("start", time.time())]
    meta, raw = fetch(match)
    times.append(("fetch", time.time()))
    convert("idsse-" + match, meta, raw, limit, times)


def convert(name, meta, raw, limit, times):
    """The dataset folder from the two XML streams: meta.json and features.npz through pipeline,
    and gt.csv beside them."""
    unusable = unusable_ball_frames(raw)
    times.append(("raw ball check", time.time()))
    dataset = load(meta, raw, limit)
    times.append(("kloppy", time.time()))
    data = tracking_arrays(dataset, unusable)
    del dataset
    gc.collect()  # kloppy's frames point at each other, so only the collector frees them
    times.append(("arrays", time.time()))

    whole_match = not limit or data["n_read"] < limit
    ball_check = check_ball(data["ball"], whole_match)
    players, pitch = data["players"], data["pitch"]
    speed = pipeline.position_speed(players, pitch)
    features = pipeline.frame_features(data["ball"], players, speed, data["team"],
                                       ~np.isnan(players[..., 0]), pitch)
    labels = labels_of(data["alive"], data["owner"])
    meta_json = {"name": name, "n_frames": len(labels), "pitch": pitch,
                 "home": data["home"], "away": data["away"], "smart_tagging": False,
                 "halves": [[period, first, first + n - 1]
                            for period, (first, _, n) in sorted(data["blocks"].items())]}
    pipeline.write_dataset(name, meta_json, features)
    baseline.write_submission(os.path.join(pipeline.dataset_dir(name), "gt.csv"), labels)
    times.append(("features and files", time.time()))
    report(name, data, labels, features["shown_count"], ball_check, whole_match, limit, times)


# -- printing ----------------------------------------------------------------------------------

def percent(mask):
    """100 x the share of True in a boolean array, nan when the array is empty."""
    return 100.0 * mask.mean() if len(mask) else float("nan")


def peak_memory_gb():
    """Peak resident memory of this process so far, in GB, or nan where the platform cannot say."""
    try:
        import resource
    except ImportError:  # Windows
        return float("nan")
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / float(2 ** 30 if sys.platform == "darwin" else 2 ** 20)  # bytes vs KiB


def report(name, data, labels, shown_count, ball_check, whole_match, limit, times):
    """Aggregates only: nothing here prints a frame."""
    blocks, alive, owner, framed = data["blocks"], data["alive"], data["owner"], data["framed"]
    live = alive == 1
    in_halves = sum(n for _, _, n in blocks.values())
    print("{}: {} (H) v {} (A), pitch {:g} x {:g} m{}".format(
        name, data["home"], data["away"], data["pitch"][0], data["pitch"][1],
        ", first {} frames only".format(limit) if limit else ""))
    print("  frames: {}, {} in all with {}-frame separators; inside the halves {} missing from "
          "the file, {} with no ball status".format(
              ", ".join("half {} {}".format(period, n)
                        for period, (_, _, n) in sorted(blocks.items())),
              len(labels), SEPARATOR, in_halves - int(framed.sum()),
              int(np.sum(framed & (alive < 0)))))
    print("  alive on {:.1f}% of frames with a ball status; home share of alive frames {:.1f}%"
          .format(percent(alive[alive >= 0] == 1), percent(owner[live & (owner >= 0)] == 0)))
    print("  alive with no owning team: {} frames, labelled X; BallPossession set on {:.1f}% of "
          "dead frames".format(int(np.sum(live & (owner < 0))), percent(owner[alive == 0] >= 0)))

    codes = pm.label_codes(labels)
    spells = pm.possession_spells(codes, pm.valid_runs(codes))
    seconds = np.array([stop - start for start, stop, _ in spells], dtype=float) / pipeline.FPS
    print("  ground truth: {} possessions, mean {:.1f} s".format(
        len(spells), seconds.mean() if len(seconds) else float("nan")))

    median, outside = ball_check
    print("  ball coordinates on {:.1f}% of alive frames; ball median ({:.2f}, {:.2f}){}, {:.2f}% "
          "outside [{:g}, {:g}]; median {:.0f} players on the pitch".format(
              percent(~np.isnan(data["ball"][live, 0])), median[0], median[1],
              "" if whole_match else " (a slice, so not held to the centre)", 100 * outside,
              BALL_RANGE[0], BALL_RANGE[1], np.median(shown_count[framed])))
    steps = ", ".join("{} {:.0f} s".format(label, now - before)
                      for (_, before), (label, now) in zip(times, times[1:]))
    print("  time: {}; total {:.0f} s; peak memory so far {:.1f} GB".format(
        steps, times[-1][1] - times[0][1], peak_memory_gb()))
    print("  wrote {}".format(pipeline.dataset_dir(name)))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--match", action="append", required=True, choices=sorted(MATCHES),
                        help="IDSSE match id; repeat the flag for several matches")
    parser.add_argument("--limit", type=int, default=0,
                        help="read only the first N frames of each match, for quick tests")
    args = parser.parse_args()
    for match in dict.fromkeys(args.match):  # in the order given, each once
        build(match, args.limit)


if __name__ == "__main__":
    main()
