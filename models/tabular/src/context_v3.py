"""The v3 context for BAR-ATH: evidence the tactical tracking alone lacks, for the decision models.

Three cues, each aimed at a failure the v1 run showed on the checks that need no ground truth:

* The broadcast tracking, aligned to the tactical clock. v1 labelled every frame three seconds
  into a ball gap as dead, yet some gaps are fast live play the tactical tracker lost. The broadcast
  camera still sees the ball in some of those, and its homography is invalid while the TV shows a
  replay or close-up -- which happens when play is stopped.
* The kick-off setup: both teams wholly inside their own halves with the ball at the centre spot.
  The 2,934 graded frames before the second-half kick-off are all dead ball; v1 called most live.
* The whole of the current ball gap, seen offline: how long it lasts, where the ball was last seen
  and whether it reappears moving.

The broadcast is a separate recording with its own timeline. The offset is found per half by
matching the two ball tracks, never hard-coded, and printed with the residual distance as a check.

    python jev_track/context_v3.py                     # writes runs/BAR-ATH/extra.npz
    python jev_track/pipeline.py ask --dataset BAR-ATH --context v3 --passes 3 ...

With --passes, pass k uses a timeline of +/-RADII[k] seconds, so the votes are not copies of one
answer. Prints aggregates only.
"""

import os
import sys
import time
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pipeline as pl  # noqa: E402
from games import GAME, GAMES, require, source_frames  # noqa: E402

# Params
COARSE_STEP = 25        # frames between candidate offsets in the first search
COARSE_SAMPLE = 25      # every n-th tactical frame scored in the first search
FINE_RADIUS = 60        # frames either side of the coarse winner
FINE_SAMPLE = 5
MIN_OVERLAP = 400       # sampled frames where both cameras see the ball
CENTRE_RADIUS = 2.0     # metres from the centre spot
HALFWAY_SLACK = 1.0     # metres a player may stand over the halfway line at a kick-off
MIN_SETUP_PLAYERS = 14  # visible players needed before calling a kick-off setup
MOVING = 1.0            # m/s
RADII = (5, 3, 7)       # timeline half-width per pass

ABOUT_V3 = (pl.ABOUT + " A second camera, the TV broadcast, runs on the same clock: it sometimes "
            "sees the ball when the tactical tracking does not, and it cuts to replays and "
            "close-ups mostly while play is stopped.")
RULES_V3 = dict(pl.RULES, **{
    "kick_off": "Before a kick-off both teams stand inside their own halves with the ball at the "
                "centre spot; the ball is dead until the kick-off is taken.",
    "reading_the_cues": "A ball that stays untracked for many seconds while players walk usually "
                        "means play has stopped. If players keep running, or the TV still sees "
                        "the ball moving, play is probably live.",
})


# -- alignment ---------------------------------------------------------------------------------

def align(tactical, broadcast, first, last):
    """(offset, median_metres, overlap) with broadcast_frame = tactical_frame - offset, for one
    half. Ball tracks in metres; the offset minimises the median distance between them."""
    def score(offsets, step):
        frames = np.arange(first, last + 1, step)
        best = (np.inf, None, 0)
        for offset in offsets:
            j = frames - offset
            ok = (j >= 0) & (j < len(broadcast))
            if ok.sum() < MIN_OVERLAP:
                continue
            a, b = tactical[frames[ok]], broadcast[j[ok]]
            both = ~np.isnan(a[:, 0]) & ~np.isnan(b[:, 0])
            if both.sum() < MIN_OVERLAP:
                continue
            distance = float(np.median(np.linalg.norm(a[both] - b[both], axis=1)))
            if distance < best[0]:
                best = (distance, int(offset), int(both.sum()))
        return best

    coarse = score(range(first - len(broadcast), last + 1, COARSE_STEP), COARSE_SAMPLE)
    fine = score(range(coarse[1] - FINE_RADIUS, coarse[1] + FINE_RADIUS + 1), FINE_SAMPLE)
    return fine[1], fine[0], fine[2]


# -- cues --------------------------------------------------------------------------------------

def speed_of(track):
    out = np.full(len(track), np.nan, dtype=np.float32)
    k = pl.SPEED_LAG
    out[k:-k] = np.linalg.norm(track[2 * k:] - track[:-2 * k], axis=1) / (2.0 * k / pl.FPS)
    return out


def gap_cues(f):
    """Per frame inside a ball gap: its total length, how the ball reappears, where it was last
    seen. 0 / -1 outside gaps."""
    seen = f["seen"]
    n = len(seen)
    total = np.zeros(n, dtype=np.int32)
    reappear = np.full(n, -1, dtype=np.int8)
    last_zone = np.full(n, -1, dtype=np.int8)
    edges = np.flatnonzero(np.diff(np.concatenate(([1], seen.astype(np.int8), [1]))))
    for start, stop in zip(edges[::2], edges[1::2]):  # runs of unseen frames, half-open
        total[start:stop] = stop - start
        if stop + pl.SPEED_LAG < n:
            speed = f["ball_speed"][stop + pl.SPEED_LAG]
            reappear[start:stop] = -1 if np.isnan(speed) else int(speed > MOVING)
        if start > 0:
            last_zone[start:stop] = f["zone"][start - 1]
    return total, reappear, last_zone


def kickoff_setup(players_m, team, shown, ball_m, pitch):
    """True where both teams stand wholly in their own halves and the ball is at the centre spot."""
    length, width = pitch
    x = np.where(shown, players_m[..., 0], np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        home_min = np.nanmin(np.where(team == 0, x, np.nan), axis=1)
        home_max = np.nanmax(np.where(team == 0, x, np.nan), axis=1)
        away_min = np.nanmin(np.where(team == 1, x, np.nan), axis=1)
        away_max = np.nanmax(np.where(team == 1, x, np.nan), axis=1)
    half = length / 2.0
    split = (((home_max < half + HALFWAY_SLACK) & (away_min > half - HALFWAY_SLACK))
             | ((away_max < half + HALFWAY_SLACK) & (home_min > half - HALFWAY_SLACK)))
    enough = shown.sum(axis=1) >= MIN_SETUP_PLAYERS
    centre = np.linalg.norm(ball_m - np.array([half, width / 2.0]), axis=1) < CENTRE_RADIUS
    return split & enough, centre


def build(dataset=GAME):
    """Write runs/<dataset>/extra.npz. BAR-ATH only: it is the one dataset with a broadcast."""
    import baseline
    import load_homography
    meta, f = pl.load_dataset(dataset)
    pitch = np.array(meta["pitch"], dtype=np.float32)
    started = time.time()

    meta_t, raw_t = require(["atd_meta", "atd_raw"])
    ball_t, players_t, _ = pl.parse_atd(raw_t, source_frames("tactical"))
    team = baseline.team_of_each_track(meta_t)
    matrices, _, valid_t = load_homography.load()
    shown = pl.on_screen(players_t, matrices, valid_t)
    _, raw_b = require(["atd_meta", "atd_raw"], source="broadcast")
    ball_b, _, _ = pl.parse_atd(raw_b, source_frames("broadcast"))
    _, _, valid_b = load_homography.load(source="broadcast")
    parsed = time.time()

    ball_t_m, ball_b_m = ball_t * pitch, ball_b * pitch
    speed_b = speed_of(ball_b_m)
    n = len(ball_t)
    b_mapped = np.zeros(n, dtype=bool)
    b_seen = np.zeros(n, dtype=bool)
    b_cut = np.zeros(n, dtype=bool)
    b_speed = np.full(n, np.nan, dtype=np.float32)
    for half, first, last in meta["halves"]:
        offset, residual, overlap = align(ball_t_m, ball_b_m, first, last)
        frames = np.arange(first, last + 1)
        j = frames - offset
        ok = (j >= 0) & (j < len(ball_b))
        b_mapped[frames[ok]] = True
        b_seen[frames[ok]] = ~np.isnan(ball_b[j[ok], 0])
        b_cut[frames[ok]] = ~valid_b[j[ok]]
        b_speed[frames[ok]] = speed_b[j[ok]]
        print("half {}: broadcast = tactical - {}; median ball distance {:.2f} m over {} sampled "
              "frames; {:.1f}% of the half has a broadcast frame".format(
                  half, offset, residual, overlap, 100 * ok.mean()))

    setup, centre = kickoff_setup(players_t * pitch, team, shown, ball_t_m, meta["pitch"])
    total, reappear, last_zone = gap_cues(f)
    extra = {"b_mapped": b_mapped, "b_seen": b_seen, "b_cut": b_cut, "b_speed": b_speed,
             "kickoff": setup & (centre | ~f["seen"]), "gap_total": total,
             "gap_reappear": reappear, "gap_last_zone": last_zone}
    np.savez_compressed(os.path.join(pl.dataset_dir(dataset), "extra.npz"), **extra)

    graded = np.zeros(n, dtype=bool)
    for _, first, last in meta["halves"]:
        graded[first:last + 1] = True
    lost = graded & ~f["seen"]
    long_gap = graded & (f["gap"] >= 3 * pl.FPS)
    print("parsed in {:.0f} s, cues in {:.0f} s".format(parsed - started, time.time() - parsed))
    print("  tactical ball missing: broadcast sees it on {:.1f}%, TV cut away on {:.1f}%".format(
        100 * b_seen[lost].mean(), 100 * b_cut[lost].mean()))
    print("  3+ s into a gap: broadcast sees the ball on {:.1f}%, TV cut away on {:.1f}%".format(
        100 * b_seen[long_gap].mean(), 100 * b_cut[long_gap].mean()))
    print("  kick-off setup on {} graded frames ({} of them before the 2nd-half kick-off)".format(
        int(extra["kickoff"][graded].sum()), int(extra["kickoff"][110808:113742].sum())))


def load_extra(dataset):
    with np.load(os.path.join(pl.dataset_dir(dataset), "extra.npz")) as data:
        return {key: data[key] for key in data.files}


# -- the v3 state ------------------------------------------------------------------------------

def share_words(share, none="not at all"):
    return "throughout" if share == 1.0 else "partly" if share > 0 else none


def tv_summary(x, start, stop):
    span = slice(start, stop)
    mapped = x["b_mapped"][span]
    if not mapped.any():
        return "no TV picture for this second"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        speed = np.nanmedian(x["b_speed"][span][mapped])
    return {
        "ball_seen_by_tv": share_words(float(x["b_seen"][span][mapped].mean())),
        "tv_shows_replay_or_close_up": share_words(float(x["b_cut"][span][mapped].mean()),
                                                   none="no"),
        "ball_speed_on_tv": pl.bucket(speed, [0.5, 3, 12], ["still", "slow", "moving", "fast"]),
    }


def gap_summary(f, x, start, stop):
    missing = np.flatnonzero(~f["seen"][start:stop])
    if not len(missing):
        return None
    i = start + int(missing[0])
    return {
        "whole_gap_lasts": pl.bucket(x["gap_total"][i] / float(pl.FPS), [1, 3, 10, 30],
                                     ["under 1 s", "1 to 3 s", "3 to 10 s", "10 to 30 s",
                                      "over 30 s"]),
        "ball_last_seen": pl.ZONES.get(int(x["gap_last_zone"][i]), "unknown"),
        "ball_reappears": {1: "moving", 0: "still"}.get(int(x["gap_reappear"][i]),
                                                       "not before the half ends"),
    }


def line_of(f, x, meta, start, stop, offset):
    """One compact timeline line, to keep prompts short."""
    s = pl.summary_of_second(f, meta, start, stop)
    tv = tv_summary(x, start, stop)
    tv_text = tv if isinstance(tv, str) else "TV ball {}, replay/close-up {}".format(
        tv["ball_seen_by_tv"], tv["tv_shows_replay_or_close_up"])
    return "{:+d} s{}: ball {} ({}); nearest {} at {}; players {}; {}".format(
        offset, " (target)" if offset == 0 else "", s["ball"], s["ball_speed"],
        s["nearest_team"], s["nearest_player_distance"], s["players"], tv_text)


def state_v3(f, x, meta, first, last, start, stop, radius):
    target = pl.state_of(f, meta, None, first, start, stop)
    target["tv_broadcast"] = tv_summary(x, start, stop)
    gap = gap_summary(f, x, start, stop)
    if gap:
        target["ball_gap"] = gap
    if x["kickoff"][start:stop].mean() > 0.5:
        target["kick_off_setup"] = ("both teams stand inside their own halves and the ball is at "
                                    "the centre spot")
    timeline = []
    for offset in range(-radius, radius + 1):
        a = start + offset * pl.WINDOW
        if first <= a <= last:
            timeline.append(line_of(f, x, meta, a, min(a + pl.WINDOW, last + 1), offset))
    return {"about": ABOUT_V3, "rules": RULES_V3, "target_second": target, "timeline": timeline}


if __name__ == "__main__":
    build()
