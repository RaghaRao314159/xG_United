"""Label possession with a System One decision model: TypeSafe's Jev, or an open model that serves
the same API (OpenJev, https://huggingface.co/openjev/openjev; decider, github.com/Mapika/decider).

The model never sees raw tracking. Code turns each second of play into named buckets -- how long
the ball has been untracked, how close each team's nearest player is, how fast the players move --
because TypeSafe's own guidance is that Jev "is not a calculator". The model answers two questions
per second (is the ball in play, which team has it), and code decodes those answers into frame
labels and moves each boundary onto the frame where the tracking shows the change.

A dataset is a folder under runs/ holding features.npz and meta.json (teams, halves, pitch), plus
gt.csv when labels exist. `features` builds the BAR-ATH one from the provided tracking;
jev_track/idsse.py builds them for the open IDSSE matches.

    python jev_track/pipeline.py features
    python jev_track/pipeline.py selftest --base-url http://127.0.0.1:3000 --model openjev
    python jev_track/pipeline.py ask      --dataset BAR-ATH --base-url http://127.0.0.1:3000 \
                                          --model openjev --name openjev
    python jev_track/pipeline.py decode   --dataset BAR-ATH --name openjev
    python jev_track/pipeline.py diagnose --dataset BAR-ATH --name openjev

For Jev itself: --base-url https://api.typesafe.ai --model jev-1.14.0 with TYPESAFE_API_KEY set,
and only on the IDSSE datasets. selftest sends an invented state with no match data in it.

Every constant below was fixed before anything was scored. None was chosen on the public window,
which the rules reserve for checking that the pipeline is correct. Scripts print aggregates only.
"""

import argparse
import json
import os
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402
import baseline  # noqa: E402  (team_of_each_track and write_submission)

# Params
RUNS = os.environ.get("MTAH_RUNS_DIR") or os.path.join(ROOT, "runs")
FPS = 25
WINDOW = FPS                 # one request per second of play
CONTEXT = 3 * FPS            # how far back "previous seconds" looks
IMAGE_W, IMAGE_H = 1920, 1080
LINE_MARGIN = 2.0            # metres from a line that counts as "near" it
SPEED_LAG = 2                # ball speed from positions this many frames either side
SWITCH_COST = 3.0            # nats the decoder charges for any change of state between seconds
SNAP = FPS                   # how far a boundary may move onto tracking evidence, in frames
RESTART_SPEED = 1.0          # m/s: the ball counts as played once it moves faster than this
PROB_FLOOR = 1e-4
H, A, D = 0, 1, 2
# Smart Tagging spells the away club "Atletic Club"; the model gets the real name.
TAG_TEAMS = {"barcelona": "Barcelona", "atletic club": "Athletic Club",
             "athletic club": "Athletic Club"}
ZONES = {0: "inside the pitch", 1: "near a touchline", 2: "near a goal line",
         3: "in a corner area", 4: "outside the lines"}

IN_PLAY = ("Is the ball in play during this second? Answer no while play is stopped: after the ball "
           "has gone out, or while a throw-in, corner, goal kick, free kick, kick-off, injury or "
           "substitution is pending and the restart has not happened yet.")
TEAM = ("Which team is in possession of the ball during this second? A team keeps possession until "
        "an opponent gains controlled possession; a touch, a deflection or a loose ball does not "
        "change it.")


def questions(home, away, context="v1", team_orders=2):
    """The team question is asked twice with the options in both orders, because TypeSafe documents
    that Jev leans toward whichever option is listed first. The two answers are averaged. With the
    v2 context the questions point at the target second of a timeline. team_orders=1 asks it once,
    for a model tuned against that bias (OpenJev reports 2.3% of answers change with option
    order) -- a third less work per request."""
    criteria = {team: "{} controls the ball, or was the last team to control it".format(team)
                for team in (home, away)}
    swapped = dict(reversed(list(criteria.items())))
    in_play, team = IN_PLAY, TEAM
    if context in ("v2", "v3"):
        in_play = in_play.replace("during this second", "during the target second")
        team = team.replace("during this second", "during the target second") + (
            " Use the timeline around the target second to judge who last had control.")
    out = {
        "in_play": {"type": "noul", "instructions": in_play},
        "team_home_first": {"type": "choice", "instructions": team, "criteria": criteria},
        "team_away_first": {"type": "choice", "instructions": team, "criteria": swapped},
    }
    if team_orders == 1:
        del out["team_away_first"]
    return out


# The v2 context: what the data is, the challenge's own rules (CHALLENGE.md), and a timeline. The
# labels are produced offline, so the timeline looks ahead as well as back.
ABOUT = ("Automatic tracking of one professional football match, summarised one second at a time. "
         "It is uncorrected: players carry no identities, and the ball is often not detected -- "
         "mostly while play is stopped (throw-ins, goal kicks, free kicks, substitutions, "
         "injuries), but sometimes during fast live play as well.")
RULES = {
    "dead_ball": "The ball is out of play from the moment the referee stops play (ball out, foul, "
                 "offside, goal, injury, substitution) until the restart is taken.",
    "possession": "While the ball is in play exactly one team has possession. It changes only when "
                  "a player of the other team gains controlled possession; a touch, a deflection, "
                  "a won aerial duel or a loose or contested ball does not change it.",
    "restarts": "Play always restarts with one team clearly in possession.",
}
TIMELINE = 5  # seconds either side of the target second


def summary_of_second(f, meta, start, stop):
    """A compact per-second line for the v2 timeline."""
    span = slice(start, stop)
    seen = f["seen"][span]
    fraction = float(seen.mean())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        speed = np.nanmedian(f["ball_speed"][span]) if seen.any() else np.nan
        nearest = np.nanmin([np.nanmedian(f["d_home"][span][seen]),
                             np.nanmedian(f["d_away"][span][seen])]) if seen.any() else np.nan
        crowd = np.nanmedian(f["crowd"][span])
    return {
        "ball": ("tracked" if fraction == 1.0 else "partly tracked" if fraction > 0
                 else "not tracked"),
        "ball_speed": bucket(speed, [0.5, 3, 12], ["still", "slow", "moving", "fast"]),
        "nearest_team": describe_team(f["nearest"][span], meta["home"], meta["away"]),
        "nearest_player_distance": bucket(nearest, [1, 2, 5, 10], ["under 1 m", "1 to 2 m",
                                                                   "2 to 5 m", "5 to 10 m",
                                                                   "over 10 m"]),
        "players": bucket(crowd, [0.7, 1.5, 3.0], ["mostly standing", "walking", "jogging",
                                                   "running"]),
    }


def state_v2(f, meta, tags, first, last, start, stop):
    """The v2 state: the data note, the rules, the target second in full and a +/-5 s timeline."""
    timeline = []
    for offset in range(-TIMELINE, TIMELINE + 1):
        a = start + offset * WINDOW
        b = min(a + WINDOW, last + 1)
        if a < first or a > last:
            continue
        entry = {"offset_seconds": offset}
        entry.update(summary_of_second(f, meta, a, b))
        timeline.append(entry)
    return {"about": ABOUT, "rules": RULES,
            "target_second": state_of(f, meta, tags, first, start, stop),
            "timeline": timeline}


# -- datasets ----------------------------------------------------------------------------------

def dataset_dir(name):
    return os.path.join(RUNS, name)


def load_dataset(name):
    """(meta, features) for a dataset folder under runs/."""
    folder = dataset_dir(name)
    with open(os.path.join(folder, "meta.json")) as handle:
        meta = json.load(handle)
    with np.load(os.path.join(folder, "features.npz")) as data:
        features = {key: data[key] for key in data.files}
    return meta, features


def write_dataset(name, meta, features):
    folder = dataset_dir(name)
    os.makedirs(folder, exist_ok=True)
    np.savez_compressed(os.path.join(folder, "features.npz"), **features)
    with open(os.path.join(folder, "meta.json"), "w") as handle:
        json.dump(meta, handle, indent=1)


# -- features ----------------------------------------------------------------------------------

def parse_atd(raw_path, n_frames):
    """(ball (n,2), players (n,slots,2), speed (n,slots)) in normalised 0-1 coordinates, NaN where the
    tracker has nothing. One streaming pass, the same reading as baseline.py."""
    ball = np.full((n_frames, 2), np.nan, dtype=np.float32)
    players = speed = None
    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\r\n").split(":")
            if len(parts) != 3:
                continue
            row = int(parts[0]) - 1
            groups = parts[1].split(";")
            if players is None:
                players = np.full((n_frames, len(groups), 2), np.nan, dtype=np.float32)
                speed = np.full((n_frames, len(groups)), np.nan, dtype=np.float32)
            for slot, group in enumerate(groups):
                if not group or group[0] == "N":
                    continue
                fields = group.split(",")
                players[row, slot] = (float(fields[0]), float(fields[1]))
                if len(fields) > 2 and fields[2] and fields[2][0] != "N":
                    speed[row, slot] = float(fields[2])
            if parts[2] and parts[2][0] != "N":
                fields = parts[2].split(",")
                ball[row] = (float(fields[0]), float(fields[1]))
    return ball, players, speed


def on_screen(players, matrices, valid):
    """True where a slot projects inside the tactical image. The tracker parks lost players off the
    image edge with stale speeds, and those would drag every crowd statistic toward zero."""
    x = players[..., 0].astype(np.float64)
    y = players[..., 1].astype(np.float64)
    m = matrices
    with np.errstate(invalid="ignore", divide="ignore"):
        den = m[:, 2, 0, None] * x + m[:, 2, 1, None] * y + m[:, 2, 2, None]
        px = (m[:, 0, 0, None] * x + m[:, 0, 1, None] * y + m[:, 0, 2, None]) / den
        py = (m[:, 1, 0, None] * x + m[:, 1, 1, None] * y + m[:, 1, 2, None]) / den
        inside = (px >= 0) & (px < IMAGE_W) & (py >= 0) & (py < IMAGE_H)
    return inside & valid[:, None]


def position_speed(players, pitch):
    """Player speed in m/s from normalised positions, for data without a speed channel (IDSSE).
    Scaled to metres first, because the two pitch axes are different lengths."""
    metres = players * np.array(pitch, dtype=np.float32)
    out = np.full(players.shape[:2], np.nan, dtype=np.float32)
    k = SPEED_LAG
    out[k:-k] = np.linalg.norm(metres[2 * k:] - metres[:-2 * k], axis=2) / (2.0 * k / FPS)
    return out


def frame_features(ball, players, speed, team, shown, pitch):
    """Per-frame numbers the state buckets are built from. Metres and m/s, NaN when unknown."""
    length, width = pitch
    scale = np.array([length, width], dtype=np.float32)
    b = ball * scale
    p = players * scale
    seen = ~np.isnan(b[:, 0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        dist = np.sqrt(((p - b[:, None, :]) ** 2).sum(axis=2))
        dist[~shown] = np.nan
        d_home = np.nanmin(np.where(team[None, :] == 0, dist, np.nan), axis=1)
        d_away = np.nanmin(np.where(team[None, :] == 1, dist, np.nan), axis=1)
        crowd = np.nanmedian(np.where(shown, speed, np.nan), axis=1)

    nearest = np.full(len(b), -1, dtype=np.int8)
    both = ~np.isnan(d_home) & ~np.isnan(d_away)
    nearest[both] = np.where(d_home[both] <= d_away[both], H, A)

    ball_speed = np.full(len(b), np.nan, dtype=np.float32)
    k = SPEED_LAG
    ball_speed[k:-k] = np.linalg.norm(b[2 * k:] - b[:-2 * k], axis=1) / (2.0 * k / FPS)

    gap = np.zeros(len(b), dtype=np.int32)  # frames since the ball was last tracked, 0 if tracked
    run = 0
    for i, tracked in enumerate(seen):
        run = 0 if tracked else run + 1
        gap[i] = run

    x, y = b[:, 0], b[:, 1]
    zone = np.full(len(b), -1, dtype=np.int8)
    with np.errstate(invalid="ignore"):
        near_touch = seen & ((y < LINE_MARGIN) | (y > width - LINE_MARGIN))
        near_goal = seen & ((x < LINE_MARGIN) | (x > length - LINE_MARGIN))
        outside = seen & ((x < 0) | (x > length) | (y < 0) | (y > width))
    zone[seen] = 0
    zone[near_touch] = 1
    zone[near_goal] = 2
    zone[near_touch & near_goal] = 3
    zone[outside] = 4

    return {"seen": seen, "d_home": d_home, "d_away": d_away, "nearest": nearest,
            "ball_speed": ball_speed, "gap": gap, "crowd": crowd, "zone": zone,
            "shown_count": shown.sum(axis=1).astype(np.int16)}


def cmd_features(args):
    """The BAR-ATH dataset, from the provided tactical tracking and homography."""
    import load_homography
    meta_path, raw_path = require(["atd_meta", "atd_raw"])
    n = source_frames("tactical")
    started = time.time()
    ball, players, speed = parse_atd(raw_path, n)
    team = baseline.team_of_each_track(meta_path)
    if players.shape[1] != len(team):
        raise ValueError("{} slots in the raw file, {} in the metadata".format(
            players.shape[1], len(team)))
    matrices, _, valid = load_homography.load()
    pitch = GAMES[GAME]["pitch"]
    features = frame_features(ball, players, speed, team, on_screen(players, matrices, valid),
                              pitch)
    meta = {"name": GAME, "n_frames": n, "pitch": list(pitch), "home": "Barcelona",
            "away": "Athletic Club", "smart_tagging": True,
            "halves": [list(block) for block in half_video_blocks(GAME)]}
    write_dataset(GAME, meta, features)

    graded = np.zeros(n, dtype=bool)
    for _, first, last in meta["halves"]:
        graded[first:last + 1] = True
    print("features: {} frames, {} slots, {:.0f} s".format(n, len(team), time.time() - started))
    print("  ball tracked on {:.1f}% of graded frames".format(100 * features["seen"][graded].mean()))
    print("  median on-screen slots per graded frame: {:.0f}".format(
        np.median(features["shown_count"][graded])))


# -- states ------------------------------------------------------------------------------------

def windows(meta):
    """[(half, first_frame_of_half, start, stop)] one-second windows covering every graded frame."""
    out = []
    for half, first, last in meta["halves"]:
        for start in range(first, last + 1, WINDOW):
            out.append((half, first, start, min(start + WINDOW, last + 1)))
    return out


def load_tags():
    """BAR-ATH Smart Tagging as (start_frame, end_frame, text) -- an automatic prior."""
    import load_smart_tagging
    out = []
    for event in load_smart_tagging.load():
        tags = event["tags"]
        detail = [tags[group] for group in ("Type",) if tags.get(group)]
        team = TAG_TEAMS.get((tags.get("Team") or "").strip().lower())
        if team:
            detail.append(team)
        text = (event["code"] or "").strip().lower()
        out.append((event["start_frame"], event["end_frame"],
                    text + (" ({})".format(", ".join(detail)) if detail else "")))
    return out


def bucket(value, edges, names):
    if value is None or np.isnan(value):
        return "unknown"
    return names[int(np.searchsorted(edges, value, side="right"))]


def describe_team(nearest, home, away):
    known = nearest[nearest >= 0]
    if not len(known):
        return "unknown, the ball is not tracked"
    share = float(np.mean(known == H))
    if share == 1.0:
        return "{} throughout".format(home)
    if share >= 0.6:
        return "mostly {}".format(home)
    if share > 0.4:
        return "both teams at times"
    if share > 0.0:
        return "mostly {}".format(away)
    return "{} throughout".format(away)


def state_of(f, meta, tags, first, start, stop):
    """The JSON state for one second of play. Every number is bucketed in code."""
    home, away = meta["home"], meta["away"]
    span = slice(start, stop)
    seen = f["seen"][span]
    fraction = float(seen.mean())
    tracked = ("throughout the second" if fraction == 1.0 else "for most of the second"
               if fraction >= 0.5 else "briefly" if fraction > 0 else "not at all")
    gap_s = f["gap"][stop - 1] / float(FPS)
    untracked = "tracked now" if gap_s == 0 else bucket(
        gap_s, [1, 3, 10], ["under 1 s", "1 to 3 s", "3 to 10 s", "over 10 s"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        speed = np.nanmedian(f["ball_speed"][span]) if seen.any() else np.nan
        d_home = np.nanmedian(f["d_home"][span][seen]) if seen.any() else np.nan
        d_away = np.nanmedian(f["d_away"][span][seen]) if seen.any() else np.nan
        crowd = np.nanmedian(f["crowd"][span])
    seen_idx = np.flatnonzero(seen)
    location = ZONES[int(f["zone"][start + seen_idx[-1]])] if len(seen_idx) else "unknown"

    def distance(metres):
        return bucket(metres, [1, 2, 5, 10], ["under 1 m", "1 to 2 m", "2 to 5 m", "5 to 10 m",
                                              "over 10 m"])

    state = {
        "ball": {
            "tracked": tracked,
            "untracked_for": untracked,
            "speed": bucket(speed, [0.5, 3, 12], ["still", "slow, under 3 m/s",
                                                  "moving, 3 to 12 m/s", "fast, over 12 m/s"]),
            "location": location,
        },
        "nearest_player_to_ball": {home: distance(d_home), away: distance(d_away)},
        "team_nearest_to_ball": {
            "this_second": describe_team(f["nearest"][span], home, away),
            "previous_3_seconds": describe_team(f["nearest"][max(first, start - CONTEXT):start],
                                                home, away),
        },
        "players": bucket(crowd, [0.7, 1.5, 3.0], ["mostly standing", "walking", "jogging",
                                                   "running"]),
    }
    if tags is not None:
        state["tactical_tags"] = sorted({text for a, b, text in tags
                                         if a < stop and b > start}) or "none"
    return state


# -- asking the model --------------------------------------------------------------------------

_local = threading.local()


def post(url, headers, body, retries=6, echo_errors=False):
    """POST with backoff on rate limits. Error bodies are printed only when asked, because a
    validation error can echo the request -- and in `ask` the request is built from match data."""
    import requests
    if not hasattr(_local, "session"):
        _local.session = requests.Session()
    for attempt in range(retries):
        try:
            response = _local.session.post(url, headers=headers, json=body, timeout=120)
        except requests.RequestException:
            time.sleep(2 ** attempt)
            continue
        if response.status_code in (429, 503, 529):
            time.sleep(2 ** attempt)
            continue
        if response.status_code >= 400:
            raise RuntimeError("HTTP {}{}".format(
                response.status_code, ": " + response.text[:500] if echo_errors else ""))
        return response.json()
    raise RuntimeError("gave up after {} attempts".format(retries))


def answers_of(response, names):
    """{question name: answer}, wherever the API puts them: at the top level or one level down."""
    if names <= set(response):
        return response
    for value in response.values():
        if isinstance(value, dict) and names <= set(value):
            return value
    raise KeyError("no answers for {} in a response with keys {}".format(
        sorted(names), sorted(response)))


def noul_value(answer):
    for key in ("noul", "probability", "p_true", "value"):
        if isinstance(answer.get(key), (int, float)):
            return float(answer[key])
    probs = answer.get("probabilities") or answer.get("probs") or {}
    for key in ("true", "yes", "True"):
        if key in probs:
            return float(probs[key])
    raise KeyError("no noul probability in {}".format(sorted(answer)))


def team_probability(answer, team):
    probs = answer.get("probabilities") or answer.get("probs")
    if not isinstance(probs, dict) or team not in probs:
        raise KeyError("no choice probabilities in {}".format(sorted(answer)))
    total = sum(float(v) for v in probs.values()) or 1.0
    return float(probs[team]) / total


def endpoint(base_url):
    """URL and headers for a System One server. Hosted Jev is reachable directly from TypeSafe
    (https://api.typesafe.ai, TYPESAFE_API_KEY) or through OpenRouter's pass-through
    (https://openrouter.ai/api, OPENROUTER_API_KEY, models such as typesafe/jev-1.14). A local
    server needs no key."""
    url = base_url.rstrip("/") + "/v1/systemone"
    headers = {"Content-Type": "application/json"}
    for host, variable in (("typesafe.ai", "TYPESAFE_API_KEY"),
                           ("openrouter.ai", "OPENROUTER_API_KEY")):
        if host in base_url:
            key = os.environ.get(variable)
            if not key:
                raise RuntimeError("{} is not set".format(variable))
            headers["Authorization"] = "Bearer " + key
    return url, headers


def cmd_selftest(args):
    """One request with an invented state, to see the server answer and the response layout."""
    home, away = "Home FC", "Away United"
    url, headers = endpoint(args.base_url)
    state = {"ball": {"tracked": "throughout the second", "untracked_for": "tracked now",
                      "speed": "slow, under 3 m/s", "location": "inside the pitch"},
             "nearest_player_to_ball": {home: "under 1 m", away: "5 to 10 m"},
             "team_nearest_to_ball": {"this_second": "{} throughout".format(home),
                                      "previous_3_seconds": "{} throughout".format(home)},
             "players": "jogging"}
    qs = questions(home, away)
    started = time.time()
    response = post(url, headers, {"model": args.model, "state": state, "questions": qs},
                    echo_errors=True)
    print("latency {:.2f} s".format(time.time() - started))
    print(json.dumps(response, indent=1)[:3000])
    answers = answers_of(response, set(qs))
    print("parsed: in_play {:.3f}  home (home first) {:.3f}  home (away first) {:.3f}".format(
        noul_value(answers["in_play"]), team_probability(answers["team_home_first"], home),
        team_probability(answers["team_away_first"], home)))


def cmd_ask(args):
    meta, f = load_dataset(args.dataset)
    tags = load_tags() if args.tags and meta.get("smart_tagging") else None
    lasts = {half: last for half, _, last in meta["halves"]}
    todo = windows(meta)
    if args.limit:
        todo = todo[:args.limit]
    out_path = os.path.join(dataset_dir(args.dataset), "answers_{}.jsonl".format(args.name))
    done = set()
    if os.path.exists(out_path):
        with open(out_path) as handle:
            done = {(row["start"], row.get("pass", 0))
                    for row in map(json.loads, filter(str.strip, handle))}
    uncertain = None
    if args.uncertain_band and os.path.exists(out_path):
        # Votes only where they can matter: passes after the first go to the seconds whose
        # answers so far sit within the band of 0.5, on either question.
        so_far = load_answers(args.dataset, args.name)
        uncertain = {start for start, (p_in, p_home) in so_far.items()
                     if abs(p_in - 0.5) < args.uncertain_band
                     or (p_in > 0.5 and abs(p_home - 0.5) < args.uncertain_band)}
        print("{}: {} of {} seconds are uncertain within +/-{}".format(
            args.dataset, len(uncertain), len(so_far), args.uncertain_band))
    todo = [(w, k) for w in todo for k in range(args.passes)
            if (w[2], k) not in done and (k == 0 or uncertain is None or w[2] in uncertain)]
    url, headers = endpoint(args.base_url)
    qs = questions(meta["home"], meta["away"], args.context, args.team_orders)
    if args.context == "v3":
        import context_v3
        extra = context_v3.load_extra(args.dataset)

    def one(task):
        (half, first, start, stop), k = task
        if args.context == "v3":
            state = context_v3.state_v3(f, extra, meta, first, lasts[half], start, stop,
                                        context_v3.RADII[k % len(context_v3.RADII)])
        elif args.context == "v2":
            state = state_v2(f, meta, tags, first, lasts[half], start, stop)
        else:
            state = state_of(f, meta, tags, first, start, stop)
        body = {"model": args.model, "state": state, "questions": qs}
        started = time.time()
        response = post(url, headers, body)
        answers = answers_of(response, set(qs))
        p_home_a = team_probability(answers["team_home_first"], meta["home"])
        return {"start": start, "pass": k, "p_in": noul_value(answers["in_play"]),
                "p_home_a": p_home_a,
                "p_home_b": (team_probability(answers["team_away_first"], meta["home"])
                             if "team_away_first" in answers else p_home_a),
                "model": response.get("model"), "latency": time.time() - started,
                "input_tokens": (response.get("usage") or {}).get("input_tokens")}

    started, failures, latencies, models, tokens = time.time(), 0, [], {}, 0
    with ThreadPoolExecutor(args.workers) as pool, open(out_path, "a") as out:
        futures = [pool.submit(one, task) for task in todo]
        for count, future in enumerate(as_completed(futures), start=1):
            try:
                row = future.result()
            except Exception as error:  # keep going; a missing second decodes as "no opinion"
                failures += 1
                if failures <= 3:
                    print("  request failed: {}".format(str(error)[:200]))
                continue
            out.write(json.dumps(row) + "\n")
            latencies.append(row["latency"])
            models[row["model"]] = models.get(row["model"], 0) + 1
            tokens += row["input_tokens"] or 0
            if count % 1000 == 0:
                print("  {} / {} answered".format(count, len(todo)), flush=True)
    print("{}: asked {} seconds in {:.0f} s, {} failed; models {}".format(
        args.dataset, len(todo), time.time() - started, failures, models))
    if latencies:
        print("  latency median {:.3f} s, p95 {:.3f} s; input tokens {}".format(
            float(np.median(latencies)), float(np.percentile(latencies, 95)), tokens))


# -- decoding ----------------------------------------------------------------------------------

VOTE_CLIP = 0.1  # a unanimous vote is strong evidence, not certainty, once it reaches the decoder


def load_answers(dataset, name, vote="mean"):
    """{start: (p_in, p_home)} with every pass of a second combined: `mean` averages the
    probabilities, `majority` turns each pass into a yes/no decision and uses the share of yes
    votes, clipped to [VOTE_CLIP, 1 - VOTE_CLIP]. With one pass, majority is that pass's decision."""
    passes = {}
    with open(os.path.join(dataset_dir(dataset), "answers_{}.jsonl".format(name))) as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                passes.setdefault(row["start"], []).append(
                    (row["p_in"], 0.5 * (row["p_home_a"] + row["p_home_b"])))
    out = {}
    for start, rows in passes.items():
        rows = np.array(rows)
        if vote == "majority":
            share = (rows > 0.5).mean(axis=0)
            out[start] = tuple(np.clip(share, VOTE_CLIP, 1 - VOTE_CLIP))
        else:
            out[start] = tuple(rows.mean(axis=0))
    return out


def viterbi(log_emission, cost):
    """Most likely H/A/D sequence when every change of state costs `cost` nats."""
    n, k = log_emission.shape
    step = -cost * (1.0 - np.eye(k))  # step[i, j]: from i to j
    score = log_emission[0].copy()
    back = np.zeros((n, k), dtype=np.int8)
    for t in range(1, n):
        candidates = score[:, None] + step
        back[t] = candidates.argmax(axis=0)
        score = candidates.max(axis=0) + log_emission[t]
    path = np.empty(n, dtype=np.int8)
    path[-1] = int(score.argmax())
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def snap(labels, f, first, last):
    """Move each boundary up to SNAP frames onto the frame where the tracking shows the change:
    the nearest player's team flipping for a hand-over, the ball moving again for a restart, and
    the ball vanishing for a stoppage. A boundary never crosses its neighbours."""
    out = labels.copy()
    edges = (np.flatnonzero(labels[1:] != labels[:-1]) + 1).tolist()
    for index, edge in enumerate(edges):
        before, after = labels[edge - 1], labels[edge]
        low = max(edges[index - 1] + 1 if index else 1, edge - SNAP)
        high = min(edges[index + 1] - 1 if index + 1 < len(edges) else len(labels) - 1,
                   edge + SNAP)
        frames = np.arange(low, high + 1)
        absolute = first + frames
        if before != D and after != D:
            team = f["nearest"][absolute]
            prior = f["nearest"][absolute - 1]
            hits = frames[(team == after) & (prior != after) & (prior >= 0)]
        elif before == D:
            moving = f["seen"][absolute] & (f["ball_speed"][absolute] > RESTART_SPEED)
            hits = frames[moving][:1]
        else:
            hits = frames[f["gap"][absolute] == 1]
        if not len(hits):
            continue
        target = int(hits[np.argmin(np.abs(hits - edge))])
        if target < edge:
            out[target:edge] = after
        elif target > edge:
            out[edge:target] = before
    return out


def cmd_decode(args):
    meta, f = load_dataset(args.dataset)
    answers = load_answers(args.dataset, args.name, args.vote)
    codes = np.full(meta["n_frames"], -1, dtype=np.int8)
    spans_all = windows(meta)
    missing = 0
    for half, first, last in meta["halves"]:
        spans = [w for w in spans_all if w[0] == half]
        emission = np.empty((len(spans), 3))
        for row, (_, _, start, _) in enumerate(spans):
            p_in, p_home = answers.get(start, (0.5, 0.5))
            missing += start not in answers
            emission[row] = (p_in * p_home, p_in * (1 - p_home), 1 - p_in)
        path = viterbi(np.log(np.clip(emission, PROB_FLOOR, 1.0)), SWITCH_COST)
        labels = np.repeat(path, [stop - start for _, _, start, stop in spans])
        codes[first:last + 1] = snap(labels, f, first, last)

    letters = np.array(["H", "A", "D"])
    out = np.where(codes >= 0, letters[np.clip(codes, 0, 2)], "X")
    folder = os.path.join(dataset_dir(args.dataset), "results")
    os.makedirs(folder, exist_ok=True)
    path = baseline.write_submission(
        os.path.join(folder, "{}.csv".format(args.out_name or args.name)), out)
    print("{}: decoded {} seconds ({} without an answer) -> {}".format(
        args.dataset, len(spans_all), missing, os.path.relpath(path, ROOT)))


def cmd_baseline(args):
    """The repository's nearest-player baseline (±50-frame vote) on any dataset, for reference."""
    meta, f = load_dataset(args.dataset)
    nearest = np.where(f["seen"], f["nearest"], -1).astype(np.int8)
    labels = baseline.smooth(np.where(nearest == H, baseline.HOME,
                                      np.where(nearest == A, baseline.AWAY, baseline.NOBODY)), 50)
    graded = np.zeros(meta["n_frames"], dtype=bool)
    for _, first, last in meta["halves"]:
        graded[first:last + 1] = True
    out = np.where(graded, labels, "X")
    folder = os.path.join(dataset_dir(args.dataset), "results")
    os.makedirs(folder, exist_ok=True)
    baseline.write_submission(os.path.join(folder, "baseline_smooth50.csv"), out)
    print("{}: wrote baseline_smooth50".format(args.dataset))


def cmd_compare(args):
    """How far two runs of the same questions disagree -- the evidence for or against the rule that
    a repository must regenerate the same file. Decisions are the 0.5 cut on each probability."""
    first = load_answers(args.dataset, args.name)
    second = load_answers(args.dataset, args.other)
    common = sorted(set(first) & set(second))
    a = np.array([first[s] for s in common])
    b = np.array([second[s] for s in common])
    diff = np.abs(a - b)
    print("{}: {} vs {} on {} common seconds".format(args.dataset, args.name, args.other,
                                                    len(common)))
    for column, label in ((0, "in play"), (1, "home has it")):
        flips = np.mean((a[:, column] > 0.5) != (b[:, column] > 0.5))
        print("  {:<12} decision flips {:.2%}   |dp| mean {:.4f}  max {:.4f}  identical {:.1%}".format(
            label, flips, diff[:, column].mean(), diff[:, column].max(),
            np.mean(diff[:, column] == 0)))


def cmd_diagnose(args):
    """Ground-truth-free checks, as aggregates."""
    sys.path.insert(0, os.path.join(ROOT, "evaluation"))
    import possession_metrics as pm
    meta, f = load_dataset(args.dataset)
    codes = pm.label_codes(pm.read_label_csv(os.path.join(
        dataset_dir(args.dataset), "results", "{}.csv".format(args.name))))
    runs = [(first, last + 1) for _, first, last in meta["halves"]]
    graded = np.zeros(len(codes), dtype=bool)
    for start, stop in runs:
        graded[start:stop] = True
    spells = pm.possession_spells(codes, runs)
    lengths = np.array([b - a for a, b, _ in spells]) / float(FPS)
    print("{} / {}: {} possessions, mean {:.1f} s, median {:.1f} s, {:.0f}% under 2 s".format(
        args.dataset, args.name, len(spells), lengths.mean(), np.median(lengths),
        100 * np.mean(lengths < 2)))
    print("  graded frames: D {:.1f}%  H {:.1f}%  A {:.1f}%".format(
        *(100 * np.mean(codes[graded] == k) for k in (pm.D, pm.H, pm.A))))
    long_gap = graded & (f["gap"] >= 3 * FPS)
    if long_gap.any():
        print("  frames 3+ s into a ball gap labelled D: {:.1f}%".format(
            100 * np.mean(codes[long_gap] == pm.D)))
    if args.dataset == GAME:
        prekick = codes[110808:113742]
        print("  2nd-half pre-kick-off frames labelled D: {:.1f}%".format(
            100 * np.mean(prekick == pm.D)))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("features")
    selftest = sub.add_parser("selftest")
    selftest.add_argument("--base-url", required=True)
    selftest.add_argument("--model", required=True)
    ask = sub.add_parser("ask")
    ask.add_argument("--dataset", required=True)
    ask.add_argument("--base-url", required=True)
    ask.add_argument("--model", required=True)
    ask.add_argument("--name", required=True)
    ask.add_argument("--workers", type=int, default=16)
    ask.add_argument("--limit", type=int, default=0)
    ask.add_argument("--tags", action="store_true",
                     help="add Smart Tagging to the state (BAR-ATH only; IDSSE has none)")
    ask.add_argument("--context", choices=("v1", "v2", "v3"), default="v1",
                     help="v2 adds the data note, the challenge's rules and a +/-5 s timeline; "
                          "v3 (BAR-ATH) adds broadcast, kick-off and gap cues "
                          "(jev_track/context_v3.py)")
    ask.add_argument("--passes", type=int, default=1, help="ask every second this many times")
    ask.add_argument("--uncertain-band", type=float, default=0.0,
                     help="passes after the first only for seconds within this of 0.5 so far")
    ask.add_argument("--team-orders", type=int, choices=(1, 2), default=2,
                     help="ask the team question in one or both option orders")
    for name in ("decode", "diagnose"):
        command = sub.add_parser(name)
        command.add_argument("--dataset", required=True)
        command.add_argument("--name", required=True)
    sub.choices["decode"].add_argument("--vote", choices=("mean", "majority"), default="mean")
    sub.choices["decode"].add_argument("--out-name", help="results file name (default: --name)")
    sub.add_parser("baseline").add_argument("--dataset", required=True)
    compare = sub.add_parser("compare")
    compare.add_argument("--dataset", required=True)
    compare.add_argument("--name", required=True)
    compare.add_argument("--other", required=True)
    args = parser.parse_args()
    {"features": cmd_features, "selftest": cmd_selftest, "ask": cmd_ask, "decode": cmd_decode,
     "diagnose": cmd_diagnose, "baseline": cmd_baseline, "compare": cmd_compare}[args.command](args)


if __name__ == "__main__":
    main()
