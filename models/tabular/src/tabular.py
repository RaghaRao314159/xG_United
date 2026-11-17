"""Label possession with tabular classifiers trained on the open IDSSE matches.

pipeline.py asks a decision model two questions about every second of play. This module puts the
same two questions to ordinary tabular classifiers -- LightGBM and the tabular foundation models
TabPFN, TabICL and TabFM -- trained on the IDSSE matches, whose every frame is labelled the way
the challenge labels BAR-ATH (jev_track/idsse.py). Each one-second window of pipeline.windows()
becomes one row of numbers computed from the per-frame features pipeline.py already builds, a
classifier gives P(H), P(A), P(D) for the row, and the answers go through pipeline.py's own
decoder, so these models are decoded exactly like the decision models they are compared against:

    p_in = 1 - P(D)        p_home = P(H) / (P(H) + P(A))

A row holds the target second (share of frames with the ball tracked, seconds since the ball was
last seen at the end of the second, median ball speed, last ball zone, median distance from each
team's nearest player, share of frames with the home side nearest, median player speed) and the
same compact numbers for every second within CONTEXT either side. Labels are produced offline, so
the context looks ahead as well as back. Nothing names a team or a match, every model is trained
on both orientations of each row (home and away swapped, H and A swapped) and answers as the mean
over both, because a home/away prior learned from seven Bundesliga matches is match-specific.

THE DOMAIN GAP is the central problem. IDSSE tracks the ball on every frame; BAR-ATH's tracking
loses it on about 41% of graded frames, mostly in long blackouts that coincide with stoppages, but
not always. A model trained on clean IDSSE has never seen a missing ball. So training, and the
degraded evaluation, use IDSSE copies with the ball hidden in stretches whose lengths are drawn
from BAR-ATH's own gap lengths (aggregates of runs/BAR-ATH/features.npz: no frame positions, no
labels) until the hidden share of graded frames matches BAR-ATH's, with DEAD_SHARE of the hidden
time inside stoppages. Every feature that depends on the ball (seen, gap, zone, ball_speed,
d_home, d_away, nearest) is recomputed after hiding, exactly as pipeline.frame_features would.

THE ASSUMPTION, plainly: how BAR-ATH's blackouts split between dead and live play is unknown.
Nothing here estimates it from BAR-ATH labels, the public five minutes included, and nothing may:
that would be tuning on the public window. DEAD_SHARE = 0.85 is a prior, not a measurement: at
41% hidden it puts about 35% of graded frames hidden in stoppages (roughly three quarters of the
44-47% dead time LaLiga ball-in-play norms imply) and 6% hidden in live play, which squares with
players moving markedly slower inside blackouts than outside them. 0.7 and 0.95 are reported as
sensitivities rather than chosen between. Where a stretch goes is the second assumption: it
starts on a random frame of the pool (dead or live) that still shows the ball and runs for a drawn
BAR-ATH length, cut at the end of that stoppage or that stretch of live play, so a long blackout
ends as play restarts and the ball reappears moving, as it does on BAR-ATH 95% of the time.

A SECOND GAP, found after the runs and not yet closed: IDSSE freezes the ball while it is dead (its
speed is exactly 0 on 99.5% of dead frames and 0.1-0.5% of live ones), BAR-ATH's visible ball is
never exactly still. Clean IDSSE alive/dead is therefore nearly free (F1 0.994) and the models
learn a cue BAR-ATH never shows, which is why they label too little of BAR-ATH dead.

shown_count is computed but not used: on IDSSE it counts the players on the pitch (22 nearly
always), on BAR-ATH the players the tactical camera shows (about 20), so it can only carry an
IDSSE artefact across. Player speed is divided by the match's own median for the same reason --
IDSSE's comes from every player's positions, BAR-ATH's from the on-screen players' speed channel.

    python jev_track/tabular.py degrade --dead-share 0.85 0.7 0.95
    python jev_track/tabular.py lomo    --model lgbm --train d85 --test d85
    python jev_track/tabular.py score   --cond clean d85
    python jev_track/tabular.py predict --model lgbm --train d85 --name tab_lgbm_d85
    python jev_track/pipeline.py diagnose --dataset BAR-ATH --name tab_lgbm_d85

Model choice is by leave-one-match-out on IDSSE only. Prints aggregates only: nothing here prints
a row, a frame or a window of BAR-ATH, and no BAR-ATH-derived row (nor any degraded IDSSE row,
whose gaps are drawn from BAR-ATH) ever leaves the machine -- every model runs locally.
Data: IDSSE, DFL, CC BY 4.0 (Bassek et al., Scientific Data 12, 195, 2025).
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
CHALLENGE_EVALUATION = os.path.abspath(os.path.join(ROOT, "..", "..", "challenge", "evaluation"))
sys.path.insert(0, CHALLENGE_EVALUATION)
import pipeline  # noqa: E402  (datasets, windows, decoder; it also puts starter/ on the path)
import possession_metrics as pm  # noqa: E402  (label codes, possession spells)

# Params
MATCHES = ("J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9")
if os.environ.get("MTAH_MATCHES"):   # e.g. the SkillCorner set: lab/exp_sc/sc_build.py --list
    MATCHES = tuple(os.environ["MTAH_MATCHES"].split(","))
TARGET = pipeline.GAME
FPS = pipeline.FPS
WINDOW = pipeline.WINDOW
H, A, D, X = pm.H, pm.A, pm.D, pm.X
CONTEXT = 5                  # seconds either side of the target second
TARGET_STATS = ("seen", "gap", "speed", "zone", "d_home", "d_away", "home", "crowd", "shown")
CONTEXT_STATS = ("seen", "gap", "speed", "zone", "dmin", "home", "crowd")
DROP = ("shown",)            # measured differently on the two sources; see the docstring
GAP_CAP = 60.0               # seconds; longer gaps all read as "long"
SPEED_CAP = 40.0             # m/s; faster is a tracking jump
DIST_CAP = 60.0              # metres
FILL = -1.0                  # every unknown number, for every model alike
DEAD_SHARE = 0.85            # share of hidden ball time inside stoppages: a prior, see above
DEAD_SHARES = (0.85, 0.7, 0.95)
SEED = 0
SEPARATOR = FPS              # X frames between matches when they are scored as one
# Training rows each model is given at most (both orientations counted), inside its documented
# limits: TabPFN v2 10,000; TabICLv2 pretrained on up to ~48K; TabFM passes every row as context.
MAX_ROWS = {"lgbm": None, "tabpfn": 10000, "tabicl": 20000, "tabfm": 10000}
LGBM = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=40,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
            deterministic=True, force_row_wise=True, n_jobs=8, verbose=-1)
PROB_EPS = 1e-9


# -- datasets ----------------------------------------------------------------------------------

def dataset_name(match, cond):
    """clean -> idsse-<match>; d85 -> idsse-<match>-d85, the copy with the ball hidden. SkillCorner
    matches are named SC<id> and live in sc-<id>[-<cond>] (lab/exp_sc/sc_build.py)."""
    stem = "sc-" + match[2:] if match.startswith("SC") else "idsse-" + match
    return stem + ("" if cond == "clean" else "-" + cond)


def cond_of(dead_share):
    return "d{:02d}".format(int(round(100 * dead_share)))


def built_matches():
    """The IDSSE matches whose clean dataset exists under runs/."""
    return [m for m in MATCHES
            if os.path.exists(os.path.join(pipeline.dataset_dir(dataset_name(m, "clean")),
                                           "features.npz"))]


def gt_codes(dataset):
    path = os.path.join(pipeline.dataset_dir(dataset), "gt.csv")
    return pm.label_codes(pm.read_label_csv(path))


def graded_mask(meta, n):
    out = np.zeros(n, dtype=bool)
    for _, first, last in meta["halves"]:
        out[first:last + 1] = True
    return out


def run_bounds(mask):
    """(starts, stops) of the runs of True in a boolean array, stops exclusive."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


# -- one row per second ------------------------------------------------------------------------

def by_second(values, meta, fill):
    """values over the graded frames as (n_windows, WINDOW), padded with fill, in the order of
    pipeline.windows(meta), plus the mask of real (unpadded) frames."""
    chunks, real = [], []
    for _, first, last in meta["halves"]:
        span = np.asarray(values[first:last + 1], dtype=np.float64)
        width = -(-len(span) // WINDOW) * WINDOW
        padded = np.full(width, fill, dtype=np.float64)
        padded[:len(span)] = span
        ok = np.zeros(width, dtype=bool)
        ok[:len(span)] = True
        chunks.append(padded.reshape(-1, WINDOW))
        real.append(ok.reshape(-1, WINDOW))
    return np.vstack(chunks), np.vstack(real)


def nanmedian_rows(values):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmedian(values, axis=1)


def second_stats(f, meta, crowd_scale):
    """{stat: array over windows}: the target-second numbers, the same ones pipeline.state_of
    buckets for the decision model, kept as numbers."""
    seen, real = by_second(f["seen"], meta, 0)
    seen = (seen > 0) & real
    ends = np.array([stop - 1 for _, _, _, stop in pipeline.windows(meta)])
    out = {"seen": seen.sum(axis=1) / real.sum(axis=1),
           "gap": np.minimum(f["gap"][ends] / float(FPS), GAP_CAP)}
    speed, _ = by_second(f["ball_speed"], meta, np.nan)
    out["speed"] = np.minimum(nanmedian_rows(speed), SPEED_CAP)
    zone, _ = by_second(f["zone"], meta, -1)
    last = WINDOW - 1 - np.argmax(seen[:, ::-1], axis=1)
    out["zone"] = np.where(seen.any(axis=1), zone[np.arange(len(zone)), last], np.nan)
    for key in ("d_home", "d_away"):
        dist, _ = by_second(f[key], meta, np.nan)
        out[key] = np.minimum(nanmedian_rows(np.where(seen, dist, np.nan)), DIST_CAP)
    out["dmin"] = np.fmin(out["d_home"], out["d_away"])
    near, _ = by_second(f["nearest"], meta, -1)
    known = (near >= 0).sum(axis=1)
    out["home"] = np.where(known > 0, (near == H).sum(axis=1) / np.maximum(known, 1), np.nan)
    crowd, _ = by_second(f["crowd"], meta, np.nan)
    out["crowd"] = nanmedian_rows(crowd) / crowd_scale
    shown, _ = by_second(f["shown_count"], meta, np.nan)
    out["shown"] = nanmedian_rows(shown)
    return out


def half_of_windows(meta):
    return np.array([half for half, _, _, _ in pipeline.windows(meta)])


def shifted(values, half, offset):
    """values moved by offset windows, NaN where that leaves the half."""
    out = np.full(len(values), np.nan)
    index = np.arange(len(values)) + offset
    ok = (index >= 0) & (index < len(values))
    ok[ok] = half[index[ok]] == half[ok]
    out[ok] = values[index[ok]]
    return out


def feature_names():
    names = [s for s in TARGET_STATS if s not in DROP]
    for offset in range(-CONTEXT, CONTEXT + 1):
        if offset:
            names += ["{}{:+d}".format(s, offset) for s in CONTEXT_STATS if s not in DROP]
    return names


def crowd_scale_of(f, meta):
    """The match's own median player speed over graded frames: speeds are compared relative to
    it, because the two sources measure them differently."""
    graded = graded_mask(meta, len(f["crowd"]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return float(np.nanmedian(f["crowd"][graded]))


def rows(f, meta):
    """(n_windows, n_features) float matrix, NaN where unknown, columns as feature_names()."""
    stats = second_stats(f, meta, crowd_scale_of(f, meta))
    half = half_of_windows(meta)
    columns = [stats[s] for s in TARGET_STATS if s not in DROP]
    for offset in range(-CONTEXT, CONTEXT + 1):
        if offset:
            columns += [shifted(stats[s], half, offset) for s in CONTEXT_STATS if s not in DROP]
    return np.column_stack(columns).astype(np.float32)


def window_labels(codes, meta):
    """Majority ground-truth label (H, A or D) of each window's frames, -1 where none is graded."""
    labels, _ = by_second(codes, meta, X)
    counts = np.stack([(labels == k).sum(axis=1) for k in (H, A, D)], axis=1)
    out = counts.argmax(axis=1)
    out[counts.sum(axis=1) == 0] = -1
    return out


def stat_of(name):
    """'home+3' -> 'home', 'd_home' -> 'd_home'."""
    for sign in "+-":
        if sign in name:
            return name.split(sign)[0]
    return name


def swap(matrix, names):
    """The same rows with home and away exchanged."""
    out = matrix.copy()
    for i, name in enumerate(names):
        if stat_of(name) == "home":
            out[:, i] = 1.0 - matrix[:, i]
    a, b = names.index("d_home"), names.index("d_away")
    out[:, a], out[:, b] = matrix[:, b], matrix[:, a]
    return out


def swap_labels(labels):
    out = labels.copy()
    out[labels == H] = A
    out[labels == A] = H
    return out


# -- ball dropout ------------------------------------------------------------------------------

def target_gaps():
    """(gap lengths in frames, hidden share of graded frames) of BAR-ATH's tracking: every run of
    frames without the ball inside the graded halves. Positions and labels are not read."""
    meta, f = pipeline.load_dataset(TARGET)
    lengths, missing, graded = [], 0, 0
    for _, first, last in meta["halves"]:
        lost = ~f["seen"][first:last + 1]
        starts, stops = run_bounds(lost)
        lengths.extend((stops - starts).tolist())
        missing += int(lost.sum())
        graded += len(lost)
    return np.array(lengths), missing / float(graded)


def describe_gaps(lengths, label):
    """One aggregate line about a set of gap lengths."""
    seconds = np.asarray(lengths, dtype=float) / FPS
    total = seconds.sum() or 1.0
    print("  {}: {} gaps, median {:.1f} s, p90 {:.0f} s; time in gaps > 3 s {:.1f}%, > 10 s "
          "{:.1f}%".format(label, len(seconds), np.median(seconds) if len(seconds) else 0,
                           np.percentile(seconds, 90) if len(seconds) else 0,
                           100 * seconds[seconds > 3].sum() / total,
                           100 * seconds[seconds > 10].sum() / total))


def place(hidden, pool, budget, lengths, rng):
    """Hide the ball in stretches until `budget` frames of `pool` are hidden. Each stretch starts
    on a random pool frame that still shows the ball, lasts a length drawn from BAR-ATH's gaps and
    stops at the end of its run of pool frames. Returns the frames hidden."""
    starts, stops = run_bounds(pool)
    run_end = np.zeros(len(pool), dtype=np.int64)
    run_end[pool] = np.repeat(stops, stops - starts)
    done = 0
    while done < budget:
        free = np.flatnonzero(pool & ~hidden)
        if not len(free):
            break
        start = int(free[rng.integers(len(free))])
        length = max(1, min(int(rng.choice(lengths)), int(round(budget - done))))
        stop = min(start + length, int(run_end[start]))
        done += int((~hidden[start:stop]).sum())
        hidden[start:stop] = True
    return done


def dropout_mask(codes, meta, lengths, share, dead_share, rng):
    """Frames whose ball is hidden: share of the graded frames, dead_share of them dead."""
    graded = graded_mask(meta, len(codes)) & (codes != X)
    budget = share * graded.sum()
    hidden = np.zeros(len(codes), dtype=bool)
    place(hidden, graded & (codes == D), dead_share * budget, lengths, rng)
    place(hidden, graded & ((codes == H) | (codes == A)), (1 - dead_share) * budget, lengths, rng)
    return hidden


def gap_of(seen):
    """Frames since the ball was last tracked, 0 where tracked: pipeline.frame_features' count."""
    index = np.arange(len(seen))
    last = np.maximum.accumulate(np.where(seen, index, -1))
    return (index - last).astype(np.int32)


def hide_ball(f, hidden):
    """The per-frame features with the ball removed on `hidden` frames, recomputed as
    pipeline.frame_features would compute them from a ball track missing there."""
    out = dict(f)
    out["seen"] = f["seen"] & ~hidden
    out["d_home"] = np.where(hidden, np.nan, f["d_home"]).astype(f["d_home"].dtype)
    out["d_away"] = np.where(hidden, np.nan, f["d_away"]).astype(f["d_away"].dtype)
    out["nearest"] = np.where(hidden, -1, f["nearest"]).astype(f["nearest"].dtype)
    out["zone"] = np.where(hidden, -1, f["zone"]).astype(f["zone"].dtype)
    k = pipeline.SPEED_LAG  # ball_speed[i] is read from the ball at i - k and i + k
    unknown = np.zeros(len(hidden), dtype=bool)
    unknown[k:-k] = hidden[:-2 * k] | hidden[2 * k:]
    out["ball_speed"] = np.where(unknown, np.nan, f["ball_speed"]).astype(f["ball_speed"].dtype)
    out["gap"] = gap_of(out["seen"])
    return out


def cmd_degrade(args):
    """Write runs/idsse-<match>-dNN/ for every built match and every dead share."""
    lengths, share = target_gaps()
    print("{}: ball missing on {:.1f}% of graded frames".format(TARGET, 100 * share))
    describe_gaps(lengths, TARGET)
    for match in built_matches():
        meta, f = pipeline.load_dataset(dataset_name(match, "clean"))
        codes = gt_codes(dataset_name(match, "clean"))
        graded = graded_mask(meta, len(codes)) & (codes != X)
        dead = graded & (codes == D)
        print("{}: D on {:.1f}% of graded frames, ball tracked on {:.1f}%".format(
            match, 100 * dead.sum() / graded.sum(), 100 * f["seen"][graded].mean()))
        for index, dead_share in enumerate(args.dead_share):
            rng = np.random.default_rng([SEED, MATCHES.index(match), index,
                                         int(round(1000 * dead_share))])
            hidden = dropout_mask(codes, meta, lengths, share, dead_share, rng)
            g = hide_ball(f, hidden)
            name = dataset_name(match, cond_of(dead_share))
            pipeline.write_dataset(name, dict(meta, name=name), g)
            link = os.path.join(pipeline.dataset_dir(name), "gt.csv")
            if not os.path.lexists(link):
                os.symlink(os.path.join("..", dataset_name(match, "clean"), "gt.csv"), link)
            lost = graded & ~g["seen"]
            print("  {}: hidden {:.1f}% of graded frames, {:.1f}% of it dead; {:.1f}% of D and "
                  "{:.1f}% of live frames hidden".format(
                      cond_of(dead_share), 100 * lost.mean() / graded.mean(),
                      100 * (lost & dead).sum() / max(1, lost.sum()),
                      100 * (lost & dead).sum() / dead.sum(),
                      100 * (lost & ~dead).sum() / max(1, (graded & ~dead).sum())))
            gaps = []
            for _, first, last in meta["halves"]:
                starts, stops = run_bounds(~g["seen"][first:last + 1])
                gaps.extend((stops - starts).tolist())
            describe_gaps(gaps, "    " + name)


# -- models ------------------------------------------------------------------------------------

def make_model(model, seed):
    """An sklearn-style classifier. Imports are local: each foundation model lives in its own
    environment, and all of them run on this machine, never through a hosted API."""
    if model == "lgbm":
        import lightgbm
        return lightgbm.LGBMClassifier(random_state=seed, **LGBM)
    if model == "tabpfn":
        from tabpfn import TabPFNClassifier
        from tabpfn.constants import ModelVersion
        return TabPFNClassifier.create_default_for_version(
            ModelVersion.V2, device="cuda", random_state=seed)
    if model == "tabicl":
        from tabicl import TabICLClassifier
        return TabICLClassifier(device="cuda", random_state=seed)
    if model == "tabfm":  # load() would put it on the CPU; 8 members, not the default 32
        from tabfm import TabFMClassifier
        from tabfm import tabfm_v1_0_0_pytorch as tabfm_v1_0_0
        return TabFMClassifier(model=tabfm_v1_0_0.load(model_type="classification",
                                                       device="cuda"),
                               n_estimators=8, random_state=seed)
    raise ValueError("unknown model {}".format(model))


def fit_predict(model, train_x, train_y, test_x, seed):
    """P(H), P(A), P(D) for every test row."""
    clf = make_model(model, seed)
    clf.fit(np.nan_to_num(train_x, nan=FILL), train_y)
    proba = clf.predict_proba(np.nan_to_num(test_x, nan=FILL))
    out = np.zeros((len(test_x), 3))
    out[:, np.asarray(clf.classes_, dtype=int)] = proba
    return out


def symmetric_proba(model, train_x, train_y, test_x, names, seed):
    """Train on both orientations of every row, answer with the mean over both orientations of
    every test row, so neither side is favoured for being listed first."""
    keep = train_y >= 0
    both_x = np.vstack([train_x[keep], swap(train_x[keep], names)])
    both_y = np.concatenate([train_y[keep], swap_labels(train_y[keep])])
    cap = MAX_ROWS.get(model)
    if cap and len(both_y) > cap:
        pick = np.sort(np.random.default_rng(seed).choice(len(both_y), cap, replace=False))
        both_x, both_y = both_x[pick], both_y[pick]
    n = len(test_x)
    proba = fit_predict(model, both_x, both_y, np.vstack([test_x, swap(test_x, names)]), seed)
    return 0.5 * (proba[:n] + proba[n:][:, [A, H, D]]), len(both_y)


# -- answers and decoding ----------------------------------------------------------------------

def write_answers(dataset, name, meta, proba):
    """answers_<name>.jsonl in the layout pipeline.load_answers reads, one line per window."""
    path = os.path.join(pipeline.dataset_dir(dataset), "answers_{}.jsonl".format(name))
    with open(path, "w") as handle:
        for (_, _, start, _), (p_h, p_a, p_d) in zip(pipeline.windows(meta), proba):
            alive = p_h + p_a
            p_home = p_h / alive if alive > PROB_EPS else 0.5
            handle.write(json.dumps({"start": int(start), "pass": 0,
                                     "p_in": float(1.0 - p_d), "p_home_a": float(p_home),
                                     "p_home_b": float(p_home)}) + "\n")


def decode(dataset, name):
    pipeline.cmd_decode(argparse.Namespace(dataset=dataset, name=name, vote="mean",
                                           out_name=None))


def table_of(match, cond, cache):
    """(rows, labels, meta) of one match in one condition, built once per process."""
    key = (match, cond)
    if key not in cache:
        dataset = dataset_name(match, cond)
        meta, f = pipeline.load_dataset(dataset)
        cache[key] = (rows(f, meta), window_labels(gt_codes(dataset), meta), meta)
    return cache[key]


def run_name(model, train, test, suffix=""):
    """tab_<model> when trained and tested alike, tab_<model>_t<train> otherwise."""
    return "tab_{}{}{}".format(model, "" if train == test else "_t" + train, suffix)


def cmd_lomo(args):
    """Leave one IDSSE match out: train on the rest, answer and decode the one left out."""
    matches = built_matches()
    names, cache = feature_names(), {}
    for held in matches:
        started = time.time()
        train = [m for m in matches if m != held]
        train_x = np.vstack([table_of(m, args.train, cache)[0] for m in train])
        train_y = np.concatenate([table_of(m, args.train, cache)[1] for m in train])
        test_x, test_y, meta = table_of(held, args.test, cache)
        proba, used = symmetric_proba(args.model, train_x, train_y, test_x, names, SEED)
        name = run_name(args.model, args.train, args.test, args.suffix)
        dataset = dataset_name(held, args.test)
        write_answers(dataset, name, meta, proba)
        decode(dataset, name)
        graded = test_y >= 0
        print("  {} held out: {} training rows, window accuracy {:.3f}, P(D) mean {:.3f} vs D "
              "share {:.3f}, {:.0f} s".format(
                  held, used, np.mean(proba[graded].argmax(axis=1) == test_y[graded]),
                  proba[graded, D].mean(), np.mean(test_y[graded] == D),
                  time.time() - started), flush=True)


def cmd_predict(args):
    """Train on every built IDSSE match in one condition and label BAR-ATH."""
    started = time.time()
    names, cache = feature_names(), {}
    matches = built_matches()
    train_x = np.vstack([table_of(m, args.train, cache)[0] for m in matches])
    train_y = np.concatenate([table_of(m, args.train, cache)[1] for m in matches])
    meta, f = pipeline.load_dataset(TARGET)
    test_x = rows(f, meta)
    proba, used = symmetric_proba(args.model, train_x, train_y, test_x, names, SEED)
    write_answers(TARGET, args.name, meta, proba)
    decode(TARGET, args.name)
    print("{} / {}: trained on {} rows of {} matches ({}); {} seconds answered, mean P(D) "
          "{:.3f}, P(D) > 0.5 on {:.1f}% of seconds; {:.0f} s".format(
              TARGET, args.name, used, len(matches), args.train, len(test_x),
              proba[:, D].mean(), 100 * np.mean(proba[:, D] > 0.5), time.time() - started))


# -- scoring -----------------------------------------------------------------------------------

def scoreboard(dataset_dir_or_path):
    path = os.path.join(dataset_dir_or_path, "score", "scoreboard.csv")
    with open(path) as handle:
        return {row["submission"]: row for row in csv.DictReader(handle)}


def official_score(gt, results, out, bootstrap):
    """The repository's scorer, exactly as a participant runs it."""
    subprocess.run([sys.executable, os.path.join(CHALLENGE_EVALUATION, "run_evaluation.py"),
                    "--gt", gt, "--results", results, "--out", out,
                    "--bootstrap", str(bootstrap), "--no-detail"],
                   check=False, stdout=subprocess.DEVNULL)


def pooled(cond, matches, bootstrap):
    """Every held-out match of one condition scored as one long match (X frames between them),
    so each model gets one spellF1 with a bootstrap interval over all the possessions."""
    folder = pipeline.dataset_dir("tab-pooled-" + cond)
    os.makedirs(os.path.join(folder, "results"), exist_ok=True)
    sep = np.full(SEPARATOR, "X")
    gts, names = [], None
    for match in matches:
        dataset = dataset_name(match, cond)
        gts.append(pm.read_label_csv(os.path.join(pipeline.dataset_dir(dataset), "gt.csv"))
                   ["label"].values)
        here = {n[:-4] for n in os.listdir(os.path.join(pipeline.dataset_dir(dataset),
                                                        "results")) if n.endswith(".csv")}
        names = here if names is None else names & here
    pipeline.baseline.write_submission(os.path.join(folder, "gt.csv"), np.concatenate(
        [part for g in gts for part in (g, sep)][:-1]))
    for name in sorted(names):
        parts = []
        for match in matches:
            path = os.path.join(pipeline.dataset_dir(dataset_name(match, cond)), "results",
                                name + ".csv")
            parts += [pm.read_label_csv(path)["label"].values, sep]
        pipeline.baseline.write_submission(os.path.join(folder, "results", name + ".csv"),
                                           np.concatenate(parts[:-1]))
    official_score(os.path.join(folder, "gt.csv"), os.path.join(folder, "results"),
                   os.path.join(folder, "score"), bootstrap)
    return folder


def cmd_score(args):
    """The official scorer on every held-out match, then on all of them pooled; prints the table."""
    matches = built_matches()
    for cond in args.cond:
        for match in matches:
            dataset = dataset_name(match, cond)
            folder = pipeline.dataset_dir(dataset)
            if not os.path.exists(os.path.join(folder, "results", "baseline_smooth50.csv")):
                pipeline.cmd_baseline(argparse.Namespace(dataset=dataset))
            official_score(os.path.join(folder, "gt.csv"), os.path.join(folder, "results"),
                           os.path.join(folder, "score"), args.bootstrap)
        pooled(cond, matches, args.bootstrap)
        print_table(cond, matches)


def print_table(cond, matches):
    """One markdown table per condition: pooled scores, then spellF1 per held-out match."""
    board = scoreboard(pipeline.dataset_dir("tab-pooled-" + cond))
    per_match = {m: scoreboard(pipeline.dataset_dir(dataset_name(m, cond))) for m in matches}
    print("\n### {} ({} matches pooled, {} true possessions)".format(
        cond, len(matches), next(iter(board.values()))["n_gt_spells"]))
    print("| model | spellF1 pooled [5-95%] | macroF1 | alive/dead F1 | team acc | poss pred/true "
          "| " + " | ".join(matches) + " |")
    print("|---" * (6 + len(matches)) + "|")
    for name, row in sorted(board.items(), key=lambda item: -float(item[1]["spell_f1_mean"])):
        cells = []
        for match in matches:
            r = per_match[match].get(name)
            cells.append("{:.3f} [{:.2f}, {:.2f}]".format(
                float(r["spell_f1_mean"]), float(r["spell_bootstrap_low"]),
                float(r["spell_bootstrap_high"])) if r and r["spell_bootstrap_low"] else "--")
        print("| {} | {:.3f} [{:.3f}, {:.3f}] | {:.3f} | {:.3f} | {:.3f} | {} / {} | {} |".format(
            name, float(row["spell_f1_mean"]), float(row["spell_bootstrap_low"]),
            float(row["spell_bootstrap_high"]), float(row["macro_f1"]),
            float(row["alive_dead_f1"]), float(row["team_acc_both_alive"]), row["poss"],
            row["n_gt_spells"], " | ".join(cells)))


def cmd_table(args):
    for cond in args.cond:
        print_table(cond, built_matches())


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    degrade = sub.add_parser("degrade")
    degrade.add_argument("--dead-share", type=float, nargs="+", default=list(DEAD_SHARES))
    for name in ("lomo", "predict"):
        command = sub.add_parser(name)
        command.add_argument("--model", required=True, choices=sorted(MAX_ROWS))
        command.add_argument("--train", default=cond_of(DEAD_SHARE))
    sub.choices["lomo"].add_argument("--test", default=cond_of(DEAD_SHARE))
    sub.choices["lomo"].add_argument("--suffix", default="")
    sub.choices["predict"].add_argument("--name", required=True)
    for name in ("score", "table"):
        command = sub.add_parser(name)
        command.add_argument("--cond", nargs="+", default=["clean", cond_of(DEAD_SHARE)])
    sub.choices["score"].add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    {"degrade": cmd_degrade, "lomo": cmd_lomo, "predict": cmd_predict, "score": cmd_score,
     "table": cmd_table}[args.command](args)


if __name__ == "__main__":
    main()
