"""The possession methods compared on the dashboard.

Every method returns a (n_frames,) array of single-letter labels: H, A, D, or X outside the two
halves (ungraded frames, so X there is free and honest).

    provided_baselines   the organisers' nearest-player + majority vote, all four windows
    user_velocity        the drafted idea: ball-direction changes hand possession to the nearest
                         player; out-of-bounds and slow players mean a dead ball; tags as a fallback
    hybrid_hmm           the drafted idea's signals turned into soft evidence and decoded jointly
                         with a 3-state HMM (H / A / D) and switch costs
    pro_soccer_hmm       adds modern pro-football research signals on top of the HMM:
                         * cross-camera broadcast ATD ball recovery (for BAR-ATH)
                         * physics-aware pass trajectory interpolation across short dropouts
                         * Spearman kinematic time-to-intercept (Pitch Control) & pass-corridor lock
                         * 50/50 duel & deflection touch dampening (per RULES.md)
                         * ball-free collective team width expansion, attacking velocity flow, and
                           fast-player convergence cluster ("virtual ball", BallRadar concept)
                         * single-player-off-pitch throw-in / corner-kick waiting & restart taker team
    pro_v2_hmm           error-analysis driven upgrade on top of pro_soccer_hmm:
                         * dead-ball kick-back & unattended rolling ball filter (distinguishes balls
                           tossed/kicked to the spot of a foul from live restarts via lookahead squad speed)
                         * full-pitch unseen-ball foul detector, extended OOB latch, and full
                           SET PIECES tag window alignment
                         * 2D aerial pass-flight protection (dampens fleeting <0.25s proximity/touches
                           when a fast ball >6.5 m/s flies past opponents in 2D projection)
                         * two-pass physical continuity & boundary snapping (merges impossible <2.5s
                           build-up pauses and dead-ball retriever touches, and snaps stoppage/turnover
                           boundaries back to the exact physical trigger frame)
    pro_v3_var_hmm       adds VAR / SAOT & league optical-tracking physics on top of pro_v2_hmm:
                         * SAOT kick-point (ball release frame) to reception-point flight ownership lock
                         * 2D-to-3D parabolic flight height z(t) estimation & 3D Euclidean player-ball
                           distance (so players standing underneath an aerial pass get zero proximity)
                         * pre-reception approach suppression (prevents switching to an intercepting
                           receiver while the ball is still in flight approaching them)
    pro_v4_cop_hmm       Controlled Ownership & Propagation (COP-HMM) v4:
                         * SAOT dwell & launch turnover impulse: preserves controlled turnovers
                           (uncontested possession >= 0.5s before launch) through interception/flight
                         * contested carry / dribble velocity vector synchronization (cosine similarity
                           between ball and closest player velocities during 50/50 duels)
                         * geometric set-piece detection: pitch-coordinate corner arc and 6-yard box
                           goal-kick setup recognition without requiring manual tag XML
                         * refined boundary snapping: fast restart touch snapping (D -> live),
                           live-pass rescue (protects verified pass flights against dead snapping),
                           and first-contact turnover snapping

None of them reads a ground truth. Every constant is a physical or football prior, written next to
where it is used -- none was chosen by scoring against the public five minutes (RULES.md).
"""

import os
import sys

import numpy as np

from .data import AWAY, HOME, ROOT
from .features import (FLIGHT_MS, REACH_M, WALKING_MS, centred_count, runs_of, set_piece_starts,
                       tag_reliability, tag_team_evidence)

sys.path.insert(0, os.path.join(ROOT, "starter"))
import baseline as provided  # noqa: E402  (starter/baseline.py, the organisers' reference)

LETTERS = np.array(["H", "A", "D"])


def _mask_ungraded(labels, f):
    out = labels.astype("<U1").copy()
    out[~f.graded] = "X"
    return out


# -- 1. the provided baseline -------------------------------------------------------------------

def provided_baselines(f, fps=25):
    """starter/baseline.py, re-run on the cached tracking. Same algorithm, same time windows."""
    out = {}
    for name, window in provided.VARIANTS:
        w = window if fps == 25 else int(round(window * fps / 25.0))
        out[name] = provided.smooth(f.nearest_team_raw, w)
    return out


# -- shared: ball direction-change ("touch") events --------------------------------------------

def direction_events(f, fps, k=None, angle_deg=30.0, min_speed=1.5, ratio=2.5, ratio_speed=3.0):
    """Frames where the ball's velocity changes: direction turns or speed jumps/collapses.

    before = p[t] - p[t-k], after = p[t+k] - p[t] over k ~ 0.16 s (4 frames at 25 fps), on the
    smoothed ball. A touch shows up as either
      * a turn of more than `angle_deg` while the ball is moving (dribble, deflection, pass), or
      * a speed ratio above `ratio` with the faster side above `ratio_speed` m/s (a kick from
        rest, a first touch killing the ball).
    Consecutive firing frames are merged into one event at the sharpest frame.
    """
    if k is None:
        k = max(1, int(round(0.16 * fps)))
    x, y = f.bx, f.by
    n = len(x)
    bx_ = np.full(n, np.nan); by_ = np.full(n, np.nan)
    ax_ = np.full(n, np.nan); ay_ = np.full(n, np.nan)
    bx_[k:] = x[k:] - x[:-k]; by_[k:] = y[k:] - y[:-k]
    ax_[:-k] = x[k:] - x[:-k]; ay_[:-k] = y[k:] - y[:-k]
    sb = np.hypot(bx_, by_) * fps / k
    sa = np.hypot(ax_, ay_) * fps / k
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = (bx_ * ax_ + by_ * ay_) / (np.hypot(bx_, by_) * np.hypot(ax_, ay_))
        angle = np.degrees(np.arccos(np.clip(cos, -1, 1)))
        fast, slow = np.fmax(sa, sb), np.fmin(sa, sb)
        turn = (sb > min_speed) & (sa > min_speed) & (angle > angle_deg)
        jump = (fast > ratio_speed) & (fast > ratio * np.maximum(slow, 0.1))
    fire = (turn | jump) & ~np.isnan(sa) & ~np.isnan(sb)
    score = np.where(turn, angle / 180.0, 0.0) + np.where(jump, 1.0, 0.0)

    events = []
    for start, stop in runs_of(fire):
        events.append(start + int(np.argmax(score[start:stop])))
    return np.array(events, dtype=np.int64)


# -- 2. the drafted idea, implemented as written ------------------------------------------------

USER_TAG_CODES = ("PLAYERS IN THE BOX", "BALL IN THE BOX", "BALL IN FINAL THIRD",
                  "CREATING CHANCES")


def user_velocity(m, f):
    """Sequential rules, exactly the drafted logic:

    * each half starts with the team of the player nearest the ball at kick-off;
    * whenever the ball's velocity direction changes, the nearest player's team has it from that
      frame on; while the ball keeps its line (a pass in flight, a dribble's carry) nothing changes;
    * dead ball when the ball is beyond a line (held until it is back in and kicked), or when the
      squad's median speed is below walking pace;
    * on a restart, the nearest player (the taker) gets the ball;
    * while the ball is invisible for more than 2 s, an attacking Smart Tag ("players in the box",
      "ball in the box", "ball in final third", "creating chances") hands it to the tagged team.
    """
    from .features import team_code

    n = m.n
    labels = np.full(n, "X", dtype="<U1")
    events = np.zeros(n, dtype=bool)
    events[direction_events(f, m.fps)] = True

    tag_team = np.full(n, -1, dtype=np.int8)
    for t in m.tags:
        if t["code"].upper() in USER_TAG_CODES:
            team = team_code(m, t["tags"].get("Team"))
            if team is not None:
                tag_team[max(0, t["start_frame"]):min(n, t["end_frame"])] = team

    slow = (np.nan_to_num(f.squad_speed, nan=99.0) < WALKING_MS) | (~m.ball_seen & (f.n_tracked < 5))
    in_bounds_seen = m.ball_seen & ~f.ball_oob
    speed = np.nan_to_num(f.ball_speed, nan=0.0)
    pre_ko = getattr(f, "pre_kickoff", np.zeros(n, dtype=bool))

    for half, first, last in m.halves:
        # kick-off team: nearest player at the first post-kickoff frame the ball is seen
        team = HOME
        ko_start = first
        while ko_start <= last and pre_ko[ko_start]:
            ko_start += 1
        for t in range(ko_start, min(last, ko_start + 10 * m.fps)):
            if f.nearest_team[t] >= 0:
                team = int(f.nearest_team[t])
                break
        oob_latch = False
        dead_prev = False
        unseen_run = 0
        for t in range(first, last + 1):
            if pre_ko[t]:
                labels[t] = "D"
                dead_prev = True
                continue
            if f.ball_oob[t]:
                oob_latch = True
            elif oob_latch and in_bounds_seen[t] and speed[t] > 4.0:
                oob_latch = False
            dead = oob_latch or slow[t]

            unseen_run = 0 if m.ball_seen[t] else unseen_run + 1
            if dead:
                labels[t] = "D"
                dead_prev = True
                continue
            if dead_prev and f.nearest_team[t] >= 0:
                team = int(f.nearest_team[t])      # restart: the taker has it
            dead_prev = False
            if events[t] and f.nearest_team[t] >= 0:
                team = int(f.nearest_team[t])
            elif unseen_run > 2 * m.fps and tag_team[t] >= 0:
                team = int(tag_team[t])
            labels[t] = "H" if team == HOME else "A"
    return labels


# -- 3. hybrid: the same signals as soft evidence, decoded with an HMM --------------------------

class HybridParams(object):
    # Evidence weights, in log-odds units per frame.
    W_PROX = 1.5      # a player of one team clearly closest to a controllable ball
    W_TOUCH = 3.0     # a direction-change event with a player of that team within reach
    W_TAG = 0.10      # per unit of tag reliability; only decisive while the ball is invisible
    PRIOR_DEAD = -1.0 # the ball is alive most of the time
    W_SLOW = 1.0      # per 0.5 m/s below walking pace
    W_OOB = 3.0       # ball beyond a line
    W_OOB_LATCH = 2.0 # ...and until it is next seen back inside
    W_STATIONARY = 1.0  # a ball placed still for a second
    W_PLAYER_OOB = 0.5  # someone standing outside the lines (throw-in / corner taker)
    W_MOVING = 2.0    # a moving ball inside the pitch is live play
    W_SETPIECE = 0.5  # just before a tagged set piece
    # Transition costs: how much accumulated evidence a change has to overcome. A change of team
    # needing ~0.3 s of clear control is the "controlled possession" rule in HMM form; a
    # touch/deflection without control cannot pay it.
    C_SWITCH = 6.0
    C_DEAD = 10.0


def hybrid_evidence(m, f, params=HybridParams, use_tags=True, reliability=None):
    P = params
    n = m.n
    seen = m.ball_seen
    speed = f.ball_speed

    # --- team evidence (+ home, - away) ---
    with np.errstate(invalid="ignore"):
        near = np.fmin(f.d_home, f.d_away)
        margin = np.abs(f.d_home - f.d_away)
        controllable = seen & (near < REACH_M) & ~(speed > FLIGHT_MS)
        strength = np.clip(1.0 - near / REACH_M, 0, 1) * np.clip(margin / 2.0, 0, 1)
        side = np.where(f.d_home < f.d_away, 1.0, -1.0)
    e_team = np.where(controllable, side * strength * P.W_PROX, 0.0)

    ev = direction_events(f, m.fps)
    touch_ok = (near[ev] < REACH_M) & np.isfinite(near[ev])
    e_team[ev[touch_ok]] += side[ev[touch_ok]] * P.W_TOUCH

    if use_tags:
        if reliability is None:
            reliability = tag_reliability(m, f)
        e_team += P.W_TAG * tag_team_evidence(m, reliability)

    # --- dead-ball evidence (log-odds dead vs alive) ---
    e_dead = np.full(n, P.PRIOR_DEAD)
    sq = f.squad_speed
    e_dead += np.where(np.isnan(sq), 0.0,
                       P.W_SLOW * np.clip((WALKING_MS - np.nan_to_num(sq)) / 0.5, -2, 2))
    # In broadcast tracking (e.g. SkillCorner), camera cuts / replays have 0 players tracked and no ball
    e_dead += np.where(~seen & (f.n_tracked < 5), 1.5, 0.0)
    e_dead += P.W_OOB * f.ball_oob
    latch = np.zeros(n, dtype=bool)
    on = False
    in_seen = seen & ~f.ball_oob
    for t in range(n):  # cheap: one pass of booleans
        if f.ball_oob[t]:
            on = True
        elif in_seen[t]:
            on = False
        latch[t] = on and not f.ball_oob[t]
    e_dead += P.W_OOB_LATCH * latch
    e_dead += P.W_STATIONARY * f.ball_stationary
    e_dead += P.W_PLAYER_OOB * f.player_oob
    e_dead -= P.W_MOVING * (in_seen & (np.nan_to_num(speed) > 4.0))
    if use_tags:
        for frame, _, _ in set_piece_starts(m):
            e_dead[max(0, frame - 8 * m.fps):min(n, frame + m.fps)] += P.W_SETPIECE
    pre_ko = getattr(f, "pre_kickoff", None)
    if pre_ko is not None and pre_ko.any():
        e_dead[pre_ko] = 50.0
    return e_team, e_dead


def viterbi(e_team, e_dead, c_switch, c_dead):
    """Best H/A/D path. Emissions: H = +e/2, A = -e/2, D = e_dead. Costs on changes of state."""
    n = len(e_team)
    back = np.zeros((n, 3), dtype=np.int8)
    eh = (0.5 * e_team).tolist()
    ea = (-0.5 * e_team).tolist()
    ed = e_dead.tolist()
    sh, sa, sd = eh[0], ea[0], ed[0]
    for t in range(1, n):
        # into H
        ch = (sh, sa - c_switch, sd - c_dead)
        bh = 0 if ch[0] >= ch[1] and ch[0] >= ch[2] else (1 if ch[1] >= ch[2] else 2)
        ca = (sh - c_switch, sa, sd - c_dead)
        ba = 1 if ca[1] >= ca[0] and ca[1] >= ca[2] else (0 if ca[0] >= ca[2] else 2)
        cd = (sh - c_dead, sa - c_dead, sd)
        bd = 2 if cd[2] >= cd[0] and cd[2] >= cd[1] else (0 if cd[0] >= cd[1] else 1)
        back[t, 0], back[t, 1], back[t, 2] = bh, ba, bd
        sh, sa, sd = ch[bh] + eh[t], ca[ba] + ea[t], cd[bd] + ed[t]
    path = np.zeros(n, dtype=np.int8)
    path[-1] = int(np.argmax([sh, sa, sd]))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def hybrid_hmm(m, f, params=HybridParams, use_tags=True, reliability=None):
    e_team, e_dead = hybrid_evidence(m, f, params, use_tags, reliability)
    labels = np.full(m.n, "X", dtype="<U1")
    scale = m.fps / 25.0
    c_sw = params.C_SWITCH * scale
    c_dd = params.C_DEAD * scale
    for _, first, last in m.halves:
        sl = slice(first, last + 1)
        path = viterbi(e_team[sl], e_dead[sl], c_sw, c_dd)
        labels[sl] = LETTERS[path]
    return labels


# -- 4. pro_soccer_hmm: modern football tracking research + single-player-oob throw/corner ------

def pro_soccer_hmm(m, f, params=HybridParams, use_tags=True, reliability=None):
    """Combines all hybrid HMM signals with five modern professional football analytics signals:

    1. 50/50 Duel & Deflection Touch Dampening (RULES.md): when both teams are within 1.0 m of
       each other at a ball-direction change, scale down the touch impulse so blocks/deflections
       do not trigger false possession flips.
    2. Spearman Kinematic Time-to-Intercept (Pitch Control) on recovered/interpolated ball frames
       (filling in unseen ball gaps using broadcast cross-camera recovery and pass interpolation).
    3. Pass Departure-to-Arrival Corridor Lock: when a short unseen gap begins and ends with the
       same team clearly controlling the ball (<2.2 m), lock that team across the flight corridor.
    4. Ball-Free Collective Team Shape & Flow (BallRadar / PathCRF concept): tactical width
       expansion (the team in possession spreads wider across the pitch while the defending block
       compacts), collective longitudinal velocity flow, and fast-player convergence cluster.
    5. Single Player Off-Pitch = Throw-In or Corner Kick Setup: when exactly one player is
       standing/walking outside the pitch lines while the ball is not in play away from them,
       boost dead-ball evidence (D), and when they release the ball back into play, credit their
       team as the restart taker for 2 s.
    """
    n = m.n
    e_team, e_dead = hybrid_evidence(m, f, params, use_tags, reliability)

    # 1. Dampen contested 50/50 touch impulses
    ev = direction_events(f, m.fps)
    with np.errstate(invalid="ignore"):
        near = np.fmin(f.d_home, f.d_away)
        margin = np.abs(f.d_home - f.d_away)
        side = np.where(f.d_home < f.d_away, 1.0, -1.0)
        contested = (near[ev] < REACH_M) & np.isfinite(near[ev]) & (margin[ev] < 1.0)
    e_team[ev[contested]] -= (
        side[ev[contested]] * params.W_TOUCH *
        (1.0 - np.clip(margin[ev[contested]] / 1.0, 0.15, 1.0))
    )

    # 2. Spearman Pitch Control on recovered/interpolated missing-ball frames
    near_i = np.fmin(f.d_home_interp, f.d_away_interp)
    e_team += np.where(f.ball_recovered & (near_i < 3.0), 0.75 * f.pitch_control, 0.0)

    # 3. Pass corridor lock across unseen-ball gaps
    e_team += 0.65 * f.pass_corridor

    # 4. Ball-free collective team width expansion + attacking flow + convergence cluster
    w_col = np.where(m.ball_seen, 0.25, 0.65)
    e_team += w_col * (0.75 * f.team_width_diff + 0.15 * f.team_flow + 0.10 * f.vball_cluster)

    # 5. Single player off-pitch waiting (throw-in / corner kick) + restart taker team
    e_dead += 1.5 * f.single_oob_waiting
    for a, b in runs_of(f.single_oob_waiting):
        if b - a >= max(2, m.fps // 3) and b < n:
            s_val = float(np.median(f.oob_taker_sign[a:b]))
            e_team[b:min(n, b + 2 * m.fps)] += 0.8 * s_val

    labels = np.full(n, "X", dtype="<U1")
    scale = m.fps / 25.0
    c_sw = params.C_SWITCH * scale
    c_dd = params.C_DEAD * scale
    for _, first, last in m.halves:
        sl = slice(first, last + 1)
        path = viterbi(e_team[sl], e_dead[sl], c_sw, c_dd)
        labels[sl] = LETTERS[path]
    return labels


# -- 5. pro_v2_hmm: stoppage-context physics, 2D flight protection & boundary snapping -----------

def _snap_boundaries(path, offset, f, m, sq, speed, near, side):
    """Two-pass physical continuity and boundary refinement on Viterbi states (0=H, 1=A, 2=D).

    Pass 1 (physical continuity):
      * A dead-ball stoppage (D) between two live spells cannot last < 2.5 s (it is a build-up pause).
      * A brief live segment (< 2.5 s, or < 5.5 s while the squad is walking or decelerating into D)
        sandwiched between two D spells is a player moving/retrieving the dead ball during a stoppage.

    Pass 2 (boundary snapping):
      * Entering D (Alive -> D): snap backward (up to 2.5 s) across OOB, unseen-ball, or unattended
        deceleration frames so the stoppage begins at the whistle/line-crossing rather than after
        players finish decelerating.
      * Turnover (H <-> A): snap backward (up to 1.0 s) to the exact frame where the winning team
        first established clean proximity (< 1.8 m), removing the Viterbi switch-cost lag.
    """
    out = path.copy()
    L = len(out)
    fps = m.fps

    for _ in range(2):
        changes = np.flatnonzero(np.diff(out) != 0) + 1
        starts = np.concatenate(([0], changes))
        stops = np.concatenate((changes, [L]))
        for idx in range(1, len(starts) - 1):
            s = int(starts[idx])
            e = int(stops[idx])
            dur = e - s
            st = out[s]
            prev_st = out[s - 1]
            next_st = out[e]
            if st == 2 and prev_st in (0, 1) and next_st in (0, 1) and dur < int(round(2.5 * fps)):
                out[s:e] = prev_st
            elif st in (0, 1) and prev_st == 2 and next_st == 2:
                med_sq = float(np.median(sq[offset + s:offset + e]))
                decel_end = (sq[offset + e - 1] < 1.58) and (sq[offset + e - 1] < sq[offset + s] - 0.5)
                if dur < int(round(2.5 * fps)) or (dur < int(round(5.5 * fps)) and (med_sq < 1.55 or decel_end)):
                    out[s:e] = 2

    max_back_d = int(round(2.5 * fps))
    max_back_sw = int(round(1.0 * fps))

    changes = np.flatnonzero(np.diff(out) != 0) + 1
    starts = np.concatenate(([0], changes))
    stops = np.concatenate((changes, [L]))

    for idx in range(len(starts)):
        s = int(starts[idx])
        e = int(stops[idx])
        st = out[s]

        if st == 2 and s > 0 and (e - s) >= int(2.5 * fps):
            lo = max(starts[idx - 1] + int(1.0 * fps), s - max_back_d)
            snap = s
            for t in range(s - 1, lo - 1, -1):
                gi = offset + t
                if f.ball_oob[gi] or (not m.ball_seen[gi]) or (near[gi] > 3.8 and sq[gi] < 1.85):
                    snap = t
                else:
                    break
            if snap < s:
                out[snap:s] = 2

        elif st in (0, 1) and s > 0 and out[s - 1] in (0, 1) and out[s - 1] != st and (e - s) >= int(1.5 * fps):
            new_sign = 1.0 if st == 0 else -1.0
            lo = max(starts[idx - 1] + int(0.8 * fps), s - max_back_sw)
            snap = s
            for t in range(s - 1, lo - 1, -1):
                gi = offset + t
                if not m.ball_seen[gi]:
                    break
                if side[gi] == -new_sign and near[gi] < 2.0:
                    break
                if side[gi] == new_sign and near[gi] < 1.8:
                    snap = t
            if snap < s:
                out[snap:s] = st

    return out


def pro_v2_hmm(m, f, params=HybridParams, use_tags=True, reliability=None):
    """Error-analysis driven upgrade on top of pro_soccer_hmm:

    1. Dead-Ball Kick-Back & Unattended Rolling Ball Filter: uses lookahead squad speed (2.2 s)
       to distinguish real restart kicks (where the squad accelerates into live play) from dead
       balls rolling or kicked back to the spot of a foul while players walk.
    2. Stoppage Context Detection: combines unattended slowdown, extended out-of-bounds latch
       (until a player receives/plays the ball in-bounds or squad speed > 1.55 m/s), full-pitch
       unseen-ball foul waits (>= 3 s unseen with >= 18 tracked players walking < 1.35 m/s), and
       full SET PIECES tag windows while gating down unseen-ball team-shape/tag priors during waits.
    3. 2D Aerial Pass-Flight Protection: dampens fleeting (< 0.24 s) proximity and touch impulses
       when a fast ball (> 6.5 m/s) flies past opposing players in 2D projection.
    4. Two-Pass Physical Continuity & Boundary Snapping: removes impossible < 2.5 s build-up D
       blips and dead-ball retriever touches between D spells, and snaps stoppage/turnover
       boundaries back to the exact physical trigger frame.
    """
    n = m.n
    fps = m.fps
    seen = m.ball_seen
    in_seen = seen & ~f.ball_oob
    speed = np.nan_to_num(f.ball_speed, nan=0.0)
    sq = np.nan_to_num(f.squad_speed, nan=2.5)
    k_look = int(round(2.2 * fps))
    sq_future = np.empty(n)
    sq_future[:-k_look] = sq[k_look:]
    sq_future[-k_look:] = sq[-1]

    if use_tags and reliability is None:
        reliability = tag_reliability(m, f)
    e_team, e_dead = hybrid_evidence(m, f, params, use_tags, reliability)

    with np.errstate(invalid="ignore"):
        near = np.fmin(f.d_home, f.d_away)
        margin = np.abs(f.d_home - f.d_away)
        side = np.where(f.d_home < f.d_away, 1.0, -1.0)

    # 1. Dead-ball kick-back & unattended rolling ball filter
    dead_ball_kick = in_seen & (speed > 4.0) & (
        ((sq < 1.45) & (sq_future < 1.50) & (near > 2.2)) |
        ((near > 4.0) & (sq_future < 1.42) & (speed < 12.0))
    )
    e_dead += params.W_MOVING * dead_ball_kick

    # 2. Stoppage context detectors (unattended slowdown, extended OOB latch, unseen foul, set-piece wait)
    unattended_stop = in_seen & (near > 3.8) & (sq < 1.50) & (sq_future < 1.35)
    ext_latch = np.zeros(n, dtype=bool)
    on_ext = False
    for t in range(n):
        if f.ball_oob[t]:
            on_ext = True
        elif on_ext and (sq[t] > 1.55 or (in_seen[t] and near[t] < 3.0 and speed[t] > 3.0)):
            on_ext = False
        ext_latch[t] = on_ext

    unseen_stop = np.zeros(n, dtype=bool)
    for a, b in runs_of(~in_seen & f.graded):
        if (b - a) >= int(3.0 * fps) and np.median(f.n_tracked[a:b]) >= 18 and np.median(sq[a:b]) < 1.35:
            unseen_stop[a:b] = True

    sp_wait_mask = np.zeros(n, dtype=bool)
    if use_tags:
        for t_tag in m.tags:
            if t_tag["code"].upper() == "SET PIECES":
                s0 = max(0, t_tag["start_frame"] - 4 * fps)
                s1 = min(n, t_tag["end_frame"])
                sp_wait = (sq[s0:s1] < 1.48) & (sq_future[s0:s1] < 1.55)
                sp_wait_mask[s0:s1] |= sp_wait

    unseen_stop_active = unseen_stop & ((sq < 1.45) | (sq_future < 1.35))
    stop_ctx = unattended_stop | (ext_latch & (sq_future < 1.45)) | unseen_stop_active | sp_wait_mask
    e_dead += np.where(stop_ctx, 1.10, 0.0)

    stop_gate = np.where(sp_wait_mask | unseen_stop_active, 0.15, 1.0)

    # 3. 50/50 duel dampening + 2D aerial pass-flight protection
    ev = direction_events(f, fps)
    with np.errstate(invalid="ignore"):
        contested = (near[ev] < REACH_M) & np.isfinite(near[ev]) & (margin[ev] < 1.0)
    e_team[ev[contested]] -= (
        side[ev[contested]] * params.W_TOUCH *
        (1.0 - np.clip(margin[ev[contested]] / 1.0, 0.15, 1.0))
    )

    w_sus = max(2, int(round(0.35 * fps)))
    h_cnt = centred_count(in_seen & (f.d_home < 2.2) & (f.d_home < f.d_away), w_sus)
    a_cnt = centred_count(in_seen & (f.d_away < 2.2) & (f.d_away < f.d_home), w_sus)
    t_cnt = np.where(side > 0, h_cnt, a_cnt)
    min_cnt = max(2, int(round(0.24 * fps)))
    fly_by = in_seen & (near < REACH_M) & (speed > 6.5) & (t_cnt < min_cnt)
    with np.errstate(invalid="ignore"):
        strength = np.clip(1.0 - near / REACH_M, 0, 1) * np.clip(margin / 2.0, 0, 1)
    e_team -= np.where(fly_by, 0.75 * side * strength * params.W_PROX, 0.0)
    fly_touch = (
        (near[ev] < REACH_M) & np.isfinite(near[ev]) &
        (margin[ev] >= 1.0) & (speed[ev] > 6.5) & (t_cnt[ev] < min_cnt)
    )
    e_team[ev[fly_touch]] -= 0.65 * side[ev[fly_touch]] * params.W_TOUCH

    # Spearman Pitch Control, pass corridor, collective team shape (gated during stoppage waits), and OOB taker
    near_i = np.fmin(f.d_home_interp, f.d_away_interp)
    e_team += np.where(f.ball_recovered & (near_i < 3.0), 0.75 * f.pitch_control, 0.0)
    e_team += 0.65 * f.pass_corridor

    w_col = np.where(m.ball_seen, 0.25, 0.65 * stop_gate)
    e_team += w_col * (0.75 * f.team_width_diff + 0.15 * f.team_flow + 0.10 * f.vball_cluster)

    if use_tags:
        tag_ev = params.W_TAG * tag_team_evidence(m, reliability)
        e_team -= np.where(~in_seen, (1.0 - stop_gate) * tag_ev, 0.0)

    e_dead += 1.5 * f.single_oob_waiting
    for a, b in runs_of(f.single_oob_waiting):
        if b - a >= max(2, fps // 3) and b < n:
            e_team[b:min(n, b + 2 * fps)] += 0.8 * float(np.median(f.oob_taker_sign[a:b]))

    labels = np.full(n, "X", dtype="<U1")
    scale = fps / 25.0
    c_sw = params.C_SWITCH * scale
    c_dd = params.C_DEAD * scale
    for _, first, last in m.halves:
        sl = slice(first, last + 1)
        path = viterbi(e_team[sl], e_dead[sl], c_sw, c_dd)
        path = _snap_boundaries(path, first, f, m, sq, speed, near, side)
        labels[sl] = LETTERS[path]
    return labels


# -- 6. pro_v3_var_hmm: SAOT kick-point flight lock & 2D-to-3D parabolic height distance --------

def _compute_var_kinematics(m, f, ev):
    """SAOT Kick-Point to Reception-Point Flight Ownership, 3D Height z(t) & Approach Suppression.

    1. Kick-Point (Ball Release Frame) Detection: finds frames where a player within tight reach
       (< 1.65 m, margin > 0.8 m) launches the ball from control (speed <= 5.5 m/s) or via a sharp
       velocity-change touch event (ev) into a fast flight trajectory (speed >= 7.5 m/s, displacement
       >= 10 m) until the next reception/contact frame t_rec.
    2. Flight Ownership Propagation: when the kicking team completes the pass (same team closest at
       t_rec) or dominates a long flight (>= 1.0 s, >= 16 m, >= 13 m/s), locks flight_team across
       [t_kick, t_rec] so the kicking team retains possession throughout the flight arc.
    3. 2D-to-3D Ballistic Height Estimation z(t): models vertical elevation z(t) along aerial passes
       (>= 14 m or peak speed >= 11 m/s) using projectile kinematics so opposing players standing
       underneath the flight path in 2D projection have 3D distance sqrt(d_2d^2 + z^2) > REACH_M.
    """
    n = m.n
    fps = m.fps
    in_seen = m.ball_seen & ~f.ball_oob
    speed = np.nan_to_num(f.ball_speed, nan=0.0)
    with np.errstate(invalid="ignore"):
        near = np.fmin(f.d_home, f.d_away)
        margin = np.abs(f.d_home - f.d_away)
        side = np.where(f.d_home < f.d_away, 1.0, -1.0)

    ev_mask = np.zeros(n, dtype=bool)
    ev_mask[ev] = True

    k_step = max(2, int(round(0.20 * fps)))
    flight_team = np.zeros(n, dtype=np.float64)
    pre_rec_suppress = np.zeros(n, dtype=bool)
    z_height = np.zeros(n, dtype=np.float64)

    t = 1
    max_flight = int(round(4.0 * fps))
    min_flight = max(3, int(round(0.35 * fps)))

    while t < n - k_step:
        if in_seen[t] and near[t] < 1.65 and margin[t] > 0.8:
            t_prev = max(0, t - max(1, k_step // 2))
            t_next = min(n - 1, t + k_step)
            had_touch_or_ctrl = (speed[t_prev] <= 5.5) or ev_mask[t] or (
                ev_mask[max(0, t - 1)] or ev_mask[min(n - 1, t + 1)]
            )
            accel_launch = (
                in_seen[t_next] and speed[t_next] >= 7.5 and
                (speed[t_next] - speed[t_prev] >= 2.8 or (speed[t_prev] <= 3.5 and speed[t_next] >= 6.5)) and
                near[t_next] > near[t] + 0.4
            )
            if had_touch_or_ctrl and accel_launch:
                kicker_side = side[t]
                t_rec = t + 1
                while t_rec < min(n, t + max_flight):
                    if not in_seen[t_rec]:
                        break
                    if t_rec >= t + k_step:
                        if speed[t_rec] < 4.2 and near[t_rec] < 2.2:
                            break
                        if near[t_rec] < 1.65 and (ev_mask[t_rec] or speed[t_rec] < 6.0):
                            break
                    t_rec += 1

                dur = t_rec - t
                if dur >= min_flight:
                    disp = np.hypot(f.bx[min(n - 1, t_rec - 1)] - f.bx[t],
                                    f.by[min(n - 1, t_rec - 1)] - f.by[t])
                    if disp >= 10.0:
                        w_end = slice(max(t + 1, t_rec - 2), min(n, t_rec + int(round(0.4 * fps)) + 1))
                        end_h = np.nanmedian(f.d_home[w_end])
                        end_a = np.nanmedian(f.d_away[w_end])
                        end_side = 1.0 if end_h < end_a else -1.0
                        peak_spd = float(np.max(speed[t:t_rec]))
                        kicker_dom = (
                            dur >= int(round(1.0 * fps)) and disp >= 16.0 and peak_spd >= 13.0 and
                            float(np.mean(side[t:t_rec] == kicker_side)) >= 0.75
                        )
                        if end_side == kicker_side or kicker_dom:
                            flight_team[t:t_rec] = kicker_side

                        if disp >= 14.0 or peak_spd >= 11.0:
                            tau = np.arange(dur, dtype=np.float64) / float(fps)
                            T_sec = dur / float(fps)
                            z_arc = np.minimum(4.0, 0.5 * 9.81 * tau * (T_sec - tau) * 0.35)
                            z_height[t:t_rec] = np.maximum(z_height[t:t_rec], z_arc)
                        if end_side == -kicker_side and kicker_dom:
                            app_start = max(t + 2, t_rec - int(round(0.40 * fps)))
                            pre_rec_suppress[app_start:t_rec] = True
                        t = t_rec
                        continue
        t += 1

    return flight_team, z_height, pre_rec_suppress


def pro_v3_var_hmm(m, f, params=HybridParams, use_tags=True, reliability=None):
    """Combines pro_v2_hmm with VAR / SAOT kick-point flight lock and 2D-to-3D parabolic height."""
    n = m.n
    fps = m.fps
    seen = m.ball_seen
    in_seen = seen & ~f.ball_oob
    speed = np.nan_to_num(f.ball_speed, nan=0.0)
    sq = np.nan_to_num(f.squad_speed, nan=2.5)
    k_look = int(round(2.2 * fps))
    sq_future = np.empty(n)
    sq_future[:-k_look] = sq[k_look:]
    sq_future[-k_look:] = sq[-1]

    if use_tags and reliability is None:
        reliability = tag_reliability(m, f)
    e_team, e_dead = hybrid_evidence(m, f, params, use_tags, reliability)

    with np.errstate(invalid="ignore"):
        near = np.fmin(f.d_home, f.d_away)
        margin = np.abs(f.d_home - f.d_away)
        side = np.where(f.d_home < f.d_away, 1.0, -1.0)

    ev = direction_events(f, fps)
    flight_team, z_height, pre_rec_suppress = _compute_var_kinematics(m, f, ev)

    # 1. Dead-ball kick-back & unattended rolling ball filter
    dead_ball_kick = in_seen & (speed > 4.0) & (
        ((sq < 1.45) & (sq_future < 1.50) & (near > 2.2)) |
        ((near > 4.0) & (sq_future < 1.42) & (speed < 12.0))
    )
    e_dead += params.W_MOVING * dead_ball_kick

    # 2. Stoppage context detectors
    unattended_stop = in_seen & (near > 3.8) & (sq < 1.50) & (sq_future < 1.35) & (flight_team == 0)
    ext_latch = np.zeros(n, dtype=bool)
    on_ext = False
    for t in range(n):
        if f.ball_oob[t]:
            on_ext = True
        elif on_ext and (sq[t] > 1.55 or (in_seen[t] and near[t] < 3.0 and speed[t] > 3.0)):
            on_ext = False
        ext_latch[t] = on_ext

    unseen_stop = np.zeros(n, dtype=bool)
    for a, b in runs_of(~in_seen & f.graded):
        if (b - a) >= int(3.0 * fps) and np.median(f.n_tracked[a:b]) >= 18 and np.median(sq[a:b]) < 1.35:
            unseen_stop[a:b] = True

    sp_wait_mask = np.zeros(n, dtype=bool)
    if use_tags:
        for t_tag in m.tags:
            if t_tag["code"].upper() == "SET PIECES":
                s0 = max(0, t_tag["start_frame"] - 4 * fps)
                s1 = min(n, t_tag["end_frame"])
                sp_wait = (sq[s0:s1] < 1.48) & (sq_future[s0:s1] < 1.55)
                sp_wait_mask[s0:s1] |= sp_wait

    unseen_stop_active = unseen_stop & ((sq < 1.45) | (sq_future < 1.35))
    stop_ctx = unattended_stop | (ext_latch & (sq_future < 1.45)) | unseen_stop_active | sp_wait_mask
    e_dead += np.where(stop_ctx, 1.10, 0.0)

    stop_gate = np.where(sp_wait_mask | unseen_stop_active, 0.15, 1.0)

    # 3. 50/50 duel dampening + 2D-to-3D parabolic height distance & flight protection
    with np.errstate(invalid="ignore"):
        contested = (near[ev] < REACH_M) & np.isfinite(near[ev]) & (margin[ev] < 1.0)
    e_team[ev[contested]] -= (
        side[ev[contested]] * params.W_TOUCH *
        (1.0 - np.clip(margin[ev[contested]] / 1.0, 0.15, 1.0))
    )

    w_sus = max(2, int(round(0.35 * fps)))
    h_cnt = centred_count(in_seen & (f.d_home < 2.2) & (f.d_home < f.d_away), w_sus)
    a_cnt = centred_count(in_seen & (f.d_away < 2.2) & (f.d_away < f.d_home), w_sus)
    t_cnt = np.where(side > 0, h_cnt, a_cnt)
    min_cnt = max(2, int(round(0.24 * fps)))

    near_3d = np.hypot(near, z_height)
    airborne_pass_by = in_seen & (near < REACH_M) & ((near_3d > REACH_M) | pre_rec_suppress)
    fly_by = (in_seen & (near < REACH_M) & (speed > 6.5) & (t_cnt < min_cnt)) | airborne_pass_by
    with np.errstate(invalid="ignore"):
        strength = np.clip(1.0 - near / REACH_M, 0, 1) * np.clip(margin / 2.0, 0, 1)
    e_team -= np.where(fly_by, 0.85 * side * strength * params.W_PROX, 0.0)
    fly_touch = (
        (near[ev] < REACH_M) & np.isfinite(near[ev]) &
        (margin[ev] >= 1.0) & (((speed[ev] > 6.5) & (t_cnt[ev] < min_cnt)) | airborne_pass_by[ev])
    )
    e_team[ev[fly_touch]] -= 0.75 * side[ev[fly_touch]] * params.W_TOUCH

    # 4. SAOT Kick-Point to Reception-Point Flight Ownership Lock
    e_team += 0.95 * flight_team
    flight_gate = np.where(flight_team != 0, 0.25, 1.0)

    near_i = np.fmin(f.d_home_interp, f.d_away_interp)
    e_team += np.where(f.ball_recovered & (near_i < 3.0), 0.75 * f.pitch_control, 0.0)
    e_team += 0.65 * f.pass_corridor

    w_col = np.where(m.ball_seen, 0.25 * flight_gate, 0.65 * stop_gate)
    e_team += w_col * (0.75 * f.team_width_diff + 0.15 * f.team_flow + 0.10 * f.vball_cluster)

    if use_tags:
        tag_ev = params.W_TAG * tag_team_evidence(m, reliability)
        e_team -= np.where(~in_seen, (1.0 - stop_gate) * tag_ev, 0.0)

    e_dead += 1.5 * f.single_oob_waiting
    for a, b in runs_of(f.single_oob_waiting):
        if b - a >= max(2, fps // 3) and b < n:
            e_team[b:min(n, b + 2 * fps)] += 0.8 * float(np.median(f.oob_taker_sign[a:b]))

    labels = np.full(n, "X", dtype="<U1")
    scale = fps / 25.0
    c_sw = params.C_SWITCH * scale
    c_dd = params.C_DEAD * scale
    for _, first, last in m.halves:
        sl = slice(first, last + 1)
        path = viterbi(e_team[sl], e_dead[sl], c_sw, c_dd)
        path = _snap_boundaries(path, first, f, m, sq, speed, near, side)
        labels[sl] = LETTERS[path]
    return labels


# -- 7. pro_v4_cop_hmm: Controlled Ownership & Propagation (COP-HMM) -----------------------------

def _compute_player_ball_sync(m, f):
    """Cosine similarity between closest player velocity and ball velocity (carry/dribble sync)."""
    n = m.n
    fps = m.fps
    bx, by = f.bx, f.by
    vbx, vby = f.vx, f.vy
    b_spd = np.nan_to_num(f.ball_speed, nan=0.0)
    home_cols = m.team == HOME

    dt = max(1, int(round(0.2 * fps)))
    vpx = np.zeros_like(m.px)
    vpy = np.zeros_like(m.py)
    vpx[dt:] = (m.px[dt:] - m.px[:-dt]) * (fps / float(dt))
    vpy[dt:] = (m.py[dt:] - m.py[:-dt]) * (fps / float(dt))

    with np.errstate(invalid="ignore"):
        dh = np.where(home_cols[None, :], np.hypot(m.px - bx[:, None], m.py - by[:, None]), np.nan)
        da = np.where(~home_cols[None, :], np.hypot(m.px - bx[:, None], m.py - by[:, None]), np.nan)

    slot_h = np.nanargmin(np.where(np.isnan(dh), np.inf, dh), axis=1)
    slot_a = np.nanargmin(np.where(np.isnan(da), np.inf, da), axis=1)

    idx = np.arange(n)
    vpx_h, vpy_h = vpx[idx, slot_h], vpy[idx, slot_h]
    vpx_a, vpy_a = vpx[idx, slot_a], vpy[idx, slot_a]

    sp_h = np.hypot(vpx_h, vpy_h)
    sp_a = np.hypot(vpx_a, vpy_a)

    dot_h = vpx_h * vbx + vpy_h * vby
    dot_a = vpx_a * vbx + vpy_a * vby
    denom_h = np.maximum(sp_h * b_spd, 1e-6)
    denom_a = np.maximum(sp_a * b_spd, 1e-6)
    cos_h = np.where((b_spd > 1.2) & (sp_h > 1.0) & (f.d_home < 2.0), dot_h / denom_h, 0.0)
    cos_a = np.where((b_spd > 1.2) & (sp_a > 1.0) & (f.d_away < 2.0), dot_a / denom_a, 0.0)
    return np.nan_to_num(cos_h, nan=0.0), np.nan_to_num(cos_a, nan=0.0)


def _detect_setpiece_geometry(m, f):
    """Detects corner kicks and 6-yard goal-kick setups purely from player/ball pitch geometry."""
    seen = m.ball_seen
    bx, by = f.bx, f.by
    sq = np.nan_to_num(f.squad_speed, nan=2.5)
    L, W = m.length, m.width

    # Corner arc zones: within 4.5m of pitch corners
    c_tl = np.hypot(bx, by) < 4.5
    c_tr = np.hypot(bx - L, by) < 4.5
    c_bl = np.hypot(bx, by - W) < 4.5
    c_br = np.hypot(bx - L, by - W) < 4.5
    near_corner = seen & (c_tl | c_tr | c_bl | c_br) & (sq < 1.45) & (f.nearest_dist < 2.5)

    # 6-yard box stationary goal kicks
    gk_left = (bx < 6.0) & (np.abs(by - W / 2.0) < 9.5)
    gk_right = (bx > L - 6.0) & (np.abs(by - W / 2.0) < 9.5)
    in_6yd = seen & (gk_left | gk_right) & (sq < 1.35) & f.ball_stationary

    return near_corner | in_6yd


def _compute_var_kinematics_v4(m, f, ev):
    """SAOT Kick-Point, Dwell & Launch Turnover Impulse, and 3D Parabolic Height."""
    n = m.n
    fps = m.fps
    seen = m.ball_seen
    in_seen = seen & ~f.ball_oob
    speed = np.nan_to_num(f.ball_speed, nan=0.0)
    with np.errstate(invalid="ignore"):
        near = np.fmin(f.d_home, f.d_away)
        margin = np.abs(f.d_home - f.d_away)
        side = np.where(f.d_home < f.d_away, 1.0, -1.0)

    ev_mask = np.zeros(n, dtype=bool)
    ev_mask[ev] = True

    k_step = max(2, int(round(0.20 * fps)))
    flight_team = np.zeros(n, dtype=np.float64)
    pre_rec_suppress = np.zeros(n, dtype=bool)
    z_height = np.zeros(n, dtype=np.float64)

    t = 1
    max_flight = int(round(4.0 * fps))
    min_flight = max(3, int(round(0.35 * fps)))

    while t < n - k_step:
        if in_seen[t] and near[t] < 1.65 and margin[t] > 0.8:
            t_prev = max(0, t - max(1, k_step // 2))
            t_next = min(n - 1, t + k_step)
            had_touch_or_ctrl = (speed[t_prev] <= 5.5) or ev_mask[t] or (
                ev_mask[max(0, t - 1)] or ev_mask[min(n - 1, t + 1)]
            )
            accel_launch = (
                in_seen[t_next] and speed[t_next] >= 7.5 and
                (speed[t_next] - speed[t_prev] >= 2.8 or (speed[t_prev] <= 3.5 and speed[t_next] >= 6.5)) and
                near[t_next] > near[t] + 0.4
            )
            if had_touch_or_ctrl and accel_launch:
                kicker_side = side[t]
                t_rec = t + 1
                while t_rec < min(n, t + max_flight):
                    if not in_seen[t_rec]:
                        break
                    if t_rec >= t + k_step:
                        if speed[t_rec] < 4.2 and near[t_rec] < 2.2:
                            break
                        if near[t_rec] < 1.65 and (ev_mask[t_rec] or speed[t_rec] < 6.0):
                            break
                    t_rec += 1

                dur = t_rec - t
                if dur >= min_flight:
                    disp = np.hypot(f.bx[min(n - 1, t_rec - 1)] - f.bx[t],
                                    f.by[min(n - 1, t_rec - 1)] - f.by[t])
                    if disp >= 10.0:
                        w_end = slice(max(t + 1, t_rec - 2), min(n, t_rec + int(round(0.4 * fps)) + 1))
                        end_h = np.nanmedian(f.d_home[w_end])
                        end_a = np.nanmedian(f.d_away[w_end])
                        end_side = 1.0 if end_h < end_a else -1.0
                        peak_spd = float(np.max(speed[t:t_rec]))
                        kicker_dom = (
                            dur >= int(round(1.0 * fps)) and disp >= 16.0 and peak_spd >= 13.0 and
                            float(np.mean(side[t:t_rec] == kicker_side)) >= 0.75
                        )
                        t_dwell_lo = max(0, t - int(round(0.50 * fps)))
                        had_dwell = (
                            t - t_dwell_lo >= max(3, int(round(0.35 * fps))) and
                            float(np.mean((side[t_dwell_lo:t] == kicker_side) & (near[t_dwell_lo:t] < 1.5))) >= 0.8
                        )
                        if end_side == kicker_side or kicker_dom or had_dwell:
                            end_lock = t_rec if (end_side == kicker_side or kicker_dom) else max(t + 1, t_rec - 2)
                            flight_team[t:end_lock] = kicker_side

                        if disp >= 14.0 or peak_spd >= 11.0:
                            tau = np.arange(dur, dtype=np.float64) / float(fps)
                            T_sec = dur / float(fps)
                            z_arc = np.minimum(4.0, 0.5 * 9.81 * tau * (T_sec - tau) * 0.35)
                            z_height[t:t_rec] = np.maximum(z_height[t:t_rec], z_arc)
                        if end_side == -kicker_side and kicker_dom:
                            app_start = max(t + 2, t_rec - int(round(0.40 * fps)))
                            pre_rec_suppress[app_start:t_rec] = True
                        t = t_rec
                        continue
        t += 1

    return flight_team, z_height, pre_rec_suppress


def _snap_boundaries_v4(path, offset, f, m, sq, speed, near, side, ev_mask, flight_team):
    """Refined boundary snapping with live-pass rescue, restart touch snap, and turnover alignment."""
    out = path.copy()
    L = len(out)
    fps = m.fps

    for _ in range(2):
        changes = np.flatnonzero(np.diff(out) != 0) + 1
        starts = np.concatenate(([0], changes))
        stops = np.concatenate((changes, [L]))
        for idx in range(1, len(starts) - 1):
            s = int(starts[idx])
            e = int(stops[idx])
            dur = e - s
            st = out[s]
            prev_st = out[s - 1]
            next_st = out[e]
            if st == 2 and prev_st in (0, 1) and next_st in (0, 1) and dur < int(round(2.5 * fps)):
                out[s:e] = prev_st
            elif st in (0, 1) and prev_st == 2 and next_st == 2:
                med_sq = float(np.median(sq[offset + s:offset + e]))
                decel_end = (sq[offset + e - 1] < 1.58) and (sq[offset + e - 1] < sq[offset + s] - 0.5)
                if dur < int(round(2.5 * fps)) or (dur < int(round(5.5 * fps)) and (med_sq < 1.55 or decel_end)):
                    out[s:e] = 2

    max_back_d = int(round(2.5 * fps))
    max_sw = int(round(1.0 * fps))
    max_restart_look = int(round(2.0 * fps))

    changes = np.flatnonzero(np.diff(out) != 0) + 1
    starts = np.concatenate(([0], changes))
    stops = np.concatenate((changes, [L]))

    for idx in range(len(starts)):
        s = int(starts[idx])
        e = int(stops[idx])
        st = out[s]

        # Stoppage begins: snap backward to OOB or deceleration, guarding against erasing live pass flights
        if st == 2 and s > 0 and (e - s) >= int(2.5 * fps):
            lo = max(starts[idx - 1] + int(1.0 * fps), s - max_back_d)
            snap = s
            for t in range(s - 1, lo - 1, -1):
                gi = offset + t
                if m.ball_seen[gi] and flight_team[gi] != 0:
                    break
                if f.ball_oob[gi] or (not m.ball_seen[gi]) or (near[gi] > 3.8 and sq[gi] < 1.85):
                    snap = t
                else:
                    break
            if snap < s:
                out[snap:s] = 2

        # Restart begins (D -> Live): snap to the first kick / throw-in touch
        elif st in (0, 1) and s > 0 and out[s - 1] == 2:
            new_sign = 1.0 if st == 0 else -1.0
            lo = max(starts[idx - 1] + int(1.5 * fps), s - max_restart_look)
            snap = s
            for t in range(s - 1, lo - 1, -1):
                gi = offset + t
                if m.ball_seen[gi] and (not f.ball_oob[gi]) and near[gi] < 1.6 and (ev_mask[gi] or speed[gi] > 3.5):
                    if side[gi] == new_sign:
                        snap = t
                elif not m.ball_seen[gi] or f.ball_oob[gi]:
                    break
            if snap < s:
                out[snap:s] = st

        # Turnover (H <-> A): snap backward to first established proximity
        elif st in (0, 1) and s > 0 and out[s - 1] in (0, 1) and out[s - 1] != st and (e - s) >= int(1.5 * fps):
            new_sign = 1.0 if st == 0 else -1.0
            lo = max(starts[idx - 1] + int(0.8 * fps), s - max_sw)
            snap = s
            for t in range(s - 1, lo - 1, -1):
                gi = offset + t
                if not m.ball_seen[gi]:
                    break
                if side[gi] == -new_sign and near[gi] < 2.0:
                    break
                if side[gi] == new_sign and near[gi] < 1.8:
                    snap = t
            if snap < s:
                out[snap:s] = st

    return out


def pro_v4_cop_hmm(m, f, params=HybridParams, use_tags=True, reliability=None):
    """Pro v4: Controlled Ownership & Propagation (COP-HMM).
    Adds SAOT dwell & launch turnover impulse, 50/50 contested carry cosine synchronization,
    geometric set-piece detection, and refined boundary/restart touch snapping.
    """
    n = m.n
    fps = m.fps
    seen = m.ball_seen
    in_seen = seen & ~f.ball_oob
    speed = np.nan_to_num(f.ball_speed, nan=0.0)
    sq = np.nan_to_num(f.squad_speed, nan=2.5)
    k_look = int(round(2.2 * fps))
    sq_future = np.empty(n)
    sq_future[:-k_look] = sq[k_look:]
    sq_future[-k_look:] = sq[-1]

    if use_tags and reliability is None:
        reliability = tag_reliability(m, f)
    e_team, e_dead = hybrid_evidence(m, f, params, use_tags, reliability)

    with np.errstate(invalid="ignore"):
        near = np.fmin(f.d_home, f.d_away)
        margin = np.abs(f.d_home - f.d_away)
        side = np.where(f.d_home < f.d_away, 1.0, -1.0)

    # 1. Contested carry / dribble velocity vector synchronization
    cos_h, cos_a = _compute_player_ball_sync(m, f)
    contested_carry = in_seen & (near < 1.8) & (margin < 0.75) & (np.maximum(cos_h, cos_a) > 0.65)
    e_team += np.where(contested_carry, 0.75 * (cos_h - cos_a), 0.0)

    ev = direction_events(f, fps)
    ev_mask = np.zeros(n, dtype=bool)
    ev_mask[ev] = True

    # 2. SAOT kick-point and dwell & launch turnover impulse
    flight_team, z_height, pre_rec_suppress = _compute_var_kinematics_v4(m, f, ev)

    # 3. Dead-ball kick-back & unattended rolling ball filter
    dead_ball_kick = in_seen & (speed > 4.0) & (
        ((sq < 1.45) & (sq_future < 1.50) & (near > 2.2)) |
        ((near > 4.0) & (sq_future < 1.42) & (speed < 12.0))
    )
    e_dead += params.W_MOVING * dead_ball_kick

    # 4. Stoppage context detectors & geometric set-piece recognition
    unattended_stop = in_seen & (near > 3.8) & (sq < 1.50) & (sq_future < 1.35) & (flight_team == 0)
    ext_latch = np.zeros(n, dtype=bool)
    on_ext = False
    for t in range(n):
        if f.ball_oob[t]:
            on_ext = True
        elif on_ext and (sq[t] > 1.55 or (in_seen[t] and near[t] < 3.0 and speed[t] > 3.0)):
            on_ext = False
        ext_latch[t] = on_ext

    unseen_stop = np.zeros(n, dtype=bool)
    for a, b in runs_of(~in_seen & f.graded):
        if (b - a) >= int(3.0 * fps) and np.median(f.n_tracked[a:b]) >= 18 and np.median(sq[a:b]) < 1.35:
            unseen_stop[a:b] = True

    sp_wait_mask = np.zeros(n, dtype=bool)
    if use_tags:
        for t_tag in m.tags:
            if t_tag["code"].upper() == "SET PIECES":
                s0 = max(0, t_tag["start_frame"] - 4 * fps)
                s1 = min(n, t_tag["end_frame"])
                sp_wait = (sq[s0:s1] < 1.48) & (sq_future[s0:s1] < 1.55)
                sp_wait_mask[s0:s1] |= sp_wait

    sp_geom = _detect_setpiece_geometry(m, f)
    sp_wait_mask |= sp_geom
    e_dead += np.where(sp_geom, 1.20, 0.0)

    unseen_stop_active = unseen_stop & ((sq < 1.45) | (sq_future < 1.35))
    stop_ctx = unattended_stop | (ext_latch & (sq_future < 1.45)) | unseen_stop_active | sp_wait_mask
    e_dead += np.where(stop_ctx, 1.10, 0.0)

    stop_gate = np.where(sp_wait_mask | unseen_stop_active, 0.15, 1.0)

    # 5. 50/50 duel dampening + 2D-to-3D parabolic height distance & flight protection
    with np.errstate(invalid="ignore"):
        contested = (near[ev] < REACH_M) & np.isfinite(near[ev]) & (margin[ev] < 1.0)
    e_team[ev[contested]] -= (
        side[ev[contested]] * params.W_TOUCH *
        (1.0 - np.clip(margin[ev[contested]] / 1.0, 0.15, 1.0))
    )

    w_sus = max(2, int(round(0.35 * fps)))
    h_cnt = centred_count(in_seen & (f.d_home < 2.2) & (f.d_home < f.d_away), w_sus)
    a_cnt = centred_count(in_seen & (f.d_away < 2.2) & (f.d_away < f.d_home), w_sus)
    t_cnt = np.where(side > 0, h_cnt, a_cnt)
    min_cnt = max(2, int(round(0.24 * fps)))

    near_3d = np.hypot(near, z_height)
    airborne_pass_by = in_seen & (near < REACH_M) & ((near_3d > REACH_M) | pre_rec_suppress)
    fly_by = (in_seen & (near < REACH_M) & (speed > 6.5) & (t_cnt < min_cnt)) | airborne_pass_by
    with np.errstate(invalid="ignore"):
        strength = np.clip(1.0 - near / REACH_M, 0, 1) * np.clip(margin / 2.0, 0, 1)
    e_team -= np.where(fly_by, 0.85 * side * strength * params.W_PROX, 0.0)
    fly_touch = (
        (near[ev] < REACH_M) & np.isfinite(near[ev]) &
        (margin[ev] >= 1.0) & (((speed[ev] > 6.5) & (t_cnt[ev] < min_cnt)) | airborne_pass_by[ev])
    )
    e_team[ev[fly_touch]] -= 0.75 * side[ev[fly_touch]] * params.W_TOUCH

    e_team += 0.95 * flight_team
    flight_gate = np.where(flight_team != 0, 0.25, 1.0)

    near_i = np.fmin(f.d_home_interp, f.d_away_interp)
    e_team += np.where(f.ball_recovered & (near_i < 3.0), 0.75 * f.pitch_control, 0.0)
    e_team += 0.65 * f.pass_corridor

    w_col = np.where(m.ball_seen, 0.25 * flight_gate, 0.65 * stop_gate)
    e_team += w_col * (0.75 * f.team_width_diff + 0.15 * f.team_flow + 0.10 * f.vball_cluster)

    if use_tags:
        tag_ev = params.W_TAG * tag_team_evidence(m, reliability)
        e_team -= np.where(~in_seen, (1.0 - stop_gate) * tag_ev, 0.0)

    e_dead += 1.5 * f.single_oob_waiting
    for a, b in runs_of(f.single_oob_waiting):
        if b - a >= max(2, fps // 3) and b < n:
            e_team[b:min(n, b + 2 * fps)] += 0.8 * float(np.median(f.oob_taker_sign[a:b]))

    labels = np.full(n, "X", dtype="<U1")
    scale = fps / 25.0
    c_sw = params.C_SWITCH * scale
    c_dd = params.C_DEAD * scale
    for _, first, last in m.halves:
        sl = slice(first, last + 1)
        path = viterbi(e_team[sl], e_dead[sl], c_sw, c_dd)
        path = _snap_boundaries_v4(path, first, f, m, sq, speed, near, side, ev_mask, flight_team)
        labels[sl] = LETTERS[path]
    return labels


# -- 8. pro_v5_arima_hmm: ARIMA(p,d,q) Time-Series Forecasting & Innovation Filtering -----------

def _compute_arima_var_kinematics(m, f, ev, ar):
    """ARIMA-enhanced SAOT Kick-Point to Reception-Point Flight Ownership & 3D Parabolic Height.

    Uses causal ARIMA control state (`ctrl_causal`) and ARIMA velocity innovation (`innov_ms`) to:
    1. Detect pressured pass launches where a pressing opponent steps within 1m right as the team
       with established ARIMA control momentum (>= 0.30) kicks a completed pass to a teammate.
    2. Preserve SAOT dwell & launch propagation while suppressing the presser's transient contact.
    """
    n = m.n
    fps = m.fps
    in_seen = m.ball_seen & ~f.ball_oob
    speed = np.nan_to_num(f.ball_speed, nan=0.0)
    ctrl_causal = ar["ctrl_causal"]
    innov_ms = ar["innov_ms"]

    with np.errstate(invalid="ignore"):
        near = np.fmin(f.d_home, f.d_away)
        margin = np.abs(f.d_home - f.d_away)
        side = np.where(f.d_home < f.d_away, 1.0, -1.0)

    ev_mask = np.zeros(n, dtype=bool)
    ev_mask[ev] = True

    k_step = max(2, int(round(0.20 * fps)))
    flight_team = np.zeros(n, dtype=np.float64)
    pressed_launch_suppress = np.zeros(n, dtype=bool)
    pre_rec_suppress = np.zeros(n, dtype=bool)
    z_height = np.zeros(n, dtype=np.float64)

    t = 1
    max_flight = int(round(4.0 * fps))
    min_flight = max(3, int(round(0.35 * fps)))

    while t < n - k_step:
        t_prev = max(0, t - max(1, k_step // 2))
        t_next = min(n - 1, t + k_step)
        pre_k = max(0, t - int(round(0.45 * fps)))
        arima_pre_val = float(np.mean(ctrl_causal[max(0, pre_k - k_step):pre_k + 1])) if pre_k > 0 else 0.0
        clean_prox = in_seen[t] and (near[t] < 1.65) and (margin[t] > 0.8)
        pressed_prox = in_seen[t] and (near[t] < 1.85) and (abs(arima_pre_val) >= 0.30)

        if clean_prox or pressed_prox:
            had_touch_or_ctrl = (
                (speed[t_prev] <= 5.5) or ev_mask[t] or
                ev_mask[max(0, t - 1)] or ev_mask[min(n - 1, t + 1)] or
                (innov_ms[t] >= 3.2)
            )
            accel_launch = (
                in_seen[t_next] and speed[t_next] >= 7.5 and
                (speed[t_next] - speed[t_prev] >= 2.8 or (speed[t_prev] <= 3.5 and speed[t_next] >= 6.5)) and
                near[t_next] > near[t] + 0.4
            )
            if had_touch_or_ctrl and accel_launch:
                kicker_side = side[t] if clean_prox else (1.0 if arima_pre_val > 0 else -1.0)
                t_rec = t + 1
                while t_rec < min(n, t + max_flight):
                    if not in_seen[t_rec]:
                        break
                    if t_rec >= t + k_step:
                        if speed[t_rec] < 4.2 and near[t_rec] < 2.2:
                            break
                        if near[t_rec] < 1.65 and (ev_mask[t_rec] or speed[t_rec] < 6.0):
                            break
                    t_rec += 1

                dur = t_rec - t
                if dur >= min_flight:
                    disp = np.hypot(f.bx[min(n - 1, t_rec - 1)] - f.bx[t],
                                    f.by[min(n - 1, t_rec - 1)] - f.by[t])
                    if disp >= 10.0:
                        w_end = slice(max(t + 1, t_rec - 2), min(n, t_rec + int(round(0.4 * fps)) + 1))
                        end_h = np.nanmedian(f.d_home[w_end])
                        end_a = np.nanmedian(f.d_away[w_end])
                        end_side = 1.0 if end_h < end_a else -1.0
                        peak_spd = float(np.max(speed[t:t_rec]))
                        kicker_dom = (
                            dur >= int(round(1.0 * fps)) and disp >= 16.0 and peak_spd >= 13.0 and
                            float(np.mean(side[t:t_rec] == kicker_side)) >= 0.75
                        )
                        t_dwell_lo = max(0, t - int(round(0.50 * fps)))
                        had_dwell = (
                            clean_prox and
                            t - t_dwell_lo >= max(3, int(round(0.35 * fps))) and
                            float(np.mean((side[t_dwell_lo:t] == kicker_side) & (near[t_dwell_lo:t] < 1.5))) >= 0.8
                        )
                        pre_side = 1.0 if arima_pre_val > 0 else -1.0
                        if abs(arima_pre_val) >= 0.30 and end_side == pre_side and disp >= 14.0:
                            kicker_side = pre_side
                            t_lock_start = max(0, t - int(round(0.6 * fps)))
                            flight_team[t_lock_start:t_rec] = pre_side
                            pressed_launch_suppress[t_lock_start:t_rec] = True
                        elif end_side == kicker_side or kicker_dom or had_dwell:
                            end_lock = t_rec if (end_side == kicker_side or kicker_dom) else max(t + 1, t_rec - 2)
                            flight_team[t:end_lock] = kicker_side

                        if disp >= 14.0 or peak_spd >= 11.0:
                            tau = np.arange(dur, dtype=np.float64) / float(fps)
                            T_sec = dur / float(fps)
                            z_arc = np.minimum(4.0, 0.5 * 9.81 * tau * (T_sec - tau) * 0.35)
                            z_height[t:t_rec] = np.maximum(z_height[t:t_rec], z_arc)
                        if end_side == -kicker_side and kicker_dom:
                            app_start = max(t + 2, t_rec - int(round(0.40 * fps)))
                            pre_rec_suppress[app_start:t_rec] = True
                        t = t_rec
                        continue
        t += 1

    return flight_team, z_height, pre_rec_suppress, pressed_launch_suppress


def _snap_boundaries_arima(path, offset, f, m, sq, speed, near, side, ev_mask, flight_team, ar):
    """ARIMA-informed physical continuity, pre-stoppage deflection filtering, and boundary snapping."""
    out = path.copy()
    L = len(out)
    fps = m.fps
    ctrl_smooth = ar["ctrl_smooth"]

    for _ in range(2):
        changes = np.flatnonzero(np.diff(out) != 0) + 1
        starts = np.concatenate(([0], changes))
        stops = np.concatenate((changes, [L]))
        for idx in range(1, len(starts) - 1):
            s = int(starts[idx])
            e = int(stops[idx])
            dur = e - s
            st = out[s]
            prev_st = out[s - 1]
            next_st = out[e]
            if st == 2 and prev_st in (0, 1) and next_st in (0, 1) and dur < int(round(2.5 * fps)):
                out[s:e] = prev_st
            elif st in (0, 1) and prev_st == 2 and next_st == 2:
                med_sq = float(np.median(sq[offset + s:offset + e]))
                decel_end = (sq[offset + e - 1] < 1.58) and (sq[offset + e - 1] < sq[offset + s] - 0.5)
                if dur < int(round(2.5 * fps)) or (dur < int(round(5.5 * fps)) and (med_sq < 1.55 or decel_end)):
                    out[s:e] = 2
            elif st in (0, 1) and prev_st in (0, 1) and prev_st != st and next_st == 2 and dur < int(round(2.2 * fps)):
                seg_spd = speed[offset + s:offset + e]
                seg_ctrl = ctrl_smooth[offset + s:offset + e]
                st_sign = 1.0 if st == 0 else -1.0
                if float(np.median(seg_spd)) > 5.5 or float(np.max(st_sign * seg_ctrl)) < 0.35:
                    out[s:e] = prev_st

    max_back_d = int(round(2.5 * fps))
    max_sw = int(round(1.0 * fps))
    max_restart_look = int(round(2.0 * fps))

    changes = np.flatnonzero(np.diff(out) != 0) + 1
    starts = np.concatenate(([0], changes))
    stops = np.concatenate((changes, [L]))

    for idx in range(len(starts)):
        s = int(starts[idx])
        e = int(stops[idx])
        st = out[s]

        if st == 2 and s > 0 and (e - s) >= int(2.5 * fps):
            lo = max(starts[idx - 1] + int(1.0 * fps), s - max_back_d)
            snap = s
            for t in range(s - 1, lo - 1, -1):
                gi = offset + t
                if m.ball_seen[gi] and flight_team[gi] != 0:
                    break
                if f.ball_oob[gi] or (not m.ball_seen[gi]) or (near[gi] > 3.8 and sq[gi] < 1.85):
                    snap = t
                else:
                    break
            if snap < s:
                out[snap:s] = 2

        elif st in (0, 1) and s > 0 and out[s - 1] == 2:
            new_sign = 1.0 if st == 0 else -1.0
            lo = max(starts[idx - 1] + int(1.5 * fps), s - max_restart_look)
            snap = s
            for t in range(s - 1, lo - 1, -1):
                gi = offset + t
                if m.ball_seen[gi] and (not f.ball_oob[gi]) and near[gi] < 1.6 and (ev_mask[gi] or speed[gi] > 3.5):
                    if side[gi] == new_sign:
                        snap = t
                elif not m.ball_seen[gi] or f.ball_oob[gi]:
                    break
            if snap < s:
                out[snap:s] = st

        elif st in (0, 1) and s > 0 and out[s - 1] in (0, 1) and out[s - 1] != st and (e - s) >= int(1.5 * fps):
            new_sign = 1.0 if st == 0 else -1.0
            lo = max(starts[idx - 1] + int(0.8 * fps), s - max_sw)
            snap = s
            for t in range(s - 1, lo - 1, -1):
                gi = offset + t
                if not m.ball_seen[gi]:
                    break
                if side[gi] == -new_sign and near[gi] < 2.0:
                    break
                if side[gi] == new_sign and near[gi] < 1.8:
                    snap = t
            if snap < s:
                out[snap:s] = st

    return out


def pro_v5_arima_hmm(m, f, params=HybridParams, use_tags=True, reliability=None):
    """Pro v5: ARIMA(p,d,q) State-Space & Innovation HMM (`pro_v5_arima_hmm`).

    Fits three self-supervised Box-Jenkins ARIMA models (`solution/arima.py`) to each match's
    tracking time series:
    1. Ball Trajectory ARIMA(1, 1, 1): estimates physical drag decay (phi_1) and tracker noise
       (theta_1), imputing missing-ball gaps bidirectionally and detecting 1-step velocity
       innovation impulses ||e_t|| / sigma_e at player touches/interceptions.
    2. Team Control ARIMA(2, 0, 1): provides causal 1-step-ahead possession momentum forecasts
       hat(u)_{t|t-1} to protect pressured pass launches and reject uncontrolled pre-stoppage
       deflections.
    3. Squad Speed ARIMA(2, 1, 1): models collective player acceleration/deceleration waves.
    """
    from .arima import compute_arima_signals

    n = m.n
    fps = m.fps
    seen = m.ball_seen
    in_seen = seen & ~f.ball_oob
    speed = np.nan_to_num(f.ball_speed, nan=0.0)
    sq = np.nan_to_num(f.squad_speed, nan=2.5)
    k_look = int(round(2.2 * fps))
    sq_future = np.empty(n)
    sq_future[:-k_look] = sq[k_look:]
    sq_future[-k_look:] = sq[-1]

    if use_tags and reliability is None:
        reliability = tag_reliability(m, f)
    e_team, e_dead = hybrid_evidence(m, f, params, use_tags, reliability)

    ar = compute_arima_signals(m, f)

    with np.errstate(invalid="ignore"):
        near = np.fmin(f.d_home, f.d_away)
        margin = np.abs(f.d_home - f.d_away)
        side = np.where(f.d_home < f.d_away, 1.0, -1.0)

    cos_h, cos_a = _compute_player_ball_sync(m, f)
    contested_carry = in_seen & (near < 1.8) & (margin < 0.75) & (np.maximum(cos_h, cos_a) > 0.65)
    e_team += np.where(contested_carry, 0.75 * (cos_h - cos_a), 0.0)

    ev = direction_events(f, fps)
    ev_mask = np.zeros(n, dtype=bool)
    ev_mask[ev] = True

    flight_team, z_height, pre_rec_suppress, pressed_launch_suppress = _compute_arima_var_kinematics(m, f, ev, ar)

    dead_ball_kick = in_seen & (speed > 4.0) & (
        ((sq < 1.45) & (sq_future < 1.50) & (near > 2.2)) |
        ((near > 4.0) & (sq_future < 1.42) & (speed < 12.0))
    )
    e_dead += params.W_MOVING * dead_ball_kick

    unattended_stop = in_seen & (near > 3.8) & (sq < 1.50) & (sq_future < 1.35) & (flight_team == 0)
    ext_latch = np.zeros(n, dtype=bool)
    on_ext = False
    for t in range(n):
        if f.ball_oob[t]:
            on_ext = True
        elif on_ext and (sq[t] > 1.55 or (in_seen[t] and near[t] < 3.0 and speed[t] > 3.0)):
            on_ext = False
        ext_latch[t] = on_ext

    unseen_stop = np.zeros(n, dtype=bool)
    for a, b in runs_of(~in_seen & f.graded):
        if (b - a) >= int(3.0 * fps) and np.median(f.n_tracked[a:b]) >= 18 and np.median(sq[a:b]) < 1.35:
            unseen_stop[a:b] = True

    sp_wait_mask = np.zeros(n, dtype=bool)
    if use_tags:
        for t_tag in m.tags:
            if t_tag["code"].upper() == "SET PIECES":
                s0 = max(0, t_tag["start_frame"] - 4 * fps)
                s1 = min(n, t_tag["end_frame"])
                sp_wait = (sq[s0:s1] < 1.48) & (sq_future[s0:s1] < 1.55)
                sp_wait_mask[s0:s1] |= sp_wait

    sp_geom = _detect_setpiece_geometry(m, f)
    sp_wait_mask |= sp_geom
    e_dead += np.where(sp_geom, 1.20, 0.0)

    unseen_stop_active = unseen_stop & ((sq < 1.45) | (sq_future < 1.35))
    stop_ctx = unattended_stop | (ext_latch & (sq_future < 1.45)) | unseen_stop_active | sp_wait_mask
    e_dead += np.where(stop_ctx, 1.10, 0.0)

    stop_gate = np.where(sp_wait_mask | unseen_stop_active, 0.15, 1.0)

    with np.errstate(invalid="ignore"):
        contested = (near[ev] < REACH_M) & np.isfinite(near[ev]) & (margin[ev] < 1.0)
    e_team[ev[contested]] -= (
        side[ev[contested]] * params.W_TOUCH *
        (1.0 - np.clip(margin[ev[contested]] / 1.0, 0.15, 1.0))
    )

    w_sus = max(2, int(round(0.35 * fps)))
    h_cnt = centred_count(in_seen & (f.d_home < 2.2) & (f.d_home < f.d_away), w_sus)
    a_cnt = centred_count(in_seen & (f.d_away < 2.2) & (f.d_away < f.d_home), w_sus)
    t_cnt = np.where(side > 0, h_cnt, a_cnt)
    min_cnt = max(2, int(round(0.24 * fps)))

    near_3d = np.hypot(near, z_height)
    pressed_opp = pressed_launch_suppress & (side != flight_team)
    airborne_pass_by = in_seen & (near < REACH_M) & ((near_3d > REACH_M) | pre_rec_suppress | pressed_opp)
    fly_by = (in_seen & (near < REACH_M) & (speed > 6.5) & (t_cnt < min_cnt)) | airborne_pass_by
    with np.errstate(invalid="ignore"):
        strength = np.clip(1.0 - near / REACH_M, 0, 1) * np.clip(margin / 2.0, 0, 1)
    e_team -= np.where(fly_by, 0.85 * side * strength * params.W_PROX, 0.0)
    fly_touch = (
        (near[ev] < REACH_M) & np.isfinite(near[ev]) &
        (margin[ev] >= 1.0) & (((speed[ev] > 6.5) & (t_cnt[ev] < min_cnt)) | airborne_pass_by[ev])
    )
    e_team[ev[fly_touch]] -= 0.75 * side[ev[fly_touch]] * params.W_TOUCH
    pressed_touch = pressed_opp[ev] & (near[ev] < REACH_M) & np.isfinite(near[ev])
    e_team[ev[pressed_touch]] -= side[ev[pressed_touch]] * params.W_TOUCH

    e_team += 0.95 * flight_team
    flight_gate = np.where(flight_team != 0, 0.25, 1.0)

    # ARIMA-filtered team control momentum on controllable ball frames
    e_team += np.where(in_seen & (speed < 6.5) & (flight_team == 0), 0.25 * ar["ctrl_smooth"], 0.0)

    # ARIMA velocity innovation impulse at extended reach (2.0..2.75m) with large uncontested margin (>2.2m)
    ext_innov = (
        in_seen & (near >= REACH_M) & (near < 2.75) & (margin > 2.2) &
        (ar["innov_ms"] > 3.2) & (flight_team == 0)
    )
    e_team += np.where(ext_innov, 0.90 * side * np.clip(ar["innov_ms"] / 5.0, 0.5, 1.5), 0.0)

    near_i = np.fmin(f.d_home_interp, f.d_away_interp)
    e_team += np.where(f.ball_recovered & (near_i < 3.0), 0.75 * f.pitch_control, 0.0)
    e_team += 0.65 * f.pass_corridor

    w_col = np.where(m.ball_seen, 0.25 * flight_gate, 0.65 * stop_gate)
    e_team += w_col * (0.75 * f.team_width_diff + 0.15 * f.team_flow + 0.10 * f.vball_cluster)

    if use_tags:
        tag_ev = params.W_TAG * tag_team_evidence(m, reliability)
        e_team -= np.where(~in_seen, (1.0 - stop_gate) * tag_ev, 0.0)

    e_dead += 1.5 * f.single_oob_waiting
    for a, b in runs_of(f.single_oob_waiting):
        if b - a >= max(2, fps // 3) and b < n:
            e_team[b:min(n, b + 2 * fps)] += 0.8 * float(np.median(f.oob_taker_sign[a:b]))

    labels = np.full(n, "X", dtype="<U1")
    scale = fps / 25.0
    c_sw = params.C_SWITCH * scale
    c_dd = params.C_DEAD * scale
    for _, first, last in m.halves:
        sl = slice(first, last + 1)
        path = viterbi(e_team[sl], e_dead[sl], c_sw, c_dd)
        path = _snap_boundaries_arima(path, first, f, m, sq, speed, near, side, ev_mask, flight_team, ar)
        labels[sl] = LETTERS[path]
    return labels


# -- 9. pro_v6_tuned_arima_hmm & pro_v7_tuned_arimax_hmm: Per-Match Tuning, Cross-Match Learning & ARIMAX --

def viterbi_dynamic(e_team, e_dead, c_switch_arr, c_dead_arr):
    """3-state Viterbi with time-varying transition penalties c_switch_arr[t] and c_dead_arr[t]."""
    n = len(e_team)
    back = np.zeros((n, 3), dtype=np.int8)
    eh = (0.5 * e_team).tolist()
    ea = (-0.5 * e_team).tolist()
    ed = e_dead.tolist()
    c_sw = c_switch_arr.tolist() if isinstance(c_switch_arr, np.ndarray) else [float(c_switch_arr)] * n
    c_dd = c_dead_arr.tolist() if isinstance(c_dead_arr, np.ndarray) else [float(c_dead_arr)] * n
    sh, sa, sd = eh[0], ea[0], ed[0]
    for t in range(1, n):
        sw = c_sw[t]
        dd = c_dd[t]
        ch = (sh, sa - sw, sd - dd)
        bh = 0 if ch[0] >= ch[1] and ch[0] >= ch[2] else (1 if ch[1] >= ch[2] else 2)
        ca = (sh - sw, sa, sd - dd)
        ba = 1 if ca[1] >= ca[0] and ca[1] >= ca[2] else (0 if ca[0] >= ca[2] else 2)
        cd = (sh - dd, sa - dd, sd)
        bd = 2 if cd[2] >= cd[0] and cd[2] >= cd[1] else (0 if cd[0] >= cd[1] else 1)
        back[t, 0], back[t, 1], back[t, 2] = bh, ba, bd
        sh, sa, sd = ch[bh] + eh[t], ca[ba] + ea[t], cd[bd] + ed[t]
    path = np.zeros(n, dtype=np.int8)
    path[-1] = int(np.argmax([sh, sa, sd]))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def _compute_arima_var_kinematics_v6(m, f, ev, ar):
    """Upgraded SAOT Kick-Point to Reception-Point Flight Ownership with ARIMA Mid-Flight Interception Break."""
    n = m.n
    fps = m.fps
    in_seen = m.ball_seen & ~f.ball_oob
    speed = np.nan_to_num(f.ball_speed, nan=0.0)
    ctrl_causal = ar["ctrl_causal"]
    innov_ms = ar["innov_ms"]

    with np.errstate(invalid="ignore"):
        near = np.fmin(f.d_home, f.d_away)
        margin = np.abs(f.d_home - f.d_away)
        side = np.where(f.d_home < f.d_away, 1.0, -1.0)

    ev_mask = np.zeros(n, dtype=bool)
    ev_mask[ev] = True

    k_step = max(2, int(round(0.20 * fps)))
    flight_team = np.zeros(n, dtype=np.float64)
    pressed_launch_suppress = np.zeros(n, dtype=bool)
    pre_rec_suppress = np.zeros(n, dtype=bool)
    z_height = np.zeros(n, dtype=np.float64)

    t = 1
    max_flight = int(round(4.0 * fps))
    min_flight = max(3, int(round(0.35 * fps)))

    while t < n - k_step:
        t_prev = max(0, t - max(1, k_step // 2))
        t_next = min(n - 1, t + k_step)
        pre_k = max(0, t - int(round(0.45 * fps)))
        arima_pre_val = float(np.mean(ctrl_causal[max(0, pre_k - k_step):pre_k + 1])) if pre_k > 0 else 0.0
        clean_prox = in_seen[t] and (
            ((near[t] < 1.65) and (margin[t] > 0.8)) or
            ((near[t] < 2.60) and (margin[t] > 2.2) and (innov_ms[t] >= 2.8) and (speed[t_prev] <= 4.5))
        )
        pressed_prox = in_seen[t] and (near[t] < 1.85) and (abs(arima_pre_val) >= 0.30)

        if clean_prox or pressed_prox:
            had_touch_or_ctrl = (
                (speed[t_prev] <= 5.5) or ev_mask[t] or
                ev_mask[max(0, t - 1)] or ev_mask[min(n - 1, t + 1)] or
                (innov_ms[t] >= 2.8)
            )
            accel_launch = (
                in_seen[t_next] and speed[t_next] >= 7.5 and
                (speed[t_next] - speed[t_prev] >= 2.8 or (speed[t_prev] <= 3.5 and speed[t_next] >= 6.5)) and
                near[t_next] > near[t] + 0.4
            )
            if had_touch_or_ctrl and accel_launch:
                kicker_side = side[t] if clean_prox else (1.0 if arima_pre_val > 0 else -1.0)
                t_rec = t + 1
                while t_rec < min(n, t + max_flight):
                    if not in_seen[t_rec]:
                        break
                    if t_rec >= t + k_step:
                        if speed[t_rec] < 4.2 and near[t_rec] < 2.2:
                            break
                        if near[t_rec] < 1.65 and (ev_mask[t_rec] or speed[t_rec] < 6.0):
                            break
                        if near[t_rec] < 0.85 and innov_ms[t_rec] >= 1.5:
                            break
                        if speed[t_rec] < 4.5 and near[t_rec] < 2.85 and innov_ms[t_rec] >= 2.5 and margin[t_rec] > 1.5:
                            break
                    t_rec += 1

                dur = t_rec - t
                if dur >= min_flight:
                    disp = np.hypot(f.bx[min(n - 1, t_rec - 1)] - f.bx[t],
                                    f.by[min(n - 1, t_rec - 1)] - f.by[t])
                    if disp >= 10.0:
                        w_end = slice(max(t + 1, t_rec - 2), min(n, t_rec + int(round(0.4 * fps)) + 1))
                        end_h = np.nanmedian(f.d_home[w_end])
                        end_a = np.nanmedian(f.d_away[w_end])
                        end_side = 1.0 if end_h < end_a else -1.0
                        peak_spd = float(np.max(speed[t:t_rec]))
                        kicker_dom = (
                            dur >= int(round(1.0 * fps)) and disp >= 16.0 and peak_spd >= 13.0 and
                            float(np.mean(side[t:t_rec] == kicker_side)) >= 0.75
                        )
                        t_dwell_lo = max(0, t - int(round(0.50 * fps)))
                        had_dwell = (
                            clean_prox and
                            t - t_dwell_lo >= max(3, int(round(0.35 * fps))) and
                            float(np.mean((side[t_dwell_lo:t] == kicker_side) & (near[t_dwell_lo:t] < 1.5))) >= 0.8
                        )
                        pre_side = 1.0 if arima_pre_val > 0 else -1.0
                        if abs(arima_pre_val) >= 0.30 and end_side == pre_side and disp >= 14.0:
                            kicker_side = pre_side
                            t_lock_start = max(0, t - int(round(0.6 * fps)))
                            flight_team[t_lock_start:t_rec] = pre_side
                            pressed_launch_suppress[t_lock_start:t_rec] = True
                        elif end_side == kicker_side or kicker_dom or had_dwell:
                            end_lock = t_rec if (end_side == kicker_side or kicker_dom) else max(t + 1, t_rec - 2)
                            flight_team[t:end_lock] = kicker_side

                        if disp >= 14.0 or peak_spd >= 11.0:
                            tau = np.arange(dur, dtype=np.float64) / float(fps)
                            T_sec = dur / float(fps)
                            z_arc = np.minimum(4.0, 0.5 * 9.81 * tau * (T_sec - tau) * 0.35)
                            z_height[t:t_rec] = np.maximum(z_height[t:t_rec], z_arc)
                        if end_side == -kicker_side and kicker_dom:
                            app_start = max(t + 2, t_rec - int(round(0.40 * fps)))
                            pre_rec_suppress[app_start:t_rec] = True
                        t = t_rec
                        continue
        t += 1

    return flight_team, z_height, pre_rec_suppress, pressed_launch_suppress


def _snap_boundaries_v6(path, offset, f, m, sq, speed, near, side, ev_mask, flight_team, ar):
    """Boundary snapping with post-snap pre-stoppage deflection cleanup and turnover touch alignment."""
    out = path.copy()
    L = len(out)
    fps = m.fps
    ctrl_smooth = ar["ctrl_smooth"]
    innov_ms = ar["innov_ms"]

    for _ in range(2):
        changes = np.flatnonzero(np.diff(out) != 0) + 1
        starts = np.concatenate(([0], changes))
        stops = np.concatenate((changes, [L]))
        for idx in range(1, len(starts) - 1):
            s = int(starts[idx])
            e = int(stops[idx])
            dur = e - s
            st = out[s]
            prev_st = out[s - 1]
            next_st = out[e]
            if st == 2 and prev_st in (0, 1) and next_st in (0, 1) and dur < int(round(2.5 * fps)):
                out[s:e] = prev_st
            elif st in (0, 1) and prev_st == 2 and next_st == 2:
                med_sq = float(np.median(sq[offset + s:offset + e]))
                decel_end = (sq[offset + e - 1] < 1.58) and (sq[offset + e - 1] < sq[offset + s] - 0.5)
                if dur < int(round(2.5 * fps)) or (dur < int(round(5.5 * fps)) and (med_sq < 1.55 or decel_end)):
                    out[s:e] = 2
            elif st in (0, 1) and prev_st in (0, 1) and prev_st != st and next_st == 2 and dur < int(round(2.2 * fps)):
                seg_spd = speed[offset + s:offset + e]
                seg_ctrl = ctrl_smooth[offset + s:offset + e]
                st_sign = 1.0 if st == 0 else -1.0
                if float(np.median(seg_spd)) > 5.5 or float(np.max(st_sign * seg_ctrl)) < 0.35:
                    out[s:e] = prev_st

    max_back_d = int(round(2.5 * fps))
    max_sw = int(round(1.0 * fps))
    max_restart_look = int(round(2.0 * fps))

    changes = np.flatnonzero(np.diff(out) != 0) + 1
    starts = np.concatenate(([0], changes))
    stops = np.concatenate((changes, [L]))

    for idx in range(len(starts)):
        s = int(starts[idx])
        e = int(stops[idx])
        st = out[s]

        if st == 2 and s > 0 and (e - s) >= int(2.5 * fps):
            lo = max(starts[idx - 1] + int(1.0 * fps), s - max_back_d)
            snap = s
            for t in range(s - 1, lo - 1, -1):
                gi = offset + t
                if m.ball_seen[gi] and flight_team[gi] != 0:
                    break
                if f.ball_oob[gi] or (not m.ball_seen[gi]) or (near[gi] > 3.8 and sq[gi] < 1.85):
                    snap = t
                else:
                    break
            if snap < s:
                out[snap:s] = 2

        elif st in (0, 1) and s > 0 and out[s - 1] == 2:
            new_sign = 1.0 if st == 0 else -1.0
            lo = max(starts[idx - 1] + int(1.5 * fps), s - max_restart_look)
            snap = s
            for t in range(s - 1, lo - 1, -1):
                gi = offset + t
                if m.ball_seen[gi] and (not f.ball_oob[gi]) and near[gi] < 1.6 and (ev_mask[gi] or speed[gi] > 3.5 or innov_ms[gi] > 1.8):
                    if side[gi] == new_sign:
                        snap = t
                elif not m.ball_seen[gi] or f.ball_oob[gi]:
                    break
            if snap < s:
                out[snap:s] = st

        elif st in (0, 1) and s > 0 and out[s - 1] in (0, 1) and out[s - 1] != st and (e - s) >= int(1.5 * fps):
            new_sign = 1.0 if st == 0 else -1.0
            lo = max(starts[idx - 1] + int(0.8 * fps), s - max_sw)
            snap = s
            for t in range(s - 1, lo - 1, -1):
                gi = offset + t
                if not m.ball_seen[gi]:
                    break
                if side[gi] == -new_sign and near[gi] < 2.0:
                    break
                if side[gi] == new_sign and near[gi] < 1.8:
                    snap = t
            if snap < s:
                out[snap:s] = st

    # Post-snap cleanup: after D is snapped backward across unseen/OOB frames, any remaining
    # <2.2s high-speed/low-control deflection right before D is absorbed into the prior team spell
    changes = np.flatnonzero(np.diff(out) != 0) + 1
    starts = np.concatenate(([0], changes))
    stops = np.concatenate((changes, [L]))
    for idx in range(1, len(starts) - 1):
        s = int(starts[idx])
        e = int(stops[idx])
        dur = e - s
        st = out[s]
        prev_st = out[s - 1]
        next_st = out[e]
        if st in (0, 1) and prev_st in (0, 1) and prev_st != st and next_st == 2 and dur < int(round(2.2 * fps)):
            seg_spd = speed[offset + s:offset + e]
            seg_ctrl = ctrl_smooth[offset + s:offset + e]
            st_sign = 1.0 if st == 0 else -1.0
            if float(np.median(seg_spd)) > 5.5 or float(np.max(st_sign * seg_ctrl)) < 0.35:
                out[s:e] = prev_st

    return out


def pro_v6_tuned_arima_hmm(m, f, params=HybridParams, use_tags=True, reliability=None):
    """Pro v6: Per-Match Auto-Tuned ARIMA(p,d,q) + 27-Match Supervised Cross-Match Learner (`pro_v6_tuned_arima_hmm`)."""
    from .arima import compute_tuned_arima_signals

    n = m.n
    fps = m.fps
    seen = m.ball_seen
    in_seen = seen & ~f.ball_oob
    speed = np.nan_to_num(f.ball_speed, nan=0.0)
    sq = np.nan_to_num(f.squad_speed, nan=2.5)
    k_look = int(round(2.2 * fps))
    sq_future = np.empty(n)
    sq_future[:-k_look] = sq[k_look:]
    sq_future[-k_look:] = sq[-1]

    if use_tags and reliability is None:
        reliability = tag_reliability(m, f)
    e_team, e_dead = hybrid_evidence(m, f, params, use_tags, reliability)

    ar = compute_tuned_arima_signals(m, f, use_arimax=False)
    learned = ar["learned"]

    with np.errstate(invalid="ignore"):
        near = np.fmin(f.d_home, f.d_away)
        margin = np.abs(f.d_home - f.d_away)
        side = np.where(f.d_home < f.d_away, 1.0, -1.0)

    cos_h, cos_a = _compute_player_ball_sync(m, f)
    contested_carry = in_seen & (near < 1.8) & (margin < 0.75) & (np.maximum(cos_h, cos_a) > 0.65)
    e_team += np.where(contested_carry, 0.75 * (cos_h - cos_a), 0.0)

    ev = direction_events(f, fps)
    ev_mask = np.zeros(n, dtype=bool)
    ev_mask[ev] = True

    flight_team, z_height, pre_rec_suppress, pressed_launch_suppress = _compute_arima_var_kinematics_v6(m, f, ev, ar)

    dead_ball_kick = in_seen & (speed > 4.0) & (
        ((sq < 1.45) & (sq_future < 1.50) & (near > 2.2)) |
        ((near > 4.0) & (sq_future < 1.42) & (speed < 12.0))
    )
    e_dead += params.W_MOVING * dead_ball_kick

    unattended_stop = in_seen & (near > 3.8) & (sq < 1.50) & (sq_future < 1.35) & (flight_team == 0)
    ext_latch = np.zeros(n, dtype=bool)
    on_ext = False
    for t in range(n):
        if f.ball_oob[t]:
            on_ext = True
        elif on_ext and (sq[t] > 1.55 or (in_seen[t] and near[t] < 3.0 and speed[t] > 3.0)):
            on_ext = False
        ext_latch[t] = on_ext

    unseen_stop = np.zeros(n, dtype=bool)
    for a, b in runs_of(~in_seen & f.graded):
        if (b - a) >= int(3.0 * fps) and np.median(f.n_tracked[a:b]) >= 18 and np.median(sq[a:b]) < 1.35:
            unseen_stop[a:b] = True

    sp_wait_mask = np.zeros(n, dtype=bool)
    if use_tags:
        for t_tag in m.tags:
            if t_tag["code"].upper() == "SET PIECES":
                s0 = max(0, t_tag["start_frame"] - 4 * fps)
                s1 = min(n, t_tag["end_frame"])
                sp_wait = (sq[s0:s1] < 1.48) & (sq_future[s0:s1] < 1.55)
                sp_wait_mask[s0:s1] |= sp_wait

    sp_geom = _detect_setpiece_geometry(m, f)
    sp_wait_mask |= sp_geom
    e_dead += np.where(sp_geom, 1.20, 0.0)

    unseen_stop_active = unseen_stop & ((sq < 1.45) | (sq_future < 1.35))
    stop_ctx = unattended_stop | (ext_latch & (sq_future < 1.45)) | unseen_stop_active | sp_wait_mask
    e_dead += np.where(stop_ctx, 1.10, 0.0)
    stop_gate = np.where(sp_wait_mask | unseen_stop_active, 0.15, 1.0)

    with np.errstate(invalid="ignore"):
        contested = (near[ev] < REACH_M) & np.isfinite(near[ev]) & (margin[ev] < 1.0)
    e_team[ev[contested]] -= (
        side[ev[contested]] * params.W_TOUCH *
        (1.0 - np.clip(margin[ev[contested]] / 1.0, 0.15, 1.0))
    )

    w_sus = max(2, int(round(0.35 * fps)))
    h_cnt = centred_count(in_seen & (f.d_home < 2.2) & (f.d_home < f.d_away), w_sus)
    a_cnt = centred_count(in_seen & (f.d_away < 2.2) & (f.d_away < f.d_home), w_sus)
    t_cnt = np.where(side > 0, h_cnt, a_cnt)
    min_cnt = max(2, int(round(0.24 * fps)))

    near_3d = np.hypot(near, z_height)
    pressed_opp = pressed_launch_suppress & (side != flight_team)
    airborne_pass_by = in_seen & (near < REACH_M) & ((near_3d > REACH_M) | pre_rec_suppress | pressed_opp)
    fly_by = (in_seen & (near < REACH_M) & (speed > 6.5) & (t_cnt < min_cnt)) | airborne_pass_by
    with np.errstate(invalid="ignore"):
        strength = np.clip(1.0 - near / REACH_M, 0, 1) * np.clip(margin / 2.0, 0, 1)
    e_team -= np.where(fly_by, 0.85 * side * strength * params.W_PROX, 0.0)
    fly_touch = (
        (near[ev] < REACH_M) & np.isfinite(near[ev]) &
        (margin[ev] >= 1.0) & (((speed[ev] > 6.5) & (t_cnt[ev] < min_cnt)) | airborne_pass_by[ev])
    )
    e_team[ev[fly_touch]] -= 0.75 * side[ev[fly_touch]] * params.W_TOUCH
    pressed_touch = pressed_opp[ev] & (near[ev] < REACH_M) & np.isfinite(near[ev])
    e_team[ev[pressed_touch]] -= side[ev[pressed_touch]] * params.W_TOUCH

    e_team += 0.95 * flight_team
    flight_gate = np.where(flight_team != 0, 0.25, 1.0)

    w_ctrl = float(learned.get("w_ctrl_v6", 0.14))
    e_team += np.where(in_seen & (speed < 6.5) & (flight_team == 0), w_ctrl * ar["ctrl_smooth"], 0.0)

    ext_innov = (
        in_seen & (near >= REACH_M) & (near < 2.75) & (margin > 2.2) &
        (ar["innov_ms"] > 3.2) & (flight_team == 0)
    )
    e_team += np.where(ext_innov, 0.90 * side * np.clip(ar["innov_ms"] / 5.0, 0.5, 1.5), 0.0)

    near_i = np.fmin(f.d_home_interp, f.d_away_interp)
    e_team += np.where(f.ball_recovered & (near_i < 3.0), 0.75 * f.pitch_control, 0.0)
    e_team += 0.65 * f.pass_corridor

    w_col = np.where(m.ball_seen, 0.25 * flight_gate, 0.65 * stop_gate)
    e_team += w_col * (0.75 * f.team_width_diff + 0.15 * f.team_flow + 0.10 * f.vball_cluster)

    if use_tags:
        tag_ev = params.W_TAG * tag_team_evidence(m, reliability)
        e_team -= np.where(~in_seen, (1.0 - stop_gate) * tag_ev, 0.0)

    e_dead += 1.5 * f.single_oob_waiting
    for a, b in runs_of(f.single_oob_waiting):
        if b - a >= max(2, fps // 3) and b < n:
            e_team[b:min(n, b + 2 * fps)] += 0.8 * float(np.median(f.oob_taker_sign[a:b]))

    labels = np.full(n, "X", dtype="<U1")
    scale = fps / 25.0
    c_sw_learned = float(learned.get("c_sw_optical", 6.14) if fps >= 20 else learned.get("c_sw_broadcast", 6.00))
    c_sw = c_sw_learned * scale
    c_dd = params.C_DEAD * scale
    for _, first, last in m.halves:
        sl = slice(first, last + 1)
        path = viterbi(e_team[sl], e_dead[sl], c_sw, c_dd)
        path = _snap_boundaries_v6(path, first, f, m, sq, speed, near, side, ev_mask, flight_team, ar)
        labels[sl] = LETTERS[path]
    return labels


def pro_v7_tuned_arimax_hmm(m, f, params=HybridParams, use_tags=True, reliability=None):
    """Pro v7: Per-Match Auto-Tuned ARIMAX(p,d,q, exog=X) + Dynamic Regime-Switching Viterbi (`pro_v7_tuned_arimax_hmm`)."""
    from .arima import compute_tuned_arima_signals

    n = m.n
    fps = m.fps
    seen = m.ball_seen
    in_seen = seen & ~f.ball_oob
    speed = np.nan_to_num(f.ball_speed, nan=0.0)
    sq = np.nan_to_num(f.squad_speed, nan=2.5)
    k_look = int(round(2.2 * fps))
    sq_future = np.empty(n)
    sq_future[:-k_look] = sq[k_look:]
    sq_future[-k_look:] = sq[-1]

    if use_tags and reliability is None:
        reliability = tag_reliability(m, f)
    e_team, e_dead = hybrid_evidence(m, f, params, use_tags, reliability)

    ar = compute_tuned_arima_signals(m, f, use_arimax=True)
    learned = ar["learned"]

    with np.errstate(invalid="ignore"):
        near = np.fmin(f.d_home, f.d_away)
        margin = np.abs(f.d_home - f.d_away)
        side = np.where(f.d_home < f.d_away, 1.0, -1.0)

    cos_h, cos_a = _compute_player_ball_sync(m, f)
    contested_carry = in_seen & (near < 1.8) & (margin < 0.75) & (np.maximum(cos_h, cos_a) > 0.65)
    e_team += np.where(contested_carry, 0.75 * (cos_h - cos_a), 0.0)

    ev = direction_events(f, fps)
    ev_mask = np.zeros(n, dtype=bool)
    ev_mask[ev] = True

    flight_team, z_height, pre_rec_suppress, pressed_launch_suppress = _compute_arima_var_kinematics_v6(m, f, ev, ar)

    dead_ball_kick = in_seen & (speed > 4.0) & (
        ((sq < 1.45) & (sq_future < 1.50) & (near > 2.2)) |
        ((near > 4.0) & (sq_future < 1.42) & (speed < 12.0))
    )
    e_dead += params.W_MOVING * dead_ball_kick

    unattended_stop = in_seen & (near > 3.8) & (sq < 1.50) & (sq_future < 1.35) & (flight_team == 0)
    ext_latch = np.zeros(n, dtype=bool)
    on_ext = False
    for t in range(n):
        if f.ball_oob[t]:
            on_ext = True
        elif on_ext and (sq[t] > 1.55 or (in_seen[t] and near[t] < 3.0 and speed[t] > 3.0)):
            on_ext = False
        ext_latch[t] = on_ext

    unseen_stop = np.zeros(n, dtype=bool)
    for a, b in runs_of(~in_seen & f.graded):
        if (b - a) >= int(3.0 * fps) and np.median(f.n_tracked[a:b]) >= 18 and np.median(sq[a:b]) < 1.35:
            unseen_stop[a:b] = True

    sp_wait_mask = np.zeros(n, dtype=bool)
    if use_tags:
        for t_tag in m.tags:
            if t_tag["code"].upper() == "SET PIECES":
                s0 = max(0, t_tag["start_frame"] - 4 * fps)
                s1 = min(n, t_tag["end_frame"])
                sp_wait = (sq[s0:s1] < 1.48) & (sq_future[s0:s1] < 1.55)
                sp_wait_mask[s0:s1] |= sp_wait

    sp_geom = _detect_setpiece_geometry(m, f)
    sp_wait_mask |= sp_geom
    e_dead += np.where(sp_geom, 1.20, 0.0)

    unseen_stop_active = unseen_stop & ((sq < 1.45) | (sq_future < 1.35))
    stop_ctx = unattended_stop | (ext_latch & (sq_future < 1.45)) | unseen_stop_active | sp_wait_mask
    e_dead += np.where(stop_ctx, 1.10, 0.0)
    stop_gate = np.where(sp_wait_mask | unseen_stop_active, 0.15, 1.0)

    with np.errstate(invalid="ignore"):
        contested = (near[ev] < REACH_M) & np.isfinite(near[ev]) & (margin[ev] < 1.0)
    e_team[ev[contested]] -= (
        side[ev[contested]] * params.W_TOUCH *
        (1.0 - np.clip(margin[ev[contested]] / 1.0, 0.15, 1.0))
    )

    w_sus = max(2, int(round(0.35 * fps)))
    h_cnt = centred_count(in_seen & (f.d_home < 2.2) & (f.d_home < f.d_away), w_sus)
    a_cnt = centred_count(in_seen & (f.d_away < 2.2) & (f.d_away < f.d_home), w_sus)
    t_cnt = np.where(side > 0, h_cnt, a_cnt)
    min_cnt = max(2, int(round(0.24 * fps)))

    near_3d = np.hypot(near, z_height)
    pressed_opp = pressed_launch_suppress & (side != flight_team)
    airborne_pass_by = in_seen & (near < REACH_M) & ((near_3d > REACH_M) | pre_rec_suppress | pressed_opp)
    fly_by = (in_seen & (near < REACH_M) & (speed > 6.5) & (t_cnt < min_cnt)) | airborne_pass_by
    with np.errstate(invalid="ignore"):
        strength = np.clip(1.0 - near / REACH_M, 0, 1) * np.clip(margin / 2.0, 0, 1)
    e_team -= np.where(fly_by, 0.85 * side * strength * params.W_PROX, 0.0)
    fly_touch = (
        (near[ev] < REACH_M) & np.isfinite(near[ev]) &
        (margin[ev] >= 1.0) & (((speed[ev] > 6.5) & (t_cnt[ev] < min_cnt)) | airborne_pass_by[ev])
    )
    e_team[ev[fly_touch]] -= 0.75 * side[ev[fly_touch]] * params.W_TOUCH
    pressed_touch = pressed_opp[ev] & (near[ev] < REACH_M) & np.isfinite(near[ev])
    e_team[ev[pressed_touch]] -= side[ev[pressed_touch]] * params.W_TOUCH

    scale = fps / 25.0
    c_sw_learned = float(learned.get("c_sw_optical", 6.14) if fps >= 20 else learned.get("c_sw_broadcast", 6.00))
    c_sw_arr = np.full(n, c_sw_learned * scale, dtype=np.float64)
    c_dd_arr = np.full(n, params.C_DEAD * scale, dtype=np.float64)

    # ARIMAX Intercept-and-Launch Contact Realignment & Dynamic Structural Break Gating (on high-rate optical tracking)
    if fps >= 20:
        innov_ms = ar["innov_ms"]
        k_step = max(2, int(round(0.20 * fps)))
        look_w = max(3, int(round(0.45 * fps)))
        dom_w = max(4, int(round(0.65 * fps)))
        for e in ev:
            if in_seen[e] and 1.25 < near[e] < REACH_M:
                hi = min(n - k_step - 1, e + look_w)
                for tc in range(e + 1, hi + 1):
                    if (in_seen[tc] and side[tc] == -side[e] and near[tc] < 1.05 and
                            near[tc] < near[e] - 0.50 and margin[tc] > 0.80 and
                            innov_ms[tc] >= 2.2 and innov_ms[tc] >= innov_ms[e]):
                        if speed[tc + k_step] >= 7.0 and (speed[tc + k_step] - speed[e]) >= 3.5:
                            dom_end = min(n, tc + dom_w)
                            if float(np.mean(side[tc:dom_end] == side[tc])) >= 0.80:
                                e_team[e] -= side[e] * params.W_TOUCH
                                s_new = side[tc]
                                launch_span = slice(max(0, e - 1), min(n, tc + int(round(0.60 * fps))))
                                e_team[launch_span] += s_new * 1.45
                                c_sw_arr[max(0, e - 2):min(n, tc + int(round(0.75 * fps)))] = 0.45 * c_sw_learned * scale
                                break

    e_team += 0.95 * flight_team
    flight_gate = np.where(flight_team != 0, 0.25, 1.0)

    w_ctrl = float(learned.get("w_ctrl_v7", 0.14))
    e_team += np.where(in_seen & (speed < 6.5) & (flight_team == 0), w_ctrl * ar["ctrl_arimax"], 0.0)

    ext_innov = (
        in_seen & (near >= REACH_M) & (near < 2.75) & (margin > 2.2) &
        (ar["innov_ms"] > 3.2) & (flight_team == 0)
    )
    e_team += np.where(ext_innov, 0.90 * side * np.clip(ar["innov_ms"] / 5.0, 0.5, 1.5), 0.0)

    near_i = np.fmin(f.d_home_interp, f.d_away_interp)
    e_team += np.where(f.ball_recovered & (near_i < 3.0), 0.75 * f.pitch_control, 0.0)
    e_team += 0.65 * f.pass_corridor

    w_col = np.where(m.ball_seen, 0.25 * flight_gate, 0.65 * stop_gate)
    e_team += w_col * (0.75 * f.team_width_diff + 0.15 * f.team_flow + 0.10 * f.vball_cluster)

    if use_tags:
        tag_ev = params.W_TAG * tag_team_evidence(m, reliability)
        e_team -= np.where(~in_seen, (1.0 - stop_gate) * tag_ev, 0.0)

    e_dead += 1.5 * f.single_oob_waiting
    for a, b in runs_of(f.single_oob_waiting):
        if b - a >= max(2, fps // 3) and b < n:
            e_team[b:min(n, b + 2 * fps)] += 0.8 * float(np.median(f.oob_taker_sign[a:b]))

    labels = np.full(n, "X", dtype="<U1")
    for _, first, last in m.halves:
        sl = slice(first, last + 1)
        path = viterbi_dynamic(e_team[sl], e_dead[sl], c_sw_arr[sl], c_dd_arr[sl])
        path = _snap_boundaries_v6(path, first, f, m, sq, speed, near, side, ev_mask, flight_team, ar)
        labels[sl] = LETTERS[path]
    return labels


# -- registry -------------------------------------------------------------------------------------

METHOD_INFO = {
    "baseline_raw": ("Provided", "Nearest tracked player to the ball; D where the ball is unseen. "
                     "No smoothing."),
    "baseline_smooth25": ("Provided", "Nearest player, then a +/-1 s majority vote."),
    "baseline_smooth50": ("Provided", "Nearest player, then a +/-2 s majority vote. The published "
                          "floor (full-match spellF1 0.2433)."),
    "baseline_smooth125": ("Provided", "Nearest player, then a +/-5 s majority vote."),
    "user_velocity": ("Your draft", "Kick-off team known; every ball-direction change hands the "
                      "ball to the nearest player; ball over a line or squad below walking pace = "
                      "dead; attacking tags fill long ball-less gaps."),
    "hybrid_no_tags": ("Hybrid (ablation)", "3-state HMM with Smart Tagging inputs removed, "
                       "showing what tracking signals alone achieve."),
    "hybrid_hmm": ("Hybrid", "Your signals (touches, proximity, out of bounds, squad speed, "
                   "tags) as soft per-frame evidence, decoded jointly by a 3-state HMM with "
                   "switch costs (the 'controlled possession' rule)."),
    "pro_soccer_hmm": ("Pro Research", "Adds broadcast cross-camera ball recovery, pass trajectory "
                       "interpolation, Spearman time-to-intercept pitch control, pass-corridor "
                       "lock, 50/50 duel dampening, collective team width/flow/convergence "
                       "(BallRadar), and single-player-off-pitch throw-in/corner detection."),
    "pro_v2_hmm": ("Pro Research v2", "Error-analysis upgrade: filters unattended/dead-ball "
                   "kick-backs via lookahead squad speed, detects in-bounds unseen fouls & "
                   "extended OOB/set-piece waits, protects 2D aerial pass flight against fleeting "
                   "proximity, and applies two-pass physical continuity & boundary snapping."),
    "pro_v3_var_hmm": ("Pro VAR / SAOT v3", "Adds SAOT kick-point (ball release frame) to "
                       "reception-point flight ownership lock, 2D-to-3D parabolic height z(t) "
                       "player-ball Euclidean distance, and pre-reception approach suppression."),
    "pro_v4_cop_hmm": ("Pro COP-HMM v4", "Controlled Ownership & Propagation (COP-HMM): "
                       "adds SAOT dwell & launch turnover impulse, contested carry / dribble cosine "
                       "velocity synchronization, geometric set-piece detection (corners and 6-yd goal "
                       "kicks), and refined boundary snapping (restart touch snapping & live pass flight protection)."),
    "pro_v5_arima_hmm": ("Pro ARIMA-HMM v5", "Box-Jenkins ARIMA(p,d,q) Time-Series & Innovation HMM: "
                         "fits self-supervised statsmodels ARIMA(1,1,1) on ball velocity/position "
                         "(bidirectional gap trajectory forecasting & 1-step touch innovation impulses), "
                         "ARIMA(2,0,1) on continuous team control (causal pre-launch momentum lock for "
                         "pressed passes & pre-stoppage deflection filtering), and ARIMA(2,1,1) on squad speed."),
    "pro_v6_tuned_arima_hmm": ("Pro Tuned-ARIMA v6", "Per-Match Auto-Tuned ARIMA(p,d,q) + 27-Match Supervised "
                               "Cross-Match Learner: searches candidate (p,d,q) orders & touch horizons per match "
                               "via AIC, learns emission & transition scales via 5-fold Out-of-Fold CV across the "
                               "27 external matches, and adds ARIMA mid-flight interception break & post-snap deflection filtering."),
    "pro_v7_tuned_arimax_hmm": ("Pro Tuned-ARIMAX v7", "Per-Match Tuned Box-Jenkins ARIMAX(p,d,q, exog=X) + "
                                "Dynamic Regime-Switching Viterbi: incorporates exogenous physical shock inputs X_t "
                                "(signed uncontested ball-touch velocity innovation shocks & pass-corridor dominance) "
                                "into the autoregressive control state, realigns intercept-and-launch contacts, and "
                                "dynamically discounts HMM switch costs during high-innovation turnovers."),
    "pro_v8a_restart": ("Pro Restart-Edge v8a", "Football Restart Rules & Touchline-Edge Geometry: "
                        "adds the last-touch out-of-bounds restart rule (the team that did not touch the ball "
                        "last before it left the pitch takes the throw-in/corner/goal-kick restart) and a "
                        "touchline-edge waiting detector (<1.5m from touchline while squad walks), preserving "
                        "Alive-vs-Dead log-odds margin invariance."),
    "pro_v8b_hsmm": ("Pro Semi-Markov HSMM v8b", "Hidden Semi-Markov Model (Segmental Minimum-Duration Viterbi): "
                     "replaces first-order geometric duration assumptions with an exact O(N) minimum-duration "
                     "semi-Markov decoder (K_team = 0.6s, K_dead = 1.2s) on top of tuned ARIMAX emissions."),
    "pro_v8c_kalman": ("Pro Kalman-GARCH v8c", "State-Space Constant-Velocity Kalman Filter ARIMA(0,2,2) + "
                       "GARCH(1,1)-Style EWMA Volatility Normalization: estimates per-match measurement noise R "
                       "from second differences, tracks ball state across missing frames natively, and normalizes "
                       "innovation shocks by local EWMA volatility."),
    "pro_v8d_stack": ("Pro ARIMAX-LGBM Stack v8d", "Physics-Protected Cross-Match LightGBM Stacker + ARIMAX Fusion: "
                      "trains a 74-feature multi-scale temporal LightGBM classifier on the 27 external matches "
                      "via 5-fold grouped Out-of-Fold CV and fuses its log-probabilities with v7 ARIMAX & Kalman "
                      "emissions using physical walking-squad stoppage protection and Alive-Dead margin invariance."),
    "pro_v8e_combo": ("Pro Ultimate Ensemble v8e", "Full v8 Synthesis: combines Tuned ARIMAX(p,d,q,X) + "
                      "State-Space Kalman-GARCH ARIMA(0,2,2) + Last-Touch Restart & Touchline-Edge Geometry + "
                      "Physics-Protected 27-Match LightGBM Temporal Stacker + Hidden Semi-Markov Minimum-Duration "
                      "Viterbi & Unseen-Aerial-Pass Guarded Boundary Snapping."),
    "pro_v9a_sensor_gk_invariance": ("Pro Sensor & GK Invariance v9a", "Physical Sensor & Dual-Camera Invariance: "
                                     "enforces goalkeeper penalty-box territorial invariance against post-gap "
                                     "optical tracker ID swaps, removes frozen phantom tracklets (>2.0s zero variance), "
                                     "fuses tactical + broadcast player proximity for touchline & 6-yard-box carriers, "
                                     "and extends pass flight ownership across one-touch teammate redirects."),
    "pro_v9b_unseen_flight_bridge": ("Pro Unseen Flight Bridge v9b", "ARIMA / Kinematic Unseen Aerial Pass Flight "
                                     "Bridge: models missing-ball intervals (~m.ball_seen, 2.5..30 m/s chord speed, "
                                     "dist >= 5m, live squad speed) as continuous ballistic trajectories owned by the "
                                     "kicking team, cancelling mid-flight defensive pitch control and arrival "
                                     "interception switch barriers."),
    "pro_v9c_arimax_kalman": ("Pro ARIMAX + Kalman-GARCH v9c", "Pure Time-Series Synthesis: combines v9a sensor/GK "
                              "invariance and v9b unseen aerial pass flight bridging with State-Space Constant-Velocity "
                              "Kalman Filter ARIMA(0,2,2) + GARCH(1,1) volatility-normalized innovation shocks, "
                              "last-touch OOB restart physics, and touchline-edge waiting detection."),
    "pro_v9d_ultimate_arimax_physics": ("Pro Ultimate ARIMAX-Physics v9d", "Complete v9 Heuristic & Box-Jenkins "
                                        "Time-Series Architecture: combines v9a Dual-Camera & GK Invariance + "
                                        "v9b ARIMA Unseen Flight Bridge + v9c State-Space Kalman-GARCH & Restart "
                                        "Physics + Two-Stage Topological & Contact-Aligned Boundary Snapping "
                                        "(D-Live-D false blip absorption before Live-D-Live bridging, and fast-pass "
                                        "pre-contact turnover guards)."),
}


def run_all(m, f, verbose=True):
    from .v8 import pro_v8a_restart, pro_v8b_hsmm, pro_v8c_kalman, pro_v8d_stack, pro_v8e_combo
    from .v9 import (
        pro_v9a_sensor_gk_invariance,
        pro_v9b_unseen_flight_bridge,
        pro_v9c_arimax_kalman,
        pro_v9d_ultimate_arimax_physics,
    )

    out = {}
    for name, labels in provided_baselines(f, fps=m.fps).items():
        out[name] = labels
    if verbose:
        print("  user_velocity ...")
    out["user_velocity"] = user_velocity(m, f)
    reliability = tag_reliability(m, f)
    if verbose:
        print("  hybrid_hmm ...")
    out["hybrid_no_tags"] = hybrid_hmm(m, f, use_tags=False)
    out["hybrid_hmm"] = hybrid_hmm(m, f, use_tags=True, reliability=reliability)
    if verbose:
        print("  pro_soccer_hmm ...")
    out["pro_soccer_hmm"] = pro_soccer_hmm(m, f, use_tags=True, reliability=reliability)
    if verbose:
        print("  pro_v2_hmm ...")
    out["pro_v2_hmm"] = pro_v2_hmm(m, f, use_tags=True, reliability=reliability)
    if verbose:
        print("  pro_v3_var_hmm ...")
    out["pro_v3_var_hmm"] = pro_v3_var_hmm(m, f, use_tags=True, reliability=reliability)
    if verbose:
        print("  pro_v4_cop_hmm ...")
    out["pro_v4_cop_hmm"] = pro_v4_cop_hmm(m, f, use_tags=True, reliability=reliability)
    if verbose:
        print("  pro_v5_arima_hmm ...")
    out["pro_v5_arima_hmm"] = pro_v5_arima_hmm(m, f, use_tags=True, reliability=reliability)
    if verbose:
        print("  pro_v6_tuned_arima_hmm ...")
    out["pro_v6_tuned_arima_hmm"] = pro_v6_tuned_arima_hmm(m, f, use_tags=True, reliability=reliability)
    if verbose:
        print("  pro_v7_tuned_arimax_hmm ...")
    out["pro_v7_tuned_arimax_hmm"] = pro_v7_tuned_arimax_hmm(m, f, use_tags=True, reliability=reliability)
    if verbose:
        print("  pro_v8a_restart ...")
    out["pro_v8a_restart"] = pro_v8a_restart(m, f, use_tags=True, reliability=reliability)
    if verbose:
        print("  pro_v8b_hsmm ...")
    out["pro_v8b_hsmm"] = pro_v8b_hsmm(m, f, use_tags=True, reliability=reliability)
    if verbose:
        print("  pro_v8c_kalman ...")
    out["pro_v8c_kalman"] = pro_v8c_kalman(m, f, use_tags=True, reliability=reliability)
    if verbose:
        print("  pro_v8d_stack ...")
    out["pro_v8d_stack"] = pro_v8d_stack(m, f, use_tags=True, reliability=reliability)
    if verbose:
        print("  pro_v8e_combo ...")
    out["pro_v8e_combo"] = pro_v8e_combo(m, f, use_tags=True, reliability=reliability)
    if verbose:
        print("  pro_v9a_sensor_gk_invariance ...")
    out["pro_v9a_sensor_gk_invariance"] = pro_v9a_sensor_gk_invariance(m, f, use_tags=True, reliability=reliability)
    if verbose:
        print("  pro_v9b_unseen_flight_bridge ...")
    out["pro_v9b_unseen_flight_bridge"] = pro_v9b_unseen_flight_bridge(m, f, use_tags=True, reliability=reliability)
    if verbose:
        print("  pro_v9c_arimax_kalman ...")
    out["pro_v9c_arimax_kalman"] = pro_v9c_arimax_kalman(m, f, use_tags=True, reliability=reliability)
    if verbose:
        print("  pro_v9d_ultimate_arimax_physics ...")
    out["pro_v9d_ultimate_arimax_physics"] = pro_v9d_ultimate_arimax_physics(m, f, use_tags=True, reliability=reliability)
    return out, reliability
