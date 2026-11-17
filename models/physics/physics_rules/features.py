"""Per-frame features shared by every method. All vectorised, all label-free.

Nothing here reads a ground truth. Thresholds are physical (walking pace, a player's reach, the
pitch lines, reaction time, max sprint speed), not fitted -- see the constants below and the
reasoning beside each one.
"""

import os
import warnings

import numpy as np

from .data import (AWAY, CACHE_DIR, GAMES, HOME, _parse_raw, _team_of_slots, anchor_video_frame,
                   asset_paths, source_frames)

# -- physical constants (not tuned) -------------------------------------------------------------
REACH_M = 2.0            # a player can play a ball within ~2 m in one or two steps
WALKING_MS = 1.4         # median squad speed below brisk walking pace => play is probably stopped
OOB_MARGIN_M = 1.0       # tracking/homography noise: only call the ball out once clearly beyond
STATIONARY_M = 0.5       # a ball moving less than this over one second is placed / held
FLIGHT_MS = 12.0         # faster than this the ball is struck, not carried
SMOOTH_BALL = 5          # frames, ball position smoothing before differencing (0.2 s at 25 fps)
MAX_PASS_MS = 28.0       # max plausible average ball speed across a short interpolated dropout
SPRINT_MS = 7.0          # typical player intercept speed (m/s) for Spearman pitch control
REACT_S = 0.20           # player reaction time (s) before accelerating toward ball trajectory


def centred_mean(x, window):
    """NaN-aware centred rolling mean over +/- window frames, O(n) with cumulative sums."""
    x = np.asarray(x, dtype=np.float64)
    ok = ~np.isnan(x)
    c_val = np.concatenate(([0.0], np.cumsum(np.where(ok, x, 0.0))))
    c_cnt = np.concatenate(([0], np.cumsum(ok)))
    idx = np.arange(len(x))
    lo = np.maximum(0, idx - window)
    hi = np.minimum(len(x), idx + window + 1)
    cnt = c_cnt[hi] - c_cnt[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (c_val[hi] - c_val[lo]) / cnt
    out[cnt == 0] = np.nan
    return out, cnt


def centred_count(mask, window):
    m = np.asarray(mask, dtype=np.int64)
    c = np.concatenate(([0], np.cumsum(m)))
    idx = np.arange(len(m))
    return c[np.minimum(len(m), idx + window + 1)] - c[np.maximum(0, idx - window)]


def runs_of(mask):
    """[(start, stop)] half-open runs where mask is True."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def _load_broadcast_ball(m):
    """For BAR-ATH, load or cache the optional broadcast ATD and align it per half to tactical."""
    if m.game not in GAMES:
        return None
    try:
        paths = asset_paths(m.game, source="broadcast")
    except Exception:
        return None
    if not (os.path.isfile(paths.get("atd_meta", "")) and os.path.isfile(paths.get("atd_raw", ""))):
        return None

    cache_b = os.path.join(CACHE_DIR, f"{m.game}_broadcast_atd.npz")
    if os.path.isfile(cache_b):
        blob = np.load(cache_b)
        pl_b, ball_b = blob["players"], blob["ball"]
    else:
        team_b, _ = _team_of_slots(paths["atd_meta"])
        nf_b = source_frames("broadcast", m.game)
        pl_b, ball_b = _parse_raw(paths["atd_raw"], nf_b, len(team_b))
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.savez_compressed(cache_b, players=pl_b, ball=ball_b, team=team_b)

    bx_b = ball_b[:, 0].astype(np.float64) * m.length
    by_b = ball_b[:, 1].astype(np.float64) * m.width
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        cx_t = np.nanmean(m.px, axis=1)
        cy_t = np.nanmean(m.py, axis=1)
        cx_b = np.nanmean(pl_b[:, :, 0].astype(np.float64) * m.length, axis=1)
        cy_b = np.nanmean(pl_b[:, :, 1].astype(np.float64) * m.width, axis=1)

    bx_out = np.full(m.n, np.nan)
    by_out = np.full(m.n, np.nan)
    bx_t, by_t = m.ball[:, 0], m.ball[:, 1]

    # Automatically find the frame offset (t_tactical - t_broadcast) for each half via ball alignment
    for half, f0, f1 in m.halves:
        t_sub = np.arange(f0 + 2000, min(f1, f0 + 18000), 5)
        best_off, best_mae = 0, 999.0
        for off in range(-2000, 35000, 5):
            b_idx = t_sub - off
            ok = (b_idx >= 0) & (b_idx < len(bx_b))
            ti, bi = t_sub[ok], b_idx[ok]
            both = (~np.isnan(bx_t[ti])) & (~np.isnan(bx_b[bi]))
            if both.sum() > 150:
                mae = float(np.median(np.abs(bx_t[ti[both]] - bx_b[bi[both]]) +
                                      np.abs(by_t[ti[both]] - by_b[bi[both]])))
                if mae < best_mae:
                    best_mae, best_off = mae, off
        for off in range(best_off - 6, best_off + 7):
            b_idx = t_sub - off
            ok = (b_idx >= 0) & (b_idx < len(bx_b))
            ti, bi = t_sub[ok], b_idx[ok]
            both = (~np.isnan(bx_t[ti])) & (~np.isnan(bx_b[bi]))
            if both.sum() > 150:
                mae = float(np.median(np.hypot(bx_t[ti[both]] - bx_b[bi[both]],
                                               by_t[ti[both]] - by_b[bi[both]])))
                if mae < best_mae:
                    best_mae, best_off = mae, off

        t_all = np.arange(f0, f1 + 1)
        b_all = t_all - best_off
        ok = (b_all >= 0) & (b_all < len(bx_b))
        t_all, b_all = t_all[ok], b_all[ok]
        # Verify broadcast is showing live play (not a replay) via 22-player centroid agreement
        live = np.hypot(cx_t[t_all] - cx_b[b_all], cy_t[t_all] - cy_b[b_all]) < 8.0
        in_pitch = ((bx_b[b_all] >= 0.5) & (bx_b[b_all] <= m.length - 0.5) &
                    (by_b[b_all] >= 0.5) & (by_b[b_all] <= m.width - 0.5))
        valid = (~np.isnan(bx_b[b_all])) & live & in_pitch
        bx_out[t_all[valid]] = bx_b[b_all[valid]]
        by_out[t_all[valid]] = by_b[b_all[valid]]
    return bx_out, by_out


class Features(object):
    pass


def compute(m):
    f = Features()
    n = m.n
    seen = m.ball_seen
    bx, by = m.ball[:, 0], m.ball[:, 1]
    home_cols = m.team == HOME

    # Ball position smoothed only across seen frames, never invented across a gap.
    smooth_half = max(1, int(round(0.1 * m.fps)))  # 0.1 s half-window (2 frames at 25 fps)
    sx, _ = centred_mean(bx, smooth_half)
    sy, _ = centred_mean(by, smooth_half)
    sx[~seen], sy[~seen] = np.nan, np.nan
    f.bx, f.by = sx, sy

    # Ball velocity: centred difference over 2 frames (m/s). NaN unless both neighbours are seen.
    vx = np.full(n, np.nan)
    vy = np.full(n, np.nan)
    vx[1:-1] = (sx[2:] - sx[:-2]) * m.fps / 2.0
    vy[1:-1] = (sy[2:] - sy[:-2]) * m.fps / 2.0
    f.vx, f.vy = vx, vy
    f.ball_speed = np.hypot(vx, vy)

    # Nearest player of each team to the ball.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        dist = np.hypot(m.px - sx[:, None], m.py - sy[:, None])
        f.d_home = np.nanmin(np.where(home_cols[None, :], dist, np.nan), axis=1)
        f.d_away = np.nanmin(np.where(~home_cols[None, :], dist, np.nan), axis=1)
        dist_inf = np.where(np.isnan(dist), np.inf, dist)
        nearest_slot = np.argmin(dist_inf, axis=1)
        any_player = np.isfinite(dist_inf).any(axis=1)
        ok = seen & any_player
        f.nearest_team = np.where(ok, m.team[nearest_slot], -1).astype(np.int8)
        f.nearest_dist = np.where(ok, dist_inf[np.arange(n), nearest_slot], np.nan)
        del dist, dist_inf

        # Plain nearest-player team on the RAW ball, exactly the provided baseline's step 1.
        raw = np.hypot(m.px - bx[:, None], m.py - by[:, None])
        raw = np.where(np.isnan(raw), np.inf, raw)
        slot = np.argmin(raw, axis=1)
        ok = seen & np.isfinite(raw).any(axis=1)
        f.nearest_team_raw = np.where(ok, m.team[slot], -1).astype(np.int8)
        del raw

        # Collective speed: median tracked-player speed, smoothed over +/- 0.5 s.
        tracked = ~np.isnan(m.px)
        med = np.nanmedian(np.where(tracked, m.pspeed, np.nan), axis=1)
    f.squad_speed, _ = centred_mean(med, m.fps // 2)
    f.n_tracked = tracked.sum(axis=1)

    # Ball beyond the lines (clearly).
    f.ball_oob = seen & ((bx < -OOB_MARGIN_M) | (bx > m.length + OOB_MARGIN_M) |
                         (by < -OOB_MARGIN_M) | (by > m.width + OOB_MARGIN_M))
    # A player standing well outside a touchline or goal line: thrower, corner taker.
    with np.errstate(invalid="ignore"):
        outside = ((m.px < -1.5) | (m.px > m.length + 1.5) |
                   (m.py < -1.5) | (m.py > m.width + 1.5))
    f.player_oob = outside.sum(axis=1) > 0

    # Ball stationary for a full second (placed for a restart, or held still).
    w = m.fps // 2
    mx, cnt = centred_mean(sx, w)
    my, _ = centred_mean(sy, w)
    with np.errstate(invalid="ignore"):
        dev = np.hypot(sx - mx, sy - my)
    max_dev, _ = centred_mean(dev, w)  # mean deviation, robust to a single jittery frame
    f.ball_stationary = (cnt >= int(0.8 * (2 * w + 1))) & (max_dev < STATIONARY_M / 2.0)

    # Pre-kickoff dead-ball frames at the start of a half (e.g. BAR-ATH 2nd half 110808..113741).
    f.pre_kickoff = np.zeros(n, dtype=bool)
    if m.game in GAMES:
        g_info = GAMES[m.game]
        for half, first, last in m.halves:
            key = "kickoff" if half == 1 else "kickoff_2nd"
            if key in g_info:
                ko_frame = anchor_video_frame(g_info[key])
                if ko_frame - first > 5 * m.fps:
                    f.pre_kickoff[first:min(last + 1, ko_frame + 1)] = True

    f.graded = m.graded_mask()

    # -- Pro Research Features (Broadcast Fusion, Trajectory Interp, Pitch Control, Team Shape) ---

    # 1. Broadcast cross-camera ball recovery + physics-aware linear pass interpolation
    bx_f, by_f = sx.copy(), sy.copy()
    seen_f = seen.copy()
    bcast = _load_broadcast_ball(m)
    if bcast is not None:
        bx_b, by_b = bcast
        fill = (~seen_f) & (~np.isnan(bx_b)) & f.graded
        bx_f[fill] = bx_b[fill]
        by_f[fill] = by_b[fill]
        seen_f[fill] = True

    bx_i, by_i = bx_f.copy(), by_f.copy()
    interp_mask = np.zeros(n, dtype=bool)
    max_gap = int(2.0 * m.fps)
    for a, b in runs_of((~seen_f) & f.graded):
        if 0 < (b - a) <= max_gap and a > 0 and b < n and seen_f[a - 1] and seen_f[b]:
            if not f.ball_oob[a - 1] and not f.ball_oob[b]:
                dist_gap = np.hypot(bx_f[b] - bx_f[a - 1], by_f[b] - by_f[a - 1])
                spd_gap = dist_gap * m.fps / float(b - a + 1)
                if 0.8 <= spd_gap <= MAX_PASS_MS:
                    weights = np.linspace(0.0, 1.0, b - a + 2)[1:-1]
                    bx_i[a:b] = bx_f[a - 1] + weights * (bx_f[b] - bx_f[a - 1])
                    by_i[a:b] = by_f[a - 1] + weights * (by_f[b] - by_f[a - 1])
                    interp_mask[a:b] = True

    f.bx_interp, f.by_interp = bx_i, by_i
    f.ball_recovered = (~seen) & (seen_f | interp_mask)

    # 2. Spearman Kinematic Time-to-Intercept (Pitch Control) & Pass Departure-to-Arrival Corridor
    dt_p = max(1, int(round(0.2 * m.fps)))
    vpx = np.zeros_like(m.px)
    vpy = np.zeros_like(m.py)
    vpx[dt_p:] = (m.px[dt_p:] - m.px[:-dt_p]) * (m.fps / float(dt_p))
    vpy[dt_p:] = (m.py[dt_p:] - m.py[:-dt_p]) * (m.fps / float(dt_p))
    bad_v = (~np.isfinite(vpx)) | (np.hypot(vpx, vpy) > 12.0)  # filter tracklet slot swaps
    vpx[bad_v] = 0.0
    vpy[bad_v] = 0.0

    vbx = np.zeros(n)
    vby = np.zeros(n)
    vbx[1:-1] = (bx_i[2:] - bx_i[:-2]) * m.fps / 2.0
    vby[1:-1] = (by_i[2:] - by_i[:-2]) * m.fps / 2.0
    vbx = np.nan_to_num(vbx, nan=0.0)
    vby = np.nan_to_num(vby, nan=0.0)
    spd_i = np.hypot(vbx, vby)

    bx_proj = bx_i + np.where(spd_i < 15.0, 0.12 * vbx, 0.0)
    by_proj = by_i + np.where(spd_i < 15.0, 0.12 * vby, 0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        px_proj = m.px + REACT_S * vpx
        py_proj = m.py + REACT_S * vpy
        t_int = REACT_S + np.hypot(px_proj - bx_proj[:, None], py_proj - by_proj[:, None]) / SPRINT_MS
        t_home = np.nanmin(np.where(home_cols[None, :], t_int, np.nan), axis=1)
        t_away = np.nanmin(np.where(~home_cols[None, :], t_int, np.nan), axis=1)
        d_home_i = np.nanmin(np.where(home_cols[None, :],
                                      np.hypot(m.px - bx_i[:, None], m.py - by_i[:, None]), np.nan), axis=1)
        d_away_i = np.nanmin(np.where(~home_cols[None, :],
                                      np.hypot(m.px - bx_i[:, None], m.py - by_i[:, None]), np.nan), axis=1)

    f.d_home_interp = d_home_i
    f.d_away_interp = d_away_i
    f.pitch_control = np.nan_to_num(np.tanh((t_away - t_home) / 0.30), nan=0.0)

    # Pass corridor across unseen gaps: same team controlling at departure and arrival
    corridor = np.zeros(n)
    for a, b in runs_of((~seen) & f.graded):
        if 1 <= (b - a) <= int(2.5 * m.fps) and a > 0 and b < n:
            th0 = (1 if d_home_i[a - 1] < min(d_away_i[a - 1], 2.2)
                   else (-1 if d_away_i[a - 1] < min(d_home_i[a - 1], 2.2) else 0))
            th1 = (1 if d_home_i[b] < min(d_away_i[b], 2.2)
                   else (-1 if d_away_i[b] < min(d_home_i[b], 2.2) else 0))
            if th0 != 0 and th0 == th1:
                corridor[a:b] = th0
    f.pass_corridor = corridor

    # 3. Ball-Free Collective Team Features (Width Expansion, Attacking Flow, Convergence Cluster)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        std_h = np.nanstd(np.where(home_cols[None, :], m.py, np.nan), axis=1)
        std_a = np.nanstd(np.where(~home_cols[None, :], m.py, np.nan), axis=1)
        cx = np.nanmean(m.px, axis=1)
        # Fast-player convergence cluster ("virtual ball" zone, BallRadar concept)
        w_spd = np.square(np.clip(np.nan_to_num(m.pspeed, nan=0.0), 0.0, 9.0)) + 0.01
        w_spd[np.isnan(m.px)] = 0.0
        w_sum = np.maximum(w_spd.sum(axis=1), 1e-6)
        vbx_c = np.nansum(m.px * w_spd, axis=1) / w_sum
        vby_c = np.nansum(m.py * w_spd, axis=1) / w_sum
        vd_h = np.nanmin(np.where(home_cols[None, :],
                                  np.hypot(m.px - vbx_c[:, None], m.py - vby_c[:, None]), np.nan), axis=1)
        vd_a = np.nanmin(np.where(~home_cols[None, :],
                                  np.hypot(m.px - vbx_c[:, None], m.py - vby_c[:, None]), np.nan), axis=1)

    v_cluster, _ = centred_mean(np.tanh((vd_a - vd_h) / 2.5), max(1, m.fps // 2))
    f.vball_cluster = np.nan_to_num(v_cluster, nan=0.0)

    w_diff, _ = centred_mean(np.tanh((std_h - std_a) / 3.0), m.fps)
    f.team_width_diff = np.nan_to_num(w_diff, nan=0.0)

    cx_sm, _ = centred_mean(cx, m.fps)
    vx_c = np.zeros(n)
    k = m.fps
    if 2 * k < n:
        vx_c[k:-k] = (cx_sm[2 * k:] - cx_sm[:-2 * k]) / 2.0
    flow = np.zeros(n)
    for _, f0, f1 in m.halves:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            hx = np.nanmean(m.px[f0:f1 + 1, home_cols])
            ax = np.nanmean(m.px[f0:f1 + 1, ~home_cols])
        adir = 1.0 if hx < ax else -1.0
        flow[f0:f1 + 1] = np.tanh(adir * vx_c[f0:f1 + 1] / 2.0)
    f.team_flow = np.nan_to_num(flow, nan=0.0)

    # 4. Single Player Off-Pitch = Throw-In or Corner Kick Setup (dead ball + restart taker team)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out_slow = (((m.px < -1.0) | (m.px > m.length + 1.0) |
                     (m.py < -1.0) | (m.py > m.width + 1.0)) &
                    (np.nan_to_num(m.pspeed, nan=0.0) < 2.8))
    single_oob = (out_slow.sum(axis=1) == 1) & f.graded
    oob_slot = np.argmax(out_slow, axis=1)
    idx = np.arange(n)
    ox = np.where(single_oob, m.px[idx, oob_slot], np.nan)
    oy = np.where(single_oob, m.py[idx, oob_slot], np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        d_ball_oob = np.hypot(sx - ox, sy - oy)
    in_seen = seen & ~f.ball_oob
    f.single_oob_waiting = single_oob & (~in_seen | (d_ball_oob < 4.0))
    f.oob_taker_sign = np.where(m.team[oob_slot] == HOME, 1.0, -1.0)

    return f


# -- Smart Tagging, turned into per-frame evidence ----------------------------------------------

PHASE_CODES = (
    "BUILD UP", "PROGRESSION", "ATTACKING TRANSITION", "LONG BALL", "BALL IN FINAL THIRD",
    "CREATING CHANCES", "BALL IN THE BOX", "PLAYERS IN THE BOX",
    "DEFENDING IN ATTACKING THIRD", "DEFENDING IN MIDDLE THIRD", "DEFENDING IN DEFENSIVE THIRD",
    "DEFENSIVE TRANSITION",
)


def team_code(m, name):
    if name is None:
        return None
    if name.strip().lower() == m.team_names[0].strip().lower() or name == "Barcelona":
        return HOME
    if name.strip().lower() == m.team_names[1].strip().lower() or name.startswith("Atl"):
        return AWAY
    return None


def tag_reliability(m, f):
    """Self-supervised: for each phase code, how often is the TAGGED team nearest the ball?

    Computed over the whole match from the tracking alone -- no ground truth involved. A value
    near 1 means the tag names the team in possession, near 0 means it names the defending team,
    near 0.5 means it says nothing about possession (PLAYERS IN THE BOX turns out to be one).
    Returns {code: (n_instances, fraction, weight)} with weight = 2 * fraction - 1 in [-1, 1].
    """
    stats = {}
    near_home = f.nearest_team == HOME
    valid = f.nearest_team >= 0
    for t in m.tags:
        code = t["code"].upper()
        team = team_code(m, t["tags"].get("Team"))
        if code not in PHASE_CODES or team is None:
            continue
        sl = slice(max(0, t["start_frame"]), min(m.n, t["end_frame"]))
        ok = valid[sl]
        if not ok.any():
            continue
        agree = (near_home[sl][ok] == (team == HOME))
        s = stats.setdefault(code, [0, 0, 0])
        s[0] += 1
        s[1] += int(agree.sum())
        s[2] += int(ok.sum())
    out = {}
    for code, (k, agree, total) in stats.items():
        frac = agree / float(total) if total else 0.5
        out[code] = (k, frac, 2.0 * frac - 1.0)
    return out


def tag_team_evidence(m, reliability, codes=None, min_abs_weight=0.15):
    """Per-frame signed evidence for HOME (+) vs AWAY (-) from the phase tags.

    Each active tag adds its reliability weight towards the team it implies. Tags whose
    reliability is within +/- min_abs_weight of zero are ignored as uninformative.
    """
    ev = np.zeros(m.n)
    for t in m.tags:
        code = t["code"].upper()
        if codes is not None and code not in codes:
            continue
        if code not in reliability:
            continue
        team = team_code(m, t["tags"].get("Team"))
        if team is None:
            continue
        w = reliability[code][2]
        if abs(w) < min_abs_weight:
            continue
        sign = 1.0 if team == HOME else -1.0
        ev[max(0, t["start_frame"]):min(m.n, t["end_frame"])] += sign * w
    return ev


def set_piece_starts(m):
    """[(frame, team or None, type)] for every SET PIECES tag."""
    out = []
    for t in m.tags:
        if t["code"].upper() == "SET PIECES":
            out.append((t["start_frame"], team_code(m, t["tags"].get("Team")),
                        t["tags"].get("Type")))
    return out
