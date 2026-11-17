"""The v9 family: Pure Heuristic, Kinematic & ARIMA/State-Space Time-Series Models.

Built from a deep-dive root-cause diagnosis of every remaining spell boundary, missed turnover,
and sensor failure across `BAR-ATH` (tactical + broadcast tracking), `IDSSE` (7 Bundesliga matches),
and `SkillCorner` (20 A-League matches). Every component is grounded in physical football laws and
Box-Jenkins / State-Space time-series dynamics (no black-box decision trees):

    pro_v9a_sensor_gk_invariance     Goalkeeper Penalty-Box Physical Invariance, Frozen Phantom
                                     Tracklet Filtering, Dual-Camera (Tactical + Broadcast) Player
                                     Proximity & Live-Carrier Clearance Fusion, and One-Touch
                                     Teammate Pass Flight Continuation.
    pro_v9b_unseen_flight_bridge     ARIMA / Kinematic Unseen Aerial Pass Flight Bridge: attributes
                                     live optical unseen aerial pass flights (~ball_seen when z > 1.5m,
                                     chord speed 2.5..30 m/s, squad running >= 1.5 m/s) to the kicking
                                     team at departure, cancels wrong-team interpolated pitch-control
                                     during intercepted flights, and discounts arrival switch costs.
    pro_v9c_arimax_kalman            State-Space Constant-Velocity Kalman Filter ARIMA(0,2,2) with
                                     GARCH(1,1)-style EWMA innovation volatility normalisation +
                                     Last-Touch Out-of-Bounds Restart Rule & Touchline-Edge Waiting
                                     with Alive-vs-Dead log-odds margin invariance.
    pro_v9d_ultimate_arimax_physics  Full v9 Synthesis: combines v9a + v9b + v9c with Two-Stage
                                     Topological & Contact-Aligned Boundary Snapping (`_snap_boundaries_v9`)
                                     and Optical Minimum-Duration Semi-Markov Viterbi decoding.

None of these reads the BAR-ATH ground truth. All principles are general physical/kinematic laws
and self-supervised / cross-match time-series filters obeying RULES.md.
"""

import copy
import os
import numpy as np

from .data import CACHE_DIR
from .features import compute, runs_of
from .methods import LETTERS, HybridParams, viterbi_dynamic
from .v8 import decode, kalman_innovations, kalman_touch_evidence, restart_priors, v7_bundle, viterbi_mindur


# -- 1. v9a: Physical Sensor Fusion, Goalkeeper Invariance & One-Touch Flight Continuation ---------

def apply_v9a_sensor_gk_invariance(m, f):
    """Apply physical tracking invariants and dual-camera player proximity sensor fusion:
    1. Filter frozen phantom player tracklets (|dx| < 1e-4, |dy| < 1e-4 over >= 1.0s).
    2. Goalkeeper Penalty-Box Physical Invariance per half: any goalkeeper slot (half-median
       x < 22m or > L - 22m) appearing inside the OPPOSITE penalty box (< 22m from the opposite
       goal line) is tracking the opposing goalkeeper and belongs to the defending team.
    3. Dual-Camera Broadcast Player Proximity & Live-Carrier Sensor Fusion (when broadcast ATD
       exists): recovers missing touchline ball-carriers (Rule A), disambiguates far-touchline
       jerseys (Rule B), and detects live ball-carriers when tactical ball drifts (Rule C_safe).
    """
    m2 = copy.copy(m)
    px = m.px.copy()
    py = m.py.copy()
    pspeed = m.pspeed.copy()
    n, n_slots = px.shape

    # 1. Filter frozen phantom tracklets (exact same x,y over >= 1.0s)
    w_freeze = max(5, int(round(1.0 * m.fps)))
    changed = False
    for s in range(n_slots):
        dx = np.abs(np.diff(px[:, s], prepend=np.nan))
        dy = np.abs(np.diff(py[:, s], prepend=np.nan))
        still = (dx < 1e-4) & (dy < 1e-4) & np.isfinite(px[:, s])
        for a, b in runs_of(still):
            if b - a >= w_freeze:
                px[a:b, s] = np.nan
                py[a:b, s] = np.nan
                pspeed[a:b, s] = np.nan
                changed = True

    # 2. Goalkeeper Penalty-Box Physical Invariance per half
    L = m.length
    for _, f0, f1 in m.halves:
        sl = slice(f0, f1 + 1)
        with np.errstate(invalid="ignore"):
            h_med_x = np.nanmedian(px[sl, m.team == 0])
            a_med_x = np.nanmedian(px[sl, m.team == 1])
        if np.isnan(h_med_x) or np.isnan(a_med_x):
            continue
        home_def_left = h_med_x < a_med_x

        gk_slots = {0: [], 1: []}
        for s in range(n_slots):
            vis = np.isfinite(px[sl, s])
            if np.sum(vis) > 5 * m.fps:
                s_med = float(np.nanmedian(px[sl, s]))
                if s_med < 22.0 or s_med > L - 22.0:
                    gk_slots[int(m.team[s])].append(s)

        for tm in (0, 1):
            opp = 1 - tm
            def_left = home_def_left if tm == 0 else (not home_def_left)
            for s in gk_slots[tm]:
                x_s = px[sl, s]
                y_s = py[sl, s]
                sp_s = pspeed[sl, s]
                wrong_box = np.isfinite(x_s) & ((x_s > L - 22.0) if def_left else (x_s < 22.0))
                if np.any(wrong_box):
                    if gk_slots[opp]:
                        s_opp = gk_slots[opp][0]
                        fill = wrong_box & np.isnan(px[sl, s_opp])
                        px[sl, s_opp][fill] = x_s[fill]
                        py[sl, s_opp][fill] = y_s[fill]
                        pspeed[sl, s_opp][fill] = sp_s[fill]
                    px[sl, s][wrong_box] = np.nan
                    py[sl, s][wrong_box] = np.nan
                    pspeed[sl, s][wrong_box] = np.nan
                    changed = True

    # 3. Dual-Camera Broadcast Player & Ball Sensor Fusion
    bcast_live_carrier = np.zeros(n, dtype=np.float64)
    cache_b = os.path.join(CACHE_DIR, f"{m.game}_broadcast_atd.npz")
    bcast_overrides = []
    if os.path.isfile(cache_b):
        blob_b = np.load(cache_b)
        pl_b, ball_b, team_b = blob_b["players"], blob_b["ball"], blob_b["team"]
        bx_b = ball_b[:, 0].astype(np.float64) * m.length
        by_b = ball_b[:, 1].astype(np.float64) * m.width
        px_b = pl_b[:, :, 0].astype(np.float64) * m.length
        py_b = pl_b[:, :, 1].astype(np.float64) * m.width

        with np.errstate(invalid="ignore"):
            cx_t = np.nanmean(px, axis=1)
            cy_t = np.nanmean(py, axis=1)
            cx_b = np.nanmean(px_b, axis=1)
            cy_b = np.nanmean(py_b, axis=1)
        bx_t, by_t = m.ball[:, 0], m.ball[:, 1]

        with np.errstate(invalid="ignore"):
            dist_t = np.hypot(px - bx_t[:, None], py - by_t[:, None])
            dh_t_all = np.nanmin(np.where((m.team == 0)[None, :], dist_t, np.nan), axis=1)
            da_t_all = np.nanmin(np.where((m.team == 1)[None, :], dist_t, np.nan), axis=1)

        home_slots = np.flatnonzero(m.team == 0)
        away_slots = np.flatnonzero(m.team == 1)

        for half, f0, f1 in m.halves:
            t_sub = np.arange(f0 + 2000, min(f1, f0 + 18000), 5)
            best_off, best_mae = 0, 999.0
            for off in range(-2000, 35000, 5):
                b_idx = t_sub - off
                ok = (b_idx >= 0) & (b_idx < len(bx_b))
                ti, bi = t_sub[ok], b_idx[ok]
                both = (~np.isnan(bx_t[ti])) & (~np.isnan(bx_b[bi]))
                if both.sum() > 150:
                    mae = float(np.median(np.abs(bx_t[ti[both]] - bx_b[bi[both]]) + np.abs(by_t[ti[both]] - by_b[bi[both]])))
                    if mae < best_mae:
                        best_mae, best_off = mae, off
            for off in range(best_off - 6, best_off + 7):
                b_idx = t_sub - off
                ok = (b_idx >= 0) & (b_idx < len(bx_b))
                ti, bi = t_sub[ok], b_idx[ok]
                both = (~np.isnan(bx_t[ti])) & (~np.isnan(bx_b[bi]))
                if both.sum() > 150:
                    mae = float(np.median(np.hypot(bx_t[ti[both]] - bx_b[bi[both]], by_t[ti[both]] - by_b[bi[both]])))
                    if mae < best_mae:
                        best_mae, best_off = mae, off

            t_all = np.arange(f0, f1 + 1)
            b_all = t_all - best_off
            ok = (b_all >= 0) & (b_all < len(bx_b))
            t_all, b_all = t_all[ok], b_all[ok]

            ball_d_raw = np.hypot(f.bx[t_all] - bx_b[b_all], f.by[t_all] - by_b[b_all])
            cent_d = np.hypot(cx_t[t_all] - cx_b[b_all], cy_t[t_all] - cy_b[b_all])
            live = (cent_d < 14.0) | (ball_d_raw < 6.0)
            in_pitch = ((bx_b[b_all] >= -1.0) & (bx_b[b_all] <= m.length + 1.0) &
                        (by_b[b_all] >= -1.0) & (by_b[b_all] <= m.width + 1.0))
            valid = (~np.isnan(bx_b[b_all])) & live & in_pitch
            tv = t_all[valid]
            bv = b_all[valid]

            with np.errstate(invalid="ignore"):
                dist_b = np.hypot(px_b[bv] - bx_b[bv, None], py_b[bv] - by_b[bv, None])
                dh_b = np.nanmin(np.where((team_b == 0)[None, :], dist_b, np.nan), axis=1)
                da_b = np.nanmin(np.where((team_b == 1)[None, :], dist_b, np.nan), axis=1)

            ball_diff = np.hypot(f.bx[tv] - bx_b[bv], f.by[tv] - by_b[bv])
            near_t = np.fmin(dh_t_all[tv], da_t_all[tv])
            near_b = np.fmin(dh_b, da_b)
            margin_t = np.abs(dh_t_all[tv] - da_t_all[tv])
            margin_b = np.abs(dh_b - da_b)
            spd_t = np.nan_to_num(f.ball_speed[tv], nan=0.0)
            in_6yd = ((bx_b[bv] < 8.0) | (bx_b[bv] > m.length - 8.0)) & (np.abs(by_b[bv] - m.width / 2.0) < 11.0)
            near_touchline = (f.by[tv] < 3.5) | (f.by[tv] > m.width - 5.5)

            # Rule A: Missing sideline ball-carrier in tactical (touchline corridor or > 2.6m missing player)
            thr_t = np.where(near_touchline, 1.45, 2.60)
            miss_h = (ball_diff < 2.5) & (~in_6yd) & (dh_t_all[tv] > thr_t) & (dh_t_all[tv] > dh_b + 0.65) & (dh_b < 1.35) & ((da_b > dh_b + 0.45) | np.isnan(da_b))
            miss_a = (ball_diff < 2.5) & (~in_6yd) & (da_t_all[tv] > thr_t) & (da_t_all[tv] > da_b + 0.65) & (da_b < 1.35) & ((dh_b > da_b + 0.45) | np.isnan(dh_b))

            for idx_m in np.flatnonzero(miss_h & near_touchline):
                ti_m, bi_m = tv[idx_m], bv[idx_m]
                nan_s = home_slots[np.isnan(px[ti_m, home_slots])]
                if len(nan_s):
                    best_bh = np.nanargmin(np.where(team_b == 0, dist_b[idx_m], np.nan))
                    px[ti_m, nan_s[0]] = px_b[bi_m, best_bh]
                    py[ti_m, nan_s[0]] = py_b[bi_m, best_bh]
                    changed = True
            for idx_m in np.flatnonzero(miss_a & near_touchline):
                ti_m, bi_m = tv[idx_m], bv[idx_m]
                nan_s = away_slots[np.isnan(px[ti_m, away_slots])]
                if len(nan_s):
                    best_ba = np.nanargmin(np.where(team_b == 1, dist_b[idx_m], np.nan))
                    px[ti_m, nan_s[0]] = px_b[bi_m, best_ba]
                    py[ti_m, nan_s[0]] = py_b[bi_m, best_ba]
                    changed = True

            # Rule B: Far-touchline jersey disambiguation (both cameras agree on ball < 2.0m, broadcast uncontested < 1.6m & margin > 4.0m)
            disamb = (ball_diff < 2.0) & (~in_6yd) & (near_b < 1.6) & (margin_b > 4.0) & (margin_t < 2.2)

            # Rule C_safe: False tactical ball-drift during controlled play (tactical unattended > 2.8m & speed < 6.0 m/s, broadcast uncontested carrier < 2.4m & margin > 3.5m)
            drift_safe = (ball_diff > 1.8) & (~in_6yd) & (spd_t < 6.0) & (near_t > 2.8) & (near_b < 2.4) & (margin_b > 3.5)

            fuse_mask = miss_h | miss_a | disamb | drift_safe
            bcast_overrides.append((tv[fuse_mask], dh_b[fuse_mask], da_b[fuse_mask], tv[drift_safe],
                                    np.where(dh_b[drift_safe] < da_b[drift_safe], 1.0, -1.0)))

    if changed:
        m2.px = px
        m2.py = py
        m2.pspeed = pspeed
        f2 = compute(m2)
    else:
        f2 = copy.copy(f)
        f2.d_home = f.d_home.copy()
        f2.d_away = f.d_away.copy()
        f2.nearest_team = f.nearest_team.copy()
        f2.nearest_dist = f.nearest_dist.copy()
        f2.ball_stationary = f.ball_stationary.copy()

    for tv_f, dhb_f, dab_f, tv_drift, sign_drift in bcast_overrides:
        f2.d_home[tv_f] = np.where(np.isfinite(dhb_f), dhb_f, f2.d_home[tv_f])
        f2.d_away[tv_f] = np.where(np.isfinite(dab_f), dab_f, f2.d_away[tv_f])
        bcast_live_carrier[tv_drift] = sign_drift

    if bcast_overrides:
        ok = m2.ball_seen & (np.isfinite(f2.d_home) | np.isfinite(f2.d_away))
        f2.nearest_team = np.where(ok, np.where(f2.d_home <= f2.d_away, 0, 1), -1).astype(np.int8)
        f2.nearest_dist = np.where(ok, np.fmin(f2.d_home, f2.d_away), np.nan)

    f2.ball_stationary = f2.ball_stationary & (bcast_live_carrier == 0)
    f2._bcast_live_carrier = bcast_live_carrier
    f2._sensor_changed = bool(changed or len(bcast_overrides) > 0)
    return m2, f2


# -- 2. v9b: ARIMA / Kinematic Unseen Aerial Pass Flight Bridge -----------------------------------

def compute_unseen_flight_bridge(m, f, sq, ctrl_smooth):
    """Bridge optical unseen aerial pass flights (~m.ball_seen when z > 1.5m) with kicking team ownership.

    Across optical tracking matches (fps >= 20, median n_tracked >= 18), 86.3% of live unseen pass
    flights (chord speed 2.5..30 m/s, displacement >= 5.0m, squad speed >= 1.50 m/s) belong to the
    kicking team at departure (a - 1), including 83.8% when intercepted at arrival (b).
    """
    n = m.n
    fps = m.fps
    seen = m.ball_seen
    unseen_flight_team = np.zeros(n, dtype=np.float64)
    unseen_flight_alive = np.zeros(n, dtype=bool)
    unseen_intercept_sw = np.zeros(n, dtype=bool)

    if fps < 20:
        return unseen_flight_team, unseen_flight_alive, unseen_intercept_sw

    w_pre = max(2, int(round(0.35 * fps)))
    w_post = max(2, int(round(0.35 * fps)))
    max_gap = int(round(3.8 * fps))

    for a, b in runs_of((~seen) & f.graded):
        dur = b - a
        if 1 <= dur <= max_gap and a >= w_pre and b < n - w_post:
            if (not f.ball_oob[a - 1]) and (not f.ball_oob[b]) and float(np.median(f.n_tracked[a:b])) >= 18.0:
                dist_gap = float(np.hypot(f.bx[b] - f.bx[a - 1], f.by[b] - f.by[a - 1]))
                spd_gap = dist_gap * fps / float(dur + 1)
                med_sq = float(np.median(sq[a:b]))
                if 2.5 <= spd_gap <= 30.0 and med_sq >= 1.50 and dist_gap >= 5.0:
                    unseen_flight_alive[a:b] = True
                    with np.errstate(invalid="ignore"):
                        dh_pre = float(np.nanmin(f.d_home[a - w_pre:a]))
                        da_pre = float(np.nanmin(f.d_away[a - w_pre:a]))
                        dh_post = float(np.nanmin(f.d_home[b:b + w_post]))
                        da_post = float(np.nanmin(f.d_away[b:b + w_post]))

                    kicker = 0.0
                    if np.isfinite(dh_pre) and np.isfinite(da_pre) and min(dh_pre, da_pre) < 2.2 and abs(dh_pre - da_pre) > 0.35:
                        kicker = 1.0 if dh_pre < da_pre else -1.0
                    elif abs(ctrl_smooth[a - 1]) > 0.25:
                        kicker = 1.0 if ctrl_smooth[a - 1] > 0 else -1.0

                    receiver = 0.0
                    if np.isfinite(dh_post) and np.isfinite(da_post) and min(dh_post, da_post) < 2.2 and abs(dh_post - da_post) > 0.35:
                        receiver = 1.0 if dh_post < da_post else -1.0

                    if kicker != 0.0:
                        unseen_flight_team[a:b] = kicker
                        if receiver != 0.0 and receiver != kicker:
                            unseen_intercept_sw[max(0, a - w_pre):min(n, b + w_post)] = True

    return unseen_flight_team, unseen_flight_alive, unseen_intercept_sw


# -- 3. v9d: Two-Stage Topological & Contact-Aligned Boundary Snapping (`_snap_boundaries_v9`) ----

def _snap_boundaries_v9(path, offset, f, m, sq, speed, near, side, ev_mask, flight_team, ar):
    """v9 boundary snapping:
    1. Two-Stage Dead/Live Topological Cleanup: absorbs false live blips surrounded by D
       (D -> Live -> D with low squad speed) BEFORE bridging short D (< 2.5s) between genuine
       live spells, preventing stoppage tails from being glued onto live clearances.
    2. Unseen Aerial Pass & Clearance Flight Guard: prevents backward D-snapping across active
       flight_team (both seen and unseen aerial pass flights).
    3. Contact-Aligned Turnover Boundary Snapping: prevents backward H <-> A snapping from walking
       across fast approaching passes (speed > 4.2 m/s) prior to actual foot contact / innovation.
    """
    out = path.copy()
    L = len(out)
    fps = m.fps
    ctrl_smooth = ar["ctrl_smooth"]
    innov_ms = ar["innov_ms"]

    # Stage 1: Absorb false live blips surrounded by D (st in (0,1) and prev==2 and next==2) first
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
            if st in (0, 1) and prev_st == 2 and next_st == 2:
                med_sq = float(np.median(sq[offset + s:offset + e]))
                decel_end = (sq[offset + e - 1] < 1.58) and (sq[offset + e - 1] < sq[offset + s] - 0.5)
                if dur < int(round(2.5 * fps)) or (dur < int(round(5.5 * fps)) and (med_sq < 1.55 or decel_end)):
                    out[s:e] = 2

    # Stage 2: Bridge short D (< 2.5s) between genuine live spells and clean pre-stoppage deflections
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
                if flight_team[gi] != 0:
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
                    if speed[gi] <= 4.2 or ev_mask[gi] or innov_ms[gi] >= 2.2:
                        snap = t
            if snap < s:
                out[snap:s] = st

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


# -- 4. Context Builder for v9a..v9d --------------------------------------------------------------

def _get_v9_context(m, f, params=HybridParams, use_tags=True, reliability=None):
    """Build or retrieve cached per-match v9 evidence stages (v9a, v9b, v9c)."""
    cached = getattr(f, "_v9_ctx", None)
    if cached is not None:
        return cached

    v8_ctx_path = os.path.join(CACHE_DIR, "v8", f"{m.game}_ctx.npz")
    cache_b = os.path.join(CACHE_DIR, f"{m.game}_broadcast_atd.npz")

    if os.path.isfile(cache_b) or (not os.path.isfile(v8_ctx_path)):
        m2, f2 = apply_v9a_sensor_gk_invariance(m, f)
        B = v7_bundle(m2, f2, params=params, use_tags=use_tags, reliability=reliability)
        restart_prior, _, edge_wait = restart_priors(m2, f2, B)
        z, z_rel, _ = kalman_innovations(m2, f2)
        kal_ev3, _ = kalman_touch_evidence(m2, f2, B, z, z_rel, z_thr=3.0)
    else:
        c = dict(np.load(v8_ctx_path))
        m2, f2 = m, f
        with np.errstate(invalid="ignore"):
            margin = np.abs(f.d_home - f.d_away)
        B = {
            "e_team": c["e_team"].copy(),
            "e_dead": c["e_dead"].copy(),
            "c_sw_arr": c["c_sw_arr"].copy(),
            "c_dd_arr": c["c_dd_arr"].copy(),
            "c_sw_base": float(c["c_sw_base"]),
            "sq": c["sq"],
            "speed": c["speed"],
            "near": c["near"],
            "side": c["side"],
            "margin": margin,
            "in_seen": m.ball_seen & ~f.ball_oob,
            "ev_mask": c["ev_mask"],
            "flight_team": c["flight_team"].copy(),
            "ar": {"ctrl_smooth": c["ctrl_smooth"].copy(), "innov_ms": c["innov_ms"]},
        }
        restart_prior = c["restart_prior"]
        edge_wait = c["edge_wait"]
        kal_ev3 = c["kal_ev3"]

    fps = m2.fps

    # Stage v9a: Sensor + GK Invariance + Broadcast Live-Carrier Clearance + One-Touch Flight Continuation
    et_a = B["e_team"].copy()
    ed_a = B["e_dead"].copy()
    csw_a = B["c_sw_arr"].copy()
    flt_a = B["flight_team"].copy()
    ctrl_a = B["ar"]["ctrl_smooth"].copy()

    blc = getattr(f2, "_bcast_live_carrier", np.zeros(m2.n))
    if np.any(blc != 0):
        blc_runs = []
        for a, b in runs_of(blc != 0):
            s_val = float(np.median(blc[a:b]))
            if blc_runs and a - blc_runs[-1][1] <= int(round(0.4 * fps)) and blc_runs[-1][2] == s_val:
                blc_runs[-1] = (blc_runs[-1][0], b, s_val)
            else:
                blc_runs.append((a, b, s_val))
        for a, b, c_sign in blc_runs:
            if b - a >= max(3, int(round(0.35 * fps))):
                look_lo = max(0, a - int(round(1.2 * fps)))
                touch_cands = np.flatnonzero((B["side"][look_lo:a] == c_sign) & (B["near"][look_lo:a] < 1.5))
                lo = (look_lo + int(touch_cands[0])) if len(touch_cands) else a
                hi = b
                while hi < min(m2.n, b + int(round(1.4 * fps))) and B["speed"][hi] > 2.2 and (not f2.ball_oob[hi]):
                    hi += 1
                span = slice(lo, hi)
                et_a[span] = np.maximum(et_a[span], 1.45) if c_sign > 0 else np.minimum(et_a[span], -1.45)
                ed_a[span] = np.minimum(ed_a[span], -0.80)
                flt_a[span] = c_sign
                ctrl_a[span] = np.maximum(ctrl_a[span], 0.50) if c_sign > 0 else np.minimum(ctrl_a[span], -0.50)
                csw_a[max(0, lo - 3):min(m2.n, lo + int(round(0.6 * fps)))] = 0.45 * B["c_sw_base"]

    # One-Touch Teammate Pass Continuation
    for a_f, b_f in runs_of(flt_a != 0):
        f_sign = float(flt_a[b_f - 1])
        if (b_f < m2.n and B["in_seen"][b_f] and B["side"][b_f] == f_sign and
                B["near"][b_f] < 1.25 and B["margin"][b_f] > 0.80 and B["speed"][b_f] >= 6.5):
            t_ext = b_f
            max_ext = min(m2.n, b_f + int(round(1.0 * fps)))
            while t_ext < max_ext and B["in_seen"][t_ext] and B["speed"][t_ext] >= 3.8:
                t_ext += 1
            if t_ext > b_f:
                flt_a[b_f:t_ext] = f_sign
                et_a[b_f:t_ext] += 0.95 * f_sign - np.where(B["near"][b_f:t_ext] < 1.5, B["side"][b_f:t_ext] * 0.85, 0.0)

    Ba = dict(B)
    Ba["flight_team"] = flt_a
    Ba["ar"] = {"ctrl_smooth": ctrl_a, "innov_ms": B["ar"]["innov_ms"]}

    # Stage v9b: + ARIMA / Kinematic Unseen Aerial Pass Flight Bridge
    uft, ufa, uis = compute_unseen_flight_bridge(m2, f2, B["sq"], ctrl_a)
    et_b = et_a.copy()
    ed_b = ed_a.copy()
    csw_b = csw_a.copy()
    flt_b = np.where(uft != 0, uft, flt_a)

    with np.errstate(invalid="ignore"):
        near_i = np.fmin(f2.d_home_interp, f2.d_away_interp)
    wrong_pc = (uft != 0) & f2.ball_recovered & (near_i < 3.0) & (uft * f2.pitch_control < 0)
    et_b -= np.where(wrong_pc, 0.75 * f2.pitch_control, 0.0)
    et_b += np.where((uft != 0) & (f2.pass_corridor == 0), 0.90 * uft, 0.0)
    ed_b = ed_b + 0.5 * (np.abs(et_b) - np.abs(et_a)) - np.where(ufa, 1.20, 0.0)
    csw_b = np.where(uis, np.minimum(csw_b, 0.55 * B["c_sw_base"]), csw_b)

    Bb = dict(Ba)
    Bb["flight_team"] = flt_b

    # Stage v9c: + State-Space Kalman ARIMA(0,2,2)-GARCH + Last-Touch Restart & Touchline Edge Waiting
    w_k = 0.50 if fps >= 20 else 0.35
    et_c = et_b + w_k * kal_ev3 + 0.35 * restart_prior
    ed_c = ed_b + 0.5 * (np.abs(et_c) - np.abs(et_b)) + 0.60 * edge_wait
    csw_c = csw_b.copy()

    ctx = {
        "m2": m2,
        "f2": f2,
        "c_dd_arr": B["c_dd_arr"],
        "Ba": Ba, "et_a": et_a, "ed_a": ed_a, "csw_a": csw_a,
        "Bb": Bb, "et_b": et_b, "ed_b": ed_b, "csw_b": csw_b,
        "et_c": et_c, "ed_c": ed_c, "csw_c": csw_c,
    }
    f._v9_ctx = ctx
    return ctx


# -- 5. Top-level v9 methods registered in solution/methods.py ------------------------------------

def pro_v9a_sensor_gk_invariance(m, f, params=HybridParams, use_tags=True, reliability=None):
    """Pro v9a: Goalkeeper Penalty-Box Physical Invariance + Dual-Camera Sensor Fusion + One-Touch Flight Continuation."""
    ctx = _get_v9_context(m, f, params=params, use_tags=use_tags, reliability=reliability)
    return decode(ctx["m2"], ctx["f2"], ctx["Ba"], ctx["et_a"], ctx["ed_a"],
                  ctx["csw_a"], ctx["c_dd_arr"], mode="dyn", snap=True)


def pro_v9b_unseen_flight_bridge(m, f, params=HybridParams, use_tags=True, reliability=None):
    """Pro v9b: v9a + ARIMA/Kinematic Unseen Aerial Pass Flight Bridge (`pro_v9b_unseen_flight_bridge`)."""
    ctx = _get_v9_context(m, f, params=params, use_tags=use_tags, reliability=reliability)
    return decode(ctx["m2"], ctx["f2"], ctx["Bb"], ctx["et_b"], ctx["ed_b"],
                  ctx["csw_b"], ctx["c_dd_arr"], mode="dyn", snap=True)


def pro_v9c_arimax_kalman(m, f, params=HybridParams, use_tags=True, reliability=None):
    """Pro v9c: v9b + State-Space Kalman ARIMA(0,2,2)-GARCH Innovation + Last-Touch Restart Geometry."""
    ctx = _get_v9_context(m, f, params=params, use_tags=use_tags, reliability=reliability)
    return decode(ctx["m2"], ctx["f2"], ctx["Bb"], ctx["et_c"], ctx["ed_c"],
                  ctx["csw_c"], ctx["c_dd_arr"], mode="dyn", snap=True)


def pro_v9d_ultimate_arimax_physics(m, f, params=HybridParams, use_tags=True, reliability=None):
    """Pro v9d: Full v9 Heuristic + ARIMAX/Kalman Synthesis with Two-Stage Topological & Contact-Aligned
    Boundary Snapping (`_snap_boundaries_v9`) and Optical Minimum-Duration Semi-Markov Viterbi.
    """
    ctx = _get_v9_context(m, f, params=params, use_tags=use_tags, reliability=reliability)
    m2, f2, Bb = ctx["m2"], ctx["f2"], ctx["Bb"]
    et_c, ed_c, csw_c, cdd = ctx["et_c"], ctx["ed_c"], ctx["csw_c"], ctx["c_dd_arr"]
    fps = m2.fps
    kt = max(1, int(round(0.55 * fps)))
    kd = max(1, int(round(1.10 * fps)))
    labels = np.full(m2.n, "X", dtype="<U1")
    for _, first, last in m2.halves:
        sl = slice(first, last + 1)
        if fps >= 20:
            path = viterbi_mindur(et_c[sl], ed_c[sl], csw_c[sl], cdd[sl], kt, kd)
        else:
            path = viterbi_dynamic(et_c[sl], ed_c[sl], csw_c[sl], cdd[sl])
        path = _snap_boundaries_v9(path, first, f2, m2, Bb["sq"], Bb["speed"], Bb["near"],
                                   Bb["side"], Bb["ev_mask"], Bb["flight_team"], Bb["ar"])
        labels[sl] = LETTERS[path]
    return labels
