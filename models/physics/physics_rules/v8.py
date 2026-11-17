"""The v8 family: modular experiments on top of `pro_v7_tuned_arimax_hmm`.

Everything here starts from the exact v7 evidence ("bundle") and adds ONE idea at a time, so each
variant's gain or loss can be attributed:

    pro_v8a_restart      Football restart rules: the team that did NOT touch the ball last before it
                         left the pitch takes the restart (throw-in, corner / goal kick, kick-off
                         after a goal), plus a touchline-edge throw-in waiting detector.
    pro_v8b_hsmm         Hidden Semi-Markov decoding: an exact minimum-duration segmental Viterbi
                         (no possession / stoppage shorter than K frames), with K learned per
                         tracking regime from the 27 external matches' spell-duration histograms.
    pro_v8c_kalman       A different time-series model for the ball: a constant-velocity Kalman
                         filter (the state-space form of ARIMA(0,2,2)) with GARCH-style EWMA
                         volatility normalisation of the innovations, turned into touch evidence
                         that also works on 10 fps broadcast tracking.
    pro_v8d_stack        Genuine supervised learning: a LightGBM classifier trained on per-frame
                         features (v7 evidence + ARIMA/ARIMAX/Kalman states + multi-scale temporal
                         context) from the 27 external matches with 5-fold grouped Out-of-Fold CV,
                         whose calibrated log-probabilities replace hand-set HMM emissions.
    pro_v8e_combo        v8d emissions + restart rules + semi-Markov decoding + v6 boundary snapping.

None of these reads the BAR-ATH ground truth. Every learned number comes from the 27 external
matches; for an external match the parameters are always those fitted WITHOUT that match.
"""

import numpy as np

from .features import REACH_M, centred_count, centred_mean, runs_of
from .methods import (LETTERS, HybridParams, _compute_arima_var_kinematics_v6,
                      _compute_player_ball_sync, _detect_setpiece_geometry, _snap_boundaries_v6,
                      direction_events, hybrid_evidence, tag_reliability, tag_team_evidence,
                      viterbi_dynamic)


# -- 1. the v7 evidence, exposed -----------------------------------------------------------------

def v7_bundle(m, f, params=HybridParams, use_tags=True, reliability=None):
    """Exactly the evidence `pro_v7_tuned_arimax_hmm` decodes, plus the context arrays it uses."""
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

    flight_team, z_height, pre_rec_suppress, pressed_launch_suppress = \
        _compute_arima_var_kinematics_v6(m, f, ev, ar)

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

    return {
        "e_team": e_team, "e_dead": e_dead, "c_sw_arr": c_sw_arr, "c_dd_arr": c_dd_arr,
        "c_sw_base": c_sw_learned * scale, "c_dd_base": params.C_DEAD * scale,
        "sq": sq, "sq_future": sq_future, "speed": speed, "near": near, "margin": margin,
        "side": side, "ev": ev, "ev_mask": ev_mask, "flight_team": flight_team,
        "z_height": z_height, "in_seen": in_seen, "stop_gate": stop_gate, "ar": ar,
        "fly_by": fly_by, "strength": strength,
    }


# -- 2. decoders: first-order HMM (v7) and an exact minimum-duration semi-Markov Viterbi ----------

def viterbi_mindur(e_team, e_dead, c_sw, c_dd, k_team, k_dead):
    """Segmental (semi-Markov) Viterbi with exact minimum segment durations.

    A segment of state s can only be entered if it then lasts at least K_s frames. Using cumulative
    sums of the emissions, entering s at t0 and maturing at t = t0 + K_s - 1 costs one lookup, so the
    whole decode is still O(n * 3 * 2) -- the same as the first-order Viterbi.
        V_s(t) = max( V_s(t-1) + e_s(t),                                  # stay (mature)
                      max_{x != s} V_x(t0 - 1) - cost(x, s, t0) + E_s[t0..t] )  # enter at t0
    """
    n = len(e_team)
    em = (0.5 * e_team, -0.5 * e_team, np.asarray(e_dead, dtype=np.float64))
    cs = [np.concatenate(([0.0], np.cumsum(x))).tolist() for x in em]
    el = [x.tolist() for x in em]
    K = (max(1, int(k_team)), max(1, int(k_team)), max(1, int(k_dead)))
    csw = c_sw.tolist() if isinstance(c_sw, np.ndarray) else [float(c_sw)] * n
    cdd = c_dd.tolist() if isinstance(c_dd, np.ndarray) else [float(c_dd)] * n
    NEG = -1e18
    V = [[NEG] * n for _ in range(3)]
    back = np.full((n, 3), -1, dtype=np.int8)   # -1 stay, 0/1/2 entered from that state, 3 = start
    V0, V1, V2 = V
    for t in range(n):
        for s in (0, 1, 2):
            Vs = V[s]
            best = Vs[t - 1] + el[s][t] if t > 0 else NEG
            kind = -1
            t0 = t - K[s] + 1
            if t0 == 0:
                cand = cs[s][t + 1]
                if cand > best:
                    best, kind = cand, 3
            elif t0 > 0:
                seg = cs[s][t + 1] - cs[s][t0]
                for x in (0, 1, 2):
                    if x == s:
                        continue
                    cost = csw[t0] if (x != 2 and s != 2) else cdd[t0]
                    cand = V[x][t0 - 1] - cost + seg
                    if cand > best:
                        best, kind = cand, x
            Vs[t] = best
            back[t, s] = kind
    path = np.zeros(n, dtype=np.int8)
    s = int(np.argmax([V0[n - 1], V1[n - 1], V2[n - 1]]))
    t = n - 1
    while t >= 0:
        kind = int(back[t, s])
        if kind == -1:
            path[t] = s
            t -= 1
            continue
        t0 = max(0, t - K[s] + 1)
        path[t0:t + 1] = s
        if kind == 3:
            break
        s = kind
        t = t0 - 1
    return path


class _Ctx(object):
    """Light stand-in for (m, f) inside the boundary snapper when decoding from cached arrays."""
    pass


def decode(ctx_m, ctx_f, B, e_team, e_dead, c_sw_arr, c_dd_arr, mode="dyn", snap=True,
           k_team=1, k_dead=1):
    """Decode H/A/D per half. `ctx_m` needs n, fps, halves, ball_seen; `ctx_f` needs ball_oob."""
    labels = np.full(ctx_m.n, "X", dtype="<U1")
    for _, first, last in ctx_m.halves:
        sl = slice(first, last + 1)
        if mode == "mindur" and (k_team > 1 or k_dead > 1):
            path = viterbi_mindur(e_team[sl], e_dead[sl], c_sw_arr[sl], c_dd_arr[sl], k_team, k_dead)
        else:
            path = viterbi_dynamic(e_team[sl], e_dead[sl], c_sw_arr[sl], c_dd_arr[sl])
        if snap:
            path = _snap_boundaries_v6(path, first, ctx_f, ctx_m, B["sq"], B["speed"], B["near"],
                                       B["side"], B["ev_mask"], B["flight_team"], B["ar"])
        labels[sl] = LETTERS[path]
    return labels


# -- 3. v8a: football restart rules ----------------------------------------------------------------

def restart_priors(m, f, B):
    """Last-touch restart rule + touchline-edge throw-in waiting.

    Returns
        restart_prior  (+1 home / -1 away) over the first 1.5 s after each restart touch
        restart_feat   the same sign, held from the ball leaving play until 1.5 s after the restart
        edge_wait      ball held / placed at the touchline while the squad walks (throw-in set-up)
    """
    n, fps = m.n, m.fps
    in_seen, near, side, speed, sq = B["in_seen"], B["near"], B["side"], B["speed"], B["sq"]
    ev_mask = B["ev_mask"]
    L, W = m.length, m.width
    restart_prior = np.zeros(n)
    restart_feat = np.zeros(n)

    oob_runs = [(a, b) for a, b in runs_of(f.ball_oob) if b - a >= max(2, int(0.2 * fps))]
    merged = []
    for a, b in oob_runs:
        if merged and a - merged[-1][1] < fps:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))

    look_back = int(4.0 * fps)
    look_fwd = int(90.0 * fps)
    hold = int(1.5 * fps)
    for idx, (a, b) in enumerate(merged):
        lo = max(0, a - look_back)
        win_ok = in_seen[lo:a] & (near[lo:a] < 1.5)
        cand = np.flatnonzero(win_ok)
        if not len(cand):
            continue
        evs = cand[ev_mask[lo:a][cand]]
        t_last = lo + int(evs[-1] if len(evs) else cand[-1])
        sign = -float(side[t_last])
        nxt_a = merged[idx + 1][0] if idx + 1 < len(merged) else n
        hi = min(n, b + look_fwd, nxt_a)
        r_cand = np.flatnonzero(in_seen[b:hi] & (speed[b:hi] > 3.0) & (near[b:hi] < 2.5))
        if not len(r_cand):
            continue
        r = b + int(r_cand[0])
        restart_prior[r:min(n, r + hold)] = sign
        restart_feat[a:min(n, r + hold)] = sign

    with np.errstate(invalid="ignore"):
        d_touch = np.fmin(np.abs(f.by), np.abs(W - f.by))
        edge_wait = (in_seen | f.ball_oob) & (d_touch < 1.5) & (speed < 1.5) & (sq < 1.5) & (near < 2.0)
    edge_wait = np.nan_to_num(edge_wait).astype(bool)
    return restart_prior, restart_feat, edge_wait


# -- 4. v8c: Kalman constant-velocity filter + EWMA (GARCH-style) volatility -----------------------

def kalman_innovations(m, f, sigma_a=4.0):
    """Constant-velocity Kalman filter on the raw ball (x, y).

    The CV Kalman filter is the state-space form of an ARIMA(0,2,2) model; unlike a fixed ARIMA it
    handles missing frames natively (predict without update) and gives a time-varying innovation
    variance S_t. We return the normalised innovation magnitude z_t = sqrt(nu' S^-1 nu) and a
    volatility-relative version z_t / sqrt(EWMA(z^2)) (a GARCH(1,1)-style rescaling, so a scramble
    with constant jostling does not look like a string of touches, and noisy broadcast tracking
    does not look like a touch every frame).
    """
    n, fps = m.n, m.fps
    dt = 1.0 / fps
    bx = m.ball[:, 0].astype(np.float64)
    by = m.ball[:, 1].astype(np.float64)
    seen = m.ball_seen & np.isfinite(bx) & np.isfinite(by)

    # Measurement noise from the second differences of consecutive seen frames (var(D2 eps) = 6R)
    d2 = bx[2:] - 2 * bx[1:-1] + bx[:-2]
    ok = seen[2:] & seen[1:-1] & seen[:-2]
    if ok.sum() > 100:
        mad = np.median(np.abs(d2[ok] - np.median(d2[ok])))
        R = float(np.clip((1.4826 * mad) ** 2 / 6.0, 0.0004, 0.25))
    else:
        R = 0.01
    q = sigma_a ** 2
    Q11, Q12, Q22 = q * dt ** 4 / 4.0, q * dt ** 3 / 2.0, q * dt ** 2

    z = np.full(n, np.nan)
    bxl, byl, seenl = bx.tolist(), by.tolist(), seen.tolist()
    zl = [float("nan")] * n
    init = False
    gap = 0
    max_gap = int(0.5 * fps)
    for t in range(n):
        if not seenl[t]:
            if init:
                gap += 1
                if gap > max_gap:
                    init = False
                else:
                    for st in (sx, sy):
                        st[0] += dt * st[1]
                    P11 = P11 + 2 * dt * P12 + dt * dt * P22 + Q11
                    P12 = P12 + dt * P22 + Q12
                    P22 = P22 + Q22
            continue
        if not init:
            sx = [bxl[t], 0.0]
            sy = [byl[t], 0.0]
            P11, P12, P22 = R, 0.0, 25.0
            init = True
            gap = 0
            continue
        gap = 0
        # predict
        sx[0] += dt * sx[1]
        sy[0] += dt * sy[1]
        P11 = P11 + 2 * dt * P12 + dt * dt * P22 + Q11
        P12 = P12 + dt * P22 + Q12
        P22 = P22 + Q22
        # update (shared covariance for x and y: same model, same noise)
        S = P11 + R
        nux = bxl[t] - sx[0]
        nuy = byl[t] - sy[0]
        zl[t] = ((nux * nux + nuy * nuy) / S) ** 0.5
        K1, K2 = P11 / S, P12 / S
        sx[0] += K1 * nux
        sx[1] += K2 * nux
        sy[0] += K1 * nuy
        sy[1] += K2 * nuy
        P22 = P22 - K2 * P12
        P12 = P12 - K1 * P12
        P11 = P11 - K1 * P11
    z = np.array(zl)
    # EWMA volatility of z^2 over ~1 s, using only the past
    lam = float(np.exp(-1.0 / fps))
    z2 = np.nan_to_num(z, nan=0.0) ** 2
    from scipy.signal import lfilter
    vol = lfilter([1.0 - lam], [1.0, -lam], z2)
    vol = np.concatenate(([vol[0]], vol[:-1]))
    z_rel = np.where(np.isfinite(z), z / np.sqrt(np.maximum(vol, 1.0)), 0.0)
    return np.nan_to_num(z, nan=0.0), z_rel, R


def kalman_touch_evidence(m, f, B, z, z_rel, z_thr=3.0):
    """Signed touch evidence at local peaks of the volatility-relative Kalman innovation."""
    n, fps = m.n, m.fps
    in_seen, near, margin, side = B["in_seen"], B["near"], B["margin"], B["side"]
    w = max(1, int(round(0.12 * fps)))
    zr = z_rel
    peak = np.zeros(n, dtype=bool)
    peak[w:-w] = zr[w:-w] >= np.max(np.stack([np.roll(zr, k) for k in range(-w, w + 1) if k != 0]), axis=0)[w:-w]
    touch = peak & (zr >= z_thr) & in_seen & (near < REACH_M) & (margin > 1.0) & (B["flight_team"] == 0) & ~B["fly_by"]
    ev = np.zeros(n)
    ev[touch] = side[touch] * np.clip(zr[touch] / (2.0 * z_thr), 0.5, 1.5)
    return ev, touch


# -- 5. v8d: per-frame features for the supervised stacker -----------------------------------------

def _past_mean(x, w):
    c = np.concatenate(([0.0], np.cumsum(x)))
    idx = np.arange(len(x))
    lo = np.maximum(0, idx - w)
    return (c[idx] - c[lo]) / np.maximum(idx - lo, 1)


def _future_mean(x, w):
    c = np.concatenate(([0.0], np.cumsum(x)))
    n = len(x)
    idx = np.arange(n)
    lo = np.minimum(n, idx + 1)
    hi = np.minimum(n, idx + 1 + w)
    return (c[hi] - c[lo]) / np.maximum(hi - lo, 1)


def frame_features(m, f, B, restart_feat, edge_wait, z, z_rel, kal_ev):
    """(n, F) float32 feature matrix + names. Windows are in seconds, so fps-invariant."""
    fps = m.fps
    L, W = m.length, m.width
    in_seen = B["in_seen"].astype(np.float64)
    near = np.nan_to_num(B["near"], nan=8.0)
    margin = np.nan_to_num(B["margin"], nan=0.0)
    side = B["side"]
    speed = np.clip(B["speed"], 0, 35)
    prox = np.where(B["in_seen"] & (near < 3.0), side * np.clip(1.0 - near / 3.0, 0, 1) * np.clip(margin / 2.0, 0, 1), 0.0)
    e_team = np.clip(B["e_team"], -8, 8)
    e_dead = np.clip(B["e_dead"], -8, 12)
    ar = B["ar"]
    sq = B["sq"]
    bxn = np.nan_to_num(f.bx, nan=L / 2.0)
    byn = np.nan_to_num(f.by, nan=W / 2.0)
    d_touch = np.clip(np.fmin(np.abs(byn), np.abs(W - byn)), 0, 40)
    d_goal = np.clip(np.fmin(np.abs(bxn), np.abs(L - bxn)), 0, 60)
    oob = f.ball_oob.astype(np.float64)
    ev_f = B["ev_mask"].astype(np.float64)

    def S(sec):
        return max(1, int(round(sec * fps)))

    cols = {
        "e_team": e_team, "e_dead": e_dead,
        "c_sw_disc": (B["c_sw_arr"] < 0.9 * B["c_sw_base"]).astype(np.float64),
        "prox": prox, "near": np.clip(near, 0, 30), "margin": np.clip(margin, 0, 20),
        "d_home": np.clip(np.nan_to_num(f.d_home, nan=30.0), 0, 30),
        "d_away": np.clip(np.nan_to_num(f.d_away, nan=30.0), 0, 30),
        "speed": speed, "sq": sq, "sq_future": B["sq_future"],
        "in_seen": in_seen, "oob": oob, "stationary": f.ball_stationary.astype(np.float64),
        "player_oob": f.player_oob.astype(np.float64),
        "single_oob": f.single_oob_waiting.astype(np.float64),
        "n_tracked": f.n_tracked / 22.0,
        "flight": B["flight_team"], "z_height": B["z_height"],
        "pitch_ctrl": f.pitch_control, "corridor": f.pass_corridor,
        "width": f.team_width_diff, "flow": f.team_flow, "vball": f.vball_cluster,
        "ctrl_arimax": ar["ctrl_arimax"], "ctrl_smooth": ar["ctrl_smooth"],
        "innov_ms": np.clip(ar["innov_ms"], 0, 20),
        "kal_z": np.clip(z, 0, 50), "kal_rel": np.clip(z_rel, 0, 20), "kal_ev": kal_ev,
        "restart": restart_feat, "edge_wait": edge_wait.astype(np.float64),
        "d_touch": d_touch, "d_goal": d_goal,
        "bx_att": (bxn / L - 0.5),
        "optical": np.full(m.n, 1.0 if fps >= 20 else 0.0),
    }
    for sec in (0.5, 1.0, 2.0, 4.0, 8.0):
        cols[f"prox_c{sec}"] = centred_mean(prox, S(sec))[0]
        cols[f"eteam_c{sec}"] = centred_mean(e_team, S(sec))[0]
    for sec in (1.0, 3.0):
        cols[f"prox_p{sec}"] = _past_mean(prox, S(sec))
        cols[f"prox_f{sec}"] = _future_mean(prox, S(sec))
        cols[f"eteam_p{sec}"] = _past_mean(e_team, S(sec))
        cols[f"eteam_f{sec}"] = _future_mean(e_team, S(sec))
        cols[f"edead_p{sec}"] = _past_mean(e_dead, S(sec))
        cols[f"edead_f{sec}"] = _future_mean(e_dead, S(sec))
        cols[f"seen_c{sec}"] = centred_mean(in_seen, S(sec))[0]
        cols[f"oob_p{sec}"] = _past_mean(oob, S(sec))
        cols[f"oob_f{sec}"] = _future_mean(oob, S(sec))
        cols[f"speed_c{sec}"] = centred_mean(speed, S(sec))[0]
        cols[f"ev_c{sec}"] = centred_mean(ev_f, S(sec))[0]
    for sec in (2.0, 5.0):
        cols[f"sq_p{sec}"] = _past_mean(sq, S(sec))
        cols[f"sq_f{sec}"] = _future_mean(sq, S(sec))
        cols[f"ctrl_c{sec}"] = centred_mean(ar["ctrl_arimax"], S(sec))[0]
    names = list(cols)
    X = np.empty((m.n, len(names)), dtype=np.float32)
    for j, k in enumerate(names):
        X[:, j] = np.nan_to_num(np.asarray(cols[k], dtype=np.float64), nan=0.0)
    return X, names


def ml_emissions(proba, lam=1.0, eps=1e-4):
    """Class probabilities (n, 3) [H, A, D] -> (e_team, e_dead) in the Viterbi's convention.

    The Viterbi scores H = +e_team/2, A = -e_team/2, D = e_dead, so with log-probabilities
    e_team = log pH - log pA and e_dead = log pD - (log pH + log pA) / 2 reproduce them exactly up
    to a per-frame constant (which no path can avoid, so it never changes the argmax).
    """
    lp = np.log(np.clip(proba, eps, 1.0))
    e_team = lam * (lp[:, 0] - lp[:, 1])
    e_dead = lam * (lp[:, 2] - 0.5 * (lp[:, 0] + lp[:, 1]))
    return e_team, e_dead


def _snap_boundaries_v8(path, offset, f, m, sq, speed, near, side, ev_mask, flight_team, ar, d_ml=None):
    """Boundary snapping with an unseen-aerial-pass guard when multi-scale ML dead-ball logits exist."""
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
                if d_ml is not None and (not m.ball_seen[gi]) and d_ml[gi] < -1.2 and (not f.ball_oob[gi]) and sq[gi] >= 1.65:
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


def _get_v8_context(m, f, params=HybridParams, use_tags=True, reliability=None):
    """Load or compute the cached v8 context and Out-of-Fold / cross-match LightGBM probabilities."""
    cached = getattr(f, "_v8_ctx", None)
    if cached is not None:
        return cached

    import os
    from .data import CACHE_DIR
    v8_dir = os.path.join(CACHE_DIR, "v8")
    ctx_path = os.path.join(v8_dir, f"{m.game}_ctx.npz")
    proba_path = os.path.join(v8_dir, f"{m.game}_proba.npy")

    if os.path.isfile(ctx_path):
        c = dict(np.load(ctx_path))
        B = {k: c[k] for k in ("sq", "speed", "near", "side", "ev_mask", "flight_team")}
        B["ar"] = {"ctrl_smooth": c["ctrl_smooth"], "innov_ms": c["innov_ms"]}
        ctx = {
            "e_team": c["e_team"],
            "e_dead": c["e_dead"],
            "c_sw_arr": c["c_sw_arr"],
            "c_dd_arr": c["c_dd_arr"],
            "restart_prior": c["restart_prior"],
            "edge_wait": c["edge_wait"],
            "kal_ev3": c["kal_ev3"],
            "B": B,
        }
    else:
        B = v7_bundle(m, f, params=params, use_tags=use_tags, reliability=reliability)
        restart_prior, restart_feat, edge_wait = restart_priors(m, f, B)
        z, z_rel, _ = kalman_innovations(m, f)
        kal_ev3, _ = kalman_touch_evidence(m, f, B, z, z_rel, z_thr=3.0)
        ctx = {
            "e_team": B["e_team"],
            "e_dead": B["e_dead"],
            "c_sw_arr": B["c_sw_arr"],
            "c_dd_arr": B["c_dd_arr"],
            "restart_prior": restart_prior,
            "edge_wait": edge_wait,
            "kal_ev3": kal_ev3,
            "B": B,
        }

    if os.path.isfile(proba_path):
        ctx["proba"] = np.load(proba_path)
    else:
        ctx["proba"] = None

    f._v8_ctx = ctx
    return ctx


# -- 6. Top-level v8 methods registered in solution/methods.py ------------------------------------

def pro_v8a_restart(m, f, params=HybridParams, use_tags=True, reliability=None):
    """Pro v8a: Last-Touch Out-of-Bounds Restart Rule + Touchline-Edge Throw-In Waiting (`pro_v8a_restart`)."""
    ctx = _get_v8_context(m, f, params=params, use_tags=use_tags, reliability=reliability)
    B = ctx["B"]
    et0, ed0 = ctx["e_team"], ctx["e_dead"]
    et1 = et0 + 0.50 * ctx["restart_prior"]
    ed1 = ed0 + 0.5 * (np.abs(et1) - np.abs(et0)) + 0.80 * ctx["edge_wait"]
    return decode(m, f, B, et1, ed1, ctx["c_sw_arr"], ctx["c_dd_arr"], mode="dyn", snap=True)


def pro_v8b_hsmm(m, f, params=HybridParams, use_tags=True, reliability=None):
    """Pro v8b: Hidden Semi-Markov Minimum-Duration Viterbi (`pro_v8b_hsmm`)."""
    ctx = _get_v8_context(m, f, params=params, use_tags=use_tags, reliability=reliability)
    B = ctx["B"]
    kt = max(1, int(round(0.60 * m.fps)))
    kd = max(1, int(round(1.20 * m.fps)))
    return decode(m, f, B, ctx["e_team"], ctx["e_dead"], ctx["c_sw_arr"], ctx["c_dd_arr"],
                  mode="mindur", snap=True, k_team=kt, k_dead=kd)


def pro_v8c_kalman(m, f, params=HybridParams, use_tags=True, reliability=None):
    """Pro v8c: State-Space Constant-Velocity Kalman Filter ARIMA(0,2,2) + GARCH Volatility (`pro_v8c_kalman`)."""
    ctx = _get_v8_context(m, f, params=params, use_tags=use_tags, reliability=reliability)
    B = ctx["B"]
    et0, ed0 = ctx["e_team"], ctx["e_dead"]
    w_k = 0.50 if m.fps >= 20 else 0.35
    et1 = et0 + w_k * ctx["kal_ev3"] + 0.35 * ctx["restart_prior"]
    ed1 = ed0 + 0.5 * (np.abs(et1) - np.abs(et0)) + 0.60 * ctx["edge_wait"]
    return decode(m, f, B, et1, ed1, ctx["c_sw_arr"], ctx["c_dd_arr"], mode="dyn", snap=True)


def pro_v8d_stack(m, f, params=HybridParams, use_tags=True, reliability=None):
    """Pro v8d: Physics-Protected Cross-Match LightGBM Stacker + ARIMAX Emission Fusion (`pro_v8d_stack`)."""
    ctx = _get_v8_context(m, f, params=params, use_tags=use_tags, reliability=reliability)
    B = ctx["B"]
    if ctx["proba"] is None:
        return pro_v8c_kalman(m, f, params=params, use_tags=use_tags, reliability=reliability)
    e_ml, d_ml = ml_emissions(ctx["proba"])
    et0 = ctx["e_team"] + 0.35 * ctx["kal_ev3"] + 0.35 * ctx["restart_prior"]
    ed0 = ctx["e_dead"] + 0.60 * ctx["edge_wait"]
    w_t, w_d = 0.40, 0.75
    et1 = (1.0 - w_t) * et0 + w_t * e_ml
    phys_stop = (ed0 > 0.0) & ((B["sq"] < 1.52) | f.ball_oob | (~m.ball_seen & (ed0 > 0.8)))
    d_ml_safe = np.where(phys_stop, np.maximum(ed0, d_ml), d_ml)
    ed_base = (1.0 - w_d) * ed0 + w_d * d_ml_safe
    ed1 = ed_base + 0.5 * (np.abs(et1) - np.abs((1.0 - w_t) * et0))
    csw = ctx["c_sw_arr"] * (1.0 + 0.65 * w_t)
    cdd = ctx["c_dd_arr"] * (1.0 + 0.80 * w_d)
    labels = np.full(m.n, "X", dtype="<U1")
    for _, first, last in m.halves:
        sl = slice(first, last + 1)
        path = viterbi_dynamic(et1[sl], ed1[sl], csw[sl], cdd[sl])
        path = _snap_boundaries_v8(path, first, f, m, B["sq"], B["speed"], B["near"], B["side"],
                                   B["ev_mask"], B["flight_team"], B["ar"], d_ml=d_ml)
        labels[sl] = LETTERS[path]
    return labels


def pro_v8e_combo(m, f, params=HybridParams, use_tags=True, reliability=None):
    """Pro v8e: Full v8 Synthesis -- Tuned ARIMAX + State-Space Kalman-GARCH + Restart/Edge Geometry +
    Physics-Protected Cross-Match LightGBM + Hidden Semi-Markov Minimum-Duration Viterbi (`pro_v8e_combo`).
    """
    ctx = _get_v8_context(m, f, params=params, use_tags=use_tags, reliability=reliability)
    B = ctx["B"]
    if ctx["proba"] is None:
        return pro_v8b_hsmm(m, f, params=params, use_tags=use_tags, reliability=reliability)
    e_ml, d_ml = ml_emissions(ctx["proba"])
    et0 = ctx["e_team"] + 0.35 * ctx["kal_ev3"] + 0.35 * ctx["restart_prior"]
    ed0 = ctx["e_dead"] + 0.60 * ctx["edge_wait"]
    w_t, w_d = 0.40, 0.75
    et1 = (1.0 - w_t) * et0 + w_t * e_ml
    phys_stop = (ed0 > 0.0) & ((B["sq"] < 1.52) | f.ball_oob | (~m.ball_seen & (ed0 > 0.8)))
    d_ml_safe = np.where(phys_stop, np.maximum(ed0, d_ml), d_ml)
    ed_base = (1.0 - w_d) * ed0 + w_d * d_ml_safe
    ed1 = ed_base + 0.5 * (np.abs(et1) - np.abs((1.0 - w_t) * et0))
    csw = ctx["c_sw_arr"] * (1.0 + 0.65 * w_t)
    cdd = ctx["c_dd_arr"] * (1.0 + 0.80 * w_d)
    kt = max(1, int(round(0.60 * m.fps)))
    kd = max(1, int(round(1.20 * m.fps)))
    labels = np.full(m.n, "X", dtype="<U1")
    for _, first, last in m.halves:
        sl = slice(first, last + 1)
        path = viterbi_mindur(et1[sl], ed1[sl], csw[sl], cdd[sl], kt, kd)
        path = _snap_boundaries_v8(path, first, f, m, B["sq"], B["speed"], B["near"], B["side"],
                                   B["ev_mask"], B["flight_team"], B["ar"], d_ml=d_ml)
        labels[sl] = LETTERS[path]
    return labels
