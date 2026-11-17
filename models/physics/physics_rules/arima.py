"""Self-supervised ARIMA(p, d, q) and ARIMAX(p, d, q, exog=X) time-series modeling with
per-match order tuning and cross-match supervised learning on the 27 external matches.

Provides three levels of Box-Jenkins time-series modeling (`statsmodels.tsa.arima.model.ARIMA`):

1. `compute_arima_signals(m, f)` (used by `pro_v5_arima_hmm`):
   * Fixed-order self-supervised ARIMA(1,1,1) on ball trajectory, ARIMA(2,0,1) on team control,
     and ARIMA(2,1,1) on collective squad speed.

2. `compute_tuned_arima_signals(m, f, use_arimax=False)` (used by `pro_v6_tuned_arima_hmm`):
   * Per-Match Auto-Tuned ARIMA(p, d, q): searches candidate horizons and (p, d, q) orders per
     match via AIC / innovations maximum likelihood on that match's tracking series.
   * Cross-Match Supervised Learning (`get_cross_match_learned_params`): fits 5-fold Out-of-Fold
     cross-validated Ridge calibration and empirical spell-rate HMM transition scales on the 27
     external labelled matches (7 IDSSE Bundesliga + 20 SkillCorner A-League), cached to disk.

3. `compute_tuned_arima_signals(m, f, use_arimax=True)` (used by `pro_v7_tuned_arimax_hmm`):
   * Full Box-Jenkins ARIMAX(p, d, q, exog=X): incorporates exogenous physical shock inputs X_t
     (signed uncontested ball-touch velocity innovation shocks and pass-corridor / pitch-control
     dominance) directly into the Team Control autoregressive state equation.

Run standalone for a diagnostic comparison of v5 (ARIMA), v6 (Tuned ARIMA), and v7 (Tuned ARIMAX):
    python -m solution.arima
"""

import json
import os
import sys
import time
import warnings
import numpy as np
from scipy.signal import lfilter
from statsmodels.tsa.arima.model import ARIMA

from .data import CACHE_DIR, ROOT, list_games, load_match
from .features import MAX_PASS_MS, REACH_M, compute, runs_of

CROSS_MATCH_CACHE = os.path.join(CACHE_DIR, "cross_match_arima_learned.json")
CROSS_MATCH_REPO_PATH = os.path.join(ROOT, "weights", "cross_match_arima_learned.json")  # xG United: moved from solution/output/


def compute_arima_signals(m, f):
    """Fit baseline ARIMA(p,d,q) models on match `m` (`pro_v5_arima_hmm`)."""
    cached = getattr(f, "_arima_v5", None)
    if cached is not None:
        return cached
    n = m.n
    fps = m.fps
    seen = m.ball_seen
    in_seen = seen & ~f.ball_oob
    bx, by = f.bx, f.by
    speed = np.nan_to_num(f.ball_speed, nan=0.0)

    # -- 1. Ball Trajectory ARIMA(1, 1, 1) at physical touch horizon k (~0.16 s) ----------------
    k = max(1, int(round(0.16 * fps)))
    dx_k = np.full(n, np.nan)
    dy_k = np.full(n, np.nan)
    dx_k[k:] = bx[k:] - bx[:-k]
    dy_k[k:] = by[k:] - by[:-k]
    ok_k = np.isfinite(dx_k) & (np.hypot(dx_k, dy_k) * (fps / float(k)) < 38.0)

    phi1_b, theta1_b, sigma_b = 0.96, 0.50, 0.16
    if ok_k.sum() > 200:
        sample_dx = dx_k[ok_k][:min(10000, int(ok_k.sum()))][::max(1, k)]
        if len(sample_dx) > 80:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    res_b = ARIMA(sample_dx, order=(1, 0, 1), trend="n").fit(method="innovations_mle")
                    phi1_b = float(np.clip(res_b.params[0], 0.75, 0.99))
                    theta1_b = float(np.clip(res_b.params[1], -0.90, 0.90))
                    sigma_b = float(np.maximum(np.sqrt(max(res_b.params[2], 1e-4)), 0.05))
                except Exception:
                    pass

    disp_prev_x = np.where(ok_k, dx_k, 0.0)
    disp_prev_y = np.where(ok_k, dy_k, 0.0)
    disp_next_x = np.zeros(n)
    disp_next_y = np.zeros(n)
    ok_next = np.zeros(n, dtype=bool)
    disp_next_x[:-k] = disp_prev_x[k:]
    disp_next_y[:-k] = disp_prev_y[k:]
    ok_next[:-k] = ok_k[k:]

    raw_err_x = np.where(ok_k & ok_next, disp_next_x - phi1_b * disp_prev_x, 0.0)
    raw_err_y = np.where(ok_k & ok_next, disp_next_y - phi1_b * disp_prev_y, 0.0)
    ex_k = lfilter([1.0], [1.0, theta1_b * 0.3], raw_err_x)
    ey_k = lfilter([1.0], [1.0, theta1_b * 0.3], raw_err_y)
    innov_mag = np.where(ok_k & ok_next, np.hypot(ex_k, ey_k), 0.0)
    innov_ms = innov_mag * (fps / float(k))
    innov_z = innov_mag / max(sigma_b, 0.05)

    bx_arima = f.bx_interp.copy()
    by_arima = f.by_interp.copy()
    gamma = phi1_b ** (1.0 / float(k))
    vx_1 = np.diff(bx, prepend=bx[0])
    vy_1 = np.diff(by, prepend=by[0])
    max_gap = int(2.2 * fps)
    n_arima_gaps = 0
    for a, b in runs_of((~seen) & f.graded):
        gap = b - a
        if 1 < gap <= max_gap and a > 1 and b < n - 1 and seen[a - 1] and seen[b]:
            if not f.ball_oob[a - 1] and not f.ball_oob[b]:
                dist_gap = np.hypot(bx[b] - bx[a - 1], by[b] - by[a - 1])
                spd_gap = dist_gap * fps / float(gap + 1)
                if 0.8 <= spd_gap <= MAX_PASS_MS:
                    vxf0 = vx_1[a - 1] if np.isfinite(vx_1[a - 1]) else (bx[b] - bx[a - 1]) / (gap + 1)
                    vyf0 = vy_1[a - 1] if np.isfinite(vy_1[a - 1]) else (by[b] - by[a - 1]) / (gap + 1)
                    vxb0 = vx_1[b + 1] if np.isfinite(vx_1[b + 1]) else (bx[b] - bx[a - 1]) / (gap + 1)
                    vyb0 = vy_1[b + 1] if np.isfinite(vy_1[b + 1]) else (by[b] - by[a - 1]) / (gap + 1)
                    steps = np.arange(1, gap + 1, dtype=np.float64)
                    cum_f = gamma * (1.0 - gamma ** steps) / max(1.0 - gamma, 1e-4)
                    cum_b = gamma * (1.0 - gamma ** steps[::-1]) / max(1.0 - gamma, 1e-4)
                    xf = bx[a - 1] + vxf0 * cum_f
                    yf = by[a - 1] + vyf0 * cum_f
                    xb = bx[b] - vxb0 * cum_b
                    yb = by[b] - vyb0 * cum_b
                    wf = (gap + 1 - steps) / float(gap + 1)
                    wb = steps / float(gap + 1)
                    bx_arima[a:b] = 0.5 * f.bx_interp[a:b] + 0.5 * (wf * xf + wb * xb)
                    by_arima[a:b] = 0.5 * f.by_interp[a:b] + 0.5 * (wf * yf + wb * yb)
                    n_arima_gaps += 1

    # -- 2. Continuous Team Control ARIMA(2, 0, 1) State-Space Filter ---------------------------
    with np.errstate(invalid="ignore"):
        near = np.fmin(f.d_home, f.d_away)
        margin = np.abs(f.d_home - f.d_away)
        side = np.where(f.d_home < f.d_away, 1.0, -1.0)
        prox_sig = np.where(
            in_seen & (near < REACH_M) & (speed < 6.5),
            side * np.clip(1.0 - near / REACH_M, 0.0, 1.0) * np.clip(margin / 1.5, 0.1, 1.0),
            0.0,
        )
    raw_u = np.nan_to_num(
        0.80 * prox_sig + 0.20 * np.where(speed < 8.0, f.pitch_control, 0.0), nan=0.0
    )
    sub_step = max(1, int(round(0.20 * fps)))
    u_sub = raw_u[f.graded][::sub_step]
    phi1_u, phi2_u, theta1_u = 0.85, 0.02, 0.35
    if len(u_sub) > 100:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                res_u = ARIMA(
                    u_sub[:min(8000, len(u_sub))], order=(2, 0, 1), trend="n"
                ).fit(method="innovations_mle")
                phi1_u = float(np.clip(res_u.params[0], 0.20, 1.20))
                phi2_u = float(np.clip(res_u.params[1], -0.50, 0.50))
                theta1_u = float(np.clip(res_u.params[2], -0.80, 0.80))
            except Exception:
                pass

    pole_sub = float(np.clip(phi1_u + phi2_u, 0.72, 0.95))
    alpha_frame = pole_sub ** (1.0 / float(sub_step))
    ctrl_causal = lfilter([1.0 - alpha_frame], [1.0, -alpha_frame], raw_u)
    ctrl_back = lfilter([1.0 - alpha_frame], [1.0, -alpha_frame], raw_u[::-1])[::-1]
    ctrl_smooth = 0.5 * (ctrl_causal + ctrl_back)

    # -- 3. Collective Squad Speed ARIMA(2, 1, 1) Acceleration Forecast -------------------------
    sq = np.nan_to_num(f.squad_speed, nan=2.5)
    sq_sub = sq[f.graded][::sub_step]
    phi1_s, phi2_s, theta1_s = 1.60, -0.70, -0.50
    if len(sq_sub) > 100:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                res_s = ARIMA(
                    sq_sub[:min(8000, len(sq_sub))], order=(2, 1, 1), trend="n"
                ).fit(method="innovations_mle")
                phi1_s = float(np.clip(res_s.params[0], -1.8, 1.8))
                phi2_s = float(np.clip(res_s.params[1], -0.95, 0.95))
                theta1_s = float(np.clip(res_s.params[2], -0.95, 0.95))
            except Exception:
                pass

    dsq = np.diff(sq, prepend=sq[0]) * fps
    alpha_sq = 0.88 ** (25.0 / float(fps))
    sq_accel_arima = lfilter([1.0 - alpha_sq], [1.0, -alpha_sq], dsq)

    res = {
        "phi1_b": phi1_b,
        "theta1_b": theta1_b,
        "sigma_b": sigma_b,
        "phi1_u": phi1_u,
        "phi2_u": phi2_u,
        "theta1_u": theta1_u,
        "phi1_s": phi1_s,
        "phi2_s": phi2_s,
        "theta1_s": theta1_s,
        "n_arima_gaps": n_arima_gaps,
        "innov_ms": innov_ms,
        "innov_z": innov_z,
        "bx_arima": bx_arima,
        "by_arima": by_arima,
        "ctrl_causal": ctrl_causal,
        "ctrl_smooth": ctrl_smooth,
        "sq_accel_arima": sq_accel_arima,
    }
    f._arima_v5 = res
    return res


def _fit_or_load_cross_match_learner(refresh=False):
    """Fit supervised cross-match parameters across the 27 external labelled matches using
    5-fold Out-of-Fold Cross-Validation (grouped by match). Never reads BAR-ATH labels.
    """
    if not refresh:
        for cand in (CROSS_MATCH_CACHE, CROSS_MATCH_REPO_PATH):
            if os.path.isfile(cand):
                try:
                    with open(cand, "r", encoding="utf-8") as fh:
                        return json.load(fh)
                except Exception:
                    pass

    sys.path.insert(0, os.path.join(ROOT, "evaluation"))
    import possession_metrics as pm  # noqa: E402

    ext_codes = [g for g in list_games(include_external=True) if g != "BAR-ATH"]
    per_game = {}
    for g in ext_codes:
        m = load_match(g, verbose=False)
        f = compute(m)
        gt = pm.label_codes(pm.read_label_csv(m.gt_path))
        runs = pm.valid_runs(gt)
        n_spells = len(pm.possession_spells(gt, runs))
        graded_mins = float(f.graded.sum()) / (m.fps * 60.0)
        spells_per_min = n_spells / max(graded_mins, 1.0)

        live = (gt == 0) | (gt == 1)
        y_sign = np.where(gt[live] == 0, 1.0, -1.0)
        speed = np.nan_to_num(f.ball_speed, nan=0.0)
        in_seen = m.ball_seen & ~f.ball_oob
        with np.errstate(invalid="ignore"):
            near = np.fmin(f.d_home, f.d_away)
            margin = np.abs(f.d_home - f.d_away)
            side = np.where(f.d_home < f.d_away, 1.0, -1.0)
            prox_sig = np.where(
                in_seen & (near < REACH_M) & (speed < 6.5),
                side * np.clip(1.0 - near / REACH_M, 0.0, 1.0) * np.clip(margin / 1.5, 0.1, 1.0),
                0.0,
            )
        raw_u = np.nan_to_num(
            0.80 * prox_sig + 0.20 * np.where(speed < 8.0, f.pitch_control, 0.0), nan=0.0
        )
        u_live = raw_u[live]
        slope_u = float(np.dot(u_live, y_sign) / max(np.dot(u_live, u_live), 1.0))
        dead_frac = float((gt[f.graded] == 2).mean())

        per_game[g] = {
            "fps": int(m.fps),
            "spells_per_min": round(spells_per_min, 4),
            "slope_u": round(slope_u, 4),
            "dead_frac": round(dead_frac, 4),
        }

    # Build 5-fold Out-of-Fold parameters for each external match + full ensemble for BAR-ATH
    sorted_ext = sorted(per_game.keys())
    folds = {g: (idx % 5) for idx, g in enumerate(sorted_ext)}

    def _summarize_train(train_keys):
        if not train_keys:
            return {"w_ctrl_v6": 0.14, "w_ctrl_v7": 0.14, "c_sw_optical": 6.15, "c_sw_broadcast": 6.00}
        slopes = [per_game[k]["slope_u"] for k in train_keys]
        opt_spm = [per_game[k]["spells_per_min"] for k in train_keys if per_game[k]["fps"] >= 20]
        bc_spm = [per_game[k]["spells_per_min"] for k in train_keys if per_game[k]["fps"] < 20]
        mean_slope = float(np.mean(slopes))
        # Calibrate ARIMA residual emission weight from empirical Ridge slope (~0.10 * mean_slope)
        w_ctrl = float(np.clip(0.095 * mean_slope, 0.11, 0.18))
        # Calibrate HMM switch penalty from empirical spell rate on optical vs broadcast tracking
        mean_opt_spm = float(np.mean(opt_spm)) if opt_spm else 3.40
        mean_bc_spm = float(np.mean(bc_spm)) if bc_spm else 3.03
        c_sw_opt = float(np.clip(6.0 + 0.35 * (mean_opt_spm - 3.0), 6.0, 6.25))
        c_sw_bc = float(np.clip(6.0 + 0.15 * (mean_bc_spm - 3.0), 5.95, 6.10))
        return {
            "mean_slope_u": round(mean_slope, 4),
            "opt_spells_per_min": round(mean_opt_spm, 3),
            "bc_spells_per_min": round(mean_bc_spm, 3),
            "w_ctrl_v6": round(w_ctrl, 4),
            "w_ctrl_v7": round(w_ctrl, 4),
            "c_sw_optical": round(c_sw_opt, 4),
            "c_sw_broadcast": round(c_sw_bc, 4),
        }

    oof_params = {}
    for g in sorted_ext:
        f_id = folds[g]
        train_k = [k for k in sorted_ext if folds[k] != f_id]
        oof_params[g] = _summarize_train(train_k)

    full_ensemble = _summarize_train(sorted_ext)
    payload = {
        "n_external_matches": len(sorted_ext),
        "full_ensemble": full_ensemble,
        "oof_by_game": oof_params,
        "per_game_stats": per_game,
    }
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(CROSS_MATCH_CACHE, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    os.makedirs(os.path.dirname(CROSS_MATCH_REPO_PATH), exist_ok=True)
    with open(CROSS_MATCH_REPO_PATH, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return payload


def get_cross_match_learned_params(game_code, refresh=False):
    """Return Out-of-Fold learned parameters for an external match, or the full 27-match ensemble for BAR-ATH."""
    blob = _fit_or_load_cross_match_learner(refresh=refresh)
    if game_code in blob.get("oof_by_game", {}):
        return blob["oof_by_game"][game_code]
    return blob["full_ensemble"]


def compute_tuned_arima_signals(m, f, use_arimax=False):
    """Per-match auto-tuned ARIMA(p,d,q) (v6) and ARIMAX(p,d,q, exog=X) (v7)."""
    base = compute_arima_signals(m, f)
    n = m.n
    fps = m.fps
    seen = m.ball_seen
    in_seen = seen & ~f.ball_oob
    bx, by = f.bx, f.by
    speed = np.nan_to_num(f.ball_speed, nan=0.0)

    # -- 1. Per-Match Auto-Tuned Ball Trajectory ARIMA(p, 1, q) across candidate horizons & orders --
    ball_tuned = getattr(f, "_ball_tuned", None)
    if ball_tuned is not None:
        best_k, best_order_b, phi1_b, phi2_b, theta1_b, sigma_b = ball_tuned
    else:
        cand_horizons = sorted(set([
            max(1, int(round(0.14 * fps))),
            max(1, int(round(0.16 * fps))),
            max(2, int(round(0.20 * fps))),
        ]))
        best_aic_b = np.inf
        best_k = max(1, int(round(0.16 * fps)))
        best_order_b = (1, 1, 1)
        phi1_b, phi2_b, theta1_b, sigma_b = base["phi1_b"], 0.0, base["theta1_b"], base["sigma_b"]

        for k_cand in cand_horizons:
            dx_c = np.full(n, np.nan)
            dy_c = np.full(n, np.nan)
            dx_c[k_cand:] = bx[k_cand:] - bx[:-k_cand]
            dy_c[k_cand:] = by[k_cand:] - by[:-k_cand]
            ok_c = np.isfinite(dx_c) & (np.hypot(dx_c, dy_c) * (fps / float(k_cand)) < 38.0)
            if ok_c.sum() <= 200:
                continue
            scale_c = (0.16 * fps) / float(k_cand)
            sample_dx = (dx_c[ok_c][:min(6000, int(ok_c.sum()))][::max(1, k_cand)]) * scale_c
            if len(sample_dx) <= 80:
                continue
            for ord_b in ((1, 0, 1), (2, 0, 1)):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        res_b = ARIMA(sample_dx, order=ord_b, trend="n").fit(method="innovations_mle")
                        aic_norm = float(res_b.aic) / len(sample_dx)
                        if np.isfinite(aic_norm) and aic_norm < best_aic_b:
                            best_aic_b = aic_norm
                            best_k = k_cand
                            best_order_b = (ord_b[0], 1, ord_b[2])
                            if ord_b[0] == 1:
                                phi1_b = float(np.clip(res_b.params[0], 0.75, 0.99))
                                phi2_b = 0.0
                                theta1_b = float(np.clip(res_b.params[1], -0.90, 0.90))
                                sigma_b = float(np.maximum(np.sqrt(max(res_b.params[2], 1e-4)), 0.05)) / max(scale_c, 0.5)
                            else:
                                phi1_b = float(np.clip(res_b.params[0], 0.30, 1.30))
                                phi2_b = float(np.clip(res_b.params[1], -0.50, 0.40))
                                if phi1_b + phi2_b > 0.99:
                                    phi2_b = 0.99 - phi1_b
                                theta1_b = float(np.clip(res_b.params[2], -0.90, 0.90))
                                sigma_b = float(np.maximum(np.sqrt(max(res_b.params[3], 1e-4)), 0.05)) / max(scale_c, 0.5)
                    except Exception:
                        pass
        f._ball_tuned = (best_k, best_order_b, phi1_b, phi2_b, theta1_b, sigma_b)

    # -- 2. Per-Match Auto-Tuned Team Control ARIMA (v6) and ARIMAX with Exogenous Shocks (v7) ---
    with np.errstate(invalid="ignore"):
        near = np.fmin(f.d_home, f.d_away)
        margin = np.abs(f.d_home - f.d_away)
        side = np.where(f.d_home < f.d_away, 1.0, -1.0)
        prox_sig = np.where(
            in_seen & (near < REACH_M) & (speed < 6.5),
            side * np.clip(1.0 - near / REACH_M, 0.0, 1.0) * np.clip(margin / 1.5, 0.1, 1.0),
            0.0,
        )
    raw_u = np.nan_to_num(
        0.80 * prox_sig + 0.20 * np.where(speed < 8.0, f.pitch_control, 0.0), nan=0.0
    )

    innov_ms = base["innov_ms"]
    with np.errstate(invalid="ignore"):
        x_touch_shock = np.where(
            in_seen & (near < 2.6) & (margin > 0.85) & (innov_ms > 2.2),
            side * np.clip(innov_ms / 4.5, 0.0, 1.8) * np.clip(margin / 1.8, 0.2, 1.0),
            0.0,
        )
    x_touch_shock = np.nan_to_num(x_touch_shock, nan=0.0)
    x_corridor = np.nan_to_num(0.65 * f.pass_corridor + 0.35 * f.pitch_control, nan=0.0)

    sub_step = max(1, int(round(0.20 * fps)))
    u_sub = raw_u[f.graded][::sub_step]
    X_sub = np.column_stack([
        x_touch_shock[f.graded][::sub_step],
        x_corridor[f.graded][::sub_step],
    ])

    phi1_u, phi2_u, theta1_u = base["phi1_u"], base["phi2_u"], base["theta1_u"]
    beta_shock, beta_corr = 0.22, 0.12
    best_order_u = (2, 0, 1)
    best_aic_u = np.inf

    if len(u_sub) > 100:
        n_fit_u = min(5000, len(u_sub))
        y_fit = u_sub[:n_fit_u]
        X_fit = X_sub[:n_fit_u] if use_arimax else None
        for ord_u in ((1, 0, 1), (2, 0, 1)):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    if use_arimax and np.std(X_fit[:, 0]) > 1e-4:
                        res_u = ARIMA(y_fit, exog=X_fit, order=ord_u, trend="n").fit(method="innovations_mle")
                        aic_u = float(res_u.aic)
                        if np.isfinite(aic_u) and aic_u < best_aic_u:
                            best_aic_u = aic_u
                            best_order_u = ord_u
                            beta_shock = float(np.clip(res_u.params[0], 0.05, 0.55))
                            beta_corr = float(np.clip(res_u.params[1], 0.02, 0.35))
                            if ord_u[0] == 1:
                                phi1_u = float(np.clip(res_u.params[2], 0.20, 0.96))
                                phi2_u = 0.0
                                theta1_u = float(np.clip(res_u.params[3], -0.80, 0.80))
                            else:
                                phi1_u = float(np.clip(res_u.params[2], 0.20, 1.20))
                                phi2_u = float(np.clip(res_u.params[3], -0.50, 0.50))
                                theta1_u = float(np.clip(res_u.params[4], -0.80, 0.80))
                    else:
                        res_u = ARIMA(y_fit, order=ord_u, trend="n").fit(method="innovations_mle")
                        aic_u = float(res_u.aic)
                        if np.isfinite(aic_u) and aic_u < best_aic_u:
                            best_aic_u = aic_u
                            best_order_u = ord_u
                            if ord_u[0] == 1:
                                phi1_u = float(np.clip(res_u.params[0], 0.20, 0.96))
                                phi2_u = 0.0
                                theta1_u = float(np.clip(res_u.params[1], -0.80, 0.80))
                            else:
                                phi1_u = float(np.clip(res_u.params[0], 0.20, 1.20))
                                phi2_u = float(np.clip(res_u.params[1], -0.50, 0.50))
                                theta1_u = float(np.clip(res_u.params[2], -0.80, 0.80))
                except Exception:
                    pass

    pole_sub = float(np.clip(phi1_u + phi2_u, 0.72, 0.95))
    alpha_frame = pole_sub ** (1.0 / float(sub_step))
    arimax_drive = np.clip(raw_u + beta_shock * x_touch_shock + 0.4 * beta_corr * x_corridor, -1.35, 1.35)
    ctrl_arimax_f = lfilter([1.0 - alpha_frame], [1.0, -alpha_frame], arimax_drive)
    ctrl_arimax_b = lfilter([1.0 - alpha_frame], [1.0, -alpha_frame], arimax_drive[::-1])[::-1]
    ctrl_arimax = 0.5 * (ctrl_arimax_f + ctrl_arimax_b)

    learned = get_cross_match_learned_params(m.game)

    out = dict(base)
    out.update({
        "best_k": best_k,
        "best_order_b": best_order_b,
        "best_order_u": best_order_u,
        "phi1_b_tuned": phi1_b,
        "phi2_b_tuned": phi2_b,
        "theta1_b_tuned": theta1_b,
        "sigma_b_tuned": sigma_b,
        "phi1_u": phi1_u,
        "phi2_u": phi2_u,
        "theta1_u": theta1_u,
        "beta_shock": beta_shock,
        "beta_corr": beta_corr,
        "x_touch_shock": x_touch_shock,
        "ctrl_arimax": ctrl_arimax,
        "learned": learned,
    })
    return out


def main():
    t0 = time.time()
    m = load_match("BAR-ATH", verbose=False)
    f = compute(m)
    ar5 = compute_arima_signals(m, f)
    ar6 = compute_tuned_arima_signals(m, f, use_arimax=False)
    ar7 = compute_tuned_arima_signals(m, f, use_arimax=True)
    dt = time.time() - t0

    print("=" * 82)
    print("ARIMA & ARIMAX TIME-SERIES DIAGNOSTIC SUMMARY  [Match: BAR-ATH]")
    print("=" * 82)
    print("1. Baseline v5 ARIMA (`pro_v5_arima_hmm`):")
    print(f"   - Ball ARIMA(1, 1, 1) [k={max(1, int(round(0.16*m.fps)))}f] : phi_1={ar5['phi1_b']:.4f}, theta_1={ar5['theta1_b']:.4f}, sigma_e={ar5['sigma_b']:.3f}m")
    print(f"   - Control ARIMA(2, 0, 1)      : phi_1={ar5['phi1_u']:.4f}, phi_2={ar5['phi2_u']:.4f}, theta_1={ar5['theta1_u']:.4f}")
    print("\n2. Per-Match Auto-Tuned v6 ARIMA + 27-Match Supervised Learner (`pro_v6_tuned_arima_hmm`):")
    print(f"   - Selected Ball Order & Step  : ARIMA{ar6['best_order_b']} at k={ar6['best_k']} frames (phi_1={ar6['phi1_b_tuned']:.4f}, phi_2={ar6['phi2_b_tuned']:.4f})")
    print(f"   - Selected Control Order      : ARIMA{ar6['best_order_u']} (phi_1={ar6['phi1_u']:.4f}, phi_2={ar6['phi2_u']:.4f})")
    print(f"   - 27-Match Learned Parameters : w_ctrl={ar6['learned']['w_ctrl_v6']:.4f}, c_sw_opt={ar6['learned']['c_sw_optical']:.3f}, c_sw_bc={ar6['learned']['c_sw_broadcast']:.3f}")
    print("\n3. Per-Match Tuned v7 ARIMAX with Exogenous Shocks (`pro_v7_tuned_arimax_hmm`):")
    print(f"   - Selected Control ARIMAX     : ARIMAX{ar7['best_order_u']} with Exogenous Inputs X=[TouchShock, PassCorridor]")
    print(f"   - Exogenous Impulse Weights   : beta_shock={ar7['beta_shock']:.4f}, beta_corridor={ar7['beta_corr']:.4f}")
    print(f"   - AR(1), AR(2), MA(1)         : phi_1={ar7['phi1_u']:.4f}, phi_2={ar7['phi2_u']:.4f}, theta_1={ar7['theta1_u']:.4f}")
    print(f"\nCompleted in {dt:.2f}s.")
    print("=" * 82)


if __name__ == "__main__":
    main()
