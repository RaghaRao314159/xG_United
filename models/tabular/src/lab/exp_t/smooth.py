"""Make IDSSE's ball track as smooth as BAR-ATH's: a Gaussian filter (sigma in frames) over each
contiguous stretch of seen ball positions in runs/idsse-<m>-d85/raw_tb.npz, sigma chosen so the
acceleration percentiles of seen frames (p50 / p90 / p99) match BAR-ATH's -- label-free. `pick`
prints the grid; `write --sigma s` builds condition d85s: runs/idsse-<m>-d85s/ with features.npz,
meta.json and gt.csv linked from d85 and the smoothed raw_tb.npz. Aggregates only.

    MTAH_ROOT=/root/mtah python smooth.py pick
    MTAH_ROOT=/root/mtah python smooth.py write --sigma 2
"""
import os as _os
ROOT = _os.environ.get("MTAH_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))))
LAB = _os.environ.get("MTAH_LAB") or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
LABCACHE = _os.path.join(ROOT, "runs", "lab-cache")

import argparse
import os
import sys

import numpy as np
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, LAB + "/exp")
sys.path.insert(0, LAB + "/exp_b")
from common import pipeline, tb  # noqa: E402
import feat_tb  # noqa: E402

SIGMAS = (0.0, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
Q = (50, 90, 99)


def load_raw(dataset):
    with np.load(os.path.join(pipeline.dataset_dir(dataset), "raw_tb.npz")) as d:
        return {k: d[k] for k in d.files}


def smooth_ball(ball, sigma):
    out = ball.copy()
    if sigma <= 0:
        return out
    seen = ~np.isnan(ball[:, 0])
    starts, stops = tb.run_bounds(seen)
    for a, b in zip(starts, stops):
        for k in (0, 1):
            out[a:b, k] = gaussian_filter1d(ball[a:b, k].astype(np.float64), sigma, mode="nearest")
    return out


def acc_of(raw, meta):
    length, width = [float(v) for v in raw["pitch"]]
    b = raw["ball"].astype(np.float32) * np.array([length, width], dtype=np.float32)
    v = feat_tb.deriv(b, feat_tb.LAG)
    spd = np.linalg.norm(v, axis=1)
    v[spd == 0] = np.nan
    acc = np.linalg.norm(feat_tb.deriv(v, feat_tb.LAG), axis=1)
    graded = np.zeros(len(b), dtype=bool)
    for _, first, last in meta["halves"]:
        graded[first:last + 1] = True
    a = acc[graded & np.isfinite(acc)]
    return np.percentile(a, Q)


def cmd_pick(args):
    meta, _ = pipeline.load_dataset("BAR-ATH")
    target = acc_of(load_raw("BAR-ATH"), meta)
    print("BAR-ATH acc p50/p90/p99 {}".format(target.round(1).tolist()))
    for sigma in SIGMAS:
        got = []
        for m in tb.built_matches():
            ds = tb.dataset_name(m, "d85")
            meta, _ = pipeline.load_dataset(ds)
            raw = load_raw(ds)
            raw["ball"] = smooth_ball(raw["ball"], sigma)
            got.append(acc_of(raw, meta))
        got = np.mean(got, axis=0)
        err = float(np.abs(np.log(got[1:] / target[1:])).sum())
        print("  sigma {:g}: IDSSE d85 acc p50/p90/p99 {} | log error (p90 + p99) {:.3f}".format(
            sigma, got.round(1).tolist(), err), flush=True)


def cmd_write(args):
    for m in tb.built_matches():
        src = pipeline.dataset_dir(tb.dataset_name(m, "d85"))
        dst = pipeline.dataset_dir(tb.dataset_name(m, "d85s"))
        os.makedirs(os.path.join(dst, "results"), exist_ok=True)
        for name in ("features.npz", "meta.json", "gt.csv"):
            if not os.path.exists(os.path.join(dst, name)):
                os.symlink(os.path.join(src, name), os.path.join(dst, name))
        raw = load_raw(tb.dataset_name(m, "d85"))
        raw["ball"] = smooth_ball(raw["ball"], args.sigma).astype(np.float32)
        np.savez_compressed(os.path.join(dst, "raw_tb.npz"), **raw)
        print("  wrote {} (sigma {:g})".format(os.path.basename(dst), args.sigma), flush=True)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("pick")
    w = sub.add_parser("write")
    w.add_argument("--sigma", type=float, required=True)
    args = parser.parse_args()
    {"pick": cmd_pick, "write": cmd_write}[args.command](args)


if __name__ == "__main__":
    main()
