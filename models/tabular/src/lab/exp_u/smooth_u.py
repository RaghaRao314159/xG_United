"""Track U: condition d85su = the unfrozen d85 copy (d85u) smoothed like BAR-ATH.

Track T's smooth.py smooths runs/idsse-<m>-d85/raw_tb.npz with a Gaussian (sigma = 6 frames, chosen
label-free so IDSSE's acceleration p90 / p99 match BAR-ATH's) over each contiguous stretch of seen
ball positions. Here the same filter (smooth.smooth_ball) is applied to the unfrozen d85u ball,
AFTER unfreezing, and -- unlike d85s, which keeps d85's features.npz -- features.npz is recomputed
from the smoothed ball with pipeline.frame_features, since BAR-ATH's features come from its own
smooth track. Writes runs/idsse-<m>-d85su/. Prints aggregates only.

    MTAH_ROOT=/root/mtah python smooth_u.py --sigma 6
"""
import os as _os
ROOT = _os.environ.get("MTAH_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))))
LAB = _os.environ.get("MTAH_LAB") or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
LABCACHE = _os.path.join(ROOT, "runs", "lab-cache")

import argparse
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, LAB + "/exp")
sys.path.insert(0, LAB + "/exp_t")
sys.path.insert(0, LAB + "/exp_u")
from common import pipeline, tb  # noqa: E402
import smooth  # noqa: E402  (Track T)
import unfreeze  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sigma", type=float, default=6.0)
    args = parser.parse_args()
    for m in tb.built_matches():
        src = tb.dataset_name(m, "d85u")
        raw = smooth.load_raw(src)
        meta, f = pipeline.load_dataset(src)
        ball = smooth.smooth_ball(raw["ball"], args.sigma).astype(np.float32)
        g = unfreeze.recompute(ball, raw)
        assert (g["seen"] == f["seen"]).all() and (g["gap"] == f["gap"]).all()
        dst = "idsse-{}-d85su".format(m)
        folder = pipeline.dataset_dir(dst)
        os.makedirs(folder, exist_ok=True)
        for name in ("meta.json", "gt.csv"):
            shutil.copyfile(os.path.join(pipeline.dataset_dir(src), name), os.path.join(folder, name))
        np.savez_compressed(os.path.join(folder, "features.npz"), **g)
        np.savez_compressed(os.path.join(folder, "raw_tb.npz"), ball=ball, ball_z=raw["ball_z"],
                            players=raw["players"], team=raw["team"], pitch=raw["pitch"])
        for name in os.listdir(folder):
            if name.startswith(("tb_cols_", "pass_cols_")):
                os.remove(os.path.join(folder, name))
        acc = smooth.acc_of({"ball": ball, "pitch": raw["pitch"]}, meta)
        print("{}: sigma {:g}, seen acc p50/p90/p99 {}".format(dst, args.sigma, acc.round(1).tolist()),
              flush=True)


if __name__ == "__main__":
    main()
