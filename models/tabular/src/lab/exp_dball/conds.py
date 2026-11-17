"""Ball-visible IDSSE conditions in the challenge's convention, for the dead-ball work.

  d50su    exp_sc/hard_idsse.py's condition (built here if missing): BAR-ATH's gap lengths and hidden
           share, but only half of the hidden time inside stoppages, so the ball stays in view through
           about half of the dead time (d85su: about a fifth)
  cleansu  exp_u/unfreeze.py's cleanu copy (the ball in view through every stoppage) smoothed like
           d85su and d50su (sigma 6, features.npz recomputed), built here the way hard_idsse.py builds
           its conditions but with nothing hidden; cleanu itself keeps IDSSE's raw, much rougher ball

For each, <cond>c gets every file of <cond> as a symlink and gt.csv = d85suc's (Track C's R2 relabel
of the same clean truth, k = 0.6; the D frames are the clean truth's in every copy). So every
condition shares one truth and differs only in what the ball does: P(alive) answers from any of them
can be decoded with the fixed d85suc team caches and pooled against each other. pfeat.py's cache
(player-only columns, identical in every copy) is linked from d85su after checking the players match.

    MTAH_ROOT=/root/mtah python conds.py
Prints aggregates only.
"""
import os as _os
ROOT = _os.environ.get("MTAH_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))))
LAB = _os.environ.get("MTAH_LAB") or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
LABCACHE = _os.path.join(ROOT, "runs", "lab-cache")

import os
import shutil
import subprocess
import sys

import numpy as np

sys.path.insert(0, LAB + "/exp")
sys.path.insert(0, LAB + "/exp_t")
sys.path.insert(0, LAB + "/exp_u")
from common import pipeline, tb  # noqa: E402

CONDS = ("d50su", "cleansu")
PF = "pf_cols_w12.npz"
TEAM_CACHE = os.path.join(LABCACHE, "exp_u", "{}_{}_{}.npz")
TEAMS = ("kk4",)
SIGMA = 6.0   # d85su's and hard_idsse.py's


def players_of(dataset):
    with np.load(os.path.join(pipeline.dataset_dir(dataset), "raw_tb.npz")) as d:
        return d["players"]


def built(cond, matches):
    return all(os.path.exists(os.path.join(pipeline.dataset_dir(tb.dataset_name(m, cond)), "features.npz"))
               for m in matches)


def build_cleansu(matches):
    import smooth  # noqa: E402  (Track T)
    import unfreeze  # noqa: E402
    for m in matches:
        src = tb.dataset_name(m, "cleanu")
        raw = smooth.load_raw(src)
        meta, _ = pipeline.load_dataset(src)
        ball = smooth.smooth_ball(raw["ball"], SIGMA).astype(np.float32)
        g = unfreeze.recompute(ball, raw)
        dst = tb.dataset_name(m, "cleansu")
        folder = pipeline.dataset_dir(dst)
        os.makedirs(os.path.join(folder, "results"), exist_ok=True)
        pipeline.write_dataset(dst, dict(meta, name=dst), g)
        shutil.copyfile(os.path.join(pipeline.dataset_dir(src), "gt.csv"), os.path.join(folder, "gt.csv"))
        np.savez_compressed(os.path.join(folder, "raw_tb.npz"), ball=ball,
                            ball_z=np.full(len(ball), np.nan, dtype=np.float32),
                            players=raw["players"], team=raw["team"], pitch=raw["pitch"])
        print("  {}: built (sigma {:g})".format(dst, SIGMA), flush=True)


def link_controlled(conds, matches):
    """<cond>c copies of each cond: symlinks, d85suc's controlled truth, pfeat cache, kk4 team caches."""
    for m in matches:
        clean = tb.gt_codes(tb.dataset_name(m, "clean"))
        ctl_dir = pipeline.dataset_dir(tb.dataset_name(m, "d85suc"))
        ctl = tb.gt_codes(tb.dataset_name(m, "d85suc"))
        assert ((clean == tb.D) == (ctl == tb.D)).all(), "d85suc D frames differ from the clean truth"
        ref_players = players_of(tb.dataset_name(m, "d85su"))
        pf_src = os.path.join(pipeline.dataset_dir(tb.dataset_name(m, "d85su")), PF)
        for cond in conds:
            src_ds = tb.dataset_name(m, cond)
            src = pipeline.dataset_dir(src_ds)
            assert (tb.gt_codes(src_ds) == clean).all(), (m, cond, "truth")
            if os.path.exists(pf_src) and not os.path.exists(os.path.join(src, PF)):
                assert np.array_equal(players_of(src_ds), ref_players, equal_nan=True), (m, cond, "players")
                os.symlink(pf_src, os.path.join(src, PF))
            dst = pipeline.dataset_dir(tb.dataset_name(m, cond + "c"))
            os.makedirs(os.path.join(dst, "results"), exist_ok=True)
            for name in os.listdir(src):
                if name in ("gt.csv", "results", "score") or name.startswith("answers_"):
                    continue
                link = os.path.join(dst, name)
                if not os.path.lexists(link):
                    os.symlink(os.path.join(src, name), link)
            shutil.copyfile(os.path.join(ctl_dir, "gt.csv"), os.path.join(dst, "gt.csv"))
            for team in TEAMS:    # the fixed team model: d85suc's caches serve every copy
                link = TEAM_CACHE.format(cond + "c", team, m)
                if not os.path.lexists(link):
                    os.symlink(TEAM_CACHE.format("d85suc", team, m), link)
            _, f = pipeline.load_dataset(src_ds)
            graded = clean != tb.pm.X
            dead = graded & (clean == tb.D)
            print("  {} {}c: ball in view on {:.1%} of graded frames, {:.1%} of dead and {:.1%} of live "
                  "frames".format(m, cond, f["seen"][graded].mean(), f["seen"][dead].mean(),
                                  f["seen"][graded & ~dead].mean()), flush=True)


def main():
    matches = tb.built_matches()
    if not built("d50su", matches):
        subprocess.run([sys.executable, "-W", "ignore", os.path.join(LAB, "exp_sc", "hard_idsse.py"),
                        "--cond", "d50su"], check=True)
    if not built("cleansu", matches):
        build_cleansu(matches)
    link_controlled(CONDS, matches)
    print("done", flush=True)


if __name__ == "__main__":
    main()
