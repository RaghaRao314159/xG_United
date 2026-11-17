"""Lets the IDSSE helpers treat BAR-ATH as one more 'match': tabular.dataset_name maps the name
BAR-ATH to runs/BAR-ATH whatever the condition, and tabular.gt_codes gives it an all-X truth
(BAR-ATH labels are never read; its rows are only ever test rows). Import after common."""
import os as _os
ROOT = _os.environ.get("MTAH_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))))
LAB = _os.environ.get("MTAH_LAB") or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
LABCACHE = _os.path.join(ROOT, "runs", "lab-cache")

import os

import sys

import numpy as np

sys.path.insert(0, LAB + "/exp")  # common.py, when imported on its own

from common import pipeline, tb

TARGET = "BAR-ATH"
_name, _gt = tb.dataset_name, tb.gt_codes


def dataset_name(match, cond):
    return TARGET if match == TARGET else _name(match, cond)


def gt_codes(dataset):
    if dataset == TARGET:
        meta, _ = pipeline.load_dataset(TARGET)
        return np.full(meta["n_frames"], tb.pm.X, dtype=np.int8)
    return _gt(dataset)


tb.dataset_name = dataset_name
tb.gt_codes = gt_codes


def ensure_raw(match, cond):
    """runs/idsse-<m>-<cond>/raw_tb.npz: the clean copy with the ball hidden exactly where that
    condition's features.npz hides it (Track B made the d85 ones; d70 / d95 are made here)."""
    folder = pipeline.dataset_dir(tb.dataset_name(match, cond))
    path = os.path.join(folder, "raw_tb.npz")
    if os.path.exists(path):
        return
    with np.load(os.path.join(pipeline.dataset_dir(tb.dataset_name(match, "clean")),
                              "raw_tb.npz")) as d:
        raw = {k: d[k] for k in d.files}
    _, f = pipeline.load_dataset(tb.dataset_name(match, cond))
    clean_seen = ~np.isnan(raw["ball"][:, 0])
    seen = f["seen"].astype(bool)
    assert len(seen) == len(clean_seen) and not (seen & ~clean_seen).any(), "hidden mask"
    raw["ball"] = raw["ball"].copy()
    raw["ball"][~seen] = np.nan
    np.savez_compressed(path, **raw)
