"""Run a lab script unchanged and save every model it fits: LightGBM boosters (save_model text) and the
Kumo Tabular in-context sets (the rows and labels handed to KumoTabular.fit), plus the column indices
chosen by kt / kt_pa select_columns and the feature names seen by the symmetric fitters.

    python wcap.py OUTDIR TAG script.py [args ...]
    python wcap.py OUTDIR TAG -c "python code"
"""
import importlib.abc
import itertools
import json
import os
import runpy
import sys
import time

import numpy as np

OUT, TAG = sys.argv[1], sys.argv[2]
SEQ = itertools.count()
os.makedirs(OUT, exist_ok=True)


def stack():
    fr, out = sys._getframe(2), []
    while fr is not None and len(out) < 14:
        out.append("{}:{}".format(os.path.basename(fr.f_code.co_filename), fr.f_code.co_name))
        fr = fr.f_back
    return out


def save_json(stem, meta):
    meta.update(tag=TAG, time=time.strftime("%Y-%m-%d %H:%M:%S"), stack=stack())
    with open(stem + ".json", "w") as handle:
        json.dump(meta, handle, indent=1, default=str)


def stem(kind):
    return os.path.join(OUT, "{}_{:03d}_{}".format(TAG, next(SEQ), kind))


# -- LightGBM ----------------------------------------------------------------------------------
import lightgbm  # noqa: E402

_lgb_fit = lightgbm.LGBMClassifier.fit


def lgb_fit(self, X, y, *a, **k):
    res = _lgb_fit(self, X, y, *a, **k)
    s = stem("lgbm")
    self.booster_.save_model(s + ".txt")
    save_json(s, {"kind": "lightgbm.LGBMClassifier", "n_rows": int(len(y)), "n_features": int(self.n_features_in_),
                  "classes": [int(c) for c in self.classes_], "params": self.get_params()})
    return res


lightgbm.LGBMClassifier.fit = lgb_fit

# -- Kumo Tabular ------------------------------------------------------------------------------
try:
    import sdm.models as _sm

    _kumo_fit = _sm.KumoTabular.fit

    def kumo_fit(self, x, y, *a, **k):
        s = stem("kumo")
        xc = x.detach().float().cpu().numpy()
        yc = y.detach().cpu().numpy()
        np.savez_compressed(s + ".npz", x_context=xc, y_context=yc)
        save_json(s, {"kind": "sdm.models.KumoTabular in-context set", "context_rows": int(len(yc)),
                      "n_columns": int(xc.shape[1]), "fit_args": [str(v) for v in a],
                      "fit_kwargs": {kk: str(vv) for kk, vv in k.items()}})
        return _kumo_fit(self, x, y, *a, **k)

    _sm.KumoTabular.fit = kumo_fit
except ImportError:
    pass


# -- lab functions, patched as their modules load ----------------------------------------------
def wrap_select(module):
    orig = module.select_columns

    def select_columns(x, y, k, seed):
        cols = orig(x, y, k, seed)
        s = stem("cols")
        np.save(s + ".npy", np.asarray(cols))
        save_json(s, {"kind": "selected column indices", "module": module.__name__, "k": k, "seed": seed,
                      "n_input_columns": int(x.shape[1])})
        return cols

    module.select_columns = select_columns


def wrap_names(module, fname, pos):
    orig = getattr(module, fname)

    def wrapped(*a, **k):
        s = stem("names")
        names = a[pos] if len(a) > pos else None
        save_json(s, {"kind": "feature names", "function": module.__name__ + "." + fname,
                      "names": list(names) if names is not None else None})
        return orig(*a, **k)

    setattr(module, fname, wrapped)


PATCH = {
    "kt": lambda m: wrap_select(m),
    "kt_pa": lambda m: (wrap_select(m), wrap_names(m, "symmetric", 4)),
    "tabular": lambda m: wrap_names(m, "symmetric_proba", 4),
}


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name not in PATCH:
            return None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(name, path, target)
            if spec is not None:
                loader_exec = spec.loader.exec_module

                def exec_module(module, loader_exec=loader_exec):
                    loader_exec(module)
                    PATCH[name](module)
                spec.loader.exec_module = exec_module
                return spec
        return None


sys.meta_path.insert(0, Finder())

# -- run ---------------------------------------------------------------------------------------
if sys.argv[3] == "-c":
    code = sys.argv[4]
    sys.argv = ["-c"] + sys.argv[5:]
    exec(compile(code, "<wcap -c>", "exec"), {"__name__": "__main__"})
else:
    script = os.path.abspath(sys.argv[3])
    sys.argv = [script] + sys.argv[4:]
    sys.path.insert(0, os.path.dirname(script))
    runpy.run_path(script, run_name="__main__")
