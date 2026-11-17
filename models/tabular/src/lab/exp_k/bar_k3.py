"""Hybrid BAR-ATH files: the team model and P(alive) may come from different training sources.

    members:  python bar_k3.py members --train idsse|both [--members lgb_b lgb_ab kt_b kt_ab]
              -> runs/lab-cache/exp_k/bar_members_<train>.npz (each member's P(H | alive) + block meta)
    decode:   python bar_k3.py decode --team-from idsse --palive-from sc --parts kt_b:1 kt_ab:1 lgb_b:1 lgb_ab:1 \
                  --cost 2 2 20 --mins 10 1 --name exp_k_kk4_tIDSSE_pSC
              P(alive) = answers_exp_k_all_pa_<palive-from>.jsonl written by bar_k2.py (LightGBM + Kumo
              3-class, logit 50/50, trained on that source); kick-off rules as bar_v3.py.
Prints aggregates only.
"""
import os as _os
LAB = _os.environ.get("MTAH_LAB") or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

import argparse
import os
import sys
import time

sys.path.insert(0, LAB + "/exp_k")
import bar_k2  # noqa: E402

import numpy as np  # noqa: E402

from bar_k2 import TARGET, bar_v2, dball, kick, kt, pipeline, tb, td, xfer, aggregates, lg  # noqa: E402

CACHE = os.path.join(os.path.dirname(kt.TABLES), "bar_members_{}.npz")
INFO = ("start", "end", "half", "win", "counts", "n_frames")


def members(args):
    bar_k2.TRAIN.update(bar_k2.train_set(args.train))
    tb.swap = xfer.ORIGINAL_SWAP
    pipeline.WINDOW = tb.WINDOW = pipeline.FPS
    started = time.time()
    out, info = {}, None
    for m in args.members:
        if m == "kft_b":
            out[m] = bar_k2.kumo_finetuned()
        elif m == "kr_b":
            out[m] = bar_k2.relational()
        else:
            out[m], info = bar_k2.team_member(m)
            bar_k2.free()
    if info is None:
        info = kt.table(TARGET, bar_k2.COND, "b")[1]
    key = args.train + ("" if bar_k2.COND == "d85suc" else "_" + bar_k2.COND)   # e.g. idsse_d85sue
    np.savez_compressed(CACHE.format(key), **out, **{"info_" + k: info[k] for k in INFO})
    print("members {} from {}: cached, {:.0f} s".format(list(out), key, time.time() - started), flush=True)


def decode(args):
    z = np.load(CACHE.format(args.team_from))
    info = {k: z["info_" + k] for k in INFO}
    parts = [(p.rpartition(":")[0], float(p.rpartition(":")[2])) for p in args.parts]
    total = sum(w for _, w in parts)
    p1 = 1.0 / (1.0 + np.exp(-sum(w / total * lg(z[m]) for m, w in parts)))
    meta, f = pipeline.load_dataset(TARGET)
    first_of = np.zeros(len(info["start"]), dtype=np.int64)
    for half, first, _ in meta["halves"]:
        first_of[info["half"] == half] = first
    block = int(np.median(info["end"] - info["start"]))
    mid = np.minimum(info["start"] + block // 2, info["end"] - 1)
    p_in = dball.lookup(pipeline.load_answers(TARGET, "exp_k_all_pa_" + args.palive_from), info["start"],
                        info["end"], first_of, 12, block // 2)
    k1, _, _, _ = kick.kick_mask(kick.load_raw(TARGET), meta, f, players_only=True)
    k2 = np.zeros(meta["n_frames"], dtype=bool)
    kicks = dball.kickoff_frames()
    for half, first, last in meta["halves"]:
        k2[first:min(kicks[half], last + 1)] = True
    forced = k1 | k2
    codes = bar_v2.decode(p_in, p1, dict(info, p1=p1), forced[mid], forced, tuple(args.cost), tuple(args.mins))
    td.write(TARGET, args.name, codes)
    print(aggregates(args.name, codes, meta, f), flush=True)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("members")
    a.add_argument("--train", required=True, choices=("idsse", "sc", "both"))
    a.add_argument("--members", nargs="+", default=["lgb_b", "lgb_ab", "kt_b", "kt_ab"])
    b = sub.add_parser("decode")
    b.add_argument("--team-from", required=True)
    b.add_argument("--palive-from", required=True)
    b.add_argument("--parts", nargs="+", required=True)
    b.add_argument("--cost", nargs=3, type=float, required=True)
    b.add_argument("--mins", nargs=2, type=int, required=True)
    b.add_argument("--name", required=True)
    args = parser.parse_args()
    members(args) if args.cmd == "members" else decode(args)


if __name__ == "__main__":
    main()
