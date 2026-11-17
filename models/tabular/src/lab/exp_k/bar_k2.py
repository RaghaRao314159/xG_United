"""BAR-ATH files for every Kumo recipe in one pass: P(alive) (LightGBM pf + Kumo 3-class, logit 50/50)
and each team member's P(H | alive) are computed once, then every recipe is blended and decoded at the
settings its controlled-IDSSE LOMO chose (ev.py logs; never the public window), with bar_v3's kick-off
rules. Members, all trained / in-context on the 7 controlled-IDSSE matches only:
    lgb_b, lgb_ab   ta5.fit_team, 800 trees / 63 leaves
    kt_b, kt_ab     Kumo Tabular large, inference only (50k context, top-100 columns, 4 estimators)
    kft_b           Kumo Tabular large fine-tuned on IDSSE (kt_ft.py --mode xfer -> ft/xktftA_xfer.pt),
                    with the columns stored in the checkpoint
    kr_b            Kumo Relational (kr.py: blocks + 10 nearest players, 1 hop, 50k context, 4 estimators)
Writes runs/BAR-ATH/results/<name>.csv; prints aggregates only.

    MTAH_ROOT=/root/mtah python bar_k2.py --train idsse|sc|both
Every source uses each recipe's IDSSE-chosen decoder, so only the training data differs between sources.
"""
import os as _os
LAB = _os.environ.get("MTAH_LAB") or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

import gc
import os
import sys
import time

sys.path.insert(0, LAB + "/exp_k")
import bar_k  # noqa: E402  (xfer/palive windows, BAR-ATH dataset names, kick-off rules)

import numpy as np  # noqa: E402

from bar_k import TARGET, COND, bar_v2, dball, kick, kt, pipeline, ta5, tb, td, xfer, aggregates, lg  # noqa: E402
import kr  # noqa: E402

FT_PATH = os.path.join(os.path.dirname(kt.TABLES), "ft", "xktftA_xfer.pt")
SC_COND = "clean"
TRAIN = {}            # match -> condition of the training set (set in main from --train)


def train_set(source):
    out = {}
    if source in ("idsse", "both"):
        out.update({m: COND for m in xfer.IDSSE})
    if source in ("sc", "both"):
        out.update({m: SC_COND for m in kt.sc_matches()})
    return out


def team_tables(variant):
    tables = {m: kt.table(m, c, variant) for m, c in TRAIN.items()}
    tables[TARGET] = kt.table(TARGET, COND, variant)
    return tables


def palive_mix(name):
    """bar_k.palive_answers with the training matches of TRAIN."""
    import argparse
    cache, tables, spec, names = {}, {}, {}, None
    for m, c in list(TRAIN.items()) + [(TARGET, COND)]:
        tables[m], n2, s = xfer.palive_table(m, c, cache)
        assert names is None or n2 == names
        names = n2
        spec.update(s)
    tb.swap = bar_k.palive.subset_swap(bar_k.run_tb.make_swap(spec), names)(names)
    train = list(TRAIN)
    x = np.vstack([tables[m][0] for m in train])
    y = np.concatenate([tables[m][1] for m in train])
    lgb, _ = tb.symmetric_proba("lgbm", x, y, tables[TARGET][0], names, tb.SEED)
    kumo = bar_k.kt_pa.Kumo3("large")
    got, _ = bar_k.kt_pa.symmetric(kumo, x, y, {TARGET: tables[TARGET][0]}, names,
                                   argparse.Namespace(k=100, ctx=50000, est=4, seed=0))
    kp = got[TARGET]
    del kumo
    free()
    lgb_in, kp_in = 1 - lgb[:, tb.D], 1 - kp[:, tb.D]
    lgb_h = lgb[:, tb.H] / np.maximum(lgb_in, 1e-9)
    kp_h = kp[:, tb.H] / np.maximum(kp_in, 1e-9)
    p_in = 1 / (1 + np.exp(-(0.5 * lg(lgb_in) + 0.5 * lg(kp_in))))
    p_h = 1 / (1 + np.exp(-(0.5 * lg(lgb_h) + 0.5 * lg(kp_h))))
    proba = np.column_stack([p_in * p_h, p_in * (1 - p_h), 1 - p_in])
    tb.write_answers(TARGET, name, tables[TARGET][2], proba)
    print("P(alive) mix50: mean P(D) {:.3f} (LightGBM {:.3f}, Kumo {:.3f})".format(
        proba[:, tb.D].mean(), lgb[:, tb.D].mean(), kp[:, tb.D].mean()), flush=True)


def team_member(member):
    kind, variant = member.split("_", 1)
    tables = team_tables(variant)
    swap = kt.u_fit.make_swap(*tables[TARGET][2])
    t2 = {m: (t[0], t[1]) for m, t in tables.items()}
    if kind == "lgb":
        ta5.LGBM.update(n_estimators=800, num_leaves=63, n_jobs=int(os.environ.get("MTAH_LGBM_JOBS", 12)))
        p = ta5.fit_team({m: (t[0], None, t[1]) for m, t in tables.items()}, list(TRAIN), [TARGET], swap)[TARGET]
    else:
        x, y, groups = kt.training_pool(t2, list(TRAIN), swap)
        cols = kt.select_columns(x, y, 100, 0)
        pick = kt.sample_context(y, groups, 50000, "random", 0)
        kumo = kt.Kumo("large")
        kumo.fit(x[pick][:, cols], y[pick], 4, 0)
        xt = t2[TARGET][0]
        p = 0.5 * (kumo.predict(xt[:, cols]) + 1.0 - kumo.predict(swap(xt)[:, cols]))
        del kumo
        free()
    print("  team {}: mean P(H | alive) {:.3f}".format(member, p.mean()), flush=True)
    return p, tables[TARGET][1]

# name: (members with weights, (c_HA, c_out, c_in), (min alive, min dead) blocks) -- IDSSE LOMO choices
RECIPES = {
    "exp_k_kk4_mix":  ({"kt_b": 1, "kt_ab": 1, "lgb_b": 1, "lgb_ab": 1}, (2, 2, 20), (10, 1)),
    "exp_k_mk3_mix":  ({"lgb_b": 1, "lgb_ab": 1, "kt_b": 1}, (2, 0.1, 20), (10, 10)),
    "exp_k_mkb_mix":  ({"lgb_b": 1, "kt_b": 1}, (2, 2, 20), (10, 1)),
    "exp_k_kt_mix":   ({"kt_b": 1}, (2, 2, 20), (10, 1)),
    "exp_k_kk_mix":   ({"kt_b": 1, "kt_ab": 1}, (2, 2, 20), (10, 1)),
    "exp_k_ktft_mix": ({"kft_b": 1}, (1, 2, 20), (10, 1)),
    "exp_k_kk4ft_mix": ({"kft_b": 1, "kt_ab": 1, "lgb_b": 1, "lgb_ab": 1}, (2, 2, 20), (10, 1)),
    "exp_k_kr_mix":   ({"kr_b": 1}, (2, 0.1, 20), (10, 1)),
    "exp_k_kk5_mix":  ({"kt_b": 1, "kt_ab": 1, "lgb_b": 1, "lgb_ab": 1, "kr_b": 1}, (2, 2, 20), (10, 1)),
}


def free():
    import torch
    gc.collect()
    torch.cuda.empty_cache()


def kumo_finetuned():
    """Fine-tuned Kumo Tabular on BAR-ATH: kt.py's xfer inference with the saved weights and columns."""
    import torch
    ck = torch.load(FT_PATH, map_location="cpu", weights_only=False)
    tables = team_tables("b")
    swap = kt.u_fit.make_swap(*tables[TARGET][2])
    t2 = {m: (t[0], t[1]) for m, t in tables.items()}
    x, y, groups = kt.training_pool(t2, list(TRAIN), swap)
    cols = np.asarray(ck["columns"])
    kumo = kt.Kumo(ck["size"])
    own = kumo.model.state_dict()
    kumo.model.load_state_dict({k: v.to(own[k].dtype) for k, v in ck["state_dict"].items()})
    kumo.model.eval()
    pick = kt.sample_context(y, groups, 50000, "random", 0)
    kumo.fit(x[pick][:, cols], y[pick], 4, 0)
    xt = t2[TARGET][0]
    p = 0.5 * (kumo.predict(xt[:, cols]) + 1.0 - kumo.predict(swap(xt)[:, cols]))
    print("  team kft_b (fine-tuned, kept step {}): mean P(H | alive) {:.3f}".format(ck.get("kept_step"), p.mean()),
          flush=True)
    del kumo
    free()
    return p


def relational():
    """Kumo Relational on BAR-ATH: kr.py's xfer inference with BAR-ATH as the only target."""
    tables = team_tables("b")
    swap = kt.u_fit.make_swap(*tables[TARGET][2])
    players = {m: kr.player_rows(m, TRAIN.get(m, COND), tables[m][1], 10) for m in tables}
    xs, ys, brs, prs, off = [], [], [], [], 0
    for m in TRAIN:
        x, info = tables[m][0], tables[m][1]
        keep, y = kt.ta5.team_target(info["counts"])
        idx = np.flatnonzero(keep)
        br, pr = kr.subset_rows(players[m][0], players[m][1], idx)
        for flip in (False, True):
            xs.append(swap(x[idx]) if flip else x[idx])
            ys.append(1 - y[idx] if flip else y[idx])
            p2 = pr.copy()
            if flip:
                p2[:, 0] *= -1
            brs.append(br + off)
            prs.append(p2)
            off += len(idx)
    x, y = np.vstack(xs), np.concatenate(ys).astype(np.int64)
    br, pr = np.concatenate(brs), np.vstack(prs)
    cols = kt.select_columns(x, y, 50, 0)
    names = ["c{}".format(i) for i in range(len(cols))]
    pick = np.sort(np.random.default_rng(0).choice(len(y), size=min(50000, len(y)), replace=False))
    cb, cp = kr.subset_rows(br, pr, pick)
    model = kr.Relational()
    model.fit(np.arange(len(pick)), x[pick][:, cols], y[pick], cb, cp, names, 4)
    xt = tables[TARGET][0]
    nb = len(xt)
    out = {}
    for flip in (False, True):
        xx = swap(xt) if flip else xt
        pb, pp = players[TARGET]
        pp = pp.copy()
        if flip:
            pp[:, 0] *= -1
        res = np.empty(nb)
        for i in range(0, nb, 4096):
            sel = np.arange(i, min(nb, i + 4096))
            qb, qp = kr.subset_rows(pb, pp, sel)
            res[sel] = model.predict(10_000_000 + sel, xx[sel][:, cols], qb, qp, names)
        out[flip] = res
    p = 0.5 * (out[False] + 1.0 - out[True])
    print("  team kr_b (Kumo Relational): mean P(H | alive) {:.3f}".format(p.mean()), flush=True)
    del model
    free()
    return p


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="idsse", choices=("idsse", "sc", "both"))
    args = parser.parse_args()
    TRAIN.update(train_set(args.train))
    started = time.time()
    print("training set {}: {} matches".format(args.train, len(TRAIN)), flush=True)
    pa_name = "exp_k_all_pa_" + args.train
    palive_mix(pa_name)
    tb.swap = xfer.ORIGINAL_SWAP
    pipeline.WINDOW = tb.WINDOW = pipeline.FPS
    members, info = {}, None
    for m in ("lgb_b", "lgb_ab", "kt_b", "kt_ab"):
        members[m], info = team_member(m)
        free()
    members["kft_b"] = kumo_finetuned()
    members["kr_b"] = relational()
    print("members done, {:.0f} s".format(time.time() - started), flush=True)

    meta, f = pipeline.load_dataset(TARGET)
    first_of = np.zeros(len(info["start"]), dtype=np.int64)
    for half, first, _ in meta["halves"]:
        first_of[info["half"] == half] = first
    block = int(np.median(info["end"] - info["start"]))
    mid = np.minimum(info["start"] + block // 2, info["end"] - 1)
    p_in = dball.lookup(pipeline.load_answers(TARGET, pa_name), info["start"], info["end"],
                        first_of, 12, block // 2)
    k1, _, _, _ = kick.kick_mask(kick.load_raw(TARGET), meta, f, players_only=True)
    k2 = np.zeros(meta["n_frames"], dtype=bool)
    kicks = dball.kickoff_frames()
    for half, first, last in meta["halves"]:
        k2[first:min(kicks[half], last + 1)] = True
    forced = k1 | k2
    print("| output | D share | D in >10 s blackouts | D pre-kick-off 2nd half | possessions | per "
          "alive-min | mean s | median s | < 2 s | < 4 s |\n|---|---|---|---|---|---|---|---|---|---|", flush=True)
    for name, (parts, cost, mins) in RECIPES.items():
        total = sum(parts.values())
        z = sum(w / total * lg(members[m]) for m, w in parts.items())
        p1 = 1.0 / (1.0 + np.exp(-z))
        codes = bar_v2.decode(p_in, p1, dict(info, p1=p1), forced[mid], forced, tuple(float(c) for c in cost),
                              tuple(mins))
        out = "{}_{}".format(name, args.train)
        td.write(TARGET, out, codes)
        print(aggregates(out, codes, meta, f), flush=True)
    print("done in {:.0f} s".format(time.time() - started), flush=True)


if __name__ == "__main__":
    main()
