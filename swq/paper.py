import argparse
import glob
import json
import os
import time

import numpy as np
import pandas as pd

from swq import data as D
from swq import model
from swq import util as U

STATION = {"gloria": "water-body name", "caravan": "gauge_id", "ssc": "site_no", "coast": "location name"}
KM_EDGES = [0, 0.1, 1, 10, 100, 1000, 1e9]
KM_BINS = ["<0.1", "0.1-1", "1-10", "10-100", "100-1000", ">1000"]
FIG4_TASKS = ["gloria/Chla", "caravan/DOC", "ssc/SSC", "coast/DOC"]


def frame(task, test, pred, n_train, **cols):
    return pd.DataFrame({**cols, "i": test, "y": task.y[test], "p": np.asarray(pred, np.float32), "n_train": n_train})


def run_unit(u):
    if os.path.exists(u["path"]):
        return "done before"
    t0 = time.time()
    task, s, k, scheme = D.get_task(u["task"]), u["seed"], u["fold"], u["scheme"]
    f = D.folds(task, scheme, s)
    test, train = np.where(f == k)[0], np.where(f != k)[0]
    hp, inner_r2 = model.tune("lgbm", task.X[train], task.y[train], task.grp[train], 100 * s + k)
    info = {"hp": hp, "inner_r2": inner_r2}
    out = [frame(task, test, model.fit_predict(task, "lgbm", hp, train, test, s), len(train), scheme=scheme, cond="base", model="lgbm")]
    if scheme != "station":
        out[0]["d_km"] = U.nn_km(task.lat[train], task.lon[train], task.lat[test], task.lon[test]).astype(np.float32)
        out.append(frame(task, test, model.location_only(task, train, test), len(train), scheme=scheme, cond="base", model="location"))
        info["families"] = {}
        for m in model.FAMILIES:
            p, sc = model.tune(m, task.X[train], task.y[train], task.grp[train], 100 * s + k)
            info["families"][m] = {"hp": p, "inner_r2": sc}
            out.append(frame(task, test, model.fit_predict(task, m, p, train, test, s), len(train), scheme=scheme, cond="base", model=m))
    if scheme == "random":
        dup = np.isin(task.key[train], task.key[test])
        rest = train[~dup]
        n_drop = min(int(dup.sum()), max(len(rest) - 50, 0))
        drop = np.random.RandomState(1000 * s + k + 7).choice(len(rest), size=n_drop, replace=False)
        control = np.concatenate([np.delete(rest, drop), train[dup]])
        info["removal"] = dict(n_train=len(train), n_removed=int(dup.sum()), n_control_removed=n_drop, size_matched=n_drop == int(dup.sum()))
        for cond, tr in (("rm", rest), ("ctrl", control)):
            out.append(frame(task, test, model.fit_predict(task, "lgbm", hp, tr, test, s), len(tr), scheme=scheme, cond=cond, model="lgbm"))
    df = pd.concat(out, ignore_index=True)
    df["seed"], df["fold"] = s, k
    U.write_parquet(df, u["path"])
    info["seconds"] = round(time.time() - t0, 1)
    with open(u["path"].replace(".parquet", "_info.json"), "w") as fh:
        json.dump(info, fh, indent=1)
    return "ok"


def analyze_task(t, res, n_boot):
    X = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(res, t, "*", "s*_k*.parquet")))], ignore_index=True)
    task = D.get_task(t)
    dom, n, idx = task.domain, task.n, np.arange(task.n)
    ds = D.DATASET[dom]
    boot = U.Boot(task.grp, n_boot)
    seeds = sorted(X.seed.unique())

    def oof(s, scheme, mdl="lgbm", cond="base"):
        g = X[(X.seed == s) & (X.scheme == scheme) & (X.model == mdl) & (X.cond == cond)].sort_values("i")
        assert len(g) == n and g.i.is_unique, f"{t}: missing predictions ({scheme}, {mdl}, {cond}, seed {s})"
        return g

    def r2(scheme, mdl="lgbm", cond="base"):
        return np.array([U.r2(g.y.values, g.p.values) for g in (oof(s, scheme, mdl, cond) for s in seeds)])

    def boot_r2(scheme, mdl="lgbm"):
        return np.mean([boot.r2(idx, g.y.values.astype(float), g.p.values.astype(float)) for g in (oof(s, scheme, mdl) for s in seeds)], 0)

    rr, rb, rs = r2("random"), r2("blocked"), r2("station")
    br, bb, bs = boot_r2("random"), boot_r2("blocked"), boot_r2("station")
    out = {}
    lo, hi = U.ci(br - bb)
    out["inflation"] = dict(dataset=ds, task=t, r2_random=rr.mean(), r2_group=rb.mean(), inflation=(rr - rb).mean(),
                            inflation_lo=lo, inflation_hi=hi, n_seeds=len(seeds))
    lo, hi = U.ci(bs - bb)
    out["station_grouping"] = dict(dataset=ds, task=t, station_definition=STATION[dom], r2_random=rr.mean(), r2_blocked=rb.mean(),
                                   r2_station=rs.mean(), extra_r2_station=(rs - rb).mean(), extra_r2_station_lo=lo, extra_r2_station_hi=hi,
                                   share_hidden_by_station_grouping=1 - (rr - rs).mean() / (rr - rb).mean(), n_seeds=len(seeds))
    out["location_only"] = dict(dataset=ds, task=t, location_r2_random=r2("random", "location").mean(),
                                location_r2_group=r2("blocked", "location").mean(), lgbm_r2_random=rr.mean(), lgbm_r2_group=rb.mean())

    removal = [json.load(open(f))["removal"] for f in glob.glob(os.path.join(res, t, "random", "s*_k*_info.json"))]
    rm, ct = r2("random", cond="rm"), r2("random", cond="ctrl")
    share = (ct - rm) / (rr - rb)
    out["twin_removal"] = dict(dataset=ds, task=t, definition="narrow", r2_random=rr.mean(), r2_removed=rm.mean(), r2_control=ct.mean(),
                               r2_group=rb.mean(), inflation=(rr - rb).mean(), net_effect=(ct - rm).mean(), share_of_inflation=share.mean(),
                               share_of_inflation_sd=share.std(ddof=1),
                               share_train_removed=float(np.mean([r["n_removed"] / r["n_train"] for r in removal])),
                               size_matched=all(r["size_matched"] for r in removal), n_seeds=len(seeds))

    out["intervals"] = []
    for cal in ("random", "blocked"):
        per_seed = []
        for s in seeds:
            ev, cb = oof(s, "blocked"), oof(s, cal)
            fold = D.folds(task, "blocked", s)
            res_ev, res_cal = np.abs(ev.y.values - ev.p.values), np.abs(cb.y.values - cb.p.values)
            q = np.zeros(n)
            for k in np.unique(fold):
                q[fold == k] = U.conformal_q(res_cal[fold != k], 0.9)
            hit = (res_ev <= q).astype(float)
            per_seed.append((hit.mean(), *U.ci(boot.mean(idx, hit))))
        c, lo, hi = np.mean(per_seed, 0)
        out["intervals"].append(dict(task=t, model="lgbm", cal="group" if cal == "blocked" else cal, eval="group", level=0.9,
                                     coverage=c, lo=lo, hi=hi, n_seeds=len(seeds)))

    out["distance"] = []
    for scheme in ("random", "blocked"):
        d = X[(X.scheme == scheme) & (X.model == "lgbm") & (X.cond == "base")].d_km
        counts = pd.cut(d, KM_EDGES, labels=KM_BINS, include_lowest=True).value_counts()
        out["distance"] += [dict(dataset=ds, task=t, split="group" if scheme == "blocked" else scheme, km_bin=b, n=int(v))
                            for b, v in counts.items() if v]
    out["models"] = []
    for m in ["lgbm"] + model.FAMILIES:
        a, b = r2("random", m), r2("blocked", m)
        lo, hi = U.ci(boot_r2("random", m) - boot_r2("blocked", m))
        out["models"].append(dict(dataset=ds, task=t, model=m, r2_random=a.mean(), r2_group=b.mean(), inflation=(a - b).mean(),
                                  inflation_lo=lo, inflation_hi=hi))

    out["near_duplicates"] = []
    for scheme in D.SCHEMES:
        for s in seeds:
            f = D.folds(task, scheme, s)
            share = np.mean([np.isin(task.key[f == k], task.key[f != k]).mean() for k in range(D.NFOLD)])
            out["near_duplicates"].append(dict(dataset=ds, task=t, scheme=scheme, seed=s, share=share))

    if t in FIG4_TASKS:
        out["fig4"] = [pd.DataFrame({"task": t, "scheme": scheme, "y": g.y.values, "p": g.p.values}) for scheme in ("random", "blocked")
                       for g in [oof(seeds[0], scheme)]]
    if dom == "caravan":
        h = pd.util.hash_pandas_object(pd.DataFrame(task.X).round(10), index=False)
        out["duplicates"] = dict(task=t, n_stations=n, share_identical_predictors=h.duplicated(keep=False).mean())
    return out


def analyze(tasks, res, tables, n_boot):
    rows = {}
    for t in tasks:
        for name, r in analyze_task(t, res, n_boot).items():
            rows.setdefault(name, []).extend(r if isinstance(r, list) else [r])
        print("analyzed", t, flush=True)
    for name in ("inflation", "station_grouping", "location_only", "twin_removal", "intervals"):
        U.write_csv(pd.DataFrame(rows[name]), os.path.join(tables, name + ".csv"))
    U.write_csv(pd.DataFrame(rows["models"]), os.path.join(tables, "models_supplement.csv"))
    U.write_csv(pd.DataFrame(rows["near_duplicates"]), os.path.join(tables, "near_duplicate_share.csv"))
    if "fig4" in rows:
        U.write_csv(pd.concat(rows["fig4"], ignore_index=True), os.path.join(tables, "fig4_predictions.csv.gz"))
    U.write_csv(pd.DataFrame(rows["duplicates"]), os.path.join(tables, "caravan_duplicate_stations.csv"))
    d = pd.DataFrame(rows["distance"]).groupby(["dataset", "split", "km_bin"], sort=False).n.sum().reset_index()
    d["share_of_test_rows"] = d.n / d.groupby(["dataset", "split"]).n.transform("sum")
    U.write_csv(d, os.path.join(tables, "distance_error_by_dataset.csv"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "analyze"])
    ap.add_argument("--out", default=os.path.join(D.ROOT, "output"))
    ap.add_argument("--tasks", default=",".join(D.ALL_TASKS))
    ap.add_argument("--seeds", help="default: 0-4 (GLORIA, Caravan-Qual), 0-2 (USGS SSC, CoastDOM)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    tasks, res = a.tasks.split(","), os.path.join(a.out, "results")
    if a.cmd == "analyze":
        return analyze(tasks, res, os.path.join(a.out, "tables"), a.boot)
    units = [dict(task=t, scheme=sc, seed=s, fold=k, path=os.path.join(res, t, sc, f"s{s}_k{k}.parquet"))
             for t in tasks for sc in D.SCHEMES
             for s in ([int(x) for x in a.seeds.split(",")] if a.seeds else D.SEEDS[t.split("/")[0]])
             for k in range(D.NFOLD)]
    raise SystemExit(1 if U.run_units(run_unit, units, a.workers) else 0)


if __name__ == "__main__":
    main()
