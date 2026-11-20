import io
import os
import warnings

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

ROOT = os.environ.get("SWQ_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.environ.get("SWQ_DATA", os.path.join(ROOT, "data"))
NFOLD = 5

CARAVAN = {"NO3N": ("NO3N_TN", "lo"), "TN": ("NO3N_TN", "up"), "TRP": ("TRP_TP", "lo"), "TP": ("TRP_TP", "up"),
           "TDP": ("TDP_TP", "lo"), "DOC": ("DOC_TOC", "lo"), "TOC": ("DOC_TOC", "up")}
TASKS = {"gloria": ["Chla", "TSS", "aCDOM440", "Secchi_depth"], "caravan": list(CARAVAN), "ssc": ["SSC"], "coast": ["DOC"]}
ALL_TASKS = [f"{d}/{t}" for d, ts in TASKS.items() for t in ts]
DATASET = {"gloria": "GLORIA", "caravan": "Caravan-Qual", "ssc": "USGS SSC", "coast": "CoastDOM"}
SEEDS = {"gloria": range(5), "caravan": range(5), "ssc": range(3), "coast": range(3)}
SCHEMES = ["random", "blocked", "station"]
SSC_FILE = "sentinel2_usgs_ssc_passed_gate_25474_hydrology_ml.csv"
COAST_FILE = "coastdom_pangaea964012.tab"


class Task:
    def __init__(self, name, ids, X, y, lat, lon, block, key, site=None):
        self.name, self.domain = name, name.split("/")[0]
        self.ids, self.X, self.y = np.asarray(ids), X.astype(np.float64), y.astype(np.float64)
        self.lat, self.lon, self.grp, self.key, self.site = lat, lon, block, key, site
        self.n = len(y)
        self._folds = {}


def _components(a, b, na, nb):
    A = coo_matrix((np.ones(len(a)), (a, np.asarray(b) + na)), shape=(na + nb, na + nb))
    return connected_components(A, directed=False)[1]


def _blocks(lat, lon, site):
    cell = np.char.add(np.floor(lat).astype(int).astype(str), np.char.add("_", np.floor(lon).astype(int).astype(str)))
    cu, ci = np.unique(cell, return_inverse=True)
    su, si = np.unique(site, return_inverse=True)
    return _components(ci, si, len(cu), len(su))[ci]


_GLORIA = {}


def _gloria_base():
    d = os.path.join(DATA, "GLORIA_2022")
    m = pd.read_csv(os.path.join(d, "GLORIA_meta_and_lab.csv"))
    rrs = pd.read_csv(os.path.join(d, "GLORIA_Rrs.csv"))
    qc = pd.read_csv(os.path.join(d, "GLORIA_qc_flags.csv"))
    ok = (m.GLORIA_ID.map(qc.drop_duplicates("GLORIA_ID").set_index("GLORIA_ID").Flagged) == 0).values
    wl = np.arange(350, 901)
    R = rrs[[f"Rrs_{w}" for w in wl]].values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        X = np.column_stack([np.nanmean(R[:, np.where(np.abs(wl - c) <= 0.5)[0]], axis=1) for c in range(400, 701, 25)])
    keep = ok & np.isfinite(X).all(1) & m.Latitude.notna().values
    return m[keep].reset_index(drop=True), np.log10(np.clip(X[keep], 1e-4, None))


def load_gloria(target):
    if not _GLORIA:
        _GLORIA["m"], _GLORIA["X"] = _gloria_base()
    has = _GLORIA["m"][target].notna().values
    m, X = _GLORIA["m"][has].reset_index(drop=True), _GLORIA["X"][has]
    lat, lon = m.Latitude.values, m.Longitude.values
    site = m.Site_name.fillna("NA").astype(str).values
    day = m.Date_Time_UTC.astype(str).str[:10].values
    key = np.array([s + "|" + d for s, d in zip(site, day)])
    y = np.log10(m[target].clip(lower=1e-3).values)
    return Task(f"gloria/{target}", m.GLORIA_ID.values, X, y, lat, lon, _blocks(lat, lon, site), key, site)


def load_caravan(target):
    pair, col = CARAVAN[target]
    d = pd.read_csv(os.path.join(DATA, f"pair_{pair}.csv"))
    si = pd.read_csv(os.path.join(DATA, "caravan", "wqms_site_info.csv")).set_index("wqms_id").loc[d.wqms_id]
    X = d[[c for c in d.columns if c not in ("wqms_id", "n_days", "lo", "up", "ratio", "country_name", "fold")]].values
    ids = [[None if x is None else f"{c}:{x}" for x in si[c].astype(str).where(si[c].notna(), None).values]
           for c in ("gauge_id", "LINKNO", "merged_LINKNO")]
    tokens, a, b = {}, [], []
    for i in range(len(d)):
        for kk in ids:
            if kk[i] is not None:
                a.append(i)
                b.append(tokens.setdefault(kk[i], len(tokens)))
    block = _components(np.array(a), np.array(b), len(d), len(tokens))[:len(d)]
    key = np.array([hash(tuple(r)) for r in np.round(X, 6)]).astype(str)
    return Task(f"caravan/{target}", d.wqms_id.values, X, np.log(d[col].values), si.wqms_lat.values, si.wqms_lon.values, block, key)


def load_ssc():
    d = pd.read_csv(os.path.join(DATA, SSC_FILE), low_memory=False)
    feats = [c for c in d.columns if c.endswith("_median")] + ["DOY_sin", "DOY_cos"]
    ok = d.ResultMeasureValue.notna() & (d.ResultMeasureValue > 0) & d.site_lat.notna() & d.site_lon.notna()
    d = d[ok].dropna(subset=feats).reset_index(drop=True)
    site = d.site_no.astype(str).values
    key = np.array([s + "|" + t for s, t in zip(site, d.ActivityStartDate.astype(str).str[:10])])
    return Task("ssc/SSC", np.arange(len(d)).astype(str), d[feats].values, np.log10(d.ResultMeasureValue.values),
                d.site_lat.values, d.site_lon.values, _blocks(d.site_lat.values, d.site_lon.values, site), key, site)


def load_coast():
    raw = open(os.path.join(DATA, COAST_FILE), encoding="utf-8", errors="replace").read()
    d = pd.read_csv(io.StringIO(raw[raw.index("*/") + 2:].lstrip("\n")), sep="\t", low_memory=False)
    d.columns = [c.split(" (")[0] if "(" in c else c for c in d.columns]
    pred = ["Temp [°C]", "Sal", "Depth water [m]", "[NO3]- + [NO2]- [µmol/l]", "[NH4]+ [µmol/l]", "[HPO4]2- [µmol/l]"]
    tgt = "DOC [µmol/l]"
    d = d[(d["QF DOC"] == 2) & (d["QF [NO3]- + [NO2]-"] == 2) & (d["QF [NH4]+"] == 2) & (d["QF [HPO4]2-"] == 2)]
    d = d.dropna(subset=pred + [tgt, "Latitude", "Longitude", "Date/Time", "Location"])
    d = d[d[tgt] > 0].reset_index(drop=True)
    site = d.Location.astype(str).values
    key = np.array([s + "|" + t for s, t in zip(site, d["Date/Time"].astype(str).str[:10])])
    return Task("coast/DOC", np.arange(len(d)).astype(str), d[pred].values, np.log10(d[tgt].values),
                d.Latitude.values, d.Longitude.values, _blocks(d.Latitude.values, d.Longitude.values, site), key, site)


LOADERS = {"gloria": load_gloria, "caravan": load_caravan, "ssc": lambda _: load_ssc(), "coast": lambda _: load_coast()}
_TASKS = {}


def get_task(name):
    if name not in _TASKS:
        dom, target = name.split("/")
        _TASKS[name] = LOADERS[dom](target)
    return _TASKS[name]


def stations(task):
    if task.domain != "caravan":
        return task.site
    si = pd.read_csv(os.path.join(DATA, "caravan", "wqms_site_info.csv")).drop_duplicates("wqms_id").set_index("wqms_id")
    gauge = si.loc[task.ids, "gauge_id"].values
    return pd.Series([f"row:{i}" if pd.isna(g) else f"gauge:{g}" for i, g in enumerate(gauge)]).factorize()[0]


def _balanced(groups, seed):
    g, inv = np.unique(groups, return_inverse=True)
    size = np.bincount(inv)
    rng = np.random.RandomState(seed)
    load, fold = np.zeros(NFOLD), np.zeros(len(g), int)
    for j in rng.permutation(len(g)):
        f = int(np.argmin(load + rng.rand(NFOLD) * 1e-6))
        fold[j] = f
        load[f] += size[j]
    return fold[inv]


def folds(task, scheme, seed):
    if (scheme, seed) not in task._folds:
        if scheme == "random":
            f = np.random.RandomState(seed).permutation(task.n) % NFOLD
        elif scheme == "blocked":
            f = _balanced(task.grp, seed)
        else:
            f = _balanced(stations(task), seed)
        task._folds[scheme, seed] = f
    return task._folds[scheme, seed]
