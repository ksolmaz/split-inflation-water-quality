import os
import re
import zipfile

import pandas as pd
import xarray as xr

from swq import data as D

MIN_DAYS = 10
PAIRS = [("NO3N", "TN"), ("TRP", "TP"), ("TDP", "TP"), ("DOC", "TOC")]
CARAVAN = os.path.join(D.DATA, "caravan")
ZARR = os.path.join(CARAVAN, "Caravan-Qual_lite.zarr")
DROP = re.compile(r"_s\d\d$|^(TDXHydroRegion|TerminalLink|TopologicalOrder|DSContArea|musk_[kx]|LengthGeodesicMeters)$")

zf = zipfile.ZipFile(os.path.join(CARAVAN, "wqms-csv.zip"))
cache = {}


def load(v):
    if v not in cache:
        with zf.open(f"wqms-csv/{v}.csv") as f:
            d = pd.read_csv(f, usecols=["wqms_id", "dates", "obs", "flag"], dtype={"flag": "string", "wqms_id": "string"}, parse_dates=["dates"])
        d = d[d.flag.isna() & d.obs.notna() & (d.obs > 0)]
        d["dates"] = d.dates.dt.normalize()
        cache[v] = d.groupby(["wqms_id", "dates"], as_index=False).obs.mean().rename(columns={"obs": v})
    return cache[v]


tables = {}
for lo, up in PAIRS:
    m = load(lo).merge(load(up), on=["wqms_id", "dates"])
    m["ratio"] = (m[lo] / m[up]).clip(upper=1.0)
    g = m.groupby("wqms_id").agg(n_days=(lo, "size"), lo=(lo, "median"), up=(up, "median"), ratio=("ratio", "median"))
    tables[(lo, up)] = g[g.n_days >= MIN_DAYS].reset_index()
    print(f"{lo}<={up}: stations with >={MIN_DAYS} common days = {len(tables[(lo, up)])}", flush=True)

uni = pd.concat([t[["wqms_id"]] for t in tables.values()]).drop_duplicates()
sites = pd.read_csv(os.path.join(CARAVAN, "wqms_site_info.csv")).drop_duplicates("wqms_id")
sites["wqms_id"] = sites.wqms_id.astype("string")
uni = uni.merge(sites[["wqms_id", "LINKNO", "merged_LINKNO", "gauge_id", "country_name"]], on="wqms_id", how="left")
uni = uni[uni.LINKNO.notna()].reset_index(drop=True)

if not os.path.isdir(ZARR):
    print("extracting Caravan-Qual_lite.zarr.zip", flush=True)
    with zipfile.ZipFile(os.path.join(CARAVAN, "Caravan-Qual_lite.zarr.zip")) as z:
        nested = z.namelist()[0].startswith("Caravan-Qual_lite.zarr/")
        z.extractall(CARAVAN if nested else ZARR)

ds = xr.open_zarr(ZARR, consolidated=True)
attr_vars = [v for v in ds.data_vars if ds[v].dims == ("LINKNO",) and not DROP.search(v)]
idx = pd.Index(ds.LINKNO.values).get_indexer(uni.LINKNO.astype("int64"))
uni = uni[idx >= 0].reset_index(drop=True)
A = ds[attr_vars].isel(LINKNO=idx[idx >= 0]).load().to_dataframe().reset_index(drop=True)
A = A.loc[:, A.nunique() > 1]
A = A.fillna(A.median())
statics = pd.concat([uni[["wqms_id", "country_name"]], A], axis=1)

for (lo, up), t in tables.items():
    out = t.merge(statics, on="wqms_id")
    out.to_csv(os.path.join(D.DATA, f"pair_{lo}_{up}.csv"), index=False)
    print(f"pair_{lo}_{up}.csv: stations={len(out)} features={A.shape[1]}")
