import os
import time
import traceback

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
from sklearn.neighbors import BallTree

R_EARTH = 6371.0


def write_csv(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print("wrote", path)


def write_parquet(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path + ".tmp")
    os.replace(path + ".tmp", path)


def run_units(fn, units, workers):
    import multiprocessing as mp
    t0, errors = time.time(), 0
    with mp.get_context("spawn").Pool(workers) as pool:
        for i, (u, status) in enumerate(pool.imap_unordered(_call, [(fn, u) for u in units]), 1):
            errors += status == "error"
            print(f"{i}/{len(units)} {u['task']} {u['scheme']} seed {u['seed']} fold {u['fold']}: {status} "
                  f"({(time.time() - t0) / 60:.1f} min)", flush=True)
    return errors


def _call(arg):
    fn, u = arg
    try:
        return u, fn(u)
    except Exception:
        traceback.print_exc()
        return u, "error"


def r2(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def nn_km(lat_ref, lon_ref, lat, lon):
    tree = BallTree(np.radians(np.c_[lat_ref, lon_ref]), metric="haversine")
    return tree.query(np.radians(np.c_[lat, lon]), k=1)[0][:, 0] * R_EARTH


class Boot:
    def __init__(self, grp, B, seed=12345):
        _, self.gi = np.unique(grp, return_inverse=True)
        self.ng = self.gi.max() + 1
        self.W = np.random.RandomState(seed).multinomial(self.ng, np.full(self.ng, 1.0 / self.ng), size=B).astype(np.float64)

    def sums(self, i, v):
        return np.bincount(self.gi[i], weights=np.asarray(v, float), minlength=self.ng)

    def total(self, i, v):
        with np.errstate(all="ignore"):
            return self.W @ self.sums(i, v)

    def mean(self, i, v):
        return self.total(i, v) / np.maximum(self.total(i, np.ones(len(i))), 1e-12)

    def r2(self, i, y, p):
        n, sy, syy = self.total(i, np.ones(len(i))), self.total(i, y), self.total(i, np.asarray(y, float) ** 2)
        return 1 - self.total(i, (np.asarray(y) - np.asarray(p)) ** 2) / (syy - sy ** 2 / n)


def ci(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))


def conformal_q(abs_res, level):
    n = len(abs_res)
    k = int(np.ceil((n + 1) * level))
    return np.inf if k > n else float(np.partition(abs_res, k - 1)[k - 1])
