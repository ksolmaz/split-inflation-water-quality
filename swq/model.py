import warnings

import numpy as np
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from swq.util import r2

warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)

FAMILIES = ["ridge", "knn", "rf", "mlp"]
GRID = {
    "lgbm": {"lr": [0.03, 0.06, 0.1], "leaves": [15, 31, 63], "mcs": [5, 20, 50], "ss": [0.7, 1.0], "cs": [0.5, 0.8, 1.0], "l2": [0.0, 1.0, 10.0]},
    "ridge": {},
    "knn": {"k": [3, 5, 10, 20], "w": ["uniform", "distance"]},
    "rf": {"leaf": [2, 5, 10], "mf": [0.33, 0.6]},
    "mlp": {"hid": [(64, 64), (128, 64)], "alpha": [1e-4, 1e-2]},
}
DEFAULT = {
    "lgbm": {"lr": 0.06, "leaves": 31, "mcs": 20, "ss": 1.0, "cs": 0.8, "l2": 1.0},
    "ridge": {},
    "knn": {"k": 10, "w": "distance"},
    "rf": {"leaf": 2, "mf": 0.6},
    "mlp": {"hid": (64, 64), "alpha": 1e-3},
}
N_CANDIDATES = {"lgbm": 6, "ridge": 1, "knn": 4, "rf": 3, "mlp": 2}


def clip(a):
    return np.clip(a, -5, 5)


def scaled(m):
    return Pipeline([("sc", StandardScaler()), ("cl", FunctionTransformer(clip)), ("m", m)])


def build(name, p, seed):
    if name == "lgbm":
        return LGBMRegressor(n_estimators=250, learning_rate=p["lr"], num_leaves=p["leaves"], min_child_samples=p["mcs"],
                             subsample=p["ss"], subsample_freq=1, colsample_bytree=p["cs"], reg_lambda=p["l2"],
                             n_jobs=1, verbose=-1, random_state=seed)
    if name == "ridge":
        return scaled(RidgeCV(alphas=np.logspace(-3, 3, 13)))
    if name == "knn":
        return scaled(KNeighborsRegressor(n_neighbors=p["k"], weights=p["w"]))
    if name == "rf":
        return RandomForestRegressor(n_estimators=150, min_samples_leaf=p["leaf"], max_features=p["mf"], n_jobs=1, random_state=seed)
    if name == "mlp":
        return scaled(MLPRegressor(hidden_layer_sizes=p["hid"], alpha=p["alpha"], max_iter=300, early_stopping=True,
                                   n_iter_no_change=15, random_state=seed))
    raise ValueError(name)


def candidates(name, seed):
    if not GRID[name]:
        return [{}]
    rng = np.random.RandomState(seed)
    out = [DEFAULT[name]]
    while len(out) < N_CANDIDATES[name]:
        out.append({k: v[rng.randint(len(v))] for k, v in GRID[name].items()})
    return out


def tune(name, X, y, blocks, seed):
    best, best_score = None, -np.inf
    for p in candidates(name, seed):
        score = np.mean([r2(y[b], build(name, p, seed).fit(X[a], y[a]).predict(X[b])) for a, b in GroupKFold(3).split(X, y, blocks)])
        if score > best_score:
            best, best_score = p, float(score)
    return best, best_score


def fit_predict(task, name, p, train, test, seed):
    return build(name, p, seed).fit(task.X[train], task.y[train]).predict(task.X[test])


def unit_vector(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.c_[np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]


def location_only(task, train, test):
    knn = KNeighborsRegressor(n_neighbors=5, weights="distance")
    knn.fit(unit_vector(task.lat[train], task.lon[train]), task.y[train])
    return knn.predict(unit_vector(task.lat[test], task.lon[test]))
