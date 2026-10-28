"""Time-based (walk-forward) evaluation, metrics, regime and error analysis."""
import numpy as np
import pandas as pd
from sklearn.base import clone

from .config import Config
from .models import BASELINES


def walk_forward(labeled: pd.DataFrame, cfg: Config, models: dict, features: list) -> pd.DataFrame:
    """Expanding-window walk-forward. For each test year Y the models are fit only on origins whose
    target was already realised before Y began (t + horizon <= first quarter of Y), so no future
    information reaches the training set. Random shuffling is never used."""
    h = cfg.horizon_q
    last = cfg.last_test_year or int(labeled["yr"].max())
    parts = []
    for Y in range(cfg.first_test_year, last + 1):
        test = labeled[labeled["yr"] == Y]
        train = labeled[labeled["t"] + h <= Y * 4]
        if test.empty or len(train) < cfg.min_train_rows:
            continue
        part = test[["place_id", "place_name", "t", "yr", "q"]].copy()
        part["actual"] = test["target"].values
        part["Persistence"] = test["yoy"].values
        part["Historical mean"] = test["hist_mean"].values
        for name, model in models.items():
            fitted = clone(model).fit(train[features], train["target"])
            part[name] = fitted.predict(test[features])
        part["n_train"] = len(train)
        parts.append(part)
        print(f"[eval] test year {Y}: train rows {len(train):,}, test rows {len(test):,}")
    if not parts:
        raise ValueError("No test years produced; adjust first_test_year / min_train_rows.")
    res = pd.concat(parts, ignore_index=True)
    return add_regimes(res, cfg)


def add_regimes(res: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Label each origin quarter by the *realised* market-wide next-year growth (median across places).
    This is an evaluation label only, never a model input."""
    lo, hi = cfg.regime_bounds
    nat = res.groupby("t")["actual"].transform("median")
    res = res.copy()
    res["national_actual"] = nat
    res["regime"] = pd.cut(nat, [-np.inf, lo, hi, np.inf], labels=["decline", "steady", "boom"])
    return res


def score(res: pd.DataFrame, pred_cols: list, by=None) -> pd.DataFrame:
    rows = []
    groups = [("all", res)] if by is None else list(res.groupby(by, observed=True))
    for key, g in groups:
        for c in pred_cols:
            e = g[c] - g["actual"]
            rows.append({"group": key, "model": c, "n": len(g), "MAE": e.abs().mean(),
                         "RMSE": float(np.sqrt((e ** 2).mean())), "Bias": e.mean()})
    out = pd.DataFrame(rows)
    base = out[out["model"] == BASELINES[0]].set_index("group")["MAE"]
    out["MAE_vs_persistence_%"] = (1 - out["MAE"] / out["group"].map(base)) * 100
    return out


def worst_places(res: pd.DataFrame, pred_cols: list, top: int = 15) -> pd.DataFrame:
    tbl = pd.DataFrame({c: (res[c] - res["actual"]).abs().groupby(res["place_name"]).mean() for c in pred_cols})
    return tbl.sort_values(pred_cols[-1], ascending=False).head(top)


def largest_errors(res: pd.DataFrame, model: str, top: int = 25) -> pd.DataFrame:
    out = res[["place_name", "yr", "q", "regime", "actual", model]].copy()
    out["error"] = out[model] - out["actual"]
    out["abs_error"] = out["error"].abs()
    return out.nlargest(top, "abs_error").reset_index(drop=True)
