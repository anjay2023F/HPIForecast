"""Feature engineering. Every feature at origin quarter t uses data up to and including t only.

Target: percent change in HPI from t to t + horizon (default 4 quarters = next year).
"""
import numpy as np
import pandas as pd

from .config import Config

FEATURES = [
    "yoy",        # growth over the last 4 quarters (this is also the persistence baseline)
    "g1q_ann",    # last quarter's growth, annualised
    "accel",      # last 2 quarters annualised minus yoy (short-run acceleration)
    "mom",        # momentum: yoy now minus yoy one year ago
    "g8_ann",     # 2-year annualised growth
    "g12_ann",    # 3-year annualised growth
    "g20_ann",    # 5-year annualised growth
    "vol8",       # std of quarterly log growth, last 8 quarters
    "vol20",      # std of quarterly log growth, last 20 quarters
    "drawdown",   # % below the trailing 5-year peak
    "hist_mean",  # expanding mean of yoy growth for the place (history-average baseline)
    "nat_yoy",    # median yoy growth across all places (market-wide trend)
    "rel_yoy",    # place yoy minus market-wide yoy
]


def _long(frame: pd.DataFrame, name: str) -> pd.Series:
    out = frame.copy()
    out.index.name = "t"
    out.columns.name = "place_id"
    return (out.reset_index()
               .melt(id_vars="t", var_name="place_id", value_name=name)
               .set_index(["t", "place_id"])[name])


def build_dataset(panel: pd.DataFrame, cfg: Config):
    """Return (all_rows, labeled_rows). all_rows has features (target may be NaN for the latest
    origins); labeled_rows additionally has a realised target."""
    h = cfg.horizon_q
    wide = panel.pivot(index="t", columns="place_id", values="index").sort_index()
    wide = wide.reindex(range(int(wide.index.min()), int(wide.index.max()) + 1))

    pct = lambda k: 100 * (wide / wide.shift(k) - 1)
    ann = lambda k: 100 * ((wide / wide.shift(k)) ** (4 / k) - 1)
    qret = 100 * np.log(wide / wide.shift(1))

    yoy = pct(4)
    nat = yoy.median(axis=1)
    nat_frame = pd.DataFrame(np.repeat(nat.values[:, None], wide.shape[1], axis=1),
                             index=wide.index, columns=wide.columns)

    feats = {
        "yoy": yoy,
        "g1q_ann": ann(1),
        "accel": ann(2) - yoy,
        "mom": yoy - yoy.shift(4),
        "g8_ann": ann(8),
        "g12_ann": ann(12),
        "g20_ann": ann(20),
        "vol8": qret.rolling(8).std(),
        "vol20": qret.rolling(20).std(),
        "drawdown": 100 * (wide / wide.rolling(20).max() - 1),
        "hist_mean": yoy.expanding(min_periods=8).mean(),
        "nat_yoy": nat_frame,
        "rel_yoy": yoy.sub(nat, axis=0),
        "target": 100 * (wide.shift(-h) / wide - 1),
    }
    df = pd.concat([_long(f, n) for n, f in feats.items()], axis=1).reset_index()

    names = panel.drop_duplicates("place_id").set_index("place_id")["place_name"]
    df["place_name"] = df["place_id"].map(names)
    df["yr"] = df["t"] // 4
    df["q"] = df["t"] % 4 + 1

    all_rows = df.dropna(subset=FEATURES).reset_index(drop=True)
    labeled = all_rows.dropna(subset=["target"]).reset_index(drop=True)
    print(f"[features] {len(labeled):,} labeled place-quarters "
          f"({labeled['yr'].min()}-{labeled['yr'].max()} origins), {len(FEATURES)} features")
    return all_rows, labeled
