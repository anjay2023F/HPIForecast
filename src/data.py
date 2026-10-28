"""Load and clean the FHFA HPI data into a tidy quarterly panel.

Output columns: place_id, place_name, yr, q, t, index
where t = yr*4 + (q-1) is an integer quarter counter (no calendar-API surprises).
"""
import numpy as np
import pandas as pd

from .config import Config

# Column aliases so the simpler per-geography FHFA files also load.
_ALIASES = {
    "year": "yr",
    "qtr": "period",
    "quarter": "period",
    "month": "period",
    "state": "place_name",
    "cbsa": "place_id",
    "msa": "place_name",
}


def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.rename(columns={k: v for k, v in _ALIASES.items() if k in df.columns and v not in df.columns})
    return df


def clean(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = df.copy()

    # 1) Filter to the requested HPI product / geography / frequency (only where the column exists).
    for col, val in [("hpi_type", cfg.hpi_type), ("hpi_flavor", cfg.hpi_flavor),
                     ("frequency", cfg.frequency), ("level", cfg.level)]:
        if col in df.columns:
            mask = df[col].astype(str).str.strip().str.lower() == val.lower()
            if not mask.any():
                raise ValueError(f"No rows with {col} == {val!r}. Available: {sorted(df[col].dropna().unique())}")
            df = df[mask]

    needed = {"place_name", "yr", "period", cfg.index_col}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Input is missing columns {sorted(missing)}; found {list(df.columns)}")
    if "place_id" not in df.columns:
        df["place_id"] = df["place_name"]

    # 2) Quarter counter. Monthly data -> keep quarter-end months only.
    df["yr"] = pd.to_numeric(df["yr"], errors="coerce")
    df["period"] = pd.to_numeric(df["period"], errors="coerce")
    if cfg.frequency.lower() == "monthly":
        df = df[df["period"].isin([3, 6, 9, 12])]
        df["q"] = (df["period"] // 3).astype(int)
    else:
        df["q"] = df["period"]
    df = df.dropna(subset=["yr", "q"])
    df = df[np.isfinite(df["yr"]) & np.isfinite(df["q"])]
    df = df[(df["yr"] % 1 == 0) & (df["q"] % 1 == 0)]
    df["yr"], df["q"] = df["yr"].astype(int), df["q"].astype(int)
    df = df[df["q"].between(1, 4)]
    df["t"] = df["yr"] * 4 + (df["q"] - 1)

    # 3) Numeric, positive index; one row per place-quarter.
    df["index"] = pd.to_numeric(df[cfg.index_col], errors="coerce")
    df = df[(df["index"] > 0) & np.isfinite(df["index"])]
    df = df.dropna(subset=["place_id", "place_name"])
    for col in ["place_id", "place_name"]:
        df[col] = df[col].astype(str).str.strip()
        df = df[df[col] != ""]
    df["place_id"] = df["place_id"].astype(str)
    df = df.drop_duplicates(["place_id", "t", "index"])
    if df.duplicated(["place_id", "t"]).any():
        raise ValueError("Conflicting HPI values for the same place-quarter.")

    # 4) Drop places with short histories.
    counts = df.groupby("place_id")["t"].count()
    keep = counts[counts >= cfg.min_years * 4].index
    dropped = len(counts) - len(keep)
    df = df[df["place_id"].isin(keep)]
    if df.empty:
        raise ValueError("No places left after cleaning; lower min_years or check filters.")

    # 5) Report internal gaps (features are computed on a contiguous grid, so gaps become NaN, not wrong).
    gaps = 0
    for _, g in df.groupby("place_id"):
        gaps += (g["t"].max() - g["t"].min() + 1) - len(g)
    print(f"[data] {df['place_id'].nunique()} places kept ({dropped} dropped for short history), "
          f"{len(df):,} place-quarters, {df['yr'].min()}Q1-{df['yr'].max()}, {gaps} internal gaps")

    return (df[["place_id", "place_name", "yr", "q", "t", "index"]]
            .sort_values(["place_id", "t"]).reset_index(drop=True))
