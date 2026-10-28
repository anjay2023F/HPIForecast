"""Independently audit a completed default FHFA pipeline run against raw data."""
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.features import FEATURES
from src.models import BASELINES, get_models


def verify(raw_path, out):
    raw = pd.read_csv(raw_path, low_memory=False)
    raw = raw[(raw.hpi_type == "traditional") & (raw.hpi_flavor == "all-transactions") &
              (raw.frequency == "quarterly") & (raw.level == "State")].copy()
    raw["t"] = raw.yr * 4 + raw.period - 1
    raw.place_id = raw.place_id.astype(str)
    truth = raw.set_index(["place_id", "t"]).index_nsa.sort_index()
    panel = pd.read_csv(out / "processed/cleaned_panel.csv")
    observed = panel.set_index(["place_id", "t"])["index"].sort_index()
    pd.testing.assert_index_equal(observed.index, truth.index)
    np.testing.assert_allclose(observed, truth, rtol=0, atol=1e-10)
    assert not panel.duplicated(["place_id", "t"]).any()

    features = pd.read_csv(out / "processed/features.csv")
    keys = pd.MultiIndex.from_frame(features[["place_id", "t"]])
    current = truth.reindex(keys).to_numpy()
    def lag(k):
        return truth.reindex(pd.MultiIndex.from_arrays(
            [features.place_id, features.t - k], names=truth.index.names)).to_numpy()
    np.testing.assert_allclose(features.yoy, 100 * (current / lag(4) - 1), atol=1e-10)
    np.testing.assert_allclose(features.g1q_ann, 100 * ((current / lag(1)) ** 4 - 1), atol=1e-10)
    np.testing.assert_allclose(features.g20_ann, 100 * ((current / lag(20)) ** .2 - 1), atol=1e-10)
    future = truth.reindex(pd.MultiIndex.from_arrays(
        [features.place_id, features.t + 4], names=truth.index.names)).to_numpy()
    np.testing.assert_allclose(features.target, 100 * (future / current - 1),
                               atol=1e-10, equal_nan=True)

    # Independently calculate the remaining features as individual time series.
    pieces = []
    for place, rows in panel.groupby("place_id"):
        series = rows.set_index("t")["index"].sort_index()
        series = series.reindex(range(int(series.index.min()), int(series.index.max()) + 1))
        yoy = 100 * (series / series.shift(4) - 1)
        log_growth = 100 * np.log(series / series.shift(1))
        expected = pd.DataFrame({
            "accel": 100 * ((series / series.shift(2)) ** 2 - 1) - yoy,
            "mom": yoy - yoy.shift(4),
            "g8_ann": 100 * ((series / series.shift(8)) ** .5 - 1),
            "g12_ann": 100 * ((series / series.shift(12)) ** (1 / 3) - 1),
            "vol8": log_growth.rolling(8).std(),
            "vol20": log_growth.rolling(20).std(),
            "drawdown": 100 * (series / series.rolling(20).max() - 1),
            "hist_mean": yoy.expanding(min_periods=8).mean(),
            "yoy": yoy,
        })
        expected["place_id"] = place
        pieces.append(expected.reset_index())
    expected = pd.concat(pieces, ignore_index=True)
    expected["nat_yoy"] = expected.groupby("t").yoy.transform("median")
    expected["rel_yoy"] = expected.yoy - expected.nat_yoy
    comparison = features.merge(expected, on=["place_id", "t"], validate="one_to_one",
                                suffixes=("", "_expected"))
    for name in ["accel", "mom", "g8_ann", "g12_ann", "vol8", "vol20",
                 "drawdown", "hist_mean", "nat_yoy", "rel_yoy"]:
        np.testing.assert_allclose(comparison[name], comparison[f"{name}_expected"], atol=1e-10)

    train = pd.read_csv(out / "processed/train.csv")
    test = pd.read_csv(out / "processed/test.csv")
    assert train.t.max() + 4 < test.t.min()
    for frame in [train, test]:
        assert np.isfinite(frame[FEATURES + ["target"]]).all().all()
    res = pd.read_csv(out / "predictions.csv")
    merged = res.merge(test, on=["place_id", "t"], validate="one_to_one", suffixes=("", "_feature"))
    assert len(merged) == len(res) == len(test)
    np.testing.assert_allclose(merged.actual, merged.target, atol=1e-10)
    np.testing.assert_allclose(merged.Persistence, merged.yoy, atol=1e-10)
    np.testing.assert_allclose(merged["Historical mean"], merged.hist_mean, atol=1e-10)
    labeled = features.dropna(subset=["target"])
    for year, group in res.groupby("yr"):
        assert (group.n_train == (labeled.t + 4 < year * 4).sum()).all()
    models = BASELINES + list(get_models())
    assert np.isfinite(res[models + ["actual"]]).all().all()
    national = res.groupby("t").actual.transform("median")
    np.testing.assert_allclose(res.national_actual, national, atol=1e-10)
    assert (res.regime == np.select([national < 0, national > 7],
                                   ["decline", "boom"], default="steady")).all()
    for filename, by in [("metrics_overall.csv", None), ("metrics_by_year.csv", "yr"),
                         ("metrics_by_regime.csv", "regime")]:
        metrics = pd.read_csv(out / filename)
        groups = [("all", res)] if by is None else res.groupby(by)
        for key, group in groups:
            table = metrics[metrics["group"].astype(str) == str(key)].set_index("model")
            baseline_mae = (group.Persistence - group.actual).abs().mean()
            for model in models:
                error = group[model] - group.actual
                assert table.loc[model, "n"] == len(group)
                np.testing.assert_allclose(
                    table.loc[model, ["MAE", "RMSE", "Bias", "MAE_vs_persistence_%"]].to_numpy(dtype=float),
                    [error.abs().mean(), np.sqrt((error ** 2).mean()), error.mean(),
                     100 * (1 - error.abs().mean() / baseline_mae)], atol=1e-10)
    forecast = pd.read_csv(out / "latest_forecast.csv")
    assert len(forecast) == panel.place_id.nunique()
    assert (forecast.yr * 4 + forecast.q - 1 == panel.t.max()).all()
    assert np.isfinite(forecast.filter(like="%")).all().all()
    latest = features[features.t == panel.t.max()].set_index("place_id")
    np.testing.assert_allclose(forecast["persistence_%"], latest.loc[forecast.place_id, "yoy"], atol=1e-10)
    for filename in ["pred_vs_actual", "national_timeseries", "mae_by_year", "mae_by_regime", "worst_places"]:
        assert (out / f"{filename}.png").stat().st_size > 0
    print(f"PASS: {len(panel):,} raw HPI observations match; features/targets independently verified.")
    print(f"PASS: {len(res):,} predictions, training cutoffs, metrics, latest forecasts, and five plots verified.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/raw/hpi_master.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    verify(args.data, args.out)
