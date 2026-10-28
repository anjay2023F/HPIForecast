"""End-to-end pipeline:  load -> clean -> features -> walk-forward evaluation -> metrics -> plots.

    python run_pipeline.py                       # uses data/raw/hpi_master.csv
    python run_pipeline.py --data path/to.csv --level MSA --first-test-year 2008
    python run_pipeline.py --synthetic           # smoke test on fake data (NOT real results)
"""
import argparse
import os
import sys

import pandas as pd
from sklearn.base import clone

from src import evaluate as ev
from src import plots
from src.config import Config
from src.data import clean, load_raw
from src.features import FEATURES, build_dataset
from src.models import BASELINES, get_models
from src.synthetic import make_synthetic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=Config.raw_path)
    ap.add_argument("--synthetic", action="store_true", help="generate fake data to test the pipeline")
    ap.add_argument("--level", default=Config.level)
    ap.add_argument("--flavor", default=Config.hpi_flavor)
    ap.add_argument("--index-col", default=Config.index_col)
    ap.add_argument("--first-test-year", type=int, default=Config.first_test_year)
    ap.add_argument("--out", default=Config.out_dir)
    a = ap.parse_args()

    cfg = Config(raw_path=a.data, level=a.level, hpi_flavor=a.flavor, index_col=a.index_col,
                 first_test_year=a.first_test_year, out_dir=a.out)
    if a.synthetic:
        cfg.raw_path = make_synthetic("data/synthetic/hpi_master_synthetic.csv")
        cfg.out_dir = a.out if a.out != Config.out_dir else "outputs_synthetic"
        print("[!] SYNTHETIC DATA: results only demonstrate that the pipeline runs.")
    if not os.path.exists(cfg.raw_path):
        sys.exit(f"Data file not found: {cfg.raw_path}\n"
                 "Download the Master HPI CSV from https://www.fhfa.gov/data/hpi/datasets "
                 "(or run scripts/download_data.py) and save it there, or pass --data.")
    os.makedirs(cfg.out_dir, exist_ok=True)

    panel = clean(load_raw(cfg.raw_path), cfg)
    all_rows, labeled = build_dataset(panel, cfg)
    models = get_models(cfg.random_state)
    pred_cols = BASELINES + list(models)

    res = ev.walk_forward(labeled, cfg, models, FEATURES)
    res.to_csv(f"{cfg.out_dir}/predictions.csv", index=False)

    overall = ev.score(res, pred_cols)
    by_regime = ev.score(res, pred_cols, by="regime")
    by_year = ev.score(res, pred_cols, by="yr")
    overall.to_csv(f"{cfg.out_dir}/metrics_overall.csv", index=False)
    by_regime.to_csv(f"{cfg.out_dir}/metrics_by_regime.csv", index=False)
    by_year.to_csv(f"{cfg.out_dir}/metrics_by_year.csv", index=False)

    best_ml = overall[overall["model"].isin(models)].sort_values("MAE").iloc[0]["model"]
    worst = ev.worst_places(res, pred_cols[:1] + [best_ml])
    worst.to_csv(f"{cfg.out_dir}/worst_places.csv")
    ev.largest_errors(res, best_ml).to_csv(f"{cfg.out_dir}/largest_errors.csv", index=False)

    plots.pred_vs_actual(res, pred_cols, cfg.out_dir)
    plots.national_timeseries(res, ["Persistence", best_ml], cfg.out_dir)
    plots.mae_by_year(by_year, pred_cols, cfg.out_dir)
    plots.mae_by_regime(by_regime, pred_cols, cfg.out_dir)
    plots.worst_places_plot(worst, pred_cols[:1] + [best_ml], cfg.out_dir)

    # Forecast for the latest available origin using the best ML model trained on all labeled rows.
    latest = all_rows[all_rows["t"] == all_rows["t"].max()]
    final = clone(models[best_ml]).fit(labeled[FEATURES], labeled["target"])
    fc = latest[["place_id", "place_name", "yr", "q"]].copy()
    fc[f"pred_next_{cfg.horizon_q}q_growth_%"] = final.predict(latest[FEATURES])
    fc["persistence_%"] = latest["yoy"].values
    fc.sort_values(fc.columns[-2], ascending=False).to_csv(f"{cfg.out_dir}/latest_forecast.csv", index=False)

    pd.set_option("display.width", 140)
    print("\n=== Overall (walk-forward, test origins "
          f"{res['yr'].min()}-{res['yr'].max()}) ===")
    print(overall.drop(columns="group").round(3).to_string(index=False))
    print("\n=== By realised market regime ===")
    print(by_regime.round(3).to_string(index=False))
    print(f"\nBest ML model by MAE: {best_ml}. Outputs in {cfg.out_dir}/")


if __name__ == "__main__":
    main()
