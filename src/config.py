"""Central configuration for the HPI forecasting pipeline."""
from dataclasses import dataclass


@dataclass
class Config:
    # --- data selection (values match the FHFA "Master HPI" CSV columns) ---
    raw_path: str = "data/raw/hpi_master.csv"
    hpi_type: str = "traditional"
    hpi_flavor: str = "all-transactions"  # or "purchase-only", "expanded-data"
    level: str = "State"                  # "State", "MSA", "USA or Census Division", ...
    frequency: str = "quarterly"          # "quarterly" or "monthly" (monthly -> quarter-end months)
    index_col: str = "index_nsa"          # all-transactions has NSA only; purchase-only also has index_sa
    min_years: int = 15                   # drop places with less history than this

    # --- target / features ---
    horizon_q: int = 4                    # forecast horizon in quarters (4 = next year)

    # --- evaluation ---
    first_test_year: int = 2005           # first origin year scored (walk-forward, refit each year)
    last_test_year: int | None = None     # None = last year with a realised target
    min_train_rows: int = 500
    regime_bounds: tuple = (0.0, 7.0)     # national realised growth (%): decline < 0 <= steady <= 7 < boom

    random_state: int = 42
    out_dir: str = "outputs"
