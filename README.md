# Forecasting next-year house price growth from FHFA HPI history

**Question:** how accurately can ML models forecast next-year house price growth using only historical FHFA HPI data, compared with simple baselines, and how does accuracy change across market regimes?

## Run from a fresh clone

Install Git and Python 3.13 first. On Windows, enable the Python launcher (`py`)
when installing Python. Internet access is needed to install dependencies and
download the FHFA data. Raw data, processed data, and `.venv` are not included
in the repository.

Clone this repository using its GitHub **Code** URL, then open a terminal in
the `HPIForecast` folder (the folder containing `run_pipeline.py` and
`requirements.txt`). Run all commands below from that folder.

### Windows (PowerShell)

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts/prepare_data.py
.\.venv\Scripts\python.exe run_pipeline.py
.\.venv\Scripts\python.exe scripts/verify_pipeline.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

These commands use the virtual environment directly; activation is not required.
In VS Code, select `.venv\Scripts\python.exe` with **Python: Select Interpreter**.

### macOS / Linux

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/prepare_data.py
.venv/bin/python run_pipeline.py
.venv/bin/python scripts/verify_pipeline.py
.venv/bin/python -m unittest discover -s tests -v
```

`prepare_data.py` downloads the real FHFA CSV automatically if it is missing,
then cleans it, creates features, and writes train/test files to `data/processed/`.
`run_pipeline.py` trains and evaluates the models, prints a metrics summary, and
writes predictions, metrics, forecasts, plots, and another copy of the prepared
data to `outputs/`. The full run can take several minutes.
`verify_pipeline.py` audits the default real-data outputs; a successful run prints
two `PASS` lines. The unit-test command should finish with `OK`.

Start with `outputs/metrics_overall.csv`, `outputs/predictions.csv`, and
`outputs/latest_forecast.csv` to inspect the results. This project is a command-line
pipeline; it does not start a web server.

For later runs, reuse `.venv` and run `run_pipeline.py` with the environment's
Python executable. Run `prepare_data.py` again when you need to regenerate the
standalone split files. It reuses an existing raw CSV; to explicitly refresh
the raw download on Windows, run:

```powershell
.\.venv\Scripts\python.exe scripts/download_data.py
.\.venv\Scripts\python.exe scripts/prepare_data.py
.\.venv\Scripts\python.exe run_pipeline.py
```

Data: the **Master HPI CSV** ("Master HPI Data" -> Data [CSV]) at https://www.fhfa.gov/data/hpi/datasets.
It contains every geography in one long file, filtered by `hpi_flavor` and `level`.
If `download_data.py` fails, copy the CSV link from that page and pass `--url`, or download it in a browser.

Useful options: `--level MSA`, `--flavor purchase-only`, `--index-col index_sa`, `--first-test-year 2008`.
Edit defaults in `src/config.py`.

For a synthetic smoke test only, run `run_pipeline.py --synthetic` with your
environment's Python executable. Its results do not measure real housing markets.

## Real data preparation

The downloader uses the official FHFA Master HPI CSV linked from the supplied
[FHFA datasets page](https://www.fhfa.gov/data/hpi/datasets?tab=additional-data).
Defaults select quarterly, traditional, state-level, all-transactions HPI
(`index_nsa`). Synthetic data is used only when explicitly requested.

Run `python scripts/prepare_data.py --first-test-year 2005` to download the real
CSV if absent and write `data/processed/cleaned_panel.csv`, `features.csv`,
`train.csv`, `test.csv`, `purged.csv`, and `metadata.json`.
The training set contains origins whose next-year target ends strictly before
2005Q1; the test set contains labeled origins from 2005Q1 onward. Origins before
the cutoff with targets reaching into the test period go into `purged.csv`.
Every state uses the same date cutoff; rows are never randomly shuffled.
Choose a later cutoff, such as `--first-test-year 2020`, for a smaller holdout.

Cleaning normalizes column names, selects a single HPI product, rejects invalid
dates, missing identifiers, and nonpositive or nonfinite HPI values, removes
identical observations, rejects conflicting duplicates, and drops places with
less than 15 years of observations. Missing quarters stay missing; no future
values are used for interpolation. Feature rows needing missing history and
rows without realized targets are excluded from train/test. Features use only
observations through the origin quarter. CSV features remain unscaled; the
linear model's existing pipeline fits its scaler on training rows only.

`run_pipeline.py` also saves these preparation artifacts under
`outputs/processed/` before running its annual expanding-window evaluation.
This rolling evaluation refits using newly available historical labels; it is
distinct from fitting a single model once on the saved training CSV. FHFA data
are revised, so this is an evaluation on the downloaded vintage, not a
historical real-time vintage backtest.

Generated raw/processed CSVs and model outputs are ignored by Git. See
[`data/README.md`](data/README.md) and the committed source manifest for the
snapshot hash and guidance on archiving a frozen dataset.

After a default real-data pipeline run, verify its saved outputs with
`python scripts/verify_pipeline.py`. This independently compares the cleaned
panel to the raw FHFA values, recomputes growth features and targets, checks
split/training cutoffs, and recomputes prediction metrics.

## Design

| Piece | Choice |
|---|---|
| Target | % change in HPI from quarter *t* to *t+4* (next year), per place |
| Features (at *t*, no look-ahead) | yoy growth, last-quarter growth, acceleration, momentum (yoy change), 2/3/5-yr annualised growth, volatility (8 and 20 quarters), drawdown from 5-yr peak, expanding historical mean growth, market-wide yoy and place-minus-market |
| Baselines | **Persistence** (next year = last year's growth); **Historical mean** (place's expanding average growth) |
| Models | Linear Regression, Random Forest, Gradient Boosting (scikit-learn) |
| Validation | Expanding-window **walk-forward**, refit each year. Training uses only origins whose target was already realised before the test year starts (`t + 4 < first quarter of test year`). No random shuffling. |
| Metrics | MAE, RMSE, bias; MAE improvement vs persistence |
| Regimes | Each origin labelled by realised national (median) next-year growth: `decline` < 0%, `steady` 0-7%, `boom` > 7% (adjust in `Config.regime_bounds`). Evaluation label only, never a feature. |
| Error analysis | by year, by regime, worst places, largest individual errors |

## Outputs (`outputs/`)

`metrics_overall.csv`, `metrics_by_regime.csv`, `metrics_by_year.csv`, `worst_places.csv`, `largest_errors.csv`, `predictions.csv`, `latest_forecast.csv` (best ML model's forecast from the latest quarter), and figures: `pred_vs_actual.png`, `national_timeseries.png`, `mae_by_year.png`, `mae_by_regime.png`, `worst_places.png`.

## Notes and caveats

- Test windows overlap (quarterly origins, 4-quarter targets), so observations are autocorrelated and metrics are less independent than the row count suggests. Do not over-read small MAE gaps.
- Places are pooled in one model per fit; the market-wide features let the models see the common cycle.
- All-transactions indexes are not seasonally adjusted. The target is year-over-year, so seasonality largely cancels.
- State-level data has only ~51 series; MSA-level (`--level MSA`) gives more rows but more gaps/short histories.
- Reading the results: if models fail to beat persistence, that is itself the finding the project proposal describes (limits of forecasting from HPI history alone).
- The best model is selected retrospectively from the reported evaluation period. Its winning score is not an independent estimate after model selection; use a separate validation period and untouched final holdout for that claim.
- Time cutoffs refer to observation quarters, not actual publication dates. Historical release lags and revised vintages are not modeled.

## Layout

```
run_pipeline.py        entry point
src/config.py          settings
src/data.py            load + clean -> tidy quarterly panel
src/features.py        features and target
src/models.py          baselines list + sklearn models
src/evaluate.py        walk-forward, metrics, regimes, error tables
src/plots.py           figures
src/synthetic.py       fake data in FHFA layout for smoke tests
scripts/download_data.py
```
