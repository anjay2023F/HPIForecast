# Forecasting next-year house price growth from FHFA HPI history

**Question:** how accurately can ML models forecast next-year house price growth using only historical FHFA HPI data, compared with simple baselines, and how does accuracy change across market regimes?

## Quick start

```bash
pip install -r requirements.txt
python scripts/download_data.py          # or save the Master HPI CSV to data/raw/hpi_master.csv yourself
python run_pipeline.py                   # states, all-transactions index, quarterly
python run_pipeline.py --synthetic       # smoke test on fake data (results are meaningless)
```

Data: the **Master HPI CSV** ("Master HPI Data" -> Data [CSV]) at https://www.fhfa.gov/data/hpi/datasets.
It contains every geography in one long file, filtered by `hpi_flavor` and `level`.
If `download_data.py` fails, copy the CSV link from that page and pass `--url`, or download it in a browser.

Useful options: `--level MSA`, `--flavor purchase-only`, `--index-col index_sa`, `--first-test-year 2008`.
Edit defaults in `src/config.py`.

## Design

| Piece | Choice |
|---|---|
| Target | % change in HPI from quarter *t* to *t+4* (next year), per place |
| Features (at *t*, no look-ahead) | yoy growth, last-quarter growth, acceleration, momentum (yoy change), 2/3/5-yr annualised growth, volatility (8 and 20 quarters), drawdown from 5-yr peak, expanding historical mean growth, market-wide yoy and place-minus-market |
| Baselines | **Persistence** (next year = last year's growth); **Historical mean** (place's expanding average growth) |
| Models | Linear Regression, Random Forest, Gradient Boosting (scikit-learn) |
| Validation | Expanding-window **walk-forward**, refit each year. Training uses only origins whose target was already realised before the test year starts (`t + 4 <= first quarter of test year`). No random shuffling. |
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
