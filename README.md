# KNN Time Series Model + GraphQL API

This repository trains and serves a K-Nearest Neighbors (KNN) time series regression model for TSLA-based features. It includes:

- Data preparation with optional lag features, time index feature, and winsorization.
- KNN regression with feature scaling and optional feature weighting.
- Post-processing (smoothing, ratio clipping) for more stable predictions.
- Training + evaluation reports with metrics and plots.
- Day-by-day (walk-forward) predictions.
- A GraphQL API and a CLI client for fetching model outputs.

## Repository layout

- `services/knn_timeseries/`
  - `train.py`: model training + evaluation and artifact saving
  - `predict_daybyday.py`: walk-forward predictions using a saved model
  - `data.py`: dataset loading, feature engineering, splits, winsorization
  - `model.py`: KNN pipeline (scaler + optional feature weighting)
  - `evaluation.py`: metrics, smoothing, ratio clipping helpers
  - `graphql_resolvers.py`: GraphQL resolvers
  - `graphql_schema.py`: GraphQL schema loader
- `configs/`: YAML configs (example: `configs/knn_tsla_tuned.yaml`)
- `models/`: saved model artifacts (`.joblib`)
- `artifacts/`: training and prediction outputs (reports, plots, CSVs)
- `schema.graphql`: GraphQL schema definition
- `app.py`: FastAPI + GraphQL entrypoint
- `run_bot.py`: CLI GraphQL client

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Configuration

Model behavior is driven by YAML config files in `configs/`. Example: `configs/knn_tsla_tuned.yaml`.

### `data` section

- `csv_path`: path to the CSV dataset
- `date_column`: date column name (parsed as datetime)
- `target_column`: target variable to predict
- `target_transform`: `none`, `ratio`, or `log_ratio`
  - `ratio`: uses `y / target_base_column`
  - `log_ratio`: uses `log(y / target_base_column)`
- `target_base_column`: required when using `ratio` or `log_ratio`
- `feature_columns`: explicit feature list (if omitted, all numeric non-target columns are used)
- `lags`: number of lag steps to add for each feature
- `lag_columns`: optional override of which features to lag
- `add_time_index`: adds a normalized time index feature
- `time_index_weight`: scales the time index feature
- `winsorize`: clip feature values to quantile bounds from training set
- `winsorize_quantiles`: low/high quantiles for winsorization
- `split_date`: explicit date split between train and test
- `train_start_date`: optional lower bound for the training window
- `test_size`: fallback split ratio if `split_date` is not used

### `model` section

- `n_neighbors`, `weights`, `metric`, `p`, `algorithm`, `leaf_size`
- The pipeline uses `StandardScaler` and a `FeatureWeighter` (optional)

### `postprocess` section

- `smoothing`: `none`, `ema`, or `rolling_median`
- `ema_alpha`, `rolling_window`
- `ratio_clip`: `none`, `quantile`, or `bounds`
- `ratio_clip_quantiles` or `ratio_clip_bounds`

## Training

Train and evaluate the model with a config:

```bash
python -m services.knn_timeseries.train --config configs/knn_tsla_tuned.yaml
```

Outputs:

- `models/knn_<csv>_<target>_<run_id>.joblib`
- `artifacts/<run_id>/report.json`
- `artifacts/<run_id>/predictions.csv`
- `artifacts/<run_id>/plots/*.png`

The `report.json` includes metrics, postprocess settings, winsorization bounds, and model metadata.

## Day-by-day prediction (walk-forward)

This runs prediction for every cutoff date (useful for time-aware evaluation):

```bash
python -m services.knn_timeseries.predict_daybyday \
  --config configs/knn_tsla_tuned.yaml \
  --only-last \
  --with-tpg
```

Outputs:

- `artifacts/daybyday_<run_id>/predictions_daybyday.csv`
- `artifacts/daybyday_<run_id>/report_daybyday.json`
- optional TPG plots + metrics if `--with-tpg` is used

Each row contains `cutoff_date`, `date`, `pred_<target>`, and `actual_<target>`.

## Full pipeline (train + day-by-day)

```bash
python -m services.knn_timeseries.run_all --config configs/knn_tsla_tuned.yaml --with-tpg
```

## GraphQL API

Start the API server:

```bash
uvicorn app:app --reload
```

- GraphQL endpoint: `http://localhost:8000/graphql/`
- Health endpoint: `http://localhost:8000/health`

### Example GraphQL queries

Train:

```graphql
mutation Train($input: TrainInput) {
  train(input: $input) {
    report
  }
}
```

Variables:

```json
{
  "input": {
    "configPath": "configs/knn_tsla_tuned.yaml",
    "runId": "optional_run_id"
  }
}
```

Predict (day-by-day):

```graphql
mutation PredictDayByDay($input: PredictDayByDayInput!) {
  predictDayByDay(input: $input) {
    columns
    rows {
      cutoffDate
      date
      predicted
      actual
    }
    report
  }
}
```

Variables:

```json
{
  "input": {
    "configPath": "configs/knn_tsla_tuned.yaml",
    "withTpg": true,
    "onlyLast": true,
    "previewLimit": 50
  }
}
```

### JSON scalar

The `report` fields are returned via a `JSON` scalar so the full nested report can be inspected from GraphQL.

## CLI GraphQL client (`run_bot.py`)

Run prediction:

```bash
python run_bot.py --mode predict --with-tpg --pretty
```

Run training:

```bash
python run_bot.py --mode train --pretty
```

Show all rows (no preview limit) by omitting `--preview-limit`:

```bash
python run_bot.py --mode predict --pretty
```

The client prints:

- Full GraphQL response
- `Correction values` (postprocess settings, winsorization bounds, feature weights)
- `TPG error rate` (derived from TPG `accuracy` metric)

## Metrics and reports

Training report (`report.json`) includes:

- `metrics.sklearn`: MAE, RMSE, MAPE, R2
- `metrics.errors.ape_summary`: APE distribution summary
- `metrics.tpg`: TPG plot metrics and image paths
- `metrics.calculated_yield`: optional yield metric
- `postprocess`: smoothing/ratio-clip settings
- `winsor_bounds`: per-feature clipping bounds

Day-by-day report (`report_daybyday.json`) includes:

- output CSV path, date range, and postprocess settings
- optional TPG metrics when `--with-tpg` is used

### TPG accuracy note

TPG metrics come from `tpg.py`. The `accuracy` value is **not** classification accuracy; it is a relative error metric computed inside `sir_parameters`. The CLI prints `TPG error rate` as `accuracy * 100` for readability.

## Notes

- Predictions use the same feature engineering rules as training.
- If `target_transform` is `ratio` or `log_ratio`, the model predicts ratios and then re-scales back to the original target using `target_base_column`.
- Smoothing and ratio clipping are applied after prediction when configured.
