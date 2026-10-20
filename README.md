# Universal Sensor Forecasting (Streamlit)

A Streamlit app for multi-indicator industrial time-series exploration, pooled model comparison, segment-wise visualisation, and recursive future forecasts.

## Features

- Upload CSV, XLSX, or XLS data.
- Map the indicator, numeric reading, and optional timestamp columns.
- Clean numeric readings and inspect data and per-indicator statistics on the Upload & Data tab.
- Compare one global/pooled model across all indicators using a chronological validation split.
- Provide a focused Model Recommendation tab with model comparison metrics, recommendation rationale, and model pros and cons.
- Visualize actual readings divided into three sequential segments.
- Train a user-selected global model on the first 40 readings per indicator to produce historical holdout predictions for later observed readings; then refit it on the full history and forecast future readings.
- Handles common column names such as `TagName`, `Timestamp`, `Reading`, `CurrentTime`, and `CurrentValue`.

The app intentionally contains no data-export or download feature. The Upload & Data tab shows the cleaned dataset and descriptive statistics; the Model Recommendation tab focuses on model comparison and recommendation.

## Run locally

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies and run:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Streamlit will display a local URL, normally `http://localhost:8501`.

## Expected input

Each row should contain an indicator/sensor identifier and its numeric reading. A timestamp is strongly recommended.

Example:

| TagName | Timestamp | Reading |
|---|---|---:|
| 200PI0802/AI1/PV | 2022-10-01 13:29:00 | 0.2371 |
| 200PI0802/AI1/PV | 2022-10-01 13:29:05 | 0.2372 |
| 200PI1002/AI1/PV | 2022-10-01 13:29:00 | 0.2817 |

For each indicator, rows should represent chronological observations. If no timestamp is available, the app uses file row order within each indicator.

## Model notes

The machine-learning models are global: one estimator is fit on pooled rows from all indicators. Inputs include recent lags scaled using training-period statistics and categorical indicator identity/type. The prediction target is the next reading's change from the latest observed value, scaled per indicator. Recursive forecasts feed each prediction into the next forecast step.

The app evaluates models on a chronological holdout using one-step predictions whose lag features use prior observed readings. This is not the same as multi-step recursive forecast validation. Treat recommendations as provisional, especially with small datasets. Forecast values are not alarms, equipment-health determinations, or safety guidance.


## If "OpenBLAS error: Memory allocation still failed after 10 retries, giving up." comes up:
For CMD:
```bash
set OPENBLAS_NUM_THREADS=1
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
streamlit run app.py
```

For Powershell:
```bash
$env:OPENBLAS_NUM_THREADS="1"
$env:OMP_NUM_THREADS="1"
$env:MKL_NUM_THREADS="1"
streamlit run app.py
```
