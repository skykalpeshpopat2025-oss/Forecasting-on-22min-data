import re
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore", category=FutureWarning)

st.set_page_config(
    page_title="Universal Sensor Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_INFO = {
    "Last-value baseline": {
        "pros": "Very reliable starting point for stable signals; simple, fast, and easy to explain.",
        "cons": "Cannot anticipate a trend or sudden change; forecasts stay flat.",
        "kind": "Baseline",
    },
    "Global Ridge Regression": {
        "pros": "One pooled model across all indicators; stable with small datasets; can learn shared linear relationships.",
        "cons": "May miss nonlinear behaviour and complex interactions.",
        "kind": "Linear ML",
    },
    "Global Random Forest": {
        "pros": "Learns nonlinear relationships and interactions; usually robust to noisy readings.",
        "cons": "Needs enough varied examples; tree models do not extrapolate trends well beyond observed patterns.",
        "kind": "Tree ensemble",
    },
    "Global Extra Trees": {
        "pros": "Learns nonlinear relationships and often provides a strong, relatively low-variance ensemble.",
        "cons": "Can smooth away small changes and may not extrapolate trends; can be unstable on very small datasets.",
        "kind": "Tree ensemble",
    },
    "Global HistGradientBoosting": {
        "pros": "Can capture nonlinear patterns and interactions in a compact global model.",
        "cons": "Usually benefits from more training examples; may be less dependable with only a few readings per indicator.",
        "kind": "Boosted trees",
    },
}

MODEL_NAMES = list(MODEL_INFO.keys())
FEATURE_NUMERIC = [
    "Lag1_Scaled",
    "Lag2_Scaled",
    "Lag3_Scaled",
    "RollingMean3_Scaled",
    "RecentChange_Scaled",
]
CATEGORICAL_FEATURES = ["TagName", "IndicatorType"]


def infer_column(columns, candidates, default_index=0):
    lower_map = {str(c).strip().lower(): c for c in columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    # Accept common partial names if exact matches are not present.
    for c in columns:
        cl = str(c).strip().lower()
        for candidate in candidates:
            if candidate.lower() in cl:
                return c
    return columns[min(default_index, len(columns) - 1)]


def load_uploaded_file(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        try:
            return pd.read_csv(uploaded_file)
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, encoding="latin-1")
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded_file)
    raise ValueError("Please upload a CSV or Excel file (.csv, .xlsx, or .xls).")


def indicator_type(value) -> str:
    text = str(value).upper()
    match = re.match(r"^\d*(PDI|PI|LI)", text)
    if match:
        return match.group(1)
    return "Other"


def prepare_data(raw_df: pd.DataFrame, indicator_col, reading_col, timestamp_col):
    if raw_df.empty:
        raise ValueError("The uploaded file contains no rows.")

    working = pd.DataFrame(index=raw_df.index)
    working["TagName"] = raw_df[indicator_col].astype("string").str.strip()
    working["Reading"] = pd.to_numeric(raw_df[reading_col], errors="coerce")
    working["Original_Row"] = np.arange(len(raw_df))

    use_timestamp = timestamp_col is not None
    if use_timestamp:
        working["Timestamp"] = pd.to_datetime(raw_df[timestamp_col], errors="coerce")
        valid_ts_rate = float(working["Timestamp"].notna().mean())
        if valid_ts_rate >= 0.9 and working["Timestamp"].nunique() > 1:
            # Invalid timestamp rows remain usable but sort after timestamped rows.
            working = working.sort_values(
                ["Timestamp", "Original_Row"], na_position="last"
            )
            timestamp_usable = True
        else:
            timestamp_usable = False
            working["Timestamp"] = pd.NaT
    else:
        working["Timestamp"] = pd.NaT
        timestamp_usable = False

    original_count = len(working)
    missing_reading = int(working["Reading"].isna().sum())
    missing_indicator = int((
        working["TagName"].isna()
        | working["TagName"].fillna("").str.strip().eq("")
    ).sum())

    working = working.dropna(subset=["Reading", "TagName"]).copy()
    working = working[working["TagName"].astype(str).str.strip().ne("")].copy()
    if working.empty:
        raise ValueError("No usable rows remain. Check the selected indicator and reading columns.")

    working["TagName"] = working["TagName"].astype(str)
    working["Original_Row"] = working["Original_Row"].astype(int)
    if timestamp_usable:
        working = working.sort_values(["TagName", "Timestamp", "Original_Row"], na_position="last")
    else:
        working = working.sort_values(["TagName", "Original_Row"])

    working["Reading_Order"] = working.groupby("TagName").cumcount() + 1
    working["IndicatorType"] = working["TagName"].map(indicator_type)
    counts = working.groupby("TagName")["Reading_Order"].transform("max")
    segment_num = np.ceil(working["Reading_Order"] / counts * 3).clip(1, 3).astype(int)
    working["Segment"] = segment_num.map({1: "Segment 1", 2: "Segment 2", 3: "Segment 3"})

    report = {
        "original_rows": original_count,
        "clean_rows": len(working),
        "missing_reading": missing_reading,
        "missing_indicator": missing_indicator,
        "dropped_rows": original_count - len(working),
        "timestamp_usable": timestamp_usable,
        "timestamp_col_selected": timestamp_col,
        "timestamp_valid_rate": float(
            pd.to_datetime(raw_df[timestamp_col], errors="coerce").notna().mean()
        ) if timestamp_col is not None else 0.0,
        "indicator_col": indicator_col,
        "reading_col": reading_col,
    }
    return working.reset_index(drop=True), report


def build_lag_table(data: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for tag, group in data.groupby("TagName", sort=False):
        group = group.sort_values("Reading_Order").copy()
        vals = group["Reading"]
        group["Lag_1"] = vals.shift(1)
        group["Lag_2"] = vals.shift(2)
        group["Lag_3"] = vals.shift(3)
        group["Rolling_Mean_3"] = group[["Lag_1", "Lag_2", "Lag_3"]].mean(axis=1)
        group["Recent_Change"] = group["Lag_1"] - group["Lag_2"]
        group["Target_Raw_Change"] = group["Reading"] - group["Lag_1"]
        pieces.append(group)
    if not pieces:
        return pd.DataFrame()
    return pd.concat(pieces, ignore_index=True).sort_values(
        ["TagName", "Reading_Order"]
    ).reset_index(drop=True)


def calculate_stats(raw_train: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    stats = {}
    for tag, group in raw_train.groupby("TagName"):
        values = group["Reading"].astype(float)
        mean = float(values.mean())
        std = float(values.std(ddof=0)) if len(values) > 1 else 0.0
        # A neutral fallback keeps constant/near-constant indicators usable.
        scale = std if np.isfinite(std) and std > 1e-12 else max(abs(mean) * 0.01, 1.0)
        stats[tag] = {"mean": mean, "scale": scale}
    return stats


def transform_features(rows: pd.DataFrame, stats: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    out = pd.DataFrame(index=rows.index)
    out["TagName"] = rows["TagName"].astype(str)
    out["IndicatorType"] = rows["IndicatorType"].astype(str)

    for tag in rows["TagName"].unique():
        if tag not in stats:
            stats[tag] = {"mean": 0.0, "scale": 1.0}
    mean_vals = rows["TagName"].map({k: v["mean"] for k, v in stats.items()}).astype(float)
    scale_vals = rows["TagName"].map({k: v["scale"] for k, v in stats.items()}).astype(float)
    scale_vals = scale_vals.replace(0, 1.0)

    out["Lag1_Scaled"] = (rows["Lag_1"].astype(float) - mean_vals) / scale_vals
    out["Lag2_Scaled"] = (rows["Lag_2"].astype(float) - mean_vals) / scale_vals
    out["Lag3_Scaled"] = (rows["Lag_3"].astype(float) - mean_vals) / scale_vals
    out["RollingMean3_Scaled"] = (rows["Rolling_Mean_3"].astype(float) - mean_vals) / scale_vals
    out["RecentChange_Scaled"] = rows["Recent_Change"].astype(float) / scale_vals
    return out


def design_matrix(features: pd.DataFrame, columns=None):
    X = pd.get_dummies(
        features,
        columns=CATEGORICAL_FEATURES,
        dtype=float,
    )
    if columns is not None:
        X = X.reindex(columns=columns, fill_value=0.0)
    return X.astype(float)


def make_estimator(model_name):
    if model_name == "Global Ridge Regression":
        return Ridge(alpha=2.0)
    if model_name == "Global Random Forest":
        return RandomForestRegressor(
            n_estimators=180, min_samples_leaf=2,
            max_features=1.0, random_state=42, n_jobs=-1
        )
    if model_name == "Global Extra Trees":
        return ExtraTreesRegressor(
            n_estimators=180, min_samples_leaf=2,
            max_features=1.0, random_state=42, n_jobs=-1
        )
    if model_name == "Global HistGradientBoosting":
        return HistGradientBoostingRegressor(
            max_iter=120, learning_rate=0.05, max_leaf_nodes=15,
            l2_regularization=1.0, random_state=42
        )
    raise ValueError(f"Unsupported estimator: {model_name}")


def time_split(data: pd.DataFrame, train_fraction=0.8):
    cuts = {}
    for tag, group in data.groupby("TagName"):
        n = len(group)
        if n < 6:
            cuts[tag] = n
        else:
            cuts[tag] = int(max(4, min(np.floor(n * train_fraction), n - 2)))
    train_mask = data.apply(lambda r: r["Reading_Order"] <= cuts[r["TagName"]], axis=1)
    return cuts, train_mask


def evaluate_models(data: pd.DataFrame):
    lagged = build_lag_table(data)
    cuts, raw_train_mask = time_split(data, 0.8)
    raw_train = data.loc[raw_train_mask].copy()
    stats = calculate_stats(raw_train)

    train = lagged[
        lagged.apply(lambda r: r["Reading_Order"] <= cuts.get(r["TagName"], 0), axis=1)
        & lagged["Lag_3"].notna()
    ].copy()
    test = lagged[
        lagged.apply(lambda r: r["Reading_Order"] > cuts.get(r["TagName"], 0), axis=1)
        & lagged["Lag_3"].notna()
    ].copy()

    if train.empty or test.empty:
        return {
            "results": pd.DataFrame(),
            "sensor_results": pd.DataFrame(),
            "recommended": "Last-value baseline",
            "reason": "There are not enough readings per indicator to form a chronological validation split. The last-value baseline is the safest starting point.",
            "train_rows": len(train),
            "test_rows": len(test),
            "data_rows": lagged,
            "cuts": cuts,
            "stats": stats,
        }

    train["Target_Normalized_Change"] = train.apply(
        lambda r: r["Target_Raw_Change"] / stats.get(r["TagName"], {"scale": 1.0})["scale"], axis=1
    )
    X_train_raw = transform_features(train, stats)
    X_test_raw = transform_features(test, stats)
    X_train = design_matrix(X_train_raw)
    X_test = design_matrix(X_test_raw, columns=X_train.columns)
    y_train = train["Target_Normalized_Change"].astype(float)

    scales = test["TagName"].map({k: v["scale"] for k, v in stats.items()}).astype(float).to_numpy()
    baseline_pred = test["Lag_1"].astype(float).to_numpy()
    actual = test["Reading"].astype(float).to_numpy()

    results = []
    predictions_by_name = {"Last-value baseline": baseline_pred}
    for name in MODEL_NAMES[1:]:
        try:
            estimator = make_estimator(name)
            estimator.fit(X_train, y_train)
            delta = estimator.predict(X_test) * scales
            pred = test["Lag_1"].astype(float).to_numpy() + delta
            predictions_by_name[name] = pred
        except Exception as exc:
            # A failed model should not block the rest of the app.
            continue

    for name, pred in predictions_by_name.items():
        err = actual - pred
        results.append({
            "Model": name,
            "MAE": mean_absolute_error(actual, pred),
            "RMSE": float(np.sqrt(mean_squared_error(actual, pred))),
            "Mean_Normalized_MAE": float(np.mean(np.abs(err) / np.maximum(scales, 1e-12))),
            "Validation_Readings": len(test),
        })

    results_df = pd.DataFrame(results).sort_values("Mean_Normalized_MAE").reset_index(drop=True)
    recommended = str(results_df.iloc[0]["Model"]) if not results_df.empty else "Last-value baseline"
    baseline_score = results_df.loc[results_df["Model"] == "Last-value baseline", "Mean_Normalized_MAE"]
    best_score = results_df.iloc[0]["Mean_Normalized_MAE"] if not results_df.empty else np.nan
    if recommended != "Last-value baseline" and len(baseline_score) and best_score < float(baseline_score.iloc[0]) * 0.98:
        reason = (
            f"{recommended} achieved the lowest mean normalized MAE on the chronological holdout. "
            "This is a candidate recommendation, not proof that it will remain best on future operating conditions."
        )
    elif recommended == "Last-value baseline" or (len(baseline_score) and best_score >= float(baseline_score.iloc[0]) * 0.98):
        recommended = "Last-value baseline" if "Last-value baseline" in set(results_df["Model"]) else recommended
        reason = (
            "No machine-learning candidate showed a clear (at least 2%) normalized-error improvement over the simple baseline on this holdout. "
            "Start with the baseline, or use the top model only as an experimental option."
        )
    else:
        reason = f"{recommended} had the lowest measured mean normalized MAE, but validation data is limited."

    sensor_results = []
    for tag, group in test.groupby("TagName"):
        idx = group.index
        positions = test.index.get_indexer(idx)
        y = group["Reading"].to_numpy(dtype=float)
        scale = group["TagName"].map({k: v["scale"] for k, v in stats.items()}).to_numpy(dtype=float)
        row = {"TagName": tag, "Test_Readings": len(group)}
        for name, pred_all in predictions_by_name.items():
            pred = pred_all[positions]
            row[f"{name} Normalized MAE"] = float(np.mean(np.abs(y - pred) / np.maximum(scale, 1e-12)))
        sensor_results.append(row)

    return {
        "results": results_df,
        "sensor_results": pd.DataFrame(sensor_results),
        "recommended": recommended,
        "reason": reason,
        "train_rows": len(train),
        "test_rows": len(test),
        "data_rows": lagged,
        "cuts": cuts,
        "stats": stats,
        "validation_predictions": predictions_by_name,
        "validation_data": test,
    }



def predict_historical_holdout(data: pd.DataFrame, model_name: str, train_readings: int = 40) -> pd.DataFrame:
    """Fit the selected model on each sensor's first N readings and predict later observed readings.

    Historical predictions are one-step-ahead: lags for each held-out point use prior ACTUAL
    readings, so they can be compared directly with the observed holdout values. Future forecasting
    is performed separately by fitting the selected model on the complete available history.
    """
    lagged = build_lag_table(data)
    if lagged.empty:
        return pd.DataFrame()

    raw_train = data[data["Reading_Order"] <= train_readings].copy()
    if raw_train.empty:
        return pd.DataFrame()

    stats = calculate_stats(raw_train)
    train = lagged[
        (lagged["Reading_Order"] <= train_readings)
        & lagged["Lag_3"].notna()
    ].copy()
    holdout = lagged[
        (lagged["Reading_Order"] > train_readings)
        & lagged["Lag_3"].notna()
    ].copy()

    if holdout.empty:
        return pd.DataFrame()

    holdout = holdout.copy()
    holdout["Historical_Prediction"] = holdout["Lag_1"].astype(float).to_numpy()

    # The baseline does not need estimator training.
    if model_name != "Last-value baseline" and not train.empty:
        train["Target_Normalized_Change"] = train.apply(
            lambda r: r["Target_Raw_Change"] / stats.get(r["TagName"], {"scale": 1.0})["scale"],
            axis=1,
        )
        X_train = design_matrix(transform_features(train, stats))
        X_holdout = design_matrix(
            transform_features(holdout, stats), columns=X_train.columns
        )
        y_train = train["Target_Normalized_Change"].astype(float)

        try:
            estimator = make_estimator(model_name)
            estimator.fit(X_train, y_train)
            scales = holdout["TagName"].map(
                {tag: values["scale"] for tag, values in stats.items()}
            ).fillna(1.0).astype(float).to_numpy()
            predicted_change = estimator.predict(X_holdout) * scales
            holdout["Historical_Prediction"] = (
                holdout["Lag_1"].astype(float).to_numpy() + predicted_change
            )
        except Exception as exc:
            raise RuntimeError(f"Could not generate historical holdout predictions: {exc}") from exc

    keep_cols = ["TagName", "Reading_Order", "Reading", "Lag_1", "Historical_Prediction"]
    if "Timestamp" in holdout.columns:
        keep_cols.insert(2, "Timestamp")
    result = holdout[keep_cols].rename(columns={"Reading": "Actual_Reading"})
    result["Training_Readings_Per_Indicator"] = train_readings
    result["Model"] = model_name
    return result.sort_values(["TagName", "Reading_Order"]).reset_index(drop=True)


def fit_full_model_and_forecast(data: pd.DataFrame, model_name: str, horizon: int):
    lagged = build_lag_table(data)
    complete = lagged[lagged["Lag_3"].notna()].copy()
    full_stats = calculate_stats(data)

    forecast_records = []
    fitted_model = None
    X_columns = None

    if model_name != "Last-value baseline":
        if complete.empty:
            raise ValueError("At least four readings per indicator are needed to train the selected global model.")
        complete["Target_Normalized_Change"] = complete.apply(
            lambda r: r["Target_Raw_Change"] / full_stats.get(r["TagName"], {"scale": 1.0})["scale"], axis=1
        )
        features = transform_features(complete, full_stats)
        X = design_matrix(features)
        X_columns = list(X.columns)
        y = complete["Target_Normalized_Change"].astype(float)
        fitted_model = make_estimator(model_name)
        fitted_model.fit(X, y)

    for tag, group in data.groupby("TagName", sort=False):
        group = group.sort_values("Reading_Order")
        recent_values = group["Reading"].astype(float).tolist()
        if not recent_values:
            continue
        last_time = group["Timestamp"].dropna().max() if group["Timestamp"].notna().any() else pd.NaT
        forecast_time_origin = last_time
        if pd.notna(last_time):
            diffs = group["Timestamp"].dropna().sort_values().diff().dt.total_seconds()
            diffs = diffs[(diffs > 0) & np.isfinite(diffs)]
            seconds_step = float(diffs.median()) if len(diffs) else 1.0
            if not np.isfinite(seconds_step) or seconds_step <= 0:
                seconds_step = 1.0
        else:
            seconds_step = 1.0
        stats = full_stats.get(tag, {"mean": 0.0, "scale": 1.0})
        scale = stats["scale"]
        for step in range(1, horizon + 1):
            last = float(recent_values[-1])
            if model_name == "Last-value baseline" or fitted_model is None or len(recent_values) < 4:
                prediction = last
            else:
                lag1, lag2, lag3 = recent_values[-1], recent_values[-2], recent_values[-3]
                one = pd.DataFrame([{
                    "TagName": str(tag),
                    "IndicatorType": indicator_type(tag),
                    "Lag_1": lag1,
                    "Lag_2": lag2,
                    "Lag_3": lag3,
                    "Rolling_Mean_3": float(np.mean([lag1, lag2, lag3])),
                    "Recent_Change": lag1 - lag2,
                }])
                xrow = design_matrix(transform_features(one, full_stats), columns=X_columns)
                delta = float(fitted_model.predict(xrow)[0]) * scale
                prediction = last + delta
            recent_values.append(float(prediction))
            if pd.notna(forecast_time_origin):
                future_time = forecast_time_origin + pd.to_timedelta(seconds_step * step, unit="s")
                time_label = future_time
            else:
                future_time = pd.NaT
                time_label = None
            forecast_records.append({
                "TagName": tag,
                "IndicatorType": indicator_type(tag),
                "Forecast_Step": step,
                "Timestamp": time_label,
                "Reading_Order": int(group["Reading_Order"].max()) + step,
                "Last_Observed_Reading": float(group["Reading"].iloc[-1]),
                "Predicted_Reading": float(prediction),
                "Model": model_name,
            })
    return pd.DataFrame(forecast_records), fitted_model


def metric_summary(data: pd.DataFrame):
    rows = []
    for tag, group in data.groupby("TagName"):
        values = group["Reading"].astype(float)
        rows.append({
            "Indicator": tag,
            "Readings": len(values),
            "Mean": values.mean(),
            "Std Dev": values.std(ddof=0),
            "Minimum": values.min(),
            "Maximum": values.max(),
            "Latest Reading": group.sort_values("Reading_Order")["Reading"].iloc[-1],
        })
    return pd.DataFrame(rows)


def display_metric_row(metrics):
    cols = st.columns(len(metrics))
    for col, (label, value) in zip(cols, metrics):
        col.metric(label, value)


st.title("Universal Sensor Forecasting")
st.caption("Upload industrial or time-series readings, inspect patterns, compare pooled models, and forecast future readings.")

with st.sidebar:
    st.header("1. Upload data")
    uploaded = st.file_uploader(
        "Upload CSV or Excel",
        type=["csv", "xlsx", "xls"],
        help="The app reads the uploaded file in memory; it does not provide data-export features.",
    )

if uploaded is None:
    st.info("Upload a CSV or Excel file to begin. The input should contain an indicator name, a numeric reading, and preferably a timestamp.")
    st.subheader("Accepted input examples")
    st.code("TagName,Timestamp,Reading\n200PI0802/AI1/PV,2022-10-01 13:29:00,0.2371\n200PI0802/AI1/PV,2022-10-01 13:29:05,0.2372", language="csv")
    st.markdown("**Also supported:** `TagName / CurrentTime / CurrentValue` and other column names, using the column selectors after upload.")
    st.stop()

try:
    raw = load_uploaded_file(uploaded)
except Exception as exc:
    st.error(f"Could not read this file: {exc}")
    st.stop()

if raw.empty or len(raw.columns) == 0:
    st.error("The file has no usable rows or columns.")
    st.stop()

with st.sidebar:
    st.header("2. Map your columns")
    columns = list(raw.columns)
    indicator_default = infer_column(columns, ["TagName", "Indicator", "Sensor", "TagGroupName", "Equipment", "TagId"])
    reading_default = infer_column(columns, ["Reading", "CurrentValue", "Current Value", "Value", "Measurement"])
    time_candidates = [None] + columns
    timestamp_default = infer_column(columns, ["Timestamp", "CurrentTime", "CreatedDateTime", "Datetime", "DateTime", "Time", "Date"])
    indicator_col = st.selectbox("Indicator / sensor column", columns, index=columns.index(indicator_default))
    reading_col = st.selectbox("Numeric reading column", columns, index=columns.index(reading_default))
    time_default_index = time_candidates.index(timestamp_default) if timestamp_default in columns else 0
    timestamp_choice = st.selectbox(
        "Timestamp column (optional)",
        time_candidates,
        index=time_default_index,
        format_func=lambda x: "Use row order (no timestamp)" if x is None else str(x),
    )

try:
    data, cleaning = prepare_data(raw, indicator_col, reading_col, timestamp_choice)
except Exception as exc:
    st.error(f"Cannot prepare this dataset: {exc}")
    st.stop()

if timestamp_choice is not None and not cleaning["timestamp_usable"]:
    st.warning("The selected timestamp column could not be parsed reliably. The app will use input row order for sequencing and charts.")
elif timestamp_choice is None:
    st.caption("No timestamp selected: readings are sequenced in their original row order within each indicator.")

# A content-based signature prevents old forecasts from being shown for a different upload
# that happens to have the same filename and row count.
signature_columns = ["TagName", "Reading", "Reading_Order", "Timestamp"]
_dataset_hash = int(pd.util.hash_pandas_object(data[signature_columns], index=True).sum())
dataset_signature = (uploaded.name, len(data), _dataset_hash, indicator_col, reading_col, str(timestamp_choice))

# Cached comparison is keyed by the clean dataset and mapping. It fits one model per candidate, globally across all sensors.
@st.cache_data(show_spinner="Analysing data and comparing generalized models...")
def cached_evaluation(clean_frame: pd.DataFrame):
    return evaluate_models(clean_frame)

try:
    evaluation = cached_evaluation(data)
except Exception as exc:
    st.error(f"Model evaluation failed: {exc}")
    st.stop()

upload_tab, analysis_tab, dashboard_tab, forecast_tab = st.tabs([
    "Upload & Data", "Model Recommendation", "Dashboard", "Forecasting"
])

with upload_tab:
    st.subheader("Dataset overview")
    display_metric_row([
        ("Usable readings", f"{len(data):,}"),
        ("Indicators", f"{data['TagName'].nunique():,}"),
        ("Original columns", f"{raw.shape[1]:,}"),
        ("Rows removed", f"{cleaning['dropped_rows']:,}"),
    ])

    if cleaning["dropped_rows"]:
        st.warning(
            f"Removed {cleaning['dropped_rows']} row(s) with a missing indicator or non-numeric/missing reading. "
            f"Missing reading values found before cleaning: {cleaning['missing_reading']}."
        )
    if data["TagName"].nunique() == 1:
        st.warning("Only one indicator was found. The app still works, but a generalized multi-indicator model cannot learn shared patterns across indicators.")

    st.markdown("#### Cleaned data preview")
    preview_cols = ["TagName", "Reading", "Reading_Order", "Segment", "IndicatorType"]
    if cleaning["timestamp_usable"]:
        preview_cols.insert(1, "Timestamp")
    st.dataframe(data[preview_cols].head(100), use_container_width=True, hide_index=True)

    st.markdown("#### Reading statistics by indicator")
    st.dataframe(metric_summary(data).round(6), use_container_width=True, hide_index=True)

    with st.expander("Show original file preview"):
        st.dataframe(raw.head(50), use_container_width=True, hide_index=True)

with analysis_tab:
    st.subheader("Model recommendation")
    st.caption(
        "This tab focuses on comparing forecasting candidates and recommending a starting model. "
        "Dataset overview and descriptive statistics are available in Upload & Data."
    )

    if evaluation["results"].empty:
        st.warning(evaluation["reason"])
        st.dataframe(
            pd.DataFrame([{
                "Model": "Last-value baseline",
                "Pros": MODEL_INFO["Last-value baseline"]["pros"],
                "Cons": MODEL_INFO["Last-value baseline"]["cons"],
            }]),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.markdown("#### Candidate model comparison")
        st.dataframe(
            evaluation["results"].round(6),
            use_container_width=True,
            hide_index=True,
        )

        recommended = evaluation["recommended"]
        best_match = evaluation["results"][
            evaluation["results"]["Model"] == recommended
        ]
        st.markdown("#### Recommended starting point")
        st.success(f"Recommended model: **{recommended}**")
        st.write(evaluation["reason"])
        if not best_match.empty:
            st.write(
                f"Validation MAE: **{best_match.iloc[0]['MAE']:.6g}** · "
                f"RMSE: **{best_match.iloc[0]['RMSE']:.6g}** · "
                f"Mean normalized MAE: **{best_match.iloc[0]['Mean_Normalized_MAE']:.4f}**"
            )
        st.info(
            "Mean normalized MAE is the primary ranking metric because indicators can have different "
            "variability scales. The 2% improvement threshold is a practical heuristic, not a guarantee "
            "of future performance."
        )

    st.markdown("#### Model advantages and limitations")
    for model_name in MODEL_NAMES:
        with st.expander(
            model_name,
            expanded=(model_name == evaluation.get("recommended", "")),
        ):
            st.markdown(f"**Type:** {MODEL_INFO[model_name]['kind']}")
            st.markdown(f"**Advantages:** {MODEL_INFO[model_name]['pros']}")
            st.markdown(f"**Limitations:** {MODEL_INFO[model_name]['cons']}")

with dashboard_tab:
    st.subheader("Segment-wise actual readings")
    available_tags = sorted(data["TagName"].unique())
    dash_c1, dash_c2 = st.columns([2, 1])
    with dash_c1:
        dashboard_tag = st.selectbox("Choose indicator", available_tags, key="dashboard_indicator")
    with dash_c2:
        show_segment_bands = st.checkbox("Show segment labels and colors", value=True)

    selected = data[data["TagName"] == dashboard_tag].sort_values("Reading_Order").copy()
    if cleaning["timestamp_usable"]:
        selected["Chart_X"] = selected["Timestamp"]
        x_title = "Timestamp"
    else:
        selected["Chart_X"] = selected["Reading_Order"]
        x_title = "Reading order"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=selected["Chart_X"], y=selected["Reading"],
        mode="lines", name="Actual trend",
        line=dict(color="#777777", width=1.8),
        hovertemplate="x=%{x}<br>Reading=%{y}<extra></extra>",
    ))
    seg_colors = {"Segment 1": "#4c78a8", "Segment 2": "#f58518", "Segment 3": "#54a24b"}
    for segment in ["Segment 1", "Segment 2", "Segment 3"]:
        part = selected[selected["Segment"] == segment]
        if part.empty:
            continue
        fig.add_trace(go.Scatter(
            x=part["Chart_X"], y=part["Reading"],
            mode="markers+lines" if show_segment_bands else "markers",
            name=segment,
            line=dict(color=seg_colors[segment], width=1.5),
            marker=dict(color=seg_colors[segment], size=7),
            hovertemplate="%{x}<br>Reading=%{y}<extra>" + segment + "</extra>",
        ))
    fig.update_layout(
        title=f"Actual readings — {dashboard_tag}",
        xaxis_title=x_title, yaxis_title="Reading",
        height=500, hovermode="x unified",
        legend_title="History segment",
        margin=dict(l=20, r=20, t=60, b=30),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Segment statistics")
    segment_stats = selected.groupby("Segment", sort=False)["Reading"].agg(
        Readings="count", Mean="mean", Std_Dev="std", Minimum="min", Maximum="max"
    ).reindex(["Segment 1", "Segment 2", "Segment 3"]).reset_index()
    st.dataframe(segment_stats.round(6), use_container_width=True, hide_index=True)
    st.caption("Segments divide each indicator's readings into three sequential portions. They are descriptive time/order segments, not detected operating regimes or alarm states.")

with forecast_tab:
    st.subheader("Historical predictions and future forecasting")
    st.write(
        "The selected model is first trained on the first 40 readings of each indicator. "
        "It predicts the remaining observed readings as a historical holdout. The model is then "
        "refitted on all available history to produce future forecasts."
    )
    st.caption(
        "Historical holdout predictions use earlier actual readings as lag inputs (one-step-ahead validation). "
        "Future predictions are recursive and use previous forecasts as inputs."
    )

    candidate_names = MODEL_NAMES
    recommended = evaluation.get("recommended", "Last-value baseline")
    default_model_index = candidate_names.index(recommended) if recommended in candidate_names else 0
    fc1, fc2 = st.columns([2, 1])
    with fc1:
        selected_model = st.selectbox(
            "Model to use", candidate_names, index=default_model_index, key="forecast_model"
        )
    with fc2:
        forecast_horizon = st.number_input(
            "Future readings per indicator", min_value=1, max_value=100, value=5, step=1
        )

    st.markdown(f"**Advantages:** {MODEL_INFO[selected_model]['pros']}")
    st.markdown(f"**Limitations:** {MODEL_INFO[selected_model]['cons']}")

    history_training_count = 40
    st.caption(
        f"Historical holdout training window: first {history_training_count} readings per indicator. "
        "The future forecasting model is refitted using all available readings."
    )

    run_forecast = st.button(
        "Train selected model and forecast", type="primary", use_container_width=False
    )
    if run_forecast:
        try:
            with st.spinner("Generating historical predictions and fitting the final forecast model..."):
                historical_predictions = predict_historical_holdout(
                    data, selected_model, train_readings=history_training_count
                )
                future_df, fitted_estimator = fit_full_model_and_forecast(
                    data, selected_model, int(forecast_horizon)
                )
            st.session_state["historical_predictions_df"] = historical_predictions
            st.session_state["forecast_df"] = future_df
            st.session_state["forecast_model_name"] = selected_model
            st.session_state["forecast_dataset_signature"] = dataset_signature
            st.success(
                f"Forecast completed: {len(future_df)} future predictions across "
                f"{future_df['TagName'].nunique()} indicators."
            )
        except Exception as exc:
            st.error(f"Forecasting failed: {exc}")

    forecast_df = st.session_state.get("forecast_df")
    historical_predictions = st.session_state.get("historical_predictions_df")
    if forecast_df is not None and not forecast_df.empty:
        signature = st.session_state.get("forecast_dataset_signature")
        if signature != dataset_signature:
            st.info("The uploaded dataset may have changed. Run forecasting again to refresh these predictions.")
        else:
            st.markdown("#### Actual readings, historical predictions, and future forecast")
            fc_tags = sorted(forecast_df["TagName"].unique())
            plot_tag = st.selectbox("Indicator to visualize", fc_tags, key="forecast_plot_indicator")
            history = data[data["TagName"] == plot_tag].sort_values("Reading_Order").copy()
            future = forecast_df[forecast_df["TagName"] == plot_tag].sort_values("Forecast_Step").copy()

            if historical_predictions is None or historical_predictions.empty:
                historical = pd.DataFrame()
            else:
                historical = historical_predictions[
                    historical_predictions["TagName"] == plot_tag
                ].sort_values("Reading_Order").copy()

            plot = go.Figure()
            if cleaning["timestamp_usable"] and history["Timestamp"].notna().any():
                x_actual = history["Timestamp"]
                x_axis_label = "Timestamp"
                x_key = "Timestamp"
            else:
                x_actual = history["Reading_Order"]
                x_axis_label = "Reading order"
                x_key = "Reading_Order"

            plot.add_trace(go.Scatter(
                x=x_actual,
                y=history["Reading"],
                mode="lines+markers",
                name="Actual readings",
                line=dict(color="#4c78a8", width=2),
                marker=dict(size=5),
                hovertemplate="%{x}<br>Actual=%{y}<extra></extra>",
            ))

            if not historical.empty:
                if x_key == "Timestamp" and historical["Timestamp"].notna().any():
                    x_hist = historical["Timestamp"]
                else:
                    x_hist = historical["Reading_Order"]
                plot.add_trace(go.Scatter(
                    x=x_hist,
                    y=historical["Historical_Prediction"],
                    mode="lines+markers",
                    name="Historical predictions (trained on first 40)",
                    line=dict(color="#f58518", width=1.8, dash="dot"),
                    marker=dict(symbol="x", size=8),
                    hovertemplate="%{x}<br>Historical prediction=%{y}<extra></extra>",
                ))
            else:
                st.warning(
                    "No historical holdout predictions are available. Each indicator needs more than 40 usable "
                    "readings to show predictions after the first 40. Future forecasting is still available."
                )

            # Plot future forecasts on the timeline after the latest observed point.
            if cleaning["timestamp_usable"] and history["Timestamp"].notna().any() and future["Timestamp"].notna().any():
                x_future = future["Timestamp"]
            else:
                x_future = future["Reading_Order"]

            last_x = x_actual.iloc[-1]
            last_y = float(history["Reading"].iloc[-1])
            plot.add_trace(go.Scatter(
                x=[last_x] + list(x_future),
                y=[last_y] + future["Predicted_Reading"].tolist(),
                mode="lines+markers",
                name="Future forecast",
                line=dict(color="#54a24b", width=2, dash="dash"),
                marker=dict(symbol="diamond", size=7),
                hovertemplate="%{x}<br>Forecast=%{y}<extra></extra>",
            ))
            plot.add_vline(
                x=last_x,
                line_dash="dot",
                line_color="#777777",
                annotation_text="Future forecast starts",
            )
            plot.update_layout(
                title=f"{plot_tag}: actual, historical predictions, and future forecast",
                xaxis_title=x_axis_label,
                yaxis_title="Reading",
                height=540,
                hovermode="x unified",
                margin=dict(l=20, r=20, t=70, b=30),
                legend_title="Series",
            )
            st.plotly_chart(plot, use_container_width=True)

            if not historical.empty:
                st.markdown("#### Historical holdout predictions")
                st.caption(
                    "These predictions cover observations after the first 40 readings. The model was trained "
                    "only on the first 40 readings per indicator; the actual holdout readings are shown for comparison."
                )
                st.dataframe(
                    historical[[
                        "TagName", "Reading_Order", "Timestamp", "Actual_Reading", "Historical_Prediction"
                    ]].round(8),
                    use_container_width=True,
                    hide_index=True,
                )

            st.markdown("#### Future forecast readings")
            st.dataframe(
                future[["TagName", "Forecast_Step", "Timestamp", "Predicted_Reading", "Model"]].round(8),
                use_container_width=True,
                hide_index=True,
            )
            st.caption(
                "The model used for future forecasts is fitted on all available historical readings. "
                "Forecasts are estimates, not process alarms or safety limits."
            )

st.divider()
st.caption("Universal Sensor Forecasting · Pooled models · CSV/Excel input · No data-export functionality")
