"""
Step 4: Evaluate the LightGBM 28-day forecast.

M5 VALIDATION MODE
==================

LightGBM internal validation:
    Historical training period
    +
    2016-03-28 -> 2016-04-24

Final evaluation forecast:
    2016-04-25 -> 2016-05-22

The LightGBM model produces the final 28-day forecast for:

    2016-04-25 -> 2016-05-22

The actual sales for this period are available in:

    sales_train_evaluation.csv

Metrics:
    - MAE
    - RMSE
    - WAPE
    - Aggregate Forecast Error
    - Bias
    - RMSSE
    - WRMSSE
    - Hierarchy-level MAE
    - Hierarchy-level RMSSE
    - Hierarchy-level Bias

Outputs:
    outputs/lgb_metrics_validation.csv
    outputs/lgb_metrics_validation.json
    outputs/lgb_hierarchy_validation.parquet

Run:

    python -m src.evaluate
"""

import json
import numpy as np
import pandas as pd

from src import config as C
from src.features import to_str


# ============================================================================
# M5 HIERARCHY
# ============================================================================

LEVELS = [
    ("L1 Total", []),

    ("L2 State", [
        "state_id"
    ]),

    ("L3 Store", [
        "store_id"
    ]),

    ("L4 Category", [
        "cat_id"
    ]),

    ("L5 Department", [
        "dept_id"
    ]),

    ("L6 State-Category", [
        "state_id",
        "cat_id"
    ]),

    ("L7 State-Department", [
        "state_id",
        "dept_id"
    ]),

    ("L8 Store-Category", [
        "store_id",
        "cat_id"
    ]),

    ("L9 Store-Department", [
        "store_id",
        "dept_id"
    ]),

    ("L10 Item", [
        "item_id"
    ]),

    ("L11 Item-State", [
        "item_id",
        "state_id"
    ]),

    ("L12 Item-Store", [
        "item_id",
        "store_id"
    ]),
]


# ============================================================================
# CALENDAR HELPERS
# ============================================================================

def load_calendar() -> pd.DataFrame:
    """
    Load M5 calendar and create d_num.
    """

    calendar_file = C.RAW / "calendar.csv"

    if not calendar_file.exists():
        raise FileNotFoundError(
            f"calendar.csv was not found:\n{calendar_file}"
        )

    calendar = pd.read_csv(
        calendar_file,
        parse_dates=["date"]
    )

    calendar["d_num"] = (
        calendar["d"]
        .str.slice(2)
        .astype(int)
    )

    return calendar


def day_to_date(
    calendar: pd.DataFrame,
    day_number: int
) -> str:
    """
    Convert M5 day number into YYYY-MM-DD.
    """

    row = calendar.loc[
        calendar["d_num"] == day_number,
        "date"
    ]

    if row.empty:
        return f"Unknown date for day {day_number}"

    return str(
        row.iloc[0].date()
    )


# ============================================================================
# LOAD ACTUAL SALES
# ============================================================================

def _load_actuals(
    ids: pd.Index,
    days: list[int]
) -> pd.DataFrame:
    """
    Load actual sales for the requested item-store series and days.

    Evaluation actuals come from:

        sales_train_evaluation.csv

    If that file is unavailable, validation data is used as fallback.
    """

    evaluation_file = (
        C.RAW /
        "sales_train_evaluation.csv"
    )

    validation_file = (
        C.RAW /
        "sales_train_validation.csv"
    )

    # ------------------------------------------------------------------------
    # Select actual-sales file
    # ------------------------------------------------------------------------

    if evaluation_file.exists():

        f = evaluation_file

    elif validation_file.exists():

        f = validation_file

    else:

        raise FileNotFoundError(
            "\nCould not find:\n"
            "sales_train_evaluation.csv\n"
            "or\n"
            "sales_train_validation.csv\n"
            "inside data/raw/"
        )

    print(
        f"loading actual sales from: {f.name}"
    )

    df = pd.read_csv(
        f
    )

    # ------------------------------------------------------------------------
    # Normalize IDs
    # ------------------------------------------------------------------------

    df["id"] = (
        df["id"]
        .str.replace(
            "_evaluation$",
            "",
            regex=True
        )
        .str.replace(
            "_validation$",
            "",
            regex=True
        )
    )

    df = df.set_index(
        "id"
    )

    # ------------------------------------------------------------------------
    # Requested day columns
    # ------------------------------------------------------------------------

    requested_columns = [
        f"d_{d}"
        for d in days
    ]

    available_columns = [
        c
        for c in requested_columns
        if c in df.columns
    ]

    if not available_columns:

        return pd.DataFrame(
            index=df.index.intersection(ids)
        )

    result = df.loc[
        df.index.intersection(ids),
        available_columns
    ]

    return result


# ============================================================================
# OVERALL METRICS
# ============================================================================

def calculate_overall_metrics(
    actual: pd.DataFrame,
    predicted: pd.DataFrame
) -> dict:
    """
    Calculate bottom-level forecast metrics.
    """

    # ------------------------------------------------------------------------
    # Common IDs
    # ------------------------------------------------------------------------

    common_ids = actual.index.intersection(
        predicted.index
    )

    # ------------------------------------------------------------------------
    # Common days
    # ------------------------------------------------------------------------

    common_days = actual.columns.intersection(
        predicted.columns
    )

    actual = actual.loc[
        common_ids,
        common_days
    ]

    predicted = predicted.loc[
        common_ids,
        common_days
    ]

    if actual.empty:

        raise ValueError(
            "No common series were found "
            "between actual and predicted data."
        )

    if len(common_days) == 0:

        raise ValueError(
            "No common forecast days were found "
            "between actual and predicted data."
        )

    # ------------------------------------------------------------------------
    # Arrays
    # ------------------------------------------------------------------------

    y_true = actual.to_numpy(
        dtype="float64"
    )

    y_pred = predicted.to_numpy(
        dtype="float64"
    )

    # ------------------------------------------------------------------------
    # Error
    # ------------------------------------------------------------------------

    error = (
        y_pred -
        y_true
    )

    abs_error = np.abs(
        error
    )

    # ------------------------------------------------------------------------
    # MAE
    # ------------------------------------------------------------------------

    mae = float(
        abs_error.mean()
    )

    # ------------------------------------------------------------------------
    # RMSE
    # ------------------------------------------------------------------------

    rmse = float(
        np.sqrt(
            np.mean(
                error ** 2
            )
        )
    )

    # ------------------------------------------------------------------------
    # Totals
    # ------------------------------------------------------------------------

    actual_total = float(
        y_true.sum()
    )

    forecast_total = float(
        y_pred.sum()
    )

    absolute_error_total = float(
        abs_error.sum()
    )

    # ------------------------------------------------------------------------
    # WAPE
    # ------------------------------------------------------------------------

    if actual_total > 0:

        wape = (
            absolute_error_total
            /
            actual_total
            *
            100.0
        )

    else:

        wape = np.nan

    # ------------------------------------------------------------------------
    # Aggregate Forecast Error
    # ------------------------------------------------------------------------

    if actual_total > 0:

        aggregate_error = (
            abs(
                forecast_total -
                actual_total
            )
            /
            actual_total
            *
            100.0
        )

    else:

        aggregate_error = np.nan

    # ------------------------------------------------------------------------
    # Bias
    # ------------------------------------------------------------------------

    if actual_total > 0:

        bias = (
            (
                forecast_total /
                actual_total
            )
            -
            1.0
        ) * 100.0

    else:

        bias = np.nan

    return {
        "mae": mae,

        "rmse": rmse,

        "wape_pct": float(
            wape
        ),

        "aggregate_forecast_error_pct": float(
            aggregate_error
        ),

        "bias_pct": float(
            bias
        ),

        "actual_total_units": actual_total,

        "forecast_total_units": forecast_total,

        "n_series": int(
            len(common_ids)
        ),

        "n_days": int(
            len(common_days)
        ),

        "n_predictions": int(
            len(common_ids)
            *
            len(common_days)
        ),
    }


# ============================================================================
# BUILD HIERARCHY GROUP
# ============================================================================

def _make_group(
    meta: pd.DataFrame,
    keys: list[str]
) -> pd.Series:
    """
    Build hierarchy node names.

    Example:

        state_id = CA
        cat_id   = FOODS

    becomes:

        CA--FOODS
    """

    if not keys:

        return pd.Series(
            "TOTAL",
            index=meta.index
        )

    pieces = [
        to_str(
            meta[k]
        )
        for k in keys
    ]

    return pd.concat(
        pieces,
        axis=1
    ).agg(
        "--".join,
        axis=1
    )


# ============================================================================
# RMSSE SCALE
# ============================================================================

def calculate_scale(
    history: pd.DataFrame
) -> pd.Series:
    """
    Calculate RMSSE scale for each hierarchy node.

    Scale:

        mean((y_t - y_(t-1))^2)

    using historical demand.
    """

    values = history.to_numpy(
        dtype="float64"
    )

    if values.shape[1] < 2:

        return pd.Series(
            np.nan,
            index=history.index
        )

    differences = np.diff(
        values,
        axis=1
    )

    squared_difference = (
        differences ** 2
    )

    scale = np.nanmean(
        squared_difference,
        axis=1
    )

    scale = pd.Series(
        scale,
        index=history.index
    )

    # Zero scale cannot be used as denominator.
    scale = scale.replace(
        0,
        np.nan
    )

    return scale


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:

    print("=" * 75)

    print(
        "M5 LIGHTGBM FORECAST EVALUATION"
    )

    print("=" * 75)

    # =========================================================================
    # LOAD CALENDAR
    # =========================================================================

    calendar = load_calendar()

    # =========================================================================
    # LOAD LIGHTGBM FORECAST
    # =========================================================================

    forecast_file = (
        C.OUT /
        f"lgb_forecast_{C.MODE}.parquet"
    )

    if not forecast_file.exists():

        raise FileNotFoundError(
            f"\nLightGBM forecast was not found:\n"
            f"{forecast_file}\n\n"
            f"Run:\n"
            f"python -m src.train\n"
            f"first."
        )

    fc = pd.read_parquet(
        forecast_file
    )

    if fc.empty:

        raise ValueError(
            "LightGBM forecast file is empty."
        )

    # =========================================================================
    # DETERMINE FORECAST PERIOD
    # =========================================================================

    train_end, first_pred = (
        C.MODES[C.MODE]
    )

    pred_days = list(
        range(
            first_pred,
            first_pred +
            C.HORIZON
        )
    )

    prediction_end = (
        first_pred
        +
        C.HORIZON
        -
        1
    )

    # =========================================================================
    # CONVERT DAY NUMBERS TO REAL DATES
    # =========================================================================

    training_end_date = day_to_date(
        calendar,
        train_end
    )

    prediction_start_date = day_to_date(
        calendar,
        first_pred
    )

    prediction_end_date = day_to_date(
        calendar,
        prediction_end
    )

    # Internal validation is the 28 days immediately before final forecast.
    validation_start = (
        train_end
        -
        C.HORIZON
        +
        1
    )

    validation_start_date = day_to_date(
        calendar,
        validation_start
    )

    # M5 starts on January 29, 2011.
    first_historical_day = int(
        calendar["d_num"].min()
    )

    first_historical_date = day_to_date(
        calendar,
        first_historical_day
    )

    # =========================================================================
    # DISPLAY DATE WINDOWS
    # =========================================================================

    print()

    print(
        "DATE WINDOWS"
    )

    print("-" * 75)

    print(
        "Historical training data:"
    )

    print(
        f"  {first_historical_date} -> "
        f"{training_end_date}"
    )

    print()

    print(
        "Internal validation:"
    )

    print(
        f"  {validation_start_date} -> "
        f"{training_end_date}"
    )

    print()

    print(
        "FINAL 28-DAY EVALUATION FORECAST:"
    )

    print(
        f"  {prediction_start_date} -> "
        f"{prediction_end_date}"
    )

    print(
        "-" * 75
    )

    print()

    print(
        f"Mode:             {C.MODE}"
    )

    print(
        f"Forecast horizon: {C.HORIZON} days"
    )

    # =========================================================================
    # VERIFY FORECAST
    # =========================================================================

    number_of_series = int(
        fc["id"].nunique()
    )

    expected_rows = (
        number_of_series
        *
        C.HORIZON
    )

    print()

    print(
        "FORECAST CHECK"
    )

    print("-" * 75)

    print(
        f"Forecast rows:   {len(fc):,}"
    )

    print(
        f"Forecast series: {number_of_series:,}"
    )

    print(
        f"Expected rows:   {expected_rows:,}"
    )

    if len(fc) != expected_rows:

        print()

        print(
            "WARNING:"
        )

        print(
            "The forecast does not contain exactly "
            f"{C.HORIZON} rows for every series."
        )

    # =========================================================================
    # LOAD ACTUAL SALES
    # =========================================================================

    ids = pd.Index(
        fc["id"].unique()
    )

    # -------------------------------------------------------------------------
    # Historical data used for RMSSE scale.
    # -------------------------------------------------------------------------

    history_days = list(
        range(
            first_historical_day,
            train_end + 1
        )
    )

    history = _load_actuals(
        ids,
        history_days
    )

    # -------------------------------------------------------------------------
    # Actual evaluation-period sales.
    # -------------------------------------------------------------------------

    truth = _load_actuals(
        ids,
        pred_days
    )

    print()

    print(
        "ACTUAL DATA CHECK"
    )

    print("-" * 75)

    print(
        f"Historical actual days: "
        f"{history.shape[1]:,}"
    )

    print(
        f"Evaluation actual days: "
        f"{truth.shape[1]:,}/{C.HORIZON}"
    )

    # =========================================================================
    # CHECK GROUND TRUTH
    # =========================================================================

    have_truth = (
        truth.shape[1]
        ==
        C.HORIZON
    )

    if not have_truth:

        print()

        print(
            "WARNING:"
        )

        print(
            "The complete 28-day actual evaluation window "
            "is not available."
        )

        print(
            "Accuracy metrics cannot be calculated."
        )

        return

    # =========================================================================
    # CONVERT FORECAST TO WIDE FORMAT
    # =========================================================================

    pred = fc.pivot(
        index="id",
        columns="d_num",
        values="forecast"
    )

    pred.columns = [
        f"d_{int(c)}"
        for c in pred.columns
    ]

    # =========================================================================
    # OVERALL METRICS
    # =========================================================================

    overall = calculate_overall_metrics(
        truth,
        pred
    )

    print()

    print("=" * 75)

    print(
        "OVERALL BOTTOM-LEVEL METRICS"
    )

    print("=" * 75)

    print(
        f"MAE:                       "
        f"{overall['mae']:.4f}"
    )

    print(
        f"RMSE:                      "
        f"{overall['rmse']:.4f}"
    )

    print(
        f"WAPE:                      "
        f"{overall['wape_pct']:.2f}%"
    )

    print(
        f"Aggregate forecast error:  "
        f"{overall['aggregate_forecast_error_pct']:.2f}%"
    )

    print(
        f"Bias:                      "
        f"{overall['bias_pct']:.2f}%"
    )

    print(
        f"Actual total:              "
        f"{overall['actual_total_units']:,.0f}"
    )

    print(
        f"Forecast total:            "
        f"{overall['forecast_total_units']:,.0f}"
    )

    # =========================================================================
    # BUILD METADATA
    # =========================================================================

    meta = (
        fc
        .drop_duplicates(
            "id"
        )
        .set_index(
            "id"
        )
        [
            [
                "item_id",
                "dept_id",
                "cat_id",
                "store_id",
                "state_id",
            ]
        ]
    )

    meta = meta.reindex(
        history.index
    )

    # =========================================================================
    # LOAD PRICE DATA
    # =========================================================================

    prices_file = (
        C.RAW /
        "sell_prices.csv"
    )

    if not prices_file.exists():

        raise FileNotFoundError(
            f"sell_prices.csv was not found:\n"
            f"{prices_file}"
        )

    prices = pd.read_csv(
        prices_file
    )

    # =========================================================================
    # WEEK MAP
    # =========================================================================

    week_map = (
        calendar
        .set_index(
            "d_num"
        )["wm_yr_wk"]
    )

    # =========================================================================
    # REVENUE WEIGHTS
    # =========================================================================
    #
    # Use the final 28 historical days before the evaluation forecast.
    #
    # This corresponds to:
    #
    # 2016-03-28 -> 2016-04-24
    #
    # =========================================================================

    last_history_days = list(
        range(
            max(
                first_historical_day,
                train_end -
                C.HORIZON +
                1
            ),
            train_end + 1
        )
    )

    revenue = pd.Series(
        0.0,
        index=history.index
    )

    print()

    print(
        "Calculating revenue weights..."
    )

    # -------------------------------------------------------------------------
    # Create efficient price lookup.
    # -------------------------------------------------------------------------

    price_map = (
        prices
        .set_index(
            [
                "store_id",
                "item_id",
                "wm_yr_wk"
            ]
        )["sell_price"]
    )

    # -------------------------------------------------------------------------
    # Calculate historical revenue.
    # -------------------------------------------------------------------------

    for d_num in last_history_days:

        day_column = (
            f"d_{d_num}"
        )

        if day_column not in history.columns:
            continue

        if d_num not in week_map.index:
            continue

        week = week_map.loc[
            d_num
        ]

        price_index = pd.MultiIndex.from_arrays(
            [
                meta["store_id"].to_numpy(),
                meta["item_id"].to_numpy(),
                np.full(
                    len(meta),
                    week
                )
            ]
        )

        prices_for_day = (
            price_map
            .reindex(
                price_index
            )
            .to_numpy(
                dtype="float64"
            )
        )

        sales_for_day = (
            history[day_column]
            .reindex(
                meta.index
            )
            .to_numpy(
                dtype="float64"
            )
        )

        revenue += np.nan_to_num(
            sales_for_day
            *
            prices_for_day,
            nan=0.0
        )

    # Free price lookup memory.
    del price_map
    del prices

    # =========================================================================
    # HIERARCHY EVALUATION
    # =========================================================================

    hierarchy_rows = []

    for level_name, keys in LEVELS:

        print()

        print(
            f"processing {level_name} ..."
        )

        # ---------------------------------------------------------------------
        # Build hierarchy group.
        # ---------------------------------------------------------------------

        history_group = _make_group(
            meta,
            keys
        )

        # ---------------------------------------------------------------------
        # Aggregate historical sales.
        # ---------------------------------------------------------------------

        historical_agg = (
            history
            .groupby(
                history_group
            )
            .sum()
        )

        # ---------------------------------------------------------------------
        # Aggregate actual evaluation sales.
        # ---------------------------------------------------------------------

        truth_group = history_group.reindex(
            truth.index
        )

        truth_agg = (
            truth
            .groupby(
                truth_group
            )
            .sum()
        )

        # ---------------------------------------------------------------------
        # Aggregate predictions.
        # ---------------------------------------------------------------------

        prediction_group = history_group.reindex(
            pred.index
        )

        pred_agg = (
            pred
            .groupby(
                prediction_group
            )
            .sum()
        )

        # ---------------------------------------------------------------------
        # Revenue weights.
        # ---------------------------------------------------------------------

        revenue_group = (
            revenue
            .groupby(
                history_group
            )
            .sum()
        )

        if revenue_group.sum() > 0:

            weights = (
                revenue_group
                /
                revenue_group.sum()
            )

        else:

            weights = revenue_group

        # ---------------------------------------------------------------------
        # RMSSE scale.
        # ---------------------------------------------------------------------

        scale = calculate_scale(
            historical_agg
        )

        # ---------------------------------------------------------------------
        # Align hierarchy nodes.
        # ---------------------------------------------------------------------

        common_nodes = (
            truth_agg.index
            .intersection(
                pred_agg.index
            )
        )

        truth_agg = truth_agg.loc[
            common_nodes
        ]

        pred_agg = pred_agg.loc[
            common_nodes
        ]

        scale = scale.reindex(
            common_nodes
        )

        weights = weights.reindex(
            common_nodes
        )

        # ---------------------------------------------------------------------
        # Arrays.
        # ---------------------------------------------------------------------

        y_true = truth_agg.to_numpy(
            dtype="float64"
        )

        y_pred = pred_agg.to_numpy(
            dtype="float64"
        )

        errors = (
            y_pred -
            y_true
        )

        abs_errors = np.abs(
            errors
        )

        # ---------------------------------------------------------------------
        # MAE.
        # ---------------------------------------------------------------------

        level_mae = float(
            abs_errors.mean()
        )

        # ---------------------------------------------------------------------
        # RMSSE.
        # ---------------------------------------------------------------------

        mse_per_series = (
            errors ** 2
        ).mean(
            axis=1
        )

        scale_values = scale.to_numpy(
            dtype="float64"
        )

        rmsse_per_series = np.sqrt(
            np.divide(
                mse_per_series,
                scale_values,
                out=np.full(
                    len(mse_per_series),
                    np.nan,
                    dtype="float64"
                ),
                where=(
                    np.isfinite(
                        scale_values
                    )
                    &
                    (
                        scale_values > 0
                    )
                )
            )
        )

        # ---------------------------------------------------------------------
        # Weighted RMSSE.
        # ---------------------------------------------------------------------

        weight_values = weights.to_numpy(
            dtype="float64"
        )

        valid = (
            np.isfinite(
                rmsse_per_series
            )
            &
            np.isfinite(
                weight_values
            )
            &
            (
                weight_values >= 0
            )
        )

        if (
            valid.any()
            and
            weight_values[valid].sum() > 0
        ):

            weighted_rmsse = float(
                np.average(
                    rmsse_per_series[valid],
                    weights=weight_values[valid]
                )
            )

        else:

            weighted_rmsse = np.nan

        # ---------------------------------------------------------------------
        # Level totals and bias.
        # ---------------------------------------------------------------------

        actual_total = float(
            y_true.sum()
        )

        forecast_total = float(
            y_pred.sum()
        )

        if actual_total > 0:

            bias_pct = (
                (
                    forecast_total /
                    actual_total
                )
                -
                1.0
            ) * 100.0

        else:

            bias_pct = np.nan

        # ---------------------------------------------------------------------
        # Store results.
        # ---------------------------------------------------------------------

        hierarchy_rows.append(
            {
                "level": level_name,

                "n_series": int(
                    len(common_nodes)
                ),

                "rmsse": weighted_rmsse,

                "mae": level_mae,

                "bias_pct": float(
                    bias_pct
                ),

                "actual_total_units": actual_total,

                "forecast_total_units": forecast_total,
            }
        )

    # =========================================================================
    # HIERARCHY RESULTS
    # =========================================================================

    hierarchy_metrics = pd.DataFrame(
        hierarchy_rows
    )

    # =========================================================================
    # FINAL WRMSSE
    # =========================================================================

    valid_wrmsse = (
        hierarchy_metrics[
            "rmsse"
        ]
        .dropna()
    )

    if len(valid_wrmsse) > 0:

        wrmsse = float(
            valid_wrmsse.mean()
        )

    else:

        wrmsse = np.nan

    # =========================================================================
    # PRINT HIERARCHY RESULTS
    # =========================================================================

    print()

    print("=" * 75)

    print(
        "WRMSSE BY HIERARCHY LEVEL"
    )

    print("=" * 75)

    print(
        hierarchy_metrics.to_string(
            index=False
        )
    )

    print()

    print(
        f"FINAL WRMSSE: "
        f"{wrmsse:.6f}"
    )

    # =========================================================================
    # SAVE METRICS CSV
    # =========================================================================

    metrics_csv = (
        C.OUT /
        f"lgb_metrics_{C.MODE}.csv"
    )

    hierarchy_metrics.to_csv(
        metrics_csv,
        index=False
    )

    print()

    print(
        "saved hierarchy metrics:"
    )

    print(
        metrics_csv
    )

    # =========================================================================
    # SAVE METRICS JSON
    # =========================================================================

    metrics_json = (
        C.OUT /
        f"lgb_metrics_{C.MODE}.json"
    )

    complete_metrics = {

        "model": "LightGBM",

        "mode": C.MODE,

        "training_start_date":
            first_historical_date,

        "training_end_date":
            training_end_date,

        "internal_validation_start_date":
            validation_start_date,

        "internal_validation_end_date":
            training_end_date,

        "evaluation_start_date":
            prediction_start_date,

        "evaluation_end_date":
            prediction_end_date,

        "horizon": int(
            C.HORIZON
        ),

        "number_of_series":
            number_of_series,

        "overall":
            overall,

        "wrmsse":
            wrmsse,

        "hierarchy_levels":
            hierarchy_metrics.to_dict(
                orient="records"
            ),
    }

    metrics_json.write_text(
        json.dumps(
            complete_metrics,
            indent=2
        )
    )

    print()

    print(
        "saved complete metrics:"
    )

    print(
        metrics_json
    )

    # =========================================================================
    # BUILD HIERARCHY FORECAST TABLE
    # =========================================================================

    print()

    print(
        "building hierarchy forecast..."
    )

    hierarchy_forecasts = []

    # Metadata indexed by ID.
    forecast_meta = (
        fc
        .drop_duplicates(
            "id"
        )
        .set_index(
            "id"
        )
        [
            [
                "item_id",
                "dept_id",
                "cat_id",
                "store_id",
                "state_id",
            ]
        ]
    )

    for level_name, keys in LEVELS:

        group = _make_group(
            forecast_meta,
            keys
        )

        id_to_group = pd.Series(
            group.to_numpy(),
            index=group.index
        )

        temp = fc.copy()

        temp["node"] = (
            temp["id"]
            .map(
                id_to_group
            )
        )

        grouped = (
            temp
            .groupby(
                [
                    "date",
                    "node"
                ],
                observed=True
            )["forecast"]
            .sum()
            .reset_index()
        )

        grouped["level"] = (
            level_name
        )

        hierarchy_forecasts.append(
            grouped[
                [
                    "level",
                    "node",
                    "date",
                    "forecast"
                ]
            ]
        )

    hierarchy_forecast = pd.concat(
        hierarchy_forecasts,
        ignore_index=True
    )

    hierarchy_file = (
        C.OUT /
        f"lgb_hierarchy_{C.MODE}.parquet"
    )

    hierarchy_forecast.to_parquet(
        hierarchy_file,
        index=False
    )

    print()

    print(
        "saved hierarchy forecast:"
    )

    print(
        hierarchy_file
    )

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================

    print()

    print("=" * 75)

    print(
        "LIGHTGBM EVALUATION COMPLETE"
    )

    print("=" * 75)

    print()

    print(
        "INTERNAL VALIDATION:"
    )

    print(
        f"{validation_start_date} -> "
        f"{training_end_date}"
    )

    print()

    print(
        "FINAL 28-DAY FORECAST EVALUATION:"
    )

    print(
        f"{prediction_start_date} -> "
        f"{prediction_end_date}"
    )

    print()

    print(
        f"WRMSSE: "
        f"{wrmsse:.6f}"
    )

    print(
        f"RMSE: "
        f"{overall['rmse']:.4f}"
    )

    print(
        f"WAPE: "
        f"{overall['wape_pct']:.2f}%"
    )

    print(
        f"MAE: "
        f"{overall['mae']:.4f}"
    )

    print(
        f"Bias: "
        f"{overall['bias_pct']:.2f}%"
    )

    print()

    print(
        "The LightGBM model has been evaluated "
        "against the actual sales for the final "
        "28-day forecast period."
    )

    print()

    print(
        "FINAL FORECAST PERIOD:"
    )

    print(
        f"{prediction_start_date} -> "
        f"{prediction_end_date}"
    )


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()