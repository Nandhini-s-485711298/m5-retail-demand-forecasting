"""
Step 3: Train LightGBM and produce a 28-day forecast.

Memory-optimized version.

Pipeline:

    Full historical data
            |
            v
    Internal validation
            |
            v
    Find best boosting iteration
            |
            v
    Retrain on ALL known historical data
            |
            v
    Forecast next 28 days

For MODE="validation":

    Full historical training:
        January 29, 2011 -> March 27, 2016

    Internal validation:
        March 28, 2016 -> April 24, 2016

    Final evaluation forecast:
        April 25, 2016 -> May 22, 2016

Important:
    We do NOT create a feature matrix for all 22.6M rows.

    Only the rows required for:
        - training
        - validation
        - final prediction

    are materialized.

Run:

    python -m src.train
"""

import gc
import json

import numpy as np
import pandas as pd
import lightgbm as lgb

from src import config as C
from src.features import feature_columns


# ============================================================================
# SETTINGS
# ============================================================================

N_THREADS = 2

MAX_BIN = 63

LEARNING_RATE = 0.05

NUM_BOOST_ROUND = 1500

EARLY_STOPPING = 80

NUM_LEAVES = 63

MAX_DEPTH = -1

MIN_DATA_IN_LEAF = 100

FEATURE_FRACTION = 0.8

BAGGING_FRACTION = 0.8

BAGGING_FREQ = 1

LGB_METRIC = "rmse"

RANDOM_SEED = 42

PREDICTION_BATCH_SIZE = 100_000


# ============================================================================
# FEATURE CONVERSION
# ============================================================================

def convert_features(
    df: pd.DataFrame,
    feats: list[str],
) -> np.ndarray:
    """
    Convert a dataframe containing only the required rows into
    a float32 NumPy matrix.

    Categorical columns are converted to integer category codes.

    This function is intentionally applied only to the required
    subset instead of the entire 22.6M-row dataset.
    """

    n_rows = len(df)

    n_features = len(feats)

    print(
        f"converting "
        f"{n_rows:,} rows x "
        f"{n_features} features ..."
    )

    X = np.empty(
        (
            n_rows,
            n_features
        ),
        dtype=np.float32
    )

    for j, c in enumerate(feats):

        col = df[c]

        # ---------------------------------------------------------------
        # Categorical columns
        # ---------------------------------------------------------------

        if isinstance(
            col.dtype,
            pd.CategoricalDtype
        ):

            values = (
                col.cat.codes
                .to_numpy(
                    dtype=np.float32
                )
            )

        # ---------------------------------------------------------------
        # Numeric columns
        # ---------------------------------------------------------------

        else:

            values = (
                pd.to_numeric(
                    col,
                    errors="coerce"
                )
                .to_numpy(
                    dtype=np.float32
                )
            )

        X[:, j] = values

        del values

        if (
            j % 5 == 0
            or j == n_features - 1
        ):

            print(
                f"  feature "
                f"{j + 1}/{n_features}: "
                f"{c}"
            )

    return X


# ============================================================================
# DATE LOOKUP
# ============================================================================

def build_date_lookup(
    df: pd.DataFrame
) -> pd.Series:

    return (
        df[
            [
                "d_num",
                "date"
            ]
        ]
        .drop_duplicates(
            "d_num"
        )
        .set_index(
            "d_num"
        )["date"]
    )


def date_text(
    date_lookup: pd.Series,
    day: int
) -> str:

    if day in date_lookup.index:

        return str(
            pd.Timestamp(
                date_lookup.loc[day]
            ).date()
        )

    return "unknown"


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:

    print("=" * 75)
    print("M5 LIGHTGBM TRAINING")
    print("=" * 75)

    print(
        f"LightGBM version: "
        f"{lgb.__version__}"
    )

    print(
        f"MODE:            "
        f"{C.MODE}"
    )

    print(
        f"MAX_BIN:         "
        f"{MAX_BIN}"
    )

    print(
        f"N_THREADS:       "
        f"{N_THREADS}"
    )

    print(
        f"NUM_LEAVES:      "
        f"{NUM_LEAVES}"
    )

    print(
        f"LEARNING_RATE:   "
        f"{LEARNING_RATE}"
    )


    # ========================================================================
    # LOAD FEATURES
    # ========================================================================

    src = (
        C.PROC
        /
        f"features_{C.MODE}.parquet"
    )

    if not src.exists():

        raise FileNotFoundError(
            f"\nFeature file not found:\n"
            f"{src}\n\n"
            f"Run:\n"
            f"python -m src.features"
        )

    print()
    print(
        f"reading {src}"
    )

    df = pd.read_parquet(
        src
    )

    print(
        f"loaded "
        f"{len(df):,} rows"
    )


    # ========================================================================
    # DETERMINE FORECAST DATES
    # ========================================================================

    cfg_train_end, first_pred = (
        C.MODES[C.MODE]
    )

    actual_days = df.loc[
        df["sales"].notna(),
        "d_num"
    ]

    if actual_days.empty:

        raise RuntimeError(
            "No historical sales values found."
        )

    train_end = min(
        cfg_train_end,
        int(
            actual_days.max()
        )
    )

    prediction_end = (
        first_pred
        +
        C.HORIZON
        -
        1
    )

    validation_start = (
        train_end
        -
        C.HORIZON
        +
        1
    )


    # ========================================================================
    # DATE LOOKUP
    # ========================================================================

    date_lookup = build_date_lookup(
        df
    )


    # ========================================================================
    # DATE WINDOWS
    # ========================================================================

    print()
    print("=" * 75)
    print("DATE WINDOWS")
    print("=" * 75)

    print()
    print(
        "FULL HISTORICAL TRAINING:"
    )

    print(
        f"  {date_text(date_lookup, 1)}"
        f" -> "
        f"{date_text(date_lookup, validation_start - 1)}"
    )

    print()
    print(
        "INTERNAL VALIDATION:"
    )

    print(
        f"  {date_text(date_lookup, validation_start)}"
        f" -> "
        f"{date_text(date_lookup, train_end)}"
    )

    print()
    print(
        "FINAL 28-DAY FORECAST:"
    )

    print(
        f"  {date_text(date_lookup, first_pred)}"
        f" -> "
        f"{date_text(date_lookup, prediction_end)}"
    )


    # ========================================================================
    # FEATURES
    # ========================================================================

    feats = feature_columns(
        df
    )

    print()
    print(
        f"number of features = "
        f"{len(feats)}"
    )

    print(
        "features:"
    )

    print(
        feats
    )


    # ========================================================================
    # TIME SPLIT
    # ========================================================================

    d = (
        df["d_num"]
        .to_numpy()
    )

    train_mask = (
        d < validation_start
    )

    validation_mask = (
        (d >= validation_start)
        &
        (d <= train_end)
    )

    prediction_mask = (
        (d >= first_pred)
        &
        (d <= prediction_end)
    )


    train_rows = np.flatnonzero(
        train_mask
    )

    validation_rows = np.flatnonzero(
        validation_mask
    )

    prediction_rows = np.flatnonzero(
        prediction_mask
    )


    n_train = len(
        train_rows
    )

    n_validation = len(
        validation_rows
    )

    n_prediction = len(
        prediction_rows
    )


    print()
    print("=" * 75)
    print("TIME SPLIT")
    print("=" * 75)

    print(
        f"training rows:      "
        f"{n_train:,}"
    )

    print(
        f"validation rows:    "
        f"{n_validation:,}"
    )

    print(
        f"prediction rows:    "
        f"{n_prediction:,}"
    )


    # ========================================================================
    # SAFETY CHECKS
    # ========================================================================

    if n_train == 0:

        raise RuntimeError(
            "Training set is empty."
        )

    if n_validation == 0:

        raise RuntimeError(
            "Validation set is empty."
        )

    if n_prediction == 0:

        raise RuntimeError(
            "Prediction set is empty."
        )


    # ========================================================================
    # FUTURE SALES LEAKAGE CHECK
    # ========================================================================

    prediction_sales = df.loc[
        prediction_mask,
        "sales"
    ]

    if prediction_sales.notna().any():

        raise RuntimeError(
            "Future-sales leakage detected. "
            "Prediction rows contain actual sales."
        )

    print()
    print(
        "Future-sales leakage check: PASSED"
    )


    # ========================================================================
    # TARGET
    # ========================================================================

    y = (
        df["sales"]
        .to_numpy(
            dtype=np.float32
        )
    )


    # ========================================================================
    # PREDICTION METADATA
    # ========================================================================

    # Keep this BEFORE deleting df.

    keep = df.loc[
        prediction_mask,
        [
            "id",
            "item_id",
            "dept_id",
            "cat_id",
            "store_id",
            "state_id",
            "d_num",
            "date",
        ]
    ].copy()


    # ========================================================================
    # LIGHTGBM PARAMETERS
    # ========================================================================

    params = {

        "objective":
            "regression",

        "metric":
            LGB_METRIC,

        "boosting_type":
            "gbdt",

        "learning_rate":
            LEARNING_RATE,

        "num_leaves":
            NUM_LEAVES,

        "max_depth":
            MAX_DEPTH,

        "min_data_in_leaf":
            MIN_DATA_IN_LEAF,

        "feature_fraction":
            FEATURE_FRACTION,

        "bagging_fraction":
            BAGGING_FRACTION,

        "bagging_freq":
            BAGGING_FREQ,

        "max_bin":
            MAX_BIN,

        "verbosity":
            -1,

        "num_threads":
            N_THREADS,

        "force_col_wise":
            True,

        "seed":
            RANDOM_SEED,

        "feature_fraction_seed":
            RANDOM_SEED,

        "bagging_seed":
            RANDOM_SEED,

        "data_random_seed":
            RANDOM_SEED,
    }


    print()
    print("=" * 75)
    print("LIGHTGBM SETTINGS")
    print("=" * 75)

    print(
        json.dumps(
            params,
            indent=2
        )
    )


    # ========================================================================
    # INTERNAL VALIDATION
    # ========================================================================

    print()
    print("=" * 75)
    print("CREATING INTERNAL VALIDATION DATA")
    print("=" * 75)


    # ------------------------------------------------------------------------
    # Training matrix
    # ------------------------------------------------------------------------

    print()
    print(
        "creating training matrix ..."
    )

    train_df = df.iloc[
        train_rows
    ]

    X_train = convert_features(
        train_df,
        feats
    )

    y_train = y[
        train_rows
    ]

    print(
        f"training matrix: "
        f"{X_train.shape}"
    )

    del train_df

    gc.collect()


    # ------------------------------------------------------------------------
    # Validation matrix
    # ------------------------------------------------------------------------

    print()
    print(
        "creating validation matrix ..."
    )

    valid_df = df.iloc[
        validation_rows
    ]

    X_valid = convert_features(
        valid_df,
        feats
    )

    y_valid = y[
        validation_rows
    ]

    print(
        f"validation matrix: "
        f"{X_valid.shape}"
    )

    del valid_df

    gc.collect()


    # ========================================================================
    # BUILD LIGHTGBM DATASETS
    # ========================================================================

    print()
    print("=" * 75)
    print("BUILDING LIGHTGBM DATASETS")
    print("=" * 75)


    train_data = lgb.Dataset(
        X_train,
        label=y_train,
        feature_name=feats,
        free_raw_data=True,
    )


    valid_data = lgb.Dataset(
        X_valid,
        label=y_valid,
        feature_name=feats,
        reference=train_data,
        free_raw_data=True,
    )


    # ========================================================================
    # INTERNAL TRAINING
    # ========================================================================

    print()
    print("=" * 75)
    print("LIGHTGBM INTERNAL TRAINING")
    print("=" * 75)

    print(
        "Training on the full historical period "
        "before the internal 28-day validation window..."
    )


    callbacks = [

        lgb.early_stopping(
            EARLY_STOPPING,
            verbose=True
        ),

        lgb.log_evaluation(
            50
        ),
    ]


    validation_model = lgb.train(

        params,

        train_data,

        num_boost_round=
            NUM_BOOST_ROUND,

        valid_sets=[
            valid_data
        ],

        valid_names=[
            "validation"
        ],

        callbacks=callbacks,
    )


    best_iteration = (
        validation_model.best_iteration
    )

    best_score = (
        validation_model
        .best_score[
            "validation"
        ][
            LGB_METRIC
        ]
    )


    print()
    print(
        "Internal validation complete."
    )

    print(
        f"Best iteration: "
        f"{best_iteration}"
    )

    print(
        f"Best validation RMSE: "
        f"{best_score:.6f}"
    )


    # ========================================================================
    # FEATURE IMPORTANCE
    # ========================================================================

    importance = pd.DataFrame(
        {
            "feature":
                feats,

            "gain":
                validation_model.feature_importance(
                    importance_type="gain"
                ),

            "split":
                validation_model.feature_importance(
                    importance_type="split"
                ),
        }
    )


    importance = (
        importance
        .sort_values(
            "gain",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )


    importance_path = (
        C.OUT
        /
        f"lgb_feature_importance_{C.MODE}.csv"
    )


    importance.to_csv(
        importance_path,
        index=False
    )


    print()
    print("=" * 75)
    print("TOP 15 FEATURES")
    print("=" * 75)

    print(
        importance
        .head(15)
        .to_string(
            index=False
        )
    )


    # ========================================================================
    # RELEASE INTERNAL TRAINING MEMORY
    # ========================================================================

    del (
        train_data,
        valid_data,
        validation_model,
        X_train,
        X_valid,
        y_train,
        y_valid,
    )

    gc.collect()


    # ========================================================================
    # FINAL FULL-HISTORY TRAINING
    # ========================================================================

    print()
    print("=" * 75)
    print("FINAL FULL-HISTORY TRAINING")
    print("=" * 75)

    print(
        "Retraining using ALL known historical sales "
        "before the final forecast period."
    )


    # ------------------------------------------------------------------------
    # Identify ALL historical rows
    # ------------------------------------------------------------------------

    historical_mask = (
        df["d_num"].to_numpy()
        <= train_end
    )


    final_train_rows = np.flatnonzero(
        historical_mask
    )


    del historical_mask

    gc.collect()


    print(
        f"final training rows: "
        f"{len(final_train_rows):,}"
    )


    # ========================================================================
    # CREATE FINAL FULL-HISTORY MATRIX
    # ========================================================================

    print()
    print(
        "creating final full-history matrix ..."
    )


    final_df = df.iloc[
        final_train_rows
    ]


    X_final = convert_features(
        final_df,
        feats
    )


    y_final = y[
        final_train_rows
    ]


    print(
        f"final matrix: "
        f"{X_final.shape}"
    )


    del final_df

    gc.collect()


    # ========================================================================
    # BUILD FINAL DATASET
    # ========================================================================

    final_data = lgb.Dataset(
        X_final,
        label=y_final,
        feature_name=feats,
        free_raw_data=True,
    )


    # We no longer need these NumPy arrays directly.

    del (
        X_final,
        y_final,
        final_train_rows,
    )

    gc.collect()


    # ========================================================================
    # FINAL MODEL
    # ========================================================================

    print()
    print(
        "Training final LightGBM model..."
    )

    final_model = lgb.train(

        params,

        final_data,

        num_boost_round=
            best_iteration,

        valid_sets=None,

        callbacks=[
            lgb.log_evaluation(
                100
            )
        ],
    )


    # ========================================================================
    # SAVE MODEL
    # ========================================================================

    model_path = (
        C.OUT
        /
        f"lgb_model_{C.MODE}.txt"
    )


    final_model.save_model(
        str(model_path)
    )


    print()
    print(
        "saved model:"
    )

    print(
        model_path
    )


    # ========================================================================
    # PREDICTION
    # ========================================================================

    print()
    print("=" * 75)
    print("PREDICTING 28 DAYS")
    print("=" * 75)


    print(
        "loading prediction rows..."
    )


    prediction_df = pd.read_parquet(
        src,
        filters=[
            (
                "d_num",
                ">=",
                first_pred
            ),
            (
                "d_num",
                "<=",
                prediction_end
            ),
        ]
    )


    # Make sure ordering is deterministic.

    prediction_df = (
        prediction_df
        .sort_values(
            [
                "id",
                "d_num"
            ],
            kind="stable"
        )
        .reset_index(
            drop=True
        )
    )


    print(
        f"prediction rows loaded: "
        f"{len(prediction_df):,}"
    )


    if len(prediction_df) != n_prediction:

        raise RuntimeError(
            "Prediction row count changed "
            "after reloading."
        )


    # ========================================================================
    # PREDICT IN BATCHES
    # ========================================================================

    predictions = []


    for start in range(
        0,
        len(prediction_df),
        PREDICTION_BATCH_SIZE
    ):

        end = min(
            start +
            PREDICTION_BATCH_SIZE,
            len(prediction_df)
        )


        batch = prediction_df.iloc[
            start:end
        ]


        X_pred = convert_features(
            batch,
            feats
        )


        pred_chunk = final_model.predict(
            X_pred,
            num_iteration=best_iteration
        )


        predictions.append(
            np.asarray(
                pred_chunk,
                dtype=np.float32
            )
        )


        del (
            batch,
            X_pred,
            pred_chunk
        )

        gc.collect()


        print(
            f"  predicted "
            f"{end:,}/"
            f"{len(prediction_df):,}"
        )


    pred = np.concatenate(
        predictions
    )


    # Demand cannot be negative.

    pred = np.clip(
        pred,
        0,
        None
    ).astype(
        np.float32
    )


    # ========================================================================
    # FORECAST OUTPUT
    # ========================================================================

    out = prediction_df[
        [
            "id",
            "item_id",
            "dept_id",
            "cat_id",
            "store_id",
            "state_id",
            "d_num",
            "date",
        ]
    ].copy()


    out["forecast"] = pred


    out["horizon"] = (
        out["d_num"]
        -
        first_pred
        +
        1
    ).astype(
        "int8"
    )


    forecast_path = (
        C.OUT
        /
        f"lgb_forecast_{C.MODE}.parquet"
    )


    out.to_parquet(
        forecast_path,
        index=False
    )


    # ========================================================================
    # F1 -> F28 SUBMISSION
    # ========================================================================

    wide = out.pivot(
        index="id",
        columns="horizon",
        values="forecast"
    )


    for h in range(
        1,
        C.HORIZON + 1
    ):

        if h not in wide.columns:

            wide[h] = np.nan


    wide = wide[
        list(
            range(
                1,
                C.HORIZON + 1
            )
        )
    ]


    wide.columns = [
        f"F{int(c)}"
        for c in wide.columns
    ]


    wide = wide.reset_index()


    submission_path = (
        C.OUT
        /
        f"lgb_submission_{C.MODE}.csv"
    )


    wide.to_csv(
        submission_path,
        index=False
    )


    # ========================================================================
    # METADATA
    # ========================================================================

    first_date = (
        pd.to_datetime(
            out["date"]
        ).min()
    )


    last_date = (
        pd.to_datetime(
            out["date"]
        ).max()
    )


    metadata = {

        "model":
            "LightGBM",

        "lightgbm_version":
            lgb.__version__,

        "mode":
            C.MODE,

        "horizon":
            int(C.HORIZON),

        "training_start_date":
            date_text(
                date_lookup,
                1
            ),

        "training_end_date":
            date_text(
                date_lookup,
                train_end
            ),

        "validation_start_date":
            date_text(
                date_lookup,
                validation_start
            ),

        "validation_end_date":
            date_text(
                date_lookup,
                train_end
            ),

        "prediction_start_date":
            date_text(
                date_lookup,
                first_pred
            ),

        "prediction_end_date":
            date_text(
                date_lookup,
                prediction_end
            ),

        "training_rows_initial":
            int(n_train),

        "validation_rows":
            int(n_validation),

        "final_training_rows":
            int(
                n_train +
                n_validation
            ),

        "prediction_rows":
            int(n_prediction),

        "number_of_features":
            int(len(feats)),

        "number_of_series":
            int(
                out["id"].nunique()
            ),

        "first_forecast_date":
            str(
                first_date.date()
            ),

        "last_forecast_date":
            str(
                last_date.date()
            ),

        "best_iteration":
            int(best_iteration),

        "best_validation_rmse":
            float(best_score),

        "max_bin":
            int(MAX_BIN),

        "num_leaves":
            int(NUM_LEAVES),

        "learning_rate":
            float(LEARNING_RATE),

        "n_threads":
            int(N_THREADS),

        "tree_method":
            "gbdt",
    }


    metadata_path = (
        C.OUT
        /
        f"lgb_train_meta_{C.MODE}.json"
    )


    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2
        )
    )


    # ========================================================================
    # CLEANUP
    # ========================================================================

    print()
    print(
        "cleaning temporary memory..."
    )


    del (
        prediction_df,
        predictions,
        pred,
        out,
        wide,
        final_model,
        final_data,
        keep,
        y,
        df,
        date_lookup,
        feats,
    )


    gc.collect()


    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================

    print()
    print("=" * 75)
    print("LIGHTGBM TRAINING COMPLETE")
    print("=" * 75)


    print()
    print(
        "Model:"
    )

    print(
        model_path
    )


    print()
    print(
        "Forecast:"
    )

    print(
        forecast_path
    )


    print()
    print(
        "Submission:"
    )

    print(
        submission_path
    )


    print()
    print(
        "Internal validation RMSE:"
    )

    print(
        f"{best_score:.6f}"
    )


    print()
    print(
        "Best boosting iterations:"
    )

    print(
        best_iteration
    )


    print()
    print(
        "FINAL FORECAST PERIOD:"
    )

    print(
        f"{first_date.date()} "
        f"-> "
        f"{last_date.date()}"
    )


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()