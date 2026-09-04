"""
Central configuration for the Walmart M5 forecasting project.

Edit values here instead of changing them throughout the project.
"""

from pathlib import Path


# ============================================================================
# PROJECT PATHS
# ============================================================================

ROOT = Path(__file__).resolve().parents[1]

RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
OUT = ROOT / "outputs"

for _p in (RAW, PROC, OUT):
    _p.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================================
# M5 FORECASTING SETUP
# ============================================================================

# We are forecasting 28 days.
HORIZON = 28


# ---------------------------------------------------------------------------
# IMPORTANT M5 DATES
# ---------------------------------------------------------------------------
#
# TRAINING:
# d_1 -> d_1913
# Jan 29, 2011 -> Apr 24, 2016
#
# EVALUATION:
# d_1914 -> d_1941
# Apr 25, 2016 -> May 22, 2016
#
# FINAL FORECAST:
# d_1942 -> d_1969
# May 23, 2016 -> Jun 19, 2016
#


D_TRAIN_END = 1913

D_VALID_END = 1941

D_EVAL_END = 1969


# ============================================================================
# FORECAST MODES
# ============================================================================

"""
The project uses two modes.

1. validation
   -------------------------
   Training history:
       d_1 -> d_1913

   Prediction:
       d_1914 -> d_1941

   Purpose:
       Evaluate and tune the model.

2. evaluation
   -------------------------
   Available history:
       d_1 -> d_1941

   Prediction:
       d_1942 -> d_1969

   Purpose:
       Final 28-day forecast.

We intentionally do NOT use the old "future" mode because it would
forecast d_1970 -> d_1997, which is outside our current project scope.
"""

MODES = {

    "validation": (
        D_TRAIN_END,
        D_TRAIN_END + 1
    ),

    "evaluation": (
        D_VALID_END,
        D_VALID_END + 1
    ),
}


# ============================================================================
# CURRENT MODE
# ============================================================================

# Start with validation.
#
# This means:
#
#   Train       -> d_1 ... d_1913
#   Predict     -> d_1914 ... d_1941
#
# After the model is validated and tuned, we switch to:
#
#   MODE = "evaluation"
#
# which means:
#
#   Train       -> d_1 ... d_1941
#   Predict     -> d_1942 ... d_1969

MODE = "validation"


# ============================================================================
# DATA SIZE / MEMORY SETTINGS
# ============================================================================

# Keep only the latest N historical days as training rows.
#
# Feature engineering still looks farther back when creating lag_364.
#
# 730 days = approximately 2 years.
#
# This is useful because the full M5 dataset is very large.
#
# Set to None if your machine has enough RAM.

TRAIN_TAIL_DAYS = 730


# ============================================================================
# STORE FILTER
# ============================================================================

# During development you can use something like:
#
# STORES = ["CA_1"]
#
# or:
#
# STORES = ["CA_1", "CA_2"]
#
# None means all 10 Walmart stores.

STORES = None


# ============================================================================
# RANDOM SEED
# ============================================================================

SEED = 42


# ============================================================================
# XGBOOST SETTINGS
# ============================================================================

"""
XGBoost will be our FIRST model.

We will tune these parameters after the initial model works.

The objective is suitable for non-negative demand/count-like targets.
"""

XGB_PARAMS = {

    "objective": "reg:squarederror",

    "eval_metric": "rmse",

    "learning_rate": 0.05,

    "max_depth": 8,

    "min_child_weight": 10,

    "subsample": 0.8,

    "colsample_bytree": 0.8,

    "reg_alpha": 0.0,

    "reg_lambda": 1.0,

    "tree_method": "hist",

    "nthread": 4,

    "seed": SEED,
}


# Maximum number of boosting rounds.
XGB_NUM_BOOST_ROUND = 1200


# Stop if validation performance does not improve for this many rounds.
XGB_EARLY_STOPPING = 100


# ============================================================================
# LIGHTGBM SETTINGS
# ============================================================================

"""
LightGBM will be tested AFTER XGBoost.

We keep the LightGBM configuration here so both models can use the
same feature set and the same train/evaluation timeline.
"""

LGB_PARAMS = {

    "objective": "tweedie",

    "tweedie_variance_power": 1.1,

    "metric": "rmse",

    "learning_rate": 0.05,

    "num_leaves": 128,

    "min_data_in_leaf": 100,

    "feature_fraction": 0.8,

    "bagging_fraction": 0.8,

    "bagging_freq": 1,

    "lambda_l2": 0.1,

    "max_bin": 127,

    "num_threads": 4,

    "force_row_wise": True,

    "verbosity": -1,

    "seed": SEED,
}


LGB_NUM_BOOST_ROUND = 1200

LGB_EARLY_STOPPING = 100