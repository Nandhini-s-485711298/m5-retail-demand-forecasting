"""
Step 2: build model features.

All demand lags are >= 28 days, so all 28 forecast days can be
predicted in one shot without recursive prediction.

Pipeline:

    base_<MODE>.parquet
            ↓
      Feature Engineering
            ↓
    features_<MODE>.parquet

Run:
    python -m src.features
"""

import gc

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src import config as C


# ============================================================================
# UTILITIES
# ============================================================================

def to_str(s: pd.Series) -> pd.Series:
    """
    Safely convert categorical/string-like Series to object strings.
    """

    if isinstance(s.dtype, pd.CategoricalDtype):

        cats = np.asarray(
            s.cat.categories.to_numpy(dtype=object),
            dtype=object
        )

        codes = s.cat.codes.to_numpy()

        out = np.where(
            codes >= 0,
            cats[codes.clip(0)],
            None
        )

        return pd.Series(
            out,
            index=s.index,
            dtype=object
        )

    return pd.Series(
        np.asarray(
            s.to_numpy(),
            dtype=object
        ),
        index=s.index
    )


# ============================================================================
# FORECAST LAG LOGIC
# ============================================================================

def min_lag(train_end: int) -> int:
    """
    Minimum safe demand lag.

    Because the forecast horizon is 28 days, the smallest demand lag
    must be at least 28 days.

    This allows all 28 future days to be predicted directly without
    recursive prediction.
    """

    first_pred = C.MODES[C.MODE][1]

    return max(
        C.HORIZON,
        first_pred - train_end - 1 + C.HORIZON
    )


def lag_set(train_end: int) -> list[int]:
    """
    Safe demand lags.

    Every lag is >= 28 days.

    The additional longer lags capture:

        12-week seasonality
        ~13-week seasonality
        ~24-week seasonality
        ~26-week seasonality
        ~2-year seasonality
    """

    m = min_lag(train_end)

    base = [
        0,
        1,
        2,
        7,
        14,
        28,
        56,
        63,
        140,
        154,
        336,
        700
    ]

    lags = [
        m + b
        for b in base
    ]

    # One-year M5 seasonal lag.
    lags.append(364)

    # Remove duplicates and sort.
    return sorted(
        set(lags)
    )


def data_train_end(
    df: pd.DataFrame
) -> int:
    """
    Find the final historical day containing actual sales.
    """

    actual = df.loc[
        df["sales"].notna(),
        "d_num"
    ]

    if actual.empty:
        raise RuntimeError(
            "No historical sales values found."
        )

    return int(
        actual.max()
    )


# ============================================================================
# FEATURE DEFINITIONS
# ============================================================================

ROLL_WINDOWS = [
    7,
    14,
    28,
    56,
    91,
    180,
    364
]

STD_WINDOWS = [
    7,
    28,
    56,
    91
]

CAT_FEATURES = [
    "item_id",
    "dept_id",
    "cat_id",
    "store_id",
    "state_id",
    "weekday",
    "event_name_1",
    "event_type_1",
    "event_name_2",
    "event_type_2",
]


# ============================================================================
# MAIN FEATURE ENGINEERING
# ============================================================================

def add_features(
    df: pd.DataFrame,
    shift: int | None = None
) -> pd.DataFrame:

    """
    Add leakage-safe demand, rolling, price and calendar features.

    IMPORTANT:

    The forecast horizon is 28 days.

    Therefore every demand-derived feature is based on information
    at least 28 days before the target date.

    This means:

        target date
             ↓
        28+ day historical gap
             ↓
        historical demand features

    No future sales are used.
    """

    # ------------------------------------------------------------------------
    # Sort
    # ------------------------------------------------------------------------

    df = df.sort_values(
        [
            "id",
            "d_num"
        ],
        kind="stable"
    ).reset_index(
        drop=True
    )

    g = df.groupby(
        "id",
        observed=True
    )["sales"]


    # ------------------------------------------------------------------------
    # Determine safe shift
    # ------------------------------------------------------------------------

    if shift is None:

        train_end = data_train_end(
            df
        )

        shift = min_lag(
            train_end
        )

        lags = lag_set(
            train_end
        )

    else:

        lags = sorted(
            set(
                [
                    shift + b
                    for b in [
                        0,
                        1,
                        2,
                        7,
                        14,
                        28,
                        56,
                        63,
                        140,
                        154,
                        336,
                        700
                    ]
                ]
                +
                [364]
            )
        )


    print(
        f"  lags {lags} "
        f"(shift={shift}) ..."
    )


    # ------------------------------------------------------------------------
    # Demand lag features
    # ------------------------------------------------------------------------

    for lag in lags:

        df[f"lag_{lag}"] = (
            g.shift(lag)
            .astype("float32")
        )


    # =========================================================================
    # ROLLING DEMAND FEATURES
    # =========================================================================

    print(
        "  rolling demand statistics ..."
    )

    # ------------------------------------------------------------------------
    # Base demand
    # ------------------------------------------------------------------------

    base = g.shift(
        shift
    ).to_numpy(
        dtype="float32"
    )


    # ------------------------------------------------------------------------
    # Series boundaries
    # ------------------------------------------------------------------------

    if isinstance(
        df["id"].dtype,
        pd.CategoricalDtype
    ):

        codes = (
            df["id"]
            .cat.codes
            .to_numpy()
        )

    else:

        codes = pd.factorize(
            df["id"],
            sort=False
        )[0]


    starts = np.flatnonzero(
        np.r_[
            True,
            codes[1:] != codes[:-1]
        ]
    )

    ends = np.r_[
        starts[1:],
        len(codes)
    ]


    pos = (
        np.arange(
            len(codes)
        )
        -
        np.repeat(
            starts,
            ends - starts
        )
    )


    # ------------------------------------------------------------------------
    # Replace NaN for cumulative calculations
    # ------------------------------------------------------------------------

    nan_mask = np.isnan(
        base
    )

    vals = np.where(
        nan_mask,
        np.float32(0),
        base
    ).astype(
        "float32"
    )

    cnt = (
        ~nan_mask
    ).astype(
        "float32"
    )


    # =========================================================================
    # MEMORY-EFFICIENT CUMULATIVE SUM
    # =========================================================================

    def _block_cumsum(a):
        """
        Cumulative sum reset at every item-store series.
        """

        c = np.cumsum(
            a,
            dtype="float32"
        )

        offset = np.empty(
            len(starts),
            dtype="float32"
        )

        offset[0] = 0.0

        if len(starts) > 1:

            offset[1:] = c[
                starts[1:] - 1
            ]

        c -= np.repeat(
            offset,
            ends - starts
        )

        return c


    ones = _block_cumsum(
        np.ones(
            len(base),
            dtype="float32"
        )
    )

    csum = _block_cumsum(
        vals
    )

    ccnt = _block_cumsum(
        cnt
    )

    csq = _block_cumsum(
        vals * vals
    )

    czero = _block_cumsum(
        (
            (base == 0)
            &
            ~nan_mask
        ).astype(
            "float32"
        )
    )


    ar = np.arange(
        len(base)
    )


    # =========================================================================
    # WINDOW FUNCTION
    # =========================================================================

    def _window(
        cumulative,
        window
    ):
        """
        Trailing window within each series.
        """

        idx = np.maximum(
            ar - window,
            0
        )

        previous = np.where(
            pos >= window,
            cumulative[idx],
            0.0
        )

        return cumulative - previous


    # =========================================================================
    # ROLLING MEANS
    # =========================================================================

    for w in ROLL_WINDOWS:

        n = _window(
            ccnt,
            w
        )

        total = _window(
            csum,
            w
        )

        df[f"rmean_{w}"] = np.divide(
            total,
            n,
            out=np.full(
                len(base),
                np.nan,
                dtype="float32"
            ),
            where=n > 0
        ).astype(
            "float32"
        )


    # =========================================================================
    # ROLLING STANDARD DEVIATION
    # =========================================================================

    for w in STD_WINDOWS:

        n = _window(
            ccnt,
            w
        )

        mean = np.divide(
            _window(
                csum,
                w
            ),
            n,
            out=np.full(
                len(base),
                np.nan,
                dtype="float32"
            ),
            where=n > 0
        )

        second_moment = np.divide(
            _window(
                csq,
                w
            ),
            n,
            out=np.full(
                len(base),
                np.nan,
                dtype="float32"
            ),
            where=n > 0
        )

        var = (
            second_moment
            -
            mean * mean
        )

        # Sample variance.
        var = np.where(
            n > 1,
            var * n / (n - 1),
            np.nan
        )

        df[f"rstd_{w}"] = np.sqrt(
            np.clip(
                var,
                0,
                None
            )
        ).astype(
            "float32"
        )


    # =========================================================================
    # ZERO DEMAND RATE
    # =========================================================================

    rows = _window(
        ones,
        28
    )

    df["zero_rate_28"] = np.divide(
        _window(
            czero,
            28
        ),
        rows,
        out=np.full(
            len(base),
            np.nan,
            dtype="float32"
        ),
        where=rows > 0
    ).astype(
        "float32"
    )


    # =========================================================================
    # DEMAND TREND FEATURES
    # =========================================================================

    print(
        "  demand trend features ..."
    )

    # ------------------------------------------------------------------------
    # Short-term vs medium-term
    # ------------------------------------------------------------------------

    df["demand_trend_7_28"] = np.divide(
        df["rmean_7"],
        df["rmean_28"],
        out=np.full(
            len(df),
            np.nan,
            dtype="float32"
        ),
        where=(
            df["rmean_28"].to_numpy(
                dtype="float32"
            ) > 0
        )
    ).astype(
        "float32"
    )


    # ------------------------------------------------------------------------
    # Medium-term vs long-term
    # ------------------------------------------------------------------------

    df["demand_trend_28_91"] = np.divide(
        df["rmean_28"],
        df["rmean_91"],
        out=np.full(
            len(df),
            np.nan,
            dtype="float32"
        ),
        where=(
            df["rmean_91"].to_numpy(
                dtype="float32"
            ) > 0
        )
    ).astype(
        "float32"
    )


    # ------------------------------------------------------------------------
    # Short-term vs long-term
    # ------------------------------------------------------------------------

    df["demand_trend_7_180"] = np.divide(
        df["rmean_7"],
        df["rmean_180"],
        out=np.full(
            len(df),
            np.nan,
            dtype="float32"
        ),
        where=(
            df["rmean_180"].to_numpy(
                dtype="float32"
            ) > 0
        )
    ).astype(
        "float32"
    )


    # ------------------------------------------------------------------------
    # Yearly demand comparison
    # ------------------------------------------------------------------------

    if "lag_364" in df.columns:

        df["demand_vs_year"] = np.divide(
            df["rmean_28"],
            df["lag_364"],
            out=np.full(
                len(df),
                np.nan,
                dtype="float32"
            ),
            where=(
                df["lag_364"].to_numpy(
                    dtype="float32"
                ) > 0
            )
        ).astype(
            "float32"
        )


    # ------------------------------------------------------------------------
    # Seasonal lag comparison
    # ------------------------------------------------------------------------

    if "lag_182" in df.columns:

        df["demand_vs_182d"] = np.divide(
            df["rmean_28"],
            df["lag_182"],
            out=np.full(
                len(df),
                np.nan,
                dtype="float32"
            ),
            where=(
                df["lag_182"].to_numpy(
                    dtype="float32"
                ) > 0
            )
        ).astype(
            "float32"
        )


    # =========================================================================
    # RELEASE MEMORY FROM ROLLING CALCULATIONS
    # =========================================================================

    del (
        base,
        vals,
        cnt,
        csum,
        ccnt,
        csq,
        czero,
        ones,
        nan_mask,
        g,
        codes,
        starts,
        ends,
        pos,
        ar
    )

    gc.collect()


    # =========================================================================
    # PRICE FEATURES
    # =========================================================================

    print(
        "  price features ..."
    )

    pg = df.groupby(
        "id",
        observed=True
    )["sell_price"]


    # ------------------------------------------------------------------------
    # Weekly price change
    # ------------------------------------------------------------------------

    previous_price = pg.shift(
        7
    )

    df["price_change_w"] = (
        df["sell_price"]
        /
        previous_price
        -
        1
    ).astype(
        "float32"
    )

    del previous_price


    # =========================================================================
    # HISTORICAL ITEM PRICE
    # =========================================================================

    historical_price_sum = (
        df.groupby(
            "id",
            observed=True
        )["sell_price"]
        .cumsum()
    )

    historical_price_count = (
        df.groupby(
            "id",
            observed=True
        )["sell_price"]
        .transform(
            lambda x:
            x.notna().cumsum()
        )
    )

    historical_item_price_mean = np.divide(
        historical_price_sum.to_numpy(
            dtype="float32"
        ),
        historical_price_count.to_numpy(
            dtype="float32"
        ),
        out=np.full(
            len(df),
            np.nan,
            dtype="float32"
        ),
        where=(
            historical_price_count
            .to_numpy()
            > 0
        )
    )

    df["price_rel_item"] = (
        df["sell_price"].to_numpy(
            dtype="float32"
        )
        /
        historical_item_price_mean
    ).astype(
        "float32"
    )

    del (
        historical_price_sum,
        historical_price_count,
        historical_item_price_mean
    )

    gc.collect()


    # =========================================================================
    # PRICE RELATIVE TO DEPARTMENT
    # =========================================================================

    dept_day_mean = (
        df.groupby(
            [
                "dept_id",
                "d_num"
            ],
            observed=True
        )["sell_price"]
        .transform(
            "mean"
        )
    )

    df["price_rel_dept"] = (
        df["sell_price"]
        /
        dept_day_mean
    ).astype(
        "float32"
    )

    del (
        pg,
        dept_day_mean
    )

    gc.collect()


    # =========================================================================
    # CALENDAR FEATURES
    # =========================================================================

    print(
        "  calendar features ..."
    )

    df["dayofweek"] = (
        df["date"]
        .dt.dayofweek
        .astype(
            "int8"
        )
    )

    df["dayofmonth"] = (
        df["date"]
        .dt.day
        .astype(
            "int8"
        )
    )

    df["weekofyear"] = (
        df["date"]
        .dt.isocalendar()
        .week
        .astype(
            "int8"
        )
    )

    df["is_weekend"] = (
        df["dayofweek"] >= 5
    ).astype(
        "int8"
    )


    # =========================================================================
    # EVENT FLAG
    # =========================================================================

    event1 = to_str(
        df["event_name_1"]
    )

    event2 = to_str(
        df["event_name_2"]
    )

    df["is_event"] = (
        event1.ne("NoEvent")
        |
        event2.ne("NoEvent")
    ).astype(
        "int8"
    )

    del (
        event1,
        event2
    )

    gc.collect()


    # =========================================================================
    # SNAP
    # =========================================================================

    state = to_str(
        df["state_id"]
    )

    df["snap"] = np.select(
        [
            state == "CA",
            state == "TX",
            state == "WI"
        ],
        [
            df["snap_CA"],
            df["snap_TX"],
            df["snap_WI"]
        ],
        default=0
    ).astype(
        "int8"
    )

    df.drop(
        columns=[
            "snap_CA",
            "snap_TX",
            "snap_WI"
        ],
        inplace=True
    )

    del state

    gc.collect()


    # =========================================================================
    # DAYS SINCE RELEASE
    # =========================================================================

    first = (
        df.loc[
            df["sell_price"].notna()
        ]
        .groupby(
            "id",
            observed=True
        )["d_num"]
        .min()
    )

    df["days_since_release"] = (
        df["d_num"]
        -
        df["id"].map(
            first
        )
    ).astype(
        "float32"
    )

    del first

    gc.collect()


    # =========================================================================
    # CLEAN NUMERIC TYPES
    # =========================================================================

    numeric_features = [
        c
        for c in df.columns
        if c.startswith("lag_")
        or c.startswith("rmean_")
        or c.startswith("rstd_")
        or c.startswith("demand_")
        or c in [
            "zero_rate_28",
            "price_change_w",
            "price_rel_item",
            "price_rel_dept",
            "days_since_release"
        ]
    ]

    for c in numeric_features:

        if c in df.columns:

            df[c] = df[c].astype(
                "float32"
            )


    gc.collect()

    return df


# ============================================================================
# FEATURE COLUMN SELECTION
# ============================================================================

def feature_columns(
    df: pd.DataFrame
) -> list[str]:

    """
    Return model feature columns.

    Excluded:

        id
        sales
        date
        wm_yr_wk
        d_num
    """

    drop = {
        "id",
        "sales",
        "date",
        "wm_yr_wk",
        "d_num"
    }

    return [
        c
        for c in df.columns
        if c not in drop
    ]


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:

    """
    Build features one store at a time.

    This keeps RAM usage manageable for the M5 dataset.
    """

    src = (
        C.PROC
        /
        f"base_{C.MODE}.parquet"
    )

    print(
        f"reading {src}"
    )


    # =========================================================================
    # FIND STORES
    # =========================================================================

    stores = pd.read_parquet(
        src,
        columns=[
            "store_id"
        ]
    )

    store_list = sorted(
        str(x)
        for x in
        stores[
            "store_id"
        ]
        .dropna()
        .unique()
    )

    del stores

    gc.collect()


    print(
        f"processing "
        f"{len(store_list)} stores one at a time"
    )


    # =========================================================================
    # OUTPUT
    # =========================================================================

    out_path = (
        C.PROC
        /
        f"features_{C.MODE}.parquet"
    )

    writer = None

    total = 0


    # =========================================================================
    # PROCESS STORES
    # =========================================================================

    for i, store in enumerate(
        store_list,
        1
    ):

        print()
        print(
            f"processing store "
            f"{i}/{len(store_list)}: "
            f"{store}"
        )


        part = pd.read_parquet(
            src,
            filters=[
                (
                    "store_id",
                    "==",
                    store
                )
            ]
        )

        if part.empty:
            continue


        # ---------------------------------------------------------------------
        # Feature engineering
        # ---------------------------------------------------------------------

        part = add_features(
            part
        )


        # ---------------------------------------------------------------------
        # Historical end
        # ---------------------------------------------------------------------

        train_end = data_train_end(
            part
        )


        # ---------------------------------------------------------------------
        # Optional tail
        # ---------------------------------------------------------------------

        if C.TRAIN_TAIL_DAYS:

            part = part[
                part["d_num"]
                >
                train_end
                -
                C.TRAIN_TAIL_DAYS
            ]


        # ---------------------------------------------------------------------
        # Remove historical rows without price.
        #
        # Keep forecast rows.
        # ---------------------------------------------------------------------

        part = part[
            part["sell_price"].notna()
            |
            (
                part["d_num"]
                >
                train_end
            )
        ]


        # ---------------------------------------------------------------------
        # Keep historical + forecast period
        # ---------------------------------------------------------------------

        first_pred = C.MODES[
            C.MODE
        ][1]

        part = part[
            (
                part["d_num"]
                <=
                train_end
            )
            |
            (
                part["d_num"]
                >=
                first_pred
            )
        ]


        part = part.reset_index(
            drop=True
        )


        # =========================================================================
        # WRITE PARQUET
        # =========================================================================

        table = pa.Table.from_pandas(
            part,
            preserve_index=False
        )


        if writer is None:

            writer = pq.ParquetWriter(
                out_path,
                table.schema,
                compression="snappy"
            )


        writer.write_table(
            table
        )


        total += len(part)


        print(
            f"  [{i}/{len(store_list)}] "
            f"{store}: "
            f"{len(part):,} rows"
        )


        del (
            part,
            table
        )

        gc.collect()


    # =========================================================================
    # CLOSE WRITER
    # =========================================================================

    if writer is not None:
        writer.close()


    print()
    print(
        f"wrote {out_path}"
    )

    print(
        f"rows={total:,}"
    )


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()