"""
Step 1: Prepare the Walmart M5 data.

This script:

1. Reads sales_train_evaluation.csv
2. Keeps only the required historical/prediction period
3. Converts the wide sales table into a tidy daily table
4. Merges calendar information
5. Merges sell-price information
6. Carries the last known price into the final forecast period
7. Saves the prepared data as Parquet

Our project timeline:

VALIDATION MODE
----------------
Training:
    d_1 -> d_1913
    Jan 29, 2011 -> Apr 24, 2016

Evaluation:
    d_1914 -> d_1941
    Apr 25, 2016 -> May 22, 2016


EVALUATION MODE
----------------
Retraining history:
    d_1 -> d_1941
    Jan 29, 2011 -> May 22, 2016

Final forecast:
    d_1942 -> d_1969
    May 23, 2016 -> Jun 19, 2016


Run:

    python -m src.prepare_data
"""

import gc

import numpy as np
import pandas as pd

from src import config as C


# ============================================================================
# ID COLUMNS
# ============================================================================

ID_COLS = [
    "id",
    "item_id",
    "dept_id",
    "cat_id",
    "store_id",
    "state_id",
]


# ============================================================================
# READ SALES DATA
# ============================================================================

def _read_sales() -> pd.DataFrame:
    """
    Read the M5 sales evaluation file.

    sales_train_evaluation.csv contains actual sales through d_1941.

    If it is unavailable, fall back to sales_train_validation.csv,
    which contains actual sales only through d_1913.
    """

    evaluation_file = (
        C.RAW /
        "sales_train_evaluation.csv"
    )

    validation_file = (
        C.RAW /
        "sales_train_validation.csv"
    )


    if evaluation_file.exists():

        f = evaluation_file

    elif validation_file.exists():

        f = validation_file

        print(
            "! sales_train_evaluation.csv not found."
        )

        print(
            "! Using sales_train_validation.csv."
        )

        print(
            "! Available sales history ends at d_1913."
        )

    else:

        raise FileNotFoundError(
            "\nCould not find either:\n"
            "  sales_train_evaluation.csv\n"
            "  sales_train_validation.csv\n\n"
            f"Place the files inside:\n{C.RAW}"
        )


    print(
        f"reading {f.name}"
    )


    df = pd.read_csv(
        f
    )


    # ------------------------------------------------------------------------
    # Normalize ID names
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


    return df


# ============================================================================
# EXTEND CALENDAR IF NECESSARY
# ============================================================================

def _extend_calendar(
    cal: pd.DataFrame,
    need_day: int
) -> pd.DataFrame:
    """
    Extend calendar only if the requested forecast period goes beyond
    calendar.csv.

    For our current project the calendar normally already contains
    d_1969, so this function should not need to generate anything.

    It is retained as a safety mechanism.
    """

    last = int(
        cal["d_num"].max()
    )


    if need_day <= last:

        return cal


    n = (
        need_day -
        last
    )


    print(
        f"! calendar.csv ends at d_{last}."
    )

    print(
        f"! Generating {n} additional calendar days."
    )


    # ------------------------------------------------------------------------
    # Start date
    # ------------------------------------------------------------------------

    last_date = (
        cal.loc[
            cal["d_num"] == last,
            "date"
        ]
        .iloc[0]
    )


    start = (
        last_date
        +
        pd.Timedelta(
            days=1
        )
    )


    dates = pd.date_range(
        start,
        periods=n,
        freq="D"
    )


    # ------------------------------------------------------------------------
    # Basic calendar fields
    # ------------------------------------------------------------------------

    ext = pd.DataFrame(
        {
            "date": dates,

            "d": [
                f"d_{last + i + 1}"
                for i in range(n)
            ],

            "d_num": np.arange(
                last + 1,
                need_day + 1,
                dtype="int16"
            ),

            "weekday": dates.day_name(),

            "wday": (
                (dates.dayofweek + 2)
                % 7
                + 1
            ),

            "month": dates.month,

            "year": dates.year,
        }
    )


    # ------------------------------------------------------------------------
    # Walmart week number
    # ------------------------------------------------------------------------

    last_wk = int(
        cal.loc[
            cal["d_num"] == last,
            "wm_yr_wk"
        ]
        .iloc[0]
    )


    days_into_week = int(
        cal.loc[
            cal["d_num"] == last,
            "wday"
        ]
        .iloc[0]
    ) - 1


    wk_offset = (
        (
            np.arange(n)
            +
            days_into_week
            +
            1
        )
        // 7
    )


    ext["wm_yr_wk"] = (
        last_wk +
        wk_offset
    )


    # Handle week-year rollover.
    ext["wm_yr_wk"] = np.where(
        ext["wm_yr_wk"] % 100 > 52,

        (
            ext["wm_yr_wk"] // 100
            +
            1
        )
        * 100
        +
        1,

        ext["wm_yr_wk"]
    )


    # ------------------------------------------------------------------------
    # Event columns
    # ------------------------------------------------------------------------

    for c in [
        "event_name_1",
        "event_type_1",
        "event_name_2",
        "event_type_2",
    ]:

        ext[c] = pd.Series(
            [None] * n,
            index=ext.index,
            dtype=object
        )


    # Known events that could be needed beyond the supplied calendar.
    future_events = {

        "2016-06-19": (
            "Father's day",
            "Cultural"
        ),

        "2016-07-04": (
            "IndependenceDay",
            "National"
        ),

        "2016-07-06": (
            "Eid al-Fitr",
            "Religious"
        ),
    }


    for date_string, (
        name,
        event_type
    ) in future_events.items():

        mask = (
            ext["date"]
            ==
            pd.Timestamp(
                date_string
            )
        )


        ext.loc[
            mask,
            "event_name_1"
        ] = name


        ext.loc[
            mask,
            "event_type_1"
        ] = event_type


    # ------------------------------------------------------------------------
    # SNAP
    # ------------------------------------------------------------------------

    day_of_month = dates.day


    # California
    ext["snap_CA"] = np.isin(
        day_of_month,
        range(1, 11)
    ).astype("int8")


    # Texas
    ext["snap_TX"] = np.isin(
        day_of_month,
        [
            1,
            3,
            5,
            6,
            7,
            9,
            11,
            12,
            13,
            15,
        ]
    ).astype("int8")


    # Wisconsin
    ext["snap_WI"] = np.isin(
        day_of_month,
        [
            2,
            3,
            5,
            6,
            8,
            9,
            11,
            12,
            14,
            15,
        ]
    ).astype("int8")


    # Keep exactly the same columns/order as calendar.csv.
    ext = ext[
        cal.columns
    ]


    return pd.concat(
        [
            cal,
            ext
        ],
        ignore_index=True
    )


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:

    print("=" * 75)

    print(
        "M5 DATA PREPARATION"
    )

    print("=" * 75)


    # =========================================================================
    # 1. READ SALES
    # =========================================================================

    sales = _read_sales()


    # =========================================================================
    # 2. OPTIONAL STORE FILTER
    # =========================================================================

    if C.STORES:

        sales = sales[
            sales["store_id"].isin(
                C.STORES
            )
        ].reset_index(
            drop=True
        )


        print(
            f"filtered to stores "
            f"{C.STORES}: "
            f"{len(sales):,} series"
        )


    # =========================================================================
    # 3. DETERMINE AVAILABLE SALES HISTORY
    # =========================================================================

    sales_day_columns = [
        c
        for c in sales.columns
        if c.startswith("d_")
    ]


    if not sales_day_columns:

        raise RuntimeError(
            "No d_ sales columns were found."
        )


    last_day = max(
        int(
            c[2:]
        )
        for c in sales_day_columns
    )


    # =========================================================================
    # 4. GET CURRENT MODE
    # =========================================================================

    train_end, first_pred = (
        C.MODES[C.MODE]
    )


    # The actual sales file may contain less history than the configured
    # training end.
    train_end = min(
        train_end,
        last_day
    )


    prediction_end = (
        first_pred
        +
        C.HORIZON
        -
        1
    )


    print()
    print(
        f"MODE: {C.MODE}"
    )

    print(
        f"Sales file ends: "
        f"d_{last_day}"
    )

    print(
        f"Training/retraining ends: "
        f"d_{train_end}"
    )

    print(
        f"Prediction starts: "
        f"d_{first_pred}"
    )

    print(
        f"Prediction ends: "
        f"d_{prediction_end}"
    )


    # =========================================================================
    # 5. DETERMINE REQUIRED DAYS
    # =========================================================================
    #
    # validation mode:
    #
    #   keep d_1 -> d_1913
    #   create empty d_1914 -> d_1941
    #
    # evaluation mode:
    #
    #   keep d_1 -> d_1941
    #   create empty d_1942 -> d_1969
    #
    # This is the key separation between actual historical sales and
    # future prediction rows.
    #


    forecast_days = [
        f"d_{d}"
        for d in range(
            first_pred,
            prediction_end + 1
        )
    ]


    # =========================================================================
    # 6. REMOVE SALES AFTER TRAINING END
    # =========================================================================
    #
    # If the source file contains sales beyond the current training boundary,
    # remove them.
    #
    # Example:
    #
    # validation mode:
    #
    # source contains d_1914 -> d_1941
    #
    # but we DO NOT allow these values into training.
    #

    columns_to_remove = [

        f"d_{d}"

        for d in range(
            train_end + 1,
            last_day + 1
        )

        if f"d_{d}" in sales.columns
    ]


    if columns_to_remove:

        print()
        print(
            f"removing {len(columns_to_remove)} "
            f"future sales columns to prevent leakage"
        )


        sales.drop(
            columns=columns_to_remove,
            inplace=True
        )


    # =========================================================================
    # 7. ADD EMPTY FORECAST COLUMNS
    # =========================================================================
    #
    # These columns intentionally contain NaN.
    #
    # They represent dates where the model must predict sales.
    #

    new_forecast_columns = [
        c
        for c in forecast_days
        if c not in sales.columns
    ]


    if new_forecast_columns:

        empty_forecast = pd.DataFrame(
            np.nan,
            index=sales.index,
            columns=new_forecast_columns
        )


        sales = pd.concat(
            [
                sales,
                empty_forecast
            ],
            axis=1,
            copy=False
        )


    # =========================================================================
    # 8. REMOVE DUPLICATE DAY COLUMNS
    # =========================================================================

    duplicate_columns = (
        sales.columns[
            sales.columns.duplicated()
        ]
        .tolist()
    )


    if duplicate_columns:

        print(
            f"! removing "
            f"{len(duplicate_columns)} "
            f"duplicate columns"
        )


        sales = sales.loc[
            :,
            ~sales.columns.duplicated()
        ]


    # =========================================================================
    # 9. KEEP ONLY THE REQUIRED TRAINING TAIL
    # =========================================================================
    #
    # We need enough historical data for:
    #
    #   lag_364
    #
    # and rolling features.
    #
    # TRAIN_TAIL_DAYS = 730
    #
    # We additionally keep 400 extra days for safe yearly-lag calculation.
    #

    if C.TRAIN_TAIL_DAYS:

        keep_from = max(
            1,
            train_end
            -
            C.TRAIN_TAIL_DAYS
            -
            400
        )


        columns_to_drop = [

            f"d_{d}"

            for d in range(
                1,
                keep_from
            )

            if f"d_{d}" in sales.columns
        ]


        if columns_to_drop:

            sales.drop(
                columns=columns_to_drop,
                inplace=True
            )


        print()
        print(
            f"keeping historical days "
            f"d_{keep_from} -> d_{train_end}"
        )

        print(
            f"plus forecast days "
            f"d_{first_pred} -> d_{prediction_end}"
        )


    # =========================================================================
    # 10. MELT SALES FROM WIDE TO LONG FORMAT
    # =========================================================================

    day_columns = sorted(
        [
            c
            for c in sales.columns
            if c.startswith("d_")
        ],
        key=lambda c: int(
            c[2:]
        )
    )


    print()
    print(
        f"melting "
        f"{len(sales):,} series x "
        f"{len(day_columns):,} days ..."
    )


    df = sales.melt(
        id_vars=ID_COLS,

        value_vars=day_columns,

        var_name="d",

        value_name="sales"
    )


    del sales

    gc.collect()


    # =========================================================================
    # 11. SALES TYPES
    # =========================================================================

    df["sales"] = (
        df["sales"]
        .astype("float32")
    )


    df["d_num"] = (
        df["d"]
        .str.slice(2)
        .astype("int16")
    )


    df.drop(
        columns=["d"],
        inplace=True
    )


    # =========================================================================
    # 12. LOAD CALENDAR
    # =========================================================================

    calendar_file = (
        C.RAW /
        "calendar.csv"
    )


    if not calendar_file.exists():

        raise FileNotFoundError(
            f"Missing calendar.csv:\n"
            f"{calendar_file}"
        )


    print()
    print(
        "reading calendar.csv ..."
    )


    cal = pd.read_csv(
        calendar_file,
        parse_dates=["date"]
    )


    cal["d_num"] = (
        cal["d"]
        .str.slice(2)
        .astype("int16")
    )


    # We only need the calendar through the final forecast day.
    cal = _extend_calendar(
        cal,
        prediction_end
    )


    cal = cal[
        [
            "d_num",
            "date",
            "wm_yr_wk",
            "weekday",
            "month",
            "year",
            "event_name_1",
            "event_type_1",
            "event_name_2",
            "event_type_2",
            "snap_CA",
            "snap_TX",
            "snap_WI",
        ]
    ]


    # =========================================================================
    # 13. MERGE CALENDAR
    # =========================================================================

    df = df.merge(
        cal,
        on="d_num",
        how="left",
        validate="many_to_one"
    )


    del cal

    gc.collect()


    # =========================================================================
    # 14. LOAD SELL PRICES
    # =========================================================================

    prices_file = (
        C.RAW /
        "sell_prices.csv"
    )


    if not prices_file.exists():

        raise FileNotFoundError(
            f"Missing sell_prices.csv:\n"
            f"{prices_file}"
        )


    print()
    print(
        "reading sell_prices.csv ..."
    )


    prices = pd.read_csv(
        prices_file
    )


    # =========================================================================
    # 15. BUILD MEMORY-EFFICIENT PRICE KEY
    # =========================================================================
    #
    # Instead of joining on three string/integer columns directly, we create
    # one integer key.
    #
    # This avoids a huge pandas MultiIndex and reduces memory usage.
    #

    store_categories = pd.Index(
        sorted(
            prices["store_id"]
            .unique()
        )
    )


    item_categories = pd.Index(
        sorted(
            prices["item_id"]
            .unique()
        )
    )


    n_items = len(
        item_categories
    )


    price_store_code = (
        store_categories
        .get_indexer(
            prices["store_id"]
        )
        .astype("int64")
    )


    price_item_code = (
        item_categories
        .get_indexer(
            prices["item_id"]
        )
        .astype("int64")
    )


    price_week = (
        prices["wm_yr_wk"]
        .to_numpy(
            dtype="int64"
        )
    )


    price_key = (
        (
            price_store_code
            *
            n_items
            +
            price_item_code
        )
        *
        100_000
        +
        price_week
    )


    price_values = (
        prices["sell_price"]
        .to_numpy(
            dtype="float32"
        )
    )


    del (
        prices,
        price_store_code,
        price_item_code,
        price_week
    )

    gc.collect()


    # Sort keys for fast search.
    order = np.argsort(
        price_key,
        kind="stable"
    )


    price_key = (
        price_key[order]
    )


    price_values = (
        price_values[order]
    )


    # =========================================================================
    # 16. BUILD PRICE KEY FOR PANEL
    # =========================================================================

    def _codes(
        column: str,
        target: pd.Index
    ):

        s = df[column]


        if isinstance(
            s.dtype,
            pd.CategoricalDtype
        ):

            categories = (
                s.cat.categories
            )

            codes = (
                s.cat.codes
                .to_numpy()
            )

        else:

            codes, categories = (
                pd.factorize(
                    s,
                    sort=False
                )
            )


        mapped = (
            target
            .get_indexer(
                pd.Index(
                    categories
                )
            )
            .astype(
                "int64"
            )
        )


        safe_codes = np.clip(
            codes,
            0,
            None
        )


        result = np.where(
            codes >= 0,
            mapped[safe_codes],
            -1
        )


        return result.astype(
            "int64"
        )


    panel_store_code = _codes(
        "store_id",
        store_categories
    )


    panel_item_code = _codes(
        "item_id",
        item_categories
    )


    panel_week = (
        df["wm_yr_wk"]
        .to_numpy(
            dtype="int64"
        )
    )


    panel_key = (
        (
            panel_store_code
            *
            n_items
            +
            panel_item_code
        )
        *
        100_000
        +
        panel_week
    )


    del (
        panel_store_code,
        panel_item_code,
        panel_week
    )

    gc.collect()


    # =========================================================================
    # 17. LOOK UP PRICE
    # =========================================================================

    positions = np.searchsorted(
        price_key,
        panel_key
    )


    np.clip(
        positions,
        0,
        len(price_key) - 1,
        out=positions
    )


    hits = (
        price_key[positions]
        ==
        panel_key
    )


    sell_price = np.full(
        len(panel_key),
        np.nan,
        dtype="float32"
    )


    sell_price[hits] = (
        price_values[
            positions[hits]
        ]
    )


    df["sell_price"] = (
        sell_price
    )


    del (
        price_key,
        price_values,
        panel_key,
        positions,
        hits,
        sell_price
    )

    gc.collect()


    # =========================================================================
    # 18. CARRY LAST KNOWN PRICE INTO FINAL FORECAST PERIOD
    # =========================================================================
    #
    # The M5 price file does not contain future prices for d_1942 -> d_1969.
    #
    # We therefore use the last observed price for each item-store series.
    #
    # IMPORTANT:
    #
    # This is an assumption for the future forecast.
    #
    # It is NOT actual future price information.
    #

    future_mask = (
        df["d_num"]
        >=
        first_pred
    )


    missing_future_prices = int(
        df.loc[
            future_mask,
            "sell_price"
        ]
        .isna()
        .sum()
    )


    if missing_future_prices:

        print()
        print(
            f"carrying last known price forward "
            f"for {missing_future_prices:,} "
            f"future rows"
        )


        df = df.sort_values(
            [
                "id",
                "d_num"
            ],
            kind="stable"
        )


        df["sell_price"] = (
            df.groupby(
                "id",
                observed=True
            )["sell_price"]
            .ffill()
            .astype("float32")
        )


        df = df.reset_index(
            drop=True
        )


    # =========================================================================
    # 19. EVENT COLUMNS
    # =========================================================================

    for c in [
        "event_name_1",
        "event_type_1",
        "event_name_2",
        "event_type_2",
    ]:

        df[c] = (
            df[c]
            .fillna("NoEvent")
            .astype("category")
        )


    # =========================================================================
    # 20. CATEGORICAL ID COLUMNS
    # =========================================================================

    for c in (
        ID_COLS
        +
        ["weekday"]
    ):

        df[c] = (
            df[c]
            .astype("category")
        )


    # =========================================================================
    # 21. COMPACT NUMERIC TYPES
    # =========================================================================

    for c in [
        "snap_CA",
        "snap_TX",
        "snap_WI",
        "month",
    ]:

        df[c] = (
            df[c]
            .astype("int8")
        )


    df["year"] = (
        df["year"]
        .astype("int16")
    )


    df["wm_yr_wk"] = (
        df["wm_yr_wk"]
        .astype("int32")
    )


    # =========================================================================
    # 22. DATA INTEGRITY CHECK
    # =========================================================================

    n_days = (
        df["d_num"]
        .nunique()
    )


    n_series = (
        df["id"]
        .nunique()
    )


    expected_rows = (
        n_days
        *
        n_series
    )


    actual_rows = len(
        df
    )


    assert actual_rows == expected_rows, (

        f"row count {actual_rows:,} "
        f"!= "
        f"{n_series:,} series x "
        f"{n_days:,} days "
        f"= {expected_rows:,}. "

        "This indicates duplicated or missing "
        "series/day rows."
    )


    print()
    print(
        f"{n_series:,} series x "
        f"{n_days:,} days"
    )


    # =========================================================================
    # 23. CHECK REQUIRED CALENDAR COVERAGE
    # =========================================================================

    required_days = list(
        range(
            first_pred,
            prediction_end + 1
        )
    )


    missing_calendar_days = [

        d

        for d in required_days

        if d not in set(
            df["d_num"].unique()
        )
    ]


    if missing_calendar_days:

        raise RuntimeError(
            "Missing calendar/data rows for "
            f"forecast days: "
            f"{missing_calendar_days[:10]}"
        )


    # =========================================================================
    # 24. SAVE PARQUET
    # =========================================================================

    output_file = (
        C.PROC /
        f"base_{C.MODE}.parquet"
    )


    df.to_parquet(
        output_file,
        index=False
    )


    memory_gb = (
        df.memory_usage(
            deep=True
        ).sum()
        /
        1e9
    )


    print()
    print(
        f"wrote {output_file}"
    )


    print(
        f"rows={len(df):,}"
    )


    print(
        f"memory={memory_gb:.2f} GB"
    )


    print()
    print("=" * 75)

    print(
        "DATA PREPARATION COMPLETE"
    )

    print("=" * 75)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()