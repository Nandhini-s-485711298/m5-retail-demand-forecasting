from pathlib import Path
from datetime import datetime
import io
import json

import pandas as pd
import numpy as np

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

WEB = ROOT / "web_data"

# Plain HTML/CSS/JS frontend:
# /app/frontend/index.html, app.js, style.css
FRONTEND = ROOT / "frontend"


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Walmart M5 Retail Demand Forecasting API",
    version="5.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# MODES
# ============================================================

MODES = {
    "validation",
    "evaluation",
}

DEFAULT_MODE = "evaluation"

_CACHE = {}


# ============================================================
# HELPERS
# ============================================================

def load_forecast(mode: str) -> pd.DataFrame:

    if mode not in MODES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid mode '{mode}'. "
                "Use validation or evaluation."
            ),
        )

    if mode in _CACHE:
        return _CACHE[mode]

    path = WEB / f"forecast_{mode}.parquet"

    if not path.exists():

        output_path = (
            ROOT
            / "outputs"
            / f"forecast_{mode}.parquet"
        )

        if output_path.exists():
            path = output_path
        else:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Forecast file not found for "
                    f"{mode}: {path}"
                ),
            )

    df = pd.read_parquet(path)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    if "forecast" in df.columns:

        df["forecast"] = pd.to_numeric(
            df["forecast"],
            errors="coerce",
        ).fillna(0)

        df["forecast"] = df["forecast"].clip(
            lower=0
        )

    _CACHE[mode] = df

    return df


def clean_value(value):
    """Convert pandas/numpy values into normal JSON-safe Python values."""
    if value is None:
        return None

    # pandas / numpy missing values
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()

    # numpy scalar -> native Python scalar
    if isinstance(value, np.generic):
        return value.item()

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    return value


def json_safe(value):
    """Recursively remove numpy/pandas objects before FastAPI serializes JSON."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    return value


def clean_records(df):

    return [
        {
            key: clean_value(value)
            for key, value in row.items()
        }
        for row in df.to_dict(orient="records")
    ]


def apply_filters(
    df,
    state="",
    store="",
    category="",
    department="",
    item="",
):

    filters = [
        ("state_id", state),
        ("store_id", store),
        ("cat_id", category),
        ("dept_id", department),
        ("item_id", item),
    ]

    result = df

    for column, value in filters:

        if value and column in result.columns:

            result = result[
                result[column].astype(str)
                == str(value)
            ]

    return result


def unique_values(df, column):

    if column not in df.columns:
        return []

    return sorted(
        str(x)
        for x in df[column]
        .dropna()
        .unique()
    )


def read_json(filename):

    candidates = [
        WEB / filename,
        ROOT / "outputs" / filename,
    ]

    for path in candidates:

        if path.exists():

            try:
                return json.loads(
                    path.read_text()
                )
            except Exception:
                return {}

    return {}


def find_file(filename):

    candidates = [
        WEB / filename,
        ROOT / "outputs" / filename,
    ]

    for path in candidates:

        if path.exists():
            return path

    return None


# ============================================================
# API ROOT
# IMPORTANT:
# "/" is reserved for the React frontend.
# ============================================================

@app.get("/api")
def api_root():

    return {
        "message":
            "Walmart M5 Retail Demand Forecasting API",

        "status":
            "running",

        "frontend":
            "HTML/CSS/JS frontend",

        "modes": [
            "validation",
            "evaluation",
        ],

        "default_mode":
            DEFAULT_MODE,

        "docs":
            "/docs",
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "backend": "FastAPI",
        "default_mode": DEFAULT_MODE,
    }


# ============================================================
# OVERVIEW
# ============================================================

@app.get("/overview")
def overview():

    df = load_forecast(DEFAULT_MODE)

    states = unique_values(
        df,
        "state_id",
    )

    stores = unique_values(
        df,
        "store_id",
    )

    categories = unique_values(
        df,
        "cat_id",
    )

    departments = unique_values(
        df,
        "dept_id",
    )

    item_count = (
        int(df["item_id"].nunique())
        if "item_id" in df.columns
        else 0
    )

    series_count = (
        int(df["id"].nunique())
        if "id" in df.columns
        else 0
    )

    return {

        "dataset": {

            "name":
                "Walmart M5 Forecasting",

            "series":
                series_count,

            "items":
                item_count,

            "stores":
                len(stores),

            "states":
                len(states),

            "categories":
                len(categories),

            "departments":
                len(departments),

            "history_days":
                1941,

            "hierarchy_levels":
                12,

            "horizon_days":
                28,

            "forecast_start":
                "2016-05-23",

            "forecast_end":
                "2016-06-19",
        },

        "states": states,

        "stores": stores,

        "categories": categories,

        "departments": departments,

        "model": {

            "algorithm":
                "LightGBM",

            "objective":
                "Tweedie",

            "features":
                38,

            "approach":
                "Item × Store forecasting with "
                "bottom-up hierarchical aggregation",
        },
    }


# ============================================================
# DASHBOARD
# ============================================================

@app.get("/dashboard")
def dashboard(

    mode: str = DEFAULT_MODE,

    state: str = "",
    store: str = "",
    category: str = "",
    department: str = "",
    item: str = "",
):

    df_full = load_forecast(mode)

    df = apply_filters(
        df_full,
        state,
        store,
        category,
        department,
        item,
    )

    if df.empty:

        return {

            "mode": mode,

            "first_date": None,

            "last_date": None,

            "total_units": 0,

            "series": 0,

            "avg_units_per_day": 0,

            "peak_day_units": 0,

            "peak_date": None,

            "daily": [],

            "by_category": [],

            "by_store": [],

            "by_state": [],

            "top_items": [],

            "filters": {

                "states":
                    unique_values(
                        df_full,
                        "state_id",
                    ),

                "stores": [],

                "categories":
                    unique_values(
                        df_full,
                        "cat_id",
                    ),

                "departments": [],

                "items": [],
            },
        }

    daily = (
        df
        .groupby(
            "date",
            as_index=False,
        )["forecast"]
        .sum()
        .sort_values("date")
    )

    peak_row = daily.loc[
        daily["forecast"].idxmax()
    ]

    def grouped(column, limit=20):

        if column not in df.columns:
            return []

        g = (
            df
            .groupby(
                column,
                observed=True,
                as_index=False,
            )["forecast"]
            .sum()
            .sort_values(
                "forecast",
                ascending=False,
            )
            .head(limit)
        )

        g.columns = [
            "name",
            "forecast",
        ]

        return clean_records(g)

    top_items = []

    if {
        "item_id",
        "store_id",
    }.issubset(df.columns):

        g = (
            df
            .groupby(
                [
                    "item_id",
                    "store_id",
                ],
                observed=True,
                as_index=False,
            )["forecast"]
            .sum()
            .sort_values(
                "forecast",
                ascending=False,
            )
            .head(20)
        )

        top_items = clean_records(g)

    return {

        "mode": mode,

        "first_date":
            clean_value(
                daily["date"].min()
            ),

        "last_date":
            clean_value(
                daily["date"].max()
            ),

        "total_units":
            float(
                df["forecast"].sum()
            ),

        "series":
            int(
                df["id"].nunique()
            )
            if "id" in df.columns
            else 0,

        "avg_units_per_day":
            float(
                daily["forecast"].mean()
            ),

        "peak_day_units":
            float(
                daily["forecast"].max()
            ),

        "peak_date":
            clean_value(
                peak_row["date"]
            ),

        "daily":
            clean_records(daily),

        "by_category":
            grouped("cat_id"),

        "by_store":
            grouped("store_id"),

        "by_state":
            grouped("state_id"),

        "top_items":
            top_items,

        "filters": {

            "states":
                unique_values(
                    df_full,
                    "state_id",
                ),

            "stores":
                unique_values(
                    apply_filters(
                        df_full,
                        state=state,
                    ),
                    "store_id",
                ),

            "categories":
                unique_values(
                    apply_filters(
                        df_full,
                        state=state,
                        store=store,
                    ),
                    "cat_id",
                ),

            "departments":
                unique_values(
                    apply_filters(
                        df_full,
                        state=state,
                        store=store,
                        category=category,
                    ),
                    "dept_id",
                ),

            "items":
                unique_values(
                    apply_filters(
                        df_full,
                        state=state,
                        store=store,
                        category=category,
                        department=department,
                    ),
                    "item_id",
                ),
        },
    }


# ============================================================
# HIERARCHY
# ============================================================

@app.get("/hierarchy")
def hierarchy(
    mode: str = DEFAULT_MODE,
):

    df = load_forecast(mode)

    levels = []

    total = float(
        df["forecast"].sum()
    )

    levels.append({

        "level": "L1 Total",

        "count": 1,

        "data": [
            {
                "name": "TOTAL",
                "forecast": total,
            }
        ],
    })

    definitions = [

        (
            "L2 State",
            ["state_id"],
        ),

        (
            "L3 Store",
            ["store_id"],
        ),

        (
            "L4 Category",
            ["cat_id"],
        ),

        (
            "L5 Department",
            ["dept_id"],
        ),

        (
            "L6 State × Category",
            ["state_id", "cat_id"],
        ),

        (
            "L7 State × Department",
            ["state_id", "dept_id"],
        ),

        (
            "L8 Store × Category",
            ["store_id", "cat_id"],
        ),

        (
            "L9 Store × Department",
            ["store_id", "dept_id"],
        ),

        (
            "L10 Item",
            ["item_id"],
        ),

        (
            "L11 Item × State",
            ["item_id", "state_id"],
        ),

        (
            "L12 Item × Store",
            ["item_id", "store_id"],
        ),
    ]

    for level_name, columns in definitions:

        if not all(
            c in df.columns
            for c in columns
        ):
            continue

        grouped = (
            df
            .groupby(
                columns,
                observed=True,
                as_index=False,
            )["forecast"]
            .sum()
            .sort_values(
                "forecast",
                ascending=False,
            )
        )

        count = len(grouped)

        display = grouped.head(30)

        rows = []

        for _, row in display.iterrows():

            name = " / ".join(
                str(row[c])
                for c in columns
            )

            rows.append({

                "name":
                    name,

                "forecast":
                    float(
                        row["forecast"]
                    ),
            })

        levels.append({

            "level":
                level_name,

            "count":
                count,

            "data":
                rows,
        })

    return {
        "mode": mode,
        "levels": levels,
    }


# ============================================================
# ACCURACY
# ============================================================

@app.get("/accuracy")
def accuracy(
    mode: str = "validation",
):

    result = {

        "mode": mode,

        "wrmsse": None,

        "rmse": None,

        "levels": [],

        "features": [],

        "message": "",
    }

    metrics_file = find_file(
        f"metrics_{mode}.json"
    )

    if metrics_file:

        try:

            data = json.loads(
                metrics_file.read_text()
            )

            result["wrmsse"] = data.get(
                "wrmsse"
            )

            result["rmse"] = data.get(
                "rmse",
                data.get("overall_rmse"),
            )

            result["levels"] = data.get(
                "levels",
                [],
            )

        except Exception:

            result["message"] = (
                f"Could not read metrics "
                f"file for {mode}."
            )

    else:

        result["message"] = (
            "Metrics file was not found for "
            f"{mode}."
        )

    importance_file = find_file(
        f"feature_importance_{mode}.csv"
    )

    if importance_file:

        fi = pd.read_csv(
            importance_file
        ).head(20)

        result["features"] = clean_records(
            fi
        )

    return result


# ============================================================
# EVENTS
# ============================================================

@app.get("/events")
def events(
    mode: str = DEFAULT_MODE,
):

    df = load_forecast(mode)

    daily = (
        df
        .groupby(
            "date",
            as_index=False,
        )["forecast"]
        .sum()
    )

    events_data = []

    snap_data = []

    calendar_path = find_file(
        "calendar.parquet"
    )

    if calendar_path:

        cal = pd.read_parquet(
            calendar_path
        )

        cal["date"] = pd.to_datetime(
            cal["date"]
        )

        merged = daily.merge(
            cal,
            on="date",
            how="left",
        )

        if "event_name_1" in merged.columns:

            event_rows = merged[
                merged["event_name_1"].notna()
            ]

            normal = merged.loc[
                merged["event_name_1"].isna(),
                "forecast",
            ].mean()

            for _, row in event_rows.iterrows():

                uplift = None

                if pd.notna(normal) and normal > 0:

                    uplift = float(
                        (
                            float(row["forecast"])
                            / float(normal)
                            - 1
                        ) * 100
                    )

                events_data.append({

                    "date":
                        clean_value(
                            row["date"]
                        ),

                    "event":
                        str(
                            row["event_name_1"]
                        ),

                    "type":
                        str(
                            row.get(
                                "event_type_1",
                                "",
                            )
                        ),

                    "forecast":
                        float(
                            row["forecast"]
                        ),

                    "uplift_pct":
                        uplift,
                })

        for state, column in [
            ("CA", "snap_CA"),
            ("TX", "snap_TX"),
            ("WI", "snap_WI"),
        ]:

            if column not in cal.columns:
                continue

            if "state_id" not in df.columns:
                continue

            state_data = (
                df[
                    df["state_id"].astype(str)
                    == state
                ]
                .groupby(
                    "date",
                    as_index=False,
                )["forecast"]
                .sum()
            )

            state_data = state_data.merge(
                cal[
                    ["date", column]
                ],
                on="date",
                how="left",
            )

            on = state_data.loc[
                state_data[column] == 1,
                "forecast",
            ].mean()

            off = state_data.loc[
                state_data[column] == 0,
                "forecast",
            ].mean()

            if (
                pd.notna(on)
                and pd.notna(off)
                and off > 0
            ):

                snap_data.append({

                    "state":
                        state,

                    "snap_day":
                        float(on),

                    "non_snap_day":
                        float(off),

                    "uplift_pct":
                        float(
                            (on / off - 1)
                            * 100
                        ),
                })

    dow = daily.copy()

    dow["day"] = dow[
        "date"
    ].dt.day_name()

    order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]

    profile = (
        dow
        .groupby("day")["forecast"]
        .mean()
        .reindex(order)
    )

    weekly = [

        {
            "day": day,
            "forecast": float(value),
        }

        for day, value
        in profile.items()

        if pd.notna(value)
    ]

    return json_safe({

        "mode": mode,

        "events":
            events_data,

        "snap":
            snap_data,

        "weekly_profile":
            weekly,
    })


# ============================================================
# SERIES SEARCH
# ============================================================

@app.get("/series/search")
def series_search(

    q: str = "",

    mode: str = DEFAULT_MODE,

    limit: int = Query(
        50,
        ge=1,
        le=500,
    ),
):

    df = load_forecast(mode)

    q = q.strip().lower()

    if not q:

        return {
            "matches": 0,
            "data": [],
        }

    mask = pd.Series(
        False,
        index=df.index,
    )

    searchable_columns = [

        "id",

        "item_id",

        "store_id",

        "state_id",

        "cat_id",

        "dept_id",
    ]

    for col in searchable_columns:

        if col in df.columns:

            mask |= (
                df[col]
                .astype(str)
                .str.lower()
                .str.contains(
                    q,
                    regex=False,
                )
            )

    hits = df[mask].copy()

    if hits.empty:

        return {
            "matches": 0,
            "data": [],
        }

    if "id" not in hits.columns:

        return {
            "matches": 0,
            "data": [],
        }

    result = (
        hits
        .groupby(
            "id",
            observed=True,
            as_index=False,
        )["forecast"]
        .agg(
            total="sum",
            average="mean",
            peak="max",
        )
        .sort_values(
            "total",
            ascending=False,
        )
        .head(limit)
    )

    result["id"] = result[
        "id"
    ].astype(str)

    return {

        "matches":
            int(
                hits["id"].nunique()
            ),

        "data":
            clean_records(result),
    }


# ============================================================
# SERIES DETAIL
# ============================================================

@app.get("/series/detail")
def series_detail(

    series_id: str,

    mode: str = DEFAULT_MODE,
):

    df = load_forecast(mode)

    if "id" not in df.columns:

        raise HTTPException(
            status_code=500,
            detail=(
                "Forecast data does not contain "
                "an 'id' column."
            ),
        )

    series_id = str(series_id)

    series = (
        df[
            df["id"].astype(str)
            == series_id
        ]
        .sort_values("date")
        .copy()
    )

    if series.empty:

        raise HTTPException(
            status_code=404,
            detail=(
                f"Series '{series_id}' "
                "was not found."
            ),
        )

    # --------------------------------------------------------
    # HISTORY
    # --------------------------------------------------------

    history = []

    history_file = find_file(
        "history.parquet"
    )

    calendar_file = find_file(
        "calendar.parquet"
    )

    if (
        history_file
        and calendar_file
    ):

        h = pd.read_parquet(
            history_file
        )

        if "id" in h.columns:

            row = h[
                h["id"].astype(str)
                == series_id
            ]

            if not row.empty:

                cal = pd.read_parquet(
                    calendar_file
                )

                if {
                    "d",
                    "date",
                }.issubset(cal.columns):

                    cal["date"] = pd.to_datetime(
                        cal["date"]
                    )

                    mapping = dict(
                        zip(
                            cal["d"].astype(str),
                            cal["date"],
                        )
                    )

                    r = row.iloc[0]

                    for column in row.columns:

                        column_str = str(column)

                        if (
                            column_str.startswith("d_")
                            and column_str in mapping
                        ):

                            try:

                                sales = float(
                                    r[column]
                                )

                            except Exception:

                                sales = 0.0

                            history.append({

                                "date":
                                    mapping[
                                        column_str
                                    ].isoformat(),

                                "sales":
                                    sales,
                            })

                    history = sorted(
                        history,
                        key=lambda x:
                            x["date"],
                    )[-120:]

    # --------------------------------------------------------
    # METADATA
    # --------------------------------------------------------

    metadata = {

        "item_id":
            clean_value(
                series["item_id"].iloc[0]
            )
            if "item_id" in series.columns
            else None,

        "store_id":
            clean_value(
                series["store_id"].iloc[0]
            )
            if "store_id" in series.columns
            else None,

        "state_id":
            clean_value(
                series["state_id"].iloc[0]
            )
            if "state_id" in series.columns
            else None,

        "cat_id":
            clean_value(
                series["cat_id"].iloc[0]
            )
            if "cat_id" in series.columns
            else None,

        "dept_id":
            clean_value(
                series["dept_id"].iloc[0]
            )
            if "dept_id" in series.columns
            else None,
    }

    return {

        "id":
            series_id,

        "forecast":
            clean_records(
                series[
                    [
                        "date",
                        "forecast",
                    ]
                ]
            ),

        "history":
            history,

        "total":
            float(
                series["forecast"].sum()
            ),

        "average":
            float(
                series["forecast"].mean()
            ),

        "peak":
            float(
                series["forecast"].max()
            ),

        "peak_date":
            clean_value(
                series.loc[
                    series["forecast"].idxmax(),
                    "date",
                ]
            ),

        "metadata":
            metadata,
    }


# ============================================================
# EXPORT
# ============================================================

@app.get("/export")
def export(

    mode: str = DEFAULT_MODE,

    state: str = "",

    store: str = "",

    category: str = "",

    department: str = "",

    item: str = "",

    fmt: str = "long",
):

    df = apply_filters(

        load_forecast(mode),

        state,

        store,

        category,

        department,

        item,
    )

    if fmt == "wide":

        if {
            "id",
            "horizon",
        }.issubset(df.columns):

            df = (
                df
                .pivot_table(
                    index="id",
                    columns="horizon",
                    values="forecast",
                )
                .reset_index()
            )

            df.columns = [

                (
                    "id"
                    if i == 0
                    else f"F{column}"
                )

                for i, column
                in enumerate(df.columns)
            ]

    buffer = io.StringIO()

    df.to_csv(
        buffer,
        index=False,
    )

    buffer.seek(0)

    return StreamingResponse(

        iter([
            buffer.getvalue()
        ]),

        media_type="text/csv",

        headers={

            "Content-Disposition":
                (
                    f'attachment; '
                    f'filename="m5_forecast_'
                    f'{mode}_{fmt}.csv"'
                ),
        },
    )


# ============================================================
# PAGINATED FORECAST
# ============================================================

@app.get("/forecast")
def forecast(

    mode: str = DEFAULT_MODE,

    limit: int = Query(
        500,
        ge=1,
        le=5000,
    ),

    offset: int = Query(
        0,
        ge=0,
    ),

    state: str = "",

    store: str = "",

    category: str = "",

    department: str = "",

    item: str = "",
):

    df = apply_filters(

        load_forecast(mode),

        state,

        store,

        category,

        department,

        item,
    )

    rows = df.iloc[
        offset:
        offset + limit
    ]

    return {

        "mode":
            mode,

        "total_rows":
            len(df),

        "offset":
            offset,

        "limit":
            limit,

        "data":
            clean_records(rows),
    }


# ============================================================
# FRONTEND
# IMPORTANT:
# This must be AFTER the API routes.
# "/" serves the HTML/CSS/JS frontend.
# ============================================================

if FRONTEND.exists():

    app.mount(

        "/",

        StaticFiles(
            directory=FRONTEND,
            html=True,
        ),

        name="frontend",
    )