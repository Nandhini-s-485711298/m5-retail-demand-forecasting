"""Generate small synthetic M5-shaped CSVs so the pipeline can be tested
without the real 450 MB download.  NOT for the final deliverable.

Run:  python -m src.make_demo_data
"""
import numpy as np
import pandas as pd

from src import config as C

RNG = np.random.default_rng(0)
N_DAYS = 1969
START = pd.Timestamp("2011-01-29")


def main() -> None:
    dates = pd.date_range(START, periods=N_DAYS, freq="D")
    cal = pd.DataFrame({
        "date": dates,
        "d": [f"d_{i+1}" for i in range(N_DAYS)],
        "weekday": dates.day_name(),
        "wday": dates.dayofweek + 1,
        "month": dates.month,
        "year": dates.year,
    })
    # Walmart weeks start on Saturday
    cal["wm_yr_wk"] = (cal["year"] * 100 +
                       ((dates - START).days // 7 % 52 + 1)).astype(int)
    ev = pd.Series(pd.NA, index=cal.index, dtype="object")
    ev[(cal["month"] == 12) & (dates.day == 25)] = "Christmas"
    ev[(cal["month"] == 7) & (dates.day == 4)] = "IndependenceDay"
    cal["event_name_1"] = ev
    cal["event_type_1"] = ev.notna().map({True: "National", False: pd.NA})
    cal["event_name_2"] = np.nan
    cal["event_type_2"] = np.nan
    for s in ["CA", "TX", "WI"]:
        cal[f"snap_{s}"] = (dates.day <= 10).astype(int)
    cal.to_csv(C.RAW / "calendar.csv", index=False)

    stores = ["CA_1", "CA_2", "TX_1", "WI_1"]
    depts = {"FOODS": ["FOODS_1", "FOODS_2"], "HOBBIES": ["HOBBIES_1"]}
    items = []
    for cat, dl in depts.items():
        for d in dl:
            for i in range(1, 26):
                items.append((f"{d}_{i:03d}", d, cat))

    rows, prices = [], []
    weeks = cal["wm_yr_wk"].unique()
    for store in stores:
        state = store.split("_")[0]
        for item_id, dept, cat in items:
            base = RNG.gamma(1.6, 1.4)
            dow = 1 + 0.28 * np.sin(np.arange(N_DAYS) * 2 * np.pi / 7)
            yr = 1 + 0.22 * np.sin(np.arange(N_DAYS) * 2 * np.pi / 365.25 - 1.1)
            trend = np.linspace(0.85, 1.15, N_DAYS)
            lam = base * dow * yr * trend
            sales = RNG.poisson(lam)
            if RNG.random() < 0.35:                     # intermittent series
                sales = sales * (RNG.random(N_DAYS) > 0.45)
            start = RNG.integers(0, 500)                # staggered release
            sales[:start] = 0
            rows.append([f"{item_id}_{store}_evaluation", item_id, dept, cat,
                         store, state, *sales.astype(int)])
            p = round(float(RNG.uniform(0.9, 9.5)), 2)
            for w in weeks[start // 7:]:
                prices.append([store, item_id, w,
                               round(p * RNG.uniform(0.95, 1.05), 2)])

    cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"] + \
           [f"d_{i+1}" for i in range(N_DAYS)]
    pd.DataFrame(rows, columns=cols).to_csv(
        C.RAW / "sales_train_evaluation.csv", index=False)
    pd.DataFrame(prices, columns=["store_id", "item_id", "wm_yr_wk",
                                  "sell_price"]).to_csv(
        C.RAW / "sell_prices.csv", index=False)
    print(f"demo data in {C.RAW}: {len(rows):,} series x {N_DAYS} days")


if __name__ == "__main__":
    main()
