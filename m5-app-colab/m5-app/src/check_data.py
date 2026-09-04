"""Sanity-check that the real Kaggle CSVs are in place and correct.

Run this FIRST, before anything else:   python -m src.check_data
"""
import sys
import pandas as pd

from src import config as C

EXPECTED = {
    "calendar.csv": 1969,
    "sell_prices.csv": None,
    "sales_train_validation.csv": 30490,
    "sales_train_evaluation.csv": 30490,
}
OK, BAD, WARN = "  [OK]  ", "  [FAIL]", "  [WARN]"


def main() -> int:
    print(f"\nChecking {C.RAW}\n" + "-" * 60)
    problems = 0

    files = sorted(p.name for p in C.RAW.glob("*.csv"))
    if not files:
        print(f"{BAD} No CSV files found.")
        print(f"\n  Copy your 5 Kaggle CSVs into:\n    {C.RAW}\n")
        return 1
    print("Files present:", ", ".join(files), "\n")

    # ---------------------------------------------------------------- calendar
    f = C.RAW / "calendar.csv"
    if not f.exists():
        print(f"{BAD} calendar.csv missing"); problems += 1
    else:
        cal = pd.read_csv(f)
        n = len(cal)
        print(f"{OK if n == 1969 else BAD} calendar.csv: {n} rows (expect 1969)")
        problems += n != 1969
        print(f"         date range {cal['date'].min()} -> {cal['date'].max()}")

    # ------------------------------------------------------------------ sales
    f = C.RAW / "sales_train_evaluation.csv"
    if not f.exists():
        print(f"{WARN} sales_train_evaluation.csv missing "
              f"(pipeline will fall back to the validation file)")
        f = C.RAW / "sales_train_validation.csv"
        if not f.exists():
            print(f"{BAD} sales_train_validation.csv also missing"); return 1

    sales = pd.read_csv(f, nrows=5)
    full = pd.read_csv(f, usecols=["id", "store_id", "item_id", "cat_id"])
    n_series, n_days = len(full), sum(c.startswith("d_") for c in sales.columns)
    last_d = max(int(c[2:]) for c in sales.columns if c.startswith("d_"))

    print(f"{OK if n_series == 30490 else WARN} {f.name}: "
          f"{n_series:,} series (expect 30,490)")
    print(f"         {n_days:,} day columns, last = d_{last_d}")
    print(f"         {full['store_id'].nunique()} stores (expect 10), "
          f"{full['item_id'].nunique():,} items (expect 3,049)")
    cats = sorted(full["cat_id"].unique())
    print(f"         categories: {cats}")
    if "HOUSEHOLD" not in cats:
        print(f"{BAD} No HOUSEHOLD category -> this is still the DEMO data!")
        problems += 1
    if n_series < 30490:
        print(f"{WARN} fewer series than the real dataset "
              f"-> demo data, or a filtered file")

    # ----------------------------------------------------------------- prices
    f = C.RAW / "sell_prices.csv"
    if not f.exists():
        print(f"{BAD} sell_prices.csv missing"); problems += 1
    else:
        size = f.stat().st_size / 1e6
        print(f"{OK if size > 100 else WARN} sell_prices.csv: {size:,.0f} MB "
              f"(real file is ~200 MB)")

    print("-" * 60)
    if problems:
        print(f"\n{problems} problem(s) found. Fix these before running the "
              f"pipeline.\n")
        return 1

    train_end, first_pred = C.MODES[C.MODE]
    print(f"\nAll good. Config: MODE={C.MODE}  STORES={C.STORES}")
    print(f"Will train on days <= d_{train_end} and predict "
          f"d_{first_pred}..d_{first_pred + C.HORIZON - 1}")
    print("\nNext:  python -m src.prepare_data\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
