# Retail Demand Forecasting — Walmart M5 (Hierarchical)

A 28-day-ahead item×store demand forecasting system with a web dashboard.
Built for the Cognizant use case #11.

| Objective (from the brief) | How this project does it |
|---|---|
| Handles hierarchical aggregation | Forecasts at item×store (L12), sums bottom-up to all 12 M5 levels; scored at every level |
| External covariates (price / promo / holiday) | `sell_price`, weekly price change, price vs dept average, `calendar.csv` event flags, SNAP flags |
| Intermittent (sparse / zero-inflated) demand | LightGBM with **Tweedie** objective + `zero_rate_28` feature + forecasts clipped at 0 |
| Accurate 28-day-ahead per store/item | Direct multi-horizon (all lags ≥ 28 days), scored with **WRMSSE** |

---

## STEP 0 — Put the data in place

You already downloaded the 5 CSVs. Copy them here:

```
m5-app/data/raw/
├── calendar.csv
├── sales_train_validation.csv
├── sales_train_evaluation.csv
├── sell_prices.csv
└── sample_submission.csv
```

> ⚠️ `data/raw/` currently holds **synthetic demo data** I generated to test the
> pipeline. **Delete those 3 files first**, then copy your real Kaggle CSVs in.

```bash
rm m5-app/data/raw/*.csv
cp /path/to/kaggle/*.csv m5-app/data/raw/
```

## STEP 1 — Install

```bash
cd m5-app
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## STEP 2 — Run the pipeline (4 commands, in order)

```bash
python -m src.prepare_data   # CSVs -> one tidy parquet table      (~5 min)
python -m src.features       # build lag/rolling/price features    (~10 min)
python -m src.train          # train LightGBM + predict 28 days    (~20 min)
python -m src.evaluate       # WRMSSE across all 12 levels         (~3 min)
```

**Low on RAM?** Open `src/config.py` and set `STORES = ["CA_1"]` for a first
run — finishes in ~3 minutes and proves everything works. Then set it back to
`None` for the full 30,490 series.

## STEP 3 — Launch the dashboard

```bash
streamlit run app.py
```

Opens at http://localhost:8501 with five tabs:

- **Forecast** — total daily demand curve, breakdown by category/store, top items
- **Hierarchy** — pick any of the 12 M5 aggregation levels and compare nodes
- **Accuracy** — WRMSSE per level + feature importance
- **Series explorer** — history vs forecast for any single item×store
- **Export** — download CSV (wide Kaggle format or long format)

---

## Which 28 days does it forecast?

`src/config.py` → `MODE`:

| MODE | Trains on | Predicts | Ground truth? |
|---|---|---|---|
| `validation` | ≤ d_1913 | d_1914–1941 (25 Apr – 22 May 2016) | ✅ yes |
| `evaluation` | ≤ d_1941 | **d_1942–1969 (23 May – 19 Jun 2016)** | ✅ yes — **default** |
| `future` | ≤ d_1969 | d_1970–1997 (**20 Jun – 17 Jul 2016**) | ❌ no |

**About your handwritten note:** the window you picked (20/06/2016 → 17/07/2016)
sits *after* all labelled M5 data, which ends 19 Jun 2016. You can produce those
numbers — that's what `MODE = "future"` does — but nobody can score them.

So do both, in this order:
1. Run with `MODE = "evaluation"` → report a real WRMSSE. **This is your proof the model works.**
2. Change to `MODE = "future"`, re-run the 4 commands → produces the
   20 Jun – 17 Jul forecast for the demo.

Both appear in the dashboard's mode dropdown.

**Also on your note:** the idea of keeping only 20/6→17/7 from each year would
throw away 93% of the data, including the recent weeks that drive short-horizon
accuracy. That instinct — yearly seasonality matters — is captured properly
here as *features* instead: `lag_364`, `weekofyear`, `month`, and event flags.
Same idea, full data.

---

## Project layout

```
m5-app/
├── app.py                    # Streamlit dashboard
├── requirements.txt
├── src/
│   ├── config.py             # ← all settings live here
│   ├── prepare_data.py       # Step 1
│   ├── features.py           # Step 2
│   ├── train.py              # Step 3
│   ├── evaluate.py           # Step 4 (WRMSSE, 12 levels)
│   └── make_demo_data.py     # synthetic data for testing only
├── data/raw/                 # ← your 5 Kaggle CSVs go here
├── data/processed/           # generated parquet
└── outputs/                  # model, forecasts, metrics, submission.csv
```

## How the model works (for your presentation)

**Approach:** direct multi-horizon gradient boosting on the bottom level,
reconciled bottom-up.

1. **Reshape** — 30,490 series × 1,941 days → long format (one row per
   item-store-day), joined to calendar and weekly prices.
2. **Features** (~35):
   - *Lags* 28, 29, 30, 35, 42, 56, **364** (same day last year)
   - *Rolling* mean over 7/14/28/60/180 days, std over 7/28 — all shifted 28
     days so no future information leaks
   - *Intermittency* — `zero_rate_28`, the fraction of zero-sale days
   - *Price* — level, weekly % change, price vs the item's own average, price
     vs department average (captures promotions)
   - *Calendar* — day-of-week, day-of-month, week-of-year, month, weekend flag,
     event name/type, SNAP benefit day, days since release
3. **Model** — LightGBM, `objective="tweedie"` (`variance_power=1.1`). Tweedie
   is the right choice for retail: it models a distribution with a point mass at
   zero plus a continuous positive part, which is exactly zero-inflated demand.
   Squared-error loss would systematically over-forecast sparse items.
4. **Why lags ≥ 28** — every one of the 28 forecast days can be predicted in a
   single pass. No recursive feeding of predictions back in, so no error
   compounding and it runs in seconds.
5. **Validation** — last 28 days of history held out, early stopping on it.
6. **Reconciliation** — bottom-level forecasts summed up the hierarchy, so all
   12 levels agree by construction.
7. **Metric** — WRMSSE: RMSSE per series, weighted by each series' share of the
   last 28 days' revenue, averaged over the 12 levels. This is the official M5
   metric — using it is what makes your result comparable to the leaderboard.

## Improving the score (if you have time)

- Train one model **per store** (10 models) — usually beats a single global model
- Add `lag_1..lag_7` and switch to recursive prediction for the first week
- Ensemble Tweedie + Poisson objectives
- Multiply the final forecasts by ~0.97 (a known M5 trick — the loss is
  asymmetric and slight under-forecasting scores better)
- Add day-of-week × store interaction features
