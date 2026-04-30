# NSE Next-Day Price Predictor

Retail investors on the Nairobi Securities Exchange often place limit orders based on gut feel or broker tips. This project asks a simpler question: **can historical price data alone tell us where a stock is likely to trade tomorrow?**

We built a machine learning system that forecasts the next trading day's closing price for ten blue-chip NSE stocks. The goal is to give retail investors a data-driven anchor when setting bid and offer prices, rather than guessing.

---

## The Problem

When you place a limit order on the NSE, you have to commit to a price before the market opens. Set it too high and you overpay. Set it too low and your order never fills. Most retail investors have no systematic way to estimate where a stock will trade the next day; they rely on yesterday's close, a hunch, or a broker's opinion.

This project gives them something better: a model-backed estimate of tomorrow's closing price, trained on four years of actual NSE data.

---

## Data

We used daily OHLCV (Open, High, Low, Close, Volume) data for all NSE-listed stocks from **January 2022 to October 2025**, downloaded from the NSE website.

| Item           | Detail                    |
| -------------- | ------------------------- |
| Raw rows       | 67,313                    |
| Stocks covered | 79                        |
| Date range     | 3 Jan 2022 → 31 Oct 2025  |
| After cleaning | 63,059 rows, zero nulls   |
| Focus stocks   | 10 blue chips (see below) |

**Blue chip stocks used:**
EQTY, KCB, SCOM, COOP, SCBK, ABSA, KNRE, KEGN, KPLC, BRIT

These ten were chosen because they have the most complete trading history, the lowest proportion of zero-volume days, and are the stocks most retail investors actually trade.

---

## Data Cleaning

The raw NSE data came in inconsistent formats across years; mixed date formats, dashes where numbers should be, percentage symbols in numeric columns, and missing values.

Key cleaning steps:

- Standardised all column names to snake_case
- Converted price and volume columns from text to numeric, replacing dashes with NaN
- Dropped 4,244 rows where core price data was missing entirely
- Forward-filled then backward-filled the 52-week high/low columns per stock
- Filled missing volume with zero (no trade recorded = zero volume)
- Removed 10 rows with logically impossible prices (e.g. day_price outside the day_low–day_high band)
- Removed duplicate stock-date combinations

---

## Exploratory Findings

A few things the data told us before we built any model:

**Prices are highly autocorrelated.** Yesterday's price correlates with today's at 0.999. This means a naive "predict tomorrow = today" strategy is genuinely hard to beat; and we used it as our primary baseline.

**Nearly 30% of trading days had zero volume.** The NSE has many illiquid stocks. Among blue chips this is less severe, but KNRE and KPLC still had meaningful gaps.

**Blue chip returns have fat tails.** All ten stocks showed kurtosis well above 3, meaning extreme single-day moves happen more often than a normal distribution would predict. KPLC and KEGN had dramatic price jumps in 2025 (1.5 → 9.3 KES and 2.6 → 6.3 KES respectively), likely driven by regulatory changes in the energy sector.

**Volatility is concentrated in small stocks.** The most volatile stocks by daily return standard deviation (SMER, NBV, TCL) are all illiquid small-caps. Among blue chips, KNRE and KPLC are the most volatile; BRIT and ABSA the calmest.

---

## Feature Engineering

All features are constructed from **past data only** - nothing from the day being predicted leaks into the inputs. Every lag and rolling calculation uses `.shift(1)` to ensure this.

Features created:

| Feature                     | Description                                                             |
| --------------------------- | ----------------------------------------------------------------------- |
| `price_lag1/2/3`            | Previous 1, 2, 3 days' closing price                                    |
| `low_lag1/2`, `high_lag1/2` | Previous days' intraday low and high                                    |
| `volume_lag1`               | Previous day's volume                                                   |
| `price_roll5_mean`          | 5-day rolling average price                                             |
| `price_roll20_mean`         | 20-day rolling average price                                            |
| `price_roll5_std`           | 5-day rolling price standard deviation                                  |
| `price_ewm5`, `price_ewm20` | Exponentially weighted means (span 5 and 20)                            |
| `price_pos_52w`             | Where today's price sits in its 52-week range (0 = at low, 1 = at high) |
| `volume_ratio`              | Today's volume relative to 20-day average                               |
| `dow_num`                   | Day of week (0 = Monday, 4 = Friday)                                    |
| `range_lag1`                | Yesterday's intraday high–low spread (volatility proxy)                 |

**Target variable:** `next_day_price` - tomorrow's closing price, created using `.shift(-1)` per stock.

---

## Train / Validation / Test Split

We split the data strictly by time. No random splitting; that would leak future prices into training.

| Split      | Period              | Rows  |
| ---------- | ------------------- | ----- |
| Train      | Jan 2022 – Dec 2023 | 4,853 |
| Validation | Jan 2024 – Dec 2024 | 2,490 |
| Test       | Jan 2025 – Oct 2025 | 2,060 |

All model selection and tuning decisions were made using validation results only. The test set was evaluated once, at the end.

---

## Models

### Baseline 1 - Persistence

Predict tomorrow's price = today's price. No learning involved. This is the hardest baseline to beat in financial forecasting because prices are so autocorrelated.

**Validation MAE: 0.511 KES | MAPE: 2.13%**

### Baseline 2 - Rolling Mean & Exponential Weighted Mean

Predict tomorrow using a 5-day rolling average or EWM. The EWM gives more weight to recent days.

| Method        | MAE   | MAPE  |
| ------------- | ----- | ----- |
| Rolling 5-day | 0.712 | 2.82% |
| EWM span=5    | 0.673 | 2.60% |

Both are worse than persistence. The 5-day average smooths out the most recent price information, which is actually the most predictive signal.

### Model 1 - Ridge Regression

Linear regression with L2 regularisation (alpha=100) to handle the high correlation between lag features. Trained on all 17 engineered features.

**Validation MAE: 0.529 KES | MAPE: 2.62%**

The dominant coefficients were `low_lag1` (0.44), `high_lag1` (0.43), and `price_lag1` (0.31) - the model is essentially estimating tomorrow's price as a weighted average of yesterday's trading range. It does not meaningfully improve on persistence.

### Model 2 - ARIMA (per stock)

We ran the Augmented Dickey-Fuller test on each stock's price series to check for stationarity. Three stocks (ABSA, COOP, KPLC) were stationary at the 5% level; the remaining seven required first differencing.

We then selected the best ARIMA order per stock using AIC grid search over p ∈ {0,1,2} and q ∈ {0,1,2}.

| Stock | Best Order   | AIC     |
| ----- | ------------ | ------- |
| ABSA  | ARIMA(1,0,2) | −388.9  |
| BRIT  | ARIMA(0,1,1) | −615.5  |
| EQTY  | ARIMA(1,1,0) | 969.0   |
| KCB   | ARIMA(0,1,1) | 767.3   |
| KEGN  | ARIMA(2,1,2) | −1850.2 |

**Average validation MAE across blue chips: 4.77 KES | MAPE: 21.7%**

ARIMA performs poorly in this setting. It is univariate i.e, it ignores volume, the 52-week position, and all the other features we built. It also has no mechanism to handle the structural price jumps seen in KPLC and KEGN, leading to large sustained errors.

### Model 3 - XGBoost (Walk-Forward)

Walk-forward XGBoost trains on all available history up to day t, predicts day t+1, then expands the window by one and repeats. This is the most realistic evaluation setup for a sequential forecasting problem.

Results per stock (validation):

| Stock | MAE   | vs Persistence | Dir. Accuracy |
| ----- | ----- | -------------- | ------------- |
| ABSA  | 0.135 | −0.008         | 38.4%         |
| EQTY  | 0.411 | −0.022         | 35.9%         |
| KCB   | 0.416 | −0.027         | 39.4%         |
| KEGN  | 0.053 | competitive    | 44.2%         |

Directional accuracy (whether we correctly predict up vs down) ranges from 36–50%; barely better than a coin flip. This is expected: predicting the _direction_ of a daily return is much harder than predicting the price level.

We also extended XGBoost to predict a **price range** (next-day low and high) per stock. Coverage — the proportion of days where the actual price fell inside the predicted range:

| Stock | Coverage |
| ----- | -------- |
| ABSA  | 83.2%    |
| COOP  | 81.7%    |
| EQTY  | 78.1%    |

A coverage above 80% means the model's predicted band contains the actual price on more than 4 out of 5 trading days, which is useful for limit order placement.

### Model 4 - LightGBM

LightGBM trained on all 50 engineered features, predicting next-day `day_price` across all blue chips pooled together.

| Set        | MAE       | MAPE  |
| ---------- | --------- | ----- |
| Validation | 3.45 KES  | —     |
| Test       | 13.16 KES | 7.13% |

The large gap between validation and test MAE signals that LightGBM overfit to 2024 patterns and did not generalise well to 2025. The structural price changes in KPLC and KEGN in 2025 are likely responsible — no model trained on 2022–2023 data could anticipate a 4× price jump.

---

## Results Summary

| Model                | Validation MAE (KES) | Notes                     |
| -------------------- | -------------------- | ------------------------- |
| Persistence          | **0.511**            | Hardest to beat           |
| Rolling 5-day        | 0.712                | Worse than persistence    |
| EWM span=5           | 0.673                | Worse than persistence    |
| Ridge Regression     | 0.529                | Marginal, not significant |
| ARIMA                | 4.767                | Poor, ignores features    |
| XGBoost Walk-Forward | ~0.14–0.42           | Competitive per stock     |
| LightGBM             | 3.449                | Overfit to 2024           |

**The persistence baseline is the most robust single predictor of next-day price.** This is consistent with the Efficient Market Hypothesis in its weak form: publicly available historical price information is already reflected in current prices, leaving little systematic signal to exploit at a daily horizon.

The XGBoost walk-forward approach, evaluated correctly per stock, is the most competitive machine learning model; particularly for lower-priced, more volatile stocks like KEGN and KNRE where there is more exploitable short-term structure.

---

## Conclusion

Daily NSE stock prices are difficult to predict beyond the persistence baseline. The most honest finding from this project is that a model saying "tomorrow's price equals today's price" is genuinely hard to improve upon using standard machine learning approaches on OHLCV data alone.

This confirms something important: retail investors who anchor their limit orders to the previous day's closing price are using a near-optimal heuristic given publicly available information. What machine learning adds is the ability to quantify **how far prices typically move** from that anchor (via range prediction and error distributions), giving investors a principled basis for how wide to set their bid-offer spread.

---

## Repository Structure

```
├── NSE_Project.ipynb        # Main notebook — cleaning, EDA, models
├── app.py                   # Streamlit web application
├── nse_models/
│   ├── lgbm_day_price.pkl   # Trained LightGBM model
│   ├── feature_cols.pkl     # Feature column list
│   └── nse_features.csv     # Processed feature data
├── nse-2022-2025-data/      # Raw CSV files (one per year)
└── requirements.txt
```

---

## Running the Project

**Install dependencies:**

```bash
pip install -r requirements.txt
```

**Run the notebook:**
Open `NSE_Project.ipynb` in Jupyter and run all cells in order. The raw data folder `nse-2022-2025-data/` must be present.

**Run the Streamlit app:**

```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser. Select a blue chip stock and a date to see the next-day price forecast.

---

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
lightgbm
xgboost
statsmodels
prophet
streamlit
joblib
```

---

## Team

Class project — Nairobi, 2025.
Data source: Nairobi Securities Exchange (NSE) daily market reports, 2022–2025.
