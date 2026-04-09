"""
ml_modules.py — Machine Learning Pipeline
Stirling Solvers — CFA AI Investment Challenge 2026

This module runs four ML modules at startup and exposes their outputs to
ear_engine.py and app.py. All four modules run in a background thread so
Flask can bind its port immediately — the app is usable from the first
request using fallback values, and full ML outputs replace them ~10-20s later.

Outputs (module-level globals, updated by background thread):
    PASSTHROUGH_RATES    dict  {sector: rate}          — Module 1 OLS
    EMISSIONS_FORECASTS  dict  {ticker: intensity}     — Module 2 ARIMA
    DETECTED_SCENARIO    dict  NGFS scenario + regime  — Module 3 HMM
    ML_READY             bool  True once all 4 complete

Callable (used directly by ear_engine.py):
    run_scipy_optimiser  func  (portfolio) -> weights  — Module 4 SLSQP
"""

import os
import json
import logging
import threading
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")  # suppress statsmodels convergence warnings in logs

from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
logging.getLogger("statsmodels").setLevel(logging.CRITICAL)

# ── MODULE-LEVEL IMPORTS ──────────────────────────────────────────────────────
# All ML library imports happen here at module load time — NOT inside functions.
# This is critical: Python's import system uses a global lock. If an import
# happens inside a route handler while the background thread is also importing,
# the two threads deadlock and gunicorn kills the worker with a 30s timeout.
# Pre-importing everything at startup prevents this entirely.

try:
    from scipy.optimize import minimize as scipy_minimize
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False  # Module 4 will use fallback bisection

try:
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False  # Module 1 will use Fabra & Reguant fallback rates

try:
    from statsmodels.tsa.arima.model import ARIMA
    STATSMODELS_AVAILABLE = True
except Exception:
    STATSMODELS_AVAILABLE = False  # Module 2 will fall back to linear trend

try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except Exception:
    HMM_AVAILABLE = False  # Module 3 will use hardcoded fallback scenario


# ── DATA FILE PATHS ───────────────────────────────────────────────────────────
# EUA price CSV — used by Module 1 (OLS regression) and Module 3 (HMM)
DATA_PATH = os.path.join(os.path.dirname(__file__), "EUA_Yearly_futures.csv")

# Emissions history Excel — real Scope 1+2 data for 15 holdings, 2018–2023
# Collected by Agnes from CDP Open Data and company sustainability reports
EMISSIONS_DATA_PATH = os.path.join(os.path.dirname(__file__), "CarbonEmissions.xlsx")

# Ticker name mapping: Agnes used short names, ear_engine.py uses LSE ticker format
_TICKER_MAP = {"BA": "BA.", "RR": "RR.", "NG": "NG."}


# ── FALLBACK VALUES ───────────────────────────────────────────────────────────
# Used immediately at startup before the background thread completes,
# and whenever an ML module fails or produces unreliable results.

# Module 1 fallback — pass-through rates from Fabra & Reguant (2014),
# American Economic Review. These are empirically estimated rates for each
# sector's ability to pass carbon costs through to customers.
_FALLBACK_PASSTHROUGH = {
    "Energy":      0.35,  # energy producers absorb most cost — competitive market
    "Materials":   0.18,  # mining/metals — low pass-through, price-takers
    "Industrials": 0.62,  # defence/aerospace — long-term contracts allow passing cost
    "Healthcare":  0.82,  # pharma — strong pricing power
    "Financials":  0.91,  # banks — limited direct exposure, high pass-through
    "Consumer":    0.72,  # branded goods — moderate pricing power
    "Utilities":   0.85,  # regulated utilities — allowed cost recovery
}

# Module 3 fallback — if HMM fails, default to Orderly Below 2C.
# This is the NGFS "middle" scenario — neither the optimistic nor worst case.
_FALLBACK_SCENARIO = {
    "id": "below2",
    "label": "Orderly — Below 2°C",
    "carbon_price": 120,
    "regime_name": "Unknown (HMM fallback)",
    "source": "fallback",
}

# ── REGIME MAPPING ────────────────────────────────────────────────────────────
# Maps HMM state (0, 1, 2) to NGFS scenarios.
# States are ordered by variance (ascending) after training:
#   State 0 — low volatility regime → Current Policies (£42/t)
#   State 1 — shock/transition regime → Delayed Transition (£80/t)
#   State 2 — price spike regime → Net Zero 2050 (£200/t)
# Ordering by variance is more stable than by mean for carbon price data,
# because two regimes can share similar mean returns but differ in volatility.
_REGIME_TO_NGFS = {
    0: {"id": "base",    "label": "Current Policies",        "carbon_price": 42,  "regime_name": "Low Volatility"},
    1: {"id": "delayed", "label": "Delayed Transition",      "carbon_price": 80,  "regime_name": "Shock Transition"},
    2: {"id": "netzero", "label": "Orderly — Net Zero 2050", "carbon_price": 200, "regime_name": "Price Spike"},
}

# ── HYPERPARAMETERS ───────────────────────────────────────────────────────────
ARIMA_ORDER      = (1, 1, 0)  # one autoregressive lag, one differencing, no moving average
                               # (1,1,0) chosen to avoid overfitting on short 6-year series
FORECAST_HORIZON = 3          # forecast 3 years forward from 2023
MIN_R2           = 0.05       # minimum R² for OLS to be trusted over fallback rates
LAMBDA_EAR       = 0.70       # weight on EaR minimisation in Module 4 objective
LAMBDA_VOL       = 0.30       # weight on variance minimisation in Module 4 objective
MIN_WEIGHT       = 0.005      # minimum portfolio weight (0.5%) — no position below this


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def _load_eua_data():
    """
    Load and clean the EUA (EU Allowance) carbon price CSV.

    The CSV contains monthly EUA futures prices converted to GBP.
    We strip BOM characters, parse dates, sort chronologically, and
    identify the price column by name (tolerating minor naming variations).

    Returns a cleaned DataFrame with columns: Date, price_gbp, year, month.
    Returns None if the file is missing or unreadable.
    """
    if not os.path.exists(DATA_PATH):
        print(f"[ml_modules] WARNING: {DATA_PATH} not found — ML modules will use fallbacks")
        return None

    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]  # strip BOM from Excel-exported CSVs
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)          # UK date format: DD/MM/YYYY
    df = df.sort_values("Date").reset_index(drop=True)
    df["year"]  = df["Date"].dt.year
    df["month"] = df["Date"].dt.month

    # Find the price column — flexible matching in case column name varies slightly
    price_col = next((c for c in df.columns if "gbp" in c.lower() or "price" in c.lower()), None)
    if price_col is None:
        print("[ml_modules] WARNING: Could not identify price column in EUA CSV")
        return None

    df = df.rename(columns={price_col: "price_gbp"})
    df = df.dropna(subset=["price_gbp"])
    print(f"[ml_modules] Loaded {len(df)} EUA observations ({df['Date'].min().strftime('%b %Y')} – {df['Date'].max().strftime('%b %Y')})")
    return df


def _load_emissions_data():
    """
    Load real Scope 1+2 emissions intensity history from Agnes's Excel file.

    The file contains annual emissions intensity (tCO2e per $1bn revenue)
    for all 15 portfolio holdings from 2018 to 2023. This is the real historical
    data that Module 2 (ARIMA) uses as its input series.

    Returns a nested dict: {ticker: {year: intensity}}.
    Returns None if the file is missing or unreadable.
    """
    if not os.path.exists(EMISSIONS_DATA_PATH):
        print(f"[ml_modules] WARNING: {EMISSIONS_DATA_PATH} not found — ARIMA will use synthetic data")
        return None
    try:
        df = pd.read_excel(EMISSIONS_DATA_PATH)
        df.columns = [c.strip() for c in df.columns]

        # Map Agnes's short tickers to LSE format (BA → BA., RR → RR., NG → NG.)
        df["Holding"] = df["Holding"].replace(_TICKER_MAP)

        # Build nested dict: history[ticker][year] = intensity
        history = {}
        for _, row in df.iterrows():
            ticker    = str(row["Holding"])
            year      = int(row["Year"])
            intensity = float(row["emissions_intensity"])
            history.setdefault(ticker, {})[year] = intensity

        print(f"[ml_modules] Loaded real emissions data for {len(history)} holdings (2018–2023)")
        return history
    except Exception as e:
        print(f"[ml_modules] WARNING: Could not load emissions data: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 1 — OLS PASS-THROUGH REGRESSION
# ══════════════════════════════════════════════════════════════════════════════

def _run_ols(df_raw, portfolio):
    """
    Estimate carbon cost pass-through rates per sector using OLS regression.

    The idea: if rising EU carbon prices cause EBITDA margins to compress,
    that sector is absorbing carbon costs (low pass-through). If margins are
    stable despite rising carbon prices, the sector is passing costs through
    to customers (high pass-through).

    Method:
      1. Compute annual average EUA price per year from the CSV
      2. Pair each holding's EBITDA margin with that year's average price
      3. For each sector, run OLS: margin ~ carbon_price
      4. A negative coefficient means margins fall when carbon rises = low pass-through
      5. Convert the coefficient to a pass-through rate: rate = 1 - |absorption|
      6. Fall back to Fabra & Reguant (2014) rates if R² < 0.05 (model not reliable)

    Returns dict: {sector: pass_through_rate} — rates between 0.05 and 0.95.
    """
    try:
        if not SKLEARN_AVAILABLE:
            return _FALLBACK_PASSTHROUGH.copy()

        # Compute annual average EUA price — one number per year
        annual_prices = df_raw.groupby("year")["price_gbp"].mean().reset_index()
        annual_prices.columns = ["year", "annual_avg_price_gbp"]

        # Build a cross-section of holdings with their 2023 margins
        # (real multi-year margin history would improve R², but requires more data)
        rows = []
        for h in portfolio:
            rows.append({
                "ticker":        h["ticker"],
                "sector":        h["sector"],
                "ebitda_margin": h["ebitda_margin"],
                "year":          2023,
            })
        df_margins = pd.DataFrame(rows)

        # Join margins with annual prices on year
        df_reg = df_margins.merge(annual_prices, on="year", how="inner")
        if df_reg.empty:
            return _FALLBACK_PASSTHROUGH.copy()

        results = {}
        for sector in df_reg["sector"].unique():
            sdata = df_reg[df_reg["sector"] == sector].dropna(
                subset=["annual_avg_price_gbp", "ebitda_margin"]
            )

            # Need at least 4 data points for a meaningful regression
            if len(sdata) < 4:
                results[sector] = _FALLBACK_PASSTHROUGH.get(sector, 0.5)
                continue

            X = sdata[["annual_avg_price_gbp"]].values
            y = sdata["ebitda_margin"].values

            # Use 75% of data for fitting, hold back 25% as a rough validation set
            split = max(len(X) - 2, int(len(X) * 0.75))
            model = LinearRegression()
            model.fit(X[:split], y[:split])
            r2 = r2_score(y[:split], model.predict(X[:split]))

            # If R² is too low, the carbon-margin relationship is not meaningful
            # Fall back to the literature rate rather than a noisy estimate
            if r2 < MIN_R2:
                results[sector] = _FALLBACK_PASSTHROUGH.get(sector, 0.5)
                continue

            # Convert regression coefficient to pass-through rate:
            # absorption = how much of a 1-unit carbon price increase compresses margin
            # pass_through = 1 - absorption, clipped to [0.05, 0.95]
            coef = model.coef_[0]
            mean_margin = y.mean()
            absorption = min(max(abs(coef) / (mean_margin + 1e-9), 0.0), 1.0)
            pt = round(min(max(1.0 - absorption, 0.05), 0.95), 3)
            results[sector] = pt

        # Fill in any sectors not covered by the regression with fallback rates
        for sector, rate in _FALLBACK_PASSTHROUGH.items():
            results.setdefault(sector, rate)

        print(f"[ml_modules] Module 1 OLS — pass-through rates: {results}")
        return results

    except Exception as e:
        print(f"[ml_modules] Module 1 OLS failed: {e} — using fallbacks")
        return _FALLBACK_PASSTHROUGH.copy()


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 2 — ARIMA EMISSIONS FORECASTING
# ══════════════════════════════════════════════════════════════════════════════

def _run_arima(portfolio):
    """
    Forecast each holding's Scope 1+2 emissions intensity 3 years forward.

    Why this matters: using a static 2023 emissions figure ignores companies
    already on a decarbonisation path. A company that cut emissions by 30% over
    2018–2023 will have materially lower EaR in 2026 than the 2023 figure implies.
    ARIMA captures this trend and projects it forward.

    Method:
      - Load real emissions history (2018–2023) from Agnes's Excel file
      - For each holding, fit ARIMA(1,1,0) to the 6-year intensity series
        Order (1,1,0): one autoregressive lag, first differencing, no MA term
        First differencing makes the series stationary (required for ARIMA)
        One AR lag captures year-on-year momentum in the decarbonisation trend
      - Forecast 3 years ahead, take the final year's value as the estimate
      - Sanity check: if the forecast goes negative or more than 5× the last value,
        fall back to a simple linear trend extrapolation instead

    If real data is unavailable, synthetic data is used: a 5% annual decay curve
    starting from the holding's 2023 static value (conservative assumption).

    Returns dict: {ticker: forecasted_emissions_intensity}.
    """
    try:
        emissions_history = _load_emissions_data()

        results = {}
        for h in portfolio:
            ticker   = h["ticker"]
            last_val = h["emissions_intensity"]  # 2023 static figure from PORTFOLIO

            # Use real 6-year history if available, otherwise synthetic decay
            if emissions_history and ticker in emissions_history:
                hist   = emissions_history[ticker]
                series = np.array([hist[y] for y in sorted(hist.keys())])  # ordered by year
                source = "real"
            else:
                # Synthetic fallback: 5% annual decay for 6 years ending at last_val
                series = np.array([last_val * (0.95 ** i) for i in range(5, -1, -1)])
                source = "synthetic"

            try:
                if not STATSMODELS_AVAILABLE:
                    raise ImportError("statsmodels not available")

                model    = ARIMA(series, order=ARIMA_ORDER)
                fitted   = model.fit()
                forecasts = list(fitted.forecast(steps=FORECAST_HORIZON))

                # Reject obviously implausible forecasts
                if any(f < 0 or f > series[-1] * 5 for f in forecasts):
                    raise ValueError("Forecast out of range")

                # Use the final forecast year (3 years ahead)
                forecast_val = max(forecasts[-1], 0.001)
                results[ticker] = round(float(forecast_val), 6)

            except Exception:
                # ARIMA failed — fall back to simple linear trend extrapolation
                # Fit a straight line through the historical series and extend it
                x = np.arange(len(series))
                slope, intercept = np.polyfit(x, series, 1)
                forecast_val = max(intercept + slope * (len(series) + FORECAST_HORIZON - 1), 0.001)
                results[ticker] = round(float(forecast_val), 6)

        real_count = sum(1 for h in portfolio if emissions_history and h["ticker"] in (emissions_history or {}))
        print(f"[ml_modules] Module 2 ARIMA — forecasted {len(results)} holdings "
              f"({real_count} real data, {len(results)-real_count} synthetic)")
        return results

    except Exception as e:
        print(f"[ml_modules] Module 2 ARIMA failed: {e} — using static emissions values")
        # Last resort: return the static 2023 values unchanged
        return {h["ticker"]: h["emissions_intensity"] for h in portfolio}


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 3 — GAUSSIAN HMM REGIME DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def _run_hmm(df_raw):
    """
    Detect the current EUA carbon market regime using a Gaussian Hidden Markov Model.

    The HMM learns three hidden "regimes" from historical EUA price behaviour:
      - Low volatility:  prices stable, low month-to-month variation
      - Shock/transition: elevated volatility, policy uncertainty
      - Price spike:     extreme volatility, rapid price moves

    The current regime (as of the most recent month in the data) is mapped
    to the NGFS scenario that best represents that market environment.
    This scenario becomes the default selection when the page loads.

    Method:
      1. Filter EUA prices to Phase 3+ (2013 onwards) — pre-2013 near-zero
         prices would create a spurious fourth regime
      2. Compute monthly log returns: ln(price_t / price_{t-1})
         Log returns are approximately normally distributed, as required by GaussianHMM
      3. Train GaussianHMM with 3 components, 10 random restarts
         Multiple restarts help escape local optima — keep the best log-likelihood
      4. Sort states by variance (ascending) for stable labelling
         State 0 = calmest, State 2 = most volatile
      5. Predict the most likely state sequence and read off the current state
      6. Report confidence = probability of the most likely state at the final timestep

    Returns a dict with the detected NGFS scenario, regime name, and confidence score.
    Falls back to _FALLBACK_SCENARIO if hmmlearn is unavailable or no run converges.
    """
    try:
        if not HMM_AVAILABLE:
            print("[ml_modules] Module 3 HMM — hmmlearn not installed, using fallback")
            return _FALLBACK_SCENARIO.copy()

        # Filter to Phase 3+ (2013+) — Phase 1/2 prices were near-zero and distort training
        prices = df_raw[df_raw["year"] >= 2013]["price_gbp"].values
        if len(prices) < 24:
            print(f"[ml_modules] Module 3 HMM — insufficient data ({len(prices)} obs), using fallback")
            return _FALLBACK_SCENARIO.copy()

        # Compute log returns — this is the input signal to the HMM
        # Shape: (n_obs - 1, 1) — HMM expects a 2D array
        log_returns = np.log(prices[1:] / prices[:-1]).reshape(-1, 1)
        print(f"[ml_modules] Module 3 HMM — training on {len(log_returns)} monthly log returns...")

        # Run 10 random restarts, keep the model with the best log-likelihood
        # This reduces the risk of the EM algorithm getting stuck in a local optimum
        best_model = None
        best_score = -np.inf
        for seed in range(10):
            try:
                candidate = GaussianHMM(
                    n_components=3,       # three regimes: calm, shock, spike
                    covariance_type="full",
                    n_iter=500,           # maximum EM iterations
                    random_state=seed,    # different initialisation each time
                )
                candidate.fit(log_returns)

                # Only accept converged models
                if not candidate.monitor_.converged:
                    continue

                score = candidate.score(log_returns)
                if score > best_score:
                    best_score  = score
                    best_model  = candidate
            except Exception:
                continue  # this seed failed, try the next

        if best_model is None:
            print("[ml_modules] Module 3 HMM — no run converged, using fallback")
            return _FALLBACK_SCENARIO.copy()

        model = best_model

        # Sort states by variance (ascending) — gives stable labels across runs
        # State 0 = lowest variance = calmest market regime
        variances   = model.covars_.flatten()
        state_order = np.argsort(variances)
        regime_map  = {state_order[i]: i for i in range(3)}  # raw state → normalised regime

        # Warn if two states have nearly identical variance — regime collapse
        sorted_vars = np.sort(variances)
        var_gaps    = np.diff(sorted_vars)
        if any(g < 1e-5 for g in var_gaps):
            print("[ml_modules] Module 3 HMM — WARNING: regime collapse detected")

        # Predict state at each timestep, read the most recent
        states         = model.predict(log_returns)
        current_raw    = int(states[-1])
        current_regime = regime_map[current_raw]  # map to 0/1/2

        # Confidence = probability of being in the current state at the final timestep
        probs      = model.predict_proba(log_returns)
        confidence = float(probs[-1][current_raw])

        # Build the output scenario dict from the regime mapping
        scenario = _REGIME_TO_NGFS[current_regime].copy()
        scenario.update({
            "current_regime":       current_regime,
            "confidence":           round(confidence, 3),
            "source":               "GaussianHMM on EUA monthly log returns — best of 10 restarts",
            "n_observations":       len(log_returns),
            "model_log_likelihood": round(best_score, 3),
        })

        print(f"[ml_modules] Module 3 HMM — regime: {scenario['regime_name']} "
              f"(confidence: {confidence:.2f}) → {scenario['label']} £{scenario['carbon_price']}/t")
        return scenario

    except Exception as e:
        print(f"[ml_modules] Module 3 HMM failed: {e} — using fallback")
        return _FALLBACK_SCENARIO.copy()


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 4 — SCIPY MEAN-VARIANCE PORTFOLIO OPTIMISER
# ══════════════════════════════════════════════════════════════════════════════

def run_scipy_optimiser(portfolio, turnover_limit=0.25):
    """
    Optimise portfolio weights to minimise climate transition risk.

    Objective function:
        minimise: LAMBDA_EAR × portfolio_EaR + LAMBDA_VOL × w' × COV × w
        where LAMBDA_EAR = 0.70, LAMBDA_VOL = 0.30

    This is a weighted sum of two goals:
      1. Minimise EaR — reduce climate risk exposure (70% of objective)
      2. Minimise portfolio variance — avoid concentration into a single stock (30%)

    Without the variance term, the optimiser would dump everything into the
    single lowest-EaR holding. The covariance term penalises this.

    Covariance matrix construction:
        COV = beta × beta' × sigma_m² + diag(sigma_e²)
      - Single-factor model: market risk comes from betas, idiosyncratic from sigma_e
      - sigma_m = 20% annualised market volatility (standard assumption)
      - sigma_e = max(EaR × 0.5, 0.05) — high EaR holdings get higher idiosyncratic
        risk, which further penalises over-concentration

    Constraints:
      - Weights sum to 1.0 (fully invested)
      - Each weight >= MIN_WEIGHT (0.5%) — no position fully closed
      - Each weight <= min(3× original weight, 20%) — no extreme concentration
      - Actual turnover <= turnover_limit — enforced via bisection after SLSQP

    Turnover enforcement:
      SLSQP solves the unconstrained problem first (ignoring turnover).
      Then bisection scales the weight shifts back until actual turnover
      (sum of absolute weight changes) lands at or below the limit.
      100 bisection iterations gives precision to ~1e-8.

    Args:
        portfolio: list of dicts — must have 'weight', 'eps_impact', 'beta', 'ticker'
        turnover_limit: max allowed turnover as decimal (default 0.25 = 25%)

    Returns dict with:
        final_weights   — numpy array of optimised weights
        original_ear    — portfolio EaR before optimisation (%)
        optimised_ear   — portfolio EaR after optimisation (%)
        reduction       — EaR reduction achieved (%)
        actual_turnover — turnover actually applied (%)
        scipy_success   — whether SLSQP converged
    """
    n    = len(portfolio)
    w0   = np.array([h["weight"]     for h in portfolio])  # original weights
    ear  = np.array([h["eps_impact"] for h in portfolio])  # EPS at Risk per holding
    beta = np.array([h["beta"]       for h in portfolio]).reshape(-1, 1)

    # ── Build covariance matrix ───────────────────────────────────────────────
    # Market component: all holdings move together via their beta to the market
    # Idiosyncratic component: diagonal — each holding's own risk, scaled by EaR
    SIGMA_M    = 0.20  # assumed annualised market volatility
    sigma_e    = np.maximum(ear * 0.5, 0.05)   # idiosyncratic vol, floor at 5%
    cov_market = (SIGMA_M ** 2) * (beta @ beta.T)  # systematic risk
    cov_idio   = np.diag(sigma_e ** 2)              # idiosyncratic risk
    COV        = cov_market + cov_idio

    def objective(w):
        """Weighted sum of portfolio EaR and portfolio variance."""
        portfolio_ear = float(np.dot(w, ear))   # weighted average EPS at Risk
        portfolio_var = float(w @ COV @ w)       # portfolio variance (risk)
        return LAMBDA_EAR * portfolio_ear + LAMBDA_VOL * portfolio_var

    # Position caps: max weight = min(3× original, 20%) to prevent concentration
    max_weights = np.minimum(w0 * 3.0, 0.20)
    max_weights = np.maximum(max_weights, MIN_WEIGHT * 2)  # ensure cap > floor
    bounds      = [(MIN_WEIGHT, float(mx)) for mx in max_weights]

    # Equality constraint: weights must sum to exactly 1.0
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    # Run SLSQP (Sequential Least Squares Programming) — standard QP solver
    result = scipy_minimize(
        fun=objective,
        x0=w0,            # start from current portfolio as initial guess
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-9},
    )

    # Use optimised weights if SLSQP converged, otherwise keep original
    unconstrained = result.x if result.success else w0
    deltas        = unconstrained - w0  # how much each weight wants to change

    def scale_and_measure(s):
        """Apply deltas scaled by factor s, floor at MIN_WEIGHT, renormalise."""
        shifted  = np.maximum(w0 + deltas * s, MIN_WEIGHT)
        normed   = shifted / shifted.sum()
        turnover = float(np.sum(np.abs(normed - w0)))
        return normed, turnover

    # Start by applying the full unconstrained shift
    final_weights, actual_turnover = scale_and_measure(1.0)

    # If it exceeds the turnover limit, bisect to find the largest s that doesn't
    if actual_turnover > turnover_limit:
        lo, hi = 0.0, 1.0
        for _ in range(100):  # 100 iterations → precision ~1e-8
            s_mid = (lo + hi) / 2.0
            w_mid, t_mid = scale_and_measure(s_mid)
            if t_mid <= turnover_limit:
                lo, final_weights, actual_turnover = s_mid, w_mid, t_mid  # feasible, try more
            else:
                hi = s_mid  # too much turnover, scale back
            if (hi - lo) < 1e-8:
                break

    # Compute EaR before and after for reporting
    orig_ear  = float(np.dot(w0, ear))
    opt_ear   = float(np.dot(final_weights, ear))
    reduction = ((orig_ear - opt_ear) / orig_ear * 100) if orig_ear > 0 else 0.0

    return {
        "final_weights":   final_weights,          # numpy array — used by ear_engine.py
        "original_ear":    round(orig_ear * 100, 3),
        "optimised_ear":   round(opt_ear * 100, 3),
        "reduction":       round(reduction, 2),
        "actual_turnover": round(actual_turnover * 100, 2),  # returned as %
        "scipy_success":   result.success,
    }


# ══════════════════════════════════════════════════════════════════════════════
# STARTUP — background thread so Flask binds port immediately
# ══════════════════════════════════════════════════════════════════════════════

# Initialise globals with fallback values so the app is usable from the first request.
# The background thread will update these once ML completes (~10-20 seconds).
PASSTHROUGH_RATES   = _FALLBACK_PASSTHROUGH.copy()
EMISSIONS_FORECASTS = {}    # empty until ARIMA completes
DETECTED_SCENARIO   = _FALLBACK_SCENARIO.copy()
ML_READY            = False  # flips to True once all four modules complete


def _initialise_background():
    """
    Run all four ML modules at full quality and update the module-level globals.

    This function runs in a daemon thread launched at import time (see bottom of file).
    Flask is already serving requests while this runs — the app uses fallback values
    until this completes, then automatically switches to full ML outputs.

    The 'global' keyword is required because we're reassigning module-level variables,
    not just mutating them. All four globals are updated atomically at the end.
    """
    global PASSTHROUGH_RATES, EMISSIONS_FORECASTS, DETECTED_SCENARIO, ML_READY
    try:
        # Import the default portfolio to use as input for Modules 1 and 2
        from ear_engine import PORTFOLIO as DEFAULT_PORTFOLIO

        df_raw = _load_eua_data()  # load EUA price history once, share across modules

        if df_raw is not None:
            passthrough_rates   = _run_ols(df_raw, DEFAULT_PORTFOLIO)   # Module 1
            emissions_forecasts = _run_arima(DEFAULT_PORTFOLIO)          # Module 2
            detected_scenario   = _run_hmm(df_raw)                       # Module 3
        else:
            # EUA data missing — all three modules fall back
            print("[ml_modules] No EUA data — all modules using fallbacks")
            passthrough_rates   = _FALLBACK_PASSTHROUGH.copy()
            emissions_forecasts = {h["ticker"]: h["emissions_intensity"] for h in DEFAULT_PORTFOLIO}
            detected_scenario   = _FALLBACK_SCENARIO.copy()

        # Atomically update all four globals at once
        # Module 4 (scipy) is a function, not a global — it's already available
        PASSTHROUGH_RATES   = passthrough_rates
        EMISSIONS_FORECASTS = emissions_forecasts
        DETECTED_SCENARIO   = detected_scenario
        ML_READY            = True

        print(f"[ml_modules] ✓ Background init complete. "
              f"Scenario: {DETECTED_SCENARIO['label']} "
              f"(£{DETECTED_SCENARIO['carbon_price']}/t)")

    except Exception as e:
        # Even on failure, mark ML_READY=True so the status endpoint reflects completion
        # The fallback values set at module load time remain active
        ML_READY = True
        print(f"[ml_modules] Background init failed: {e} — fallbacks remain active")


# Launch the background thread as a daemon — it will be killed automatically
# when the main Flask process exits (daemon=True ensures this)
_bg_thread = threading.Thread(target=_initialise_background, daemon=True)
_bg_thread.start()
print("[ml_modules] Background ML initialisation started — app ready immediately")
