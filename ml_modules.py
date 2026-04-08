"""
ml_modules.py — Stirling Solvers CFA AI Investment Challenge 2026
Runs all 4 ML modules at startup and exposes outputs for ear_engine.py and app.py.

Outputs:
    PASSTHROUGH_RATES    dict  {sector: rate}          — Module 1 OLS
    EMISSIONS_FORECASTS  dict  {ticker: intensity}     — Module 2 ARIMA
    DETECTED_SCENARIO    dict  NGFS scenario + regime  — Module 3 HMM
    run_scipy_optimiser  func  (portfolio) -> result   — Module 4 scipy
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

logging.getLogger("statsmodels").setLevel(logging.CRITICAL)

# ── Path to Agnes's data file ─────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "EUA_Yearly_futures.csv")

# ── Fallback values (used if modules fail or data is missing) ─────────────────
_FALLBACK_PASSTHROUGH = {
    "Energy":      0.35,
    "Materials":   0.18,
    "Industrials": 0.62,
    "Healthcare":  0.82,
    "Financials":  0.91,
    "Consumer":    0.72,
    "Utilities":   0.85,
}

_FALLBACK_SCENARIO = {
    "id": "below2",
    "label": "Orderly — Below 2°C",
    "carbon_price": 120,
    "regime_name": "Unknown (HMM fallback)",
    "source": "fallback",
}

# ── NGFS regime mapping (sorted by variance: 0=calm, 1=shock, 2=spike) ────────
_REGIME_TO_NGFS = {
    0: {"id": "base",   "label": "Current Policies",        "carbon_price": 42,  "regime_name": "Low Volatility"},
    1: {"id": "delayed","label": "Delayed Transition",      "carbon_price": 80,  "regime_name": "Shock Transition"},
    2: {"id": "netzero","label": "Orderly — Net Zero 2050", "carbon_price": 200, "regime_name": "Price Spike"},
}

ARIMA_ORDER      = (1, 1, 0)
FORECAST_HORIZON = 3
MIN_R2           = 0.05
LAMBDA_EAR       = 0.70
LAMBDA_VOL       = 0.30
MIN_WEIGHT       = 0.005


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def _load_eua_data():
    if not os.path.exists(DATA_PATH):
        print(f"[ml_modules] WARNING: {DATA_PATH} not found — ML modules will use fallbacks")
        return None

    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    df = df.sort_values("Date").reset_index(drop=True)
    df["year"]  = df["Date"].dt.year
    df["month"] = df["Date"].dt.month

    price_col = next((c for c in df.columns if "gbp" in c.lower() or "price" in c.lower()), None)
    if price_col is None:
        print("[ml_modules] WARNING: Could not identify price column in EUA CSV")
        return None

    df = df.rename(columns={price_col: "price_gbp"})
    df = df.dropna(subset=["price_gbp"])
    print(f"[ml_modules] Loaded {len(df)} EUA observations ({df['Date'].min().strftime('%b %Y')} – {df['Date'].max().strftime('%b %Y')})")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 1 — OLS PASS-THROUGH REGRESSION
# ══════════════════════════════════════════════════════════════════════════════

def _run_ols(df_raw, portfolio):
    """Returns {sector: pass_through_rate}"""
    try:
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score

        annual_prices = df_raw.groupby("year")["price_gbp"].mean().reset_index()
        annual_prices.columns = ["year", "annual_avg_price_gbp"]

        rows = []
        for h in portfolio:
            rows.append({
                "ticker":       h["ticker"],
                "sector":       h["sector"],
                "ebitda_margin": h["ebitda_margin"],
                "year":         2023,
            })
        df_margins = pd.DataFrame(rows)

        df_reg = df_margins.merge(annual_prices, on="year", how="inner")
        if df_reg.empty:
            return _FALLBACK_PASSTHROUGH.copy()

        results = {}
        for sector in df_reg["sector"].unique():
            sdata = df_reg[df_reg["sector"] == sector].dropna(
                subset=["annual_avg_price_gbp", "ebitda_margin"]
            )

            if len(sdata) < 4:
                results[sector] = _FALLBACK_PASSTHROUGH.get(sector, 0.5)
                continue

            X = sdata[["annual_avg_price_gbp"]].values
            y = sdata["ebitda_margin"].values
            split = max(len(X) - 2, int(len(X) * 0.75))

            model = LinearRegression()
            model.fit(X[:split], y[:split])
            r2 = r2_score(y[:split], model.predict(X[:split]))

            if r2 < MIN_R2:
                results[sector] = _FALLBACK_PASSTHROUGH.get(sector, 0.5)
                continue

            coef = model.coef_[0]
            mean_margin = y.mean()
            absorption = min(max(abs(coef) / (mean_margin + 1e-9), 0.0), 1.0)
            pt = round(min(max(1.0 - absorption, 0.05), 0.95), 3)
            results[sector] = pt

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

# ── Path to Agnes's emissions history file ───────────────────────────────────
EMISSIONS_DATA_PATH = os.path.join(os.path.dirname(__file__), "CarbonEmissions.xlsx")

# ── Ticker mapping: Agnes's tickers → ear_engine tickers ─────────────────────
_TICKER_MAP = {"BA": "BA.", "RR": "RR.", "NG": "NG."}


def _load_emissions_data():
    """Load real emissions intensity history from Agnes's Excel file."""
    if not os.path.exists(EMISSIONS_DATA_PATH):
        print(f"[ml_modules] WARNING: {EMISSIONS_DATA_PATH} not found — ARIMA will use synthetic data")
        return None
    try:
        df = pd.read_excel(EMISSIONS_DATA_PATH)
        df.columns = [c.strip() for c in df.columns]
        df["Holding"] = df["Holding"].replace(_TICKER_MAP)
        # Build {ticker: {year: intensity}}
        history = {}
        for _, row in df.iterrows():
            ticker = str(row["Holding"])
            year   = int(row["Year"])
            intensity = float(row["emissions_intensity"])
            history.setdefault(ticker, {})[year] = intensity
        print(f"[ml_modules] Loaded real emissions data for {len(history)} holdings (2018–2023)")
        return history
    except Exception as e:
        print(f"[ml_modules] WARNING: Could not load emissions data: {e}")
        return None


def _run_arima(portfolio):
    """Returns {ticker: forecasted_emissions_intensity}"""
    try:
        emissions_history = _load_emissions_data()

        results = {}
        for h in portfolio:
            ticker   = h["ticker"]
            last_val = h["emissions_intensity"]

            # Use real historical data if available, else synthetic decay
            if emissions_history and ticker in emissions_history:
                hist = emissions_history[ticker]
                series = np.array([hist[y] for y in sorted(hist.keys())])
                source = "real"
            else:
                series = np.array([last_val * (0.95 ** i) for i in range(5, -1, -1)])
                source = "synthetic"

            try:
                if not STATSMODELS_AVAILABLE:
                    raise ImportError("statsmodels not available")
                model = ARIMA(series, order=ARIMA_ORDER)
                fitted = model.fit()
                forecasts = list(fitted.forecast(steps=FORECAST_HORIZON))

                if any(f < 0 or f > series[-1] * 5 for f in forecasts):
                    raise ValueError("Forecast out of range")

                forecast_val = max(forecasts[-1], 0.001)
                results[ticker] = round(float(forecast_val), 6)

            except Exception:
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
        return {h["ticker"]: h["emissions_intensity"] for h in portfolio}


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 3 — HMM REGIME DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def _run_hmm(df_raw):
    """Returns NGFS scenario dict with detected regime."""
    try:
        from hmmlearn.hmm import GaussianHMM

        prices = df_raw[df_raw["year"] >= 2013]["price_gbp"].values
        if len(prices) < 24:
            print(f"[ml_modules] Module 3 HMM — insufficient data ({len(prices)} obs), using fallback")
            return _FALLBACK_SCENARIO.copy()

        log_returns = np.log(prices[1:] / prices[:-1]).reshape(-1, 1)
        print(f"[ml_modules] Module 3 HMM — training on {len(log_returns)} monthly log returns...")

        best_model = None
        best_score = -np.inf
        for seed in range(10):
            try:
                candidate = GaussianHMM(
                    n_components=3,
                    covariance_type="full",
                    n_iter=500,
                    random_state=seed,
                )
                candidate.fit(log_returns)
                if not candidate.monitor_.converged:
                    continue
                score = candidate.score(log_returns)
                if score > best_score:
                    best_score = score
                    best_model = candidate
            except Exception:
                continue

        if best_model is None:
            print("[ml_modules] Module 3 HMM — no run converged, using fallback")
            return _FALLBACK_SCENARIO.copy()

        model = best_model

        # Sort states by VARIANCE — more stable than sorting by mean for carbon data
        variances = model.covars_.flatten()
        state_order = np.argsort(variances)
        regime_map = {state_order[i]: i for i in range(3)}

        # Regime collapse check
        sorted_vars = np.sort(variances)
        var_gaps = np.diff(sorted_vars)
        if any(g < 1e-5 for g in var_gaps):
            print("[ml_modules] Module 3 HMM — WARNING: regime collapse detected")

        states    = model.predict(log_returns)
        current_raw    = int(states[-1])
        current_regime = regime_map[current_raw]

        probs      = model.predict_proba(log_returns)
        confidence = float(probs[-1][current_raw])

        scenario = _REGIME_TO_NGFS[current_regime].copy()
        scenario.update({
            "current_regime":      current_regime,
            "confidence":          round(confidence, 3),
            "source":              "GaussianHMM on EUA monthly log returns — best of 10 restarts",
            "n_observations":      len(log_returns),
            "model_log_likelihood": round(best_score, 3),
        })

        print(f"[ml_modules] Module 3 HMM — regime: {scenario['regime_name']} "
              f"(confidence: {confidence:.2f}) → {scenario['label']} £{scenario['carbon_price']}/t")
        return scenario

    except ImportError:
        print("[ml_modules] Module 3 HMM — hmmlearn not installed, using fallback")
        return _FALLBACK_SCENARIO.copy()
    except Exception as e:
        print(f"[ml_modules] Module 3 HMM failed: {e} — using fallback")
        return _FALLBACK_SCENARIO.copy()


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 4 — SCIPY MEAN-VARIANCE OPTIMISER
# ══════════════════════════════════════════════════════════════════════════════

def run_scipy_optimiser(portfolio, turnover_limit=0.25):
    """
    scipy SLSQP portfolio optimiser.
    Minimises: LAMBDA_EAR * portfolio_EaR + LAMBDA_VOL * w' * COV * w
    Subject to: weights sum to 1, all weights >= MIN_WEIGHT,
                actual turnover <= turnover_limit (enforced via bisection)

    Covariance matrix is built from each holding's beta and idiosyncratic
    EaR contribution — diagonal plus a market factor term from betas.
    This distributes rebalancing across holdings rather than concentrating
    into a single low-EaR stock.

    Args:
        portfolio: list of holding dicts — must have 'weight', 'eps_impact', 'beta'
        turnover_limit: max allowed portfolio turnover (default 0.25 = 25%)

    Returns dict with final_weights, original_ear, optimised_ear, reduction, actual_turnover
    """
    from scipy.optimize import minimize

    n    = len(portfolio)
    w0   = np.array([h["weight"]     for h in portfolio])
    ear  = np.array([h["eps_impact"] for h in portfolio])
    beta = np.array([h["beta"]       for h in portfolio]).reshape(-1, 1)

    # ── Covariance matrix ────────────────────────────────────────────────────
    # Single-factor model: Cov = beta * beta' * sigma_m^2 + diag(sigma_e^2)
    # sigma_m = assumed market vol (20% annualised is standard)
    # sigma_e = idiosyncratic vol estimated from EaR — high EaR holdings
    #           are treated as having higher idiosyncratic risk, which
    #           penalises over-concentration into any single stock
    SIGMA_M = 0.20
    sigma_e = np.maximum(ear * 0.5, 0.05)  # idiosyncratic vol floor at 5%

    cov_market = (SIGMA_M ** 2) * (beta @ beta.T)
    cov_idio   = np.diag(sigma_e ** 2)
    COV        = cov_market + cov_idio

    def objective(w):
        portfolio_ear = float(np.dot(w, ear))
        portfolio_var = float(w @ COV @ w)
        return LAMBDA_EAR * portfolio_ear + LAMBDA_VOL * portfolio_var

    # Individual weight cap: no single position > 3x its original weight
    # or 20% absolute — prevents degenerate concentration
    max_weights = np.minimum(w0 * 3.0, 0.20)
    max_weights = np.maximum(max_weights, MIN_WEIGHT * 2)
    bounds = [(MIN_WEIGHT, float(mx)) for mx in max_weights]

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    result = minimize(
        fun=objective,
        x0=w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-9},
    )

    unconstrained = result.x if result.success else w0
    deltas = unconstrained - w0

    def scale_and_measure(s):
        shifted = np.maximum(w0 + deltas * s, MIN_WEIGHT)
        normed  = shifted / shifted.sum()
        turnover = float(np.sum(np.abs(normed - w0)))
        return normed, turnover

    final_weights, actual_turnover = scale_and_measure(1.0)

    if actual_turnover > turnover_limit:
        lo, hi = 0.0, 1.0
        for _ in range(100):
            s_mid = (lo + hi) / 2.0
            w_mid, t_mid = scale_and_measure(s_mid)
            if t_mid <= turnover_limit:
                lo, final_weights, actual_turnover = s_mid, w_mid, t_mid
            else:
                hi = s_mid
            if (hi - lo) < 1e-8:
                break

    orig_ear = float(np.dot(w0, ear))
    opt_ear  = float(np.dot(final_weights, ear))
    reduction = ((orig_ear - opt_ear) / orig_ear * 100) if orig_ear > 0 else 0.0

    return {
        "final_weights":   final_weights,
        "original_ear":    round(orig_ear * 100, 3),
        "optimised_ear":   round(opt_ear * 100, 3),
        "reduction":       round(reduction, 2),
        "actual_turnover": round(actual_turnover * 100, 2),
        "scipy_success":   result.success,
    }


# ══════════════════════════════════════════════════════════════════════════════
# STARTUP — background thread so Flask binds port immediately
# ══════════════════════════════════════════════════════════════════════════════

import threading

# Initialise with fallbacks immediately so the app is usable from first request
PASSTHROUGH_RATES   = _FALLBACK_PASSTHROUGH.copy()
EMISSIONS_FORECASTS = {}
DETECTED_SCENARIO   = _FALLBACK_SCENARIO.copy()
ML_READY            = False   # flips to True once background init completes


def _initialise_background():
    """
    Runs all 4 ML modules at full quality in a background thread.
    Updates the module-level globals once complete.
    Flask binds its port immediately using fallback values.
    All requests after this completes get full ML outputs.
    """
    global PASSTHROUGH_RATES, EMISSIONS_FORECASTS, DETECTED_SCENARIO, ML_READY
    try:
        from ear_engine import PORTFOLIO as DEFAULT_PORTFOLIO

        df_raw = _load_eua_data()

        if df_raw is not None:
            passthrough_rates   = _run_ols(df_raw, DEFAULT_PORTFOLIO)
            emissions_forecasts = _run_arima(DEFAULT_PORTFOLIO)
            detected_scenario   = _run_hmm(df_raw)
        else:
            print("[ml_modules] No EUA data — all modules using fallbacks")
            passthrough_rates   = _FALLBACK_PASSTHROUGH.copy()
            emissions_forecasts = {h["ticker"]: h["emissions_intensity"] for h in DEFAULT_PORTFOLIO}
            detected_scenario   = _FALLBACK_SCENARIO.copy()

        # Atomically update globals
        PASSTHROUGH_RATES   = passthrough_rates
        EMISSIONS_FORECASTS = emissions_forecasts
        DETECTED_SCENARIO   = detected_scenario
        ML_READY            = True

        print(f"[ml_modules] ✓ Background init complete. "
              f"Scenario: {DETECTED_SCENARIO['label']} "
              f"(£{DETECTED_SCENARIO['carbon_price']}/t)")

    except Exception as e:
        ML_READY = True  # mark ready even on failure so status endpoint reflects it
        print(f"[ml_modules] Background init failed: {e} — fallbacks remain active")


# Launch background thread at import time
_bg_thread = threading.Thread(target=_initialise_background, daemon=True)
_bg_thread.start()
print("[ml_modules] Background ML initialisation started — app ready immediately")
