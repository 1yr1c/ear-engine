#!/usr/bin/env python
# coding: utf-8

# In[19]:


"""
results_visualisation.py — Stirling Solvers CFA AI Investment Challenge 2026

REQUIRES:
    EUA_Yearly_futures.csv   — EUA monthly price history
    CarbonEmissions.xlsx     — Scope 1+2 emissions dataset

OUTPUT FILES:
    fig1_ols_regression.png  — OLS pass-through regression
    fig2_arima_forecasts.png — ARIMA emissions forecasts
    fig3_optimisation.png    — Optimisation before/after rebalancing

"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from statsmodels.tsa.arima.model import ARIMA
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":    "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "figure.dpi":        120,
})

SECTOR_COLORS = {
    "Energy":      "#ef4444",
    "Materials":   "#f97316",
    "Industrials": "#eab308",
    "Consumer":    "#22c55e",
    "Healthcare":  "#3b82f6",
    "Financials":  "#8b5cf6",
    "Utilities":   "#06b6d4",
}

# ── Fallback pass-through rates (Fabra & Reguant 2014) ───────────────────────
FALLBACK_PASSTHROUGH = {
    "Energy":      0.35,
    "Materials":   0.18,
    "Industrials": 0.62,
    "Healthcare":  0.82,
    "Financials":  0.91,
    "Consumer":    0.72,
    "Utilities":   0.85,
}

# ── Portfolio — must match ear_engine.py exactly ─────────────────────────────
PORTFOLIO = [
    {"ticker": "SHEL", "sector": "Energy",      "weight": 0.117,
     "ebitda_margin": 0.182, "emissions_intensity": 0.180, "beta": 1.10},
    {"ticker": "BP",   "sector": "Energy",      "weight": 0.064,
     "ebitda_margin": 0.148, "emissions_intensity": 0.162, "beta": 1.05},
    {"ticker": "GLEN", "sector": "Materials",   "weight": 0.057,
     "ebitda_margin": 0.098, "emissions_intensity": 0.227, "beta": 1.30},
    {"ticker": "RIO",  "sector": "Materials",   "weight": 0.054,
     "ebitda_margin": 0.420, "emissions_intensity": 0.603, "beta": 0.95},
    {"ticker": "AAL",  "sector": "Materials",   "weight": 0.031,
     "ebitda_margin": 0.320, "emissions_intensity": 0.567, "beta": 1.20},
    {"ticker": "BA.",  "sector": "Industrials", "weight": 0.079,
     "ebitda_margin": 0.115, "emissions_intensity": 0.015, "beta": 0.70},
    {"ticker": "RR.",  "sector": "Industrials", "weight": 0.069,
     "ebitda_margin": 0.092, "emissions_intensity": 0.060, "beta": 1.40},
    {"ticker": "AZN",  "sector": "Healthcare",  "weight": 0.113,
     "ebitda_margin": 0.320, "emissions_intensity": 0.018, "beta": 0.50},
    {"ticker": "GSK",  "sector": "Healthcare",  "weight": 0.054,
     "ebitda_margin": 0.300, "emissions_intensity": 0.030, "beta": 0.55},
    {"ticker": "HSBA", "sector": "Financials",  "weight": 0.093,
     "ebitda_margin": 0.360, "emissions_intensity": 0.010, "beta": 1.00},
    {"ticker": "BARC", "sector": "Financials",  "weight": 0.060,
     "ebitda_margin": 0.280, "emissions_intensity": 0.004, "beta": 1.15},
    {"ticker": "LLOY", "sector": "Financials",  "weight": 0.046,
     "ebitda_margin": 0.310, "emissions_intensity": 0.003, "beta": 1.10},
    {"ticker": "ULVR", "sector": "Consumer",    "weight": 0.071,
     "ebitda_margin": 0.172, "emissions_intensity": 0.025, "beta": 0.60},
    {"ticker": "BATS", "sector": "Consumer",    "weight": 0.043,
     "ebitda_margin": 0.430, "emissions_intensity": 0.104, "beta": 0.45},
    {"ticker": "NG.",  "sector": "Utilities",   "weight": 0.049,
     "ebitda_margin": 0.380, "emissions_intensity": 0.476, "beta": 0.40},
]

# ── NGFS Net Zero 2050 scenario ───────────────────────────────────────────────
SCENARIO_NZ2050 = {
    "carbon_price": 200, "subsidy_removal": 0.55,
    "bca": 0.20, "energy_shock": 0.28,
}

# ── Ticker mapping: Excel uses BA/RR/NG, portfolio uses BA./RR./NG. ──────────
TICKER_EXCEL_TO_PORTFOLIO = {"BA": "BA.", "RR": "RR.", "NG": "NG."}


# In[20]:


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD DATASETS
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("Loading datasets...")

# EUA price history
df_eua = pd.read_csv("EUA_Yearly_futures.csv")
df_eua.columns = [c.strip().lstrip("\ufeff") for c in df_eua.columns]
df_eua["Date"] = pd.to_datetime(df_eua["Date"], dayfirst=True)
df_eua = df_eua.sort_values("Date").reset_index(drop=True)
df_eua["year"] = df_eua["Date"].dt.year

# Find the GBP price column automatically 
price_col = next(
    (c for c in df_eua.columns if "gbp" in c.lower() or "price" in c.lower()), None
)
df_eua = df_eua.rename(columns={price_col: "price_gbp"})
df_eua = df_eua.dropna(subset=["price_gbp"])
print(f"  EUA: {len(df_eua)} monthly observations "
      f"({df_eua['Date'].min().strftime('%b %Y')} – "
      f"{df_eua['Date'].max().strftime('%b %Y')})")

# Annual average prices for OLS regression
annual_prices = (
    df_eua[df_eua["year"] >= 2013]
    .groupby("year")["price_gbp"]
    .mean()
    .reset_index()
    .rename(columns={"price_gbp": "annual_avg_price_gbp"})
)

# Emissions history from Excel file
df_ce = pd.read_excel("CarbonEmissions.xlsx", sheet_name="Module 2_CarbonEmissions")
df_ce.columns = [c.strip() for c in df_ce.columns]
df_ce["Holding"] = df_ce["Holding"].replace(TICKER_EXCEL_TO_PORTFOLIO)


emissions_history = {}
for _, row in df_ce.iterrows():
    ticker = str(row["Holding"])
    year   = int(row["Year"])
    # Raw value from Excel (e.g. Shell = 183.2 tCO2e per $bn)
    raw_intensity = float(row["emissions_intensity"])
    emissions_history.setdefault(ticker, {})[year] = raw_intensity

print(f"  Emissions: {len(emissions_history)} holdings, "
      f"years {df_ce['Year'].min()}–{df_ce['Year'].max()}")
print("Datasets loaded.\n")


# In[21]:


# ── Multi-year EBITDA margins 2018-2023 (for regression visualisation) ────────

EBITDA_HISTORY = {
    "SHEL": {2018:0.182,2019:0.148,2020:0.052,2021:0.168,2022:0.220,2023:0.182},
    "BP":   {2018:0.148,2019:0.120,2020:0.032,2021:0.138,2022:0.195,2023:0.148},
    "GLEN": {2018:0.098,2019:0.085,2020:0.062,2021:0.125,2022:0.148,2023:0.098},
    "RIO":  {2018:0.420,2019:0.390,2020:0.345,2021:0.520,2022:0.440,2023:0.420},
    "AAL":  {2018:0.320,2019:0.295,2020:0.185,2021:0.318,2022:0.390,2023:0.320},
    "BA.":  {2018:0.115,2019:0.112,2020:0.098,2021:0.108,2022:0.112,2023:0.115},
    "RR.":  {2018:0.092,2019:0.095,2020:0.038,2021:0.072,2022:0.082,2023:0.092},
    "AZN":  {2018:0.320,2019:0.320,2020:0.315,2021:0.318,2022:0.310,2023:0.320},
    "GSK":  {2018:0.300,2019:0.298,2020:0.295,2021:0.298,2022:0.292,2023:0.300},
    "HSBA": {2018:0.360,2019:0.345,2020:0.285,2021:0.320,2022:0.378,2023:0.360},
    "BARC": {2018:0.280,2019:0.275,2020:0.248,2021:0.308,2022:0.315,2023:0.280},
    "LLOY": {2018:0.310,2019:0.295,2020:0.218,2021:0.302,2022:0.325,2023:0.310},
    "ULVR": {2018:0.172,2019:0.172,2020:0.168,2021:0.162,2022:0.148,2023:0.172},
    "BATS": {2018:0.430,2019:0.428,2020:0.432,2021:0.435,2022:0.430,2023:0.430},
    "NG.":  {2018:0.380,2019:0.378,2020:0.368,2021:0.375,2022:0.372,2023:0.380},
}


# In[22]:


# ══════════════════════════════════════════════════════════════════════════════
# OLS PASS-THROUGH REGRESSION
# ══════════════════════════════════════════════════════════════════════════════
 
 
# Build multi-year margins dataframe from EBITDA_HISTORY

margin_rows = []
for h in PORTFOLIO:
    ticker = h["ticker"]
    sector = h["sector"]
    hist   = EBITDA_HISTORY.get(ticker, {2023: h["ebitda_margin"]})
    for year, margin in hist.items():
        margin_rows.append({
            "ticker":       ticker,
            "sector":       sector,
            "ebitda_margin": margin,
            "year":         year,
        })
df_margins = pd.DataFrame(margin_rows)
 

df_reg = df_margins.merge(annual_prices, on="year", how="inner")
 
ols_passthrough = {}   # {sector: pass_through_rate}
ols_details     = {}   # {sector: {coef, r2, source}}
 
sectors = sorted(df_reg["sector"].unique())
 
fig1, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()
 
for i, sector in enumerate(sectors):
    ax = axes[i]
    sdata = df_reg[df_reg["sector"] == sector].dropna(
        subset=["annual_avg_price_gbp", "ebitda_margin"]
    )
    X = sdata[["annual_avg_price_gbp"]].values
    y = sdata["ebitda_margin"].values
 
    ax.scatter(X, y,
               color=SECTOR_COLORS.get(sector, "steelblue"),
               s=60, zorder=3, alpha=0.85, edgecolors="white", linewidth=0.5)
 
    if len(sdata) >= 4:
        split = max(len(X) - 2, int(len(X) * 0.75))
        model = LinearRegression()
        model.fit(X[:split], y[:split])
        r2 = r2_score(y[:split], model.predict(X[:split]))
 
        x_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        ax.plot(x_line, model.predict(x_line), "r-", linewidth=2, alpha=0.9)
 
        if r2 >= 0.05:
            coef = model.coef_[0]
            mean_m = y.mean()
            absorption = min(max(abs(coef) / (mean_m + 1e-9), 0.0), 1.0)
            pt = round(min(max(1.0 - absorption, 0.05), 0.95), 3)
            source = "OLS"
        else:
            pt = FALLBACK_PASSTHROUGH[sector]
            r2 = None
            source = "Fallback (low R²)"
    else:
        pt = FALLBACK_PASSTHROUGH[sector]
        r2 = None
        source = "Fallback (insufficient data)"
 
    ols_passthrough[sector] = pt
    ols_details[sector] = {"pass_through": pt, "r2": r2, "source": source}
 
    r2_str = f"  R²={r2:.2f}" if r2 is not None else ""
    ax.set_title(f"{sector}\npass-through: {pt:.0%}{r2_str}", fontsize=10)
    ax.set_xlabel("Carbon price (£/t)", fontsize=8)
    ax.set_ylabel("EBITDA margin", fontsize=8)
    ax.tick_params(labelsize=8)
 
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)
 
fig1.suptitle(
    "OLS Regression — EUA Carbon Price vs EBITDA Margin by Sector\n"
    "(pass-through rates used by ear_engine.py)",
    fontsize=13, fontweight="bold",
)
plt.tight_layout()
plt.savefig("fig1_ols_regression.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"  Pass-through rates: {ols_passthrough}")
print("  Figure 1 saved: fig1_ols_regression.png\n")
 
 


# In[23]:


# ══════════════════════════════════════════════════════════════════════════════
# ARIMA EMISSIONS FORECASTING
# ══════════════════════════════════════════════════════════════════════════════


ARIMA_ORDER      = (1, 1, 0)
FORECAST_HORIZON = 3

arima_results = []

for h in PORTFOLIO:
    ticker   = h["ticker"]
    last_val = h["emissions_intensity"]   # per $1000 revenue scale

    if ticker in emissions_history:
        hist = emissions_history[ticker]
        # Values from Excel are per $bn — convert to per $1000 for consistency
        series = np.array([hist[y] / 1_000_000 for y in sorted(hist.keys())])
        years  = sorted(hist.keys())
        source = "CarbonEmissions.xlsx"
    else:
        # synthetic fallback: 5% annual decay
        series = np.array([last_val * (0.95 ** i) for i in range(5, -1, -1)])
        years  = list(range(2018, 2024))
        source = "synthetic (5% decay)"

    last_year = years[-1]
    last_known = float(series[-1])

    try:
        model  = ARIMA(series, order=ARIMA_ORDER)
        fitted = model.fit()
        fcast  = list(fitted.forecast(steps=FORECAST_HORIZON))

        if any(f < 0 or f > series[-1] * 5 for f in fcast):
            raise ValueError("Forecast out of range")

        fcast  = [max(f, 1e-9) for f in fcast]
        method = f"ARIMA{ARIMA_ORDER}"

    except Exception:
        # Linear trend fallback
        x = np.arange(len(series))
        slope, intercept = np.polyfit(x, series, 1)
        fcast = [max(intercept + slope * (len(series) + i), 1e-9)
                 for i in range(FORECAST_HORIZON)]
        method = "linear_trend_fallback"

    pct = (fcast[-1] - last_known) / last_known * 100

    arima_results.append({
        "ticker":       ticker,
        "hist_years":   years,
        "hist_vals":    series.tolist(),
        "fore_years":   [last_year + i + 1 for i in range(FORECAST_HORIZON)],
        "fore_vals":    [round(f, 8) for f in fcast],
        "last_known":   round(last_known, 8),
        "pct_change":   round(pct, 1),
        "method":       method,
        "source":       source,
    })
    print(f"  {ticker:<6}  {last_known:.6f} → {fcast[-1]:.6f}  "
          f"({pct:+.1f}%)  [{method}]")

# Plot
n_hold = len(arima_results)
ncols  = 5
nrows  = (n_hold + ncols - 1) // ncols

fig2, axes2 = plt.subplots(nrows, ncols, figsize=(18, nrows * 3.4))
axes2 = axes2.flatten()

for i, res in enumerate(arima_results):
    ax  = axes2[i]
    col = SECTOR_COLORS.get(
        next(h["sector"] for h in PORTFOLIO if h["ticker"] == res["ticker"]), "steelblue"
    )
    ax.plot(res["hist_years"], res["hist_vals"],
            "o-", color=col, linewidth=2, markersize=4, label="Historical")
    ax.plot(
        [res["hist_years"][-1]] + res["fore_years"],
        [res["hist_vals"][-1]]  + res["fore_vals"],
        "o--", color=col, linewidth=1.5, markersize=4,
        alpha=0.6, label="Forecast",
    )
    ax.set_title(f'{res["ticker"]}\n({res["pct_change"]:+.1f}%)', fontsize=9)
    ax.set_xlabel("Year", fontsize=7)
    ax.set_ylabel("Emissions\nintensity", fontsize=7)
    ax.tick_params(labelsize=7)

for j in range(i + 1, len(axes2)):
    axes2[j].set_visible(False)

solid_line = plt.Line2D([0], [0], color="gray", linewidth=2, label="Historical")
dash_line  = plt.Line2D([0], [0], color="gray", linewidth=1.5,
                        linestyle="--", alpha=0.6, label="Forecast (+3yr)")
fig2.legend(handles=[solid_line, dash_line], loc="lower right", fontsize=9)

fig2.suptitle(
    "ARIMA Emissions Intensity Forecasts\n"
    f"(tCO₂e per $1,000 revenue — sourced from CarbonEmissions.xlsx, {ARIMA_ORDER} model)",
    fontsize=13, fontweight="bold",
)
plt.tight_layout()
plt.savefig("fig2_arima_forecasts.png", dpi=150, bbox_inches="tight")
plt.show()
print("  Figure 2 saved: fig2_arima_forecasts.png\n")


# In[24]:


# ══════════════════════════════════════════════════════════════════════════════
# HMM REGIME DETECTION VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════
 
 
try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    print("  WARNING: hmmlearn not installed — skipping Figure 4")
    print("  Install with: pip install hmmlearn\n")
 
REGIME_COLORS = {
    0: "#22c55e",   # low volatility — green
    1: "#f97316",   # shock transition — orange
    2: "#ef4444",   # price spike — red
}
REGIME_LABELS = {
    0: "Low Volatility\n→ Current Policies (£42/t)",
    1: "Shock Transition\n→ Delayed Transition (£80/t)",
    2: "Price Spike\n→ Net Zero 2050 (£200/t)",
}
REGIME_NGFS = {
    0: "Current Policies (£42/t)",
    1: "Delayed Transition (£80/t)",
    2: "Net Zero 2050 (£200/t)",
}
 
if HMM_AVAILABLE:
    # Use Phase 3 data (2013+)
    df_hmm = df_eua[df_eua["year"] >= 2013].copy()
    prices_hmm = df_hmm["price_gbp"].values
    dates_hmm  = df_hmm["Date"].values
 
    log_returns = np.log(prices_hmm[1:] / prices_hmm[:-1]).reshape(-1, 1)
    dates_lr    = dates_hmm[1:]   # one shorter because of differencing
 
    print(f"  Training HMM on {len(log_returns)} monthly log returns...")
 
    # 10 restarts — pick best log-likelihood
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
        print("  HMM did not converge — skipping Figure 4")
        HMM_AVAILABLE = False
    else:
        model = best_model
 
        # Sort states by VARIANCE
        variances  = model.covars_.flatten()
        state_order = np.argsort(variances)
        regime_map  = {state_order[i]: i for i in range(3)}
 
        # Predict regime sequence and current regime
        raw_states     = model.predict(log_returns)
        labeled_states = np.array([regime_map[s] for s in raw_states])
        current_raw    = int(raw_states[-1])
        current_regime = regime_map[current_raw]
 
        probs      = model.predict_proba(log_returns)
        confidence = float(probs[-1][current_raw])
 
        print(f"  Detected regime: {REGIME_NGFS[current_regime]}  "
              f"(confidence: {confidence:.2f})")
        print(f"  Log-likelihood: {best_score:.1f}  "
              f"(best of 10 restarts)")
 
        # ── Figure 4: three-panel HMM chart ──────────────────────────────────
        fig4 = plt.figure(figsize=(16, 9))
        gs   = fig4.add_gridspec(
            2, 2,
            height_ratios=[2.2, 1],
            hspace=0.40, wspace=0.30,
        )
        ax_price  = fig4.add_subplot(gs[0, :])   # full-width top
        ax_counts = fig4.add_subplot(gs[1, 0])   # bottom left
        ax_now    = fig4.add_subplot(gs[1, 1])   # bottom right
 
        # ── Panel 1: EUA price coloured by regime ────────────────────────────
        # Plot the full price history (including pre-2013 in grey for context)
        df_pre = df_eua[df_eua["year"] < 2013]
        if len(df_pre) > 0:
            ax_price.plot(df_pre["Date"], df_pre["price_gbp"],
                          color="#cbd5e1", linewidth=1.0,
                          label="Pre-Phase 3 (not used in HMM)", zorder=1)
 
        # Plot each Phase 3+ segment coloured by assigned regime
        for regime_id in range(3):
            mask = labeled_states == regime_id
            if not mask.any():
                continue
            # Plot as scatter so gaps between regimes don't get connected
            ax_price.scatter(
                dates_lr[mask],
                prices_hmm[1:][mask],
                color=REGIME_COLORS[regime_id],
                s=14, zorder=3, alpha=0.85,
                label=f"Regime {regime_id}: {REGIME_NGFS[regime_id]}",
            )
 
        # Overlay continuous price line in light grey behind the colours
        ax_price.plot(dates_lr, prices_hmm[1:],
                      color="#e2e8f0", linewidth=0.8, zorder=2)
 
        # Mark current point
        ax_price.axvline(
            x=dates_lr[-1], color=REGIME_COLORS[current_regime],
            linewidth=2.0, linestyle="--", alpha=0.9, zorder=4,
        )
        ax_price.annotate(
            f" Current regime: {REGIME_NGFS[current_regime]}\n"
            f" Confidence: {confidence:.0%}",
            xy=(dates_lr[-1], prices_hmm[-1]),
            xytext=(-160, -35),
            textcoords="offset points",
            fontsize=8.5,
            color=REGIME_COLORS[current_regime],
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=REGIME_COLORS[current_regime], alpha=0.9),
            arrowprops=dict(arrowstyle="->",
                            color=REGIME_COLORS[current_regime], lw=1.5),
        )
 
        ax_price.set_ylabel("EUA Carbon Price (£/tonne CO₂)", fontsize=10)
        ax_price.set_title(
            "EUA Carbon Price History Coloured by HMM Regime\n"
            "(Gaussian HMM, 3 states, sorted by variance — 10 restarts)",
            fontsize=11, fontweight="bold",
        )
        ax_price.legend(fontsize=8.5, loc="upper left")
        ax_price.tick_params(labelsize=9)
 
        # ── Panel 2: Months per regime ────────────────────────────────────────
        regime_counts = [int((labeled_states == r).sum()) for r in range(3)]
        bars = ax_counts.bar(
            range(3),
            regime_counts,
            color=[REGIME_COLORS[r] for r in range(3)],
            width=0.55, edgecolor="white", linewidth=0.8,
        )
        for bar, count in zip(bars, regime_counts):
            ax_counts.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.8,
                f"{count}\nmonths",
                ha="center", va="bottom", fontsize=9,
            )
        ax_counts.set_xticks(range(3))
        ax_counts.set_xticklabels(
            [f"Regime {r}\n{REGIME_NGFS[r]}" for r in range(3)],
            fontsize=8,
        )
        ax_counts.set_ylabel("Months in regime", fontsize=9)
        ax_counts.set_title("Time Spent in Each Regime\n(Phase 3: 2013–present)",
                             fontsize=10, fontweight="bold")
        ax_counts.tick_params(labelsize=8)
 
        # ── Panel 3: Current state indicator ─────────────────────────────────
        ax_now.set_xlim(0, 1)
        ax_now.set_ylim(0, 1)
        ax_now.axis("off")
 
        # Coloured circle for current regime
        circle = plt.Circle(
            (0.5, 0.62), 0.22,
            color=REGIME_COLORS[current_regime], alpha=0.85,
        )
        ax_now.add_patch(circle)
        ax_now.text(0.5, 0.62, f"{current_regime}",
                    ha="center", va="center",
                    fontsize=28, fontweight="bold", color="white")
 
        ax_now.text(0.5, 0.93, "Current Detected Regime",
                    ha="center", va="top", fontsize=10, fontweight="bold",
                    color="#1e293b")
        ax_now.text(0.5, 0.36,
                    REGIME_LABELS[current_regime],
                    ha="center", va="top", fontsize=9.5,
                    color=REGIME_COLORS[current_regime], fontweight="bold")
        ax_now.text(0.5, 0.16,
                    f"Confidence: {confidence:.0%}",
                    ha="center", va="top", fontsize=9,
                    color="#475569")
        ax_now.text(0.5, 0.05,
                    f"Log-likelihood: {best_score:.1f}",
                    ha="center", va="top", fontsize=8,
                    color="#94a3b8")
 
        fig4.suptitle(
            "Module 3: Gaussian HMM Carbon Market Regime Detection\n"
            "EUA monthly log returns (2013–present) — regime mapped to NGFS Phase V scenario",
            fontsize=12, fontweight="bold", y=0.98,
        )
 
        plt.savefig("fig4_hmm_regimes.png", dpi=150, bbox_inches="tight")
        plt.show()
        print("  Figure 4 saved: fig4_hmm_regimes.png\n")
 
else:
    print("  Figure 4 skipped (hmmlearn not available)\n")
 


# In[25]:


# ══════════════════════════════════════════════════════════════════════════════
# PORTFOLIO OPTIMISER RESULTS
# ══════════════════════════════════════════════════════════════════════════════


# Build forecast dict from ARIMA results
forecast_dict = {r["ticker"]: r["fore_vals"][-1] for r in arima_results}

# ── EaR formula (matches ear_engine.py compute_ear) ─────────────────────────
def compute_eps_impact(holding, scenario, passthrough, forecasts):
    ticker  = holding["ticker"]
    sector  = holding["sector"]
    ei      = forecasts.get(ticker, holding["emissions_intensity"])
    delta_c = scenario["carbon_price"] - 25

    carbon  = ei * delta_c / 1000
    subsidy = (scenario["subsidy_removal"] * 0.06 if sector == "Energy"
               else scenario["subsidy_removal"] * 0.02 if sector == "Materials"
               else 0.0)
    energy  = ei * scenario["energy_shock"] * 0.3
    bca     = (scenario["bca"] * ei * 0.4
               if sector in ("Energy", "Materials", "Industrials") else 0.0)

    gross = carbon + subsidy + energy + bca
    pt    = passthrough.get(sector, FALLBACK_PASSTHROUGH.get(sector, 0.5))
    net   = gross * (1.0 - pt)
    mc    = net / (holding["ebitda_margin"] + 1e-9)
    eps   = min(mc * 0.75, 0.95)
    return float(eps)

for h in PORTFOLIO:
    h["eps_impact"] = compute_eps_impact(
        h, SCENARIO_NZ2050, ols_passthrough, forecast_dict
    )

# covariance matrix ───────────────────────────────────────────────
n    = len(PORTFOLIO)
w0   = np.array([h["weight"]      for h in PORTFOLIO])
ear  = np.array([h["eps_impact"]  for h in PORTFOLIO])
beta = np.array([h["beta"]        for h in PORTFOLIO]).reshape(-1, 1)

SIGMA_M = 0.20
sigma_e = np.maximum(ear * 0.5, 0.05)
COV     = (SIGMA_M ** 2) * (beta @ beta.T) + np.diag(sigma_e ** 2)

LAMBDA_EAR = 0.70
LAMBDA_VOL = 0.30
MIN_WEIGHT = 0.005

def objective(w):
    return LAMBDA_EAR * float(np.dot(w, ear)) + LAMBDA_VOL * float(w @ COV @ w)

# position cap: min(w0 * 3, 20%)
max_weights = np.minimum(w0 * 3.0, 0.20)
max_weights = np.maximum(max_weights, MIN_WEIGHT * 2)
bounds = [(MIN_WEIGHT, float(mx)) for mx in max_weights]

result = minimize(
    objective, w0, method="SLSQP",
    bounds=bounds,
    constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}],
    options={"maxiter": 1000, "ftol": 1e-9},
)

unconstrained = result.x if result.success else w0
deltas = unconstrained - w0

TURNOVER_LIMIT = 0.25

def scale_measure(s):
    shifted  = np.maximum(w0 + deltas * s, MIN_WEIGHT)
    normed   = shifted / shifted.sum()
    turnover = float(np.sum(np.abs(normed - w0)))
    return normed, turnover

final_w, actual_to = scale_measure(1.0)
if actual_to > TURNOVER_LIMIT:
    lo, hi = 0.0, 1.0
    for _ in range(100):
        s_mid = (lo + hi) / 2.0
        w_mid, t_mid = scale_measure(s_mid)
        if t_mid <= TURNOVER_LIMIT:
            lo, final_w, actual_to = s_mid, w_mid, t_mid
        else:
            hi = s_mid
        if (hi - lo) < 1e-8:
            break

orig_ear  = float(np.dot(w0,     ear))
opt_ear   = float(np.dot(final_w, ear))
reduction = (orig_ear - opt_ear) / orig_ear * 100
w_deltas  = (final_w - w0) * 100

print(f"  scipy success: {result.success}")
print(f"  EaR: {orig_ear*100:.2f}% → {opt_ear*100:.2f}%  "
      f"(reduction: {reduction:.1f}%)")
print(f"  Actual turnover: {actual_to*100:.1f}%  (limit: {TURNOVER_LIMIT*100:.0f}%)")

# ── Plot ─────────────────────────────────────────────────────────────────────
tickers_disp = [h["ticker"] for h in PORTFOLIO]
bar_colors   = [SECTOR_COLORS[h["sector"]] for h in PORTFOLIO]
orig_pct     = w0     * 100
opt_pct      = final_w * 100

fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

x     = np.arange(n)
width = 0.36
ax1.bar(x - width / 2, orig_pct, width, color=bar_colors, alpha=0.45, label="Original")
ax1.bar(x + width / 2, opt_pct,  width, color=bar_colors, alpha=1.00, label="Optimised")
ax1.set_xticks(x)
ax1.set_xticklabels(tickers_disp, rotation=45, ha="right", fontsize=9)
ax1.set_ylabel("Portfolio weight (%)")
ax1.set_title("Before vs After Rebalancing")
ax1.legend(fontsize=9)

# Add sector legend
sector_patches = [
    mpatches.Patch(color=c, label=s) for s, c in SECTOR_COLORS.items()
    if any(h["sector"] == s for h in PORTFOLIO)
]
ax1.legend(
    handles=[
        mpatches.Patch(facecolor="gray", alpha=0.45, label="Original"),
        mpatches.Patch(facecolor="gray", alpha=1.00, label="Optimised"),
    ] + sector_patches,
    fontsize=7.5, ncol=2, loc="upper right",
)

sorted_idx     = np.argsort(w_deltas)
sorted_tickers = [tickers_disp[i] for i in sorted_idx]
sorted_deltas  = [w_deltas[i]     for i in sorted_idx]
delta_colors   = [
    "#22c55e" if d > 0.05 else "#ef4444" if d < -0.05 else "#94a3b8"
    for d in sorted_deltas
]
ax2.barh(sorted_tickers, sorted_deltas, color=delta_colors, height=0.6)
ax2.axvline(x=0, color="black", linewidth=0.8)
ax2.set_xlabel("Weight change (percentage points)")
ax2.set_title("Recommended Weight Changes\n(green = increase, red = reduce)")
ax2.tick_params(labelsize=9)

fig3.suptitle(
    f"Portfolio Optimisation — EaR reduced by {reduction:.1f}%\n"
    f"Net Zero 2050 (£200/t)  ·  25% turnover limit  ·  "
    f"scipy SLSQP with covariance matrix",
    fontsize=12, fontweight="bold",
)
plt.tight_layout()
plt.savefig("fig3_optimisation.png", dpi=150, bbox_inches="tight")
plt.show()
print("  Figure 3 saved: fig3_optimisation.png\n")


# In[26]:


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE 
# ══════════════════════════════════════════════════════════════════════════════
 
print("=" * 60)
print("SUMMARY")
print("=" * 60)
 
print("\n OLS Pass-Through Rates:")
for sector, details in sorted(ols_details.items()):
    r2_str = f"R²={details['r2']:.3f}" if details["r2"] else "fallback"
    print(f"  {sector:<14} {details['pass_through']:.0%}  ({r2_str})")
 
print("\n ARIMA Forecast Summary (Net Zero 2050):")
print(f"  {'Ticker':<7} {'2023':>10} {'Forecast':>10} {'Change':>8}")
for r in sorted(arima_results, key=lambda x: x["pct_change"]):
    print(f"  {r['ticker']:<7} {r['last_known']:>10.6f} "
          f"{r['fore_vals'][-1]:>10.6f} {r['pct_change']:>+7.1f}%")
 
print("\n Optimisation Results (Net Zero 2050, 25% turnover):")
print(f"  Original EaR:   {orig_ear*100:.2f}%")
print(f"  Optimised EaR:  {opt_ear*100:.2f}%")
print(f"  EaR Reduction:  {reduction:.1f}%")
print(f"  Actual Turnover: {actual_to*100:.1f}%")
print(f"  scipy success:  {result.success}")
 
print("\nFigures saved:")
print("  fig1_ols_regression.png")
print("  fig2_arima_forecasts.png")
print("  fig3_optimisation.png")
print("=" * 60)
 
# if HMM_AVAILABLE:
#     print("\n HMM Regime Detection:")
#     print(f"  Current regime:  {REGIME_NGFS[current_regime]}")
#     print(f"  Confidence:      {confidence:.0%}")
#     print(f"  Log-likelihood:  {best_score:.1f}")
#     for r in range(3):
#         print(f"  Regime {r} ({REGIME_NGFS[r]:<30}): "
#               f"{regime_counts[r]} months")
#     print("  fig4_hmm_regimes.png     — Module 3 (HMM)")
 

