# Climate Transition EaR Engine
**Stirling Solvers — CFA Institute AI Investment Challenge 2026**

A three-layer quantitative decision-support engine that converts regulatory policy assumptions into portfolio earnings impact, then solves for the optimal rebalancing response.

---

## Quick Start

```bash
git clone https://github.com/1yr1c/ear-engine
cd ear-engine
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key_here
python app.py
```

Then open http://localhost:5000

---

## Architecture

### Layer 1 — Regulatory Shock Simulation
Five NGFS Phase V (Nov 2024) aligned scenarios from Current Policies (£42/t) to Disorderly Fragmented (£300/t). Each scenario carries four parameters: carbon price, subsidy removal rate, border carbon adjustment, and energy input shock.

### Layer 2 — Earnings-at-Risk Engine
Per-holding formula chain:
1. `carbon_cost_impact = emissions_intensity × Δcarbon_price / 1000`
2. `+ subsidy_impact + energy_input_impact + bca_impact = total_cost_increase`
3. `× (1 − pass_through_rate) = net_cost_increase`
4. `÷ ebitda_margin = margin_compression`
5. `× 0.75 (tax/leverage adjustment) = EPS at Risk` (capped at 95%)
6. `× portfolio_weight = holding EaR contribution`

### Layer 3 — Portfolio Optimiser
scipy SLSQP mean-variance optimiser minimising λ·EaR + (1−λ)·w'Σw subject to a user-defined turnover constraint. Uses a single-factor covariance matrix built from beta and EaR-scaled idiosyncratic risk to prevent concentration into a single low-EaR holding.

---

## ML Pipeline

All four ML modules run in a **background thread at startup** so Flask binds its port immediately. The app is fully usable from the first request using fallback values. Full ML outputs are available within ~10–20 seconds of startup.

### Startup Sequence
```
App starts
    ↓ immediately
Flask binds port — app usable with fallbacks
    ↓ background thread
ML modules run at full quality
    ↓ ~10–20 seconds later
PASSTHROUGH_RATES, EMISSIONS_FORECASTS, DETECTED_SCENARIO updated
All subsequent requests use full ML outputs
```

Check `/ml/status` to see whether ML has finished initialising (`ml_ready: true`).

---

### Module 1 — OLS Pass-Through Regression
**File:** `ml_modules.py → _run_ols()`  
**Library:** scikit-learn `LinearRegression`  
**Input:** Annual average EUA carbon price (£/t) vs sector EBITDA margins  
**Output:** `PASSTHROUGH_RATES` — dict of `{sector: rate}` for 7 sectors  
**Where used:** `pass_through` field in `compute_holding_ear()` in `ear_engine.py`  
**Effect on UI:** Pass-thru column, Net Cost Δ, Margin Compression, EPS at Risk  

**Method:** Regresses annual EUA prices against EBITDA margins per sector. Estimates how much of a carbon cost increase each sector can pass on to customers vs absorb. Negative coefficient = margins compress as carbon rises = company absorbs cost. Falls back to Fabra & Reguant (2014) literature rates when R² < 0.05.

**Fallback:** Fabra & Reguant (2014) sector rates (Energy: 0.35, Materials: 0.18, Industrials: 0.62, Healthcare: 0.82, Financials: 0.91, Consumer: 0.72, Utilities: 0.85)

**To improve:** Provide real annual EBITDA margin history per holding for 2018–2023 (source: Macrotrends.net or company annual reports). Currently uses single-year cross-section which limits R².

---

### Module 2 — ARIMA Emissions Intensity Forecasting
**File:** `ml_modules.py → _run_arima()`  
**Library:** statsmodels `ARIMA`  
**Order:** (1, 1, 0) — one lag, one differencing, no moving average  
**Input:** Historical Scope 1+2 emissions intensity per holding (tCO2e / revenue $bn)  
**Output:** `EMISSIONS_FORECASTS` — dict of `{ticker: forecasted_intensity}`  
**Where used:** `emis` variable in `compute_holding_ear()` in `ear_engine.py`  
**Effect on UI:** Emis.Int column, Net Cost Δ, Margin Compression, EPS at Risk  

**Method:** Fits ARIMA(1,1,0) on each holding's 6-year emissions intensity history and projects 3 years forward. ARIMA(1,1,0) chosen to avoid overfitting on short series. Includes sanity check — if forecast goes negative or more than triples, falls back to linear trend extrapolation.

**Fallback:** Linear trend extrapolation on synthetic 5% annual decay series (placeholder until real historical data provided)

**To improve:** Provide real Scope 1+2 emissions and revenue data for 2018–2023 per holding from CDP Open Data (cdp.net) or company sustainability reports. See `ear_engine.py` PORTFOLIO dict for sources per holding.

---

### Module 3 — Gaussian HMM Carbon Market Regime Detection
**File:** `ml_modules.py → _run_hmm()`  
**Library:** hmmlearn `GaussianHMM`  
**Input:** Monthly EUA carbon price log returns from `EUA_Yearly_futures.csv` (Phase 3+, 2013 onwards)  
**Output:** `DETECTED_SCENARIO` — NGFS scenario dict with regime label, carbon price, and confidence  
**Where used:** Default scenario selection on page load; `_DEFAULT_SCENARIO_ID` in `app.py`; `/ml/status` endpoint  
**Effect on UI:** "Use ML Recommended" button; pre-selected scenario on load  

**Method:** Trains a 3-component Gaussian HMM on monthly EUA log returns. Runs 10 random initialisations and selects the best log-likelihood to avoid local optima. States are ordered by variance (ascending) rather than mean — more stable for carbon price data where two regimes can have similar mean returns but different volatility. Regime collapse check included.

**Regime mapping:**
| Regime | Characteristic | NGFS Scenario | Carbon Price |
|--------|---------------|---------------|-------------|
| 0 | Low volatility | Current Policies | £42/t |
| 1 | Shock transition | Delayed Transition | £80/t |
| 2 | Price spike | Orderly Net Zero 2050 | £200/t |

**Fallback:** Orderly Below 2°C (£120/t) if hmmlearn unavailable or convergence fails

**Design note:** Pre-2013 EUA prices excluded — near-zero Phase 1/2 prices would distort the regime detection by creating a spurious fourth regime.

---

### Module 4 — scipy SLSQP Mean-Variance Portfolio Optimiser
**File:** `ml_modules.py → run_scipy_optimiser()`  
**Library:** scipy `minimize` with SLSQP method  
**Input:** Per-holding `weight`, `eps_impact` (EaR from M1+M2), `beta`  
**Output:** `final_weights` array, EaR reduction %, actual turnover  
**Where used:** `optimise_portfolio()` in `ear_engine.py` — replaces the inverse-EaR bisection  
**Effect on UI:** Optimiser tab — Opt. Weight column, Delta column, EaR Reduction %  

**Objective:** Minimise `λ·portfolio_EaR + (1−λ)·w'Σw`  
**Default λ:** 0.70 (70% weight on EaR minimisation, 30% on variance)  

**Covariance matrix:** Single-factor model `Cov = β·β'·σ_m² + diag(σ_e²)`  
- Market factor: `σ_m = 0.20` (20% annualised market vol)  
- Idiosyncratic: `σ_e = max(EaR × 0.5, 0.05)` — high EaR holdings get higher idiosyncratic risk, penalising concentration  

**Constraints:**
- Weights sum to 1
- Minimum position: 0.5%
- Maximum position: min(3× original weight, 20%)
- Turnover ≤ user-defined limit (bisection-enforced)

**Fallback:** Inverse-EaR bisection (original method) if scipy unavailable

---

## ML Status Endpoint

`GET /ml/status` returns:

```json
{
  "ml_active": true,
  "ml_ready": true,
  "detected_scenario": {
    "id": "base",
    "label": "Current Policies",
    "carbon_price": 42,
    "regime_name": "Low Volatility",
    "confidence": 0.87,
    "current_regime": 0,
    "n_observations": 143,
    "model_log_likelihood": 312.4,
    "source": "GaussianHMM on EUA monthly log returns — best of 10 restarts"
  },
  "passthrough_rates": {
    "Energy": 0.35,
    "Materials": 0.18,
    "Industrials": 0.62,
    "Healthcare": 0.82,
    "Financials": 0.91,
    "Consumer": 0.72,
    "Utilities": 0.85
  },
  "emissions_forecasts": {
    "SHEL": 0.162,
    "BP": 0.146,
    ...
  }
}
```

`ml_ready: false` means the background thread is still running — fallback values are in use. `ml_ready: true` means all modules have completed.

---

## Data Sources

| Data | Source | Used by |
|------|--------|---------|
| EUA monthly carbon prices | investing.com EUA Yearly Futures + EUR/GBP conversion | M3 HMM |
| Emissions intensity 2023 | Company sustainability reports (see PORTFOLIO in ear_engine.py) | M2 ARIMA baseline |
| EBITDA margins 2023 | Company annual reports 2023 | M1 OLS, EaR engine |
| Pass-through fallback rates | Fabra & Reguant (2014), American Economic Review | M1 fallback |
| NGFS scenarios | NGFS Phase V (November 2024) | All scenarios |
| Carbon price baseline | UK ETS Authority (2025) — £41.84/t for 2025 scheme year | EaR engine |
| Validation benchmark | Bank of England CBES 2021 | EaR engine validation |

---

## Improving ML Quality

To fully activate Modules 1 and 2 with real data:

**Module 1 — EBITDA margin history (2018–2023):**
- Source: Macrotrends.net — search any ticker for annual margin history
- Required: Annual gross or EBITDA margin per holding for 6 years
- Effort: ~1 hour for 15 holdings

**Module 2 — Emissions intensity history (2018–2023):**
- Source: CDP Open Data (cdp.net/en/responses) or company sustainability reports
- Required: Scope 1+2 tCO2e and revenue ($bn) per holding per year
- Effort: ~2–3 hours for 15 holdings × 6 years = 90 data points

**Module 4 — Beta estimates:**
- Source: `yfinance` Python library — one line per holding
- Required: 3–5 year weekly returns vs FTSE 100 index
- Effort: ~30 minutes with a script

---

## Portfolio CSV Format

Upload your own portfolio via the web interface. Required columns:

| Column | Type | Description |
|--------|------|-------------|
| ticker | string | Stock ticker |
| name | string | Company name |
| sector | string | Energy / Materials / Industrials / Healthcare / Financials / Consumer / Utilities |
| weight | float | Portfolio weight (must sum to 1.0 ±0.02) |
| emissions_intensity | float | Scope 1+2 tCO2e ÷ revenue ($bn) |
| ebitda_margin | float | EBITDA as fraction of revenue |
| pass_through | float | Cost pass-through rate (0–1) |

Optional: `beta`, `source`

See `portfolio_template.csv` for an example.

---

## Team

| Name | Role |
|------|------|
| Oliver | Architecture, ML integration, Flask app |
| Tracey | ML modules development |
| Agnes | Emissions data, EUA data, scenario validation |
| Michael | Narrative, presentation |

University of Stirling

---

## Licence

MIT Licence — see LICENSE file.

---

## Disclaimer

This tool is for research and educational purposes only. All AI-generated content is clearly labelled. Nothing in this tool constitutes investment advice.
