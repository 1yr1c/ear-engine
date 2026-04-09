# ============================================================
# ear_engine.py — Climate Transition EaR Engine
# Stirling Solvers — CFA AI Investment Challenge 2026
#
# This is the core quantitative module. It does three things:
#   1. Defines the default portfolio of 15 FTSE 100 holdings
#   2. Defines the 5 NGFS climate scenarios
#   3. Runs the full EaR formula chain and portfolio optimiser
#
# The formula chain converts a carbon price assumption into
# a per-holding EPS impact, then sums to a portfolio-level
# Earnings-at-Risk (EaR) figure.
# ============================================================


# ── DEFAULT PORTFOLIO ────────────────────────────────────────────────────────
# 15 FTSE 100 holdings representing a diversified UK equity portfolio.
# Weights sum to 1.0 (100% of portfolio NAV).
# Each holding includes:
#   - emissions_intensity: Scope 1+2 tCO2e per $1bn revenue (from sustainability reports)
#   - ebitda_margin: EBITDA as a fraction of revenue (from annual reports)
#   - pass_through: fraction of carbon cost the company can pass to customers (0–1)
#   - beta: market sensitivity vs FTSE 100 (used in Module 4 covariance matrix)
# Sources cited per holding for audit trail.

PORTFOLIO = [
    {"ticker": "SHEL", "name": "Shell plc",                "sector": "Energy",      "weight": 0.117, "emissions_intensity": 0.180, "ebitda_margin": 0.182, "pass_through": 0.35, "beta": 1.10, "source": "Shell Sustainability Report 2023"},
    {"ticker": "BP",   "name": "BP plc",                   "sector": "Energy",      "weight": 0.064, "emissions_intensity": 0.162, "ebitda_margin": 0.148, "pass_through": 0.30, "beta": 1.05, "source": "BP ESG Datasheet 2023"},
    {"ticker": "GLEN", "name": "Glencore plc",             "sector": "Materials",   "weight": 0.057, "emissions_intensity": 0.227, "ebitda_margin": 0.098, "pass_through": 0.15, "beta": 1.30, "source": "Glencore Sustainability Report 2023"},
    {"ticker": "RIO",  "name": "Rio Tinto plc",            "sector": "Materials",   "weight": 0.054, "emissions_intensity": 0.603, "ebitda_margin": 0.420, "pass_through": 0.20, "beta": 0.95, "source": "Rio Tinto Climate Report 2023"},
    {"ticker": "AAL",  "name": "Anglo American plc",       "sector": "Materials",   "weight": 0.031, "emissions_intensity": 0.567, "ebitda_margin": 0.320, "pass_through": 0.18, "beta": 1.20, "source": "Anglo American ESG Report 2023"},
    {"ticker": "BA.",  "name": "BAE Systems plc",          "sector": "Industrials", "weight": 0.079, "emissions_intensity": 0.015, "ebitda_margin": 0.115, "pass_through": 0.65, "beta": 0.70, "source": "BAE Systems ESG Data 2023"},
    {"ticker": "RR.",  "name": "Rolls-Royce Holdings",     "sector": "Industrials", "weight": 0.069, "emissions_intensity": 0.060, "ebitda_margin": 0.092, "pass_through": 0.60, "beta": 1.40, "source": "Rolls-Royce ESG Report 2023"},
    {"ticker": "AZN",  "name": "AstraZeneca plc",          "sector": "Healthcare",  "weight": 0.113, "emissions_intensity": 0.018, "ebitda_margin": 0.320, "pass_through": 0.85, "beta": 0.50, "source": "AstraZeneca ESG Data Summary 2023"},
    {"ticker": "GSK",  "name": "GSK plc",                  "sector": "Healthcare",  "weight": 0.054, "emissions_intensity": 0.030, "ebitda_margin": 0.300, "pass_through": 0.80, "beta": 0.55, "source": "GSK ESG Performance Report 2023"},
    {"ticker": "HSBA", "name": "HSBC Holdings plc",        "sector": "Financials",  "weight": 0.093, "emissions_intensity": 0.010, "ebitda_margin": 0.360, "pass_through": 0.92, "beta": 1.00, "source": "HSBC ESG Report 2023"},
    {"ticker": "BARC", "name": "Barclays plc",             "sector": "Financials",  "weight": 0.060, "emissions_intensity": 0.004, "ebitda_margin": 0.280, "pass_through": 0.90, "beta": 1.15, "source": "Barclays TCFD Report 2023"},
    {"ticker": "LLOY", "name": "Lloyds Banking Group",     "sector": "Financials",  "weight": 0.046, "emissions_intensity": 0.003, "ebitda_margin": 0.310, "pass_through": 0.93, "beta": 1.10, "source": "Lloyds ESG Report 2023"},
    {"ticker": "ULVR", "name": "Unilever plc",             "sector": "Consumer",    "weight": 0.071, "emissions_intensity": 0.025, "ebitda_margin": 0.172, "pass_through": 0.70, "beta": 0.60, "source": "Unilever Annual Report 2023"},
    {"ticker": "BATS", "name": "British American Tobacco", "sector": "Consumer",    "weight": 0.043, "emissions_intensity": 0.104, "ebitda_margin": 0.430, "pass_through": 0.75, "beta": 0.45, "source": "BAT ESG Report 2023"},
    {"ticker": "NG.",  "name": "National Grid plc",        "sector": "Utilities",   "weight": 0.049, "emissions_intensity": 0.476, "ebitda_margin": 0.380, "pass_through": 0.85, "beta": 0.40, "source": "National Grid ESG Data 2023"},
]


# ── NGFS SCENARIOS ────────────────────────────────────────────────────────────
# Five climate transition scenarios aligned with NGFS Phase V (November 2024).
# Each scenario defines four policy shock parameters:
#   - carbon_price: UK ETS carbon price in £/tonne CO2e
#   - subsidy_removal: fraction of fossil fuel subsidies removed (0–1)
#   - bca: Border Carbon Adjustment rate — import tariff on carbon-intensive goods
#   - energy_shock: proportional increase in energy input costs (0–1)
# Scenarios range from "do nothing" (base) to "worst-case disorderly" (fragmented).
# Carbon price baseline: UK ETS official rate £41.84/t for 2025 scheme year.
# Source: UK ETS Authority (2025); NGFS Phase V scenarios (Nov 2024).

SCENARIOS = {
    "base": {
        "label": "Current Policies",
        "carbon_price": 42,          # £/tonne — current UK ETS rate
        "subsidy_removal": 0.0,      # no subsidy reform
        "bca": 0.0,                  # no border carbon adjustment
        "energy_shock": 0.0,         # no additional energy cost shock
        "temp": "2.9C",
        "description": "NGFS Current Policies. UK ETS official rate £41.84/t (2025). No additional transition shock."
    },
    "delayed": {
        "label": "Delayed Transition",
        "carbon_price": 80,          # emissions flat until 2030, then sharp rise
        "subsidy_removal": 0.20,     # partial subsidy removal
        "bca": 0.06,                 # early EU CBAM-style border adjustment
        "energy_shock": 0.10,        # modest energy cost increase
        "temp": "1.8C",
        "description": "NGFS Delayed Transition. Emissions flat until 2030 then sharp acceleration. Carbon ~£80/t."
    },
    "below2": {
        "label": "Orderly Below 2C",
        "carbon_price": 120,         # coordinated early policy action
        "subsidy_removal": 0.35,
        "bca": 0.12,
        "energy_shock": 0.18,
        "temp": "1.7C",
        "description": "NGFS Below 2C Orderly. Early coordinated policy action. Carbon ~£120/t."
    },
    "netzero": {
        "label": "Orderly Net Zero 2050",
        "carbon_price": 200,         # full decarbonisation pathway
        "subsidy_removal": 0.55,
        "bca": 0.20,                 # EU CBAM fully operational
        "energy_shock": 0.28,
        "temp": "1.5C",
        "description": "NGFS Net Zero 2050. Full subsidy reform, EU CBAM operational. Carbon ~£200/t."
    },
    "fragmented": {
        "label": "Disorderly Fragmented",
        "carbon_price": 300,         # highest cost — uncoordinated abrupt policy
        "subsidy_removal": 0.80,
        "bca": 0.40,
        "energy_shock": 0.40,
        "temp": "2.0C",
        "description": "NGFS Disorderly Fragmented. Divergent uncoordinated policy. Carbon ~£300/t."
    },
}

import numpy as np
import csv

# The carbon price baseline represents pre-ETS levels.
# Delta carbon = scenario carbon price minus this baseline,
# so the formula captures only the incremental policy shock.
BASELINE_CARBON = 25  # £/tonne — pre-ETS reference price

# Try to import ML outputs at module load time.
# If ml_modules is available, Module 1 (OLS) and Module 2 (ARIMA) outputs
# will be used in the EaR formula — otherwise static portfolio values are used.
try:
    from ml_modules import PASSTHROUGH_RATES, EMISSIONS_FORECASTS, run_scipy_optimiser
    ML_AVAILABLE = True
except Exception:
    ML_AVAILABLE = False
    PASSTHROUGH_RATES   = {}   # fallback: use static pass_through from PORTFOLIO
    EMISSIONS_FORECASTS = {}   # fallback: use static emissions_intensity from PORTFOLIO


def compute_holding_ear(holding, scenario, pass_through_override=None):
    """
    Compute Earnings-at-Risk for a single holding under a given climate scenario.

    This is the core formula chain. It works in 9 steps:

    Step 1 — Carbon cost impact
        How much does a higher carbon price increase this company's direct costs?
        = emissions_intensity × delta_carbon / 1000
        (divide by 1000 to convert from tCO2e/$bn to a fraction of revenue)

    Step 2 — Subsidy removal impact
        Energy and Materials companies receive fossil fuel subsidies.
        Removing these adds a further cost on top of the carbon price.

    Step 3 — Energy input shock
        Higher carbon prices raise energy costs across the economy.
        This hits all companies in proportion to their emissions intensity.

    Step 4 — Border Carbon Adjustment (BCA)
        Traded sectors (Energy, Materials, Industrials) face import tariffs
        from carbon-intensive trading partners under schemes like EU CBAM.

    Step 5 — Total gross cost increase
        Sum of all four cost channels above.

    Step 6 — Net cost after pass-through
        Not all of this cost lands on the company — some is passed to customers
        via higher prices. pass_through = 1 means 100% passed on (no EaR).
        pass_through = 0 means the company absorbs everything.
        net_cost = total_cost × (1 - pass_through)

    Step 7 — Margin compression
        Net cost as a fraction of EBITDA margin — how much of current profit
        is eroded by this additional cost?

    Step 8 — EPS at Risk
        Multiply margin compression by 0.75 to account for tax and leverage effects.
        Cap at 95% — a company cannot lose more than 95% of EPS from this alone.

    Step 9 — EaR contribution
        Weight the holding's EPS at Risk by its portfolio weight.
        This is the holding's contribution to total portfolio EaR.

    Returns a dict with all intermediate values for full transparency in the UI.
    """
    margin = holding["ebitda_margin"]
    weight = holding["weight"]
    sector = holding["sector"]

    # Module 2 (ARIMA) — use ML-forecasted emissions intensity if available.
    # This replaces the static 2023 figure with a 3-year forward projection,
    # capturing decarbonisation trends already underway at each company.
    if ML_AVAILABLE and holding["ticker"] in EMISSIONS_FORECASTS:
        emis = EMISSIONS_FORECASTS[holding["ticker"]]
    else:
        emis = holding["emissions_intensity"]

    # Module 1 (OLS) — use ML-estimated pass-through rate if available.
    # The OLS regression estimates how much of carbon cost each sector
    # can pass on to customers, based on EU ETS price history vs margins.
    # pass_through_override takes priority (used in sensitivity analysis).
    if pass_through_override is not None:
        pt = pass_through_override
    elif ML_AVAILABLE and sector in PASSTHROUGH_RATES:
        pt = PASSTHROUGH_RATES[sector]
    else:
        pt = holding["pass_through"]

    carbon = scenario["carbon_price"]
    subsidy = scenario["subsidy_removal"]
    bca     = scenario["bca"]
    energy  = scenario["energy_shock"]

    # Step 1 — carbon cost impact
    # delta_carbon is the increase above the pre-ETS baseline of £25/t.
    # Dividing by 1000 converts from tCO2e per $1bn revenue to a decimal fraction.
    delta_carbon = carbon - BASELINE_CARBON
    carbon_cost_impact = (emis * delta_carbon) / 1000

    # Step 2 — subsidy removal impact
    # Only Energy and Materials sectors receive material fossil fuel subsidies.
    # The 0.06 and 0.02 factors represent the average subsidy as a fraction
    # of revenue for each sector, scaled by the scenario's removal rate.
    if sector == "Energy":
        subsidy_impact = subsidy * 0.06
    elif sector == "Materials":
        subsidy_impact = subsidy * 0.02
    else:
        subsidy_impact = 0.0

    # Step 3 — energy input shock
    # All companies face higher energy input costs as the carbon price rises.
    # The 0.3 factor reflects that energy costs are typically ~30% of the
    # direct carbon exposure for a given emissions intensity level.
    energy_input_impact = emis * energy * 0.3

    # Step 4 — border carbon adjustment
    # Only traded sectors (Energy, Materials, Industrials) face BCA risk.
    # The 0.4 factor represents the typical traded-goods share of revenue
    # that is exposed to border carbon tariffs.
    if sector in ("Energy", "Materials", "Industrials"):
        bca_impact = bca * emis * 0.4
    else:
        bca_impact = 0.0

    # Step 5 — total gross cost increase (sum of all four channels)
    total_cost_increase = carbon_cost_impact + subsidy_impact + energy_input_impact + bca_impact

    # Step 6 — net cost after pass-through
    # The company absorbs (1 - pass_through) of the total cost increase.
    # High pass-through = competitive market with pricing power (e.g. utilities).
    # Low pass-through = company absorbs cost = higher EaR (e.g. materials).
    net_cost_increase = total_cost_increase * (1 - pt)

    # Step 7 — margin compression
    # Express net cost as a fraction of EBITDA margin.
    # Example: net cost of 2% of revenue against a 10% EBITDA margin = 20% compression.
    if margin > 0:
        margin_compression = net_cost_increase / margin
    else:
        margin_compression = 0.0

    # Step 8 — EPS at Risk
    # Multiply by 0.75 to adjust for corporation tax (reduces EPS impact)
    # and leverage effects (amplify EPS impact for debt-heavy companies).
    # Net effect of these two factors is approximately 0.75 on average.
    # Cap at 0.95 (95%) — prevents unrealistic extreme values.
    eps_impact = min(margin_compression * 0.75, 0.95)

    # Step 9 — holding's contribution to portfolio EaR
    # Weight the EPS impact by this holding's share of the portfolio.
    ear = eps_impact * weight

    # Financials flag — banks report only their own operational emissions,
    # not the financed emissions of their loan books (Scope 3 Category 15).
    # This means their EaR is systematically understated. We flag it in the UI.
    return {
        "ticker":               holding["ticker"],
        "name":                 holding["name"],
        "sector":               sector,
        "weight":               weight,
        "emissions_intensity":  holding["emissions_intensity"],  # static 2023 figure
        "emis_used":            round(emis, 6),                  # actual value used (may be ARIMA forecast)
        "carbon_cost_impact":   round(carbon_cost_impact, 6),
        "subsidy_impact":       round(subsidy_impact, 6),
        "energy_input_impact":  round(energy_input_impact, 6),
        "bca_impact":           round(bca_impact, 6),
        "total_cost_increase":  round(total_cost_increase, 6),
        "net_cost_increase":    round(net_cost_increase, 6),
        "margin_compression":   round(margin_compression, 6),
        "eps_impact":           round(eps_impact, 4),
        "ear":                  round(ear, 6),
        "pass_through_used":    round(pt, 4),  # actual pass-through used (may be OLS estimate)
        "scope3_flag":          sector == "Financials",  # true = EaR is understated
    }


def compute_portfolio_ear(portfolio, scenario, pass_through_overrides=None):
    """
    Run compute_holding_ear for every holding and aggregate to portfolio level.

    pass_through_overrides: optional dict of {ticker: rate} — used when running
    sensitivity analysis with custom pass-through assumptions.

    Returns a dict with:
        holdings      — list of per-holding results (all intermediate values)
        portfolio_ear — sum of all holding EaR contributions
        scenario      — label of the scenario used
        carbon_price  — carbon price used
    """
    results = []
    for holding in portfolio:
        # Check if a per-ticker pass-through override was provided
        pt_override = None
        if pass_through_overrides and holding["ticker"] in pass_through_overrides:
            pt_override = pass_through_overrides[holding["ticker"]]

        result = compute_holding_ear(holding, scenario, pt_override)
        results.append(result)

    # Portfolio EaR is the sum of all weighted EaR contributions
    portfolio_ear = sum(r["ear"] for r in results)

    return {
        "holdings":      results,
        "portfolio_ear": round(portfolio_ear, 6),
        "scenario":      scenario["label"],
        "carbon_price":  scenario["carbon_price"],
    }


def optimise_portfolio(portfolio, scenario, turnover_limit=0.25, pass_through_overrides=None):
    """
    Compute EaR, then optimise portfolio weights to minimise it.

    Two optimisation paths:
      - If Module 4 (scipy SLSQP) is available: uses mean-variance optimisation
        minimising λ·EaR + (1−λ)·w'Σw subject to turnover and weight constraints.
        This is the primary path — see run_scipy_optimiser in ml_modules.py.

      - Fallback: simple inverse-EaR weighting with bisection-enforced turnover.
        Holdings with higher EaR get lower target weights. A 50-iteration bisection
        scales the rebalancing until actual turnover lands at or below the limit.

    turnover_limit: maximum allowed portfolio turnover as a decimal (0.25 = 25%).
    Turnover = sum of absolute weight changes. A 25% limit means at most 25% of
    the portfolio is repositioned.

    Returns a dict with original results, optimised results, trade recommendations,
    actual turnover achieved, and EaR reduction percentage.
    """
    # Compute EaR for the current (pre-optimisation) portfolio
    original = compute_portfolio_ear(portfolio, scenario, pass_through_overrides)
    w0  = np.array([h["weight"] for h in portfolio])
    eps = np.array([r["eps_impact"] for r in original["holdings"]])

    if ML_AVAILABLE:
        # Primary path — Module 4 scipy SLSQP mean-variance optimiser
        # Pass each holding's weight, EPS at Risk, and beta to the optimiser
        holdings_for_scipy = [
            {
                "weight":     h["weight"],
                "eps_impact": r["eps_impact"],
                "beta":       h.get("beta", 1.0),  # default beta=1 if not provided
                "ticker":     h["ticker"],
            }
            for h, r in zip(portfolio, original["holdings"])
        ]
        scipy_result   = run_scipy_optimiser(holdings_for_scipy, turnover_limit=turnover_limit)
        final_w        = scipy_result["final_weights"]
        actual_turnover = scipy_result["actual_turnover"] / 100  # convert from % to decimal
    else:
        # Fallback path — inverse-EaR bisection
        # Normalise EaR to [0,1] range, then set target weight = 1 - normalised_EaR
        # This gives lower weight to higher-risk holdings
        eps_min, eps_max = eps.min(), eps.max()
        if eps_max == eps_min:
            # All holdings have equal EaR — no rebalancing needed
            raw_t = np.ones(len(eps))
        else:
            raw_t = 1 - (eps - eps_min) / (eps_max - eps_min) + 0.05  # +0.05 floor
        target = raw_t / raw_t.sum()  # normalise to sum=1

        deltas = target - w0  # how much each weight needs to change

        def apply_scale(s):
            # Scale the deltas by factor s, floor weights at 0.5%, renormalise
            w_new = np.maximum(w0 + deltas * s, 0.005)
            w_new = w_new / w_new.sum()
            actual_turnover = np.abs(w_new - w0).sum()
            return w_new, actual_turnover

        # Bisection: find the largest scale factor s such that turnover <= limit
        lo, hi = 0.0, 1.0
        for _ in range(50):
            mid = (lo + hi) / 2
            _, turn = apply_scale(mid)
            if turn <= turnover_limit:
                lo = mid  # can afford more rebalancing
            else:
                hi = mid  # too much turnover, scale back

        final_w, actual_turnover = apply_scale(lo)

    # Build the optimised portfolio — same holdings, new weights
    opt_portfolio = []
    for i, h in enumerate(portfolio):
        opt_h = dict(h)
        opt_h["weight"] = round(float(final_w[i]), 6)
        opt_portfolio.append(opt_h)

    # Compute EaR for the optimised portfolio to measure improvement
    optimised = compute_portfolio_ear(opt_portfolio, scenario, pass_through_overrides)

    # Build trade-level recommendations — INCREASE / REDUCE / HOLD
    trades = []
    for i, h in enumerate(portfolio):
        delta = float(final_w[i]) - h["weight"]
        trades.append({
            "ticker":      h["ticker"],
            "name":        h["name"],
            "weight_from": round(h["weight"], 4),
            "weight_to":   round(float(final_w[i]), 4),
            "delta":       round(delta, 4),
            "direction":   "INCREASE" if delta > 0.001 else ("REDUCE" if delta < -0.001 else "HOLD"),
        })

    # EaR reduction — how much did the optimiser improve things?
    ear_reduction = original["portfolio_ear"] - optimised["portfolio_ear"]
    ear_reduction_pct = (ear_reduction / original["portfolio_ear"] * 100) if original["portfolio_ear"] > 0 else 0

    return {
        "original":          original,
        "optimised":         optimised,
        "trades":            trades,
        "actual_turnover":   round(float(actual_turnover), 4),
        "turnover_limit":    turnover_limit,
        "ear_reduction":     round(ear_reduction, 6),
        "ear_reduction_pct": round(ear_reduction_pct, 2),
    }


def parse_portfolio_csv(filepath):
    """
    Parse and validate a user-uploaded portfolio CSV file.

    Required columns: ticker, name, sector, weight, emissions_intensity,
                      ebitda_margin, pass_through
    Optional columns: beta (defaults to 1.0), source

    Validation checks:
      - All required columns present
      - Numeric fields are valid numbers
      - weight in (0, 1], ebitda_margin in (0, 1], pass_through in [0, 1]
      - sector is one of the seven recognised sectors
      - All weights sum to 1.0 ± 0.02

    Raises ValueError with a clear message if any check fails.
    Returns a list of holding dicts compatible with compute_portfolio_ear.
    """
    required = {"ticker", "name", "sector", "weight", "emissions_intensity", "ebitda_margin", "pass_through"}
    valid_sectors = {"Energy", "Materials", "Industrials", "Healthcare", "Financials", "Consumer", "Utilities"}

    holdings = []
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = set(reader.fieldnames or [])

        # Check all required columns are present before reading any rows
        missing = required - headers
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")

        for i, row in enumerate(reader, start=2):  # start=2 because row 1 is headers
            # Parse numeric fields — raise a clear error if conversion fails
            try:
                weight = float(row["weight"])
                emis   = float(row["emissions_intensity"])
                margin = float(row["ebitda_margin"])
                pt     = float(row["pass_through"])
            except ValueError as e:
                raise ValueError(f"Row {i}: numeric conversion failed — {e}")

            # Range validation
            if not (0 < weight <= 1):
                raise ValueError(f"Row {i}: weight must be between 0 and 1, got {weight}")
            if emis < 0:
                raise ValueError(f"Row {i}: emissions_intensity cannot be negative")
            if not (0 < margin <= 1):
                raise ValueError(f"Row {i}: ebitda_margin must be between 0 and 1")
            if not (0 <= pt <= 1):
                raise ValueError(f"Row {i}: pass_through must be between 0 and 1")
            if row["sector"] not in valid_sectors:
                raise ValueError(f"Row {i}: sector must be one of {valid_sectors}")

            holdings.append({
                "ticker":               row["ticker"].strip().upper(),
                "name":                 row["name"].strip(),
                "sector":               row["sector"].strip(),
                "weight":               weight,
                "emissions_intensity":  emis,
                "ebitda_margin":        margin,
                "pass_through":         pt,
                "beta":                 float(row.get("beta", 1.0) or 1.0),  # default to market beta
                "source":               row.get("source", "User uploaded").strip(),
            })

    # Final check: weights must sum to approximately 1.0
    # Allow ±2% tolerance for rounding in the CSV
    total_weight = sum(h["weight"] for h in holdings)
    if not (0.98 <= total_weight <= 1.02):
        raise ValueError(f"Portfolio weights sum to {total_weight:.4f} — must sum to 1.0 (±0.02)")

    return holdings


# ── QUICK TEST ────────────────────────────────────────────────────────────────
# Run this file directly to verify the engine produces sensible outputs.
# Example: python ear_engine.py
if __name__ == "__main__":
    scenario = SCENARIOS["netzero"]
    result = optimise_portfolio(PORTFOLIO, scenario, turnover_limit=0.25)

    print(f"\nScenario: {result['original']['scenario']} at £{result['original']['carbon_price']}/t")
    print(f"Portfolio EaR (original):  {result['original']['portfolio_ear']*100:.2f}%")
    print(f"Portfolio EaR (optimised): {result['optimised']['portfolio_ear']*100:.2f}%")
    print(f"EaR reduction:             {result['ear_reduction_pct']:.1f}%")
    print(f"Actual turnover:           {result['actual_turnover']*100:.1f}%")
    print(f"\nTrade recommendations:")
    for t in sorted(result["trades"], key=lambda x: abs(x["delta"]), reverse=True):
        if t["direction"] != "HOLD":
            print(f"  {t['ticker']:6} {t['direction']:8} {t['weight_from']*100:.1f}% -> {t['weight_to']*100:.1f}% ({t['delta']*100:+.1f}pp)")
