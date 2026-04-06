# ============================================================
# Climate Transition EaR Engine — Core Python Module
# Stirling Solvers — CFA AI Investment Challenge 2026
# ============================================================

# DEFAULT PORTFOLIO — 15 FTSE 100 holdings
# Weights sum to 1.0
# Sources cited per holding

PORTFOLIO = [
    # ticker, name, sector, weight, emissions_intensity, ebitda_margin, pass_through, beta, source
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

# NGFS Phase V (Nov 2024) aligned scenarios
# Source: NGFS Climate Scenarios Phase V, November 2024
# Carbon price baseline: UK ETS official rate £41.84/t for 2025 scheme year
# Source: UK ETS Authority (2025) UK ETS Carbon Prices for Civil Penalties

SCENARIOS = {
    "base": {
        "label": "Current Policies",
        "carbon_price": 42,
        "subsidy_removal": 0.0,
        "bca": 0.0,
        "energy_shock": 0.0,
        "temp": "2.9C",
        "description": "NGFS Current Policies. UK ETS official rate £41.84/t (2025). No additional transition shock."
    },
    "delayed": {
        "label": "Delayed Transition",
        "carbon_price": 80,
        "subsidy_removal": 0.20,
        "bca": 0.06,
        "energy_shock": 0.10,
        "temp": "1.8C",
        "description": "NGFS Delayed Transition. Emissions flat until 2030 then sharp acceleration. Carbon ~£80/t."
    },
    "below2": {
        "label": "Orderly Below 2C",
        "carbon_price": 120,
        "subsidy_removal": 0.35,
        "bca": 0.12,
        "energy_shock": 0.18,
        "temp": "1.7C",
        "description": "NGFS Below 2C Orderly. Early coordinated policy action. Carbon ~£120/t."
    },
    "netzero": {
        "label": "Orderly Net Zero 2050",
        "carbon_price": 200,
        "subsidy_removal": 0.55,
        "bca": 0.20,
        "energy_shock": 0.28,
        "temp": "1.5C",
        "description": "NGFS Net Zero 2050. Full subsidy reform, EU CBAM operational. Carbon ~£200/t."
    },
    "fragmented": {
        "label": "Disorderly Fragmented",
        "carbon_price": 300,
        "subsidy_removal": 0.80,
        "bca": 0.40,
        "energy_shock": 0.40,
        "temp": "2.0C",
        "description": "NGFS Disorderly Fragmented. Divergent uncoordinated policy. Carbon ~£300/t."
    },
}

BASELINE_CARBON = 25  # pre-ETS baseline for delta calculation

try:
    from ml_modules import PASSTHROUGH_RATES, EMISSIONS_FORECASTS, run_scipy_optimiser
    ML_AVAILABLE = True
except Exception:
    ML_AVAILABLE = False
    PASSTHROUGH_RATES   = {}
    EMISSIONS_FORECASTS = {}


def compute_holding_ear(holding, scenario, pass_through_override=None):
    """
    Compute Earnings-at-Risk for a single holding under a given scenario.

    Formula chain:
    1. carbon_cost_impact = emissions_intensity * delta_carbon / 1000
    2. subsidy_impact = subsidy_removal * sector_factor
    3. energy_input_impact = emissions_intensity * energy_shock * 0.3
    4. bca_impact = bca * emissions_intensity * 0.4 (Energy/Materials/Industrials only)
    5. total_cost_increase = sum of above
    6. net_cost_increase = total_cost_increase * (1 - pass_through)
    7. margin_compression = net_cost_increase / ebitda_margin
    8. eps_impact = min(margin_compression * 0.75, 0.95)  -- tax/leverage adj, capped at 95%
    9. ear = eps_impact * weight

    Returns dict with all intermediate values for transparency.
    """
    margin = holding["ebitda_margin"]
    weight = holding["weight"]
    sector = holding["sector"]

    # Module 2 ARIMA — use forecasted emissions intensity if available
    if ML_AVAILABLE and holding["ticker"] in EMISSIONS_FORECASTS:
        emis = EMISSIONS_FORECASTS[holding["ticker"]]
    else:
        emis = holding["emissions_intensity"]

    # Module 1 OLS — use ML pass-through rate if available
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
    delta_carbon = carbon - BASELINE_CARBON
    carbon_cost_impact = (emis * delta_carbon) / 1000

    # Step 2 — subsidy removal impact (sector-specific)
    if sector == "Energy":
        subsidy_impact = subsidy * 0.06
    elif sector == "Materials":
        subsidy_impact = subsidy * 0.02
    else:
        subsidy_impact = 0.0

    # Step 3 — energy input shock
    energy_input_impact = emis * energy * 0.3

    # Step 4 — border carbon adjustment (traded sectors only)
    if sector in ("Energy", "Materials", "Industrials"):
        bca_impact = bca * emis * 0.4
    else:
        bca_impact = 0.0

    # Step 5 — total gross cost increase
    total_cost_increase = carbon_cost_impact + subsidy_impact + energy_input_impact + bca_impact

    # Step 6 — net cost after pass-through
    net_cost_increase = total_cost_increase * (1 - pt)

    # Step 7 — margin compression
    if margin > 0:
        margin_compression = net_cost_increase / margin
    else:
        margin_compression = 0.0

    # Step 8 — EPS impact (tax/leverage adjusted, capped)
    eps_impact = min(margin_compression * 0.75, 0.95)

    # Step 9 — weighted EaR contribution
    ear = eps_impact * weight

    return {
        "ticker":               holding["ticker"],
        "name":                 holding["name"],
        "sector":               sector,
        "weight":               weight,
        "carbon_cost_impact":   round(carbon_cost_impact, 6),
        "subsidy_impact":       round(subsidy_impact, 6),
        "energy_input_impact":  round(energy_input_impact, 6),
        "bca_impact":           round(bca_impact, 6),
        "total_cost_increase":  round(total_cost_increase, 6),
        "net_cost_increase":    round(net_cost_increase, 6),
        "margin_compression":   round(margin_compression, 6),
        "eps_impact":           round(eps_impact, 4),
        "ear":                  round(ear, 6),
        "pass_through_used":    round(pt, 4),
        "scope3_flag":          sector == "Financials",
    }


def compute_portfolio_ear(portfolio, scenario, pass_through_overrides=None):
    """
    Compute EaR for all holdings and return portfolio-level summary.
    pass_through_overrides: dict of {ticker: rate} from regression model (optional)
    """
    results = []
    for holding in portfolio:
        pt_override = None
        if pass_through_overrides and holding["ticker"] in pass_through_overrides:
            pt_override = pass_through_overrides[holding["ticker"]]
        result = compute_holding_ear(holding, scenario, pt_override)
        results.append(result)

    portfolio_ear = sum(r["ear"] for r in results)

    return {
        "holdings":      results,
        "portfolio_ear": round(portfolio_ear, 6),
        "scenario":      scenario["label"],
        "carbon_price":  scenario["carbon_price"],
    }


def optimise_portfolio(portfolio, scenario, turnover_limit=0.25, pass_through_overrides=None):
    """
    Bisection-enforced EaR-ranked portfolio optimiser.

    Computes target weights inversely proportional to each holding's EaR.
    Scales the rebalancing using bisection to guarantee actual turnover
    lands at or below turnover_limit. Floor at 0.005, renormalised to sum=1.

    Returns original results, optimised results, and trade recommendations.
    """
    import numpy as np

    # Step 1 — compute original EaR
    original = compute_portfolio_ear(portfolio, scenario, pass_through_overrides)
    w0 = np.array([h["weight"] for h in portfolio])
    eps = np.array([r["eps_impact"] for r in original["holdings"]])

    # Step 2 — compute target weights
    if ML_AVAILABLE:
        # Module 4 — scipy SLSQP mean-variance optimiser
        holdings_for_scipy = [
            {
                "weight":     h["weight"],
                "eps_impact": r["eps_impact"],
                "beta":       h.get("beta", 1.0),
                "ticker":     h["ticker"],
            }
            for h, r in zip(portfolio, original["holdings"])
        ]
        scipy_result = run_scipy_optimiser(holdings_for_scipy, turnover_limit=turnover_limit)
        final_w = scipy_result["final_weights"]
        actual_turnover = scipy_result["actual_turnover"] / 100
    else:
        # Fallback — inverse-EaR bisection
        eps_min, eps_max = eps.min(), eps.max()
        if eps_max == eps_min:
            raw_t = np.ones(len(eps))
        else:
            raw_t = 1 - (eps - eps_min) / (eps_max - eps_min) + 0.05
        target = raw_t / raw_t.sum()

        deltas = target - w0

        def apply_scale(s):
            w_new = np.maximum(w0 + deltas * s, 0.005)
            w_new = w_new / w_new.sum()
            actual_turnover = np.abs(w_new - w0).sum()
            return w_new, actual_turnover

        lo, hi = 0.0, 1.0
        for _ in range(50):
            mid = (lo + hi) / 2
            _, turn = apply_scale(mid)
            if turn <= turnover_limit:
                lo = mid
            else:
                hi = mid

        final_w, actual_turnover = apply_scale(lo)

    # Step 4 — build optimised portfolio with new weights
    opt_portfolio = []
    for i, h in enumerate(portfolio):
        opt_h = dict(h)
        opt_h["weight"] = round(float(final_w[i]), 6)
        opt_portfolio.append(opt_h)

    optimised = compute_portfolio_ear(opt_portfolio, scenario, pass_through_overrides)

    # Step 5 — trade recommendations
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
    Parse a user-uploaded portfolio CSV.

    Required columns: ticker, name, sector, weight, emissions_intensity, ebitda_margin, pass_through
    Optional columns: beta, source

    Returns list of holding dicts compatible with compute_portfolio_ear.
    Raises ValueError with a clear message if validation fails.
    """
    import csv

    required = {"ticker", "name", "sector", "weight", "emissions_intensity", "ebitda_margin", "pass_through"}
    valid_sectors = {"Energy", "Materials", "Industrials", "Healthcare", "Financials", "Consumer", "Utilities"}

    holdings = []
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = set(reader.fieldnames or [])

        missing = required - headers
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")

        for i, row in enumerate(reader, start=2):
            try:
                weight = float(row["weight"])
                emis   = float(row["emissions_intensity"])
                margin = float(row["ebitda_margin"])
                pt     = float(row["pass_through"])
            except ValueError as e:
                raise ValueError(f"Row {i}: numeric conversion failed — {e}")

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
                "beta":                 float(row.get("beta", 1.0) or 1.0),
                "source":               row.get("source", "User uploaded").strip(),
            })

    # Validate weights sum to approximately 1
    total_weight = sum(h["weight"] for h in holdings)
    if not (0.98 <= total_weight <= 1.02):
        raise ValueError(f"Portfolio weights sum to {total_weight:.4f} — must sum to 1.0 (±0.02)")

    return holdings


# ============================================================
# QUICK TEST — run directly to verify outputs
# ============================================================
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
