# Climate Transition EaR Engine
**Stirling Solvers — CFA Institute AI Investment Challenge 2026**

A three-layer quantitative decision-support engine that converts regulatory policy assumptions into portfolio earnings impact, then solves for the optimal rebalancing response.

---

## Quick Start

```bash
git clone https://github.com/stirling-solvers/ear-engine
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
scipy mean-variance optimiser treating EaR as a risk factor alongside return and volatility. Bisection algorithm enforces user-defined turnover constraint on final measured weights.

---

## ML Features (Stage 2)

| Feature | Method | Status |
|---------|--------|--------|
| Pass-through regression | OLS on EU ETS price history vs gross margins | Planned |
| Emissions forecasting | ARIMA per holding | Planned |
| Carbon regime detection | Gaussian HMM on EU ETS history | Planned |
| Portfolio optimiser | scipy mean-variance | Implemented |

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

See `portfolio_template.csv` for an example.

---

## Data Sources

- Emissions intensity: 2023 company sustainability reports and CDP disclosures
- EBITDA margins: Company annual reports 2023
- NGFS Phase V scenarios: NGFS (November 2024)
- Pass-through rates: Fabra, N. and Reguant, M. (2014). Pass-Through of Emissions Costs in Electricity Markets. *American Economic Review*, 104(9), 2872–2899.
- Carbon price baseline: UK ETS Authority (2025). UK ETS Carbon Prices for Civil Penalties 2025.
- Validation benchmark: Bank of England (2022). Results of the 2021 Climate Biennial Exploratory Scenario (CBES).

---

## Validation

Model outputs for Energy and Materials holdings are cross-validated against the Bank of England CBES 2021 results under equivalent carbon price assumptions.

---

## Team

| Name | Role |
|------|------|
| Oliver | Architecture & AI Integration |
| Tracey | Lead Developer (ML features) |
| Agnes | Emissions, Scenarios & Validation |
| Michael | Narrative & Presentation |

University of Stirling

---

## Licence

MIT Licence — see LICENSE file.

---

## Disclaimer

This tool is for research and educational purposes only. All AI-generated content is clearly labelled. Nothing in this tool constitutes investment advice.
