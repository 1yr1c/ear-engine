"""
Climate Transition EaR Engine — Flask Application
Stirling Solvers — CFA AI Investment Challenge 2026
"""

import os
import json
import tempfile
from flask import Flask, render_template, request, jsonify
from ear_engine import (
    PORTFOLIO, SCENARIOS,
    compute_portfolio_ear, optimise_portfolio, parse_portfolio_csv
)

try:
    import anthropic as _anthropic
    ANTHROPIC_AVAILABLE = True
except Exception:
    ANTHROPIC_AVAILABLE = False

try:
    from ml_modules import DETECTED_SCENARIO
    _detected_id = DETECTED_SCENARIO.get("id", "netzero")
    # Validate it's a real scenario — fallback to netzero if not
    _DEFAULT_SCENARIO_ID = _detected_id if _detected_id in SCENARIOS else "netzero"
except Exception:
    _DEFAULT_SCENARIO_ID = "netzero"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024  # 1MB upload limit

# ============================================================
# MEMO CACHE — keyed by scenario_id + turnover_limit
# ============================================================
memo_cache = {}


def generate_memo(result):
    """
    Call Claude Sonnet via Anthropic API to generate a plain English investment memo.
    Returns cached memo if available. Falls back gracefully if API unavailable.
    """
    cache_key = f"{result['original']['scenario']}_{result['turnover_limit']}"
    if cache_key in memo_cache:
        return memo_cache[cache_key]

    try:
        if not ANTHROPIC_AVAILABLE:
            return "[AI memo unavailable: anthropic package not installed.]"
        client = _anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
            timeout=60.0,
        )
        prompt = f"""You are a senior investment analyst writing a concise portfolio risk memo.

Based on the following climate transition risk analysis, write a professional 150-200 word investment memo.
Write in plain English. Do not reproduce the numbers verbatim — interpret what they mean for the fund manager.
Do not make specific buy or sell recommendations. End with one sentence on key risks to monitor.

Scenario: {result['original']['scenario']} (carbon price £{result['original']['carbon_price']}/t)
Portfolio EaR before rebalancing: {result['original']['portfolio_ear']*100:.1f}%
Portfolio EaR after rebalancing: {result['optimised']['portfolio_ear']*100:.1f}%
EaR reduction achieved: {result['ear_reduction_pct']:.1f}%
Actual turnover applied: {result['actual_turnover']*100:.1f}%

Top holdings by EaR before rebalancing:
{chr(10).join([f"- {h['ticker']} ({h['sector']}): {h['eps_impact']*100:.1f}% EPS at Risk" for h in sorted(result['original']['holdings'], key=lambda x: x['eps_impact'], reverse=True)[:5]])}

Key rebalancing actions:
{chr(10).join([f"- {t['ticker']}: {t['direction']} from {t['weight_from']*100:.1f}% to {t['weight_to']*100:.1f}%" for t in result['trades'] if t['direction'] != 'HOLD'][:5])}

Note: Any financial sector holdings may understate risk due to Scope 3 financed emissions exclusion.

Write the memo now. Label it clearly as AI-assisted analysis. Not investment advice."""

        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}]
        )
        memo = message.content[0].text
        memo_cache[cache_key] = memo
        return memo

    except Exception as e:
        return f"[AI memo unavailable: {str(e)}. Please set ANTHROPIC_API_KEY environment variable.]"


# ============================================================
# ROUTES
# ============================================================

@app.route("/")
def index():
    return render_template("index.html", scenarios=SCENARIOS)


@app.route("/analyse", methods=["POST"])
def analyse():
    """
    Main analysis endpoint.
    Accepts JSON: { scenario_id, turnover_limit, portfolio (optional) }
    Returns full EaR analysis + optimised portfolio + trade recommendations.
    """
    data = request.get_json()
    scenario_id    = data.get("scenario_id", _DEFAULT_SCENARIO_ID)
    turnover_limit = float(data.get("turnover_limit", 0.25))
    custom_portfolio = data.get("portfolio", None)

    # Silently fall back to netzero if scenario_id is invalid
    if scenario_id not in SCENARIOS:
        scenario_id = "netzero"
    if not (0.05 <= turnover_limit <= 1.0):
        return jsonify({"error": "Turnover limit must be between 5% and 100%"}), 400

    portfolio = custom_portfolio if custom_portfolio else PORTFOLIO
    scenario  = SCENARIOS[scenario_id]

    try:
        result = optimise_portfolio(portfolio, scenario, turnover_limit=turnover_limit)
        memo   = generate_memo(result)
        result["memo"] = memo
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/upload", methods=["POST"])
def upload():
    """
    Accept a user-uploaded portfolio CSV.
    Returns parsed portfolio JSON or validation errors.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if not file.filename.endswith(".csv"):
        return jsonify({"error": "File must be a .csv"}), 400

    try:
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            file.save(tmp.name)
            portfolio = parse_portfolio_csv(tmp.name)
        os.unlink(tmp.name)
        return jsonify({"portfolio": portfolio, "count": len(portfolio)})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500


@app.route("/chat", methods=["POST"])
def chat():
    """
    Portfolio analyst chatbot endpoint.
    Accepts: { message, history, context }
    Returns: { reply }
    """
    if not ANTHROPIC_AVAILABLE:
        return jsonify({"reply": "[Chat unavailable: anthropic package not installed.]"})

    data = request.get_json()
    message = data.get("message", "")
    history = data.get("history", [])
    ctx     = data.get("context", {})

    if not message:
        return jsonify({"reply": ""}), 400

    system = f"""You are a senior climate finance analyst assistant embedded in a portfolio risk tool.
You have full context of the current analysis and can answer questions about it clearly and concisely.

CURRENT ANALYSIS CONTEXT:
Scenario: {ctx.get('scenario')} (carbon price £{ctx.get('carbon_price')}/t)
Portfolio EaR (before): {ctx.get('portfolio_ear')}%
Portfolio EaR (after optimisation): {ctx.get('optimised_ear')}%
EaR Reduction: {ctx.get('ear_reduction')}%

Holdings (sorted by EPS at Risk):
{chr(10).join([f"- {h['ticker']} ({h['sector']}): {h['eps_impact']}% EPS at Risk, {h['weight']}% weight, emissions intensity {h['emissions_intensity']}, pass-through {h['pass_through']}%" for h in sorted(ctx.get('holdings',[]), key=lambda x: float(x.get('eps_impact',0)), reverse=True)])}

Rebalancing trades:
{chr(10).join([f"- {t['ticker']}: {t['direction']} {t['from']}% → {t['to']}%" for t in ctx.get('trades',[])])}

Investment memo:
{ctx.get('memo','')}

INSTRUCTIONS:
- Answer questions about this specific portfolio and analysis
- Be concise — 2-4 sentences unless a detailed explanation is needed
- You can explain methodology, interpret numbers, discuss scenarios, or explore hypotheticals
- Always clarify this is not investment advice
- Do not make specific buy/sell recommendations"""

    try:
        client = _anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
            timeout=60.0,
        )

        messages = history + [{"role": "user", "content": message}]

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=400,
            system=system,
            messages=messages,
        )
        reply = response.content[0].text
        return jsonify({"reply": reply})

    except Exception as e:
        return jsonify({"reply": f"[Chat error: {str(e)}]"})


@app.route("/scenarios")
def get_scenarios():
    return jsonify(SCENARIOS)


@app.route("/portfolio/default")
def get_default_portfolio():
    return jsonify(PORTFOLIO)


@app.route("/ml/status")
def ml_status():
    try:
        import ml_modules
        return jsonify({
            "detected_scenario":   ml_modules.DETECTED_SCENARIO,
            "passthrough_rates":   ml_modules.PASSTHROUGH_RATES,
            "emissions_forecasts": ml_modules.EMISSIONS_FORECASTS,
            "ml_active":           True,
            "ml_ready":            getattr(ml_modules, "ML_READY", False),
        })
    except Exception as e:
        return jsonify({"ml_active": False, "ml_ready": False, "error": str(e)})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
