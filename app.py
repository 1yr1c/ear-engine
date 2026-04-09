"""
app.py — Flask Web Application
Climate Transition EaR Engine
Stirling Solvers — CFA AI Investment Challenge 2026

This is the web server layer. It has five responsibilities:
  1. Serve the frontend (index.html) with scenario data injected server-side
  2. Run climate risk analysis via the /analyse endpoint
  3. Accept user-uploaded portfolio CSVs via /upload
  4. Power the AI chatbot and memo generation via /chat and /analyse
  5. Expose ML pipeline status via /ml/status so the frontend can poll it

All heavy computation happens in ear_engine.py and ml_modules.py.
This file just handles HTTP routing and API calls.
"""

import os
import json
import tempfile
from flask import Flask, render_template, request, jsonify
from ear_engine import (
    PORTFOLIO, SCENARIOS,
    compute_portfolio_ear, optimise_portfolio, parse_portfolio_csv
)

# ── ANTHROPIC CLIENT SETUP ────────────────────────────────────────────────────
# Import the Anthropic SDK at module level — NOT inside the route handler.
# Lazy imports inside handlers risk a thread import lock conflict with the
# ml_modules background thread, causing gunicorn worker timeouts.
try:
    import anthropic as _anthropic
    ANTHROPIC_AVAILABLE = True
except Exception:
    ANTHROPIC_AVAILABLE = False  # app works fine without it — memo shows fallback message

# ── DEFAULT SCENARIO SETUP ───────────────────────────────────────────────────
# Try to read the HMM-detected scenario at startup. If ml_modules hasn't
# finished yet (it runs in a background thread), this will catch the fallback.
# The /ml/status polling in the frontend handles the live update separately.
try:
    from ml_modules import DETECTED_SCENARIO
    _detected_id = DETECTED_SCENARIO.get("id", "netzero")
    # Validate that the detected scenario ID exists in SCENARIOS dict
    _DEFAULT_SCENARIO_ID = _detected_id if _detected_id in SCENARIOS else "netzero"
except Exception:
    _DEFAULT_SCENARIO_ID = "netzero"  # safe default if ml_modules import fails

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024  # 1 MB max upload size

# ── MEMO CACHE ────────────────────────────────────────────────────────────────
# Cache generated memos in memory to avoid redundant API calls.
# Key format: "{scenario_id}_{turnover_limit}" — unique per analysis configuration.
# Cache is in-process only — resets on each redeploy.
memo_cache = {}


def generate_memo(result):
    """
    Call Claude Sonnet to generate a plain English investment memo for the analysis.

    The memo is generated from the deterministic model outputs — Claude interprets
    the numbers but does not produce any numbers itself. This distinction is important
    for regulatory compliance and is noted in the UI disclaimer.

    Caches the result so re-running the same scenario+turnover doesn't re-call the API.
    Falls back gracefully with a clear error message if the API key is missing or the
    call times out.

    Args:
        result: the full output dict from optimise_portfolio()

    Returns: memo text string (may be an error message if API unavailable)
    """
    cache_key = f"{result['original']['scenario']}_{result['turnover_limit']}"
    if cache_key in memo_cache:
        return memo_cache[cache_key]  # return cached version, no API call needed

    try:
        if not ANTHROPIC_AVAILABLE:
            return "[AI memo unavailable: anthropic package not installed.]"

        client = _anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
            timeout=60.0,  # 60s timeout — well within gunicorn's 120s worker timeout
        )

        # Build the prompt — give Claude the key numbers and instruct it to interpret,
        # not reproduce. The top 5 holdings by EaR and key trades provide enough context.
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
        memo_cache[cache_key] = memo  # cache for future requests
        return memo

    except Exception as e:
        return f"[AI memo unavailable: {str(e)}. Please set ANTHROPIC_API_KEY environment variable.]"


# ══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    """
    Serve the main application page.
    The SCENARIOS dict is passed to the Jinja template so scenario buttons
    are rendered server-side — no extra API call needed on page load.
    """
    return render_template("index.html", scenarios=SCENARIOS)


@app.route("/analyse", methods=["POST"])
def analyse():
    """
    Main analysis endpoint — runs the full EaR engine and optimiser.

    Accepts JSON body:
        scenario_id     (str)   — which NGFS scenario to use
        turnover_limit  (float) — max rebalancing as decimal, e.g. 0.25
        portfolio       (list)  — optional custom portfolio (from /upload)
        custom_carbon   (int)   — optional carbon price override in £/t

    Returns JSON with:
        original        — pre-optimisation EaR results per holding
        optimised       — post-optimisation EaR results per holding
        trades          — trade recommendations (INCREASE/REDUCE/HOLD)
        ear_reduction_pct — how much the optimiser reduced EaR (%)
        actual_turnover — actual turnover applied (decimal)
        memo            — AI-generated investment memo text
    """
    data             = request.get_json()
    scenario_id      = data.get("scenario_id", _DEFAULT_SCENARIO_ID)
    turnover_limit   = float(data.get("turnover_limit", 0.25))
    custom_portfolio = data.get("portfolio", None)
    custom_carbon    = data.get("custom_carbon", None)  # optional override in £/t

    # Silently fall back to netzero if the frontend sends an unrecognised scenario ID
    if scenario_id not in SCENARIOS:
        scenario_id = "netzero"

    if not (0.05 <= turnover_limit <= 1.0):
        return jsonify({"error": "Turnover limit must be between 5% and 100%"}), 400

    # Use custom portfolio if uploaded, otherwise use the default 15-holding portfolio
    portfolio = custom_portfolio if custom_portfolio else PORTFOLIO

    # Copy scenario dict before mutating — we don't want to change the global SCENARIOS
    scenario = dict(SCENARIOS[scenario_id])

    # Apply carbon price override if provided (from the "Custom Carbon Price" input)
    # This lets users test a specific carbon price without switching scenarios
    if custom_carbon and isinstance(custom_carbon, (int, float)) and 1 <= custom_carbon <= 1000:
        scenario["carbon_price"] = int(custom_carbon)
        scenario["label"] = f"{scenario['label']} (custom £{int(custom_carbon)}/t)"

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
    Accept and validate a user-uploaded portfolio CSV file.

    The CSV is saved to a temporary file, parsed by parse_portfolio_csv(),
    and the result returned as JSON. The temp file is deleted immediately after.

    Returns JSON with:
        portfolio  — list of holding dicts (same format as PORTFOLIO in ear_engine.py)
        count      — number of holdings parsed

    Returns 400 with an error message if:
        - No file attached
        - File is not a .csv
        - Required columns missing
        - Numeric values out of range
        - Weights don't sum to 1.0 ±0.02
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if not file.filename.endswith(".csv"):
        return jsonify({"error": "File must be a .csv"}), 400

    try:
        # Write to a temp file — parse_portfolio_csv expects a filepath, not a file object
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            file.save(tmp.name)
            portfolio = parse_portfolio_csv(tmp.name)
        os.unlink(tmp.name)  # clean up immediately after parsing
        return jsonify({"portfolio": portfolio, "count": len(portfolio)})
    except ValueError as e:
        # Validation errors from parse_portfolio_csv — return as user-readable message
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500


@app.route("/chat", methods=["POST"])
def chat():
    """
    Portfolio analyst chatbot endpoint.

    Accepts a user message, conversation history, and the full analysis context.
    Returns a response from Claude Sonnet with full awareness of the current portfolio,
    EaR figures, optimiser trades, and the generated memo.

    Accepts JSON body:
        message  (str)   — the user's question
        history  (list)  — previous messages [{role, content}, ...]
        context  (dict)  — current analysis data (scenario, holdings, trades, memo)

    The system prompt gives Claude the full analysis context so it can answer
    specific questions like "why is Shell so high?" or "what happens at £300/t?"
    with accurate, portfolio-specific responses.

    Conversation history is capped at 20 turns on the frontend to prevent
    the context window from growing unbounded.
    """
    if not ANTHROPIC_AVAILABLE:
        return jsonify({"reply": "[Chat unavailable: anthropic package not installed.]"})

    data    = request.get_json()
    message = data.get("message", "")
    history = data.get("history", [])  # list of {role, content} dicts
    ctx     = data.get("context", {})  # current analysis data from the frontend

    if not message:
        return jsonify({"reply": ""}), 400

    # Build a detailed system prompt from the current analysis context.
    # This gives Claude full visibility of the portfolio without the user
    # needing to re-explain what they're looking at.
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

        # Prepend conversation history so Claude has full conversational context
        messages = history + [{"role": "user", "content": message}]

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=400,  # keep responses concise
            system=system,
            messages=messages,
        )
        reply = response.content[0].text
        return jsonify({"reply": reply})

    except Exception as e:
        return jsonify({"reply": f"[Chat error: {str(e)}]"})


@app.route("/scenarios")
def get_scenarios():
    """Return all NGFS scenarios as JSON — used by frontend SCENARIOS constant."""
    return jsonify(SCENARIOS)


@app.route("/portfolio/default")
def get_default_portfolio():
    """Return the default 15-holding FTSE 100 portfolio — used by frontend reset."""
    return jsonify(PORTFOLIO)


@app.route("/ml/status")
def ml_status():
    """
    Return the current state of the ML pipeline.

    The frontend polls this endpoint every 3 seconds until ml_ready is True.
    Once ready, the HMM-recommended scenario button appears and the module dots
    turn green.

    Returns JSON with:
        ml_active          — True if ml_modules imported successfully
        ml_ready           — True once all 4 modules have completed
        detected_scenario  — NGFS scenario dict from HMM (or fallback)
        passthrough_rates  — {sector: rate} from OLS (or fallback)
        emissions_forecasts — {ticker: intensity} from ARIMA (or empty)
    """
    try:
        import ml_modules  # import the module object so we read live global values
        return jsonify({
            "detected_scenario":    ml_modules.DETECTED_SCENARIO,
            "passthrough_rates":    ml_modules.PASSTHROUGH_RATES,
            "emissions_forecasts":  ml_modules.EMISSIONS_FORECASTS,
            "ml_active":            True,
            "ml_ready":             getattr(ml_modules, "ML_READY", False),
        })
    except Exception as e:
        return jsonify({"ml_active": False, "ml_ready": False, "error": str(e)})


if __name__ == "__main__":
    # Run in debug mode locally — gunicorn is used in production (see gunicorn.conf.py)
    app.run(debug=True, port=5000)
