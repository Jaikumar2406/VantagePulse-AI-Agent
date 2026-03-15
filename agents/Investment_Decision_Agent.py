"""
Investment Decision Agent
--------------------------
The final agent in the pipeline. Synthesizes all prior agent outputs into a
unified VC-style investment decision report — no web search needed since all
intelligence has already been gathered upstream.

LLM : Groq  (llama-3.3-70b-versatile via LangChain)

Install dependencies:
    pip install langchain-groq langchain-core python-dotenv

Environment variables required (.env):
    agent2_llm = <your Groq API key>
"""

import os
import re
import textwrap
from dotenv import load_dotenv
from state.state import AgentState
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()


# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════

GROQ_MODEL  = "llama-3.3-70b-versatile"
MAX_TOKENS  = 4096
TEMPERATURE = 0


# ═════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT
# ═════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = textwrap.dedent("""\
    You are a Managing Partner at a top-tier venture capital firm. You are
    writing the final investment committee memo for a startup that has been
    put through a comprehensive multi-agent due diligence pipeline.

    You will receive a CONSOLIDATED INTELLIGENCE CONTEXT containing structured
    outputs from up to seven specialist agents:
      1. Startup Data Agent       — product, business model, founder overview
      2. Market Research Agent    — TAM/SAM/SOM, CAGR, opportunity flag
      3. Competitor Analysis Agent — competitive landscape, intensity score
      4. Financial Estimation Agent — revenue, burn, runway, unit economics
      5. Risk Analysis Agent      — risk scores per category, red flags
      6. Growth Prediction Agent  — success probability, valuation projection
      7. Founder Intelligence Agent (if available)

    Your task is to synthesize ALL of this intelligence into a single,
    authoritative INVESTMENT DECISION REPORT.

    ─────────────────────────────────────────────────────────────
    CRITICAL RULES — READ BEFORE WRITING:
    ─────────────────────────────────────────────────────────────
    - Do NOT fabricate data absent from the provided context.
    - If a metric is unavailable, write: "Data not available."
    - Use conservative, evidence-based reasoning throughout.
    - Cite which agent's analysis supports each claim.
    - Produce a professional investment committee memo tone.

    ─────────────────────────────────────────────────────────────
    REPORT STRUCTURE (use these exact section headers):
    ─────────────────────────────────────────────────────────────

    ## 1. Startup Overview
    In 4–6 sentences, summarise:
      - What the startup builds and for whom
      - The core value proposition and revenue model
      - Funding stage and key traction signal (if any)
      - The market it operates in and its size/growth rate

    ## 2. Key Strengths
    List 4–7 major positive signals as a numbered list. For each strength:
      - State the signal clearly (one line)
      - Cite the supporting agent (e.g., "per Market Research Agent")
      - Add one sentence of supporting evidence

    ## 3. Key Risks
    List 4–7 critical risks as a numbered list. For each risk:
      - State the risk clearly (one line)
      - Cite the supporting agent (e.g., "per Risk Analysis Agent")
      - Add one sentence of context or severity assessment

    ## 4. Investment Score
    Calculate a composite score from 1–10 using the six sub-dimensions below.
    For each sub-dimension assign a score (1–10) and one sentence of reasoning:

      - Market Opportunity      : X/10
      - Competitive Positioning : X/10
      - Financial Health        : X/10
      - Risk Profile            : X/10  (invert risk score: 10 - risk_score)
      - Growth Potential        : X/10
      - Founder Capability      : X/10

    Then compute:
        Overall Investment Score = weighted average
        Weights: Market 20%, Competition 15%, Financial 15%, Risk 20%, Growth 20%, Founder 10%

    Format as:
        Overall Investment Score: X.X/10 — [Weak | Moderate | Strong] Opportunity

    Score interpretation:
      1.0–3.9  → Weak     (significant concerns, not recommended)
      4.0–6.9  → Moderate (promising but needs validation)
      7.0–10.0 → Strong   (compelling, recommend investment)

    Show the full calculation clearly.

    ## 5. Recommended Investment Structure
    If sufficient financial signals exist, estimate:
      - Recommended Investment Amount   : $XM
      - Estimated Equity Stake          : X%
      - Implied Pre-Money Valuation     : $XM
      - Investment Instrument           : SAFE / Convertible Note / Equity Round

    Ground estimates in:
      - Financial signals from the Financial Estimation Agent
      - Growth projections from the Growth Prediction Agent
      - Comparable deal structures for this stage and industry

    If financial data is insufficient:
    "Investment structure cannot be accurately estimated due to missing
    financial data. Recommend requesting detailed financials before proceeding."

    ## 6. Expected Return Potential
    Estimate the potential ROI range if the startup achieves its growth
    projections. Choose one:
      - Conservative: 2x – 4x
      - Base Case   : 5x – 10x
      - Optimistic  : 10x – 25x+

    Show the reasoning:
      - Entry valuation assumption
      - Exit valuation assumption (from growth agent or comparable M&A/IPO data)
      - Time horizon (3–5 years)
      - Key assumptions required for the optimistic case

    ## 7. Final Recommendation
    State exactly one of:
        ✅ INVEST   — strong opportunity with manageable risks
        👀 WATCH    — promising but requires more validation or traction
        ❌ REJECT   — high risk or limited growth potential

    Follow with 4–6 sentences explaining:
      - The primary reason for this recommendation
      - The most important condition that must hold for success
      - What milestone or evidence would change the recommendation
      - A one-sentence investment thesis (if INVEST or WATCH)

    ─────────────────────────────────────────────────────────────
    FORMAT NOTES:
    ─────────────────────────────────────────────────────────────
    - Total report: 800–1400 words.
    - Use the exact section headers listed above.
    - Professional VC investment committee tone throughout.
    - Every scored item must appear in the format shown above.
    - The Final Recommendation line must appear exactly as shown.
    ─────────────────────────────────────────────────────────────
8. STRUCTURED INVESTMENT METRICS (FOR VISUALIZATION)
─────────────────────────────────────────────────────────────

After the report, output a JSON block called:

INVESTMENT_METRICS

This will be used for analytics dashboards.

Format exactly as:

INVESTMENT_METRICS:
{
  "overall_investment_score": number (1-10),
  "market_score": number (1-10),
  "competition_score": number (1-10),
  "financial_score": number (1-10),
  "risk_score": number (1-10),
  "growth_score": number (1-10),
  "founder_score": number (1-10),
  "final_recommendation": "INVEST | WATCH | REJECT",
  "expected_return_low": number,
  "expected_return_high": number
}

Rules:
- Scores must match the reasoning in the report.
- JSON must be valid and parseable.
- Return values should represent expected exit multiple (e.g., 5x, 10x).                           
""")


# ═════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + " … [truncated]"


# ═════════════════════════════════════════════════════════════════════════════
# CONTEXT BUILDER
# ═════════════════════════════════════════════════════════════════════════════

def build_consolidated_context(system_state: dict) -> str:
    """
    Assemble every available agent output and signal into one consolidated
    intelligence brief for the LLM to synthesize.

    Includes both full reports (truncated) AND compact structured signals
    so the LLM gets both narrative depth and machine-readable numbers.
    """
    parts: list[str] = []

    # ── Core metadata ────────────────────────────────────────────────────────
    name    = system_state.get("startup_name",    "Unknown Startup")
    stage   = system_state.get("funding_stage")
    website = system_state.get("startup_website")

    parts.append(f"STARTUP NAME   : {name}")
    if stage:
        parts.append(f"FUNDING STAGE  : {stage}")
    if website:
        parts.append(f"WEBSITE        : {website}")

    # ── AGENT 1: Startup Data Agent ──────────────────────────────────────────
    summary = system_state.get("startup_summary")
    if summary:
        parts.append(f"\n{'─'*50}\nAGENT 1 — STARTUP INTELLIGENCE SUMMARY\n{'─'*50}\n{summary}")

    # ── AGENT 2: Market Research Agent ───────────────────────────────────────
    market_report = system_state.get("market_research_report")
    opp_flag      = system_state.get("market_opportunity_flag")
    if market_report or opp_flag:
        header = f"\n{'─'*50}\nAGENT 2 — MARKET RESEARCH\n{'─'*50}"
        body   = []
        if opp_flag:
            body.append(f"Market Opportunity Flag : {opp_flag}")
        if market_report:
            body.append(_truncate(market_report, 2000))
        parts.append(header + "\n" + "\n".join(body))

    # ── AGENT 3: Competitor Analysis Agent ───────────────────────────────────
    comp_report   = system_state.get("competitor_analysis_report")
    comp_intensity = system_state.get("competition_intensity")
    if comp_report or comp_intensity:
        header = f"\n{'─'*50}\nAGENT 3 — COMPETITOR ANALYSIS\n{'─'*50}"
        body   = []
        if comp_intensity:
            body.append(
                f"Competition Intensity Score : "
                f"{comp_intensity.get('score', 'N/A')}/10 — "
                f"{comp_intensity.get('level', 'UNKNOWN')}"
            )
        if comp_report:
            body.append(_truncate(comp_report, 2000))
        parts.append(header + "\n" + "\n".join(body))

    # ── AGENT 4: Financial Estimation Agent ──────────────────────────────────
    fin_report  = system_state.get("financial_estimation_report")
    fin_signals = system_state.get("financial_signals")
    if fin_report or fin_signals:
        header = f"\n{'─'*50}\nAGENT 4 — FINANCIAL ESTIMATION\n{'─'*50}"
        body   = []
        if fin_signals:
            body.append(
                f"Sustainability Stage : {fin_signals.get('sustainability_stage', 'UNKNOWN')}\n"
                f"Runway Class         : {fin_signals.get('runway_class', 'UNKNOWN')}\n"
                f"Valuation Available  : {fin_signals.get('valuation_available', False)}"
            )
        if fin_report:
            body.append(_truncate(fin_report, 2000))
        parts.append(header + "\n" + "\n".join(body))

    # ── AGENT 5: Risk Analysis Agent ─────────────────────────────────────────
    risk_report  = system_state.get("risk_analysis_report")
    risk_signals = system_state.get("risk_signals")
    if risk_report or risk_signals:
        header = f"\n{'─'*50}\nAGENT 5 — RISK ANALYSIS\n{'─'*50}"
        body   = []
        if risk_signals:
            body.append(
                f"Overall Risk Score : {risk_signals.get('overall_score', 'N/A')}/10 — "
                f"{risk_signals.get('overall_level', 'UNKNOWN')}\n"
                f"Red Flags Count    : {risk_signals.get('red_flags_count', 0)}\n"
                f"Risk Levels        : {risk_signals.get('risk_levels', {})}"
            )
        if risk_report:
            body.append(_truncate(risk_report, 2000))
        parts.append(header + "\n" + "\n".join(body))

    # ── AGENT 6: Growth Prediction Agent ─────────────────────────────────────
    growth_report  = system_state.get("growth_prediction_report")
    growth_signals = system_state.get("growth_signals")
    if growth_report or growth_signals:
        header = f"\n{'─'*50}\nAGENT 6 — GROWTH PREDICTION\n{'─'*50}"
        body   = []
        if growth_signals:
            body.append(
                f"Growth Classification : {growth_signals.get('growth_classification', 'UNKNOWN')}\n"
                f"Success Probability   : {growth_signals.get('success_probability', 'N/A')}%\n"
                f"Projected Valuation   : {growth_signals.get('valuation_range', 'N/A')}\n"
                f"Traction Level        : {growth_signals.get('traction_level', 'UNKNOWN')}\n"
                f"Team Strength         : {growth_signals.get('team_strength', 'UNKNOWN')}\n"
                f"Competitive Position  : {growth_signals.get('competitive_position', 'UNKNOWN')}"
            )
        if growth_report:
            body.append(_truncate(growth_report, 2000))
        parts.append(header + "\n" + "\n".join(body))

    # ── AGENT 7: Founder Intelligence Agent (optional) ───────────────────────
    founder_report = system_state.get("founder_intelligence_report")
    founder_signals = system_state.get("founder_signals")
    if founder_report or founder_signals:
        header = f"\n{'─'*50}\nAGENT 7 — FOUNDER INTELLIGENCE\n{'─'*50}"
        body   = []
        if founder_signals:
            body.append(str(founder_signals))
        if founder_report:
            body.append(_truncate(founder_report, 1500))
        parts.append(header + "\n" + "\n".join(body))

    # ── Raw pitch deck (bonus signal for Investment Agent) ───────────────────
    raw = system_state.get("startup_raw_data", {})
    pitch_text = raw.get("pitch_text")
    if pitch_text:
        parts.append(
            f"\n{'─'*50}\nPITCH DECK CONTENT (OCR)\n{'─'*50}\n"
            f"{_truncate(pitch_text, 2000)}"
        )

    if len(parts) <= 3:
        parts.append(
            "\n[Note: Very limited intelligence available. "
            "Investment decision will rely on startup name and funding stage only. "
            "Confidence in this assessment is LOW.]"
        )

    return "\n\n".join(parts)


# ═════════════════════════════════════════════════════════════════════════════
# LLM SYNTHESIS
# ═════════════════════════════════════════════════════════════════════════════

async def generate_investment_report(
    consolidated_context: str,
    groq_api_key:    str,
) -> str:
    """Synthesize startup context + live search evidence into the report."""
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name=GROQ_MODEL,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
    )

    user_message = (
        "Produce a final Investment Decision Report using the consolidated "
        "intelligence context provided below. Synthesize ALL agent outputs "
        "into a unified VC investment committee memo.\n\n"
        "══════════════════════════════════════════\n"
        "CONSOLIDATED INTELLIGENCE CONTEXT\n"
        "══════════════════════════════════════════\n"
        f"{consolidated_context}"
    )

    response = await llm.ainvoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ])
    return response.content.strip()


# ═════════════════════════════════════════════════════════════════════════════
# DECISION SIGNAL EXTRACTOR
# ═════════════════════════════════════════════════════════════════════════════

def extract_decision_signals(report: str) -> dict:
    """Parse the final structured decision signals for storage and display."""
    signals = {
        "final_recommendation":   "UNKNOWN",
        "investment_score":       None,
        "investment_level":       "UNKNOWN",
        "return_potential":       None,
        "implied_valuation":      None,
        "recommended_investment": None,
    }

    report_upper = report.upper()

    # Final Recommendation
    if "✅ INVEST"   in report or "INVEST"  in report_upper.split("FINAL RECOMMENDATION")[-1][:100]:
        signals["final_recommendation"] = "INVEST"
    elif "👀 WATCH"  in report or "WATCH"  in report_upper.split("FINAL RECOMMENDATION")[-1][:100]:
        signals["final_recommendation"] = "WATCH"
    elif "❌ REJECT" in report or "REJECT" in report_upper.split("FINAL RECOMMENDATION")[-1][:100]:
        signals["final_recommendation"] = "REJECT"

    # Overall Investment Score
    score_match = re.search(
        r"OVERALL\s+INVESTMENT\s+SCORE[:\s]+([\d\.]+)\s*/\s*10",
        report_upper,
    )
    if score_match:
        score = float(score_match.group(1))
        signals["investment_score"] = score
        if score < 4.0:
            signals["investment_level"] = "WEAK"
        elif score < 7.0:
            signals["investment_level"] = "MODERATE"
        else:
            signals["investment_level"] = "STRONG"

    # Return potential (look for Nx – Mx patterns)
    ret_match = re.search(
        r"(\d+x\s*[–\-]+\s*\d+x|\d+x\+)",
        report,
        re.IGNORECASE,
    )
    if ret_match:
        signals["return_potential"] = ret_match.group(1)

    # Implied valuation
    val_match = re.search(
        r"(?:PRE.MONEY\s+VALUATION|IMPLIED\s+VALUATION)[:\s]+(\$[\d,\.]+\s*[MB]?)",
        report,
        re.IGNORECASE,
    )
    if val_match:
        signals["implied_valuation"] = val_match.group(1).strip()

    # Recommended investment amount
    inv_match = re.search(
        r"RECOMMENDED\s+INVESTMENT\s+AMOUNT[:\s]+(\$[\d,\.]+\s*[MB]?)",
        report,
        re.IGNORECASE,
    )
    if inv_match:
        signals["recommended_investment"] = inv_match.group(1).strip()

    return signals

# ═════════════════════════════════════════════════════════════════════════════
# extract_investment_metrics
# ═════════════════════════════════════════════════════════════════════════════

def extract_investment_metrics(report):

    import json
    import re

    match = re.search(
        r"INVESTMENT_METRICS:\s*(\{[\s\S]*?\})",
        report
    )

    if not match:
        return {}

    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        print("⚠️ Invalid INVESTMENT_METRICS JSON")
        return {}

# ═════════════════════════════════════════════════════════════════════════════
# MAIN AGENT ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

async def run_investment_decision_agent(system_state: AgentState) -> AgentState:
    """
    Investment Decision Agent entry point.

    Reads from `system_state` (all prior agent outputs):
        startup_name / funding_stage / startup_website
        startup_summary                ← Agent 1
        market_research_report         ← Agent 2
        market_opportunity_flag        ← Agent 2
        competitor_analysis_report     ← Agent 3
        competition_intensity          ← Agent 3
        financial_estimation_report    ← Agent 4
        financial_signals              ← Agent 4
        risk_analysis_report           ← Agent 5
        risk_signals                   ← Agent 5
        growth_prediction_report       ← Agent 6
        growth_signals                 ← Agent 6
        founder_intelligence_report    ← Agent 7 (optional)
        founder_signals                ← Agent 7 (optional)
        startup_raw_data               ← Startup Data Agent

    Environment variable:
        agent2_llm — Groq API key

    Writes to `system_state`:
        investment_decision_report  (str)  — full VC investment memo
        investment_decision_signals (dict) — extracted decision signals

    Returns the updated system_state.
    """
    from agents.key_manager import get_groq_key
    groq_api_key = get_groq_key()
    if not groq_api_key:
        raise ValueError("Groq API key missing. Set 'agent2_llm' env var.")

    startup_name = system_state.get("startup_name", "Unknown Startup")

    print(f"\n{'='*60}")
    print(f"  Investment Decision Agent — {startup_name}")
    print(f"{'='*60}")

    # Step 1 — Build consolidated context
    print("\n[1/2] Building consolidated intelligence context …")
    consolidated_context = build_consolidated_context(system_state)
    print(f"      → {len(consolidated_context):,} chars assembled")

    # Log which prior agent outputs are present
    agent_coverage = {
        "Agent 1 — Startup Summary":   bool(system_state.get("startup_summary")),
        "Agent 2 — Market Research":   bool(system_state.get("market_research_report")),
        "Agent 3 — Competitor":        bool(system_state.get("competitor_analysis_report")),
        "Agent 4 — Financial":         bool(system_state.get("financial_estimation_report")),
        "Agent 5 — Risk":              bool(system_state.get("risk_analysis_report")),
        "Agent 6 — Growth":            bool(system_state.get("growth_prediction_report")),
        "Agent 7 — Founder":           bool(system_state.get("founder_intelligence_report")),
    }
    for agent, available in agent_coverage.items():
        status = "✅" if available else "⬜"
        print(f"      {status} {agent}")

    # Step 2 — Generate investment memo
    print("\n[2/2] Generating investment decision report with Groq LLM …")
    report = await generate_investment_report(consolidated_context, groq_api_key)
    print(f"      → {len(report):,} chars generated")

    # Extract structured signals
    signals = extract_decision_signals(report)
    metrics = extract_investment_metrics(report)

    # Store results
    system_state["investment_decision_report"]  = report
    system_state["investment_decision_signals"] = signals
    system_state["investment_metrics"] = metrics


    print(f"\n{'='*60}")
    print(f"  FINAL INVESTMENT DECISION")
    print(f"{'='*60}")
    recommendation = signals["final_recommendation"]
    emoji = {"INVEST": "✅", "WATCH": "👀", "REJECT": "❌"}.get(recommendation, "❓")
    print(f"  {emoji} Recommendation    : {recommendation}")
    print(f"  📊 Investment Score   : {signals['investment_score']}/10 — {signals['investment_level']}")
    print(f"  💰 Return Potential   : {signals['return_potential'] or 'N/A'}")
    print(f"  🏷️  Implied Valuation  : {signals['implied_valuation'] or 'N/A'}")
    print(f"  💵 Recommended Inv.   : {signals['recommended_investment'] or 'N/A'}")
    print(f"{'='*60}\n")
    print(f"  Stored → system_state['investment_decision_report']")

    return system_state

