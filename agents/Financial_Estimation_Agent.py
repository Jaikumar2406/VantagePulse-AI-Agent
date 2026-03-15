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

GROQ_MODEL   = "llama-3.3-70b-versatile"
MAX_TOKENS   = 4096
TEMPERATURE  = 0


# ═════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT
# ═════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = textwrap.dedent("""\
    You are a senior financial analyst inside a venture capital startup
    evaluation system. You specialize in early-stage startup financial
    assessment.

    You will receive a STARTUP CONTEXT section that may include:
      - Startup name and funding stage
      - A startup intelligence summary
      - Website scraped content
      - Pitch deck OCR-extracted content
      - Founder LinkedIn profile data
      - Market research and competitor analysis reports from prior agents

    Your task is to produce a structured FINANCIAL ESTIMATION REPORT based
    ONLY on the information present in the provided context.

    ─────────────────────────────────────────────────────────────
    CRITICAL RULES — READ BEFORE WRITING:
    ─────────────────────────────────────────────────────────────
    - NEVER fabricate financial numbers.
    - NEVER assume revenue, burn rate, CAC, LTV, or valuation without
      explicit evidence in the provided context.
    - If a metric cannot be derived from the available data, write the
      exact phrase: "Information not provided."
    - Only perform arithmetic estimates when BOTH required inputs are
      explicitly present (e.g., MRR = customer count × price per customer,
      only if both values appear in context).
    - When estimating burn rate, use team size and regional salary norms
      as the signal — but state the assumption clearly.
    - Use conservative, venture-analyst-style reasoning throughout.
    - Cite which part of the context each data point comes from
      (e.g., "per pitch deck", "per startup summary", "per website content").

    ─────────────────────────────────────────────────────────────
    REPORT STRUCTURE (use these exact section headers):
    ─────────────────────────────────────────────────────────────

    ## 1. Revenue Indicators
    Identify signals related to revenue:
      - MRR / ARR (if stated or derivable)
      - Pricing tiers or subscription costs
      - Number of paying customers
      - Revenue stage: pre-revenue / early revenue / growing revenue

    If customer count AND pricing are both available, compute estimated MRR/ARR
    and show the formula clearly:
        Estimated ARR = [N customers] × [$price/year] = $X

    If revenue data is missing: "Information not provided."

    ## 2. Burn Rate Estimation
    Estimate monthly cash burn only if sufficient signals exist:
      - Team size × estimated average monthly salary for the industry/region
      - Known infrastructure or COGS costs
      - Marketing and operational spend indicators

    Show reasoning step by step. State all assumptions explicitly.
    If insufficient data: "Burn rate information not available."

    ## 3. Runway Estimation
    If BOTH total funding raised AND monthly burn rate are known:
        Runway (months) = Total Funding Raised / Monthly Burn Rate

    Show calculation. Classify runway as:
      - Critical  (<6 months)
      - Tight     (6–12 months)
      - Adequate  (12–18 months)
      - Healthy   (>18 months)

    If either input is missing:
    "Runway cannot be estimated due to missing financial data."

    ## 4. Unit Economics
    Evaluate only if sufficient signals exist:
      - Customer Acquisition Cost (CAC)
      - Lifetime Value (LTV)
      - LTV:CAC ratio (healthy benchmark: >3:1)
      - Payback period
      - Gross margin indicators

    If data is missing: "Unit economics information not provided."

    ## 5. Valuation Range
    Estimate a rough valuation range ONLY if revenue or strong growth
    metrics are available. Apply appropriate industry revenue multiples:
      - SaaS         : 5–15× ARR at early stage
      - Hardware+SaaS: 3–8× ARR
      - Marketplace  : 3–6× GMV or revenue
      - Pre-revenue  : reference comparable seed-stage funding rounds

    Express as a range: "$XM – $YM"
    If insufficient data: "Insufficient data to estimate valuation."

    ## 6. Financial Sustainability Assessment
    Synthesize all findings into a qualitative assessment.
    Classify as exactly one of:
      - Pre-Revenue / Concept Stage
      - Early Traction
      - Uncertain Financial Position
      - Strong Financial Momentum

    Provide 3–5 sentences explaining:
      - The classification rationale
      - Key financial strengths identified
      - Key financial risks or gaps
      - What additional data would most improve confidence in the assessment

    ─────────────────────────────────────────────────────────────
    FORMAT NOTES:
    ─────────────────────────────────────────────────────────────
    - Total report: 500–900 words.
    - Use the exact section headers listed above.
    - Be concise but thorough. Avoid padding.
    - Neutral, VC analyst tone throughout.
    ─────────────────────────────────────────────────────────────
7. STRUCTURED FINANCIAL METRICS (FOR VISUALIZATION)
─────────────────────────────────────────────────────────────

After completing the report, output a JSON block called:

FINANCIAL_METRICS

This JSON will be used to generate financial analytics charts.

Format exactly as:

FINANCIAL_METRICS:
{
  "revenue_stage": "Pre-Revenue | Early Revenue | Growing Revenue",
  "estimated_mrr": number | null,
  "estimated_arr": number | null,
  "estimated_burn_rate_monthly": number | null,
  "estimated_runway_months": number | null,
  "ltv_cac_ratio": number | null,
  "valuation_low": number | null,
  "valuation_high": number | null
}

Rules:
- Use null if the information cannot be derived.
- All numbers should be realistic and grounded in the analysis.
- JSON must be valid and parseable.
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

def build_startup_context(system_state: dict) -> str:
    """
    Assemble all available startup signals from system_state into one block.
    Priority order (highest signal first):
        1. startup_summary         (Startup Data Agent)
        2. market_research_report  (Market Research Agent)
        3. competitor_analysis_report (Competitor Analysis Agent)
        4. pitch_text              (raw OCR from pitch deck — richest financial data)
        5. website_text            (scraped pages)
        6. linkedin_text           (founder background)
    """
    parts: list[str] = []

    # ── Core metadata ────────────────────────────────────────────────────────
    name    = system_state.get("startup_name",    "Unknown Startup")
    stage   = system_state.get("funding_stage")
    website = system_state.get("startup_website")

    parts.append(f"STARTUP NAME: {name}")
    if stage:
        parts.append(f"FUNDING STAGE: {stage}")
    if website:
        parts.append(f"WEBSITE: {website}")

    # ── Startup intelligence summary ─────────────────────────────────────────
    summary = system_state.get("startup_summary")
    if summary:
        parts.append(f"\n=== STARTUP INTELLIGENCE SUMMARY ===\n{summary}")

    # ── Market research report ───────────────────────────────────────────────
    market_report = system_state.get("market_research_report")
    if market_report:
        parts.append(
            f"\n=== MARKET RESEARCH REPORT ===\n"
            f"{_truncate(market_report, 2000)}"
        )

    # ── Competitor analysis report ───────────────────────────────────────────
    comp_report = system_state.get("competitor_analysis_report")
    if comp_report:
        parts.append(
            f"\n=== COMPETITOR ANALYSIS REPORT ===\n"
            f"{_truncate(comp_report, 1500)}"
        )

    # ── Raw scraped / OCR data ────────────────────────────────────────────────
    raw = system_state.get("startup_raw_data", {})

    pitch_text = raw.get("pitch_text")
    if pitch_text:
        # Pitch deck is highest value for financial signals — give it more space
        parts.append(
            f"\n=== PITCH DECK CONTENT (OCR) ===\n"
            f"{_truncate(pitch_text, 4000)}"
        )

    website_text = raw.get("website_text")
    if website_text:
        parts.append(
            f"\n=== WEBSITE CONTENT ===\n"
            f"{_truncate(website_text, 2000)}"
        )

    linkedin_text = raw.get("linkedin_text")
    if linkedin_text:
        parts.append(
            f"\n=== FOUNDER LINKEDIN PROFILE ===\n"
            f"{_truncate(linkedin_text, 1000)}"
        )

    if len(parts) <= 3:
        parts.append(
            "\n[Note: Very limited startup data available. "
            "Base analysis only on startup name and funding stage.]"
        )

    return "\n\n".join(parts)


# ═════════════════════════════════════════════════════════════════════════════
# LLM SYNTHESIS
# ═════════════════════════════════════════════════════════════════════════════

async def generate_financial_report(
    startup_context: str,
    groq_api_key:    str,
) -> str:
    """Synthesize startup context into the financial report."""
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name=GROQ_MODEL,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
    )

    user_message = (
        "Produce a full Financial Estimation Report using ONLY startup "
        "context provided below. Do not use any external information.\n\n"
        "════════════════════════════════════════\n"
        "STARTUP CONTEXT\n"
        "════════════════════════════════════════\n"
        f"{startup_context}"
    )

    response = await llm.ainvoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ])
    return response.content.strip()


# ═════════════════════════════════════════════════════════════════════════════
# FINANCIAL SIGNAL EXTRACTOR
# ═════════════════════════════════════════════════════════════════════════════

def extract_financial_signals(report: str) -> dict:
    """
    Parse key structured signals from the report for downstream agents.
    Returns a dict with best-effort extracted values.
    """
    signals = {
        "sustainability_stage": "UNKNOWN",
        "runway_class":         "UNKNOWN",
        "valuation_available":  False,
    }

    report_upper = report.upper()

    # Sustainability stage classification
    for label in [
        "STRONG FINANCIAL MOMENTUM",
        "EARLY TRACTION",
        "UNCERTAIN FINANCIAL POSITION",
        "PRE-REVENUE",
        "PRE-REVENUE / CONCEPT STAGE",
    ]:
        if label in report_upper:
            signals["sustainability_stage"] = label
            break

    # Runway classification
    for label in ["CRITICAL", "TIGHT", "ADEQUATE", "HEALTHY"]:
        if label in report_upper:
            signals["runway_class"] = label
            break

    # Check whether a valuation range was produced
    if "INSUFFICIENT DATA TO ESTIMATE VALUATION" not in report_upper:
        if re.search(r"\$\s*\d+[\.,]?\d*\s*[MB]", report, re.IGNORECASE):
            signals["valuation_available"] = True

    return signals

# ═════════════════════════════════════════════════════════════════════════════
# MAIN AGENT ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════


def extract_financial_metrics(report):

    import json
    import re

    match = re.search(
        r"FINANCIAL_METRICS:\s*(\{[\s\S]*?\})",
        report
    )

    if not match:
        return {}

    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        print("Invalid FINANCIAL_METRICS JSON")
        return {}
# ═════════════════════════════════════════════════════════════════════════════
# MAIN AGENT ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

async def run_financial_estimation_agent(system_state: AgentState) -> AgentState:
    """
    Financial Estimation Agent entry point.

    Reads from `system_state`:
        startup_name                (str, required)
        funding_stage               (str, optional)
        startup_website             (str, optional)
        startup_summary             (str, optional)  ← Startup Data Agent
        market_research_report      (str, optional)  ← Market Research Agent
        competitor_analysis_report  (str, optional)  ← Competitor Analysis Agent
        startup_raw_data            (dict, optional) ← website / pitch / linkedin text

    Environment variable:
        agent2_llm — Groq API key

    Writes to `system_state`:
        financial_estimation_report  (str)  — full structured 6-section report
        financial_signals            (dict) — extracted structured signals

    Returns the updated system_state.
    """
    from agents.key_manager import get_groq_key
    groq_api_key = get_groq_key()
    if not groq_api_key:
        raise ValueError("Groq API key missing. Set 'agent2_llm' env var.")

    startup_name = system_state.get("startup_name", "Unknown Startup")

    print(f"\n{'='*60}")
    print(f"  Financial Estimation Agent — {startup_name}")
    print(f"{'='*60}")

    # Step 1 — Assemble context from system state
    print("\n[1/2] Assembling startup context from system state …")
    startup_context = build_startup_context(system_state)
    print(f"      → {len(startup_context):,} chars of context assembled")

    # Log which sources are available
    raw = system_state.get("startup_raw_data", {})
    sources = []
    if system_state.get("startup_summary"):             sources.append("startup_summary")
    if system_state.get("market_research_report"):      sources.append("market_research_report")
    if system_state.get("competitor_analysis_report"):  sources.append("competitor_analysis_report")
    if raw.get("pitch_text"):                           sources.append("pitch_deck_ocr")
    if raw.get("website_text"):                         sources.append("website_text")
    if raw.get("linkedin_text"):                        sources.append("linkedin_profile")
    print(f"      → Sources used: {', '.join(sources) if sources else 'metadata only'}")

    # Step 2 — Generate report with Groq LLM
    print("\n[2/2] Generating financial estimation report with Groq LLM …")
    report = await generate_financial_report(startup_context, groq_api_key)
    print(f"      → {len(report):,} chars generated")

    # Extract structured signals for downstream agents
    signals = extract_financial_signals(report)
    metrics = extract_financial_metrics(report)


    # Store results
    system_state["financial_estimation_report"] = report
    system_state["financial_signals"]           = signals
    system_state["financial_metrics"] = metrics

    print(f"\n[Agent] Done.")
    print(f"        Sustainability Stage : {signals['sustainability_stage']}")
    print(f"        Runway Class         : {signals['runway_class']}")
    print(f"        Valuation Available  : {signals['valuation_available']}")
    print(f"        Stored → system_state['financial_estimation_report']")
    print(f"{'='*60}\n")

    return system_state
