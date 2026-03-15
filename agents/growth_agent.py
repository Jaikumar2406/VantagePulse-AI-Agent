import os
import re
import textwrap
from dotenv import load_dotenv
from state.state import AgentState
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from tavily import TavilyClient

load_dotenv()


# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════

GROQ_MODEL         = "llama-3.3-70b-versatile"
MAX_TOKENS         = 4096
TEMPERATURE        = 0

SEARCH_MAX_RESULTS = 5
SEARCH_MAX_CHARS   = 1_200
TOTAL_SEARCH_CAP   = 18_000


# ═════════════════════════════════════════════════════════════════════════════
# SEARCH QUERY TEMPLATES
# Growth-focused queries covering all key prediction dimensions
# ═════════════════════════════════════════════════════════════════════════════

QUERY_TEMPLATES = [
    # Market growth context
    "{industry} market growth forecast new one",
    "{industry} technology adoption trends emerging markets",

    # Comparable company trajectories
    "{industry} successful startups growth trajectory",
    "{industry} startup valuation benchmarks series A B",

    # Traction & ecosystem signals
    "{product} adoption trends customer growth",
    "{industry} startup ecosystem partnerships 2025",

    # Expansion potential
    "{industry} geographic expansion opportunities",
    "{industry} adjacent markets new opportunities",

    # Investor sentiment
    "{industry} venture capital investment trends 2025",
    "{startup_name} growth potential investors",
]


# ═════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT
# ═════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = textwrap.dedent("""\
    You are a senior growth analyst inside a venture capital startup evaluation
    system. Your role is to forecast the 3–5 year growth trajectory of a startup
    and assess its potential to become a large, scalable company.

    You will receive:
      1. STARTUP CONTEXT — all available data from prior intelligence agents
         (startup summary, market research, competitor analysis, financial signals,
         risk analysis).
      2. WEB SEARCH RESULTS — recently retrieved data on market growth trends,
         comparable company trajectories, industry adoption rates, and investor
         sentiment in this space.

    Your task is to produce a structured GROWTH PREDICTION REPORT grounded in
    both the startup context AND the web search evidence.

    ─────────────────────────────────────────────────────────────
    CRITICAL RULES — READ BEFORE WRITING:
    ─────────────────────────────────────────────────────────────
    - Do NOT fabricate metrics not present in context or search results.
    - If data is missing for a section, explicitly state that.
    - Cite the source of key claims (e.g., "per market research report",
      "per web search", "per pitch deck").
    - Use conservative but evidence-grounded reasoning.
    - Maintain a neutral, VC analyst tone throughout.

    ─────────────────────────────────────────────────────────────
    REPORT STRUCTURE (use these exact section headers):
    ─────────────────────────────────────────────────────────────

    ## 1. Market Growth Context
    Analyze the growth dynamics of the startup's market.
    Use web search results and market research to evaluate:
      - Overall market size and trajectory
      - CAGR and where the market is headed by 2028–2030
      - Emerging technology or behavioral trends accelerating growth
      - Technology adoption curve position (early adopter / early majority / etc.)
    Explain clearly whether the market environment supports rapid startup growth.
    Cite specific figures from web search where available.

    ## 2. Startup Traction Signals
    Identify evidence of early traction from the startup context:
      - Customer numbers, growth rate, or pilot deployments
      - Product usage or engagement signals
      - Revenue indicators or pricing validation
      - Partnerships, integrations, or enterprise pilots
      - Media coverage or award recognition signals

    If traction data is absent: "Startup traction data not available."
    Classify traction as: None / Early / Growing / Strong

    ## 3. Founder and Team Strength
    Evaluate the founding team's capability to lead long-term growth:
      - Domain expertise and industry credibility
      - Prior startup or corporate experience
      - Technical capability relative to product complexity
      - Team completeness (technical, commercial, operational)
      - Evidence of ability to hire and scale teams
    Classify team strength as: Weak / Developing / Strong / Exceptional

    ## 4. Competitive Position
    Analyse the startup's positioning relative to competitors using prior
    competitor analysis and web search findings:
      - Degree of product differentiation
      - Identified market gaps or whitespace the startup occupies
      - Competition intensity score (if available from prior agent)
      - Potential moats: network effects, data, switching costs, IP
      - Risk of being outcompeted or copied by larger players
    Classify competitive position as: Weak / Moderate / Strong / Dominant

    ## 5. Financial Growth Potential
    Using financial signals from the financial estimation agent:
      - Projected revenue growth trajectory (if signals available)
      - Scalability of the business model (unit economics, gross margin)
      - Ability to grow without proportional cost increases
      - Path to profitability or next funding milestone

    If financial indicators are missing:
    "Financial growth indicators not available."

    ## 6. Expansion Potential
    Evaluate the startup's ability to grow beyond its initial beachhead:
      - Geographic expansion: which new markets are accessible and how soon?
      - Product line extension: what adjacent features or products could be built?
      - Customer segment expansion: from initial niche toward broader markets?
      - Partnership or platform ecosystem potential?
    Use web search results about geographic and market expansion trends in
    this industry to inform the analysis.

    ## 7. Growth Prediction
    Based on all sections above, provide:

    ### Success Probability
    State a probability (0–100%) that the startup becomes a successful
    growth-stage company (Series B+ or equivalent) within 3–5 years.
    Format as:
        Success Probability: X%
    Show the key factors driving this estimate (positive and negative).

    ### Projected Valuation Range (3–5 Years)
    If sufficient signals exist, estimate a valuation range the startup
    could reach if it executes successfully:
        Projected Valuation Range: $XM – $YM
    Ground this in:
      - Comparable company valuations from web search
      - Revenue multiple assumptions (state the multiple used)
      - Growth rate assumptions
    If insufficient data: "Insufficient data to project valuation range."

    ## 8. Growth Classification
    Classify the startup as exactly one of:
        HIGH GROWTH POTENTIAL
        MODERATE GROWTH POTENTIAL
        LIMITED GROWTH POTENTIAL

    Format as:
        Growth Classification: [HIGH | MODERATE | LIMITED] GROWTH POTENTIAL

    Provide 4–6 sentences explaining the classification, covering:
      - The strongest growth tailwinds
      - The most significant growth constraints
      - What would need to be true for the startup to reach its full potential
      - One-line investment perspective

    ─────────────────────────────────────────────────────────────
    FORMAT NOTES:
    ─────────────────────────────────────────────────────────────
    - Total report: 700–1100 words.
    - Use the exact section headers listed above.
    - Every classification must appear on its own line in the specified format.
    - Be concise but thorough. Avoid padding.
    - Neutral, VC analyst tone throughout.
                                
    ─────────────────────────────────────────────────────────────
9. STRUCTURED GROWTH METRICS (FOR VISUALIZATION)
─────────────────────────────────────────────────────────────

After completing the report, output a JSON block called:

GROWTH_METRICS

This JSON will be used by the system to generate analytics charts.
Only include realistic values grounded in the analysis above.

Format exactly as:

GROWTH_METRICS:
{
  "success_probability": number,
  "traction_score": number (0-10),
  "team_strength_score": number (0-10),
  "competitive_position_score": number (0-10),
  "market_growth_score": number (0-10),
  "growth_projection": {
    "2025": number,
    "2026": number,
    "2027": number,
    "2028": number
  }
}

Rules:
- All scores must be between 0 and 10.
- growth_projection values represent estimated valuation in millions USD.
- JSON must be valid and parseable.
""")


# ═════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + " … [truncated]"


def _infer_keywords(system_state: dict) -> tuple[str, str]:
    """Derive industry and product search phrases from available context."""
    name       = system_state.get("startup_name", "startup")
    summary    = system_state.get("startup_summary", "")
    raw        = system_state.get("startup_raw_data", {})
    seed       = summary[:300] or (raw.get("website_text") or "")[:300] or name
    seed_lower = seed.lower()

    keyword_map = {
        "agri":          ("agri-tech precision farming",       "precision farming IoT"),
        "farm":          ("agri-tech precision farming",       "precision farming IoT"),
        "health":        ("digital health",                    "health tech platform"),
        "medtech":       ("medical technology",                "medical device software"),
        "fintech":       ("fintech financial technology",      "fintech payment platform"),
        "edtech":        ("edtech e-learning",                 "online education platform"),
        "climate":       ("climate tech clean energy",         "climate technology software"),
        "logistics":     ("logistics supply chain",            "supply chain SaaS"),
        "cybersecurity": ("cybersecurity",                     "cloud security software"),
        "ecommerce":     ("e-commerce technology",             "online retail platform"),
        "proptech":      ("proptech real estate",              "property management software"),
        "saas":          ("SaaS cloud software",               "cloud SaaS platform"),
        "ai":            ("artificial intelligence software",  "AI SaaS platform"),
        "data":          ("data analytics",                    "data intelligence SaaS"),
        "hr":            ("HR technology",                     "HR SaaS platform"),
        "legal":         ("legaltech",                         "legal technology software"),
        "insurance":     ("insurtech",                         "insurance technology platform"),
        "retail":        ("retail technology",                 "retail SaaS platform"),
    }
    for keyword, (ind, prod) in keyword_map.items():
        if keyword in seed_lower:
            return ind, prod
    return f"{name} technology", f"{name} platform"


# ═════════════════════════════════════════════════════════════════════════════
# WEB SEARCH MODULE  (Tavily)
# ═════════════════════════════════════════════════════════════════════════════

async def run_growth_searches(
    industry: str,
    product: str,
    startup_name: str,
    tavily_api_key: str,
) -> str:
    """Execute all growth-focused queries and return deduplicated evidence."""
    client    = TavilyClient(api_key=tavily_api_key)
    collected : list[str] = []
    seen_urls : set[str]  = set()

    for template in QUERY_TEMPLATES:
        query = template.format(
            industry=industry,
            product=product,
            startup_name=startup_name,
        )
        print(f"    🔍  {query}")

        try:
            resp    = await client.async_search(
                query=query,
                search_depth="basic",
                max_results=SEARCH_MAX_RESULTS,
            )
            results = resp.get("results", [])

            block = [f"\n### Search: {query}"]
            for r in results:
                url     = r.get("url", "")
                title   = r.get("title", "No title")
                content = _truncate(r.get("content", ""), SEARCH_MAX_CHARS)
                if url in seen_urls or not content.strip():
                    continue
                seen_urls.add(url)
                block.append(f"Source : {title}\nURL    : {url}\n{content}\n")

            collected.append("\n".join(block))

        except Exception as exc:
            collected.append(f"\n### Search: {query}\n[Search error: {exc}]")

    combined = "\n".join(collected)
    return _truncate(combined, TOTAL_SEARCH_CAP)


# ═════════════════════════════════════════════════════════════════════════════
# CONTEXT BUILDER
# ═════════════════════════════════════════════════════════════════════════════

def build_startup_context(system_state: dict) -> str:
    """
    Assemble all available signals from system_state.
    Pulls from ALL prior agents for maximum signal richness.
    """
    parts: list[str] = []

    # Core metadata
    name    = system_state.get("startup_name",    "Unknown Startup")
    stage   = system_state.get("funding_stage")
    website = system_state.get("startup_website")

    parts.append(f"STARTUP NAME: {name}")
    if stage:
        parts.append(f"FUNDING STAGE: {stage}")
    if website:
        parts.append(f"WEBSITE: {website}")

    # Startup summary
    summary = system_state.get("startup_summary")
    if summary:
        parts.append(f"\n=== STARTUP INTELLIGENCE SUMMARY ===\n{summary}")

    # Risk analysis (most recent prior agent)
    risk_report = system_state.get("risk_analysis_report")
    if risk_report:
        parts.append(
            f"\n=== RISK ANALYSIS REPORT ===\n"
            f"{_truncate(risk_report, 2000)}"
        )

    risk_signals = system_state.get("risk_signals")
    if risk_signals:
        parts.append(
            f"\n=== RISK SIGNALS ===\n"
            f"Overall Score  : {risk_signals.get('overall_score', 'N/A')}/10 — "
            f"{risk_signals.get('overall_level', 'UNKNOWN')}\n"
            f"Red Flags      : {risk_signals.get('red_flags_count', 0)}\n"
            f"Risk Levels    : {risk_signals.get('risk_levels', {})}"
        )

    # Financial estimation
    fin_report = system_state.get("financial_estimation_report")
    if fin_report:
        parts.append(
            f"\n=== FINANCIAL ESTIMATION REPORT ===\n"
            f"{_truncate(fin_report, 2000)}"
        )

    fin_signals = system_state.get("financial_signals")
    if fin_signals:
        parts.append(
            f"\n=== FINANCIAL SIGNALS ===\n"
            f"Sustainability Stage : {fin_signals.get('sustainability_stage', 'UNKNOWN')}\n"
            f"Runway Class         : {fin_signals.get('runway_class', 'UNKNOWN')}\n"
            f"Valuation Available  : {fin_signals.get('valuation_available', False)}"
        )

    # Competitor analysis
    comp_report = system_state.get("competitor_analysis_report")
    if comp_report:
        parts.append(
            f"\n=== COMPETITOR ANALYSIS REPORT ===\n"
            f"{_truncate(comp_report, 1500)}"
        )

    comp_intensity = system_state.get("competition_intensity")
    if comp_intensity:
        parts.append(
            f"\n=== COMPETITION INTENSITY ===\n"
            f"Score : {comp_intensity.get('score', 'N/A')}/10 — "
            f"{comp_intensity.get('level', 'UNKNOWN')}"
        )

    # Market research
    market_report = system_state.get("market_research_report")
    if market_report:
        parts.append(
            f"\n=== MARKET RESEARCH REPORT ===\n"
            f"{_truncate(market_report, 2000)}"
        )

    opp_flag = system_state.get("market_opportunity_flag")
    if opp_flag:
        parts.append(f"\n=== MARKET OPPORTUNITY FLAG ===\n{opp_flag}")

    # Raw scraped / OCR data
    raw = system_state.get("startup_raw_data", {})

    pitch_text = raw.get("pitch_text")
    if pitch_text:
        parts.append(
            f"\n=== PITCH DECK CONTENT (OCR) ===\n"
            f"{_truncate(pitch_text, 2500)}"
        )

    website_text = raw.get("website_text")
    if website_text:
        parts.append(
            f"\n=== WEBSITE CONTENT ===\n"
            f"{_truncate(website_text, 1200)}"
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
            "Growth prediction will rely heavily on industry benchmarks "
            "and web search findings.]"
        )

    return "\n\n".join(parts)


# ═════════════════════════════════════════════════════════════════════════════
# LLM SYNTHESIS
# ═════════════════════════════════════════════════════════════════════════════

async def generate_growth_report(
    startup_context: str,
    search_results:  str,
    groq_api_key:    str,
) -> str:
    """Synthesize startup context + live search evidence into growth report."""
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name=GROQ_MODEL,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
    )

    user_message = (
        "Produce a full Growth Prediction Report using the startup context and "
        "web search evidence provided below.\n\n"
        "════════════════════════════════════════\n"
        "STARTUP CONTEXT\n"
        "══════════════════════════════════════════\n"
        f"{startup_context}\n\n"
        "══════════════════════════════════════════\n"
        "WEB SEARCH RESULTS\n"
        "════════════════════════════════════════\n"
        f"{search_results}"
    )

    response = await llm.ainvoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ])
    return response.content.strip()


# ═════════════════════════════════════════════════════════════════════════════
# GROWTH SIGNAL EXTRACTOR
# ═════════════════════════════════════════════════════════════════════════════

def extract_growth_signals(report: str) -> dict:
    """Parse structured growth signals from report text for downstream agents."""
    signals = {
        "growth_classification":  "UNKNOWN",
        "success_probability":    None,
        "valuation_range":        None,
        "traction_level":         "UNKNOWN",
        "team_strength":          "UNKNOWN",
        "competitive_position":   "UNKNOWN",
    }

    report_upper = report.upper()

    # Growth Classification
    for label in ["HIGH GROWTH POTENTIAL", "MODERATE GROWTH POTENTIAL", "LIMITED GROWTH POTENTIAL"]:
        if label in report_upper:
            signals["growth_classification"] = label
            break

    # Success Probability
    prob_match = re.search(
        r"SUCCESS\s+PROBABILITY[:\s]+(\d{1,3})\s*%",
        report_upper,
    )
    if prob_match:
        signals["success_probability"] = int(prob_match.group(1))

    # Valuation Range — e.g. "$10M – $50M" or "$10M - $50M"
    val_match = re.search(
        r"PROJECTED\s+VALUATION\s+RANGE[:\s]+(\$[\d,\.]+\s*[MB]?\s*[–\-]+\s*\$[\d,\.]+\s*[MB]?)",
        report,
        re.IGNORECASE,
    )
    if val_match:
        signals["valuation_range"] = val_match.group(1).strip()

    # Traction level
    for label in ["STRONG", "GROWING", "EARLY", "NONE"]:
        pattern = rf"TRACTION.*?:\s*{label}|{label}.*?TRACTION"
        if re.search(pattern, report_upper):
            signals["traction_level"] = label
            break

    # Team strength
    for label in ["EXCEPTIONAL", "STRONG", "DEVELOPING", "WEAK"]:
        pattern = rf"TEAM\s+STRENGTH.*?:\s*{label}|{label}.*?TEAM\s+STRENGTH"
        if re.search(pattern, report_upper):
            signals["team_strength"] = label
            break

    # Competitive position
    for label in ["DOMINANT", "STRONG", "MODERATE", "WEAK"]:
        pattern = rf"COMPETITIVE\s+POSITION.*?:\s*{label}|{label}.*?COMPETITIVE\s+POSITION"
        if re.search(pattern, report_upper):
            signals["competitive_position"] = label
            break

    return signals


# ═════════════════════════════════════════════════════════════════════════════
# extract_growth_metrics
# ═════════════════════════════════════════════════════════════════════════════

def extract_growth_metrics(report):

    import json
    import re

    match = re.search(r"GROWTH_METRICS:\s*(\{.*\})", report, re.DOTALL)

    if not match:
        return {}

    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        print("⚠️ Invalid JSON in GROWTH_METRICS")
        return {}

# ═════════════════════════════════════════════════════════════════════════════
# MAIN AGENT ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

async def run_growth_prediction_agent(system_state: AgentState) -> AgentState:
    """
    Growth Prediction Agent (Web-Search Enhanced) entry point.

    Reads from `system_state`:
        startup_name                 (str, required)
        funding_stage                (str, optional)
        startup_website              (str, optional)
        startup_summary              (str, optional)  ← Startup Data Agent
        market_research_report       (str, optional)  ← Market Research Agent
        market_opportunity_flag      (str, optional)
        competitor_analysis_report   (str, optional)  ← Competitor Analysis Agent
        competition_intensity        (dict, optional)
        financial_estimation_report  (str, optional)  ← Financial Estimation Agent
        financial_signals            (dict, optional)
        risk_analysis_report         (str, optional)  ← Risk Analysis Agent
        risk_signals                 (dict, optional)
        startup_raw_data             (dict, optional)

    Environment variables:
        agent2_llm     — Groq API key
        TAVILY_API_KEY — Tavily Search API key

    Writes to `system_state`:
        growth_prediction_report  (str)   — full structured 8-section report
        growth_search_results     (str)   — raw search evidence (audit trail)
        growth_signals            (dict)  — extracted structured growth signals

    Returns the updated system_state.
    """
    from agents.key_manager import get_groq_key
    groq_api_key   = get_groq_key()
    tavily_api_key = os.getenv("TAVILY_API_KEY")

    if not groq_api_key:
        raise ValueError("Groq API key missing. Set 'agent2_llm' env var.")
    if not tavily_api_key:
        raise ValueError("Tavily API key missing. Set 'TAVILY_API_KEY' env var.")

    startup_name = system_state.get("startup_name", "Unknown Startup")

    print(f"\n{'='*60}")
    print(f"  Growth Prediction Agent — {startup_name}")
    print(f"{'='*60}")

    # Step 1 — Infer keywords
    print("\n[1/3] Inferring industry & product keywords …")
    industry, product = _infer_keywords(system_state)
    print(f"      industry → {industry!r}")
    print(f"      product  → {product!r}")

    # Step 2 — Web searches
    print(f"\n[2/3] Running {len(QUERY_TEMPLATES)} growth-focused searches via Tavily …")
    search_results = await run_growth_searches(industry, product, startup_name, tavily_api_key)
    print(f"      → {len(search_results):,} chars of search evidence collected")

    # Step 3 — LLM synthesis
    print("\n[3/3] Generating growth prediction report with Groq LLM …")
    startup_context = build_startup_context(system_state)

    # Log available prior agent outputs
    available = []
    if system_state.get("startup_summary"):              available.append("startup_summary")
    if system_state.get("market_research_report"):       available.append("market_research")
    if system_state.get("competitor_analysis_report"):   available.append("competitor_analysis")
    if system_state.get("financial_estimation_report"):  available.append("financial_estimation")
    if system_state.get("risk_analysis_report"):         available.append("risk_analysis")
    raw = system_state.get("startup_raw_data", {})
    if raw.get("pitch_text"):    available.append("pitch_deck_ocr")
    if raw.get("website_text"):  available.append("website_text")
    if raw.get("linkedin_text"): available.append("linkedin_profile")
    print(f"      → Prior outputs: {', '.join(available) if available else 'none'}")

    report = await generate_growth_report(startup_context, search_results, groq_api_key)
    print(f"      → {len(report):,} chars generated")

    # Extract structured signals
    signals = extract_growth_signals(report)
    metrics = extract_growth_metrics(report)

    # Store results
    system_state["growth_prediction_report"] = report
    system_state["growth_search_results"]    = search_results
    system_state["growth_signals"]           = signals
    system_state["growth_metrics"]           = metrics

    print(f"\n[Agent] Done.")
    print(f"        Growth Classification : {signals['growth_classification']}")
    print(f"        Success Probability   : {signals['success_probability']}%")
    print(f"        Projected Valuation   : {signals['valuation_range'] or 'N/A'}")
    print(f"        Traction Level        : {signals['traction_level']}")
    print(f"        Team Strength         : {signals['team_strength']}")
    print(f"        Competitive Position  : {signals['competitive_position']}")
    print(f"        Stored → system_state['growth_prediction_report']")
    print(f"{'='*60}\n")

    return system_state

