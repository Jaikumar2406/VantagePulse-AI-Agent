"""
Risk Analysis Agent (Web-Search Enhanced)
------------------------------------------
Performs targeted web searches to gather current risk signals — regulatory
changes, technology maturity, competitive threats, financial red flags —
then combines them with all prior agent outputs to generate a comprehensive
risk analysis report via Groq LLM.

Search provider : Tavily Search API  (https://tavily.com)
LLM             : Groq  (llama-3.3-70b-versatile via LangChain)

Install dependencies:
    pip install langchain-groq langchain-core tavily-python python-dotenv

Environment variables required (.env):
    agent2_llm     = <your Groq API key>
    TAVILY_API_KEY = <your Tavily API key>
"""

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
# Risk-focused queries covering all 6 risk categories
# ═════════════════════════════════════════════════════════════════════════════

QUERY_TEMPLATES = [
    # Market Risk
    "{industry} market risks challenges ",
    "{industry} market saturation growth concerns",

    # Technology Risk
    "{product} technology risks limitations",
    "{product} technical challenges implementation",

    # Regulatory Risk
    "{industry} regulations compliance requirements ",
    "{industry} legal barriers government policy",

    # Competition Risk
    "{industry} competitive threats incumbents",
    "{startup_name} risks investor concerns",

    # Financial & Execution Risk
    "{industry} startup failure reasons",
    "{industry} startup execution challenges scaling",
]


# ═════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT
# ═════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = textwrap.dedent("""\
    You are a senior risk analyst inside a venture capital startup evaluation
    system. Your role is to identify, evaluate, and score the key risks
    associated with investing in a startup.

    You will receive:
      1. STARTUP CONTEXT — all available data from prior intelligence agents.
      2. WEB SEARCH RESULTS — recently retrieved information about industry
         risks, regulatory changes, technology challenges, and competitive threats.

    Your task is to produce a structured RISK ANALYSIS REPORT grounded in
    both the startup context AND the web search evidence.

    ─────────────────────────────────────────────────────────────
    CRITICAL RULES — READ BEFORE WRITING:
    ─────────────────────────────────────────────────────────────
    - Do NOT fabricate information absent from context or search results.
    - If data is missing for a category, state:
      "Information not available to assess this risk fully."
    - Cite the source of key claims (e.g., "per pitch deck", "per web search").
    - Assign risk levels using ONLY: Low / Moderate / High.
    - Maintain a neutral, evidence-based VC analyst tone.

    ─────────────────────────────────────────────────────────────
    REPORT STRUCTURE (use these exact section headers):
    ─────────────────────────────────────────────────────────────

    ## 1. Market Risk
    Analyze whether the startup operates in a viable, scalable market.
    Evaluate:
      - Market size (TAM / SAM) and venture-scale potential
      - Market growth rate and trajectory (from research + web search)
      - Market maturity (emerging vs. saturated)
      - Customer demand uncertainty
      - Market timing risk (too early or too late)
    Conclude with: Market Risk Level — Low / Moderate / High
    Provide 3–5 sentences of reasoning, citing sources.

    ## 2. Technology Risk
    Evaluate the technological feasibility of the startup's product.
    Consider:
      - Complexity and maturity of the core technology
      - Dependence on unproven or emerging technologies
      - Reliance on third-party platforms, APIs, or hardware
      - Technical depth of the founding team
      - Product development stage (concept / prototype / live)
    Use web search results to assess technology maturity in this space.
    Conclude with: Technology Risk Level — Low / Moderate / High
    Provide 3–5 sentences of reasoning, citing sources.

    ## 3. Execution Risk
    Evaluate whether the team can execute the business plan.
    Consider:
      - Founder experience and domain expertise
      - Team size relative to operational complexity
      - Track record of building and scaling
      - Go-to-market complexity
    Reference web search findings on typical execution challenges in this industry.
    Conclude with: Execution Risk Level — Low / Moderate / High
    Provide 3–5 sentences of reasoning.

    ## 4. Regulatory Risk
    Evaluate legal and regulatory barriers using both context and web search.
    Consider:
      - Industry-specific compliance requirements
      - Government approvals or certifications needed
      - Data privacy laws (GDPR, CCPA, local regulations)
      - Cross-border or export restrictions
      - Political or policy risk in target geographies
    Prioritise web search results for current regulatory developments.
    Conclude with: Regulatory Risk Level — Low / Moderate / High
    Provide 3–5 sentences of reasoning, citing sources.

    ## 5. Financial Risk
    Evaluate the startup's financial sustainability.
    Consider:
      - Burn rate and runway (from financial signals)
      - Revenue traction stage
      - Funding stage and capitalisation
      - Unit economics health
      - Dependency on future fundraising

    If financial data is unavailable:
    "Financial risk cannot be fully assessed due to missing financial data."

    Conclude with: Financial Risk Level — Low / Moderate / High (or "Indeterminate")
    Provide 3–5 sentences of reasoning.

    ## 6. Competition Risk
    Analyse competitive threats using prior competitor analysis AND web search.
    Consider:
      - Number and strength of direct competitors
      - Presence of large, well-funded incumbents
      - Barriers to differentiation
      - Risk of commoditisation
      - Recent competitive moves or new entrants (from web search)
    Conclude with: Competition Risk Level — Low / Moderate / High
    Provide 3–5 sentences of reasoning, citing sources.

    ## 7. Overall Risk Score
    Synthesize all six categories into a single Overall Risk Score.

    Scoring scale:
      1–3  → Low Risk      (manageable, strong fundamentals)
      4–6  → Moderate Risk (notable but manageable risks)
      7–10 → High Risk     (significant concerns requiring mitigation)

    Format as:
        Overall Risk Score: X/10 — [Low Risk | Moderate Risk | High Risk]

    Show the weight given to each category and reasoning for the final score.
    Provide 3–5 sentences of justification.

    ## 8. Major Red Flags
    List only genuine, evidence-backed critical issues in a numbered list.
    Examples:
      - TAM < $500M with limited expansion path
      - No differentiation from well-funded direct competitors
      - Core market blocked by pending regulation
      - Founding team with no relevant domain experience
      - Runway < 6 months with no bridge plan disclosed

    If no critical red flags: "No critical red flags identified."

    ## 9. Final Risk Assessment
    Provide a 4–6 sentence investor-facing summary covering:
      - The startup's two or three biggest risks
      - Whether those risks appear manageable
      - Overall risk outlook: Favourable / Neutral / Cautious
      - Recommended due diligence priorities

    ─────────────────────────────────────────────────────────────
    FORMAT NOTES:
    ─────────────────────────────────────────────────────────────
    - Total report: 700–1100 words.
    - Use the exact section headers listed above.
    - Every risk level conclusion must appear on its own line:
      "[Category] Risk Level — Low / Moderate / High"
                                
    ─────────────────────────────────────────────────────────────
10. STRUCTURED RISK METRICS (FOR VISUALIZATION)
─────────────────────────────────────────────────────────────

After completing the report, output a JSON block called:

RISK_METRICS

This JSON will be used by the system to generate risk analytics charts.

Format exactly as:

RISK_METRICS:
{
  "market_risk_score": number (1-10),
  "technology_risk_score": number (1-10),
  "execution_risk_score": number (1-10),
  "regulatory_risk_score": number (1-10),
  "financial_risk_score": number (1-10),
  "competition_risk_score": number (1-10),
  "overall_risk_score": number (1-10),
  "overall_risk_level": "Low | Moderate | High",
  "red_flags_count": number
}

Rules:
- Scores must align with the reasoning in the report.
- Scores must be between 1 and 10.
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
    name    = system_state.get("startup_name", "startup")
    summary = system_state.get("startup_summary", "")
    raw     = system_state.get("startup_raw_data", {})
    seed    = summary[:300] or (raw.get("website_text") or "")[:300] or name
    seed_lower = seed.lower()

    keyword_map = {
        "agri":          ("agri-tech precision farming",        "precision farming IoT"),
        "farm":          ("agri-tech precision farming",        "precision farming IoT"),
        "health":        ("digital health",                     "health tech platform"),
        "medtech":       ("medical technology",                 "medical device software"),
        "fintech":       ("fintech financial technology",       "fintech payment platform"),
        "edtech":        ("edtech e-learning",                  "online education platform"),
        "climate":       ("climate tech clean energy",          "climate technology software"),
        "logistics":     ("logistics supply chain",             "supply chain SaaS"),
        "cybersecurity": ("cybersecurity",                      "cloud security software"),
        "ecommerce":     ("e-commerce technology",              "online retail platform"),
        "proptech":      ("proptech real estate",               "property management software"),
        "saas":          ("SaaS cloud software",                "cloud SaaS platform"),
        "ai":            ("artificial intelligence software",   "AI SaaS platform"),
        "data":          ("data analytics",                     "data intelligence SaaS"),
        "hr":            ("HR technology",                      "HR SaaS platform"),
        "legal":         ("legaltech",                          "legal technology software"),
        "insurance":     ("insurtech",                          "insurance technology platform"),
        "retail":        ("retail technology",                  "retail SaaS platform"),
    }
    for keyword, (ind, prod) in keyword_map.items():
        if keyword in seed_lower:
            return ind, prod
    return f"{name} technology", f"{name} platform"


# ═════════════════════════════════════════════════════════════════════════════
# WEB SEARCH MODULE  (Tavily)
# ═════════════════════════════════════════════════════════════════════════════

async def run_risk_searches(
    industry: str,
    product: str,
    startup_name: str,
    tavily_api_key: str,
) -> str:
    """
    Execute all risk-focused search queries and return deduplicated evidence.
    """
    client    = TavilyClient(api_key=tavily_api_key)
    collected : list[str] = []
    seen_urls : set[str]  = set()

    for query in QUERY_TEMPLATES:
        formatted = query.format(industry=industry, product=product, startup_name=startup_name)
        print(f"    🔍  {query}")

        try:
            resp    = await client.async_search(
                query=formatted,
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
    Assemble all available startup signals from system_state.
    Pulls from all prior agents to maximise signal richness.
    """
    parts: list[str] = []

    name    = system_state.get("startup_name",    "Unknown Startup")
    stage   = system_state.get("funding_stage")
    website = system_state.get("startup_website")

    parts.append(f"STARTUP NAME: {name}")
    if stage:
        parts.append(f"FUNDING STAGE: {stage}")
    if website:
        parts.append(f"WEBSITE: {website}")

    summary = system_state.get("startup_summary")
    if summary:
        parts.append(f"\n=== STARTUP INTELLIGENCE SUMMARY ===\n{summary}")

    fin_report = system_state.get("financial_estimation_report")
    if fin_report:
        parts.append(
            f"\n=== FINANCIAL ESTIMATION REPORT ===\n"
            f"{_truncate(fin_report, 2500)}"
        )

    fin_signals = system_state.get("financial_signals")
    if fin_signals:
        parts.append(
            f"\n=== FINANCIAL SIGNALS ===\n"
            f"Sustainability Stage : {fin_signals.get('sustainability_stage', 'UNKNOWN')}\n"
            f"Runway Class         : {fin_signals.get('runway_class', 'UNKNOWN')}\n"
            f"Valuation Available  : {fin_signals.get('valuation_available', False)}"
        )

    comp_report = system_state.get("competitor_analysis_report")
    if comp_report:
        parts.append(
            f"\n=== COMPETITOR ANALYSIS REPORT ===\n"
            f"{_truncate(comp_report, 2000)}"
        )

    comp_intensity = system_state.get("competition_intensity")
    if comp_intensity:
        parts.append(
            f"\n=== COMPETITION INTENSITY ===\n"
            f"Score : {comp_intensity.get('score', 'N/A')}/10 — "
            f"{comp_intensity.get('level', 'UNKNOWN')}"
        )

    market_report = system_state.get("market_research_report")
    if market_report:
        parts.append(
            f"\n=== MARKET RESEARCH REPORT ===\n"
            f"{_truncate(market_report, 2000)}"
        )

    opp_flag = system_state.get("market_opportunity_flag")
    if opp_flag:
        parts.append(f"\n=== MARKET OPPORTUNITY FLAG ===\n{opp_flag}")

    raw = system_state.get("startup_raw_data", {})

    pitch_text = raw.get("pitch_text")
    if pitch_text:
        parts.append(
            f"\n=== PITCH DECK CONTENT (OCR) ===\n"
            f"{_truncate(pitch_text, 3000)}"
        )

    website_text = raw.get("website_text")
    if website_text:
        parts.append(
            f"\n=== WEBSITE CONTENT ===\n"
            f"{_truncate(website_text, 1500)}"
        )

    linkedin_text = raw.get("linkedin_text")
    if linkedin_text:
        parts.append(
            f"\n=== FOUNDER LINKEDIN PROFILE ===\n"
            f"{_truncate(linkedin_text, 1000)}"
        )

    return "\n\n".join(parts)


# ═════════════════════════════════════════════════════════════════════════════
# LLM SYNTHESIS
# ═════════════════════════════════════════════════════════════════════════════

async def generate_risk_report(
    startup_context: str,
    search_results:  str,
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
        "Produce a full Risk Analysis Report using startup context and "
        "web search evidence provided below.\n\n"
        "══════════════════════════════════════════\n"
        "STARTUP CONTEXT\n"
        "══════════════════════════════════════════\n"
        f"{startup_context}\n\n"
        "══════════════════════════════════════════\n"
        "WEB SEARCH RESULTS\n"
        "══════════════════════════════════════════\n"
        f"{search_results}"
    )

    response = await llm.ainvoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ])
    return response.content.strip()


# ═════════════════════════════════════════════════════════════════════════════
# RISK SIGNAL EXTRACTOR
# ═════════════════════════════════════════════════════════════════════════════

def extract_risk_signals(report: str) -> dict:
    """Parse structured risk signals from report text for downstream agents."""
    signals = {
        "overall_score":  None,
        "overall_level":  "UNKNOWN",
        "risk_levels": {
            "market":      "UNKNOWN",
            "technology":  "UNKNOWN",
            "execution":   "UNKNOWN",
            "regulatory":  "UNKNOWN",
            "financial":   "UNKNOWN",
            "competition": "UNKNOWN",
        },
        "red_flags_count": 0,
    }

    report_upper = report.upper()

    # Overall Risk Score
    score_match = re.search(
        r"OVERALL\s+RISK\s+SCORE[:\s]+(\d{1,2})\s*/\s*10",
        report_upper,
    )
    if score_match:
        score = int(score_match.group(1))
        signals["overall_score"] = score
        signals["overall_level"] = (
            "LOW" if score <= 3 else "MODERATE" if score <= 6 else "HIGH"
        )

    # Individual risk levels
    category_patterns = {
        "market":      r"MARKET\s+RISK\s+LEVEL\s*[—\-]+\s*(LOW|MODERATE|HIGH)",
        "technology":  r"TECHNOLOGY\s+RISK\s+LEVEL\s*[—\-]+\s*(LOW|MODERATE|HIGH)",
        "execution":   r"EXECUTION\s+RISK\s+LEVEL\s*[—\-]+\s*(LOW|MODERATE|HIGH)",
        "regulatory":  r"REGULATORY\s+RISK\s+LEVEL\s*[—\-]+\s*(LOW|MODERATE|HIGH)",
        "financial":   r"FINANCIAL\s+RISK\s+LEVEL\s*[—\-]+\s*(LOW|MODERATE|HIGH|INDETERMINATE)",
        "competition": r"COMPETITION\s+RISK\s+LEVEL\s*[—\-]+\s*(LOW|MODERATE|HIGH)",
    }
    for category, pattern in category_patterns.items():
        match = re.search(pattern, report_upper)
        if match:
            signals["risk_levels"][category] = match.group(1)

    # Count red flags
    red_flags_section = re.search(
        r"## 8\. MAJOR RED FLAGS(.*?)## 9\.",
        report,
        re.DOTALL | re.IGNORECASE,
    )
    if red_flags_section:
        items = re.findall(
            r"^\s*\d+[\.\)]\s+.+",
            red_flags_section.group(1),
            re.MULTILINE,
        )
        signals["red_flags_count"] = len(items)

    return signals


# ═════════════════════════════════════════════════════════════════════════════
# extract_risk_metrics
# ═════════════════════════════════════════════════════════════════════════════


def extract_risk_metrics(report):

    import json
    import re

    match = re.search(
        r"RISK_METRICS:\s*(\{[\s\S]*?\})",
        report
    )

    if not match:
        return {}

    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        print("⚠️ Invalid RISK_METRICS JSON")
        return {}

# ═════════════════════════════════════════════════════════════════════════════
# MAIN AGENT ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

async def run_risk_analysis_agent(system_state: AgentState) -> AgentState:
    """
    Risk Analysis Agent (Web-Search Enhanced) entry point.

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
        startup_raw_data             (dict, optional)

    Environment variables:
        agent2_llm     — Groq API key
        TAVILY_API_KEY — Tavily Search API key

    Writes to `system_state`:
        risk_analysis_report  (str)   — full structured 9-section report
        risk_search_results   (str)   — raw search evidence (audit trail)
        risk_signals          (dict)  — extracted structured risk signals

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
    print(f"  Risk Analysis Agent — {startup_name}")
    print(f"{'='*60}")

    # Step 1 — Infer keywords
    print("\n[1/3] Inferring industry & product keywords …")
    industry, product = _infer_keywords(system_state)
    print(f"      industry → {industry!r}")
    print(f"      product  → {product!r}")

    # Step 2 — Web searches
    print(f"\n[2/3] Running {len(QUERY_TEMPLATES)} risk-focused searches via Tavily …")
    search_results = await run_risk_searches(industry, product, startup_name, tavily_api_key)
    print(f"      → {len(search_results):,} chars of search evidence collected")

    # Step 3 — LLM synthesis
    print("\n[3/3] Generating risk analysis report with Groq LLM …")
    startup_context = build_startup_context(system_state)
    
    # Log which prior agent outputs are present
    available = []
    if system_state.get("startup_summary"):              available.append("startup_summary")
    if system_state.get("market_research_report"):       available.append("market_research")
    if system_state.get("competitor_analysis_report"):   available.append("competitor_analysis")
    if system_state.get("financial_estimation_report"):  available.append("financial_estimation")
    raw = system_state.get("startup_raw_data", {})
    if raw.get("pitch_text"):    available.append("pitch_deck_ocr")
    if raw.get("website_text"):  available.append("website_text")
    if raw.get("linkedin_text"): available.append("linkedin_profile")
    print(f"      → Prior outputs: {', '.join(available) if available else 'none'}")

    report = await generate_risk_report(startup_context, search_results, groq_api_key)
    print(f"      → {len(report):,} chars generated")

    # Extract structured signals
    signals = extract_risk_signals(report)
    metrics = extract_risk_metrics(report)

    # Store results
    system_state["risk_analysis_report"] = report
    system_state["risk_search_results"]  = search_results
    system_state["risk_signals"]         = signals
    system_state["risk_metrics"] = metrics

    print(f"\n[Agent] Done.")
    print(f"        Overall Risk Score : {signals['overall_score']}/10 — {signals['overall_level']}")
    print(f"        Red Flags Found    : {signals['red_flags_count']}")
    print(f"        Risk Levels        :")
    for cat, level in signals["risk_levels"].items():
        print(f"          {cat:<12} → {level}")
    print(f"        Stored → system_state['risk_analysis_report']")
    print(f"{'='*60}\n")

    return system_state

