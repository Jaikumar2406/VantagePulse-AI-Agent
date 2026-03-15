import os
import textwrap
from typing import Optional
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from tavily import TavilyClient
from state.state import AgentState
load_dotenv()


# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════

GROQ_MODEL         = "llama-3.3-70b-versatile"
MAX_TOKENS         = 4096
TEMPERATURE        = 0

SEARCH_MAX_RESULTS = 5        # results per query
SEARCH_MAX_CHARS   = 1_200    # chars kept per search snippet
TOTAL_SEARCH_CAP   = 20_000   # hard cap on total search content sent to LLM


# ═════════════════════════════════════════════════════════════════════════════
# SEARCH QUERY TEMPLATES
# ═════════════════════════════════════════════════════════════════════════════
# {industry} and {product} are filled at runtime from the startup context.

QUERY_TEMPLATES = [
    "{industry} market size 2025",
    "{industry} CAGR growth rate forecast",
    "{industry} market trends 2025",
    "{product} competitors landscape",
    "{industry} startup adoption trends",
    "{industry} funding trends venture capital 2024 2025",
    "{product} market entry barriers regulations",
    "{industry} TAM SAM market opportunity report",
]


# ═════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT
# ═════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = textwrap.dedent("""\
    You are a senior market research analyst inside a venture capital startup
    intelligence system. You have been given:

      1. A STARTUP CONTEXT section describing the startup.
      2. A WEB SEARCH RESULTS section containing real, recently retrieved data
         from industry reports, news articles, and funding databases.

    Your task is to produce a comprehensive MARKET RESEARCH REPORT grounded
    in the web search evidence provided.

    ─────────────────────────────────────────────────────────────
    REPORT STRUCTURE (use these exact section headers):
    ─────────────────────────────────────────────────────────────

    ## 1. Industry & Market Category
    Identify the industry, sub-sector, and product category. Describe the
    broader market ecosystem and where this startup fits.

    ## 2. Market Size Estimates
    Using the web search data, estimate:
      - TAM (Total Addressable Market): Global market if fully captured.
      - SAM (Serviceable Available Market): Segment the startup can serve.
      - SOM (Serviceable Obtainable Market): Realistic 3–5 year capture.
    Cite specific numbers from the search results where available. Show
    assumptions clearly when exact figures are absent.
    Express all values in USD (e.g., "$4.2B", "$800M", "$45M").

    ## 3. Market Growth Trends
    State the estimated CAGR from search results. List key growth drivers,
    technology shifts, and regulatory changes.
    Classify growth pace as one of:
      - Rapid Growth (>15% CAGR)
      - Moderate Growth (7–15% CAGR)
      - Slow Growth (<7% CAGR)

    ## 4. Customer Segments
    Identify primary and secondary customer groups (enterprise, SMB, consumer,
    developer, government, etc.). For each segment estimate:
      - Approximate number of potential buyers
      - Typical annual spend / willingness to pay
      - Key buying criteria

    ## 5. Competitive Landscape
    List the top 5–8 direct and indirect competitors. For each include:
      - Company name and brief description
      - Approximate scale, funding, or market share
      - How they compete with or compare to this startup
    Characterize market structure: fragmented / concentrated / emerging.

    ## 6. Demand Signals
    Using the web search results, identify:
      - Recent funding rounds in the sector (with amounts where available)
      - New product launches or major partnerships
      - Industry adoption trends
      - Regulatory initiatives or government mandates
      - Media or analyst attention trends

    ## 7. Market Entry Barriers
    Evaluate difficulty of entry across:
      - Regulatory / compliance requirements
      - Capital intensity
      - Technology complexity or IP moats
      - Switching costs and network effects held by incumbents
      - Sales cycle and procurement friction
    Rate overall barrier height: Low / Medium / High

    ## 8. Venture-Scale Opportunity Assessment
    Synthesize findings into a final verdict:
      - Is the TAM large enough for venture-scale returns (typically >$1B)?
      - Does the growth rate support a compelling VC narrative?
      - What is the single biggest market risk?
    End with exactly one of:
        STRONG OPPORTUNITY
        MODERATE OPPORTUNITY
        WEAK OPPORTUNITY
    Then provide a 2–3 sentence investment thesis for the market.

    ─────────────────────────────────────────────────────────────
    RULES:
    ─────────────────────────────────────────────────────────────
    - Ground every major claim in the web search results provided.
    - When citing a figure, note its source (e.g., "per Grand View Research").
    - Never fabricate specific statistics not found in the search data.
      State assumptions clearly when estimating.
    - Maintain a neutral, evidence-based VC analyst tone.
    - Total report: 700–1200 words, clearly structured with the headers above.
    ─────────────────────────────────────────────────────────────
9. STRUCTURED MARKET METRICS (FOR VISUALIZATION)
─────────────────────────────────────────────────────────────

After completing the report, output a JSON block called:

MARKET_METRICS

This JSON will be used by the system to generate analytics charts.

Format exactly as:

MARKET_METRICS:
{
  "tam_usd_billion": number | null,
  "sam_usd_billion": number | null,
  "som_usd_billion": number | null,
  "cagr_percent": number | null,
  "growth_classification": "Rapid | Moderate | Slow",
  "market_structure": "Fragmented | Concentrated | Emerging",
  "entry_barrier_level": "Low | Medium | High",
  "opportunity_rating": "STRONG | MODERATE | WEAK"
}

Rules:
- Use null if exact numbers cannot be determined.
- Values must be grounded in the search evidence or clearly stated assumptions.
- JSON must be valid and parseable.
""")


# ═════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + " … [truncated]"


def _infer_industry_product(system_state: dict) -> tuple[str, str]:
    """
    Derive search keyword strings from available startup context.
    Checks startup_summary → website_text → startup_name (fallback).
    """
    name    = system_state.get("startup_name", "startup")
    summary = system_state.get("startup_summary", "")
    raw     = system_state.get("startup_raw_data", {})
    seed    = summary[:400] or (raw.get("website_text") or "")[:400] or name

    seed_lower = seed.lower()

    # Keyword → (industry_phrase, product_phrase)
    keyword_map = {
        "agri":          ("agri-tech precision farming",        "precision farming IoT platform"),
        "farm":          ("agri-tech precision farming",        "precision farming IoT platform"),
        "health":        ("digital health technology",          "health tech platform"),
        "medtech":       ("medical technology",                 "medical device software"),
        "fintech":       ("fintech financial technology",       "fintech payment platform"),
        "edtech":        ("edtech e-learning",                  "online education platform"),
        "climate":       ("climate tech clean energy",          "climate technology software"),
        "logistics":     ("logistics supply chain tech",        "supply chain SaaS"),
        "cybersecurity": ("cybersecurity",                      "cloud security software"),
        "ecommerce":     ("e-commerce technology",              "online retail platform"),
        "proptech":      ("proptech real estate technology",    "property management software"),
        "saas":          ("SaaS cloud software",                "cloud SaaS platform"),
        "ai":            ("artificial intelligence software",   "AI SaaS platform"),
        "data":          ("data analytics platform",            "data intelligence SaaS"),
        "hr":            ("HR technology workforce management", "HR SaaS platform"),
        "legal":         ("legaltech",                          "legal technology software"),
    }

    for keyword, (ind, prod) in keyword_map.items():
        if keyword in seed_lower:
            return ind, prod

    # Generic fallback — use startup name as the search seed
    return f"{name} software technology", f"{name} platform"


# ═════════════════════════════════════════════════════════════════════════════
# WEB SEARCH MODULE  (Tavily)
# ═════════════════════════════════════════════════════════════════════════════

async def run_web_searches(
    industry: str,
    product: str,
    startup_name: str,
    tavily_api_key: str,
) -> str:
    """
    Execute all market research queries and return deduplicated evidence.
    """
    client    = TavilyClient(api_key=tavily_api_key)
    collected : list[str] = []
    seen_urls : set[str]  = set()

    for query in QUERY_TEMPLATES:
        formatted = query.format(industry=industry, product=product)
        print(f"    🔍  {formatted}")

        try:
            resp    = await client.async_search(
                query=formatted,
                search_depth="basic",
                max_results=SEARCH_MAX_RESULTS,
            )
            results = resp.get("results", [])

            block = [f"\n### Search: {formatted}"]
            for r in results:
                url     = r.get("url", "")
                title   = r.get("title", "No title")
                content = _truncate(r.get("content", ""), SEARCH_MAX_CHARS)
                if url in seen_urls or not content.strip():
                    continue
                seen_urls.add(url)
                block.append(f"**{title}**\n{content}\nSource: {url}\n")

            collected.append("\n".join(block))

        except Exception as exc:
            collected.append(f"\n### Search: {formatted}\n[Search error: {exc}]")

    combined = "\n".join(collected)
    return _truncate(combined, TOTAL_SEARCH_CAP)


# ═════════════════════════════════════════════════════════════════════════════
# STARTUP CONTEXT BUILDER
# ═════════════════════════════════════════════════════════════════════════════

def build_startup_context(system_state: dict) -> str:
    """Merge all startup signals from system_state into one context block."""
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

    raw = system_state.get("startup_raw_data", {})
    if raw.get("website_text"):
        parts.append(
            f"\n=== WEBSITE CONTENT ===\n{_truncate(raw['website_text'], 3000)}"
        )
    if raw.get("pitch_text"):
        parts.append(
            f"\n=== PITCH DECK CONTENT ===\n{_truncate(raw['pitch_text'], 3000)}"
        )
    if raw.get("linkedin_text"):
        parts.append(
            f"\n=== FOUNDER LINKEDIN PROFILE ===\n{_truncate(raw['linkedin_text'], 1500)}"
        )

    return "\n\n".join(parts)


# ═════════════════════════════════════════════════════════════════════════════
# LLM SYNTHESIS
# ═════════════════════════════════════════════════════════════════════════════

async def generate_market_report(
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
        "Produce a full Market Research Report using the startup context and "
        "web search evidence below.\n\n"
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
# OPPORTUNITY FLAG EXTRACTOR
# ═════════════════════════════════════════════════════════════════════════════

def extract_opportunity_flag(report: str) -> str:
    upper = report.upper()
    if "STRONG OPPORTUNITY"   in upper:
        return "STRONG"
    if "MODERATE OPPORTUNITY" in upper:
        return "MODERATE"
    if "WEAK OPPORTUNITY"     in upper:
        return "WEAK"
    return "UNKNOWN"

# ═════════════════════════════════════════════════════════════════════════════
# extract_market_metrics
# ═════════════════════════════════════════════════════════════════════════════

def extract_market_metrics(report):

    import json
    import re

    match = re.search(
        r"MARKET_METRICS:\s*(\{[\s\S]*?\})",
        report
    )

    if not match:
        return {}

    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        print("⚠️ Invalid MARKET_METRICS JSON")
        return {}
    

# ═════════════════════════════════════════════════════════════════════════════
# MAIN AGENT ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

async def run_market_research_agent(system_state: AgentState) -> AgentState:
    """
    Market Research Agent (Web-Search Enhanced) entry point.

    Reads from `system_state`:
        startup_name          (str, required)
        funding_stage         (str, optional)
        startup_website       (str, optional)
        startup_summary       (str, optional)  ← from Startup Data Agent
        startup_raw_data      (dict, optional) ← website / pitch / linkedin text

    Environment variables:
        agent2_llm     — Groq API key
        TAVILY_API_KEY — Tavily Search API key

    Writes to `system_state`:
        market_research_report  (str)  — full structured market report
        market_search_results   (str)  — raw search evidence (audit trail)
        market_opportunity_flag (str)  — "STRONG" | "MODERATE" | "WEAK" | "UNKNOWN"

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
    print(f"  Market Research Agent — {startup_name}")
    print(f"{'='*60}")

    # Step 1 — Infer search keywords
    print("\n[1/3] Inferring industry & product keywords …")
    industry, product = _infer_industry_product(system_state)
    print(f"      industry → {industry!r}")
    print(f"      product  → {product!r}")

    # Step 2 — Web searches
    print(f"\n[2/3] Running {len(QUERY_TEMPLATES)} web searches via Tavily …")
    search_results = await run_web_searches(industry, product, startup_name, tavily_api_key)
    print(f"      → {len(search_results):,} chars of search evidence collected")

    # Step 3 — LLM synthesis
    print("\n[3/3] Synthesizing report with Groq LLM …")
    startup_context = build_startup_context(system_state)
    report          = await generate_market_report(startup_context, search_results, groq_api_key)
    print(f"      → {len(report):,} chars generated")

    flag = extract_opportunity_flag(report)
    metrics = extract_market_metrics(report)

    # Store results
    system_state["market_research_report"]  = report
    system_state["market_search_results"]   = search_results
    system_state["market_opportunity_flag"] = flag
    system_state["market_metrics"] = metrics

    print(f"\n[Agent] Done.  Opportunity flag : {flag}")
    print(f"        Stored  → system_state['market_research_report']")
    print(f"{'='*60}\n")

    return system_state