import os
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

SEARCH_MAX_RESULTS = 6        # results per query
SEARCH_MAX_CHARS   = 1_200    # chars kept per search snippet
TOTAL_SEARCH_CAP   = 20_000   # hard cap on total search content sent to LLM


# ═════════════════════════════════════════════════════════════════════════════
# SEARCH QUERY TEMPLATES
# ═════════════════════════════════════════════════════════════════════════════

QUERY_TEMPLATES = [
    "{industry} startups 2024 2025",
    "{product} competitors",
    "companies similar to {startup_name}",
    "{industry} market leaders",
    "{product} platforms alternatives",
    "{industry} top funded startups",
    "{startup_name} competitors analysis",
    "{product} direct indirect competitors venture capital",
]


# ═════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT
# ═════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = textwrap.dedent("""\
    You are a senior competitive intelligence analyst inside a venture capital
    startup evaluation system.

    You will receive:
      1. STARTUP CONTEXT — name, summary, website, pitch deck, founder data.
      2. WEB SEARCH RESULTS — real, recently retrieved data about competitors,
         market leaders, and companies in the same space.

    Your task is to produce a structured COMPETITOR ANALYSIS REPORT.

    ─────────────────────────────────────────────────────────────
    REPORT STRUCTURE (use these exact section headers):
    ─────────────────────────────────────────────────────────────

    ## 1. Industry & Competitive Category
    State the industry, product category, and competitive environment.
    Explain how companies in this space typically compete (on price,
    features, data, distribution, brand, etc.).

    ## 2. Direct Competitors
    List 5–8 companies offering similar products or targeting the same customers.
    For each include:
      - Company name & one-line description
      - Target customers
      - Estimated scale or funding (state "not publicly available" if unknown)

    ## 3. Indirect Competitors
    Identify 3–5 companies solving the same problem via different approaches
    or technologies. Explain how each is an alternative to this startup.

    ## 4. Product & Feature Comparison
    Compare the startup against its top 3–4 direct competitors across:
      - Core product features
      - Technology approach
      - Pricing model (if known)
      - Target customer segment
      - Geographic focus
    Use a short prose comparison or a structured list — whichever is clearer.
    Explicitly call out competitive advantages and disadvantages for the startup.

    ## 5. Funding & Market Position
    Summarize the funding landscape among competitors:
      - Who are the best-funded players?
      - Are competitors at seed, Series A/B/C, or public?
      - Does the startup face well-capitalised incumbents?
    If specific funding figures are unavailable from search results, state that
    clearly rather than estimating.

    ## 6. Competitive Advantages
    Identify 3–5 potential advantages the startup may hold, such as:
      - Proprietary technology or unique data
      - Better pricing or go-to-market model
      - Underserved customer segment or geography
      - Distribution or partnership advantage
      - Founder domain expertise
    Be specific and tie each advantage to evidence from the startup context.

    ## 7. Competition Intensity Score
    Assign a score from 1–10:
      1–3  → Low    (few players, emerging market)
      4–6  → Medium (several competitors, room for differentiation)
      7–10 → High   (many well-funded competitors, commoditising market)

    Format as:
        Competition Intensity Score: X/10 — [Low | Medium | High]
    Follow with 3–5 sentences explaining your reasoning.

    ## 8. Strategic Competitive Insights
    Provide 3–5 bullet-point insights for investors, such as:
      - Whether the market is fragmented or dominated by large players
      - Whether the startup is entering a crowded or whitespace category
      - Clear market gaps or underserved niches
      - Recommended competitive positioning or differentiation strategy
      - Biggest competitive threat to watch

    ─────────────────────────────────────────────────────────────
    RULES:
    ─────────────────────────────────────────────────────────────
    - Only name companies found in the web search results or clearly stated
      in the startup context. Do NOT invent companies or funding numbers.
    - If a competitor's funding is not in the search results, write
      "Funding not publicly available."
    - Cite data sources by name where possible (e.g., "per Crunchbase",
      "per TechCrunch report").
    - Maintain a neutral, evidence-based VC analyst tone.
    - Total report: 600–1000 words, clearly structured with the headers above.
─────────────────────────────────────────────────────────────
9. STRUCTURED COMPETITION METRICS (FOR VISUALIZATION)
─────────────────────────────────────────────────────────────

After completing the report, output a JSON block called:

COMPETITION_METRICS

Format exactly as:

COMPETITION_METRICS:
{
  "competition_intensity_score": number (1-10),
  "market_fragmentation_score": number (0-10),
  "startup_differentiation_score": number (0-10),
  "incumbent_strength_score": number (0-10),
  "top_competitors": [
    {"name": "Competitor1", "strength": number (0-10)},
    {"name": "Competitor2", "strength": number (0-10)},
    {"name": "Competitor3", "strength": number (0-10)}
  ]
}

Rules:
- Scores must be realistic and grounded in the analysis.
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
    Priority: startup_summary → website_text → startup_name fallback.
    """
    name    = system_state.get("startup_name", "startup")
    summary = system_state.get("startup_summary", "")
    raw     = system_state.get("startup_raw_data", {})
    seed    = summary[:400] or (raw.get("website_text") or "")[:400] or name

    seed_lower = seed.lower()

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
        "retail":        ("retail technology",                  "retail SaaS platform"),
        "insurance":     ("insurtech",                          "insurance technology platform"),
    }

    for keyword, (ind, prod) in keyword_map.items():
        if keyword in seed_lower:
            return ind, prod

    return f"{name} technology software", f"{name} platform"


# ═════════════════════════════════════════════════════════════════════════════
# WEB SEARCH MODULE  (Tavily)
# ═════════════════════════════════════════════════════════════════════════════

async def run_competitor_searches(
    industry: str,
    product: str,
    startup_name: str,
    tavily_api_key: str,
) -> str:
    """
    Execute all QUERY_TEMPLATES, deduplicate results, and return a single
    formatted evidence string ready for the LLM prompt.
    """
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
# STARTUP CONTEXT BUILDER
# ═════════════════════════════════════════════════════════════════════════════

def build_startup_context(system_state: dict) -> str:
    """Merge all available startup signals into one context block."""
    parts: list[str] = []

    name    = system_state.get("startup_name",    "Unknown Startup")
    stage   = system_state.get("funding_stage")
    website = system_state.get("startup_website")

    parts.append(f"STARTUP NAME: {name}")
    if stage:
        parts.append(f"FUNDING STAGE: {stage}")
    if website:
        parts.append(f"WEBSITE: {website}")

    # Highest-signal source: pre-generated startup summary
    summary = system_state.get("startup_summary")
    if summary:
        parts.append(f"\n=== STARTUP INTELLIGENCE SUMMARY ===\n{summary}")

    # Market research report from previous agent (if available)
    market_report = system_state.get("market_research_report")
    if market_report:
        parts.append(
            f"\n=== MARKET RESEARCH REPORT (from Market Research Agent) ===\n"
            f"{_truncate(market_report, 3000)}"
        )

    # Raw scraped / OCR data
    raw = system_state.get("startup_raw_data", {})
    if raw.get("website_text"):
        parts.append(
            f"\n=== WEBSITE CONTENT ===\n{_truncate(raw['website_text'], 2000)}"
        )
    if raw.get("pitch_text"):
        parts.append(
            f"\n=== PITCH DECK CONTENT ===\n{_truncate(raw['pitch_text'], 2000)}"
        )
    if raw.get("linkedin_text"):
        parts.append(
            f"\n=== FOUNDER LINKEDIN PROFILE ===\n{_truncate(raw['linkedin_text'], 1000)}"
        )

    return "\n\n".join(parts)


# ═════════════════════════════════════════════════════════════════════════════
# LLM SYNTHESIS
# ═════════════════════════════════════════════════════════════════════════════

async def generate_competitor_report(
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
        "Produce a full Competitor Analysis Report using the startup context "
        "and web search evidence provided below.\n\n"
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
# COMPETITION INTENSITY EXTRACTOR
# ═════════════════════════════════════════════════════════════════════════════

def extract_competition_score(report: str) -> dict:
    """
    Parse the Competition Intensity Score and level from the report text.
    Returns {"score": int | None, "level": str}.
    """
    import re
    score = None
    level = "UNKNOWN"

    # Match patterns like "Score: 7/10" or "Score: 7/10 — High"
    match = re.search(
        r"Competition\s+Intensity\s+Score[:\s]+(\d{1,2})\s*/\s*10",
        report,
        re.IGNORECASE,
    )
    if match:
        score = int(match.group(1))
        if score <= 3:
            level = "LOW"
        elif score <= 6:
            level = "MEDIUM"
        else:
            level = "HIGH"

    return {"score": score, "level": level}
# ═════════════════════════════════════════════════════════════════════════════
# extract_competition_metrics
# ═════════════════════════════════════════════════════════════════════════════


def extract_competition_metrics(report):

    import json
    import re

    match = re.search(
        r"COMPETITION_METRICS:\s*(\{[\s\S]*?\})",
        report
    )

    if not match:
        return {}

    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        print("⚠️ Invalid COMPETITION_METRICS JSON")
        return {}

# ═════════════════════════════════════════════════════════════════════════════
# MAIN AGENT ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

async def run_competitor_analysis_agent(system_state: AgentState) -> AgentState:
    """
    Competitor Analysis Agent entry point.

    Reads from `system_state`:
        startup_name            (str, required)
        funding_stage           (str, optional)
        startup_website         (str, optional)
        startup_summary         (str, optional)  ← Startup Data Agent output
        market_research_report  (str, optional)  ← Market Research Agent output
        startup_raw_data        (dict, optional)

    Environment variables:
        agent2_llm     — Groq API key
        TAVILY_API_KEY — Tavily Search API key

    Writes to `system_state`:
        competitor_analysis_report  (str)   — full structured report
        competitor_search_results   (str)   — raw evidence (audit trail)
        competition_intensity       (dict)  — {"score": int, "level": str}

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
    print(f"  Competitor Analysis Agent — {startup_name}")
    print(f"{'='*60}")

    # Step 1 — Infer search keywords
    print("\n[1/3] Inferring industry & product keywords …")
    industry, product = _infer_industry_product(system_state)
    print(f"      industry → {industry!r}")
    print(f"      product  → {product!r}")

    # Step 2 — Web searches
    print(f"\n[2/3] Running {len(QUERY_TEMPLATES)} competitor searches via Tavily …")
    search_results = await run_competitor_searches(
        industry, product, startup_name, tavily_api_key
    )
    print(f"      → {len(search_results):,} chars of search evidence collected")

    # Step 3 — LLM synthesis
    print("\n[3/3] Synthesizing competitor analysis report with Groq LLM …")
    startup_context = build_startup_context(system_state)
    report          = await generate_competitor_report(
        startup_context, search_results, groq_api_key
    )
    print(f"      → {len(report):,} chars generated")

    intensity = extract_competition_score(report)
    metrics = extract_competition_metrics(report)

    # Store results
    system_state["competitor_analysis_report"] = report
    system_state["competitor_search_results"]  = search_results
    system_state["competition_intensity"]      = intensity
    system_state["competition_metrics"] = metrics

    print(f"\n[Agent] Done.")
    print(f"        Competition Intensity : {intensity['score']}/10 — {intensity['level']}")
    print(f"        Stored → system_state['competitor_analysis_report']")
    print(f"{'='*60}\n")

    return system_state
