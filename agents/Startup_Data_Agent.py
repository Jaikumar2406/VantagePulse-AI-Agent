import os
import re
import asyncio
import textwrap
from typing import Optional
from state.state import AgentState

# ── Playwright (async) ────────────────────────────────────────────────────────
from playwright.async_api import async_playwright, TimeoutError as PWTimeout

# ── OCR-based PDF parsing ─────────────────────────────────────────────────────
from pdf2image import convert_from_path
import pytesseract

# ── Groq LLM (LangChain wrapper) ─────────────────────────────────────────────
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()


# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════

GROQ_MODEL         = "llama-3.3-70b-versatile"
MAX_TOKENS         = 2048
PAGE_LOAD_TIMEOUT  = 20_000            # ms — Playwright navigation timeout
MAX_SCRAPED_CHARS  = 8_000             # total website text cap
MAX_PDF_CHARS      = 10_000            # total pitch-deck OCR text cap
MAX_LINKEDIN_CHARS = 4_000             # LinkedIn profile text cap
OCR_DPI            = 200               # DPI for pdf2image rendering

PAGES_TO_SCRAPE = [
    "",            # homepage
    "about",
    "about-us",
    "product",
    "features",
    "pricing",
    "team",
    "mission",
]

# Playwright browser launch args (headless, sandbox-safe)
BROWSER_ARGS = ["--no-sandbox", "--disable-setuid-sandbox"]


# ═════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _clean(text: str) -> str:
    """Collapse whitespace and strip control characters."""
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n... [truncated — {len(text) - max_chars} chars omitted]"


def _normalize_url(url: str) -> str:
    url = url.strip().rstrip("/")
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    return url


def _run(coro):
    """
    Run an async coroutine from sync code.
    Handles both normal environments and nested event loops (e.g. Jupyter).
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
    else:
        return asyncio.run(coro)


# ═════════════════════════════════════════════════════════════════════════════
# SOURCE 1 — WEBSITE SCRAPER  (Playwright)
# ═════════════════════════════════════════════════════════════════════════════

async def _scrape_page_async(browser, url: str, slug: str) -> str:
    """Open a single URL in Playwright and return cleaned visible text."""
    page = await browser.new_page()
    try:
        await page.goto(url, timeout=PAGE_LOAD_TIMEOUT, wait_until="domcontentloaded")

        # Give JS-heavy SPAs time to hydrate
        await page.wait_for_timeout(1_500)

        # Remove noisy structural elements
        await page.evaluate("""() => {
            const selectors = [
                'script', 'style', 'nav', 'footer', 'header',
                'aside', 'noscript', '[role="banner"]', '[role="navigation"]'
            ];
            selectors.forEach(sel => {
                document.querySelectorAll(sel).forEach(el => el.remove());
            });
        }""")

        text = await page.inner_text("body")
        cleaned = _clean(text)

        if len(cleaned) > 200:
            return f"[Page: /{slug}]\n{cleaned}"
        return ""                                   # skip near-empty pages

    except PWTimeout:
        return f"[Page: /{slug} — timed out after {PAGE_LOAD_TIMEOUT}ms]"
    except Exception as exc:
        return f"[Page: /{slug} — error: {exc}]"
    finally:
        await page.close()


async def _scrape_website_async(startup_website: str) -> str:
    base = _normalize_url(startup_website)
    collected: list[str] = []

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True, args=BROWSER_ARGS)
        try:
            for slug in PAGES_TO_SCRAPE:
                url = f"{base}/{slug}" if slug else base
                result = await _scrape_page_async(browser, url, slug or "")
                if result:
                    collected.append(result)
                await asyncio.sleep(0.6)            # polite crawl delay
        finally:
            await browser.close()

    combined = "\n\n".join(collected)
    return _truncate(combined, MAX_SCRAPED_CHARS)


def scrape_website(startup_website: str) -> str:
    """Sync entry point — scrape startup website pages using Playwright."""
    return _run(_scrape_website_async(startup_website))


# ═════════════════════════════════════════════════════════════════════════════
# SOURCE 2 — PITCH DECK PARSER  (pdf2image + pytesseract OCR)
# ═════════════════════════════════════════════════════════════════════════════

def parse_pitch_deck(pdf_path: str) -> str:
    """
    Convert each PDF slide to a rasterised image (pdf2image / poppler),
    then extract text via Tesseract OCR (pytesseract).

    Works on both text-based and fully image / design-heavy pitch decks.

    System requirements:
        apt-get install -y tesseract-ocr poppler-utils   # Linux
        brew install tesseract poppler                   # macOS
    """
    if not os.path.isfile(pdf_path):
        return f"[Pitch deck not found at path: {pdf_path}]"

    pages_text: list[str] = []

    try:
        images = convert_from_path(pdf_path, dpi=OCR_DPI)
    except Exception as exc:
        return f"[pdf2image conversion error: {exc}]"

    for i, img in enumerate(images, 1):
        try:
            raw  = pytesseract.image_to_string(img, lang="eng")
            text = _clean(raw)
            if text:
                pages_text.append(f"[Slide {i}]\n{text}")
        except Exception as exc:
            pages_text.append(f"[Slide {i} — OCR error: {exc}]")

    if not pages_text:
        return "[Pitch deck parsed but no text was extracted via OCR.]"

    combined = "\n\n".join(pages_text)
    return _truncate(combined, MAX_PDF_CHARS)


# ═════════════════════════════════════════════════════════════════════════════
# SOURCE 3 — LINKEDIN PROFILE ANALYZER  (Playwright)
# ═════════════════════════════════════════════════════════════════════════════

async def _analyze_linkedin_async(linkedin_url: str) -> str:
    """
    Load the LinkedIn public profile page via Playwright.
    LinkedIn requires JavaScript rendering — Playwright handles that natively.

    Note: LinkedIn may still gate full profile data behind a login wall.
    For production, replace with the LinkedIn API or a compliant enrichment
    provider (e.g. Proxycurl, Apollo).
    """
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True, args=BROWSER_ARGS)
        page    = await browser.new_page()
        try:
            # Mimic a real browser fingerprint to reduce bot detection
            await page.set_extra_http_headers({
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                )
            })
            await page.set_viewport_size({"width": 1280, "height": 800})

            response = await page.goto(
                linkedin_url.strip(),
                timeout=PAGE_LOAD_TIMEOUT,
                wait_until="domcontentloaded",
            )

            if response and response.status in (403, 429, 999):
                return (
                    f"[LinkedIn blocked automated access (HTTP {response.status}). "
                    "Integrate the LinkedIn API or a compliant data provider for "
                    "reliable founder data.]"
                )

            # Allow dynamic content to settle
            await page.wait_for_timeout(2_500)

            # Strip noise
            await page.evaluate("""() => {
                ['script', 'style', 'nav', 'footer', 'header'].forEach(sel => {
                    document.querySelectorAll(sel).forEach(el => el.remove());
                });
            }""")

            text = await page.inner_text("body")
            return _truncate(_clean(text), MAX_LINKEDIN_CHARS)

        except PWTimeout:
            return "[LinkedIn page timed out.]"
        except Exception as exc:
            return f"[LinkedIn fetch error: {exc}]"
        finally:
            await page.close()
            await browser.close()


def analyze_linkedin(linkedin_url: str) -> str:
    """Sync entry point — analyze founder LinkedIn profile using Playwright."""
    return _run(_analyze_linkedin_async(linkedin_url))


# ═════════════════════════════════════════════════════════════════════════════
# SOURCE ASSEMBLER
# ═════════════════════════════════════════════════════════════════════════════

def assemble_context(
    startup_name: str,
    funding_stage: Optional[str],
    website_text:  Optional[str],
    pitch_text:    Optional[str],
    linkedin_text: Optional[str],
) -> str:
    parts = [f"STARTUP NAME: {startup_name}"]
    if funding_stage:
        parts.append(f"FUNDING STAGE: {funding_stage}")
    if website_text:
        parts.append(f"\n=== WEBSITE CONTENT ===\n{website_text}")
    if pitch_text:
        parts.append(f"\n=== PITCH DECK CONTENT ===\n{pitch_text}")
    if linkedin_text:
        parts.append(f"\n=== FOUNDER LINKEDIN PROFILE ===\n{linkedin_text}")
    return "\n\n".join(parts)


# ═════════════════════════════════════════════════════════════════════════════
# GROQ LLM — SYNTHESIS  (LangChain-Groq)
# ═════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = textwrap.dedent("""
You are a professional startup intelligence analyst working inside a 
venture capital startup evaluation system.

Your task is to analyze raw startup data collected from:

1. Startup Website
2. Pitch Deck (OCR extracted text)
3. Founder LinkedIn Profile

Use the information to produce a structured STARTUP INTELLIGENCE SUMMARY
that helps downstream AI agents perform deeper analysis.

─────────────────────────────────────────────────────────────
CRITICAL RULES
─────────────────────────────────────────────────────────────
- Only use information present in the input context.
- Do NOT fabricate missing data.
- If information is unavailable, explicitly state:
  "Not mentioned in available data."
- Be factual, concise, and analytical.
- Maintain a professional venture-capital analyst tone.

─────────────────────────────────────────────────────────────
REPORT STRUCTURE
─────────────────────────────────────────────────────────────

## 1. Product / Service
Describe what the startup builds or offers.

## 2. Industry & Market
Identify the sector, industry category, and target market.

## 3. Business Model
Explain how the startup generates revenue (SaaS, marketplace,
transaction fees, subscription, hardware, etc.).

## 4. Problem & Solution
Explain the problem the startup solves and how its product
addresses that problem.

## 5. Traction & Growth Signals
Identify any signals of traction such as:
- user adoption
- partnerships
- pilots
- customer testimonials
- revenue signals

If none exist, state:
"Traction signals not mentioned."

## 6. Founder Background
Summarize founder experience including:
- past companies
- technical or industry expertise
- notable achievements

## 7. Competitors & Differentiation
Identify potential competitors and explain how the startup
differentiates itself.

## 8. Funding Stage & Use of Funds
Identify the current funding stage and possible priorities
for capital usage.

## 9. Risks & Information Gaps
Highlight:
- missing critical data
- unclear business model
- operational or market risks

─────────────────────────────────────────────────────────────
STRUCTURED STARTUP METRICS (FOR ANALYTICS)
─────────────────────────────────────────────────────────────

After the report, output a JSON block called:

STARTUP_METRICS

This will be used for dashboard visualization.

Format exactly as:

STARTUP_METRICS:
{
  "traction_score": number (0-10),
  "team_experience_score": number (0-10),
  "product_clarity_score": number (0-10),
  "market_clarity_score": number (0-10),
  "risk_score": number (0-10)
}

Rules:
- Scores must be between 0 and 10.
- Scores should reflect the analysis above.
- JSON must be valid and parseable.

─────────────────────────────────────────────────────────────
OUTPUT FORMAT
─────────────────────────────────────────────────────────────

1. Write the full STARTUP INTELLIGENCE SUMMARY.
2. After the report, output the STARTUP_METRICS JSON block.
3. Do not include any additional commentary.

Total report length: 400–600 words.
""")


async def synthesize_with_groq(context: str, groq_api_key: str) -> str:
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name=GROQ_MODEL,
        max_tokens=MAX_TOKENS,
        temperature=0,
    )
    response = await llm.ainvoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=context),
    ])
    return response.content.strip()

# ═════════════════════════════════════════════════════════════════════════════
# extract_growth_metrics
# ═════════════════════════════════════════════════════════════════════════════

def extract_startup_metrics(report):

    import json
    import re

    match = re.search(r"STARTUP_METRICS:\s*(\{.*\})", report, re.DOTALL)

    if not match:
        return {}

    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        print("⚠️ Invalid JSON in STARTUP_METRICS")
        return {}
    
# ═════════════════════════════════════════════════════════════════════════════
# MAIN AGENT ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def run_startup_data_agent(system_state: AgentState) -> AgentState:
    """
    Main agent function.

    Reads from `system_state`:
        startup_name      (str, required)
        startup_website   (str, optional)
        pitch_deck_pdf    (str, optional) — local file path
        founder_linkedin  (str, optional) — public profile URL
        funding_stage     (str, optional)

    API key is read from the environment variable: agent2_llm

    Writes to `system_state`:
        startup_summary   (str)  — enriched intelligence summary
        startup_raw_data  (dict) — intermediate scraped / parsed data

    Returns the updated system_state.
    """
    startup_name  = system_state.get("startup_name", "Unknown Startup")
    website       = system_state.get("startup_website")
    pitch_pdf     = system_state.get("pitch_deck_pdf")
    linkedin      = system_state.get("founder_linkedin")
    funding_stage = system_state.get("funding_stage")
    from agents.key_manager import get_groq_key
    groq_api_key = get_groq_key()

    if not groq_api_key:
        raise ValueError(
            "Groq API key not found. Set the 'agent2_llm' environment variable."
        )

    print(f"\n{'='*60}")
    print(f"  Startup Data Agent — {startup_name}")
    print(f"{'='*60}")

    website_text  = None
    pitch_text    = None
    linkedin_text = None

    # ── Source 1: Website (Playwright) ───────────────────────────────────────
    if website:
        print(f"[1/3] Scraping website with Playwright: {website}")
        website_text = scrape_website(website)
        print(f"      → {len(website_text):,} chars extracted")
    else:
        print("[1/3] Website not provided — skipping")

    # ── Source 2: Pitch Deck (pdf2image + pytesseract OCR) ───────────────────
    if pitch_pdf:
        print(f"[2/3] Parsing pitch deck with OCR (pdf2image + pytesseract): {pitch_pdf}")
        pitch_text = parse_pitch_deck(pitch_pdf)
        print(f"      → {len(pitch_text):,} chars extracted")
    else:
        print("[2/3] Pitch deck not provided — skipping")

    # ── Source 3: LinkedIn (Playwright) ─────────────────────────────────────
    if linkedin:
        print(f"[3/3] Analyzing LinkedIn with Playwright: {linkedin}")
        linkedin_text = analyze_linkedin(linkedin)
        print(f"      → {len(linkedin_text):,} chars extracted")
    else:
        print("[3/3] LinkedIn not provided — skipping")

    # ── Assemble context & synthesize ─────────────────────────────────────────
    context = assemble_context(
        startup_name=startup_name,
        funding_stage=funding_stage,
        website_text=website_text,
        pitch_text=pitch_text,
        linkedin_text=linkedin_text,
    )

    if len(context.strip()) < 100:
        summary = (
            "Insufficient data collected. Please provide at least one of: "
            "startup_website, pitch_deck_pdf, or founder_linkedin."
        )
    else:
        print("\n[LLM] Sending context to Groq for synthesis …")
        summary = _run(synthesize_with_groq(context, groq_api_key))
        print(f"[LLM] Summary generated ({len(summary):,} chars)")
    metrics = extract_startup_metrics(summary)
    # ── Write results back into system state ─────────────────────────────────
    system_state["startup_summary"] = summary
    system_state["startup_raw_data"] = {
        "website_text":  website_text,
        "pitch_text":    pitch_text,
        "linkedin_text": linkedin_text,
    }
    system_state["startup_metrics"]           = metrics

    print("\n[Agent] Done. Results stored in system_state['startup_summary']")
    print(f"{'='*60}\n")
    return system_state
