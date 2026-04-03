from fastapi import APIRouter, Depends

from backend.core.dependencies import get_startup_id, get_pipeline_result
from backend.utils.extract import extract_section, parse_markdown_to_insights, parse_metrics

router = APIRouter(prefix="/api", tags=["Market Research"])


@router.get("/market/industry")
def get_market_industry(startup_id: str = Depends(get_startup_id)):
    res = get_pipeline_result(startup_id)
    report = res.get("market_research_report", "")
    return {"industry_market_category": extract_section(report, "## 1. Industry & Market Category")}


@router.get("/market/market-size")
def get_market_size(startup_id: str = Depends(get_startup_id)):
    res = get_pipeline_result(startup_id)
    report = res.get("market_research_report", "")
    return {"market_size_estimates": extract_section(report, "## 2. Market Size Estimates")}


@router.get("/market/growth")
def get_market_growth(startup_id: str = Depends(get_startup_id)):
    res = get_pipeline_result(startup_id)
    report = res.get("market_research_report", "")
    return {"market_growth_trends": extract_section(report, "## 3. Market Growth Trends")}


@router.get("/market/customer-segments")
def get_customer_segments(startup_id: str = Depends(get_startup_id)):
    res = get_pipeline_result(startup_id)
    report = res.get("market_research_report", "")
    return {"customer_segments": extract_section(report, "## 4. Customer Segments")}


@router.get("/market/competition")
def get_competition(startup_id: str = Depends(get_startup_id)):
    res = get_pipeline_result(startup_id)
    report = res.get("market_research_report", "")
    return {"competitive_landscape": extract_section(report, "## 5. Competitive Landscape")}


@router.get("/market/demand")
def get_demand(startup_id: str = Depends(get_startup_id)):
    res = get_pipeline_result(startup_id)
    report = res.get("market_research_report", "")
    return {"demand_signals": extract_section(report, "## 6. Demand Signals")}


@router.get("/market/barriers")
def get_barriers(startup_id: str = Depends(get_startup_id)):
    res = get_pipeline_result(startup_id)
    report = res.get("market_research_report", "")
    return {"market_entry_barriers": extract_section(report, "## 7. Market Entry Barriers")}


@router.get("/market/opportunity")
def get_opportunity(startup_id: str = Depends(get_startup_id)):
    res = get_pipeline_result(startup_id)
    return {"market_opportunity_flag": res.get("market_opportunity_flag")}


@router.get("/market/metrics")
def get_market_metrics(startup_id: str = Depends(get_startup_id)):
    res = get_pipeline_result(startup_id)
    return res.get("market_metrics", {})


# ─── Combined for frontend ────────────────────────────────────

@router.get("/market-analysis")
def get_market_analysis_combined(startup_id: str = Depends(get_startup_id)):
    res = get_pipeline_result(startup_id)
    return {
        "status": "completed",
        "market": {
            "opportunity_flag": res.get("market_opportunity_flag"),
            "metrics": parse_metrics(res.get("market_metrics")),
            "analysis": parse_markdown_to_insights(res.get("market_research_report", ""))
        }
    }
