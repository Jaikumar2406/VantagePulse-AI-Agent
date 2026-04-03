from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from backend.core.database import db
from backend.core.dependencies import get_startup_id, get_pipeline_result
from backend.services.competitor.tasks import process_competitor_background
from backend.utils.extract import extract_section, parse_markdown_to_insights, parse_metrics

router = APIRouter(prefix="/api", tags=["Competitor Analysis"])


@router.post("/competitor/analyze")
async def analyze_competitor(
    background_tasks: BackgroundTasks,
    startup_id: str = Depends(get_startup_id),
):
    if startup_id not in db:
        raise HTTPException(status_code=404, detail="Startup ID not found")
    background_tasks.add_task(process_competitor_background, startup_id)
    return {"startup_id": startup_id, "agent": "competitor_analysis", "status": "processing"}


@router.get("/competitor/industry")
def get_competitor_industry(startup_id: str = Depends(get_startup_id)):
    res = get_pipeline_result(startup_id)
    return {"industry_category": extract_section(res.get("competitor_analysis_report", ""), "## 1. Industry & Competitive Category")}


@router.get("/competitor/direct")
def get_competitor_direct(startup_id: str = Depends(get_startup_id)):
    res = get_pipeline_result(startup_id)
    return {"direct_competitors": extract_section(res.get("competitor_analysis_report", ""), "## 2. Direct Competitors")}


@router.get("/competitor/indirect")
def get_competitor_indirect(startup_id: str = Depends(get_startup_id)):
    res = get_pipeline_result(startup_id)
    return {"indirect_competitors": extract_section(res.get("competitor_analysis_report", ""), "## 3. Indirect Competitors")}


@router.get("/competitor/comparison")
def get_competitor_comparison(startup_id: str = Depends(get_startup_id)):
    res = get_pipeline_result(startup_id)
    return {"product_feature_comparison": extract_section(res.get("competitor_analysis_report", ""), "## 4. Product & Feature Comparison")}


@router.get("/competitor/funding")
def get_competitor_funding(startup_id: str = Depends(get_startup_id)):
    res = get_pipeline_result(startup_id)
    return {"funding_market_position": extract_section(res.get("competitor_analysis_report", ""), "## 5. Funding & Market Position")}


@router.get("/competitor/advantages")
def get_competitor_advantages(startup_id: str = Depends(get_startup_id)):
    res = get_pipeline_result(startup_id)
    return {"competitive_advantages": extract_section(res.get("competitor_analysis_report", ""), "## 6. Competitive Advantages")}


@router.get("/competitor/intensity")
def get_competitor_intensity(startup_id: str = Depends(get_startup_id)):
    res = get_pipeline_result(startup_id)
    return {"competition_intensity_analysis": extract_section(res.get("competitor_analysis_report", ""), "## 7. Competition Intensity Score")}


@router.get("/competitor/insights")
def get_competitor_insights(startup_id: str = Depends(get_startup_id)):
    res = get_pipeline_result(startup_id)
    return {"strategic_competitive_insights": extract_section(res.get("competitor_analysis_report", ""), "## 8. Strategic Competitive Insights")}


@router.get("/competitor/metrics")
def get_competitor_metrics(startup_id: str = Depends(get_startup_id)):
    res = get_pipeline_result(startup_id)
    return res.get("competition_metrics", {})


@router.get("/competitor/intensity-score")
def get_competitor_intensity_score(startup_id: str = Depends(get_startup_id)):
    res = get_pipeline_result(startup_id)
    return res.get("competition_intensity", {})


# ─── Combined for frontend ────────────────────────────────────

@router.get("/competitor-analysis")
def get_competitor_analysis_combined(startup_id: str = Depends(get_startup_id)):
    res = get_pipeline_result(startup_id)
    return {
        "status": "completed",
        "competitor": {
            "competition_intensity": res.get("competition_intensity"),
            "metrics": parse_metrics(res.get("competition_metrics")),
            "analysis": parse_markdown_to_insights(res.get("competitor_analysis_report", ""))
        }
    }
