from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from backend.core.database import db
from backend.core.dependencies import get_startup_id, get_pipeline_result
from backend.services.financial.tasks import process_financial_background
from backend.utils.extract import extract_section, parse_markdown_to_insights, parse_metrics

router = APIRouter(prefix="/api", tags=["Financial Estimation"])


@router.post("/financial/analyze")
async def analyze_financial(
    background_tasks: BackgroundTasks,
    startup_id: str = Depends(get_startup_id),
):
    if startup_id not in db:
        raise HTTPException(status_code=404, detail="Startup ID not found")
    background_tasks.add_task(process_financial_background, startup_id)
    return {"startup_id": startup_id, "agent": "financial_estimation", "status": "processing"}


@router.get("/financial/revenue")
def get_financial_revenue(startup_id: str = Depends(get_startup_id)):
    res = get_pipeline_result(startup_id)
    return {"revenue_indicators": extract_section(res.get("financial_estimation_report", ""), "## 1. Revenue Indicators")}


@router.get("/financial/burn-rate")
def get_financial_burn_rate(startup_id: str = Depends(get_startup_id)):
    res = get_pipeline_result(startup_id)
    return {"burn_rate_estimation": extract_section(res.get("financial_estimation_report", ""), "## 2. Burn Rate Estimation")}


@router.get("/financial/runway")
def get_financial_runway(startup_id: str = Depends(get_startup_id)):
    res = get_pipeline_result(startup_id)
    return {"runway_estimation": extract_section(res.get("financial_estimation_report", ""), "## 3. Runway Estimation")}


@router.get("/financial/unit-economics")
def get_financial_unit_economics(startup_id: str = Depends(get_startup_id)):
    res = get_pipeline_result(startup_id)
    return {"unit_economics": extract_section(res.get("financial_estimation_report", ""), "## 4. Unit Economics")}


@router.get("/financial/valuation")
def get_financial_valuation(startup_id: str = Depends(get_startup_id)):
    res = get_pipeline_result(startup_id)
    return {"valuation_range": extract_section(res.get("financial_estimation_report", ""), "## 5. Valuation Range")}


@router.get("/financial/sustainability")
def get_financial_sustainability(startup_id: str = Depends(get_startup_id)):
    res = get_pipeline_result(startup_id)
    return {"financial_sustainability": extract_section(res.get("financial_estimation_report", ""), "## 6. Financial Sustainability Assessment")}


@router.get("/financial/metrics")
def get_financial_metrics(startup_id: str = Depends(get_startup_id)):
    res = get_pipeline_result(startup_id)
    return res.get("financial_metrics", {})


@router.get("/financial/signals")
def get_financial_signals(startup_id: str = Depends(get_startup_id)):
    res = get_pipeline_result(startup_id)
    return res.get("financial_signals", {})


# ─── Combined for frontend ────────────────────────────────────

@router.get("/financial-estimation")
def get_financial_estimation_combined(startup_id: str = Depends(get_startup_id)):
    res = get_pipeline_result(startup_id)
    return {
        "status": "completed",
        "financial": {
            "metrics": parse_metrics(res.get("financial_metrics")),
            "analysis": parse_markdown_to_insights(res.get("financial_estimation_report", ""))
        }
    }
