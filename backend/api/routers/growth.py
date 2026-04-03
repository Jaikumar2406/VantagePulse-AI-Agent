from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from backend.core.database import db
from backend.core.dependencies import get_startup_id, get_pipeline_result
from backend.services.growth.tasks import process_growth_background
from backend.utils.extract import extract_section, parse_markdown_to_insights, parse_metrics

router = APIRouter(prefix="/api", tags=["Growth Prediction"])


@router.post("/growth/analyze")
async def analyze_growth(
    background_tasks: BackgroundTasks,
    startup_id: str = Depends(get_startup_id),
):
    if startup_id not in db:
        raise HTTPException(status_code=404, detail="Startup ID not found")
    background_tasks.add_task(process_growth_background, startup_id)
    return {"startup_id": startup_id, "agent": "growth_prediction", "status": "processing"}


@router.get("/growth/market-context")
def get_growth_market_context(startup_id: str = Depends(get_startup_id)):
    res = get_pipeline_result(startup_id)
    return {"market_growth_context": extract_section(res.get("growth_prediction_report", ""), "## 1. Market Growth Context")}


@router.get("/growth/traction")
def get_growth_traction(startup_id: str = Depends(get_startup_id)):
    res = get_pipeline_result(startup_id)
    return {"startup_traction": extract_section(res.get("growth_prediction_report", ""), "## 2. Startup Traction Signals")}


@router.get("/growth/team")
def get_growth_team(startup_id: str = Depends(get_startup_id)):
    res = get_pipeline_result(startup_id)
    return {"team_strength": extract_section(res.get("growth_prediction_report", ""), "## 3. Founder and Team Strength")}


@router.get("/growth/competition")
def get_growth_competition(startup_id: str = Depends(get_startup_id)):
    res = get_pipeline_result(startup_id)
    return {"competitive_position": extract_section(res.get("growth_prediction_report", ""), "## 4. Competitive Position")}


@router.get("/growth/financial")
def get_growth_financial(startup_id: str = Depends(get_startup_id)):
    res = get_pipeline_result(startup_id)
    return {"financial_growth": extract_section(res.get("growth_prediction_report", ""), "## 5. Financial Growth Potential")}


@router.get("/growth/expansion")
def get_growth_expansion(startup_id: str = Depends(get_startup_id)):
    res = get_pipeline_result(startup_id)
    return {"expansion_potential": extract_section(res.get("growth_prediction_report", ""), "## 6. Expansion Potential")}


@router.get("/growth/prediction")
def get_growth_prediction(startup_id: str = Depends(get_startup_id)):
    res = get_pipeline_result(startup_id)
    return {"growth_prediction": extract_section(res.get("growth_prediction_report", ""), "## 7. Growth Prediction")}


@router.get("/growth/classification")
def get_growth_classification(startup_id: str = Depends(get_startup_id)):
    res = get_pipeline_result(startup_id)
    return {"growth_classification": extract_section(res.get("growth_prediction_report", ""), "## 8. Growth Classification")}


@router.get("/growth/metrics")
def get_growth_metrics(startup_id: str = Depends(get_startup_id)):
    res = get_pipeline_result(startup_id)
    return res.get("growth_metrics", {})


@router.get("/growth/signals")
def get_growth_signals(startup_id: str = Depends(get_startup_id)):
    res = get_pipeline_result(startup_id)
    return res.get("growth_signals", {})


# ─── Combined for frontend ────────────────────────────────────

@router.get("/growth-prediction")
def get_growth_prediction_combined(startup_id: str = Depends(get_startup_id)):
    res = get_pipeline_result(startup_id)
    return {
        "status": "completed",
        "growth": {
            "metrics": parse_metrics(res.get("growth_metrics")),
            "analysis": parse_markdown_to_insights(res.get("growth_prediction_report", ""))
        }
    }
