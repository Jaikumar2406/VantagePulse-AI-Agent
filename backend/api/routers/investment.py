from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from backend.core.database import db
from backend.core.dependencies import get_startup_id, get_pipeline_result
from backend.services.investment.tasks import process_investment_background
from backend.utils.extract import extract_section, parse_investment_markdown, parse_metrics

router = APIRouter(prefix="/api", tags=["Investment Decision"])


@router.post("/investment/analyze")
async def analyze_investment(
    background_tasks: BackgroundTasks,
    startup_id: str = Depends(get_startup_id),
):
    if startup_id not in db:
        raise HTTPException(status_code=404, detail="Startup ID not found")
    background_tasks.add_task(process_investment_background, startup_id)
    return {"startup_id": startup_id, "agent": "investment_decision", "status": "processing"}


@router.get("/investment/overview")
def get_investment_overview(startup_id: str = Depends(get_startup_id)):
    res = get_pipeline_result(startup_id)
    return {"startup_overview": extract_section(res.get("investment_decision_report", ""), "## 1. Startup Overview")}


@router.get("/investment/strengths")
def get_investment_strengths(startup_id: str = Depends(get_startup_id)):
    res = get_pipeline_result(startup_id)
    return {"key_strengths": extract_section(res.get("investment_decision_report", ""), "## 2. Key Strengths")}


@router.get("/investment/risks")
def get_investment_risks(startup_id: str = Depends(get_startup_id)):
    res = get_pipeline_result(startup_id)
    return {"key_risks": extract_section(res.get("investment_decision_report", ""), "## 3. Key Risks")}


@router.get("/investment/score")
def get_investment_score(startup_id: str = Depends(get_startup_id)):
    res = get_pipeline_result(startup_id)
    return {"investment_score": extract_section(res.get("investment_decision_report", ""), "## 4. Investment Score")}


@router.get("/investment/structure")
def get_investment_structure(startup_id: str = Depends(get_startup_id)):
    res = get_pipeline_result(startup_id)
    return {"investment_structure": extract_section(res.get("investment_decision_report", ""), "## 5. Recommended Investment Structure")}


@router.get("/investment/returns")
def get_investment_returns(startup_id: str = Depends(get_startup_id)):
    res = get_pipeline_result(startup_id)
    return {"expected_returns": extract_section(res.get("investment_decision_report", ""), "## 6. Expected Return Potential")}


@router.get("/investment/final")
def get_investment_final(startup_id: str = Depends(get_startup_id)):
    res = get_pipeline_result(startup_id)
    return {"final_recommendation": extract_section(res.get("investment_decision_report", ""), "## 7. Final Recommendation")}


@router.get("/investment/metrics")
def get_investment_metrics(startup_id: str = Depends(get_startup_id)):
    res = get_pipeline_result(startup_id)
    return res.get("investment_metrics", {})


@router.get("/investment/signals")
def get_investment_signals(startup_id: str = Depends(get_startup_id)):
    res = get_pipeline_result(startup_id)
    return res.get("investment_decision_signals", {})


# ─── Combined for frontend ────────────────────────────────────

@router.get("/investment-decision")
def get_investment_decision_combined(startup_id: str = Depends(get_startup_id)):
    res = get_pipeline_result(startup_id)
    return {
        "status": "completed",
        "investment": {
            "metrics": parse_metrics(res.get("investment_metrics")),
            "decision": parse_investment_markdown(res.get("investment_decision_report", ""))
        }
    }
