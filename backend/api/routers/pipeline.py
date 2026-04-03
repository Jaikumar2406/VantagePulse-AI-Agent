from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from backend.core.database import db
from backend.core.dependencies import get_startup_id
from backend.services.pipeline.tasks import process_full_pipeline_background
from backend.utils.extract import parse_markdown_to_insights, parse_investment_markdown, parse_metrics

router = APIRouter(prefix="/api", tags=["Pipeline"])


@router.post(
    "/startup/run-full-pipeline",
    summary="Run full analysis pipeline",
    description=(
        "Triggers all 6 agents sequentially (market → competitor → financial → "
        "risk → growth → investment). Startup data agent must have already completed. "
        "All GET endpoints return HTTP 202 until this pipeline completes."
    ),
)
async def run_full_pipeline(
    background_tasks: BackgroundTasks,
    startup_id: str = Depends(get_startup_id),
):
    if startup_id not in db:
        raise HTTPException(status_code=404, detail="Startup ID not found")

    if db[startup_id].get("status") != "completed":
        raise HTTPException(
            status_code=400,
            detail=(
                "Startup Data Agent has not completed yet. "
                "Wait for POST /api/startup/analyze to finish first, "
                "then poll GET /api/startup/status until status = 'completed'."
            ),
        )

    background_tasks.add_task(process_full_pipeline_background, startup_id)

    return {
        "startup_id":      startup_id,
        "pipeline_status": "running",
        "message":         "Full pipeline started. Poll GET /api/startup/pipeline-status to track progress.",
    }


@router.get(
    "/startup/pipeline-status",
    summary="Poll pipeline progress",
    description="Returns current pipeline status. Call this until status = 'completed'.",
)
def get_pipeline_status(startup_id: str = Depends(get_startup_id)):
    if startup_id not in db:
        raise HTTPException(status_code=404, detail="Startup ID not found")

    global_status = db[startup_id].get("global_status", "not_started")
    mapped_status = "processing" if global_status in ("not_started", "running") else global_status

    return {
        "startup_id":      startup_id,
        "status":          mapped_status,  # Frontend strictly polls for { "status": "processing" | "completed" }
        "pipeline_status": mapped_status,  # Kept for legacy compatibility
        "current_agent":   db[startup_id].get("current_agent"),
        "error":           db[startup_id].get("global_error") or db[startup_id].get("pipeline_error"),
    }


@router.get(
    "/startup/full-report",
    summary="Full combined report (requires completed pipeline)",
    description=(
        "Returns ALL agent results in one response. "
        "Returns HTTP 202 if the pipeline has not yet completed."
    ),
)
def get_full_report(startup_id: str = Depends(get_startup_id)):
    if startup_id not in db:
        raise HTTPException(status_code=404, detail="Startup ID not found")

    global_status = db[startup_id].get("global_status", "not_started")

    if global_status == "failed":
        raise HTTPException(
            status_code=500,
            detail=db[startup_id].get("global_error", "Pipeline failed"),
        )

    if global_status != "completed":
        return {
            "startup_id":      startup_id,
            "status":          "processing",
            "pipeline_status": "processing",
            "current_agent":   db[startup_id].get("current_agent"),
            "message":         "Pipeline not completed yet. Check back soon.",
        }

    result = db[startup_id].get("result", {})

    return {
        "startup_id":      startup_id,
        "status":          "completed",
        "pipeline_status": "completed",

        "startup_name":   result.get("startup_name", "Unknown Startup"),
        "industry":       result.get("industry", "Technology"),
        "funding_stage":  result.get("funding_stage", "Unknown"),
        "business_model": result.get("business_model", "Data Not Available"),
        "target_market":  result.get("target_market", "Data Not Available"),

        "startup": {
            "summary": result.get("startup_summary", ""),
            "metrics": parse_metrics(result.get("startup_metrics"))
        },
        "market": {
            "opportunity_flag": result.get("market_opportunity_flag"),
            "metrics": parse_metrics(result.get("market_metrics")),
            "analysis": parse_markdown_to_insights(result.get("market_research_report", ""))
        },
        "competitor": {
            "competition_intensity": result.get("competition_intensity"),
            "metrics": parse_metrics(result.get("competition_metrics")),
            "analysis": parse_markdown_to_insights(result.get("competitor_analysis_report", ""))
        },
        "financial": {
            "metrics": parse_metrics(result.get("financial_metrics")),
            "analysis": parse_markdown_to_insights(result.get("financial_estimation_report", ""))
        },
        "risk": {
            "signals": parse_metrics(result.get("risk_signals")),
            "metrics": parse_metrics(result.get("risk_metrics")),
            "analysis": parse_markdown_to_insights(result.get("risk_analysis_report", ""))
        },
        "growth": {
            "signals": parse_metrics(result.get("growth_signals")),
            "metrics": parse_metrics(result.get("growth_metrics")),
            "analysis": parse_markdown_to_insights(result.get("growth_prediction_report", ""))
        },
        "investment": {
            "signals": parse_metrics(result.get("investment_signals")),
            "metrics": parse_metrics(result.get("investment_metrics")),
            "decision": parse_investment_markdown(result.get("investment_decision_report", ""))
        }
    }
