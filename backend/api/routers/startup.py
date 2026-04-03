import os
import uuid
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile

from backend.core.database import db
from backend.core.dependencies import get_startup_id, get_pipeline_result
from backend.services.startup.tasks import process_complete_analysis
from backend.utils.extract import extract_section
from state.state import AgentState

router = APIRouter(prefix="/api", tags=["Startup"])

UPLOAD_DIR = os.path.join(os.getcwd(), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ─── Analyze Startup (creates startup_id — no header needed) ─

@router.post(
    "/startup/analyze",
    summary="Submit a startup for analysis",
    description=(
        "Creates a new analysis job. Returns `startup_id` — copy this value and "
        "click **Authorize** (🔒) at the top of the page to set `x-startup-id`. "
        "Then call `POST /api/startup/run-full-pipeline` and poll "
        "`GET /api/startup/pipeline-status`."
    ),
)
async def analyze_startup(
    background_tasks: BackgroundTasks,
    startup_name: str = Form(...),
    startup_website: Optional[str] = Form(None),
    founder_linkedin: Optional[str] = Form(None),
    funding_stage: Optional[str] = Form(None),
    pitch_deck_pdf: UploadFile = File(None),
):
    startup_id = str(uuid.uuid4())
    pitch_deck_path = ""

    if pitch_deck_pdf and pitch_deck_pdf.filename:
        filename = f"{startup_id}_{pitch_deck_pdf.filename}"
        pitch_deck_path = os.path.join(UPLOAD_DIR, filename)
        contents = await pitch_deck_pdf.read()
        with open(pitch_deck_path, "wb") as buffer:
            buffer.write(contents)

    state: AgentState = {
        "startup_name":     startup_name,
        "startup_website":  startup_website or "",
        "pitch_deck_pdf":   pitch_deck_path,
        "founder_linkedin": founder_linkedin or "",
        "funding_stage":    funding_stage or "",
    }

    # Initialize the global status lock required by dependencies.py
    db[startup_id] = {
        "status": "processing",  # legacy frontend checks
        "global_status": "processing",
        "pipeline_status": "not_started"
    }

    background_tasks.add_task(process_complete_analysis, startup_id, state)

    return {
        "startup_id": startup_id,
        "status":     "processing",
        "next_step":  "Set x-startup-id header, then poll GET /api/startup/pipeline-status until {'status': 'completed'}",
    }


# ─── Status (lightweight check, no pipeline gate needed) ─────

@router.get("/startup/status")
def get_status(startup_id: str = Depends(get_startup_id)):
    if startup_id not in db:
        raise HTTPException(status_code=404, detail="Startup ID not found")
    return {
        "startup_id":      startup_id,
        "status":          db[startup_id]["status"],
        "pipeline_status": db[startup_id].get("pipeline_status", "not_started"),
    }


# ─── All routes below are pipeline-gated (all-or-nothing) ────

@router.get("/startup/summary")
def get_summary(startup_id: str = Depends(get_startup_id)):
    res = get_pipeline_result(startup_id)
    return {"startup_summary": res.get("startup_summary")}


@router.get("/startup-data")
def get_startup_data(startup_id: str = Depends(get_startup_id)):
    res = get_pipeline_result(startup_id)

    metrics_dict = res.get("startup_metrics", {})
    metrics_array = []
    if isinstance(metrics_dict, dict):
        for k, v in metrics_dict.items():
            metrics_array.append({
                "subject": k.capitalize(),
                "score": float(v) * 10 if isinstance(v, (int, float)) else 0,
            })

    summary = res.get(
        "startup_summary",
        "Baseline data ingested. Pending deeper intelligence extraction.",
    )

    return {
        "status":         "completed",
        "startup_name":   res.get("startup_name", "Unknown Startup"),
        "industry":       res.get("industry", "Technology"),
        "funding_stage":  res.get("funding_stage", "Unknown"),
        "business_model": res.get("business_model", "Data Not Available"),
        "target_market":  res.get("target_market", "Data Not Available"),
        "summary":        summary,
        "metrics":        metrics_array,
        "key_details": [
            {
                "title":   "AI Summary Extraction",
                "summary": "Core intelligence signals extracted from raw data sources.",
                "details": summary,
            }
        ],
    }


@router.get("/startup/product")
def get_product(startup_id: str = Depends(get_startup_id)):
    res = get_pipeline_result(startup_id)
    summary = res.get("startup_summary", "")
    return {"product": extract_section(summary, "## 1. Product / Service")}


@router.get("/startup/industry")
def get_industry(startup_id: str = Depends(get_startup_id)):
    res = get_pipeline_result(startup_id)
    summary = res.get("startup_summary", "")
    return {"industry": extract_section(summary, "## 2. Industry & Market")}


@router.get("/startup/business-model")
def get_business_model(startup_id: str = Depends(get_startup_id)):
    res = get_pipeline_result(startup_id)
    summary = res.get("startup_summary", "")
    return {"business_model": extract_section(summary, "## 3. Business Model")}


@router.get("/startup/problem-solution")
def get_problem_solution(startup_id: str = Depends(get_startup_id)):
    res = get_pipeline_result(startup_id)
    summary = res.get("startup_summary", "")
    return {"problem_solution": extract_section(summary, "## 4. Problem & Solution")}


@router.get("/startup/metrics")
def get_metrics(startup_id: str = Depends(get_startup_id)):
    res = get_pipeline_result(startup_id)
    return res.get("startup_metrics", {})
