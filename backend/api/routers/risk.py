from fastapi import APIRouter, Depends

from backend.core.dependencies import get_startup_id, get_pipeline_result
from backend.utils.extract import extract_section, parse_markdown_to_insights, parse_metrics

router = APIRouter(prefix="/api", tags=["Risk Analysis"])


@router.get("/risk/market")
def get_risk_market(startup_id: str = Depends(get_startup_id)):
    res = get_pipeline_result(startup_id)
    return {"market_risk": extract_section(res.get("risk_analysis_report", ""), "## 1. Market Risk")}


@router.get("/risk/technology")
def get_risk_technology(startup_id: str = Depends(get_startup_id)):
    res = get_pipeline_result(startup_id)
    return {"technology_risk": extract_section(res.get("risk_analysis_report", ""), "## 2. Technology Risk")}


@router.get("/risk/execution")
def get_risk_execution(startup_id: str = Depends(get_startup_id)):
    res = get_pipeline_result(startup_id)
    return {"execution_risk": extract_section(res.get("risk_analysis_report", ""), "## 3. Execution Risk")}


@router.get("/risk/regulatory")
def get_risk_regulatory(startup_id: str = Depends(get_startup_id)):
    res = get_pipeline_result(startup_id)
    return {"regulatory_risk": extract_section(res.get("risk_analysis_report", ""), "## 4. Regulatory Risk")}


@router.get("/risk/financial")
def get_risk_financial(startup_id: str = Depends(get_startup_id)):
    res = get_pipeline_result(startup_id)
    return {"financial_risk": extract_section(res.get("risk_analysis_report", ""), "## 5. Financial Risk")}


@router.get("/risk/competition")
def get_risk_competition(startup_id: str = Depends(get_startup_id)):
    res = get_pipeline_result(startup_id)
    return {"competition_risk": extract_section(res.get("risk_analysis_report", ""), "## 6. Competition Risk")}


@router.get("/risk/overall")
def get_risk_overall(startup_id: str = Depends(get_startup_id)):
    res = get_pipeline_result(startup_id)
    return {"overall_risk": extract_section(res.get("risk_analysis_report", ""), "## 7. Overall Risk Score")}


@router.get("/risk/red-flags")
def get_risk_red_flags(startup_id: str = Depends(get_startup_id)):
    res = get_pipeline_result(startup_id)
    return {"red_flags": extract_section(res.get("risk_analysis_report", ""), "## 8. Major Red Flags")}


@router.get("/risk/final")
def get_risk_final(startup_id: str = Depends(get_startup_id)):
    res = get_pipeline_result(startup_id)
    return {"final_assessment": extract_section(res.get("risk_analysis_report", ""), "## 9. Final Risk Assessment")}


@router.get("/risk/metrics")
def get_risk_metrics(startup_id: str = Depends(get_startup_id)):
    res = get_pipeline_result(startup_id)
    return res.get("risk_metrics", {})


@router.get("/risk/signals")
def get_risk_signals(startup_id: str = Depends(get_startup_id)):
    res = get_pipeline_result(startup_id)
    return res.get("risk_signals", {})


# ─── Combined for frontend ────────────────────────────────────

@router.get("/risk-analysis")
def get_risk_analysis_combined(startup_id: str = Depends(get_startup_id)):
    res = get_pipeline_result(startup_id)
    return {
        "status": "completed",
        "risk": {
            "metrics": parse_metrics(res.get("risk_metrics")),
            "analysis": parse_markdown_to_insights(res.get("risk_analysis_report", ""))
        }
    }
