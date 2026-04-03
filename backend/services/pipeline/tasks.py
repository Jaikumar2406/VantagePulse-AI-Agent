import asyncio

from agents.Market_Research_Agent     import run_market_research_agent
from agents.Competitor_Analysis_Agent import run_competitor_analysis_agent
from agents.Financial_Estimation_Agent import run_financial_estimation_agent
from agents.Risk_Analysis_Agent       import run_risk_analysis_agent
from agents.growth_agent              import run_growth_prediction_agent
from agents.Investment_Decision_Agent import run_investment_decision_agent
from backend.core.database import db


def process_full_pipeline_background(startup_id: str):
    """
    Runs every agent sequentially after startup data is ready.
    ALL-OR-NOTHING: only sets pipeline_status = "completed" after
    every step succeeds. Any failure marks pipeline_status = "failed".
    No intermediate results are ever marked as ready.
    """
    agents = [
        ("market_research",      run_market_research_agent),
        ("competitor_analysis",  run_competitor_analysis_agent),
        ("financial_estimation", run_financial_estimation_agent),
        ("risk_analysis",        run_risk_analysis_agent),
        ("growth_prediction",    run_growth_prediction_agent),
        ("investment_decision",  run_investment_decision_agent),
    ]

    db[startup_id]["pipeline_status"] = "running"

    for agent_name, agent_fn in agents:
        try:
            print(f"[{startup_id}] Pipeline: running {agent_name} ...")
            db[startup_id]["pipeline_current_agent"] = agent_name
            system_state = db[startup_id]["result"]
            updated = asyncio.run(agent_fn(system_state))
            db[startup_id]["result"] = updated
            print(f"[{startup_id}] Pipeline: {agent_name} done ✓")
        except Exception as e:
            # ALL-OR-NOTHING: a single failure marks the whole pipeline failed
            db[startup_id]["pipeline_status"] = "failed"
            db[startup_id]["pipeline_current_agent"] = agent_name
            db[startup_id]["pipeline_error"] = str(e)
            print(f"[{startup_id}] Pipeline FAILED in {agent_name}: {e}")
            return

    # Only reach here if ALL agents succeeded
    db[startup_id]["pipeline_status"] = "completed"
    db[startup_id]["pipeline_current_agent"] = None
    print(f"[{startup_id}] Full pipeline completed ✓  — data now available on all endpoints")
