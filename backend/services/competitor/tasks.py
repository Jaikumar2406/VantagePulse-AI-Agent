import asyncio

from agents.Competitor_Analysis_Agent import run_competitor_analysis_agent
from backend.core.database import db


def process_competitor_background(startup_id: str):
    try:
        print(f"[{startup_id}] Running Competitor Analysis Agent...")
        system_state = db[startup_id]["result"]
        updated_system_state = asyncio.run(run_competitor_analysis_agent(system_state))
        db[startup_id]["result"] = updated_system_state
        print(f"[{startup_id}] Competitor analysis completed")
    except Exception as e:
        db[startup_id]["status"] = "error"
        db[startup_id]["error"] = str(e)
        print(f"[{startup_id}] Error:", str(e))
