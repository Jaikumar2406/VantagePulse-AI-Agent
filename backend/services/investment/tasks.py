import asyncio

from agents.Investment_Decision_Agent import run_investment_decision_agent
from backend.core.database import db


def process_investment_background(startup_id: str):
    try:
        print(f"[{startup_id}] Running Investment Decision Agent...")
        system_state = db[startup_id]["result"]
        updated_system_state = asyncio.run(run_investment_decision_agent(system_state))
        db[startup_id]["result"] = updated_system_state
        print(f"[{startup_id}] Investment decision completed")
    except Exception as e:
        db[startup_id]["status"] = "error"
        db[startup_id]["error"] = str(e)
        print(f"[{startup_id}] Error:", str(e))
