import asyncio

from agents.growth_agent import run_growth_prediction_agent
from backend.core.database import db


def process_growth_background(startup_id: str):
    try:
        print(f"[{startup_id}] Running Growth Prediction Agent...")
        system_state = db[startup_id]["result"]
        updated_system_state = asyncio.run(run_growth_prediction_agent(system_state))
        db[startup_id]["result"] = updated_system_state
        print(f"[{startup_id}] Growth prediction completed")
    except Exception as e:
        db[startup_id]["status"] = "error"
        db[startup_id]["error"] = str(e)
        print(f"[{startup_id}] Error:", str(e))
