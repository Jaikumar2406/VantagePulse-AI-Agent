import asyncio

from agents.Startup_Data_Agent         import run_startup_data_agent
from agents.Market_Research_Agent      import run_market_research_agent
from agents.Competitor_Analysis_Agent  import run_competitor_analysis_agent
from agents.Financial_Estimation_Agent import run_financial_estimation_agent
from agents.Risk_Analysis_Agent        import run_risk_analysis_agent
from agents.growth_agent               import run_growth_prediction_agent
from agents.Investment_Decision_Agent  import run_investment_decision_agent
from backend.core.database import db
from state.state import AgentState


# ─── Individual trigger (kept for manual use) ────────────────

def process_startup_background(startup_id: str, state: AgentState):
    """Runs only the startup data agent. Used by individual manual triggers."""
    try:
        print(f"[{startup_id}] Running Startup Data Agent...")
        result = run_startup_data_agent(state)
        db[startup_id]["status"] = "completed"
        db[startup_id]["result"] = result
        print(f"[{startup_id}] Startup analysis completed")
    except Exception as e:
        db[startup_id]["status"] = "error"
        db[startup_id]["error"] = str(e)
        print(f"[{startup_id}] Error:", str(e))


# ─── COMBINED COMPLETE ANALYSIS (all 7 agents, one trigger) ──

async def _run_domain_agents(startup_id: str, state: AgentState):
    """
    Async implementation of the complete analysis pipeline.
    Runs in a single event loop to prevent httpx/playwright 'Event loop is closed' errors.
    """
    """
    ALL-OR-NOTHING pipeline triggered by POST /api/startup/analyze.

    Runs all 7 agents sequentially:
      1. Startup Data Agent
      2. Market Research
      3. Competitor Analysis
      4. Financial Estimation
      5. Risk Analysis
      6. Growth Prediction
      7. Investment Decision

    • global_status = "processing" for the entire duration.
    • global_status = "completed"  ONLY after ALL 7 agents succeed.
    • global_status = "failed"     immediately on any error.
    • No intermediate results are ever exposed through the API.
    """
    db[startup_id]["global_status"] = "processing"
    print(f"[{startup_id}] === Starting complete analysis pipeline ===")

    # Step 1 — Startup Data Agent (sync, no asyncio.run needed)
    try:
        print(f"[{startup_id}] [1/7] Running Startup Data Agent...")
        result = run_startup_data_agent(state)
        db[startup_id]["status"] = "completed"   # kept for legacy status check
        db[startup_id]["result"] = result
        print(f"[{startup_id}] [1/7] Startup Data Agent ✓")
    except Exception as e:
        db[startup_id]["global_status"] = "failed"
        db[startup_id]["global_error"] = f"Startup Data Agent failed: {e}"
        print(f"[{startup_id}] [1/7] FAILED: {e}")
        return

    # Steps 2-7 — Domain Agents
    domain_agents = [
        ("Market Research",      run_market_research_agent),
        ("Competitor Analysis",  run_competitor_analysis_agent),
        ("Financial Estimation", run_financial_estimation_agent),
        ("Risk Analysis",        run_risk_analysis_agent),
        ("Growth Prediction",    run_growth_prediction_agent),
        ("Investment Decision",  run_investment_decision_agent),
    ]

    for idx, (name, agent_fn) in enumerate(domain_agents, start=2):
        try:
            print(f"[{startup_id}] [{idx}/7] Running {name}...")
            db[startup_id]["current_agent"] = name
            system_state = db[startup_id]["result"]
            updated = await agent_fn(system_state)
            db[startup_id]["result"] = updated
            print(f"[{startup_id}] [{idx}/7] {name} ✓")
        except Exception as e:
            db[startup_id]["global_status"] = "failed"
            db[startup_id]["global_error"] = f"{name} failed: {e}"
            db[startup_id]["current_agent"] = name
            print(f"[{startup_id}] [{idx}/7] {name} FAILED: {e}")
            return

    # All 7 agents succeeded — release the data
    db[startup_id]["global_status"]   = "completed"
    db[startup_id]["current_agent"]   = None
    db[startup_id]["pipeline_status"] = "completed"   # keep for legacy compat
    print(f"[{startup_id}] === Complete analysis DONE ✓ — all endpoints unlocked ===")


def process_complete_analysis(startup_id: str, state: AgentState):
    """
    Synchronous wrapper for FastAPI BackgroundTasks.
    Creates a single, clean event loop for all 7 agents to share, preventing 
    'Event loop is closed' errors from httpx connection pools.
    """
    try:
        asyncio.run(_run_domain_agents(startup_id, state))
    except Exception as e:
        print(f"[{startup_id}] Fatal pipeline execution error: {e}")
        db[startup_id]["global_status"] = "failed"
        db[startup_id]["global_error"] = f"Fatal async wrapper error: {e}"
