import os
from contextlib import asynccontextmanager

from backend.core.state import server_state
from backend.core.database import db


@asynccontextmanager
async def lifespan(app):
    """
    All startup checks run here.
    Server only accepts traffic after this completes successfully.
    """
    print("\n[Startup] Beginning initialization checks...")

    try:
        # 1. Verify required environment variables
        print("[Startup] Checking environment variables...")
        required_env = ["GROQ_API_KEY", "TAVILY_API_KEY"]
        missing = []
        for var in required_env:
            aliases = {
                "GROQ_API_KEY": [
                    "GROQ_API_KEY",
                    "agent1_llm",
                    "agent2_llm",
                    "agent3_llm",
                    "agent4_llm",
                    "agent5_llm",
                    "agent6_llm",
                    "agent7_llm",
                ],
                "TAVILY_API_KEY": ["TAVILY_API_KEY"],
            }
            found = any(os.getenv(alias) for alias in aliases.get(var, [var]))
            if not found:
                missing.append(var)

        if missing:
            raise EnvironmentError(
                f"Missing required environment variable(s): {', '.join(missing)}. "
                "Please check your .env file."
            )
        server_state.checks_passed.append("environment_variables")
        print("[Startup] ✓ Environment variables OK")

        # 2. Ensure upload directory exists and is writable
        print("[Startup] Checking upload directory...")
        upload_dir = os.path.join(os.getcwd(), "uploads")
        os.makedirs(upload_dir, exist_ok=True)
        test_file = os.path.join(upload_dir, ".write_test")
        with open(test_file, "w") as f:
            f.write("ok")
        os.remove(test_file)
        server_state.checks_passed.append("upload_directory")
        print("[Startup] ✓ Upload directory writable")

        # 3. Verify agent modules are importable
        print("[Startup] Verifying agent imports...")
        from agents.Startup_Data_Agent       import run_startup_data_agent        # noqa: F401
        from agents.Market_Research_Agent    import run_market_research_agent     # noqa: F401
        from agents.Risk_Analysis_Agent      import run_risk_analysis_agent       # noqa: F401
        from agents.growth_agent             import run_growth_prediction_agent   # noqa: F401
        from agents.Financial_Estimation_Agent import run_financial_estimation_agent  # noqa: F401
        from agents.Competitor_Analysis_Agent  import run_competitor_analysis_agent   # noqa: F401
        from agents.Investment_Decision_Agent  import run_investment_decision_agent   # noqa: F401
        server_state.checks_passed.append("agent_imports")
        print("[Startup] ✓ All agent modules importable")

        server_state.ready = True
        print("[Startup] ✓ Server is READY — accepting requests\n")

    except Exception as exc:
        server_state.ready = False
        server_state.init_error = str(exc)
        print(f"[Startup] ✗ Initialization FAILED: {exc}")
        print("[Startup]   Server is running but NOT ready. Fix the error and restart.\n")

    yield  # ← server is live from this point

    # Shutdown
    print("[Shutdown] Clearing in-memory database...")
    db.clear()
    print("[Shutdown] Done.")
