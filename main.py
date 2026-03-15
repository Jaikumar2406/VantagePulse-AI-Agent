import asyncio

from graph import build_workflow
from state.state import AgentState


async def main():

    print("\n" + "="*60)
    print("     AI VC Startup Intelligence System")
    print("="*60)

    # user input
    startup_name     = input("Startup Name: ")
    startup_website  = input("Startup Website: ")
    pitch_deck_pdf   = input("Pitch Deck Path: ")
    founder_linkedin = input("Founder LinkedIn URL: ")
    funding_stage    = input("Funding Stage (Pre-seed / Seed / Series A): ")

    # initial state
    state: AgentState = {
        "startup_name":     startup_name,
        "startup_website":  startup_website,
        "pitch_deck_pdf":   pitch_deck_pdf,
        "founder_linkedin": founder_linkedin,
        "funding_stage":    funding_stage
    }

    print("\n[System] Building agent workflow...\n")

    # build workflow
    workflow = build_workflow()

    # ── Run workflow asynchronously ──────────────────────────────
    # All agent nodes are async def, so ainvoke() must be used.
    result = await workflow.ainvoke(state)

    print("\n" + "="*60)
    print("        FINAL INVESTMENT REPORT")
    print("="*60)

    # print important outputs
    if "startup_summary" in result:
        print("\nStartup Summary:\n")
        print(result["startup_summary"])

    if "market_research_report" in result:
        print("\nMarket Research:\n")
        print(result["market_research_report"])

    if "competitor_analysis_report" in result:
        print("\nCompetitor Analysis:\n")
        print(result["competitor_analysis_report"])

    if "financial_estimation_report" in result:
        print("\nFinancial Estimation:\n")
        print(result["financial_estimation_report"])

    if "risk_analysis_report" in result:
        print("\nRisk Analysis:\n")
        print(result["risk_analysis_report"])

    if "growth_prediction_report" in result:
        print("\nGrowth Prediction:\n")
        print(result["growth_prediction_report"])

    if "investment_decision_report" in result:
        print("\nInvestment Decision:\n")
        print(result["investment_decision_report"])

    print("\n" + "="*60)
    print("Analysis Completed")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())