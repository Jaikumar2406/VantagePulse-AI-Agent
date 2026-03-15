from agents.Competitor_Analysis_Agent import run_competitor_analysis_agent
from agents.Financial_Estimation_Agent import run_financial_estimation_agent
from agents.growth_agent import run_growth_prediction_agent
from agents.Investment_Decision_Agent import run_investment_decision_agent
from agents.Risk_Analysis_Agent import run_risk_analysis_agent
from agents.Market_Research_Agent import run_market_research_agent
from agents.Startup_Data_Agent import run_startup_data_agent

from state.state import AgentState

from langgraph.graph import StateGraph, START, END


def build_workflow():

    graph = StateGraph(AgentState)

    # Nodes (Agents)
    graph.add_node("startup_data", run_startup_data_agent)
    graph.add_node("market_research", run_market_research_agent)
    graph.add_node("competitor_analysis", run_competitor_analysis_agent)
    graph.add_node("financial_estimation", run_financial_estimation_agent)
    graph.add_node("risk_analysis", run_risk_analysis_agent)
    graph.add_node("growth_prediction", run_growth_prediction_agent)
    graph.add_node("investment_decision", run_investment_decision_agent)

    # Workflow pipeline
    graph.add_edge(START, "startup_data")

    graph.add_edge("startup_data", "market_research")
    graph.add_edge("market_research", "competitor_analysis")
    graph.add_edge("competitor_analysis", "financial_estimation")
    graph.add_edge("financial_estimation", "risk_analysis")
    graph.add_edge("risk_analysis", "growth_prediction")
    graph.add_edge("growth_prediction", "investment_decision")

    graph.add_edge("investment_decision", END)

    return graph.compile()