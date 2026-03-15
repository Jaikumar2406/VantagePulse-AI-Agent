from typing import Literal,Optional
from pydantic import BaseModel, Field
# import pandas as pd
# from typing import TypedDict, Optional, List, Dict, Any
from typing_extensions import TypedDict


# ── LangGraph State ─────────────────────────

class AgentState(TypedDict, total=False):
    # user input
    startup_name: str
    startup_website: str
    pitch_deck_pdf: str
    founder_linkedin :str
    funding_stage: str

    # collected data
    website_content: str
    pitch_deck_content: str
    founder_profile_data: str

    # final output
    startup_summary: str
    startup_raw_data: str

    market_research_report:str
    market_search_results:str
    market_opportunity_flag: str

    competitor_analysis_report:str
    competitor_search_results:str
    competition_intensity:str

    financial_estimation_report:str
    financial_signals:str
    
    risk_analysis_report:str
    risk_search_results:str
    risk_signals:str

    growth_prediction_report:str
    growth_search_results:str
    growth_signals:str

    investment_decision_report:str
    investment_decision_signals:str

    growth_metrics:str #for market visualization
    startup_metrics: str # for startup visualization
    competition_metrics: str # for competition visualization
    financial_metrics : str # for financial visulization
    market_metrics:str #for market visualization
    investment_metrics:str #for investment visualization
    risk_metrics:str # for risk visualization


