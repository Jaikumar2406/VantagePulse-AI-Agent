import re


def extract_section(report: str, title: str) -> str:
    """
    Generic section extractor used across all domain report parsers.
    Extracts text between a markdown heading `title` and the next ## heading (or end of string).
    """
    pattern = rf"{title}(.*?)(?=##|\Z)"
    match = re.search(pattern, report, re.DOTALL)
    if match:
        return match.group(1).strip()
    return "Section not found"

import json
from typing import Dict, List, Any

def parse_metrics(metrics_data: Any) -> Dict:
    """Safely extracts dictionary from stringified JSON metrics."""
    if isinstance(metrics_data, dict):
        return metrics_data
    if isinstance(metrics_data, str) and metrics_data.strip():
        try:
            return json.loads(metrics_data)
        except Exception:
            pass
    return {}

def parse_markdown_to_insights(markdown: str) -> List[Dict]:
    """Parses a standard agent AI Markdown report into structured JSON `{title, summary, details}` blocks."""
    if not markdown:
        return [{"title": "Analysis Pending", "summary": "Awaiting data", "details": "No report generated."}]
    
    sections = [s for s in markdown.split('## ') if s.strip()]
    insights = []
    
    for i, sec in enumerate(sections):
        lines = sec.strip().split("\n")
        title = lines[0].replace("**", "").strip()
        content = " ".join(lines[1:]).strip()
        
        # very simple summary extraction (first sentence)
        match = re.search(r'[^.!?]*[.!?]', content)
        summary = match.group(0).strip() if match else "Insight details."
        details = content[len(summary):].strip() or content
        
        insights.append({
            "id": i,
            "title": title,
            "summary": summary,
            "details": details
        })
    return insights

def parse_investment_markdown(markdown: str) -> Dict:
    """Parses the specific Strengths/Risks/Thesis markdown from Investment Agent."""
    if not markdown:
        return {"key_strengths": [], "key_risks": [], "investment_thesis": "Analysis Pending..."}
    
    lines = markdown.split("\n")
    strengths, risks = [], []
    thesis = ""
    
    current_section = "thesis"
    for line in lines:
        l = line.lower()
        if "strength" in l or "pro" in l or "advantage" in l:
            current_section = "strengths"
        elif "risk" in l or "con" in l or "weakness" in l:
            current_section = "risks"
        elif "thesis" in l or "conclusion" in l:
            current_section = "thesis"
        elif line.strip().startswith("-") or line.strip().startswith("*"):
            bullet_text = line.strip()[1:].strip()
            if current_section == "strengths":
                strengths.append(bullet_text)
            if current_section == "risks":
                risks.append(bullet_text)
        elif line.strip() and current_section == "thesis" and not line.strip().startswith("#"):
            thesis += line.strip() + " "
            
    return {
        "key_strengths": strengths if strengths else ["No specific strengths extracted format."],
        "key_risks": risks if risks else ["No specific risks extracted format."],
        "investment_thesis": thesis.strip() if thesis else markdown[:300] + "..."
    }

