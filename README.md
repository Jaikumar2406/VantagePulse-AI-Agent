# 🚀 VantagePulse: AI-Powered Startup Intelligence

## ⚠️ The Problem: The "Due Diligence Bottleneck"

In the fast-paced world of Venture Capital, making a wrong investment decision is expensive, but missing out on a "unicorn" because of slow due diligence is even worse. Traditional startup evaluation suffers from:

1.  **Manual Overload**: Analysts spend hundreds of hours manually scraping websites, reading 50-page pitch decks, and searching for competitors.
2.  **Fragmented Data**: Information is scattered across LinkedIn, news articles, financial reports, and visual PDFs, making it hard to form a cohesive picture.
3.  **Human Bias & Fatigue**: Critical red flags (like a 4-month runway or a crowded market) are often missed during the 10th review of the day.
4.  **Static Analysis**: Most reports are out-of-date the moment they are written as market conditions and competitor funding change daily.

---

## ✅ The Solution: Autonomous Multi-Agent Diligence

**VantagePulse** solves the bottleneck by replacing manual research with an **Autonomous Multi-Agent Intelligence System**. 

Instead of a single AI trying to "do everything," we deploy a team of **seven specialized agents** that work collaboratively. They scrape the live web, "see" into pitch decks via OCR, and perform real-time financial forensic analysis—mimicking the workflow of a high-performance VC investment committee, but at 100x the speed.

---

## 🛠️ Detailed Solution: The Agent Pipeline

The system utilizes **LangGraph** to orchestrate a stateful, sequential workflow. Every startup is processed through a structured 7-stage intelligence funnel:

### 1. Data Capture & Extraction (The Startup Data Agent)
We don't rely on static databases. Our first agent uses **Playwright** to crawl the startup’s live website and **Tesseract OCR** to "read" visual pitch decks. By pulling the latest data from the source, we ensure the analysis is based on the most current product information and founder backgrounds.

### 2. Live Market Intelligence (The Market Research Agent)
This agent performs an "Industrial Sweep." It uses **Tavily AI Search** to find real-time market sizes (TAM/SAM/SOM), current CAGR growth rates, and emerging trends for 2025. It identifies whether the market is large enough to support a venture-scale return before the analysis proceeds.

### 3. Competitive Mapping (The Competitor Analysis Agent)
The agent maps the battlefield. It identifies both direct and indirect competitors, calculating a **Competition Intensity Score**. It looks for "Moats"—proprietary technology, network effects, or unique distribution—that give the startup a surviving edge in crowded sectors.

### 4. Financial Forensics (The Financial Estimation Agent)
Operating on a **Negative Constraint** model (no fabrication allowed), this agent cross-references pricing signals with traction data. It estimates the Monthly Burn Rate and calculates the **Cash Runway**—identifying exactly how many months the startup has before it needs more capital.

### 5. Risk & Red Flag Detection (The Risk Analysis Agent)
The "Devil's Advocate" of the system. It executes 10+ risk-specific search queries to find hidden traps in regulations, technology limitations, or execution gaps. It assigns a risk heatmap to the investment, ensuring no "silent killer" goes unnoticed.

### 6. Growth Forecasting (The Growth Prediction Agent)
Moving from data to prediction, this agent analyzes historical comparable startup trajectories to estimate a **Success Probability (0-100%)**. It projects the company’s valuation over a 3-year horizon based on current market tailwinds.

### 7. The Final Verdict (The Investment Decision Agent)
The "Managing Partner" agent. It synthesizes the collective intelligence of the previous six agents into a weighted scoring model. It issues a final recommendation—**INVEST**, **WATCH**, or **REJECT**—accompanied by a professional Investment Committee (IC) Memo.

---

## 🤖 Agent Architecture

The system operates as a collaborative **multi-agent pipeline** powered by **LangGraph**. Each agent is a specialized "AI employee" with a specific domain of expertise.

### 1. Startup Data Agent
The **Startup Data Agent** serves as the initial collection point for the pipeline. It utilizes **Playwright** for deep-web scraping of the startup's official site, **Tesseract OCR** to extract text from visual pitch deck slides, and browser automation to pull founder professional history from LinkedIn. It synthesizes these raw inputs into a structured "Source of Truth" summary that informs all subsequent agents.

### 2. Market Research Agent
The **Market Research Agent** acts as an industry landscape specialist. It infers the startup's sector and triggers **Tavily AI Search** to gather real-time market data, including TAM, SAM, SOM estimates, and CAGR growth rates. This agent identifies market entry barriers and emerging trends to assess whether the opportunity supports venture-scale returns.

### 3. Competitor Analysis Agent
The **Competitor Analysis Agent** performs real-time competitive intelligence. It identifies direct and indirect competitors by searching through current news and startup databases. The agent calculates a **Competition Intensity Score (1-10)** and maps out the startup's unique moats, helping investors understand the "crowdedness" of the space.

### 4. Financial Estimation Agent
The **Financial Estimation Agent** functions like a forensic accountant. It is strictly constrained to avoid data fabrication, instead deriving ARR/MRR by cross-referencing website pricing with pitch deck traction signals. It estimates **Burn Rate** based on team size and regional norms to calculate a realistic **Runway** and valuation range.

### 5. Risk Analysis Agent
The **Risk Analysis Agent** serves as the professional "Devil's Advocate." It executes targeted risk searches to identify potential "showstoppers" across six dimensions: Market, Technology, Execution, Regulatory, Financial, and Competition. It highlights **Major Red Flags** and produces a risk heatmap to guard against oversight.

### 6. Growth Prediction Agent
The **Growth Prediction Agent** handles future-state forecasting. By analyzing historical comparable trajectories and current market tailwinds, it estimates a **Success Probability (0-100%)**. This agent projects the startup's valuation over a 3-year horizon, providing a data-backed vision of the company's long-term potential.

### 7. Investment Decision Agent
The **Investment Decision Agent** acts as the final "Managing Partner." It does not perform new searches; instead, it synthesizes the collective intelligence of the previous six agents. By applying a **weighted scoring model** (Market, Risk, Growth, Competition, Finance, Team), it issues a final recommendation: **INVEST**, **WATCH**, or **REJECT**.

---

## 📂 Project Structure

```text
Project-Root/
├── agents/             # specialized AI agents logic
│   ├── Startup_Data_Agent.py
│   ├── Market_Research_Agent.py
│   ├── Risk_Analysis_Agent.py
│   └── ... 
├── frontend/           # Next.js web application
├── state/              # LangGraph state definitions
├── uploads/            # Temporary storage for pitch decks
├── graph.py            # Workflow orchestration logic
├── main.py             # CLI Entry point
└── main2.py            # FastAPI Backend Entry point
```

---

## 🚦 Getting Started

### Prerequisites

1. **Groq API Key**: Obtain a key from [Groq Console](https://console.groq.com/).
2. **Tesseract OCR**: 
    - Windows: [Install via UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki).
    - Mac: `brew install tesseract`
3. **Poppler**:
    - Mac: `brew install poppler`
    - Windows: [Download binaries](http://blog.alivate.com.au/poppler-windows/) and add to PATH.

### Backend Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/jaikumar2406/startup-intelligence.git
   cd startup-intelligence
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   # Ensure playwright is initialized
   playwright install chromium
   ```

3. **Configure Environment**:
   Create a `.env` file in the root:
   ```env
   agent2_llm=your_groq_api_key_here
   ```

4. **Run the API**:
   ```bash
   uvicorn main2:app --reload
   ```

### Frontend Setup

1. **Navigate to frontend**:
   ```bash
   cd frontend
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Run the development server**:
   ```bash
   npm run dev
   ```

---

## 📖 Usage

### Web Interface
1. Open `http://localhost:3000` in your browser.
2. Enter the Startup Name, Website, and Founder LinkedIn.
3. Upload the Pitch Deck (PDF).
4. Monitor the live analysis as agents process the data.

### CLI Mode
For a quick terminal-based report:
```bash
python main.py
```

---

## ⚖️ License

[MIT License](LICENSE)
