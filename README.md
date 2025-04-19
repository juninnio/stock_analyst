# AI-Powered Stock Analysis Assistant

A chat-based stock analysis assistant that combines real-time financial data, automated metric calculation, and Large Language Model (LLM) insights to help users understand a company's financial health, performance, and investment potential.

---

## Overview

This project integrates traditional financial analysis with modern AI models to simulate the experience of having a personal financial analyst on demand.

### Workflow:

1. User submits a stock symbol (e.g., `AAPL`, `MSFT`) via chat.
2. Backend fetches the company’s **5-year financial history** (Income Statement, Balance Sheet, Cash Flow) via Alpha Vantage API.
3. Computes key financial metrics including:
    - Profitability Ratios: Gross Margin, Net Margin, Operating Margin.
    - Liquidity Ratios: Current Ratio, Quick Ratio.
    - Solvency Ratios: Debt-to-Equity, Debt-to-Assets.
    - Valuation Multiples: P/E, P/B, Dividend Yield.
4. Passes the financial context to **Google Gemini LLM** for AI-driven investment-grade analysis.
5. Displays the LLM-generated insights in the chat interface.

---

## ⚙️ Tech Stack

| Layer             | Technology                         |
|-------------------|-------------------------------------|
| Backend           | Python, Flask                      |
| Data Processing   | Pandas, Numpy, Alpha Vantage API   |
| Machine Learning  | TensorFlow (Technical Prediction with RNN-LSTM)  |
| LLM Integration   | Google Gemini API                  |
| Frontend          | HTML, CSS, JavaScript (Chat UI)    |

---

## Setup Instructions
1. Clone repo
2. Install dependencies (pip install -r requirements.txt)
3. Get AlphaVantage API key and Gemini LLM API key
4. Create .env file for API keys or hard-code the API keys in stock_analyst.py
5. Start flask server
6. Open the link in your browser

## Features
1. Stock analysis with real-time data
2. Next-day price prediction with RNN-LSTM model
3. Automated metric calculation for in-depth financial health assessment
4. Investment advice and company evaluation with Google Gemini LLM API
5. Clean and simple UI for easy usage

## Future Enhancements
- Integrate real-time price plots using Plotly or Chart.js.
- Enable multi-stock comparisons.
- Build a Retrieval-Augmented Generation (RAG) pipeline using vector databases for large document analysis (e.g., 10-K filings, earnings transcripts).
- Integrate financial news and sentiment analysis to enrich model context.
- Package the project for deployment (Docker / Cloud Hosting).

## Contributing
If you have any ideas or improvement, feel free to contact me via [LinkedIn]([url](https://www.linkedin.com/in/juninnio-harris/))
