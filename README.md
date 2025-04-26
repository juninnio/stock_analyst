# AI-Powered Stock Analysis Assistant

A chat-based stock analysis assistant that combines real-time financial data, automated metric calculation, and Large Language Model (LLM) insights to help users understand a company's financial health, performance, and investment potential.

---

## Overview

This project integrates traditional financial analysis with modern AI models to simulate the experience of having a personal financial analyst on demand.

### Workflow:

1. User submits a stock symbol (e.g., `AAPL`, `MSFT`) via chat interface
2. Backend fetches the company's **5-year financial history** (Income Statement, Balance Sheet, Cash Flow) via Alpha Vantage API
3. Generates interactive visualizations:
    - Historical price movements (Candlestick chart)
    - Revenue vs Net Income trends
    - Profit Margins analysis (Gross, Operating, Net margins)
4. Computes key financial metrics including:
    - Profitability Ratios: Gross Margin, Net Margin, Operating Margin
    - Liquidity Ratios: Current Ratio, Quick Ratio
    - Solvency Ratios: Debt-to-Equity, Debt-to-Assets
    - Valuation Multiples: P/E, P/B, P/S, Dividend Yield
    - Return Metrics: ROA, ROE
5. Predicts next-day closing price using an RNN-LSTM model
6. Generates AI-driven investment analysis using Google Gemini LLM, including:
    - Technical analysis of price movements
    - Financial health assessment
    - Risk analysis
    - Growth potential evaluation
    - Investment recommendation (Strong Buy to Strong Sell)

---

## ⚙️ Tech Stack

| Layer             | Technology                         |
|-------------------|-------------------------------------|
| Backend           | Python, Flask                      |
| Data Processing   | Pandas, Numpy, Alpha Vantage API   |
| Visualization     | Plotly                            |
| Machine Learning  | TensorFlow, Sklearn (RNN-LSTM)    |
| LLM Integration   | Google Gemini API                  |
| Frontend          | HTML, CSS, JavaScript, Marked.js   |

---

## Setup Instructions

1. Clone the repository
2. Install dependencies
3. Set up API keys:
   - Get an Alpha Vantage API key from [Alpha Vantage](https://www.alphavantage.co/)
   - Get a Google Gemini API key from [Google AI Studio](https://ai.google.dev/)
4. Configure API keys:
   - Create a `.env` file or
   - Update the API keys in `app.py`
5. Start the Flask server
6. Open your browser and navigate to `http://localhost:5000`

## Features

1. **Interactive Chat Interface**
   - Clean, modern design
   - Real-time response generation
   - Markdown support for formatted output

2. **Comprehensive Financial Analysis**
   - Real-time financial data fetching
   - Automated metric calculations
   - Interactive Plotly visualizations
   - Next-day price predictions
   - AI-generated investment insights

3. **Visual Analytics**
   - Candlestick charts for price analysis
   - Revenue and profitability trends
   - Margin analysis over time

## Future Enhancements

- Multi-stock comparison analysis
- Integration of financial news and sentiment analysis
- RAG pipelines
- Docker containerization for easy deployment

## Contributing

If you'd like to contribute or have suggestions for improvements, please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

For other inquiries, connect with me on [LinkedIn](https://www.linkedin.com/in/juninnio-harris/)
