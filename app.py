from flask import Flask, request, jsonify, render_template
from stock_analyst import Fundamental_Analyst, TechnicalAnalysis
from google import genai
import re


app = Flask(__name__)

API_KEY = "alphavantage_api_key"
GEMINI_API_KEY ="gemini_api_key"

client = genai.Client(api_key=GEMINI_API_KEY)


@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    message = data.get("message", "").upper().strip()

    try:
        fundamentals = Fundamental_Analyst(message, API_KEY)
        technicals = TechnicalAnalysis(message, API_KEY)

        metrics = fundamentals.get_metrics()
        prediction = technicals.get_prediction()

        metrics['Year'] = metrics['Year'].astype(int)
        print(metrics['Year'])

        historical_plot_html = technicals.make_historical_plot()

        response = f"## Analysis for {message}:\n\n"
        for key, value in metrics.items():
            if key not in ['Gross Profit Margin', 'Operating Profit Margin', 'Net Profit Margin','Return on Assets','Return on Equity']:
                response += f"- {key}: {round(value, 2) if (isinstance(value, float)) else value}\n"
            else:
                response += f"- {key}: {round(value, 2)}% \n"


        response += f"\n### Predicted next close price: **{round(prediction, 2)}**\n\n"

        revenue_plot = fundamentals.make_line_plot('Year',['Revenue','Net Income'],'Revenue vs Net Income','Year','Amount')

        margins_plot = fundamentals.make_line_plot('Year',['Gross Profit Margin','Operating Profit Margin','Net Profit Margin'],'Profit Margins vs Year','Year','Profit Margins')
    
        financials = fundamentals.get_financials()

        prompt=f"""You are an experienced financial analyst.
        Based on the following financial data and last year's metrics, provide a thorough investment analysis for {message}:
        Metrics: {metrics}
        Financial Data: {financials}
        This candlestick chart shows the historical price movement of {message}: {historical_plot_html}.
        These plots will also be included at the bottom of your response which you also mention in your analysis: {revenue_plot, margins_plot}.

        Focus on financial healths, risks, growth potential, and long-term investment advice.
        First provide an analysis of the candlestick chart. Then, give a seamless analysis of the company based on the data provided and only use the plot as a reference.
        At the end, give conclusions and signal whether it's Strong Buy, Buy, Neutral, Sell, or Strong Sell
        """

        llm_response = client.models.generate_content(
            model = "gemini-2.0-flash",
            contents=[prompt]
        )

        llm_reply = llm_response.text

        return jsonify({
            "reply": response or "No analysis available.",
            "plot": revenue_plot or "",
            "candlestick": historical_plot_html or "",
            "llm": llm_reply or "",
            "margins_plot": margins_plot or ""
        })


    except Exception as e:
        return jsonify({
            "reply": f"Couldn't process symbol `{message}`.\nError: {str(e)}",
            "plot": "",
            "candlestick": "",
            "llm": "",
            "margins_plot": ""
        })

