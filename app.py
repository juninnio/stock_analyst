from flask import Flask, request, jsonify, render_template
from stock_analyst import Fundamental_Analyst, TechnicalAnalysis
from google import genai
import re


app = Flask(__name__)

API_KEY = "alphavantage_api_key"
GEMINI_API_KEY = "gemini_api_key"

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

        response = f"## Analysis for {message}:\n\n"
        for key, value in metrics.items():
            response += f"- {key}: {round(value, 2) if isinstance(value, float) else value}\n"
        response += f"\n### Predicted next close price: **{round(prediction, 2)}**\n\n"


        financials = fundamentals.get_financials()

        prompt=f"""You are an experienced financial analyst.
        Based on the following financial data and last year's metrics, provide a thorough investment analysis for {message}:
        Metrics: {metrics}
        Financial Data: {financials}

        Focus on financial healths, risks, growth potential, and long-term investment advice.
        At the end, give conclusions and signal whether it's Strong Buy, Buy, Neutral, Sell, or Strong Sell
        """

        llm_response = client.models.generate_content(
            model = "gemini-2.0-flash",
            contents=[prompt]
        )

        llm_reply = llm_response.text
        response += llm_reply




    except Exception as e:
        response = f"Couldn't process symbol `{message}`.\nError: {str(e)}"

    return jsonify({"reply": response})
