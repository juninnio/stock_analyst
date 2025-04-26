import requests
import pandas as pd
import pandas_ta as ta
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import plotly.graph_objects as go
import plotly.express as px


class Fundamental_Analyst:
    def  __init__(self, symbol, api_key):
        self.API_KEY = api_key
        self.symbol = symbol
        self.base_url = "https://www.alphavantage.co/query"
        self.income_statement = self.get_income_statement(symbol)
        self.balance_sheet = self.get_balance_sheet(symbol)
        self.cash_flow = self.get_cash_flow(symbol)
        self.overview = self.get_overview(symbol)
        self.financial_data = self.metrics()


    def get_income_statement(self, symbol):
        params = {
            "function" : "INCOME_STATEMENT",
            "symbol" : symbol,
            "apikey" : self.API_KEY
        }

        response = requests.get(self.base_url, params=params)
        data = response.json()

        return data
    
    def get_balance_sheet(self, symbol):
        params = {
            "function" : "BALANCE_SHEET",
            "symbol" : symbol,
            "apikey" : self.API_KEY
        }

        response = requests.get(self.base_url, params=params)
        data = response.json()

        return data
    
    def get_cash_flow(self, symbol):
        params = {
            "function" : "CASH_FLOW",
            "symbol" : symbol,
            "apikey" : self.API_KEY
        }

        response = requests.get(self.base_url, params=params)
        data = response.json()

        return data
    
    def get_overview(self, symbol):
        params = {
            "function" : "OVERVIEW",
            "symbol" : symbol,
            "apikey" : self.API_KEY
        }

        response = requests.get(self.base_url, params=params)
        data = response.json()

        return data

    
    def metrics(self):
        
        if len(self.income_statement['annualReports']) < 6:
            r = len(self.income_statement['annualReports'])
        else:
            r = 5

        data ={
                "Year":[],
                "Revenue":[],
                "Net Income":[],
                "Gross Profit Margin":[],
                "Operating Profit Margin":[],
                "Net Profit Margin":[],
                "Return on Assets":[],
                "Return on Equity":[],
                "Current Ratio":[],
                "Quick Ratio":[],
                "Debt-to-Equity Ratio": [],
                "Debt-to-Assets Ratio": [],
                "P/E Ratio": [],
                "P/B Ratio":[],
                "P/S Ratio":[],
                "Dividend Yield":[],
                "EPS": [],
            }

        for i in range(r):
            income_statement = self.income_statement['annualReports'][i]
            last_balance_sheet = self.balance_sheet['annualReports'][i+1]
            balance_sheet = self.balance_sheet['annualReports'][i]
            overview = self.overview

            revenue = float(income_statement['totalRevenue'])
            year = pd.to_datetime(income_statement['fiscalDateEnding']).year
            net_income = float(income_statement['netIncome'])

        
            #profitability
            gross_profit_margin = (float(income_statement['grossProfit']))/float(income_statement['totalRevenue'])

            operating_profit_margin = (float(income_statement['operatingIncome']))/float(income_statement['totalRevenue'])

            net_profit_margin = (float(income_statement['netIncome']))/float(income_statement['totalRevenue'])

            roa = (float(income_statement['netIncome']))/(float(balance_sheet['totalAssets'])+float(last_balance_sheet['totalAssets']))/2

            roe = (float(income_statement['netIncome']))/((float(balance_sheet['totalShareholderEquity'])+float(last_balance_sheet['totalShareholderEquity'])))/2

            #liquidity 
            curr_ratio = float(balance_sheet['totalCurrentAssets'])/float(last_balance_sheet['totalCurrentLiabilities'])

            cash = float(balance_sheet['cashAndCashEquivalentsAtCarryingValue'])
            short_term_investments = float(balance_sheet['shortTermInvestments'])
            accounts_receivable = float(balance_sheet['currentNetReceivables']) if balance_sheet['currentNetReceivables'] != 'None' else 0

            quick_ratio = (cash +short_term_investments +accounts_receivable)/float(balance_sheet['totalCurrentLiabilities'])

            #solvency
            de_ratio = float(balance_sheet['totalLiabilities'])/float(balance_sheet['totalShareholderEquity'])
            da_ratio = float(balance_sheet['totalLiabilities'])/float(balance_sheet['totalAssets'])

            #valuation
            pe = float(overview['PERatio'])
            pb = float(overview['PriceToBookRatio'])
            ps = float(overview['PriceToSalesRatioTTM'])
            dividend_yield = float(overview['DividendYield']) if overview['DividendYield'] != "None" else 0

            eps = float(income_statement['netIncome'])/float(balance_sheet['commonStockSharesOutstanding'])

            data['Revenue'].append(revenue)
            data['Year'].append(year)
            data['Net Income'].append(net_income)
            data["Gross Profit Margin"].append(gross_profit_margin*100)
            data["Operating Profit Margin"].append(operating_profit_margin*100)
            data["Net Profit Margin"].append(net_profit_margin*100)
            data["Return on Assets"].append(roa*100)
            data["Return on Equity"].append(roe*100)
            data["Current Ratio"].append(curr_ratio)
            data["Quick Ratio"].append(quick_ratio)
            data["Debt-to-Equity Ratio"].append(de_ratio)
            data["Debt-to-Assets Ratio"].append(da_ratio)
            data["P/E Ratio"].append(pe)
            data["P/B Ratio"].append(pb)
            data["P/S Ratio"].append(ps)
            data["Dividend Yield"].append(dividend_yield*100)
            data["EPS"].append(round(eps,2))


        df = pd.DataFrame(data=data)
        return df
        
    def get_metrics(self):
        return dict(self.financial_data.loc[0])
    
    def get_financials(self):
        report_key = "annualReports"
        balance_sheet, cash_flow, income_statement = [],[],[]
        for i in range(5):
            balance_sheet.append(self.balance_sheet[report_key][i])
            cash_flow.append(self.cash_flow[report_key][i])
            income_statement.append(self.income_statement[report_key][i])
        return {'Company Overview': self.overview, "Balance Sheet":balance_sheet, "Cash Flow":cash_flow,
                'Income Statement': income_statement}

    
    def make_line_plot(self, x_var, y_var, title, x_title=None, y_title=None):
        #Revenue vs Net Profit
        fig1 = px.line(self.financial_data, x=x_var, y=y_var)
        fig1.update_layout(
            title=title,
            xaxis=dict(
              tickmode='array',
              tickvals= self.financial_data['Year'].values  
            ),
            xaxis_title = f"{x_var if not x_title else x_title}",
            yaxis_title = f"{y_var if not y_title else y_title}",
            template = 'plotly_dark',
            height=500,
            width=1000,
            margin=dict(l=50, r=50, b=50, t=50, pad=4)
        )

        html_str = fig1.to_html(
            full_html=False,
            include_plotlyjs = False,
            config = {'responsive': True}
        )

        return html_str

        

    def __str__(self):
        return f"{self.financial_data}"


class TechnicalAnalysis:
    """
    Call to get technical analysis and future prediction of a stock
    1. Get dataframe
    2. Construct Technical Indicators
    3. Preprocessing
    4. Train Model
    5. Get Prediction
    """
    def __init__(self, symbol, api_key):
        self.API_KEY = api_key
        self.symbol = symbol
        self.data = self.get_data()
        self.scaled_data, self.scaler = self.scale_data(self.data)
        self.model, self.test_loss , self.test_mae = self.train_test_model()
        self.next_close_pred = self.get_prediction()
    

    def get_splits(self):
        split_url = f"https://www.alphavantage.co/query?function=SPLITS&symbol={self.symbol}&apikey={self.API_KEY}"
        r = requests.get(split_url)
        split_data = r.json()
        splits = split_data['data'][::-1]
        return splits
        

    def get_data(self):
        """
        get data and add technical indicators
        """
        stock_url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={self.symbol}&datatype=csv&outputsize=full&apikey={self.API_KEY}"
        stock_data = pd.read_csv(stock_url)
        stock_data['timestamp'] = pd.to_datetime(stock_data['timestamp'])
        stock_data = stock_data[::-1].reset_index(drop=True)

        splits = self.get_splits()
        for split in splits:
            eff_date = pd.to_datetime(split['effective_date'])
            factor = float(split['split_factor'])

            temp =stock_data['timestamp'] < eff_date
            stock_data.loc[temp, ['open','high','low','close']] = stock_data.loc[temp, ['open','high','low','close']] / factor 
            stock_data.loc[temp, 'volume'] = stock_data.loc[temp,'volume'] * factor


        stock_data["SMA50"] = ta.sma(stock_data['close'], length=50)
        stock_data["SMA100"] = ta.sma(stock_data['close'], length=100)
        stock_data["SMA200"] = ta.sma(stock_data['close'], length=200)

        stock_data['EMA50'] = ta.ema(stock_data['close'],length=50)
        stock_data['EMA100'] = ta.ema(stock_data['close'],length=100)
        stock_data['EMA200'] = ta.ema(stock_data['close'],length=200)

        stock_data['RSI'] = ta.rsi(stock_data['close'], length=14)
        macd_df = ta.macd(stock_data['close'], fast=12, slow=26, signal=9)
        stock_data = pd.concat([stock_data, macd_df], axis=1)

        stock_data.rename(columns={
            'MACD_12_26_9': 'MACD',
            'MACDs_12_26_9': 'MACD_signal',
            'MACDh_12_26_9': 'MACD_histogram'
        }, inplace=True)


        bb = ta.bbands(stock_data['close'], length=20, std=2)
        stock_data['BB_upper'] = bb['BBU_20_2.0']
        stock_data['BB_middle'] = bb['BBM_20_2.0']
        stock_data['BB_lower'] = bb['BBL_20_2.0']

        stock_data['OBV'] = ta.obv(stock_data['close'], stock_data['volume'])

        # Bullish crossover: when 50-day SMA crosses above 200-day SMA
        stock_data['SMAcrossover'] = None
        stock_data.loc[(stock_data['SMA50'] > stock_data['SMA200']) & (stock_data['SMA50'].shift(1) < stock_data['SMA200'].shift(1)), 'SMAcrossover'] = 'Bullish'

        # Bearish crossover: when 50-day SMA crosses below 200-day SMA
        stock_data.loc[(stock_data['SMA50'] < stock_data['SMA200']) & (stock_data['SMA50'].shift(1) > stock_data['SMA200'].shift(1)), 'SMAcrossover'] = 'Bearish'

        #RSI signal for overbought/ oversold
        stock_data['RSI_signal'] = None
        stock_data.loc[stock_data['RSI'] > 70, 'RSI_signal'] = 'Overbought'
        stock_data.loc[stock_data['RSI'] < 30, 'RSI_signal'] = 'Oversold'

        return stock_data
    


    def scale_data(self, data):
        features = ['close', 'open','high','low','SMA50','SMA100', 'SMA200', 'EMA50', 'EMA100','EMA200', 'RSI', 'MACD', 'MACD_signal', 'BB_upper', 'BB_middle', 'BB_lower', 'OBV']
        scaled_df = data[features].dropna()
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(scaled_df)
        scaled_df = pd.DataFrame(scaled_data, columns=features)

        return scaled_df, scaler
    
    def create_dataset(self, data, timestep =90):
        X,y = [], []
        for i in range(timestep, len(data)):
            X.append(data[i-timestep:i])
            y.append(data[i][0])

        return np.array(X), np.array(y)
    
    def train_test_model(self):

        X,y = self.create_dataset(self.scaled_data.values)

        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.15, shuffle=False)

        input_shape = (X.shape[1],X.shape[2])
        model= Sequential([
            layers.LSTM(64, return_sequences = True, input_shape = input_shape),
            layers.LSTM(64, return_sequences = False),
            layers.Dropout(0.3),
            layers.Dense(32),
            layers.Dense(1)
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='mean_squared_error', metrics=['mae'])        

        early_stop = EarlyStopping(
            monitor='val_mae',   
            patience=10,          
            restore_best_weights=True,
            verbose=1
        )

        history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=100, batch_size=64, verbose=1, callbacks = [early_stop])

        test_loss, test_mae = model.evaluate(x_test, y_test)

        return model, test_loss, test_mae
    

    
    def get_prediction(self):
        last_sequence = self.scaled_data[-90:]  
        x = np.expand_dims(last_sequence, axis=0) 
        pred = self.model.predict(x)
        pred = self.scaler.inverse_transform(np.concatenate([pred, 
                                                        np.zeros((len(pred), 16))], axis=1))[:, 0]
        
        return pred[0]
    
    def make_historical_plot(self):
        data = self.data[self.data['timestamp'] > self.data['timestamp'].values[-1] - pd.to_timedelta(365*5, unit='d')]
        fig = go.Figure(data=[go.Candlestick(x=data['timestamp'],
                                    open=data['open'],
                                    high=data['high'],
                                    low=data['low'],
                                    close=data['close'])])
        fig.update_layout(
        title=f"{self.symbol} Stock Price - Last 5 Years",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        xaxis_rangeslider_visible=False,
        template="plotly_dark",  # Dark theme to match your UI
        height=500,
        width=1000,
        margin=dict(l=50, r=50, b=50, t=50, pad=4)
    )
    
        # Important: Return the HTML directly without using show()
        html_str = fig.to_html(
            full_html=False,
            include_plotlyjs=False,
            config={'responsive': True}
    )
        return html_str
    

    def __str__(self):
        return(f"""Test MSE (loss): {self.test_loss:.4f}
Test MAE: {self.test_mae:.4f}
Predicted next close price: {self.next_close_pred}""")
    
"""API_KEY = "EWFT2QD72ZJLZE32"

fd = Fundamental_Analyst('AAPL', API_KEY)
a = fd.get_metrics()
response = ""
for key,value in a.items():
           
    if key not in ['Gross Profit Margin', 'Operating Profit Margin', 'Net Profit Margin','Return on Assets','Return on Equity']:
        response += f"- {key}: {round(value, 2) if isinstance(value, float) else value}\n"
    else:
        response += f"- {key}: {round(value, 2)}% \n"

fd.make_line_plot('year',['Revenue','Net Income'],'Revenue vs Net Income','Year','Amount')"""