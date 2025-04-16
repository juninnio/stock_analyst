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



class Fundamental_Analyst:
    def  __init__(self, symbol, api_key):
        self.API_KEY = api_key
        self.symbol = symbol
        self.base_url = "https://www.alphavantage.co/query"
        self.income_statement = self.get_income_statement(symbol)
        self.balance_sheet = self.get_balance_sheet(symbol)
        self.cash_flow = self.get_cash_flow(symbol)
        self.overview = self.get_overview(symbol)


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
        
        income_statement = self.income_statement['annualReports'][0]
        last_balance_sheet = self.balance_sheet['annualReports'][1]
        balance_sheet = self.balance_sheet['annualReports'][0]
        overview = self.overview



        """if len(annual_reports) >=10:
            financials = [annual_reports[i] for i in range(10)]
        elif 5<= len(annual_reports) < 10:
            financials = [annual_reports[i] for i in range(5)]
        else:
            financials = [annual_reports[i] for i in range(len(annual_reports))]"""
        
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

        metrics ={
            "Gross Profit Margin": f"{gross_profit_margin*100:.2f}%",
            "Operating Profit Margin":f"{operating_profit_margin*100:.2f}%",
            "Net Profit Margin":f"{net_profit_margin*100:.2f}%",
            "Return on Assets":f"{roa*100:.2f}%",
            "Return on Equity":f"{roe*100:.2f}%",
            "Current Ratio":curr_ratio,
            "Quick Ratio":quick_ratio,
            "Debt-to-Equity Ratio": de_ratio,
            "Debt-to-Assets Ratio": da_ratio,
            "P/E": pe,
            "P/B Ratio":pb,
            "P/S Ratio":ps,
            "Dividend Yield":dividend_yield*100
        }

        return metrics
        
    def get_metrics(self):
        return self.metrics()
    
    def get_financials(self):
        report_key = "annualReports"
        balance_sheet, cash_flow, income_statement = [],[],[]
        for i in range(5):
            balance_sheet.append(self.balance_sheet[report_key][i])
            cash_flow.append(self.cash_flow[report_key][i])
            income_statement.append(self.income_statement[report_key][i])
        return {'Company Overview': self.overview, "Balance Sheet":balance_sheet, "Cash Flow":cash_flow,
                'Income Statement': income_statement}


    def __str__(self):
        return f"{self.get_metrics()}"


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
        


    def get_data(self):
        """
        get data and add technical indicators
        """
        stock_url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={self.symbol}&datatype=csv&outputsize=full&apikey={self.API_KEY}"
        stock_data = pd.read_csv(stock_url)
        stock_data['timestamp'] = pd.to_datetime(stock_data['timestamp'])
        stock_data = stock_data[::-1].reset_index(drop=True)

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
    

    def __str__(self):
        return(f"""Test MSE (loss): {self.test_loss:.4f}
Test MAE: {self.test_mae:.4f}
Predicted next close price: {self.next_close_pred}""")

