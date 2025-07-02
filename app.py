

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import datetime

st.set_page_config(page_title="Stock Price Predictor", layout="centered")

st.title("📈 Stock Price Prediction App")
st.write("This app predicts future stock prices using historical data and Linear Regression.")

ticker_symbol = st.text_input("Enter Stock Ticker Symbol (e.g., RELIANCE.NS, AAPL):", "RELIANCE.NS")

start_date = st.date_input("Start Date", datetime.date(2018, 1, 1))
end_date = st.date_input("End Date", datetime.date.today())
future_days = st.slider("Days to Predict", 7, 60, 30)

if st.button("Predict"):

    st.info(f"Downloading data for {ticker_symbol}...")
    data = yf.download(ticker_symbol, start=start_date, end=end_date)

    if data.empty:
        st.error("No data found. Please check the ticker symbol.")
    else:
        stock = yf.Ticker(ticker_symbol)
        end_price = stock.history(period="1d")["Close"].iloc[-1]
        st.success(f"📈 Latest Closing Price: ₹{end_price:.2f}")

        stock = yf.Ticker(ticker_symbol)
        start_price = stock.history(period="1d")["Open"].iloc[-1]
        st.success(f"📈 Latest starting Price: ₹{start_price:.2f}")

        stock = yf.Ticker(ticker_symbol)
        letest_volume = stock.history(period="1d")["Volume"].iloc[-1]
        st.info(f"📈 Latest volume: ₹{letest_volume:.2f}")






        
        data.reset_index(inplace=True)
        data['Date_Ordinal'] = pd.to_datetime(data['Date']).map(datetime.datetime.toordinal)

        X = np.array(data['Date_Ordinal']).reshape(-1, 1)
        y = np.array(data['Close'])

        model = LinearRegression()
        model.fit(X, y)

        last_date = data['Date'].max()
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_days)
        future_ordinals = future_dates.map(datetime.datetime.toordinal).to_numpy().reshape(-1, 1)
        predicted_prices = model.predict(future_ordinals)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(data['Date'], y, label="Actual Prices", color='blue')
        ax.plot(future_dates, predicted_prices, label="Predicted Prices", color='red', linestyle='dashed')
        ax.set_title(f"{ticker_symbol} Stock Price Prediction")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        from sklearn.metrics import mean_squared_error, r2_score
        y_pred = model.predict(X)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        st.success(f"Model Performance: R2 score={r2:.2f}")

