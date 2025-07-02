

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import datetime

ticker_symbol = "RELIANCE.NS"  
start_date = "end_date -"
end_date = "datetime.date.today()"
future_days = 30

print(f"Downloading data for {ticker_symbol}...")
data = yf.download(ticker_symbol, start=start_date, end=end_date)

if data.empty:
    print("No data found. Please check the ticker symbol.")
    exit()

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

y_pred = model.predict(X)
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

plt.figure(figsize=(12, 6))
plt.plot(data['Date'], y, label="Actual Prices", color='black')
plt.plot(future_dates, predicted_prices, label="Predicted Prices", color='red', linestyle='dashed')
plt.title(f"{ticker_symbol} Stock Price Prediction")
plt.xlabel("Date")
plt.ylabel("Stock Price (INR)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("stock_prediction_plot.png")
plt.show()

print(f"Model Performance for {ticker_symbol}:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.4f}")
