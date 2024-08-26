import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

import warnings
warnings.filterwarnings('ignore')

# DATA LOADING AND EXPLORATION
# Load the stock price data
df = pd.read_csv('tesla stock/data/tesla-stock-historical-data.csv', parse_dates=['Date'], index_col='Date')
print(df.head())

# Exploratory analysis
print(df.info())
print(df.describe())

# Plot the closing price
plt.figure(figsize=(10, 6))
plt.plot(df['Close'])
plt.title('Tesla Stock Closing Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

# TIME SERIES ANALYSIS
# check for stationarity
from statsmodels.tsa.stattools import adfuller

def test_stationarity(timeseries):
    # Determine rolling statistics
    rolmean = timeseries.rolling(window=12).mean()
    rolstd = timeseries.rolling(window=12).std()

    # Plot rolling statistics
    plt.figure(figsize=(10, 6))
    plt.plot(timeseries, color='purple', label='Original')
    plt.plot(rolmean, color='red', label='Rolling Mean')
    plt.plot(rolstd, color='black', label='Reolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)

    # Perform Augmented Dickey-Fuller Test
    print('Results of Augmented Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Obsevations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)

# Apply the test
test_stationarity(df['Close'])

# DATA PROCESSING
# Differencing to make the series stationary
df['Close_diff'] = df['Close'] - df['Close'].shift(1)
df['Close_diff'].dropna(inplace=True)

# Check stationarity after differencing
test_stationarity(df['Close_diff'].dropna())

# ARIMA MODEL DEVELOPMENT
from statsmodels.tsa.arima.model import ARIMA

# ARIMA model
model = ARIMA(df['Close_diff'].dropna(), order=(1, 1, 1))
results_ARIMA = model.fit()

# Plot the predictions on top of the actual data
plt.figure(figsize=(10, 6))
plt.plot(df['Close_diff'].dropna(), label='Original')
plt.plot(results_ARIMA.fittedvalues, color='red', label='Fitted')
plt.title('ARIMA Model - Fitted vs Original')
plt.legend()
plt.show()

# MODEL EVALUATION
# Forecasting the next steps
forecast_results = results_ARIMA.get_forecast(steps=30) #get_forecast returns a 'PredictionResults' object

# Extracting the forecasted mean, standard error, and confidence intervals
forecast = forecast_results.predicted_mean #predicted_mean extracts the forecasted values (mean)
conf_int = forecast_results.conf_int()

plt.figure(figsize=(10, 6))
plt.plot(df['Close'], label='Historical')
plt.plot(forecast, label='Forecasted')
plt.title('Stock Price Forecast')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()