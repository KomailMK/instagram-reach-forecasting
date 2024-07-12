import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.sarimax import SARIMAXResults

# Load the dataset
file_path = 'Instagram-Reach.csv'  # Ensure this path is correct
data = pd.read_csv(file_path)

# Check for null values
print("Null values in the dataset:")
print(data.isnull().sum())

# Display column info
print("\nColumn information:")
print(data.info())

# Descriptive statistics
print("\nDescriptive statistics:")
print(data.describe())

# Convert the Date column to datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Line chart of Instagram reach over time
plt.figure(figsize=(12, 6))
plt.plot(data['Date'], data['Instagram reach'], label='Instagram Reach')
plt.xlabel('Date')
plt.ylabel('Instagram Reach')
plt.title('Trend of Instagram Reach Over Time')
plt.legend()
plt.grid(True)
plt.show()

# Bar chart of Instagram reach for each day
plt.figure(figsize=(12, 6))
plt.bar(data['Date'], data['Instagram reach'])
plt.xlabel('Date')
plt.ylabel('Instagram Reach')
plt.title('Instagram Reach for Each Day')
plt.xticks(rotation=90)
plt.grid(True)
plt.show()

# Box plot of Instagram reach
plt.figure(figsize=(8, 6))
plt.boxplot(data['Instagram reach'])
plt.ylabel('Instagram Reach')
plt.title('Distribution of Instagram Reach')
plt.grid(True)
plt.show()

# Create a day column
data['Day'] = data['Date'].dt.day_name()

# Group by the Day column and calculate statistics
day_stats = data.groupby('Day')['Instagram reach'].agg(['mean', 'median', 'std']).reindex(
    ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

print("\nInstagram Reach statistics by day of the week:")
print(day_stats)

# Bar chart to visualize the reach for each day of the week
plt.figure(figsize=(12, 6))
plt.bar(day_stats.index, day_stats['mean'], yerr=day_stats['std'], capsize=5)
plt.xlabel('Day of the Week')
plt.ylabel('Mean Instagram Reach')
plt.title('Mean Instagram Reach by Day of the Week')
plt.grid(True)
plt.show()

# Decompose the time series to observe trend and seasonality
decomposition = sm.tsa.seasonal_decompose(data.set_index('Date')['Instagram reach'], model='additive')
fig = decomposition.plot()
fig.set_size_inches(14, 8)
plt.show()

# Determine the order of differencing (d) by observing the autocorrelation plot
sm.graphics.tsa.plot_acf(data['Instagram reach'])
plt.show()

# Determine the order of AR (p) and MA (q) by observing the partial autocorrelation plot
sm.graphics.tsa.plot_pacf(data['Instagram reach'])
plt.show()

# Fit the SARIMA model
p = 1  # determined from PACF plot
d = 1  # generally starting with 1 for differencing
q = 1  # determined from ACF plot
seasonal_p = 1
seasonal_d = 1
seasonal_q = 1
seasonal_period = 12  # Assuming monthly seasonality for simplicity

model = SARIMAX(data['Instagram reach'], order=(p, d, q), seasonal_order=(seasonal_p, seasonal_d, seasonal_q, seasonal_period))
results = model.fit()

# Print model summary
print("\nModel Summary:")
print(results.summary())

# Save the model to a file
results.save('sarima_model.pkl')

# Make predictions
forecast_steps = 30  # Number of periods to forecast
forecast = results.get_forecast(steps=forecast_steps)
forecast_index = pd.date_range(start=data['Date'].iloc[-1], periods=forecast_steps+1, freq='D')[1:]
forecast_series = pd.Series(forecast.predicted_mean.values, index=forecast_index)

# Print the predictions
print("\nForecasted Reach for the Next 30 Days:")
print(forecast_series)

# Plot the actual data and the forecast
plt.figure(figsize=(12, 6))
plt.plot(data['Date'], data['Instagram reach'], label='Actual Reach')
plt.plot(forecast_series, label='Forecasted Reach', color='red')
plt.xlabel('Date')
plt.ylabel('Instagram Reach')
plt.title('Instagram Reach Forecast')
plt.legend()
plt.grid(True)
plt.show()

# Load the model from a file
loaded_model = SARIMAXResults.load('sarima_model.pkl')

# Use the loaded model to make predictions
forecast_loaded = loaded_model.get_forecast(steps=30)
forecast_loaded_index = pd.date_range(start=data['Date'].iloc[-1], periods=31, freq='D')[1:]
forecast_loaded_series = pd.Series(forecast_loaded.predicted_mean.values, index=forecast_loaded_index)

# Print the predictions from the loaded model
print("\nForecasted Reach for the Next 30 Days (Loaded Model):")
print(forecast_loaded_series)

# Plot the actual data and the forecast from the loaded model
plt.figure(figsize=(12, 6))
plt.plot(data['Date'], data['Instagram reach'], label='Actual Reach')
plt.plot(forecast_loaded_series, label='Forecasted Reach (Loaded Model)', color='green')
plt.xlabel('Date')
plt.ylabel('Instagram Reach')
plt.title('Instagram Reach Forecast (Loaded Model)')
plt.legend()
plt.grid(True)
plt.show()
