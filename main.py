from joblib import load
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

# Load the model with the correct path
model = load("prophet_sales_predict.joblib")  # Ensure the path is correct
print("Model loaded successfully")

# Get user input for the number of months to forecast
input_months = int(input("Enter the number of months you want: "))

# Create a future dataframe for forecasting
future = model.make_future_dataframe(periods=input_months, freq='M')

# Generate forecasts
forecast = model.predict(future)

# Filter the forecasted data for the future period
future_forecast = forecast[forecast['ds'] > future['ds'].max() - pd.Timedelta(days=30*input_months)]

# Display the forecasted data
print("Forecasted Data:")
print(future_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])

# Plot the forecast
fig, ax = plt.subplots(figsize=(12, 6))

# Plot the forecasted line
ax.plot(
    future_forecast['ds'],
    future_forecast['yhat'],
    color='blue',
    linewidth=2,
    label='Forecast'
)

# Highlight confidence intervals
ax.fill_between(
    future_forecast['ds'],
    future_forecast['yhat_lower'],
    future_forecast['yhat_upper'],
    color='skyblue',
    alpha=0.3,
    label='Confidence Interval'
)

# Add labels, legend, and grid
ax.set_title(f'Forecast for Next {input_months} Months', fontsize=16)
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Sales', fontsize=12)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

# Show the plot
plt.show()