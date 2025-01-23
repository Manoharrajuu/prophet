import streamlit as st
from joblib import load
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from datetime import datetime

# Streamlit app title
st.title("Sales Forecasting with Prophet")

# Custom styling
st.markdown(
    """
    <style>
    .stButton button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        padding: 10px 24px;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load the model with a spinner
with st.spinner("Loading model..."):
    try:
        model = load("prophet_sales_model.joblib")
        st.sidebar.success("Model loaded successfully.")
    except Exception as e:
        st.sidebar.error(f"Error loading the model: {e}")

# Show historical data if the model is loaded successfully
if 'model' in locals():
    if hasattr(model, 'history'):
        st.subheader("Historical Sales Data")
        st.write(model.history[['ds', 'y']])

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.lineplot(x=model.history['ds'], y=model.history['y'], color='black', label='Historical Sales', ax=ax)
        ax.set_title('Historical Sales Data', fontsize=16)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Sales', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        st.pyplot(fig)

# Date range selector
start_date = st.date_input("Select start date for forecast", datetime.now())
end_date = st.date_input("Select end date for forecast", datetime.now() + pd.Timedelta(days=365))

# Generate forecast
if st.button("Generate Forecast"):
    if 'model' in locals():
        with st.spinner("Generating forecast..."):
            future = model.make_future_dataframe(periods=(end_date - start_date).days, freq='D')
            forecast = model.predict(future)
            future_forecast = forecast[(forecast['ds'] >= pd.to_datetime(start_date)) & (forecast['ds'] <= pd.to_datetime(end_date))]
        st.success("Forecast generated successfully!")

        # Display forecasted data
        st.subheader("Forecasted Data:")
        st.write(future_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])

        # Plot forecast
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.lineplot(x=future_forecast['ds'], y=future_forecast['yhat'], color='blue', label='Forecast', ax=ax)
        ax.fill_between(future_forecast['ds'], future_forecast['yhat_lower'], future_forecast['yhat_upper'], color='skyblue', alpha=0.3, label='Confidence Interval')
        ax.set_title(f'Forecast from {start_date} to {end_date}', fontsize=16)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Sales', fontsize=12)
        ax.legend(fontsize=10)
        st.pyplot(fig)

        # Download forecasted data
        csv = future_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(index=False)
        st.download_button(
            label="Download Forecasted Data as CSV",
            data=csv,
            file_name="forecasted_sales.csv",
            mime="text/csv"
        )
    else:
        st.error("No model loaded. Please ensure the model file is available.")