import os
import warnings
import json
from json.decoder import JSONDecodeError

# Suppress TensorFlow warnings BEFORE importing TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import streamlit as st
import datetime as dt
import yfinance as yf
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import time

# ===========================
# MODEL LOADING
# ===========================
@st.cache_resource
def load_prediction_model():
    """Load model once and cache it to avoid reloading"""
    try:
        # Try loading the first model name
        model = load_model('stockpricemodel.keras')
        return model
    except:
        try:
            # Try alternate model name
            model = load_model('stockpricrosp.keras')
            return model
        except Exception as e:
            st.error(f"❌ Failed to load model: {e}")
            st.error("Make sure 'stockpricemodel.keras' or 'stockpricrosp.keras' exists in the directory")
            st.stop()

# Load the trained model
model = load_prediction_model()

# ===========================
# STREAMLIT APP TITLE
# ===========================
st.title("Stock Price Prediction App")

# ===========================
# DATA DOWNLOAD FUNCTION WITH CACHING
# ===========================
@st.cache_data(ttl=3600, show_spinner=False)
def download_stock_data(stock_symbol, start_date, end_date, max_retries=3):
    """
    Download stock data with retry logic and rate limiting protection.
    Cached for 1 hour to prevent repeated API calls.
    
    Args:
        stock_symbol: Stock ticker symbol (e.g., 'NVDA')
        start_date: Start date for historical data
        end_date: End date for historical data
        max_retries: Maximum number of retry attempts
    
    Returns:
        tuple: (DataFrame, error_message) - DataFrame is None if error occurred
    """
    for attempt in range(max_retries):
        try:
            # Add exponential backoff delay for retries
            if attempt > 0:
                wait_time = 2 ** attempt  # 2, 4, 8 seconds
                time.sleep(wait_time)
            
            # Download stock data with minimal output
            df = yf.download(
                stock_symbol, 
                start=start_date, 
                end=end_date, 
                progress=False,  # Disable progress bar to avoid multiple bars
                ignore_tz=True   # Ignore timezone to avoid warnings
            )
            
            # Check if data was retrieved
            if df.empty:
                return None, f"No data found for symbol '{stock_symbol}'. Please verify the ticker symbol."
            
            return df, None
            
        except JSONDecodeError as je:
            # Handle JSON decoding errors from Yahoo Finance API
            if attempt < max_retries - 1:
                continue  # Retry
            else:
                return None, "Yahoo Finance API error (JSON decode failed). The service may be temporarily unavailable. Please try again in a few minutes."
                
        except Exception as e:
            error_msg = str(e)
            
            # Check for rate limiting
            if "rate limit" in error_msg.lower() or "too many requests" in error_msg.lower():
                if attempt < max_retries - 1:
                    continue  # Retry with backoff
                else:
                    return None, "Rate limit exceeded. Please wait 2-3 minutes before trying again."
            
            # Check for connection errors
            elif "connection" in error_msg.lower() or "timeout" in error_msg.lower():
                if attempt < max_retries - 1:
                    continue  # Retry
                else:
                    return None, f"Connection error: {error_msg}. Please check your internet connection."
            
            # Other errors
            else:
                if attempt < max_retries - 1:
                    continue  # Retry anyway
                else:
                    return None, f"Error retrieving data: {error_msg}"
    
    return None, "Failed to download data after multiple attempts. Please try again later."

# ===========================
# USER INPUT
# ===========================
stock = st.text_input("Enter Stock Symbol (e.g., NVDA)", "NVDA").upper().strip()

# Define start and end dates
start = dt.datetime(2000, 1, 1)
end = dt.datetime.now()

# ===========================
# DOWNLOAD STOCK DATA
# ===========================
with st.spinner(f"Loading data for {stock}..."):
    df, error = download_stock_data(stock, start, end)

# Handle download errors
if error:
    st.error(f"❌ {error}")
    st.info("💡 **Tips to avoid issues:**")
    st.write("- Wait 2-3 minutes between requests")
    st.write("- Avoid refreshing the page frequently")
    st.write("- Verify the stock symbol is correct (e.g., AAPL, GOOGL, MSFT)")
    st.write("- Try again during market hours")
    st.write("- Data is automatically cached for 1 hour to reduce API calls")
    st.stop()

# ===========================
# DISPLAY DATASET SUMMARY
# ===========================
st.subheader("Stock Data Summary")
st.write(df.describe())

# ===========================
# CALCULATE EXPONENTIAL MOVING AVERAGES (EMA)
# ===========================
df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
df['EMA_100'] = df['Close'].ewm(span=100, adjust=False).mean()
df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()

# ===========================
# CANDLESTICK CHART WITH 20 & 50 EMA
# ===========================
st.subheader("Closing Price with 20 & 50 Days EMA")
fig1 = go.Figure()
fig1.add_trace(go.Candlestick(
    x=df.index, 
    open=df['Open'], 
    high=df['High'], 
    low=df['Low'], 
    close=df['Close'], 
    name='Candlestick'
))
fig1.add_trace(go.Scatter(
    x=df.index, 
    y=df['EMA_20'], 
    mode='lines', 
    name='EMA 20', 
    line=dict(color='green')
))
fig1.add_trace(go.Scatter(
    x=df.index, 
    y=df['EMA_50'], 
    mode='lines', 
    name='EMA 50', 
    line=dict(color='red')
))
fig1.update_layout(xaxis_rangeslider_visible=False)
st.plotly_chart(fig1, use_container_width=True)

# ===========================
# CANDLESTICK CHART WITH 100 & 200 EMA
# ===========================
st.subheader("Closing Price with 100 & 200 Days EMA")
fig2 = go.Figure()
fig2.add_trace(go.Candlestick(
    x=df.index, 
    open=df['Open'], 
    high=df['High'], 
    low=df['Low'], 
    close=df['Close'], 
    name='Candlestick'
))
fig2.add_trace(go.Scatter(
    x=df.index, 
    y=df['EMA_100'], 
    mode='lines', 
    name='EMA 100', 
    line=dict(color='blue')
))
fig2.add_trace(go.Scatter(
    x=df.index, 
    y=df['EMA_200'], 
    mode='lines', 
    name='EMA 200', 
    line=dict(color='purple')
))
fig2.update_layout(xaxis_rangeslider_visible=False)
st.plotly_chart(fig2, use_container_width=True)

# ===========================
# DATA PREPARATION FOR MODEL
# ===========================
# Split data into training and testing (70/30 split)
data_training = df[['Close']][:int(len(df) * 0.70)]
data_testing = df[['Close']][int(len(df) * 0.70):]

# Scale the training data
scaler = MinMaxScaler(feature_range=(0, 1))
data_training_array = scaler.fit_transform(data_training)

# ===========================
# PREPARE TEST DATA
# ===========================
# Get last 100 days from training data
past_100_days = data_training.tail(100)

# Combine with testing data
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)

# Scale the final dataframe
input_data = scaler.transform(final_df)

# Create sequences for prediction
x_test, y_test = [], []
for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i - 100:i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

# ===========================
# MAKE PREDICTIONS ON TEST DATA
# ===========================
y_predicted = model.predict(x_test, verbose=0)

# ===========================
# REVERSE SCALING
# ===========================
scale_factor = 1 / scaler.scale_[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

# ===========================
# EXTRACT CORRESPONDING DATES
# ===========================
test_dates = df.index[-len(y_test):]

# ===========================
# PLOT PREDICTIONS VS ACTUAL
# ===========================
st.subheader("Prediction vs Original Trend")
fig3 = go.Figure()
fig3.add_trace(go.Scatter(
    x=test_dates, 
    y=y_test, 
    mode='lines', 
    name="Actual Price", 
    line=dict(color="blue")
))
fig3.add_trace(go.Scatter(
    x=test_dates, 
    y=y_predicted.flatten(), 
    mode='lines', 
    name="Predicted Price", 
    line=dict(color="orange")
))
fig3.update_layout(
    title="Stock Price Prediction",
    xaxis_title="Date",
    yaxis_title="Price",
    xaxis_rangeslider_visible=False
)
st.plotly_chart(fig3, use_container_width=True)

# ===========================
# GET CURRENT PRICE AND DATE
# ===========================
# Get last closing price correctly as float
current_price = df['Close'].iloc[-1]
current_price = float(current_price) if isinstance(current_price, (pd.Series, pd.DataFrame)) else current_price

# Get current date (the last date in our dataset)
current_date = df.index[-1]

# Get the prediction for the current day (the last prediction in y_predicted)
current_day_prediction = float(y_predicted[-1][0])

# ===========================
# PREDICT NEXT 10 DAYS RECURSIVELY
# ===========================
last_100_days = input_data[-100:]
future_predictions = []
current_input = last_100_days.reshape(1, 100, 1)

# Predict next 10 days
previous_price = y_test[-1]
for _ in range(10):
    # Make prediction
    next_prediction = model.predict(current_input, verbose=0)[0][0] * scale_factor
    
    # ✅ Add realistic fluctuation (including falls)
    noise = np.random.uniform(-5, 5)  # Simulate realistic market fluctuation
    next_prediction += noise
    
    # ✅ Limit the difference between consecutive prices to prevent unrealistic jumps
    if len(future_predictions) > 0:
        next_prediction = np.clip(
            next_prediction, 
            future_predictions[-1] - 5.043,  # Max decrease
            future_predictions[-1] + 5.597   # Max increase
        )
    else:
        next_prediction = np.clip(
            next_prediction, 
            previous_price - 5.673,  # Max decrease from last known price
            previous_price + 5.068   # Max increase from last known price
        )
    
    future_predictions.append(next_prediction)
    
    # Update input for next prediction (sliding window)
    next_scaled = next_prediction / scale_factor
    current_input = np.append(current_input[:, 1:, :], [[[next_scaled]]], axis=1)

# Convert to numpy array
future_predictions = np.array(future_predictions)

# ===========================
# GENERATE PREDICTION DATES
# ===========================
# Get the current time
now = dt.datetime.now()
current_time = now.time()
midnight = dt.time(0, 1)  # 12:01 AM

# Generate dates for predictions based on current time
if current_time >= midnight:
    # After midnight, start future dates from today (current day)
    future_dates = pd.date_range(start=now.date(), periods=6)
    # Use only future predictions for the table
    target_dates = list(future_dates)
    target_prices = list(future_predictions[:6])
    # The first prediction in the table is the current day prediction
    top_prediction = target_prices[0]
else:
    # Before midnight, include current day + future days
    future_dates = pd.date_range(start=df.index[-1] + dt.timedelta(days=1), periods=10)
    target_dates = [current_date] + list(future_dates[:5])
    target_prices = [current_day_prediction] + list(future_predictions[:5])
    # The first prediction in the table is the current day prediction
    top_prediction = current_day_prediction

# ===========================
# CREATE PREDICTION TABLE
# ===========================
target_df = pd.DataFrame({
    'Date': [d.strftime('%Y-%m-%d') for d in target_dates],
    'Predicted Target Price': [f"{price:.2f}" for price in target_prices]
})

# Display prediction table
st.subheader("Predicted Prices (Next 6 Days)")
st.dataframe(target_df, use_container_width=True)

# ===========================
# GET CURRENCY INFORMATION
# ===========================
try:
    ticker = yf.Ticker(stock)
    ticker_info = ticker.info
    currency = ticker_info.get("currency", "USD")
except:
    currency = "USD"
    
# Currency symbol mapping
currency_symbols = {
    "USD": "$",
    "INR": "₹",
    "EUR": "€",
    "GBP": "£",
    "JPY": "¥",
    "AUD": "A$",
    "CAD": "C$",
    "CHF": "CHF",
    "SGD": "S$",
    "HKD": "HK$",
}

# Get the appropriate currency symbol
currency_symbol = currency_symbols.get(currency, currency)

# ===========================
# SINGLE-DAY GAIN/LOSS CALCULATOR
# ===========================
st.subheader("📈 Single-Day Gain/Loss Calculator")

# Display current stock price
formatted_price = f"Current Stock Price: **{currency_symbol}{current_price:.2f}**"
st.write(formatted_price)

# User input for number of shares
shares_bought = st.number_input(
    "Enter the number of shares you want to buy:", 
    min_value=1, 
    step=1, 
    value=1
)

# Determine which prediction to use based on time
if current_time >= midnight:
    # After midnight, use first future prediction
    prediction_for_calc = future_predictions[0]
    day_label = "today"
else:
    # Before midnight, use current day prediction
    prediction_for_calc = current_day_prediction
    day_label = "current day"

# Calculate profit/loss
profit_loss = (prediction_for_calc - current_price) * shares_bought

# Display profit/loss
if profit_loss > 0:
    st.success(f"🎉 Expected **Profit** for {day_label}: **{currency_symbol}{profit_loss:.2f}**")
elif profit_loss < 0:
    st.error(f"⚠️ Expected **Loss** for {day_label}: **{currency_symbol}{profit_loss:.2f}**")
else:
    st.info(f"No gain or loss expected for {day_label}.")

# ===========================
# DOWNLOAD DATASET AS CSV
# ===========================
csv_file_path = f"{stock}_dataset.csv"
df.to_csv(csv_file_path)

with open(csv_file_path, 'rb') as f:
    st.download_button(
        label="Download Dataset as CSV", 
        data=f, 
        file_name=csv_file_path, 
        mime='text/csv'
    )

# ===========================
# SAVE PREDICTION DATA TO CSV LOG
# ===========================
# Create a unique filename based on the stock symbol
prediction_data_filename = f"{stock}_prediction_log.csv"

# Create prediction record with timestamp
prediction_timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
prediction_record = {
    'Timestamp': prediction_timestamp,
    'Stock': stock,
    'Current_Price': current_price,
    'Current_Day_Prediction': top_prediction  # Using the top prediction from the table
}

# Add future dates and prices to the record
for i, (date, price) in enumerate(zip(target_dates, target_prices)):
    day_number = i + 1
    prediction_record[f'Date_{day_number}'] = date.strftime('%Y-%m-%d') if isinstance(date, dt.datetime) else date
    prediction_record[f'Price_{day_number}'] = price

# Check if file exists to append or create new
if os.path.exists(prediction_data_filename):
    # Append to existing file
    existing_data = pd.read_csv(prediction_data_filename)
    updated_data = pd.concat([existing_data, pd.DataFrame([prediction_record])], ignore_index=True)
    updated_data.to_csv(prediction_data_filename, index=False)
else:
    # Create new file
    pd.DataFrame([prediction_record]).to_csv(prediction_data_filename, index=False)

# ===========================
# DISPLAY PREDICTION LOG INFO
# ===========================
st.subheader("✅ Prediction Log")
st.success(f"Prediction record saved to {prediction_data_filename} at {prediction_timestamp}")
st.info(f"Current day prediction (top of table): {currency_symbol}{top_prediction:.2f}")

# ===========================
# DOWNLOAD PREDICTION LOG
# ===========================
with open(prediction_data_filename, 'rb') as file:
    st.download_button(
        label="Download Prediction Log",
        data=file,
        file_name=prediction_data_filename,
        mime='text/csv'
    )