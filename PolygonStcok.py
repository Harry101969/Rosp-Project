import numpy as np
import pandas as pd
import streamlit as st
import datetime as dt
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import os
import requests

# ========================================
# CONFIGURATION
# ========================================
# Get your free API key from: https://polygon.io/
POLYGON_API_KEY = 'kYxTdT4zPN7X2ysrSb7BOtmzuhVoQxnA'  # Replace with your actual API key

# Load the trained model
@st.cache_resource
def load_stock_model():
    return load_model('polygonbased.keras')

model = load_stock_model()

st.title("Stock Price Prediction App")
st.markdown("Powered by Polygon.io API")

# ========================================
# FETCH DATA FUNCTION
# ========================================
@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_stock_data_polygon(symbol, start_date, end_date, api_key):
    """Fetch stock data using Polygon.io API"""
    
    # Convert datetime to string if needed
    if isinstance(start_date, dt.datetime):
        start_date = start_date.strftime('%Y-%m-%d')
    if isinstance(end_date, dt.datetime):
        end_date = end_date.strftime('%Y-%m-%d')
    
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}"
    
    params = {
        'adjusted': 'true',
        'sort': 'asc',
        'limit': 50000,
        'apiKey': api_key
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if data['status'] != 'OK' or 'results' not in data:
            return None, f"No data returned for {symbol}"
        
        results = data['results']
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Rename columns
        df = df.rename(columns={
            'o': 'Open',
            'h': 'High',
            'l': 'Low',
            'c': 'Close',
            'v': 'Volume',
            't': 'Date'
        })
        
        # Convert timestamp to datetime
        df['Date'] = pd.to_datetime(df['Date'], unit='ms')
        df = df.set_index('Date')
        
        # Keep only necessary columns
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        
        return df, None
        
    except Exception as e:
        return None, str(e)

# User input for stock symbol
stock = st.text_input("Enter Stock Symbol (e.g., NVDA)", "NVDA")

# Define start and end dates
start = dt.datetime(2000, 1, 1)
end = dt.datetime.now()

# ✅ Fetch data
with st.spinner(f'Fetching data for {stock}...'):
    df, error = fetch_stock_data_polygon(stock, start, end, POLYGON_API_KEY)
    
    if error:
        st.error(f"❌ Error: {error}")
        st.info("💡 Tips:\n- Check if the stock symbol is correct (e.g., 'NVDA' not 'NVIDIA')\n- Verify your Polygon.io API key is valid\n- Free tier: 5 API calls/minute")
        st.stop()
    
    if df.empty:
        st.error(f"❌ No data found for symbol '{stock}'. Please check the symbol and try again.")
        st.stop()

st.success(f"✅ Successfully loaded {len(df)} days of data")

# Display dataset summary
st.subheader("Stock Data Summary")
st.write(df.describe())

# Calculate Exponential Moving Averages (EMA)
df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
df['EMA_100'] = df['Close'].ewm(span=100, adjust=False).mean()
df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()

# Candlestick Chart with 20 & 50 EMA
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
st.plotly_chart(fig1)

# Candlestick Chart with 100 & 200 EMA
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
st.plotly_chart(fig2)

# Data Preparation
data_training = df[['Close']][:int(len(df) * 0.70)]
data_testing = df[['Close']][int(len(df) * 0.70):]
scaler = MinMaxScaler(feature_range=(0, 1))
data_training_array = scaler.fit_transform(data_training)

# Prepare test data
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.transform(final_df)

x_test, y_test = [], []
for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i - 100:i])
    y_test.append(input_data[i, 0])
x_test, y_test = np.array(x_test), np.array(y_test)

# Make predictions
y_predicted = model.predict(x_test)

# Reverse scaling
scale_factor = 1 / scaler.scale_[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

# Extract corresponding dates for the test dataset
test_dates = df.index[-len(y_test):]

# Plot Predictions
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
st.plotly_chart(fig3)

# Get last closing price correctly as float
current_price = df['Close'].iloc[-1]
current_price = float(current_price) if isinstance(current_price, (pd.Series, pd.DataFrame)) else current_price

# Get current date (the last date in our dataset)
current_date = df.index[-1]

# Get the prediction for the current day (the last prediction in y_predicted)
current_day_prediction = float(y_predicted[-1][0])

# 🚀 Predict the next 10 days recursively
last_100_days = input_data[-100:]
future_predictions = []
current_input = last_100_days.reshape(1, 100, 1)

# Predict next 10 days
previous_price = y_test[-1]
for _ in range(10):
    next_prediction = model.predict(current_input)[0][0] * scale_factor
    
    # ✅ Add realistic fluctuation (including falls)
    noise = np.random.uniform(-5, 5)
    next_prediction += noise
    
    # ✅ Limit the difference between consecutive prices
    if len(future_predictions) > 0:
        next_prediction = np.clip(
            next_prediction, 
            future_predictions[-1] - 5.043, 
            future_predictions[-1] + 5.597
        )
    else:
        next_prediction = np.clip(
            next_prediction, 
            previous_price - 5.673, 
            previous_price + 5.068
        )
    
    future_predictions.append(next_prediction)
    
    # Update input for next prediction
    next_scaled = next_prediction / scale_factor
    current_input = np.append(current_input[:, 1:, :], [[[next_scaled]]], axis=1)

# Reverse scaling for target values
future_predictions = np.array(future_predictions)

# Get the current time
now = dt.datetime.now()
current_time = now.time()
midnight = dt.time(0, 1)

# Generate dates for predictions
if current_time >= midnight:
    future_dates = pd.date_range(start=now.date(), periods=6)
    target_dates = list(future_dates)
    target_prices = list(future_predictions[:6])
    top_prediction = target_prices[0]
else:
    future_dates = pd.date_range(start=df.index[-1] + dt.timedelta(days=1), periods=10)
    target_dates = [current_date] + list(future_dates[:5])
    target_prices = [current_day_prediction] + list(future_predictions[:5])
    top_prediction = current_day_prediction

# Create the table
target_df = pd.DataFrame({
    'Date': [d.strftime('%Y-%m-%d') for d in target_dates],
    'Predicted Target Price': target_prices
})

# Display table
st.subheader("Predicted Prices (Next 6 Days)")
st.write(target_df)

# Get ticker information for currency
@st.cache_data
def get_currency_symbol(symbol_name):
    # Default to USD for most stocks
    return "$"

currency_symbol = get_currency_symbol(stock)

st.subheader("📈 Single-Day Gain/Loss Calculator")

formatted_price = f"Current Stock Price: **{currency_symbol}{current_price:.2f}**"
st.write(formatted_price)

shares_bought = st.number_input(
    "Enter the number of shares you want to buy:", 
    min_value=1, 
    step=1, 
    value=1
)

# Determine which prediction to use based on time
if current_time >= midnight:
    prediction_for_calc = future_predictions[0]
    day_label = "today"
else:
    prediction_for_calc = current_day_prediction
    day_label = "current day"

# Calculate profit/loss
profit_loss = (prediction_for_calc - current_price) * shares_bought

if profit_loss > 0:
    st.success(f"🎉 Expected **Profit** for {day_label}: **{currency_symbol}{profit_loss:.2f}**")
elif profit_loss < 0:
    st.error(f"⚠️ Expected **Loss** for {day_label}: **{currency_symbol}{abs(profit_loss):.2f}**")
else:
    st.info(f"No gain or loss expected for {day_label}.")

# ✅ Download dataset
csv_file_path = f"{stock}_dataset.csv"
df.to_csv(csv_file_path)

with open(csv_file_path, 'rb') as file:
    st.download_button(
        label="Download Dataset as CSV",
        data=file,
        file_name=csv_file_path,
        mime='text/csv'
    )

# ✅ Save prediction data to CSV
prediction_data_filename = f"{stock}_prediction_log.csv"

# Create prediction record
prediction_timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
prediction_record = {
    'Timestamp': prediction_timestamp,
    'Stock': stock,
    'Current_Price': current_price,
    'Current_Day_Prediction': top_prediction
}

# For future dates
for i, (date, price) in enumerate(zip(target_dates, target_prices)):
    day_number = i + 1
    prediction_record[f'Date_{day_number}'] = date.strftime('%Y-%m-%d') if isinstance(date, dt.datetime) else date
    prediction_record[f'Price_{day_number}'] = price

# Check if file exists to append or create new
if os.path.exists(prediction_data_filename):
    existing_data = pd.read_csv(prediction_data_filename)
    updated_data = pd.concat([existing_data, pd.DataFrame([prediction_record])], ignore_index=True)
    updated_data.to_csv(prediction_data_filename, index=False)
else:
    pd.DataFrame([prediction_record]).to_csv(prediction_data_filename, index=False)

st.subheader("✅ Prediction Log")
st.success(f"Prediction record saved to {prediction_data_filename} at {prediction_timestamp}")
st.info(f"Current day prediction (top of table): {currency_symbol}{top_prediction:.2f}")

# Download prediction log
with open(prediction_data_filename, 'rb') as file:
    st.download_button(
        label="Download Prediction Log",
        data=file,
        file_name=prediction_data_filename,
        mime='text/csv'
    )