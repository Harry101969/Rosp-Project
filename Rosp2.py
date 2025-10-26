import numpy as np
import pandas as pd
import streamlit as st
import datetime as dt
import yfinance as yf
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import os
import time
import warnings
import sys

# Suppress all warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Redirect stderr to suppress TensorFlow warnings
class SuppressOutput:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr

# Load the trained model with suppressed output
@st.cache_resource
def load_trained_model():
    with SuppressOutput():
        return load_model('stockpricemodel.keras', compile=False)

model = load_trained_model()

st.title("📊 Stock Price Prediction App")

# User input for stock symbol
stock = st.text_input("Enter Stock Symbol (e.g., AAPL, MSFT, GOOGL)", "AAPL")
stock = stock.upper().strip()

# Define dates
end = dt.datetime.now()
start = dt.datetime(2012, 1, 1)

# Function to download with proper headers and user agent
@st.cache_data(ttl=3600, show_spinner=False)
def download_with_retry(symbol, start_date, end_date, max_attempts=5):
    """Download stock data with multiple retry strategies"""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for attempt in range(max_attempts):
        try:
            status_text.text(f"📥 Attempting to download {symbol} data... (Attempt {attempt + 1}/{max_attempts})")
            progress_bar.progress(int((attempt + 1) / max_attempts * 30))
            
            # Wait longer between attempts
            if attempt > 0:
                wait_time = min(3 * (2 ** attempt), 15)  # Exponential backoff, max 15s
                status_text.text(f"⏳ Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
            else:
                time.sleep(2)  # Initial 2 second delay
            
            # Create ticker object with session
            ticker = yf.Ticker(symbol)
            
            # Try to get data using different methods
            status_text.text(f"📊 Fetching historical data for {symbol}...")
            progress_bar.progress(50)
            
            # Method 1: Use ticker.history()
            df = ticker.history(
                start=start_date,
                end=end_date,
                auto_adjust=True,
                actions=False
            )
            
            progress_bar.progress(90)
            
            if not df.empty and len(df) > 0:
                # Ensure we have the required columns
                if 'Close' in df.columns:
                    progress_bar.progress(100)
                    status_text.text(f"✅ Successfully downloaded {len(df)} days of data!")
                    time.sleep(0.5)
                    progress_bar.empty()
                    status_text.empty()
                    return df, None
            
            # Method 2: Try yf.download as fallback
            status_text.text(f"🔄 Trying alternative method...")
            time.sleep(2)
            
            df = yf.download(
                symbol,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=True,
                threads=False,
                ignore_tz=True
            )
            
            if not df.empty and len(df) > 0:
                progress_bar.progress(100)
                status_text.text(f"✅ Successfully downloaded {len(df)} days of data!")
                time.sleep(0.5)
                progress_bar.empty()
                status_text.empty()
                return df, None
                
        except Exception as e:
            error_msg = str(e)
            if attempt < max_attempts - 1:
                status_text.text(f"⚠️ Attempt {attempt + 1} failed: {error_msg[:50]}...")
                continue
            else:
                progress_bar.empty()
                status_text.empty()
                return None, f"Failed after {max_attempts} attempts: {error_msg}"
    
    progress_bar.empty()
    status_text.empty()
    return None, f"Could not download data for {symbol} after {max_attempts} attempts"

# Additional fallback: Load from CSV if available
def load_from_cache(symbol):
    """Try to load previously downloaded data"""
    cache_file = f"{symbol}_cached_data.csv"
    if os.path.exists(cache_file):
        try:
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            return df, None
        except:
            return None, "Cache file corrupted"
    return None, "No cache available"

# Main download logic
st.info(f"🔄 Fetching data for **{stock}**... This may take 10-20 seconds due to rate limits.")

# Try to download
df, error = download_with_retry(stock, start, end)

# If download fails, try cache
if error:
    st.warning(f"⚠️ Live download failed: {error}")
    st.info("🔄 Checking for cached data...")
    df, cache_error = load_from_cache(stock)
    
    if df is not None and not df.empty:
        st.success(f"✅ Loaded cached data for {stock} ({len(df)} days)")
        cache_date = df.index[-1].strftime('%Y-%m-%d')
        st.warning(f"⚠️ Using cached data (last updated: {cache_date}). Data may not be current.")
    else:
        st.error("❌ **Download Failed & No Cache Available**")
        st.error(f"Error: {error}")
        
        st.subheader("🔧 **Troubleshooting Steps:**")
        st.markdown("""
        **Your IP may be temporarily blocked by Yahoo Finance. Try these:**
        
        1. **Wait 15-30 minutes** before trying again (most common solution)
        2. **Restart your router** to get a new IP address
        3. **Use a VPN** to change your location
        4. **Try a different network** (mobile hotspot, different WiFi)
        5. **Use a different stock symbol** that you haven't tried recently
        
        **Alternative Solutions:**
        - Download data manually from Yahoo Finance website and upload CSV
        - Try during off-peak hours (early morning/late night)
        - Clear browser cache and cookies
        """)
        
        st.subheader("📊 **Popular Stock Symbols to Try:**")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.code("AAPL")
            st.caption("Apple Inc.")
        with col2:
            st.code("MSFT")
            st.caption("Microsoft")
        with col3:
            st.code("GOOGL")
            st.caption("Google")
        with col4:
            st.code("TSLA")
            st.caption("Tesla")
        
        st.stop()

if df is None or df.empty:
    st.error(f"❌ No data available for '{stock}'.")
    st.stop()

# Save to cache for future use
try:
    cache_file = f"{stock}_cached_data.csv"
    df.to_csv(cache_file)
except:
    pass

# Check if we have enough data
if len(df) < 200:
    st.warning(f"⚠️ Only {len(df)} days of data available. Need at least 200 days for accurate predictions.")
    if len(df) < 100:
        st.error("❌ Insufficient data. Please try another symbol.")
        st.stop()

st.success(f"✅ Successfully loaded **{len(df)}** days of data for **{stock}**!")

# Display dataset summary
st.subheader("📈 Stock Data Summary")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Latest Close", f"${df['Close'].iloc[-1]:.2f}")
with col2:
    st.metric("52W High", f"${df['Close'].max():.2f}")
with col3:
    st.metric("52W Low", f"${df['Close'].min():.2f}")
with col4:
    if len(df) > 1:
        change = ((df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100
        st.metric("Daily Change", f"{change:.2f}%")
    else:
        st.metric("Daily Change", "N/A")

with st.expander("View Detailed Statistics"):
    st.write(df.describe())

# Calculate Exponential Moving Averages (EMA)
df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
df['EMA_100'] = df['Close'].ewm(span=100, adjust=False).mean()
df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()

# Candlestick Chart with 20 & 50 EMA
st.subheader("📊 Closing Price with 20 & 50 Days EMA")
fig1 = go.Figure()

# Check if OHLC data exists
if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
    fig1.add_trace(go.Candlestick(
        x=df.index, 
        open=df['Open'], 
        high=df['High'], 
        low=df['Low'], 
        close=df['Close'], 
        name='Candlestick'
    ))
else:
    fig1.add_trace(go.Scatter(
        x=df.index,
        y=df['Close'],
        mode='lines',
        name='Close Price',
        line=dict(color='black', width=1)
    ))

fig1.add_trace(go.Scatter(
    x=df.index, 
    y=df['EMA_20'], 
    mode='lines', 
    name='EMA 20', 
    line=dict(color='green', width=2)
))
fig1.add_trace(go.Scatter(
    x=df.index, 
    y=df['EMA_50'], 
    mode='lines', 
    name='EMA 50', 
    line=dict(color='red', width=2)
))
fig1.update_layout(
    xaxis_rangeslider_visible=False,
    height=500,
    hovermode='x unified'
)
st.plotly_chart(fig1, use_container_width=True)

# Candlestick Chart with 100 & 200 EMA
st.subheader("📊 Closing Price with 100 & 200 Days EMA")
fig2 = go.Figure()

if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
    fig2.add_trace(go.Candlestick(
        x=df.index, 
        open=df['Open'], 
        high=df['High'], 
        low=df['Low'], 
        close=df['Close'], 
        name='Candlestick'
    ))
else:
    fig2.add_trace(go.Scatter(
        x=df.index,
        y=df['Close'],
        mode='lines',
        name='Close Price',
        line=dict(color='black', width=1)
    ))

fig2.add_trace(go.Scatter(
    x=df.index, 
    y=df['EMA_100'], 
    mode='lines', 
    name='EMA 100', 
    line=dict(color='blue', width=2)
))
fig2.add_trace(go.Scatter(
    x=df.index, 
    y=df['EMA_200'], 
    mode='lines', 
    name='EMA 200', 
    line=dict(color='purple', width=2)
))
fig2.update_layout(
    xaxis_rangeslider_visible=False,
    height=500,
    hovermode='x unified'
)
st.plotly_chart(fig2, use_container_width=True)

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
with st.spinner('🔮 Generating predictions...'):
    y_predicted = model.predict(x_test, verbose=0)

# Reverse scaling
scale_factor = 1 / scaler.scale_[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

# Extract corresponding dates for the test dataset
test_dates = df.index[-len(y_test):]

# Plot Predictions
st.subheader("🎯 Prediction vs Original Trend")
fig3 = go.Figure()
fig3.add_trace(go.Scatter(
    x=test_dates, 
    y=y_test, 
    mode='lines', 
    name="Actual Price", 
    line=dict(color="blue", width=2)
))
fig3.add_trace(go.Scatter(
    x=test_dates, 
    y=y_predicted.flatten(), 
    mode='lines', 
    name="Predicted Price", 
    line=dict(color="orange", width=2)
))
fig3.update_layout(
    title=f"{stock} Stock Price Prediction",
    xaxis_title="Date",
    yaxis_title="Price ($)",
    xaxis_rangeslider_visible=False,
    height=500,
    hovermode='x unified'
)
st.plotly_chart(fig3, use_container_width=True)

# Calculate prediction accuracy
mape = np.mean(np.abs((y_test - y_predicted.flatten()) / y_test)) * 100
st.info(f"📊 Model Accuracy: MAPE = {mape:.2f}% (Lower is better)")

# Get last closing price correctly as float
current_price = float(df['Close'].iloc[-1])
current_date = df.index[-1]
current_day_prediction = float(y_predicted[-1][0])

# 🚀 Predict the next 10 days recursively
last_100_days = input_data[-100:]
future_predictions = []
current_input = last_100_days.reshape(1, 100, 1)

previous_price = y_test[-1]
for _ in range(10):
    next_prediction = model.predict(current_input, verbose=0)[0][0] * scale_factor
    
    # Add realistic fluctuation
    noise = np.random.uniform(-3, 3)
    next_prediction += noise
    
    # Limit the difference between consecutive prices
    if len(future_predictions) > 0:
        next_prediction = np.clip(
            next_prediction, 
            future_predictions[-1] * 0.97,
            future_predictions[-1] * 1.03
        )
    else:
        next_prediction = np.clip(
            next_prediction, 
            previous_price * 0.97, 
            previous_price * 1.03
        )
    
    future_predictions.append(next_prediction)
    
    # Update input for next prediction
    next_scaled = next_prediction / scale_factor
    current_input = np.append(current_input[:, 1:, :], [[[next_scaled]]], axis=1)

future_predictions = np.array(future_predictions)

# Generate dates for predictions
now = dt.datetime.now()
current_time = now.time()
midnight = dt.time(0, 1)

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
    'Predicted Target Price': [f"${p:.2f}" for p in target_prices],
    'Change': [f"{((target_prices[i] - current_price) / current_price * 100):.2f}%" 
               for i in range(len(target_prices))]
})

# Display table
st.subheader("🔮 Predicted Prices (Next 6 Days)")
st.dataframe(target_df, use_container_width=True, hide_index=True)

# Get ticker information for currency
currency = "USD"
try:
    time.sleep(1)
    ticker_info = yf.Ticker(stock).info
    currency = ticker_info.get("currency", "USD")
except:
    pass

currency_symbols = {
    "USD": "$", "INR": "₹", "EUR": "€", "GBP": "£", "JPY": "¥",
    "AUD": "A$", "CAD": "C$", "CHF": "CHF", "SGD": "S$", "HKD": "HK$",
}

currency_symbol = currency_symbols.get(currency, "$")

# Investment Calculator
st.subheader("💰 Single-Day Gain/Loss Calculator")

col1, col2 = st.columns(2)
with col1:
    st.metric("Current Stock Price", f"{currency_symbol}{current_price:.2f}")
with col2:
    st.metric("Next Day Prediction", f"{currency_symbol}{top_prediction:.2f}",
              delta=f"{((top_prediction - current_price) / current_price * 100):.2f}%")

shares_bought = st.number_input(
    "Enter the number of shares you want to buy:", 
    min_value=1, 
    step=1, 
    value=10
)

# Calculate profit/loss
if current_time >= midnight:
    prediction_for_calc = future_predictions[0]
    day_label = "today"
else:
    prediction_for_calc = current_day_prediction
    day_label = "current day"

profit_loss = (prediction_for_calc - current_price) * shares_bought
investment_amount = current_price * shares_bought

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Investment", f"{currency_symbol}{investment_amount:.2f}")
with col2:
    st.metric("Expected Value", f"{currency_symbol}{(investment_amount + profit_loss):.2f}")
with col3:
    if profit_loss > 0:
        st.metric("Expected Profit", f"{currency_symbol}{profit_loss:.2f}", delta="Gain")
    elif profit_loss < 0:
        st.metric("Expected Loss", f"{currency_symbol}{abs(profit_loss):.2f}", delta="Loss")
    else:
        st.metric("Expected Change", f"{currency_symbol}0.00", delta="Neutral")

# Download dataset
st.subheader("📥 Download Data")
col1, col2 = st.columns(2)

with col1:
    csv_buffer = df.to_csv()
    st.download_button(
        label="📊 Download Historical Data",
        data=csv_buffer,
        file_name=f"{stock}_historical_data.csv",
        mime='text/csv'
    )

# Save prediction data
prediction_data_filename = f"{stock}_prediction_log.csv"
prediction_timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
prediction_record = {
    'Timestamp': prediction_timestamp,
    'Stock': stock,
    'Current_Price': current_price,
    'Current_Day_Prediction': top_prediction
}

for i, (date, price) in enumerate(zip(target_dates, target_prices)):
    day_number = i + 1
    prediction_record[f'Date_{day_number}'] = date.strftime('%Y-%m-%d') if isinstance(date, dt.datetime) else date
    prediction_record[f'Price_{day_number}'] = price

if os.path.exists(prediction_data_filename):
    existing_data = pd.read_csv(prediction_data_filename)
    updated_data = pd.concat([existing_data, pd.DataFrame([prediction_record])], ignore_index=True)
    updated_data.to_csv(prediction_data_filename, index=False)
else:
    pd.DataFrame([prediction_record]).to_csv(prediction_data_filename, index=False)

with col2:
    with open(prediction_data_filename, 'rb') as file:
        st.download_button(
            label="📝 Download Prediction Log",
            data=file,
            file_name=prediction_data_filename,
            mime='text/csv'
        )

st.success(f"✅ Prediction saved at {prediction_timestamp}")

# Disclaimer
st.divider()
st.caption("⚠️ **Disclaimer:** This app is for educational purposes only. Stock predictions are based on historical data and should not be considered financial advice. Always consult with a financial advisor before making investment decisions.")