import numpy as np
import pandas as pd
import streamlit as st
import datetime as dt
import yfinance as yf
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import os
import warnings

# Complete warning suppression
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('yfinance').setLevel(logging.ERROR)

# Suppress stderr
import sys
from io import StringIO

class SuppressStderr:
    def __enter__(self):
        self.old_stderr = sys.stderr
        sys.stderr = StringIO()
        return self
    def __exit__(self, *args):
        sys.stderr = self.old_stderr

# Load model with suppression
@st.cache_resource
def load_trained_model():
    with SuppressStderr():
        return load_model('stockpricemodel.keras', compile=False)

model = load_trained_model()

st.title("📊 Stock Price Prediction App")

# Stock input
stock = st.text_input("Enter Stock Symbol (e.g., AAPL, MSFT, GOOGL)", "AAPL")
stock = stock.upper().strip()

# Session state initialization
if 'data_cache' not in st.session_state:
    st.session_state.data_cache = {}

# Download function - NO CACHING to avoid multiple calls
def download_stock_simple(symbol):
    """Simple, single download - no retries, no caching"""
    try:
        with SuppressStderr():
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                period="10y",  # Last 10 years
                interval="1d",
                auto_adjust=True,
                actions=False
            )
        
        if not df.empty and len(df) > 100:
            return df, None
        else:
            return None, "NO_DATA"
            
    except Exception as e:
        error_str = str(e).lower()
        if 'json' in error_str or '429' in error_str or 'rate' in error_str:
            return None, "RATE_LIMITED"
        else:
            return None, str(e)[:100]

# Check if we have cached data for this stock
cache_key = f"{stock}_data"

if cache_key in st.session_state.data_cache:
    df = st.session_state.data_cache[cache_key]
    st.info(f"📂 Using cached data for {stock} (last updated: {df.index[-1].strftime('%Y-%m-%d')})")
else:
    # Download new data
    with st.spinner(f"📥 Downloading {stock} data..."):
        df, error = download_stock_simple(stock)
    
    if error == "RATE_LIMITED":
        st.error("🚫 **Rate Limited by Yahoo Finance**")
        st.warning("Your IP has been temporarily blocked due to too many requests.")
        
        st.subheader("✅ Quick Solutions:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🌐 Option 1: Use VPN
            1. **Connect to VPN** (Proton VPN recommended)
            2. **Select US or UK server**
            3. **Refresh this page** (press R)
            4. Enter stock symbol again
            """)
        
        with col2:
            st.markdown("""
            ### 📤 Option 2: Upload CSV
            1. Go to [Yahoo Finance](https://finance.yahoo.com/quote/AAPL/history)
            2. Download CSV (max period)
            3. Upload below 👇
            """)
        
        # File uploader
        st.subheader("📂 Upload Historical Data")
        uploaded_file = st.file_uploader("Choose CSV file", type=['csv'], key="csv_upload")
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                # Handle different CSV formats
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'])
                    df.set_index('Date', inplace=True)
                else:
                    df.index = pd.to_datetime(df.index)
                
                st.success(f"✅ Loaded {len(df)} days from CSV!")
                st.session_state.data_cache[cache_key] = df
                st.rerun()
            except Exception as e:
                st.error(f"Error reading CSV: {e}")
                st.stop()
        else:
            st.info("💡 Or wait 1 hour and your IP will be unblocked")
            st.stop()
    
    elif error == "NO_DATA":
        st.error(f"❌ No data found for '{stock}'. Please check the symbol.")
        st.info("Try: AAPL, MSFT, GOOGL, TSLA, AMZN, META, NFLX")
        st.stop()
    
    elif error:
        st.error(f"❌ Download failed: {error}")
        st.info("Try a different symbol or upload CSV manually")
        st.stop()
    
    # Cache the downloaded data
    st.session_state.data_cache[cache_key] = df

# Verify data
if df is None or df.empty:
    st.error("No data available")
    st.stop()

if len(df) < 200:
    st.warning(f"⚠️ Only {len(df)} days of data. Need at least 200 for accurate predictions.")
    if len(df) < 100:
        st.error("❌ Insufficient data")
        st.stop()

st.success(f"✅ Loaded **{len(df):,}** days of data for **{stock}**")

# Summary metrics
st.subheader("📈 Stock Data Summary")
col1, col2, col3, col4 = st.columns(4)

current_price = df['Close'].iloc[-1]
high_52w = df['Close'].tail(252).max() if len(df) >= 252 else df['Close'].max()
low_52w = df['Close'].tail(252).min() if len(df) >= 252 else df['Close'].min()

with col1:
    st.metric("Latest Close", f"${current_price:.2f}")
with col2:
    st.metric("52W High", f"${high_52w:.2f}")
with col3:
    st.metric("52W Low", f"${low_52w:.2f}")
with col4:
    if len(df) > 1:
        change = ((df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100
        st.metric("Daily Change", f"{change:.2f}%")

with st.expander("📊 View Statistics"):
    st.write(df[['Open', 'High', 'Low', 'Close', 'Volume']].describe())

# Calculate EMAs
df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
df['EMA_100'] = df['Close'].ewm(span=100, adjust=False).mean()
df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()

# Chart 1: 20 & 50 EMA
st.subheader("📊 Price with 20 & 50 Day EMA")
fig1 = go.Figure()

if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
    fig1.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], 
        low=df['Low'], close=df['Close'], name='Price'
    ))

fig1.add_trace(go.Scatter(x=df.index, y=df['EMA_20'], mode='lines', 
                          name='EMA 20', line=dict(color='green', width=2)))
fig1.add_trace(go.Scatter(x=df.index, y=df['EMA_50'], mode='lines',
                          name='EMA 50', line=dict(color='red', width=2)))
fig1.update_layout(xaxis_rangeslider_visible=False, height=500, hovermode='x unified')
st.plotly_chart(fig1, use_container_width=True)

# Chart 2: 100 & 200 EMA
st.subheader("📊 Price with 100 & 200 Day EMA")
fig2 = go.Figure()

if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
    fig2.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'], name='Price'
    ))

fig2.add_trace(go.Scatter(x=df.index, y=df['EMA_100'], mode='lines',
                          name='EMA 100', line=dict(color='blue', width=2)))
fig2.add_trace(go.Scatter(x=df.index, y=df['EMA_200'], mode='lines',
                          name='EMA 200', line=dict(color='purple', width=2)))
fig2.update_layout(xaxis_rangeslider_visible=False, height=500, hovermode='x unified')
st.plotly_chart(fig2, use_container_width=True)

# Model predictions
data_training = df[['Close']][:int(len(df) * 0.70)]
data_testing = df[['Close']][int(len(df) * 0.70):]

scaler = MinMaxScaler(feature_range=(0, 1))
data_training_array = scaler.fit_transform(data_training)

past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.transform(final_df)

x_test, y_test = [], []
for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i - 100:i])
    y_test.append(input_data[i, 0])
x_test, y_test = np.array(x_test), np.array(y_test)

# Predictions
with st.spinner('🔮 Generating predictions...'):
    with SuppressStderr():
        y_predicted = model.predict(x_test, verbose=0)

scale_factor = 1 / scaler.scale_[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

test_dates = df.index[-len(y_test):]

# Plot predictions
st.subheader("🎯 Prediction vs Actual")
fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=test_dates, y=y_test, mode='lines',
                          name="Actual", line=dict(color="blue", width=2)))
fig3.add_trace(go.Scatter(x=test_dates, y=y_predicted.flatten(), mode='lines',
                          name="Predicted", line=dict(color="orange", width=2)))
fig3.update_layout(title=f"{stock} Price Prediction", xaxis_title="Date",
                   yaxis_title="Price ($)", height=500, hovermode='x unified')
st.plotly_chart(fig3, use_container_width=True)

# Accuracy
mape = np.mean(np.abs((y_test - y_predicted.flatten()) / y_test)) * 100
st.info(f"📊 Model Accuracy: MAPE = {mape:.2f}%")

# Future predictions
current_price = float(df['Close'].iloc[-1])
current_date = df.index[-1]
current_day_prediction = float(y_predicted[-1][0])

last_100_days = input_data[-100:]
future_predictions = []
current_input = last_100_days.reshape(1, 100, 1)

previous_price = y_test[-1]
for _ in range(10):
    with SuppressStderr():
        next_pred = model.predict(current_input, verbose=0)[0][0] * scale_factor
    
    noise = np.random.uniform(-2, 2)
    next_pred += noise
    
    if len(future_predictions) > 0:
        next_pred = np.clip(next_pred, future_predictions[-1] * 0.97, future_predictions[-1] * 1.03)
    else:
        next_pred = np.clip(next_pred, previous_price * 0.97, previous_price * 1.03)
    
    future_predictions.append(next_pred)
    
    next_scaled = next_pred / scale_factor
    current_input = np.append(current_input[:, 1:, :], [[[next_scaled]]], axis=1)

future_predictions = np.array(future_predictions)

# Generate dates
now = dt.datetime.now()
if now.time() >= dt.time(0, 1):
    future_dates = pd.date_range(start=now.date(), periods=6)
    target_dates = list(future_dates)
    target_prices = list(future_predictions[:6])
    top_prediction = target_prices[0]
else:
    future_dates = pd.date_range(start=current_date + dt.timedelta(days=1), periods=10)
    target_dates = [current_date] + list(future_dates[:5])
    target_prices = [current_day_prediction] + list(future_predictions[:5])
    top_prediction = current_day_prediction

# Prediction table
target_df = pd.DataFrame({
    'Date': [d.strftime('%Y-%m-%d') for d in target_dates],
    'Predicted Price': [f"${p:.2f}" for p in target_prices],
    'Change %': [f"{((p - current_price) / current_price * 100):.2f}%" for p in target_prices]
})

st.subheader("🔮 6-Day Price Forecast")
st.dataframe(target_df, use_container_width=True, hide_index=True)

# Investment calculator
st.subheader("💰 Investment Calculator")

col1, col2 = st.columns(2)
with col1:
    st.metric("Current Price", f"${current_price:.2f}")
with col2:
    st.metric("Next Day Prediction", f"${top_prediction:.2f}",
              delta=f"{((top_prediction - current_price) / current_price * 100):.2f}%")

shares = st.number_input("Number of shares:", min_value=1, value=10, step=1)

profit_loss = (top_prediction - current_price) * shares
investment = current_price * shares

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Investment", f"${investment:.2f}")
with col2:
    st.metric("Expected Value", f"${investment + profit_loss:.2f}")
with col3:
    if profit_loss > 0:
        st.metric("Expected Profit", f"${profit_loss:.2f}", delta="📈")
    elif profit_loss < 0:
        st.metric("Expected Loss", f"${abs(profit_loss):.2f}", delta="📉")
    else:
        st.metric("No Change", "$0.00")

# Downloads
st.subheader("📥 Download Data")
col1, col2 = st.columns(2)

with col1:
    csv_data = df.to_csv()
    st.download_button("📊 Historical Data", csv_data, f"{stock}_data.csv", "text/csv")

# Prediction log
log_file = f"{stock}_prediction_log.csv"
timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
record = {'Timestamp': timestamp, 'Stock': stock, 'Current_Price': current_price, 
          'Current_Day_Prediction': top_prediction}

for i, (date, price) in enumerate(zip(target_dates, target_prices), 1):
    record[f'Date_{i}'] = date.strftime('%Y-%m-%d') if isinstance(date, dt.datetime) else date
    record[f'Price_{i}'] = price

if os.path.exists(log_file):
    log_df = pd.read_csv(log_file)
    log_df = pd.concat([log_df, pd.DataFrame([record])], ignore_index=True)
    log_df.to_csv(log_file, index=False)
else:
    pd.DataFrame([record]).to_csv(log_file, index=False)

with col2:
    with open(log_file, 'rb') as f:
        st.download_button("📝 Prediction Log", f, log_file, "text/csv")

st.success(f"✅ Prediction logged at {timestamp}")

st.divider()
st.caption("⚠️ Educational purposes only. Not financial advice.")