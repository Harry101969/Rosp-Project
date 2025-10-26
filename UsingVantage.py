import numpy as np
import pandas as pd
import streamlit as st
import datetime as dt
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import os
from alpha_vantage.timeseries import TimeSeries
import time

# ========================================
# CONFIGURATION
# ========================================
# Get your free API key from: https://www.alphavantage.co/support/#api-key
ALPHA_VANTAGE_API_KEY = st.secrets.get("ALPHA_VANTAGE_API_KEY", "YOUR_API_KEY_HERE")

# If you're not using Streamlit secrets, uncomment the line below and add your API key
# ALPHA_VANTAGE_API_KEY = "YOUR_API_KEY_HERE"

st.set_page_config(page_title="Stock Price Prediction", page_icon="📈", layout="wide")

# ========================================
# LOAD MODEL
# ========================================
@st.cache_resource
def load_stock_model():
    try:
        return load_model('stockpricemodel.keras')
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_stock_model()

if model is None:
    st.error("❌ Failed to load the model. Please ensure 'stockpricemodel.keras' exists.")
    st.stop()

# ========================================
# TITLE AND DESCRIPTION
# ========================================
st.title("📈 Stock Price Prediction App")
st.markdown("Powered by Alpha Vantage & LSTM Neural Network")

# ========================================
# SIDEBAR CONFIGURATION
# ========================================
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Key input
    if ALPHA_VANTAGE_API_KEY == "YOUR_API_KEY_HERE":
        api_key = st.text_input(
            "Enter Alpha Vantage API Key", 
            type="password",
            help="Get your free API key from https://www.alphavantage.co/support/#api-key"
        )
        if not api_key:
            st.warning("⚠️ Please enter your Alpha Vantage API key")
            st.markdown("[Get Free API Key](https://www.alphavantage.co/support/#api-key)")
            st.stop()
    else:
        api_key = ALPHA_VANTAGE_API_KEY
    
    st.success("✅ API Key configured")
    
    st.markdown("---")
    st.markdown("### 📊 Supported Stocks")
    st.markdown("""
    - **US Stocks**: NVDA, AAPL, GOOGL, MSFT, TSLA
    - **Exchange format**: Symbol only (e.g., 'NVDA')
    """)
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("This app uses LSTM neural networks to predict stock prices based on historical data.")

# ========================================
# STOCK INPUT
# ========================================
col1, col2 = st.columns([3, 1])
with col1:
    stock = st.text_input("Enter Stock Symbol", "NVDA", key="stock_symbol")
with col2:
    st.write("")
    st.write("")
    fetch_button = st.button("🔍 Fetch Data", use_container_width=True)

# ========================================
# FETCH STOCK DATA
# ========================================
@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_stock_data_alphavantage(symbol, api_key_param):
    """Fetch stock data using Alpha Vantage API"""
    
    try:
        ts = TimeSeries(key=api_key_param, output_format='pandas')
        
        # Get daily adjusted data (full history)
        df, meta_data = ts.get_daily_adjusted(symbol=symbol, outputsize='full')
        
        # Rename columns to standard format
        df = df.rename(columns={
            '1. open': 'Open',
            '2. high': 'High',
            '3. low': 'Low',
            '4. close': 'Close',
            '5. adjusted close': 'Adj Close',
            '6. volume': 'Volume',
            '7. dividend amount': 'Dividend',
            '8. split coefficient': 'Split'
        })
        
        # Sort by date (oldest to newest)
        df = df.sort_index()
        
        # Keep only necessary columns
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        
        return df, meta_data
        
    except Exception as e:
        return None, str(e)

# Fetch data with progress indicator
if 'df' not in st.session_state or fetch_button:
    with st.spinner(f'🔄 Fetching data for {stock}... This may take a moment...'):
        df, meta_info = fetch_stock_data_alphavantage(stock, api_key)
        
        if df is None:
            st.error(f"❌ Failed to fetch data: {meta_info}")
            st.info("💡 Tips:\n- Check if the stock symbol is correct\n- Ensure your API key is valid\n- Free API keys have rate limits (25 requests/day, 5 requests/minute)")
            st.stop()
        
        st.session_state['df'] = df
        st.session_state['stock'] = stock
        time.sleep(1)  # Small delay to respect API rate limits

df = st.session_state.get('df')
stock = st.session_state.get('stock', stock)

if df is None or df.empty:
    st.warning("⚠️ No data loaded. Please enter a stock symbol and click 'Fetch Data'.")
    st.stop()

st.success(f"✅ Successfully loaded {len(df)} days of data for {stock}")

# ========================================
# DISPLAY DATA SUMMARY
# ========================================
with st.expander("📊 View Stock Data Summary", expanded=False):
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Latest Close", f"${df['Close'].iloc[-1]:.2f}")
    with col2:
        st.metric("Highest Ever", f"${df['Close'].max():.2f}")
    with col3:
        st.metric("Lowest Ever", f"${df['Close'].min():.2f}")
    with col4:
        st.metric("Total Days", len(df))
    
    st.dataframe(df.describe(), use_container_width=True)

# ========================================
# CALCULATE EMAs
# ========================================
df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
df['EMA_100'] = df['Close'].ewm(span=100, adjust=False).mean()
df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()

# ========================================
# CANDLESTICK CHARTS
# ========================================
st.subheader("📈 Price Analysis with EMAs")

tab1, tab2 = st.tabs(["20 & 50 Day EMA", "100 & 200 Day EMA"])

with tab1:
    fig1 = go.Figure()
    fig1.add_trace(go.Candlestick(
        x=df.index, 
        open=df['Open'], 
        high=df['High'], 
        low=df['Low'], 
        close=df['Close'], 
        name='Price'
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

with tab2:
    fig2 = go.Figure()
    fig2.add_trace(go.Candlestick(
        x=df.index, 
        open=df['Open'], 
        high=df['High'], 
        low=df['Low'], 
        close=df['Close'], 
        name='Price'
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

# ========================================
# PREPARE DATA FOR PREDICTION
# ========================================
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

# ========================================
# MAKE PREDICTIONS
# ========================================
with st.spinner('🤖 Generating predictions...'):
    y_predicted = model.predict(x_test, verbose=0)

# Reverse scaling
scale_factor = 1 / scaler.scale_[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

# Extract corresponding dates for the test dataset
test_dates = df.index[-len(y_test):]

# ========================================
# PLOT PREDICTIONS
# ========================================
st.subheader("🎯 Model Predictions vs Actual Prices")

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
    title=f"Stock Price Prediction for {stock}",
    xaxis_title="Date",
    yaxis_title="Price ($)",
    xaxis_rangeslider_visible=False,
    height=500,
    hovermode='x unified'
)
st.plotly_chart(fig3, use_container_width=True)

# Calculate accuracy metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(y_test, y_predicted)
rmse = np.sqrt(mean_squared_error(y_test, y_predicted))
r2 = r2_score(y_test, y_predicted)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Mean Absolute Error", f"${mae:.2f}")
with col2:
    st.metric("Root Mean Squared Error", f"${rmse:.2f}")
with col3:
    st.metric("R² Score", f"{r2:.4f}")

# ========================================
# FUTURE PREDICTIONS
# ========================================
st.subheader("🔮 Future Price Predictions")

# Get current price
current_price = float(df['Close'].iloc[-1])
current_date = df.index[-1]
current_day_prediction = float(y_predicted[-1][0])

# Predict next 10 days recursively
last_100_days = input_data[-100:]
future_predictions = []
current_input = last_100_days.reshape(1, 100, 1)

previous_price = y_test[-1]
for _ in range(10):
    next_prediction = model.predict(current_input, verbose=0)[0][0] * scale_factor
    
    # Add realistic fluctuation
    noise = np.random.uniform(-5, 5)
    next_prediction += noise
    
    # Limit price changes
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
    day_label = "today"
else:
    future_dates = pd.date_range(start=df.index[-1] + dt.timedelta(days=1), periods=10)
    target_dates = [current_date] + list(future_dates[:5])
    target_prices = [current_day_prediction] + list(future_predictions[:5])
    top_prediction = current_day_prediction
    day_label = "current day"

# Create prediction table
target_df = pd.DataFrame({
    'Date': [d.strftime('%Y-%m-%d') for d in target_dates],
    'Predicted Price': [f"${price:.2f}" for price in target_prices],
    'Change from Current': [f"${price - current_price:+.2f}" for price in target_prices],
    'Change %': [f"{((price - current_price) / current_price * 100):+.2f}%" for price in target_prices]
})

st.dataframe(target_df, use_container_width=True, hide_index=True)

# ========================================
# PROFIT/LOSS CALCULATOR
# ========================================
st.subheader("💰 Profit/Loss Calculator")

col1, col2 = st.columns(2)

with col1:
    st.metric("Current Stock Price", f"${current_price:.2f}")
    shares_bought = st.number_input(
        "Number of shares to buy:", 
        min_value=1, 
        step=1, 
        value=100
    )

with col2:
    prediction_for_calc = future_predictions[0] if current_time >= midnight else current_day_prediction
    
    st.metric(f"Predicted Price ({day_label})", f"${prediction_for_calc:.2f}")
    
    investment = current_price * shares_bought
    st.info(f"💵 Total Investment: **${investment:,.2f}**")

# Calculate profit/loss
profit_loss = (prediction_for_calc - current_price) * shares_bought
profit_loss_percent = (profit_loss / investment) * 100

col1, col2 = st.columns(2)

with col1:
    if profit_loss > 0:
        st.success(f"🎉 Expected **Profit**: **${profit_loss:.2f}**")
    elif profit_loss < 0:
        st.error(f"⚠️ Expected **Loss**: **${abs(profit_loss):.2f}**")
    else:
        st.info("No significant change expected")

with col2:
    if profit_loss_percent > 0:
        st.success(f"📊 Return: **+{profit_loss_percent:.2f}%**")
    elif profit_loss_percent < 0:
        st.error(f"📊 Return: **{profit_loss_percent:.2f}%**")
    else:
        st.info("📊 Return: **0.00%**")

# ========================================
# DOWNLOAD SECTION
# ========================================
st.subheader("📥 Download Data & Logs")

col1, col2, col3 = st.columns(3)

with col1:
    # Download historical data
    csv_buffer = df.to_csv()
    st.download_button(
        label="📊 Download Historical Data",
        data=csv_buffer,
        file_name=f"{stock}_historical_data.csv",
        mime='text/csv',
        use_container_width=True
    )

with col2:
    # Download predictions
    predictions_df = pd.DataFrame({
        'Date': test_dates,
        'Actual_Price': y_test,
        'Predicted_Price': y_predicted.flatten()
    })
    predictions_csv = predictions_df.to_csv(index=False)
    
    st.download_button(
        label="🎯 Download Predictions",
        data=predictions_csv,
        file_name=f"{stock}_predictions.csv",
        mime='text/csv',
        use_container_width=True
    )

with col3:
    # Save and download prediction log
    prediction_timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    prediction_record = {
        'Timestamp': [prediction_timestamp],
        'Stock': [stock],
        'Current_Price': [current_price],
        'Current_Day_Prediction': [top_prediction]
    }
    
    # Add future predictions
    for i, (date, price) in enumerate(zip(target_dates, target_prices)):
        day_number = i + 1
        prediction_record[f'Date_{day_number}'] = [date.strftime('%Y-%m-%d') if isinstance(date, dt.datetime) else date]
        prediction_record[f'Price_{day_number}'] = [price if isinstance(price, (int, float)) else float(price.replace(', ''))]
    
    prediction_log_df = pd.DataFrame(prediction_record)
    prediction_log_csv = prediction_log_df.to_csv(index=False)
    
    st.download_button(
        label="📋 Download Prediction Log",
        data=prediction_log_csv,
        file_name=f"{stock}_prediction_log_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime='text/csv',
        use_container_width=True
    )

# ========================================
# ADDITIONAL INSIGHTS
# ========================================
with st.expander("📊 Additional Market Insights", expanded=False):
    st.subheader("Price Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("52-Week High", f"${df['Close'].tail(252).max():.2f}")
        st.metric("52-Week Low", f"${df['Close'].tail(252).min():.2f}")
    
    with col2:
        recent_return = ((df['Close'].iloc[-1] - df['Close'].iloc[-30]) / df['Close'].iloc[-30]) * 100
        st.metric("30-Day Return", f"{recent_return:+.2f}%")
        
        volatility = df['Close'].tail(30).std()
        st.metric("30-Day Volatility", f"${volatility:.2f}")
    
    with col3:
        avg_volume = df['Volume'].tail(30).mean()
        st.metric("Avg Daily Volume (30d)", f"{avg_volume:,.0f}")
        
        current_vs_ema200 = ((df['Close'].iloc[-1] - df['EMA_200'].iloc[-1]) / df['EMA_200'].iloc[-1]) * 100
        st.metric("Price vs 200-Day EMA", f"{current_vs_ema200:+.2f}%")

# ========================================
# FOOTER
# ========================================
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p><strong>⚠️ Disclaimer:</strong> This app is for educational purposes only. 
    Stock predictions are based on historical data and machine learning models, 
    which cannot guarantee future performance. Always do your own research before making investment decisions.</p>
    <p><em>Data provided by Alpha Vantage | Predictions powered by LSTM Neural Network</em></p>
</div>
""", unsafe_allow_html=True)

# ========================================
# SESSION STATE INFO (DEBUG)
# ========================================
if st.checkbox("🔧 Show Debug Info", value=False):
    st.write("### Session State")
    st.write(f"Stock: {stock}")
    st.write(f"Data shape: {df.shape}")
    st.write(f"Date range: {df.index[0]} to {df.index[-1]}")
    st.write(f"Model loaded: {model is not None}")
    st.write(f"Predictions generated: {len(y_predicted)}")