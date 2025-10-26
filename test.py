import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import os
import joblib
import pandas_datareader.data as web
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint

# ------------------------
# Streamlit Configuration
# ------------------------
st.set_page_config(page_title="Stock Price Prediction App", layout="wide")
st.title("📈 Stock Price Prediction App")

# ------------------------
# Sidebar Inputs
# ------------------------
st.sidebar.header("Input Parameters")
stock = st.sidebar.text_input("Enter Stock Symbol", "NVDA")

start = dt.datetime(2000, 1, 1)
end = dt.datetime.now()

# ------------------------
# Download Data Function (Stooq only)
# ------------------------
@st.cache_data(ttl=3600)
def get_stock_data(symbol, start, end):
    try:
        df = web.DataReader(symbol, 'stooq', start=start, end=end)
        df = df.sort_index(ascending=True)
        if df.empty:
            return None
        return df
    except:
        return None

df = get_stock_data(stock, start, end)
if df is None:
    st.error(f"❌ No data found for {stock} from Stooq. Check symbol or try later.")
    st.stop()

st.success(f"✅ Data loaded successfully ({len(df)} rows)")
st.subheader("Data Preview")
st.dataframe(df.tail())

# ------------------------
# EMA Calculations
# ------------------------
for span in [20, 50, 100, 200]:
    df[f'EMA_{span}'] = df['Close'].ewm(span=span, adjust=False).mean()

# Plot EMAs with Plotly
def plot_ema(df, ema_short, ema_long):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Candlestick'
    ))
    fig.add_trace(go.Scatter(x=df.index, y=df[ema_short], mode='lines', name=ema_short, line=dict(color='green')))
    fig.add_trace(go.Scatter(x=df.index, y=df[ema_long], mode='lines', name=ema_long, line=dict(color='red')))
    fig.update_layout(xaxis_rangeslider_visible=False)
    return fig

st.subheader("EMA Charts")
st.plotly_chart(plot_ema(df, 'EMA_20', 'EMA_50'))
st.plotly_chart(plot_ema(df, 'EMA_100', 'EMA_200'))

# ------------------------
# Load or Train LSTM Model
# ------------------------
model_file = f"{stock}_lstm.keras"
scaler_file = f"{stock}_scaler.save"

def train_lstm_model(data):
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data)
    
    x_train, y_train = [], []
    for i in range(100, len(scaled_data)):
        x_train.append(scaled_data[i-100:i])
        y_train.append(scaled_data[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(x_train.shape[1],1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    checkpoint = ModelCheckpoint(model_file, save_best_only=True, monitor='loss', mode='min')
    model.fit(x_train, y_train, epochs=5, batch_size=32, callbacks=[checkpoint], verbose=0)
    
    joblib.dump(scaler, scaler_file)
    return model, scaler

if os.path.exists(model_file) and os.path.exists(scaler_file):
    model = load_model(model_file)
    scaler = joblib.load(scaler_file)
    st.success("✅ Model & Scaler loaded successfully")
else:
    st.warning("⚠ Model not found, training a new one...")
    model, scaler = train_lstm_model(df[['Close']].values)
    st.success("✅ Model trained successfully")

# ------------------------
# Prepare Data for Prediction
# ------------------------
data_training = df[['Close']][:int(len(df)*0.7)]
data_testing = df[['Close']][int(len(df)*0.7):]

past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.transform(final_df)

x_test, y_test = [], []
for i in range(100, len(input_data)):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])
x_test, y_test = np.array(x_test), np.array(y_test)

y_pred = model.predict(x_test)
scale_factor = 1 / scaler.scale_[0]
y_pred_scaled = y_pred.flatten() * scale_factor
y_test_scaled = np.array(y_test) * scale_factor
test_dates = df.index[-len(y_test):]

# ------------------------
# Plot Predictions
# ------------------------
st.subheader("Prediction vs Actual")
fig_pred = go.Figure()
fig_pred.add_trace(go.Scatter(x=test_dates, y=y_test_scaled, mode='lines', name='Actual', line=dict(color='blue')))
fig_pred.add_trace(go.Scatter(x=test_dates, y=y_pred_scaled, mode='lines', name='Predicted', line=dict(color='orange')))
fig_pred.update_layout(title="LSTM Predictions vs Actual", xaxis_title="Date", yaxis_title="Price", xaxis_rangeslider_visible=False)
st.plotly_chart(fig_pred)

# ------------------------
# Recursive 10-Day Forecast
# ------------------------
last_100_days = input_data[-100:]
future_predictions = []
current_input = last_100_days.reshape(1,100,1)
prev_price = y_test_scaled[-1]

for _ in range(10):
    next_pred = model.predict(current_input)[0][0] * scale_factor
    noise = np.random.uniform(-5,5)
    next_pred += noise
    next_pred = np.clip(next_pred, prev_price-5, prev_price+5)
    future_predictions.append(next_pred)
    
    next_scaled = next_pred / scale_factor
    current_input = np.append(current_input[:,1:,:], [[[next_scaled]]], axis=1)
    prev_price = next_pred

future_dates = pd.date_range(start=df.index[-1]+dt.timedelta(days=1), periods=10)
future_df = pd.DataFrame({'Date': future_dates.strftime('%Y-%m-%d'), 'Predicted Price': future_predictions})
st.subheader("Next 10 Days Prediction")
st.dataframe(future_df)

# ------------------------
# Single-Day Gain/Loss Calculator
# ------------------------
st.subheader("📈 Gain/Loss Calculator")
current_price = float(df['Close'].iloc[-1])
st.write(f"Current Stock Price: *${current_price:.2f}*")
shares = st.number_input("Number of shares to buy:", min_value=1, value=1, step=1)

pred_for_calc = future_predictions[0]
profit_loss = (pred_for_calc - current_price) * shares
if profit_loss > 0:
    st.success(f"🎉 Expected Profit: ${profit_loss:.2f}")
elif profit_loss < 0:
    st.error(f"⚠ Expected Loss: ${profit_loss:.2f}")
else:
    st.info("No gain or loss expected.")

# ------------------------
# Save Prediction Log
# ------------------------
log_file = f"{stock}_prediction_log.csv"
record = {'Timestamp': dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
          'Stock': stock,
          'Current Price': current_price,
          'Next 10 Days Prediction': future_predictions}
if os.path.exists(log_file):
    df_log = pd.read_csv(log_file)
    df_log = pd.concat([df_log, pd.DataFrame([record])], ignore_index=True)
else:
    df_log = pd.DataFrame([record])
df_log.to_csv(log_file, index=False)
st.subheader("✅ Prediction Log Saved")
st.download_button("Download Prediction Log", data=open(log_file,'rb'), file_name=log_file, mime='text/csv')