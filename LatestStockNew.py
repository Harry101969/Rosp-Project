import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import os
import pandas_datareader.data as web
import plotly.graph_objects as go
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

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
# Download Data Function (Stooq)
# ------------------------
@st.cache_data(ttl=3600)
def get_stock_data(symbol, start, end):
    try:
        df = web.DataReader(symbol, 'stooq', start=start, end=end)
        df = df.sort_index(ascending=True)
        if df.empty:
            return None
        df = df.reset_index()
        df = df[['Date', 'Close']]
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        return df
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

@st.cache_data(ttl=3600)
def get_full_stock_data(symbol, start, end):
    try:
        df = web.DataReader(symbol, 'stooq', start=start, end=end)
        df = df.sort_index(ascending=True)
        if df.empty:
            return None
        return df
    except:
        return None

df_full = get_full_stock_data(stock, start, end)
df = get_stock_data(stock, start, end)

if df is None or df.empty:
    st.error(f"❌ No data found for {stock} from Stooq. Check symbol or try later.")
    st.stop()

st.success(f"✅ Data loaded successfully ({len(df)} rows)")
st.subheader("Data Preview")
st.dataframe(df.tail(10))

# ------------------------
# EMA Calculations
# ------------------------
if df_full is not None:
    for span in [20, 50, 100, 200]:
        df_full[f'EMA_{span}'] = df_full['Close'].ewm(span=span, adjust=False).mean()

    def plot_ema(df, ema_short, ema_long):
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df.index, 
            open=df['Open'], 
            high=df['High'], 
            low=df['Low'], 
            close=df['Close'], 
            name='Candlestick'
        ))
        fig.add_trace(go.Scatter(
            x=df.index, 
            y=df[ema_short], 
            mode='lines', 
            name=ema_short, 
            line=dict(color='green')
        ))
        fig.add_trace(go.Scatter(
            x=df.index, 
            y=df[ema_long], 
            mode='lines', 
            name=ema_long, 
            line=dict(color='red')
        ))
        fig.update_layout(xaxis_rangeslider_visible=False, height=500)
        return fig

    st.subheader("EMA Charts")
    st.plotly_chart(plot_ema(df_full, 'EMA_20', 'EMA_50'), use_container_width=True)
    st.plotly_chart(plot_ema(df_full, 'EMA_100', 'EMA_200'), use_container_width=True)

# ------------------------
# Windowed DataFrame Creation
# ------------------------
def df_to_windowed_df(dataframe, n=3):
    """Convert dataframe to windowed format for LSTM training"""
    dates = []
    X, Y = [], []
    
    for i in range(n, len(dataframe)):
        df_subset = dataframe.iloc[i-n:i+1]
        
        if len(df_subset) != n + 1:
            continue
            
        values = df_subset['Close'].values
        x, y = values[:-1], values[-1]
        
        dates.append(dataframe.index[i])
        X.append(x)
        Y.append(y)
    
    ret_df = pd.DataFrame({'Target Date': dates})
    
    X = np.array(X)
    for i in range(n):
        ret_df[f'Target-{n - i}'] = X[:, i]
    
    ret_df['Target'] = Y
    
    return ret_df

def windowed_df_to_date_X_y(windowed_dataframe):
    """Convert windowed dataframe to dates, X, y format"""
    df_as_np = windowed_dataframe.to_numpy()
    
    dates = df_as_np[:, 0]
    middle_matrix = df_as_np[:, 1:-1]
    X = middle_matrix.reshape((len(dates), middle_matrix.shape[1], 1))
    Y = df_as_np[:, -1]
    
    return dates, X.astype(np.float32), Y.astype(np.float32)

# ------------------------
# Load or Train LSTM Model
# ------------------------
model_file = f"{stock}_lstm_windowed.keras"
window_size = 3

def train_lstm_model(df, window_size=3):
    """Train LSTM model using windowed approach"""
    st.info("🔄 Training new model... This may take a few minutes.")
    
    windowed_df = df_to_windowed_df(df, n=window_size)
    dates, X, y = windowed_df_to_date_X_y(windowed_df)
    
    q_80 = int(len(dates) * 0.8)
    q_90 = int(len(dates) * 0.9)
    
    X_train, y_train = X[:q_80], y[:q_80]
    X_val, y_val = X[q_80:q_90], y[q_80:q_90]
    X_test, y_test = X[q_90:], y[q_90:]
    
    model = Sequential([
        LSTM(64, input_shape=(window_size, 1)),
        Dense(32, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    
    model.compile(
        loss='mse',
        optimizer=Adam(learning_rate=0.001),
        metrics=['mean_absolute_error']
    )
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    epochs = 100
    for epoch in range(epochs):
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=1,
            batch_size=32,
            verbose=0
        )
        
        progress = (epoch + 1) / epochs
        progress_bar.progress(progress)
        status_text.text(f"Training: {epoch + 1}/{epochs} epochs - Loss: {history.history['loss'][0]:.6f}")
    
    progress_bar.empty()
    status_text.empty()
    
    model.save(model_file)
    
    return model, dates, X, y, q_80, q_90

if os.path.exists(model_file):
    model = load_model(model_file)
    st.success("✅ Model loaded successfully")
    
    windowed_df = df_to_windowed_df(df, n=window_size)
    dates, X, y = windowed_df_to_date_X_y(windowed_df)
    q_80 = int(len(dates) * 0.8)
    q_90 = int(len(dates) * 0.9)
else:
    st.warning("⚠ Model not found, training a new one...")
    model, dates, X, y, q_80, q_90 = train_lstm_model(df, window_size)
    st.success("✅ Model trained successfully")

dates_train, X_train, y_train = dates[:q_80], X[:q_80], y[:q_80]
dates_val, X_val, y_val = dates[q_80:q_90], X[q_80:q_90], y[q_80:q_90]
dates_test, X_test, y_test = dates[q_90:], X[q_90:], y[q_90:]

# ------------------------
# Make Predictions
# ------------------------
train_predictions = model.predict(X_train, verbose=0).flatten()
val_predictions = model.predict(X_val, verbose=0).flatten()
test_predictions = model.predict(X_test, verbose=0).flatten()

# ------------------------
# Plot Predictions
# ------------------------
st.subheader("Model Performance: Prediction vs Actual")

fig_pred = go.Figure()

fig_pred.add_trace(go.Scatter(
    x=dates_train, 
    y=y_train, 
    mode='lines', 
    name='Training Actual', 
    line=dict(color='blue', width=2)
))
fig_pred.add_trace(go.Scatter(
    x=dates_train, 
    y=train_predictions, 
    mode='lines', 
    name='Training Predicted', 
    line=dict(color='lightblue', dash='dot')
))

fig_pred.add_trace(go.Scatter(
    x=dates_val, 
    y=y_val, 
    mode='lines', 
    name='Validation Actual', 
    line=dict(color='green', width=2)
))
fig_pred.add_trace(go.Scatter(
    x=dates_val, 
    y=val_predictions, 
    mode='lines', 
    name='Validation Predicted', 
    line=dict(color='lightgreen', dash='dot')
))

fig_pred.add_trace(go.Scatter(
    x=dates_test, 
    y=y_test, 
    mode='lines', 
    name='Test Actual', 
    line=dict(color='red', width=2)
))
fig_pred.add_trace(go.Scatter(
    x=dates_test, 
    y=test_predictions, 
    mode='lines', 
    name='Test Predicted', 
    line=dict(color='orange', dash='dot')
))

fig_pred.update_layout(
    title="LSTM Predictions vs Actual",
    xaxis_title="Date",
    yaxis_title="Price",
    xaxis_rangeslider_visible=False,
    height=500
)
st.plotly_chart(fig_pred, use_container_width=True)

# ------------------------
# Model Performance Metrics
# ------------------------
st.subheader("📊 Model Performance Metrics")

col1, col2, col3 = st.columns(3)

with col1:
    st.write("**Training Set:**")
    train_mse = mean_squared_error(y_train, train_predictions)
    train_mae = mean_absolute_error(y_train, train_predictions)
    train_r2 = r2_score(y_train, train_predictions)
    train_mape = mean_absolute_percentage_error(y_train, train_predictions) * 100
    train_accuracy = 100 - train_mape
    st.metric("MSE", f"{train_mse:.4f}")
    st.metric("MAE", f"{train_mae:.4f}")
    st.metric("R² Score", f"{train_r2:.4f}")
    st.metric("Accuracy", f"{train_accuracy:.2f}%")
    st.metric("Error Rate", f"{train_mape:.2f}%")

with col2:
    st.write("**Validation Set:**")
    val_mse = mean_squared_error(y_val, val_predictions)
    val_mae = mean_absolute_error(y_val, val_predictions)
    val_r2 = r2_score(y_val, val_predictions)
    val_mape = mean_absolute_percentage_error(y_val, val_predictions) * 100
    val_accuracy = 100 - val_mape
    st.metric("MSE", f"{val_mse:.4f}")
    st.metric("MAE", f"{val_mae:.4f}")
    st.metric("R² Score", f"{val_r2:.4f}")
    st.metric("Accuracy", f"{val_accuracy:.2f}%")
    st.metric("Error Rate", f"{val_mape:.2f}%")

with col3:
    st.write("**Test Set:**")
    test_mse = mean_squared_error(y_test, test_predictions)
    test_mae = mean_absolute_error(y_test, test_predictions)
    test_r2 = r2_score(y_test, test_predictions)
    test_mape = mean_absolute_percentage_error(y_test, test_predictions) * 100
    test_accuracy = 100 - test_mape
    st.metric("MSE", f"{test_mse:.4f}")
    st.metric("MAE", f"{test_mae:.4f}")
    st.metric("R² Score", f"{test_r2:.4f}")
    st.metric("Accuracy", f"{test_accuracy:.2f}%")
    st.metric("Error Rate", f"{test_mape:.2f}%")

# ------------------------
# Improved Recursive 10-Day Forecast
# ------------------------
st.subheader("📅 Future Price Prediction (Next 10 Days)")

# Get the last known actual price from data
current_price = float(df['Close'].iloc[-1])
current_date = df.index[-1]

# Get last window for prediction
last_window = X_train[-1].copy()

# Predict price changes instead of absolute prices
future_predictions = []
recursive_dates = []

# Base price starts from current actual price
base_price = current_price

for i in range(10):
    # Get model's predicted price
    model_prediction = model.predict(np.array([last_window]), verbose=0)[0][0]
    
    # Calculate the predicted change from the last value in window
    last_window_price = last_window[-1][0]
    predicted_change = model_prediction - last_window_price
    
    # Add realistic noise (smaller percentage-based noise)
    noise_factor = np.random.uniform(-0.005, 0.005)  # ±0.5% noise
    noise = base_price * noise_factor
    
    # Apply the predicted change with noise to base price
    next_price = base_price + predicted_change + noise
    
    # Constrain the change to be realistic (max ±5% per day)
    max_change = base_price * 0.05
    next_price = np.clip(next_price, base_price - max_change, base_price + max_change)
    
    future_predictions.append(next_price)
    current_date += dt.timedelta(days=1)
    recursive_dates.append(current_date)
    
    # Update base price and window for next iteration
    base_price = next_price
    last_window = np.roll(last_window, -1, axis=0)
    last_window[-1] = next_price

# Create future predictions dataframe
future_df = pd.DataFrame({
    'Date': [d.strftime('%Y-%m-%d') for d in recursive_dates],
    'Predicted Price': [f"${p:.2f}" for p in future_predictions]
})

st.dataframe(future_df, use_container_width=True)

# Plot future predictions
fig_future = go.Figure()

# Historical data (last 60 days)
recent_data = df['Close'].tail(60)
fig_future.add_trace(go.Scatter(
    x=recent_data.index,
    y=recent_data.values,
    mode='lines',
    name='Historical Price',
    line=dict(color='blue', width=2)
))

# Current price marker
fig_future.add_trace(go.Scatter(
    x=[df.index[-1]],
    y=[current_price],
    mode='markers',
    name='Current Price',
    marker=dict(color='red', size=12, symbol='circle')
))

# Future predictions
fig_future.add_trace(go.Scatter(
    x=recursive_dates,
    y=future_predictions,
    mode='lines+markers',
    name='Predicted Price',
    line=dict(color='orange', dash='dash', width=2),
    marker=dict(size=8)
))

fig_future.update_layout(
    title="10-Day Price Forecast",
    xaxis_title="Date",
    yaxis_title="Price",
    xaxis_rangeslider_visible=False,
    height=500
)
st.plotly_chart(fig_future, use_container_width=True)

# ------------------------
# Gain/Loss Calculator
# ------------------------
st.subheader("📈 Gain/Loss Calculator")

next_day_prediction = future_predictions[0]
profit_loss_pct = ((next_day_prediction - current_price) / current_price) * 100

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Current Price", f"${current_price:.2f}")

with col2:
    st.metric(
        "Predicted Price (Tomorrow)", 
        f"${next_day_prediction:.2f}", 
        f"{profit_loss_pct:+.2f}%"
    )

with col3:
    shares = st.number_input("Number of shares:", min_value=1, value=100, step=1)

profit_loss = (next_day_prediction - current_price) * shares

st.write("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Investment Amount", f"${current_price * shares:,.2f}")

with col2:
    st.metric("Expected Value (Tomorrow)", f"${next_day_prediction * shares:,.2f}")

with col3:
    if profit_loss > 0:
        st.metric("Expected Profit/Loss", f"${profit_loss:.2f}", f"+${profit_loss:.2f}")
    else:
        st.metric("Expected Profit/Loss", f"${profit_loss:.2f}", f"${profit_loss:.2f}")

if profit_loss > 0:
    st.success(f"🎉 Expected Profit for {shares} shares: **${profit_loss:.2f}** ({profit_loss_pct:+.2f}%)")
elif profit_loss < 0:
    st.error(f"⚠️ Expected Loss for {shares} shares: **${abs(profit_loss):.2f}** ({profit_loss_pct:.2f}%)")
else:
    st.info("📊 No significant gain or loss expected.")

# ------------------------
# Save Prediction Log with Accuracy Metrics
# ------------------------
log_file = f"{stock}_prediction_log.csv"

record = {
    'Timestamp': dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    'Stock': stock,
    'Current_Price': current_price,
    'Next_Day_Prediction': next_day_prediction,
    'Expected_Change_Pct': profit_loss_pct,
    'Model_Accuracy': f"{test_accuracy:.2f}%",
    'Model_Error_Rate': f"{test_mape:.2f}%",
    'Model_MAE': f"{test_mae:.4f}",
    'Model_MSE': f"{test_mse:.4f}",
    'Model_R2_Score': f"{test_r2:.4f}",
}

# Add 10-day predictions
for i, (date, price) in enumerate(zip(recursive_dates, future_predictions)):
    record[f'Day_{i+1}_Date'] = date.strftime('%Y-%m-%d')
    record[f'Day_{i+1}_Price'] = f"{price:.2f}"
    if i > 0:
        daily_change = ((price - future_predictions[i-1]) / future_predictions[i-1]) * 100
        record[f'Day_{i+1}_Change_Pct'] = f"{daily_change:+.2f}%"
    else:
        daily_change = profit_loss_pct
        record[f'Day_{i+1}_Change_Pct'] = f"{daily_change:+.2f}%"

if os.path.exists(log_file):
    df_log = pd.read_csv(log_file)
    df_log = pd.concat([df_log, pd.DataFrame([record])], ignore_index=True)
else:
    df_log = pd.DataFrame([record])

df_log.to_csv(log_file, index=False)

st.subheader("✅ Prediction Log Saved")
st.success(f"Prediction saved at {record['Timestamp']}")

# Display model accuracy info
col1, col2 = st.columns(2)
with col1:
    st.info(f"📊 Model Accuracy: **{test_accuracy:.2f}%**")
with col2:
    st.info(f"⚠️ Model Error Rate: **{test_mape:.2f}%**")

with open(log_file, 'rb') as f:
    st.download_button(
        label="📥 Download Prediction Log",
        data=f,
        file_name=log_file,
        mime='text/csv'
    )

# Download dataset
csv_file_path = f"{stock}_dataset.csv"
df.to_csv(csv_file_path)
with open(csv_file_path, 'rb') as f:
    st.download_button(
        label="📥 Download Historical Dataset",
        data=f,
        file_name=csv_file_path,
        mime='text/csv'
    )

# ------------------------
# Retrain Model Button
# ------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("Model Management")

if st.sidebar.button("🔄 Retrain Model"):
    if os.path.exists(model_file):
        os.remove(model_file)
    st.sidebar.info("Retraining model...")
    st.rerun()

if os.path.exists(model_file):
    st.sidebar.success(f"✅ Model loaded: {model_file}")
    file_size = os.path.getsize(model_file) / 1024
    st.sidebar.info(f"Model size: {file_size:.2f} KB")
    st.sidebar.metric("Model Accuracy", f"{test_accuracy:.2f}%")
    st.sidebar.metric("Error Rate", f"{test_mape:.2f}%")