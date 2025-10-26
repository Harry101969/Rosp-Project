# # import numpy as np
# # import pandas as pd
# # import streamlit as st
# # import datetime as dt
# # import yfinance as yf
# # import plotly.graph_objects as go
# # from sklearn.preprocessing import MinMaxScaler
# # from tensorflow.keras.models import load_model
# # import os

# # # Load the trained model
# # model = load_model('stockpricemodel.keras')

# # st.title("Stock Price Prediction App")

# # # User input for stock symbol
# # stock = st.text_input("Enter Stock Symbol (e.g., NVDA)", "NVDA")

# # # Define start and end dates
# # start = dt.datetime(2000, 1, 1)
# # end = dt.datetime.now()

# # # ✅ Error Handling for Invalid Symbols
# # try:
# #     df = yf.download(stock, start=start, end=end)
# #     if df.empty:
# #         st.error(f"❌ No data found for symbol '{stock}'. Please check the symbol and try again.")
# #         st.stop()
# # except Exception as e:
# #     st.error(f"❌ Error retrieving data for symbol '{stock}': {e}")
# #     st.stop()

# # # Display dataset summary
# # st.subheader("Stock Data Summary")
# # st.write(df.describe())

# # # Calculate Exponential Moving Averages (EMA)
# # df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
# # df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
# # df['EMA_100'] = df['Close'].ewm(span=100, adjust=False).mean()
# # df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()

# # # Candlestick Chart with 20 & 50 EMA
# # st.subheader("Closing Price with 20 & 50 Days EMA")
# # fig1 = go.Figure()
# # fig1.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Candlestick'))
# # fig1.add_trace(go.Scatter(x=df.index, y=df['EMA_20'], mode='lines', name='EMA 20', line=dict(color='green')))
# # fig1.add_trace(go.Scatter(x=df.index, y=df['EMA_50'], mode='lines', name='EMA 50', line=dict(color='red')))
# # fig1.update_layout(xaxis_rangeslider_visible=False)
# # st.plotly_chart(fig1)

# # # Candlestick Chart with 100 & 200 EMA
# # st.subheader("Closing Price with 100 & 200 Days EMA")
# # fig2 = go.Figure()
# # fig2.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Candlestick'))
# # fig2.add_trace(go.Scatter(x=df.index, y=df['EMA_100'], mode='lines', name='EMA 100', line=dict(color='blue')))
# # fig2.add_trace(go.Scatter(x=df.index, y=df['EMA_200'], mode='lines', name='EMA 200', line=dict(color='purple')))
# # fig2.update_layout(xaxis_rangeslider_visible=False)
# # st.plotly_chart(fig2)

# # # Data Preparation
# # data_training = df[['Close']][:int(len(df) * 0.70)]
# # data_testing = df[['Close']][int(len(df) * 0.70):]
# # scaler = MinMaxScaler(feature_range=(0, 1))
# # data_training_array = scaler.fit_transform(data_training)

# # # Prepare test data
# # past_100_days = data_training.tail(100)
# # final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
# # input_data = scaler.transform(final_df)

# # x_test, y_test = [], []
# # for i in range(100, input_data.shape[0]):
# #     x_test.append(input_data[i - 100:i])
# #     y_test.append(input_data[i, 0])
# # x_test, y_test = np.array(x_test), np.array(y_test)

# # # Make predictions
# # y_predicted = model.predict(x_test)

# # # Reverse scaling
# # scale_factor = 1 / scaler.scale_[0]
# # y_predicted = y_predicted * scale_factor
# # y_test = y_test * scale_factor

# # # Extract corresponding dates for the test dataset
# # test_dates = df.index[-len(y_test):]

# # # Plot Predictions
# # st.subheader("Prediction vs Original Trend")
# # fig3 = go.Figure()
# # fig3.add_trace(go.Scatter(
# #     x=test_dates, y=y_test, mode='lines', name="Actual Price", line=dict(color="blue")
# # ))
# # fig3.add_trace(go.Scatter(
# #     x=test_dates, y=y_predicted.flatten(), mode='lines', name="Predicted Price", line=dict(color="orange")
# # ))

# # fig3.update_layout(title="Stock Price Prediction",
# #                    xaxis_title="Date",
# #                    yaxis_title="Price",
# #                    xaxis_rangeslider_visible=False)
# # st.plotly_chart(fig3)

# # # Get last closing price correctly as float
# # current_price = df['Close'].iloc[-1]
# # current_price = float(current_price) if isinstance(current_price, (pd.Series, pd.DataFrame)) else current_price

# # # Get current date (the last date in our dataset)
# # current_date = df.index[-1]

# # # Get the prediction for the current day (the last prediction in y_predicted)
# # current_day_prediction = float(y_predicted[-1][0])

# # # 🚀 Predict the next 10 days recursively
# # last_100_days = input_data[-100:]
# # future_predictions = []
# # current_input = last_100_days.reshape(1, 100, 1)

# # # Predict next 10 days
# # previous_price = y_test[-1]
# # for _ in range(10):
# #     next_prediction = model.predict(current_input)[0][0] * scale_factor
    
# #     # ✅ Add realistic fluctuation (including falls)
# #     noise = np.random.uniform(-5, 5)  # Simulate realistic fluctuation
# #     next_prediction += noise
    
# #     # ✅ Limit the difference between consecutive prices to ±20
# #     if len(future_predictions) > 0:
# #         next_prediction = np.clip(next_prediction, future_predictions[-1] - 5.043, future_predictions[-1] + 5.597)
# #     else:
# #         next_prediction = np.clip(next_prediction, previous_price - 5.673, previous_price + 5.068)
    
# #     future_predictions.append(next_prediction)
    
# #     # Update input for next prediction
# #     next_scaled = next_prediction / scale_factor
# #     current_input = np.append(current_input[:, 1:, :], [[[next_scaled]]], axis=1)

# # # Reverse scaling for target values
# # future_predictions = np.array(future_predictions)

# # # Get the current time
# # now = dt.datetime.now()
# # current_time = now.time()
# # midnight = dt.time(0, 1)  # 12:01 AM

# # # Generate dates for predictions
# # # If current time is after midnight (00:01), shift the predictions forward
# # if current_time >= midnight:
# #     # Start future dates from today (current day)
# #     future_dates = pd.date_range(start=now.date(), periods=6)
# #     # Use only future predictions for the table
# #     target_dates = list(future_dates)
# #     target_prices = list(future_predictions[:6])
# #     # The first prediction in the table is the current day prediction
# #     top_prediction = target_prices[0]
# # else:
# #     # Include current day + future days
# #     future_dates = pd.date_range(start=df.index[-1] + dt.timedelta(days=1), periods=10)
# #     target_dates = [current_date] + list(future_dates[:5])
# #     target_prices = [current_day_prediction] + list(future_predictions[:5])
# #     # The first prediction in the table is the current day prediction
# #     top_prediction = current_day_prediction

# # # Create the table
# # target_df = pd.DataFrame({
# #     'Date': [d.strftime('%Y-%m-%d') for d in target_dates],
# #     'Predicted Target Price': target_prices
# # })

# # # Display table
# # st.subheader("Predicted Prices (Next 6 Days)")
# # st.write(target_df)

# # # Get ticker information for currency
# # ticker_info = yf.Ticker(stock).info
# # currency = ticker_info.get("currency", "USD") 
# # currency_symbols = {
# #     "USD": "$",
# #     "INR": "₹",
# #     "EUR": "€",
# #     "GBP": "£",
# #     "JPY": "¥",
# #     "AUD": "A$",
# #     "CAD": "C$",
# #     "CHF": "CHF",
# #     "SGD": "S$",
# #     "HKD": "HK$",
# # }

# # # Get the appropriate currency symbol
# # currency_symbol = currency_symbols.get(currency, currency)
# # st.subheader("📈 Single-Day Gain/Loss Calculator")

# # formatted_price = f"Current Stock Price: **{currency_symbol}{current_price}**"
# # st.write(formatted_price)

# # shares_bought = st.number_input("Enter the number of shares you want to buy:", min_value=1, step=1, value=1)

# # # Determine which prediction to use based on time
# # if current_time >= midnight:
# #     # After midnight, use first future prediction
# #     prediction_for_calc = future_predictions[0]
# #     day_label = "today"
# # else:
# #     # Before midnight, use current day prediction
# #     prediction_for_calc = current_day_prediction
# #     day_label = "current day"

# # # Calculate profit/loss
# # profit_loss = (prediction_for_calc - current_price) * shares_bought

# # if profit_loss > 0:
# #     st.success(f"🎉 Expected **Profit** for {day_label}: **{currency_symbol}{profit_loss:.2f}**")
# # elif profit_loss < 0:
# #     st.error(f"⚠️ Expected **Loss** for {day_label}: **{currency_symbol}{profit_loss:.2f}**")
# # else:
# #     st.info(f"No gain or loss expected for {day_label}.")

# # # ✅ Download dataset
# # csv_file_path = f"{stock}_dataset.csv"
# # df.to_csv(csv_file_path)
# # st.download_button(label="Download Dataset as CSV", data=open(csv_file_path, 'rb'), file_name=csv_file_path, mime='text/csv')

# # # ✅ Save prediction data to CSV
# # # Create a unique filename based on the stock symbol
# # prediction_data_filename = f"{stock}_prediction_log.csv"

# # # Create prediction record
# # prediction_timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
# # prediction_record = {
# #     'Timestamp': prediction_timestamp,
# #     'Stock': stock,
# #     'Current_Price': current_price,
# #     'Current_Day_Prediction': top_prediction  # Using the top prediction from the table
# # }

# # # For future dates
# # for i, (date, price) in enumerate(zip(target_dates, target_prices)):
# #     day_number = i+1
# #     prediction_record[f'Date_{day_number}'] = date.strftime('%Y-%m-%d') if isinstance(date, dt.datetime) else date
# #     prediction_record[f'Price_{day_number}'] = price

# # # Check if file exists to append or create new
# # if os.path.exists(prediction_data_filename):
# #     # Append to existing file
# #     existing_data = pd.read_csv(prediction_data_filename)
# #     updated_data = pd.concat([existing_data, pd.DataFrame([prediction_record])], ignore_index=True)
# #     updated_data.to_csv(prediction_data_filename, index=False)
# # else:
# #     # Create new file
# #     pd.DataFrame([prediction_record]).to_csv(prediction_data_filename, index=False)

# # st.subheader("✅ Prediction Log")
# # st.success(f"Prediction record saved to {prediction_data_filename} at {prediction_timestamp}")
# # st.info(f"Current day prediction (top of table): {currency_symbol}{top_prediction:.2f}")

# # # Download prediction log
# # with open(prediction_data_filename, 'rb') as file:
# #     st.download_button(
# #         label="Download Prediction Log",
# #         data=file,
# #         file_name=prediction_data_filename,
# #         mime='text/csv'
# #     )
    
# import numpy as np
# import pandas as pd
# import streamlit as st
# import datetime as dt
# import yfinance as yf
# import plotly.graph_objects as go
# from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras.models import load_model
# import os

# # Load the trained model
# model = load_model('stockpricemodel.keras')

# st.title("Stock Price Prediction App")

# # User input for stock symbol
# stock = st.text_input("Enter Stock Symbol (e.g., NVDA)", "NVDA")

# # Define start and end dates
# start = dt.datetime(2000, 1, 1)
# end = dt.datetime.now()

# # ✅ Error Handling for Invalid Symbols
# try:
#     df = yf.download(stock, start=start, end=end)
#     if df.empty:
#         st.error(f"❌ No data found for symbol '{stock}'. Please check the symbol and try again.")
#         st.stop()
# except Exception as e:
#     st.error(f"❌ Error retrieving data for symbol '{stock}': {e}")
#     st.stop()

# # Display dataset summary
# st.subheader("Stock Data Summary")
# st.write(df.describe())

# # Calculate Exponential Moving Averages (EMA)
# df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
# df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
# df['EMA_100'] = df['Close'].ewm(span=100, adjust=False).mean()
# df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()

# # Candlestick Chart with 20 & 50 EMA
# st.subheader("Closing Price with 20 & 50 Days EMA")
# fig1 = go.Figure()
# fig1.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Candlestick'))
# fig1.add_trace(go.Scatter(x=df.index, y=df['EMA_20'], mode='lines', name='EMA 20', line=dict(color='green')))
# fig1.add_trace(go.Scatter(x=df.index, y=df['EMA_50'], mode='lines', name='EMA 50', line=dict(color='red')))
# fig1.update_layout(xaxis_rangeslider_visible=False)
# st.plotly_chart(fig1)

# # Candlestick Chart with 100 & 200 EMA
# st.subheader("Closing Price with 100 & 200 Days EMA")
# fig2 = go.Figure()
# fig2.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Candlestick'))
# fig2.add_trace(go.Scatter(x=df.index, y=df['EMA_100'], mode='lines', name='EMA 100', line=dict(color='blue')))
# fig2.add_trace(go.Scatter(x=df.index, y=df['EMA_200'], mode='lines', name='EMA 200', line=dict(color='purple')))
# fig2.update_layout(xaxis_rangeslider_visible=False)
# st.plotly_chart(fig2)

# # Data Preparation
# data_training = df[['Close']][:int(len(df) * 0.70)]
# data_testing = df[['Close']][int(len(df) * 0.70):]
# scaler = MinMaxScaler(feature_range=(0, 1))
# data_training_array = scaler.fit_transform(data_training)

# # Prepare test data
# past_100_days = data_training.tail(100)
# final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
# input_data = scaler.transform(final_df)

# x_test, y_test = [], []
# for i in range(100, input_data.shape[0]):
#     x_test.append(input_data[i - 100:i])
#     y_test.append(input_data[i, 0])
# x_test, y_test = np.array(x_test), np.array(y_test)

# # Make predictions
# y_predicted = model.predict(x_test)

# # Reverse scaling
# scale_factor = 1 / scaler.scale_[0]
# y_predicted = y_predicted * scale_factor
# y_test = y_test * scale_factor

# # Extract corresponding dates for the test dataset
# test_dates = df.index[-len(y_test):]

# # Plot Predictions
# st.subheader("Prediction vs Original Trend")
# fig3 = go.Figure()
# fig3.add_trace(go.Scatter(
#     x=test_dates, y=y_test, mode='lines', name="Actual Price", line=dict(color="blue")
# ))
# fig3.add_trace(go.Scatter(
#     x=test_dates, y=y_predicted.flatten(), mode='lines', name="Predicted Price", line=dict(color="orange")
# ))

# fig3.update_layout(title="Stock Price Prediction",
#                    xaxis_title="Date",
#                    yaxis_title="Price",
#                    xaxis_rangeslider_visible=False)
# st.plotly_chart(fig3)

# # Get last closing price correctly as float
# current_price = df['Close'].iloc[-1]
# current_price = float(current_price) if isinstance(current_price, (pd.Series, pd.DataFrame)) else current_price

# # Get current date (the last date in our dataset)
# current_date = df.index[-1]

# # Get the prediction for the current day (the last prediction in y_predicted)
# current_day_prediction = float(y_predicted[-1][0])

# # 🚀 Predict the next 10 days recursively
# last_100_days = input_data[-100:]
# future_predictions = []
# current_input = last_100_days.reshape(1, 100, 1)

# # Predict next 10 days
# previous_price = y_test[-1]
# for _ in range(10):
#     next_prediction = model.predict(current_input)[0][0] * scale_factor
    
#     # ✅ Add realistic fluctuation (including falls)
#     noise = np.random.uniform(-5, 5)  # Simulate realistic fluctuation
#     next_prediction += noise
    
#     # ✅ Limit the difference between consecutive prices to ±20
#     if len(future_predictions) > 0:
#         next_prediction = np.clip(next_prediction, future_predictions[-1] - 5.043, future_predictions[-1] + 5.597)
#     else:
#         next_prediction = np.clip(next_prediction, previous_price - 5.673, previous_price + 5.068)
    
#     future_predictions.append(next_prediction)
    
#     # Update input for next prediction
#     next_scaled = next_prediction / scale_factor
#     current_input = np.append(current_input[:, 1:, :], [[[next_scaled]]], axis=1)

# # Reverse scaling for target values
# future_predictions = np.array(future_predictions)

# # Get the current time
# now = dt.datetime.now()
# current_time = now.time()
# midnight = dt.time(0, 1)  # 12:01 AM

# # Generate dates for predictions
# # If current time is after midnight (00:01), shift the predictions forward
# if current_time >= midnight:
#     # Start future dates from today (current day)
#     future_dates = pd.date_range(start=now.date(), periods=6)
#     # Use only future predictions for the table
#     target_dates = list(future_dates)
#     target_prices = list(future_predictions[:6])
#     # The first prediction in the table is the current day prediction
#     top_prediction = target_prices[0]
# else:
#     # Include current day + future days
#     future_dates = pd.date_range(start=df.index[-1] + dt.timedelta(days=1), periods=10)
#     target_dates = [current_date] + list(future_dates[:5])
#     target_prices = [current_day_prediction] + list(future_predictions[:5])
#     # The first prediction in the table is the current day prediction
#     top_prediction = current_day_prediction

# # Create the table
# target_df = pd.DataFrame({
#     'Date': [d.strftime('%Y-%m-%d') for d in target_dates],
#     'Predicted Target Price': target_prices
# })

# # Display table
# st.subheader("Predicted Prices (Next 6 Days)")
# st.write(target_df)

# # Get ticker information for currency
# ticker_info = yf.Ticker(stock).info
# currency = ticker_info.get("currency", "USD") 
# currency_symbols = {
#     "USD": "$",
#     "INR": "₹",
#     "EUR": "€",
#     "GBP": "£",
#     "JPY": "¥",
#     "AUD": "A$",
#     "CAD": "C$",
#     "CHF": "CHF",
#     "SGD": "S$",
#     "HKD": "HK$",
# }

# # Get the appropriate currency symbol
# currency_symbol = currency_symbols.get(currency, currency)
# st.subheader("📈 Single-Day Gain/Loss Calculator")

# formatted_price = f"Current Stock Price: **{currency_symbol}{current_price}**"
# st.write(formatted_price)

# shares_bought = st.number_input("Enter the number of shares you want to buy:", min_value=1, step=1, value=1)

# # Determine which prediction to use based on time
# if current_time >= midnight:
#     # After midnight, use first future prediction
#     prediction_for_calc = future_predictions[0]
#     day_label = "today"
# else:
#     # Before midnight, use current day prediction
#     prediction_for_calc = current_day_prediction
#     day_label = "current day"

# # Calculate profit/loss
# profit_loss = (prediction_for_calc - current_price) * shares_bought

# if profit_loss > 0:
#     st.success(f"🎉 Expected **Profit** for {day_label}: **{currency_symbol}{profit_loss:.2f}**")
# elif profit_loss < 0:
#     st.error(f"⚠️ Expected **Loss** for {day_label}: **{currency_symbol}{profit_loss:.2f}**")
# else:
#     st.info(f"No gain or loss expected for {day_label}.")

# # ✅ Download dataset
# csv_file_path = f"{stock}_dataset.csv"
# df.to_csv(csv_file_path)
# st.download_button(label="Download Dataset as CSV", data=open(csv_file_path, 'rb'), file_name=csv_file_path, mime='text/csv')

# # ✅ Save prediction data to CSV
# # Create a unique filename based on the stock symbol
# prediction_data_filename = f"{stock}_prediction_log.csv"

# # Create prediction record
# prediction_timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
# prediction_record = {
#     'Timestamp': prediction_timestamp,
#     'Stock': stock,
#     'Current_Price': current_price,
#     'Current_Day_Prediction': top_prediction  # Using the top prediction from the table
# }

# # For future dates
# for i, (date, price) in enumerate(zip(target_dates, target_prices)):
#     day_number = i+1
#     prediction_record[f'Date_{day_number}'] = date.strftime('%Y-%m-%d') if isinstance(date, dt.datetime) else date
#     prediction_record[f'Price_{day_number}'] = price

# # Check if file exists to append or create new
# if os.path.exists(prediction_data_filename):
#     # Append to existing file
#     existing_data = pd.read_csv(prediction_data_filename)
#     updated_data = pd.concat([existing_data, pd.DataFrame([prediction_record])], ignore_index=True)
#     updated_data.to_csv(prediction_data_filename, index=False)
# else:
#     # Create new file
#     pd.DataFrame([prediction_record]).to_csv(prediction_data_filename, index=False)

# st.subheader("✅ Prediction Log")
# st.success(f"Prediction record saved to {prediction_data_filename} at {prediction_timestamp}")
# st.info(f"Current day prediction (top of table): {currency_symbol}{top_prediction:.2f}")

# # Download prediction log
# with open(prediction_data_filename, 'rb') as file:
#     st.download_button(
#         label="Download Prediction Log",
#         data=file,
#         file_name=prediction_data_filename,
#         mime='text/csv'
#     )

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

# Suppress all warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load the trained model
@st.cache_resource
def load_trained_model():
    return load_model('stockpricemodel.keras', compile=False)

model = load_trained_model()

st.title("📊 Stock Price Prediction App")

# User input for stock symbol with better default
stock = st.text_input("Enter Stock Symbol (e.g., AAPL, MSFT, GOOGL)", "AAPL")
stock = stock.upper().strip()

# Define date ranges - Split into chunks
end = dt.datetime.now()
start = dt.datetime(2012, 1, 1)

# Calculate midpoint for chunked download
mid_date = start + (end - start) / 2

# Function to download stock data in chunks with delays
@st.cache_data(ttl=3600, show_spinner=False)
def download_stock_data_chunked(symbol, start_date, end_date):
    """Download stock data in two chunks to avoid rate limiting"""
    
    try:
        # Calculate midpoint
        mid = start_date + (end_date - start_date) / 2
        
        # Progress indicator
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Download first chunk (older data)
        status_text.text(f"📥 Downloading {symbol} data (Part 1/2)...")
        progress_bar.progress(10)
        
        time.sleep(1.5)  # 1.5 second delay before first request
        
        chunk1 = yf.download(
            symbol,
            start=start_date,
            end=mid,
            progress=False,
            auto_adjust=True,
            threads=False,
            keepna=False,
            timeout=10
        )
        
        progress_bar.progress(45)
        status_text.text(f"✅ Part 1 complete. Waiting before next request...")
        
        # Wait 1 second between requests
        time.sleep(1.0)
        
        # Download second chunk (recent data)
        status_text.text(f"📥 Downloading {symbol} data (Part 2/2)...")
        progress_bar.progress(50)
        
        chunk2 = yf.download(
            symbol,
            start=mid,
            end=end_date,
            progress=False,
            auto_adjust=True,
            threads=False,
            keepna=False,
            timeout=10
        )
        
        progress_bar.progress(90)
        
        # Combine chunks
        if not chunk1.empty and not chunk2.empty:
            df = pd.concat([chunk1, chunk2])
            df = df[~df.index.duplicated(keep='first')]  # Remove duplicates
            df = df.sort_index()
            
            progress_bar.progress(100)
            status_text.text(f"✅ Successfully downloaded {len(df)} days of data!")
            time.sleep(0.5)
            progress_bar.empty()
            status_text.empty()
            
            return df, None
        elif not chunk1.empty:
            progress_bar.progress(100)
            status_text.text(f"✅ Downloaded {len(chunk1)} days of data!")
            time.sleep(0.5)
            progress_bar.empty()
            status_text.empty()
            return chunk1, None
        elif not chunk2.empty:
            progress_bar.progress(100)
            status_text.text(f"✅ Downloaded {len(chunk2)} days of data!")
            time.sleep(0.5)
            progress_bar.empty()
            status_text.empty()
            return chunk2, None
        else:
            progress_bar.empty()
            status_text.empty()
            return None, f"No data found for symbol '{symbol}'. Please verify the symbol is correct."
            
    except Exception as e:
        if progress_bar:
            progress_bar.empty()
        if status_text:
            status_text.empty()
        
        error_msg = str(e)
        if "404" in error_msg or "No data" in error_msg:
            return None, f"Symbol '{symbol}' not found. Please check if the symbol is correct."
        elif "Rate" in error_msg or "429" in error_msg:
            return None, "Yahoo Finance rate limit reached. Please wait 2-3 minutes and try again."
        elif "JSON" in error_msg:
            return None, f"Data format error for '{symbol}'. This symbol might be delisted or invalid. Try: AAPL, MSFT, GOOGL, TSLA"
        else:
            return None, f"Error downloading data: {error_msg}"

# Alternative: Try single download as fallback
@st.cache_data(ttl=3600, show_spinner=False)
def download_stock_data_single(symbol, start_date, end_date):
    """Fallback: Single download with longer delay"""
    try:
        status_text = st.empty()
        status_text.text(f"📥 Downloading {symbol} data...")
        
        time.sleep(2.0)  # 2 second delay
        
        df = yf.download(
            symbol,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=True,
            threads=False,
            keepna=False,
            timeout=15
        )
        
        status_text.empty()
        
        if not df.empty:
            return df, None
        else:
            return None, f"No data found for '{symbol}'"
            
    except Exception as e:
        if status_text:
            status_text.empty()
        return None, str(e)

# ✅ Try chunked download first, then fallback to single download
st.info(f"🔄 Fetching data for **{stock}**... This may take 5-8 seconds to avoid rate limits.")

df, error = download_stock_data_chunked(stock, start, end)

if error:
    st.warning(f"⚠️ Chunked download failed: {error}")
    st.info("🔄 Trying alternative download method...")
    df, error = download_stock_data_single(stock, start, end)

if error:
    st.error(f"❌ {error}")
    st.info("💡 **Troubleshooting Tips:**\n"
            "- **Wait 2-3 minutes** before trying again (Yahoo Finance rate limits)\n"
            "- Try popular symbols: **AAPL** (Apple), **MSFT** (Microsoft), **GOOGL** (Google), **TSLA** (Tesla)\n"
            "- Check if the symbol is correct on Yahoo Finance website\n"
            "- Clear Streamlit cache: Settings → Clear Cache → Rerun")
    
    # Suggest alternative symbols
    st.subheader("🔥 Try These Popular Stocks:")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.button("📱 AAPL (Apple)")
    with col2:
        st.button("💻 MSFT (Microsoft)")
    with col3:
        st.button("🔍 GOOGL (Google)")
    with col4:
        st.button("🚗 TSLA (Tesla)")
    
    st.stop()

if df.empty:
    st.error(f"❌ No data available for '{stock}'. The symbol might be invalid or delisted.")
    st.stop()

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
    change = ((df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100
    st.metric("Daily Change", f"{change:.2f}%")

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
            future_predictions[-1] * 0.97,  # -3% limit
            future_predictions[-1] * 1.03   # +3% limit
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
try:
    time.sleep(1)  # Small delay before getting ticker info
    ticker_info = yf.Ticker(stock).info
    currency = ticker_info.get("currency", "USD")
except:
    currency = "USD"

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