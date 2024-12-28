import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import subprocess
from datetime import datetime

# URLs to Colab notebooks
regression_url = "https://colab.research.google.com/drive/1P4V8wPwiiFKM9K4vjuZfujDaK0jU-aHq?usp=sharing"
decisionTree_url = "https://colab.research.google.com/drive/1ge2j5P8Jz8Hj6EcYBUBWf4i64f9tJgAa?usp=sharing"
LSTM_url = "https://colab.research.google.com/drive/1wMa_Bfa4qAIUS71fCeGUCrE2MXsovHPL?usp=sharing"

st.title("Real-time Stock Price Tracker & Prediction")

# List of popular tickers
popular_tickers = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", 
    "META", "NVDA", "JPM", "V", "JNJ", 
    "PG", "XOM", "BRK-B", "BAC", "WMT"
]

# Dropdown to select a ticker
ticker_symbol = st.selectbox("Select a Ticker", popular_tickers)

# Try loading the selected ticker's data
try:
    # Construct the filename based on the selected ticker
    filename = f"{ticker_symbol}.csv"
    df = pd.read_csv(filename)

    # Strip leading and trailing spaces from column names
    df.columns = df.columns.str.strip()

    # Show the DataFrame in Streamlit
    st.dataframe(df)

    # Ensure 'Date' column exists in the data
    if 'Date' in df.columns and 'Close/Last' in df.columns and 'Volume' in df.columns:
        # Convert the 'Date' column to datetime format
        df['Date'] = pd.to_datetime(df['Date'])

        # Remove the dollar sign from 'Close/Last' and convert to float
        df['Close/Last'] = df['Close/Last'].replace({'\$': ''}, regex=True).astype(float)

        # Ensure 'Volume' is a string before removing commas and converting to int
        df['Volume'] = df['Volume'].astype(str).str.replace(',', '').astype(int)

        # Set 'Date' as the index
        df.set_index('Date', inplace=True)

        # Extract 'Close/Last' and 'Volume' columns
        close_prices = df['Close/Last']
        volume = df['Volume']

        # Calculate RSI (Relative Strength Index)
        delta = close_prices.diff(1)
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=14, min_periods=1).mean()
        avg_loss = loss.rolling(window=14, min_periods=1).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        # Calculate MACD (Moving Average Convergence Divergence)
        short_ema = close_prices.ewm(span=12, adjust=False).mean()  # 12-day EMA
        long_ema = close_prices.ewm(span=26, adjust=False).mean()   # 26-day EMA
        macd = short_ema - long_ema
        signal = macd.ewm(span=9, adjust=False).mean()              # Signal line

        # Create a plot with closing prices, volume, RSI, and MACD
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))

        # Plot closing prices on the first subplot
        ax1.set_title(f'{ticker_symbol} Closing Prices and Volume')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Closing Price ($)', color='blue')
        ax1.plot(close_prices, label='Closing Prices', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.legend(loc='upper left')

        # Plot volume on the first subplot
        ax1_twin = ax1.twinx()
        ax1_twin.set_ylabel('Volume', color='green')
        ax1_twin.bar(close_prices.index, volume, alpha=0.3, color='green', label='Volume')
        ax1_twin.tick_params(axis='y', labelcolor='green')
        ax1_twin.legend(loc='upper right')

        # Plot RSI on the second subplot
        ax2.set_title('Relative Strength Index (RSI)')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('RSI')
        ax2.plot(rsi, label='RSI', color='purple')
        ax2.axhline(70, linestyle='--', color='red', label='Overbought')
        ax2.axhline(30, linestyle='--', color='green', label='Oversold')
        ax2.legend(loc='upper left')

        # Plot MACD on the third subplot
        ax3.set_title('MACD (Moving Average Convergence Divergence)')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('MACD')
        ax3.plot(macd, label='MACD', color='blue')
        ax3.plot(signal, label='Signal Line', color='orange')
        ax3.bar(close_prices.index, macd - signal, label='Histogram', color='gray', alpha=0.5)
        ax3.legend(loc='upper left')

        plt.tight_layout()

        # Display the plots in Streamlit
        st.pyplot(fig)
    else:
        st.error("'Date', 'Close/Last', or 'Volume' column not found in the dataset.")

except FileNotFoundError:
    st.error(f"CSV file for {ticker_symbol} not found.")
except Exception as e:
    st.error(f"Error loading data: {e}")

# Prediction Section
st.subheader("Predict Future Stock Price")
input_date = st.date_input("Select Date (yyyy-mm-dd)", datetime.today())
input_price = st.number_input("Enter the Current Stock Price ($)", min_value=0.0, step=0.01)

def predict_stock_price(date, price, model_type):
    """Placeholder function for model predictions."""
    if model_type == "Regression":
        return round(price * 1.02, 2)  # Example: 2% increase
    elif model_type == "Decision Tree":
        return round(price * 1.05, 2)  # Example: 5% increase
    elif model_type == "LSTM":
        return round(price * 1.03, 2)  # Example: 3% increase

if input_price > 0:
    regression_prediction = predict_stock_price(input_date, input_price, "Regression")
    decision_tree_prediction = predict_stock_price(input_date, input_price, "Decision Tree")
    lstm_prediction = predict_stock_price(input_date, input_price, "LSTM")

    st.write(f"**Regression Model Prediction**: ${regression_prediction}")
    st.write(f"**Decision Tree Model Prediction**: ${decision_tree_prediction}")
    st.write(f"**LSTM Model Prediction**: ${lstm_prediction}")

    # Plot predictions
    st.subheader("Prediction Comparison")
    fig, ax = plt.subplots()
    models = ['Regression', 'Decision Tree', 'LSTM']
    predictions = [regression_prediction, decision_tree_prediction, lstm_prediction]
    ax.bar(models, predictions, color=['blue', 'green', 'purple'])
    ax.set_title('Model Prediction Comparison')
    ax.set_ylabel('Predicted Price ($)')
    st.pyplot(fig)
else:
    st.warning("Enter a valid stock price to predict.")

# Colab Links
st.header("Model Details")
st.write(f"[Regression Model Notebook]({regression_url})")
st.write(f"[Decision Tree Model Notebook]({decisionTree_url})")
st.write(f"[LSTM Model Notebook]({LSTM_url})")