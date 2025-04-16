import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import date

# App title
st.set_page_config(page_title="Live Stock Market Dashboard", layout="wide")
st.title("📈 Live Stock Market Dashboard")

# Sidebar - Stock ticker input
st.sidebar.header("Settings")

# Dropdown of popular tickers
popular_tickers = ["AAPL", "TSLA", "MSFT", "GOOG", "AMZN", "META", "NFLX", "NVDA"]
selected_ticker = st.sidebar.selectbox("Select a Popular Ticker", popular_tickers)

# Optional custom ticker
custom_ticker = st.sidebar.text_input("Or Enter Custom Ticker")

# Use custom ticker if provided, else use selected one
ticker = custom_ticker.upper() if custom_ticker else selected_ticker


# Sidebar - Date range selection
start_date = st.sidebar.date_input("Start Date", date(2020, 1, 1))
end_date = st.sidebar.date_input("End Date", date.today())

# Sidebar - Chart type
chart_type = st.sidebar.radio("Select Chart Type", ["Line Chart", "Candlestick", "Volume"])

# Download data from Yahoo Finance
@st.cache_data
def load_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    return data

data = load_data(ticker, start_date, end_date)

# Main Dashboard
st.subheader(f"Stock Data for {ticker.upper()} from {start_date} to {end_date}")

# Display data table
st.dataframe(data.tail(10))

# Summary statistics
st.markdown("### 📊 Summary Statistics")
st.write(data.describe())

# Plotting
st.markdown("### 📉 Visualization")
if chart_type == "Line Chart":
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name="Close Price"))
    fig.update_layout(title=f"{ticker.upper()} Close Price Over Time", xaxis_title="Date", yaxis_title="Price (USD)")
    st.plotly_chart(fig, use_container_width=True)

elif chart_type == "Candlestick":
    fig = go.Figure(data=[go.Candlestick(x=data.index,
                                         open=data['Open'],
                                         high=data['High'],
                                         low=data['Low'],
                                         close=data['Close'])])
    fig.update_layout(title=f"{ticker.upper()} Candlestick Chart", xaxis_title="Date", yaxis_title="Price (USD)")
    st.plotly_chart(fig, use_container_width=True)

elif chart_type == "Volume":
    fig = go.Figure()
    fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name="Volume"))
    fig.update_layout(title=f"{ticker.upper()} Volume Traded", xaxis_title="Date", yaxis_title="Volume")
    st.plotly_chart(fig, use_container_width=True)

# Download data
csv = data.to_csv().encode()
st.download_button(
    label="📥 Download Data as CSV",
    data=csv,
    file_name=f"{ticker}_data.csv",
    mime='text/csv'
)
