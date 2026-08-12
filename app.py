import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time

st.set_page_config(page_title="Live Stock Dashboard", layout="wide")

st.title("📈 Live Stock Dashboard")
st.caption("Latest Price | 3-Period EMA | 21-Period WMA | 9-Period RSI")

# ---------- Sidebar inputs ----------
st.sidebar.header("Settings")
symbol = st.sidebar.text_input(
    "Enter Stock Symbol (Yahoo Finance format)",
    value="RELIANCE.NS",
    help="NSE stocks need '.NS' suffix. Example: TCS.NS, INFY.NS, RELIANCE.NS"
)
interval = st.sidebar.selectbox(
    "Candle Interval",
    options=["1m", "5m", "15m", "1d"],
    index=1
)
refresh_seconds = st.sidebar.number_input(
    "Auto-refresh every (seconds)",
    min_value=10, max_value=300, value=30
)
run = st.sidebar.checkbox("Start Live Updates", value=True)


# ---------- Indicator functions ----------
def calculate_indicators(df):
    df = df.copy()

    # 3-period EMA
    df["EMA_3"] = df["Close"].ewm(span=3, adjust=False).mean()

    # 21-period WMA (Weighted Moving Average)
    weights = np.arange(1, 22)  # 1 to 21
    df["WMA_21"] = df["Close"].rolling(21).apply(
        lambda prices: np.dot(prices, weights) / weights.sum(), raw=True
    )

    # 9-period RSI
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=9).mean()
    avg_loss = loss.rolling(window=9).mean()
    rs = avg_gain / avg_loss
    df["RSI_9"] = 100 - (100 / (1 + rs))

    return df


# ---------- Fetch + display ----------
def fetch_and_display():
    try:
        data = yf.download(
            tickers=symbol,
            period="5d",
            interval=interval,
            progress=False
        )

        if data.empty:
            st.error("No data found. Check the symbol (e.g. use RELIANCE.NS for NSE stocks).")
            return

        data = calculate_indicators(data)
        latest = data.iloc[-1]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Latest Price", f"{latest['Close']:.2f}")
        col2.metric("3-EMA", f"{latest['EMA_3']:.2f}")
        col3.metric("21-WMA", f"{latest['WMA_21']:.2f}")
        col4.metric("9-RSI", f"{latest['RSI_9']:.2f}")

        st.subheader("Price Chart")
        st.line_chart(data[["Close", "EMA_3", "WMA_21"]].dropna())

        st.subheader("RSI (9)")
        st.line_chart(data[["RSI_9"]].dropna())

        st.subheader("Raw Data (latest rows)")
        st.dataframe(
            data[["Close", "EMA_3", "WMA_21", "RSI_9"]].tail(15).sort_index(ascending=False)
        )

        st.caption(f"Last updated: {pd.Timestamp.now()}")

    except Exception as e:
        st.error(f"Error fetching data: {e}")


# ---------- Live loop ----------
placeholder = st.empty()

if run:
    while True:
        with placeholder.container():
            fetch_and_display()
        time.sleep(refresh_seconds)
        st.rerun()
else:
    fetch_and_display()
