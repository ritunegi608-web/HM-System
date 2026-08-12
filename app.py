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

# Yahoo Finance does not natively provide 10m / 2h / 3h / 4h candles.
# We fetch the closest available candle size and combine ("resample")
# it into the requested duration ourselves.
INTERVAL_CONFIG = {
    "1m":  {"yf_interval": "1m",  "period": "5d",   "resample": None},
    "5m":  {"yf_interval": "5m",  "period": "60d",  "resample": None},
    "10m": {"yf_interval": "5m",  "period": "60d",  "resample": "10min"},
    "15m": {"yf_interval": "15m", "period": "60d",  "resample": None},
    "30m": {"yf_interval": "30m", "period": "60d",  "resample": None},
    "1h":  {"yf_interval": "60m", "period": "180d", "resample": None},
    "2h":  {"yf_interval": "60m", "period": "180d", "resample": "2h"},
    "3h":  {"yf_interval": "60m", "period": "180d", "resample": "3h"},
    "4h":  {"yf_interval": "60m", "period": "180d", "resample": "4h"},
    "1d":  {"yf_interval": "1d",  "period": "6mo",  "resample": None},
}

interval = st.sidebar.selectbox(
    "Candle Interval",
    options=list(INTERVAL_CONFIG.keys()),
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

    # 3-period EMA (this also acts as "PRICE" in the signal logic below)
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


# ---------- Signal logic ----------
def get_signal(rsi, price, wma):
    """
    PRICE = 3-period EMA of price (as specified).
    BUY: RSI > 50 AND RSI > PRICE > WMA
    SELL: RSI < 50 AND WMA > PRICE > RSI
    CORRECTION MIGHT BE OVER / SHORT COVERING: RSI < 50 AND RSI > PRICE > WMA
    """
    if pd.isna(rsi) or pd.isna(price) or pd.isna(wma):
        return "NOT ENOUGH DATA", "info"

    if rsi > 50 and rsi > price > wma:
        return "BUY SIGNAL", "success"
    elif rsi < 50 and wma > price > rsi:
        return "SELL SIGNAL", "error"
    elif rsi < 50 and rsi > price > wma:
        return "CORRECTION MIGHT BE OVER / SHORT COVERING", "warning"
    else:
        return "NO CLEAR SIGNAL", "info"


# ---------- Data fetch ----------
def fetch_data(symbol, interval):
    cfg = INTERVAL_CONFIG[interval]

    data = yf.download(
        tickers=symbol,
        period=cfg["period"],
        interval=cfg["yf_interval"],
        progress=False
    )

    if data.empty:
        return data

    # Flatten multi-level columns (newer yfinance versions can return
    # these even for a single ticker)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # Combine smaller candles into the requested bigger duration
    if cfg["resample"]:
        data = data.resample(cfg["resample"]).agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum"
        }).dropna(how="all")

    return data


# ---------- Fetch + display ----------
def fetch_and_display():
    try:
        data = fetch_data(symbol, interval)

        if data.empty:
            st.error("No data found. Check the symbol (e.g. use RELIANCE.NS for NSE stocks).")
            return

        data = calculate_indicators(data)
        latest = data.iloc[-1]

        # Force plain Python floats so formatting never breaks
        latest_close = float(latest['Close'])
        latest_ema = float(latest['EMA_3'])
        latest_wma = float(latest['WMA_21']) if not pd.isna(latest['WMA_21']) else float('nan')
        latest_rsi = float(latest['RSI_9']) if not pd.isna(latest['RSI_9']) else float('nan')

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Latest Price", f"{latest_close:.2f}")
        col2.metric("3-EMA (PRICE)", f"{latest_ema:.2f}")
        col3.metric("21-WMA", f"{latest_wma:.2f}" if not pd.isna(latest_wma) else "N/A")
        col4.metric("9-RSI", f"{latest_rsi:.2f}" if not pd.isna(latest_rsi) else "N/A")

        # ---------- Signal ----------
        st.subheader("Signal")
        signal_text, signal_type = get_signal(latest_rsi, latest_ema, latest_wma)
        if signal_type == "success":
            st.success(f"📈 {signal_text}")
        elif signal_type == "error":
            st.error(f"📉 {signal_text}")
        elif signal_type == "warning":
            st.warning(f"⚠️ {signal_text}")
        else:
            st.info(f"ℹ️ {signal_text}")

        st.caption(
            "BUY: RSI > 50 and RSI > PRICE > WMA  |  "
            "SELL: RSI < 50 and WMA > PRICE > RSI  |  "
            "CORRECTION/SHORT COVERING: RSI < 50 and RSI > PRICE > WMA  "
            "(PRICE = 3-period EMA). This is a mechanical rule-based signal, "
            "not financial advice."
        )

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
