import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time
from plotly.subplots import make_subplots
import plotly.graph_objects as go

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

    # 3-period EMA of Close price
    df["EMA_3"] = df["Close"].ewm(span=3, adjust=False).mean()

    # 21-period WMA (Weighted Moving Average) of Close price
    weights = np.arange(1, 22)  # 1 to 21
    df["WMA_21"] = df["Close"].rolling(21).apply(
        lambda prices: np.dot(prices, weights) / weights.sum(), raw=True
    )

    # 9-period RSI using Wilder's smoothing (same method TradingView uses)
    # -> gives a much smoother line than a simple rolling average.
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 9, min_periods=9, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 9, min_periods=9, adjust=False).mean()
    rs = avg_gain / avg_loss
    df["RSI_9"] = 100 - (100 / (1 + rs))

    return df


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

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    if cfg["resample"]:
        data = data.resample(cfg["resample"]).agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum"
        }).dropna(how="all")

    return data


# ---------- Chart (TradingView "Hilega Milega" style) ----------
def build_chart(data):
    idx = data.index
    close = data["Close"]
    rsi = data["RSI_9"]
    wma = data["WMA_21"]
    ema = data["EMA_3"]

    # Masked series so the fill collapses to nothing on the "wrong" side of 50
    rsi_upper = rsi.clip(lower=50)
    rsi_lower = rsi.clip(upper=50)
    midline = pd.Series(50, index=idx)

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.55, 0.45],
        vertical_spacing=0.04,
        specs=[[{"secondary_y": False}], [{"secondary_y": True}]],
    )

    # ---- Upper section: only Last Traded Price ----
    fig.add_trace(
        go.Scatter(x=idx, y=close, name="Price", line=dict(color="#00BFFF", width=1.6)),
        row=1, col=1
    )

    # ---- Lower section: RSI zones (orange above 50, light green below 50) ----
    fig.add_trace(go.Scatter(x=idx, y=midline, line=dict(width=0), showlegend=False,
                              hoverinfo="skip"), row=2, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=idx, y=rsi_upper, line=dict(width=0), fill="tonexty",
                              fillcolor="rgba(255,165,0,0.35)", name="Overbought (>50)",
                              showlegend=True, hoverinfo="skip"), row=2, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=idx, y=midline, line=dict(width=0), showlegend=False,
                              hoverinfo="skip"), row=2, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=idx, y=rsi_lower, line=dict(width=0), fill="tonexty",
                              fillcolor="rgba(144,238,144,0.35)", name="Oversold (<50)",
                              showlegend=True, hoverinfo="skip"), row=2, col=1, secondary_y=False)

    # Midline at 50
    fig.add_trace(
        go.Scatter(x=idx, y=midline, mode="lines", line=dict(color="gray", width=1, dash="dot"),
                   name="Midline (50)"), row=2, col=1, secondary_y=False
    )

    # RSI line itself
    fig.add_trace(
        go.Scatter(x=idx, y=rsi, mode="lines", line=dict(color="white", width=1.8),
                   name="RSI (9)"), row=2, col=1, secondary_y=False
    )

    # 21 WMA (red) and 3 EMA (violet) of price, plotted on their own scale
    # in the same RSI section (like the "Hilega Milega" indicator)
    fig.add_trace(
        go.Scatter(x=idx, y=wma, mode="lines", line=dict(color="red", width=1.5),
                   name="21 WMA (Price)"), row=2, col=1, secondary_y=True
    )
    fig.add_trace(
        go.Scatter(x=idx, y=ema, mode="lines", line=dict(color="violet", width=1.5),
                   name="3 EMA (Price)"), row=2, col=1, secondary_y=True
    )

    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Price (EMA/WMA)", row=2, col=1, secondary_y=True, showgrid=False)

    fig.update_layout(
        height=700,
        template="plotly_dark",
        hovermode="x unified",
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig


# ---------- Fetch + display ----------
def fetch_and_display():
    try:
        data = fetch_data(symbol, interval)

        if data.empty:
            st.error("No data found. Check the symbol (e.g. use RELIANCE.NS for NSE stocks).")
            return

        data = calculate_indicators(data)
        latest = data.iloc[-1]

        latest_close = float(latest['Close'])
        latest_ema = float(latest['EMA_3'])
        latest_wma = float(latest['WMA_21']) if not pd.isna(latest['WMA_21']) else float('nan')
        latest_rsi = float(latest['RSI_9']) if not pd.isna(latest['RSI_9']) else float('nan')

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Latest Price", f"{latest_close:.2f}")
        col2.metric("3-EMA", f"{latest_ema:.2f}")
        col3.metric("21-WMA", f"{latest_wma:.2f}" if not pd.isna(latest_wma) else "N/A")
        col4.metric("9-RSI", f"{latest_rsi:.2f}" if not pd.isna(latest_rsi) else "N/A")

        st.plotly_chart(build_chart(data), use_container_width=True)

        with st.expander("Raw Data (latest rows)"):
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
