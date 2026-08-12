import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time
import requests
import io
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
    index=list(INTERVAL_CONFIG.keys()).index("1d")
)
refresh_seconds = st.sidebar.number_input(
    "Auto-refresh every (seconds)",
    min_value=10, max_value=300, value=30
)
num_bars = st.sidebar.slider(
    "Candles to show on chart",
    min_value=30, max_value=300, value=45,
    help="Fewer candles = cleaner, less zig-zag chart (like TradingView's default zoomed view)"
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

    # "Hilega Milega" style: EMA(3) and WMA(21) applied to the RSI itself
    # (not to price) so both lines sit on the same 0-100 scale as RSI.
    df["RSI_EMA_3"] = df["RSI_9"].ewm(span=3, adjust=False).mean()
    df["RSI_WMA_21"] = df["RSI_9"].rolling(21).apply(
        lambda vals: np.dot(vals, weights) / weights.sum(), raw=True
    )

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
def build_chart(data, is_intraday=False):
    idx = data.index
    close = data["Close"]
    rsi = data["RSI_9"]
    rsi_wma = data["RSI_WMA_21"]
    rsi_ema = data["RSI_EMA_3"]

    # Masked series so the fill collapses to nothing on the "wrong" side of 50
    rsi_upper = rsi.clip(lower=50)
    rsi_lower = rsi.clip(upper=50)
    midline = pd.Series(50, index=idx)

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.55, 0.45],
        vertical_spacing=0.04,
    )

    # ---- Upper section: Price as a candlestick chart ----
    fig.add_trace(
        go.Candlestick(
            x=idx, open=data["Open"], high=data["High"], low=data["Low"], close=data["Close"],
            name="Price",
            increasing_line_color="#26A69A", decreasing_line_color="#EF5350"
        ),
        row=1, col=1
    )

    # ---- Lower section: RSI zones (orange above 50, light green below 50) ----
    fig.add_trace(go.Scatter(x=idx, y=midline, line=dict(width=0), showlegend=False,
                              hoverinfo="skip"), row=2, col=1)
    fig.add_trace(go.Scatter(x=idx, y=rsi_upper, line=dict(width=0), fill="tonexty",
                              fillcolor="rgba(144,238,144,0.35)", name="Overbought (>50)",
                              showlegend=True, hoverinfo="skip"), row=2, col=1)
    fig.add_trace(go.Scatter(x=idx, y=midline, line=dict(width=0), showlegend=False,
                              hoverinfo="skip"), row=2, col=1)
    fig.add_trace(go.Scatter(x=idx, y=rsi_lower, line=dict(width=0), fill="tonexty",
                              fillcolor="rgba(255,200,120,0.35)", name="Oversold (<50)",
                              showlegend=True, hoverinfo="skip"), row=2, col=1)

    # Midline at 50
    fig.add_trace(
        go.Scatter(x=idx, y=midline, mode="lines", line=dict(color="gray", width=1, dash="dot"),
                   name="Midline (50)"), row=2, col=1
    )

    # RSI line itself
    fig.add_trace(
        go.Scatter(x=idx, y=rsi, mode="lines", line=dict(color="white", width=1.1),
                   name="RSI (9)"), row=2, col=1
    )

    # WMA(21) and EMA(3) of the RSI itself -- same 0-100 scale, like the
    # "Hilega Milega" TradingView indicator (close, 9, 3, 21)
    fig.add_trace(
        go.Scatter(x=idx, y=rsi_wma, mode="lines", line=dict(color="red", width=1),
                   name="21 WMA (of RSI)"), row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=idx, y=rsi_ema, mode="lines", line=dict(color="violet", width=1),
                   name="3 EMA (of RSI)"), row=2, col=1
    )

    fig.update_yaxes(title_text="Price", row=1, col=1, fixedrange=True)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1, fixedrange=True)
    fig.update_xaxes(rangeslider_visible=False, row=1, col=1)

    # Remove blank gaps for days/hours when the market is closed
    # (weekends, and outside 9:15-15:30 for intraday candles) so
    # scrolling left never lands on empty space.
    rangebreaks = [dict(bounds=["sat", "mon"])]
    if is_intraday:
        rangebreaks.append(dict(bounds=[15.5, 9.25], pattern="hour"))
    fig.update_xaxes(rangebreaks=rangebreaks, row=1, col=1)
    fig.update_xaxes(rangebreaks=rangebreaks, row=2, col=1)

    fig.update_layout(
        height=700,
        template="plotly_dark",
        hovermode="x unified",
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        dragmode="pan",  # single-finger drag = scroll left/right, not a distorting zoom-box
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
        col2.metric("3-EMA (Price)", f"{latest_ema:.2f}")
        col3.metric("21-WMA (Price)", f"{latest_wma:.2f}" if not pd.isna(latest_wma) else "N/A")
        col4.metric("9-RSI", f"{latest_rsi:.2f}" if not pd.isna(latest_rsi) else "N/A")

        st.plotly_chart(
            build_chart(
                data.tail(num_bars),
                is_intraday=(interval in ["1m", "5m", "10m", "15m", "30m", "1h", "2h", "3h", "4h"])
            ),
            use_container_width=True,
            config={
                "scrollZoom": True,      # pinch (two-finger) to zoom in/out
                "displayModeBar": False  # cleaner mobile view
            }
        )

        with st.expander("Raw Data (latest rows)"):
            st.dataframe(
                data[["RSI_9", "RSI_EMA_3", "RSI_WMA_21"]]
                .tail(15).sort_index(ascending=False)
                .style.format("{:.2f}")
            )

        st.caption(f"Last updated: {pd.Timestamp.now()}")

    except Exception as e:
        st.error(f"Error fetching data: {e}")


# ============================================================
# ---------- Market Scanner: Indexes + Nifty 500 ----------
# ============================================================

# Major market-cap based & sectoral indexes (verified Yahoo Finance symbols)
MAJOR_INDEXES = {
    "NIFTY 50": "^NSEI",
    "NIFTY NEXT 50": "^NSMIDCP",
    "NIFTY 100": "^CNX100",
    "NIFTY 200": "^CNX200",
    "NIFTY 500": "^CRSLDX",
    "NIFTY MIDCAP 50": "^NSEMDCP50",
    "SENSEX": "^BSESN",
    "NIFTY BANK": "^NSEBANK",
    "NIFTY IT": "^CNXIT",
    "NIFTY AUTO": "^CNXAUTO",
    "NIFTY METAL": "^CNXMETAL",
    "NIFTY FMCG": "^CNXFMCG",
    "NIFTY PHARMA": "^CNXPHARMA",
    "NIFTY ENERGY": "^CNXENERGY",
    "NIFTY REALTY": "^CNXREALTY",
    "NIFTY MEDIA": "^CNXMEDIA",
    "NIFTY PSU BANK": "^CNXPSUBANK",
    "NIFTY FIN SERVICE": "^CNXFIN",
    "NIFTY INFRA": "^CNXINFRA",
    "NIFTY COMMODITIES": "^CNXCMDT",
}

# Small backup list used only if the live NSE fetch below fails/is blocked
FALLBACK_SYMBOLS = [
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "HINDUNILVR", "ITC",
    "SBIN", "BHARTIARTL", "KOTAKBANK", "LT", "AXISBANK", "BAJFINANCE",
    "ASIANPAINT", "MARUTI", "HCLTECH", "SUNPHARMA", "TITAN", "ULTRACEMCO",
    "NESTLEIND"
]


@st.cache_data(ttl=86400, show_spinner=False)
def get_nifty500_symbols():
    """Fetch the live Nifty 500 constituent list directly from NSE.
    Falls back to a small known list if NSE blocks the request."""
    url = "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
    }
    try:
        session = requests.Session()
        session.headers.update(headers)
        session.get("https://www.nseindia.com", timeout=10)  # sets cookies
        resp = session.get(url, timeout=15)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        symbols = df["Symbol"].astype(str).str.strip().tolist()
        return symbols
    except Exception:
        return FALLBACK_SYMBOLS


# Shared signal -> color mapping (hex, used for both on-screen table and Excel)
SIGNAL_COLORS_EXCEL = {
    "BUY": "#008000",                        # green
    "SELL": "#FF0000",                       # red
    "CORRECTION/SHORT COVERING": "#0000FF",  # blue
    "WAIT": "#000000",                       # black (in the downloaded Excel)
}

SIGNAL_COLORS_DISPLAY = {
    "BUY": "#00FF00",                        # bright green (visible on dark bg)
    "SELL": "#FF4B4B",                        # red
    "CORRECTION/SHORT COVERING": "#4DA6FF",   # blue
    "WAIT": "#FFFFFF",                        # white (so it's visible on the dashboard)
}


def get_scanner_signal(rsi, price, wma):
    """
    PRICE here = EMA(3) of RSI (same 0-100 scale as RSI and WMA-of-RSI,
    consistent with the main chart's RSI-based signal lines).
    BUY: RSI > 50 and RSI > PRICE > WMA
    SELL: RSI < 50 and WMA > PRICE > RSI
    CORRECTION/SHORT COVERING: RSI < 50 and RSI > PRICE > WMA
    """
    if pd.isna(rsi) or pd.isna(price) or pd.isna(wma):
        return "WAIT"
    if rsi > 50 and rsi > price > wma:
        return "BUY"
    elif rsi < 50 and wma > price > rsi:
        return "SELL"
    elif rsi < 50 and rsi > price > wma:
        return "CORRECTION/SHORT COVERING"
    else:
        return "WAIT"


def style_signal_rows(df):
    """Colour every row's text based on its Signal column, and show all
    numeric values rounded to exactly 2 decimal places (for on-screen display)."""
    def _apply(row):
        color = SIGNAL_COLORS_DISPLAY.get(row.get("Signal", "WAIT"), "#FFFFFF")
        return [f"color: {color}"] * len(row)

    numeric_cols = df.select_dtypes(include="number").columns
    fmt = {col: "{:.2f}" for col in numeric_cols}
    return df.style.apply(_apply, axis=1).format(fmt)


def scan_symbols(symbols, yf_suffix=".NS", interval="1d"):
    """Batch-fetch data (at the chosen timeframe) and compute Latest Price,
    RSI(9), EMA(3) of RSI, WMA(21) of RSI and Signal for each symbol."""
    cfg = INTERVAL_CONFIG[interval]
    tickers = [
        f"{s}{yf_suffix}" if yf_suffix and not str(s).startswith("^") else s
        for s in symbols
    ]
    weights = np.arange(1, 22)

    try:
        raw = yf.download(
            tickers=" ".join(tickers),
            period=cfg["period"],
            interval=cfg["yf_interval"],
            group_by="ticker",
            threads=True,
            progress=False,
        )
    except Exception as e:
        st.error(f"Scanner fetch failed: {e}")
        return pd.DataFrame()

    rows = []
    for sym, tkr in zip(symbols, tickers):
        try:
            if len(tickers) == 1:
                df = raw
            else:
                if tkr not in raw.columns.get_level_values(0):
                    continue
                df = raw[tkr]

            df = df.dropna(how="all")

            # Combine smaller candles into the requested bigger duration
            # (same resample logic as the main single-stock chart)
            if cfg["resample"]:
                df = df.resample(cfg["resample"]).agg({
                    "Open": "first", "High": "max", "Low": "min",
                    "Close": "last", "Volume": "sum"
                }).dropna(how="all")

            close = df["Close"].dropna()
            if len(close) < 10:
                continue

            delta = close.diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.ewm(alpha=1 / 9, min_periods=9, adjust=False).mean()
            avg_loss = loss.ewm(alpha=1 / 9, min_periods=9, adjust=False).mean()
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

            rsi_ema3 = rsi.ewm(span=3, adjust=False).mean()
            rsi_wma21 = rsi.rolling(21).apply(
                lambda v: np.dot(v, weights) / weights.sum(), raw=True
            )

            latest_rsi = rsi.iloc[-1]
            latest_ema = rsi_ema3.iloc[-1]
            latest_wma = rsi_wma21.iloc[-1]
            signal = get_scanner_signal(latest_rsi, latest_ema, latest_wma)

            rows.append({
                "Symbol": sym,
                "Latest Price": round(float(close.iloc[-1]), 2),
                "RSI (9)": round(float(latest_rsi), 2) if not pd.isna(latest_rsi) else None,
                "EMA(3) of RSI": round(float(latest_ema), 2) if not pd.isna(latest_ema) else None,
                "WMA(21) of RSI": round(float(latest_wma), 2) if not pd.isna(latest_wma) else None,
                "Signal": signal,
            })
        except Exception:
            continue

    return pd.DataFrame(rows)


st.divider()
st.header("🔍 Market Scanner — Indexes & Nifty 500")
st.caption(
    "Separate from the live auto-refresh chart above. Scanning 500+ stocks "
    "can take a few minutes and only refreshes when you click the button "
    "(continuous live scanning of 500 stocks would get rate-limited by "
    "Yahoo Finance)."
)

scanner_interval = st.selectbox(
    "Scanner Timeframe",
    options=list(INTERVAL_CONFIG.keys()),
    index=list(INTERVAL_CONFIG.keys()).index("1d"),
    help="Data (and the Excel download) will be based on this timeframe."
)

if "scanner_index_df" not in st.session_state:
    st.session_state.scanner_index_df = None
if "scanner_stocks_df" not in st.session_state:
    st.session_state.scanner_stocks_df = None
if "scanner_interval_used" not in st.session_state:
    st.session_state.scanner_interval_used = None

if st.button("▶️ Run / Refresh Scanner"):
    with st.spinner("Scanning major indexes..."):
        index_names = list(MAJOR_INDEXES.keys())
        index_symbols = list(MAJOR_INDEXES.values())
        idx_df = scan_symbols(index_symbols, yf_suffix="", interval=scanner_interval)
        if not idx_df.empty:
            name_map = dict(zip(index_symbols, index_names))
            idx_df.insert(0, "Index", idx_df["Symbol"].map(name_map))
            idx_df = idx_df.drop(columns=["Symbol"])
        st.session_state.scanner_index_df = idx_df

    with st.spinner("Scanning Nifty 500 stocks (this can take a few minutes)..."):
        nifty500 = get_nifty500_symbols()
        stocks_df = scan_symbols(nifty500, yf_suffix=".NS", interval=scanner_interval)
        st.session_state.scanner_stocks_df = stocks_df

    st.session_state.scanner_interval_used = scanner_interval

if st.session_state.scanner_index_df is not None and not st.session_state.scanner_index_df.empty:
    st.caption(f"Data timeframe: **{st.session_state.scanner_interval_used}**")

    tab_all, tab_idx, tab_stocks = st.tabs(["All", "Indexes", "Nifty 500 Stocks"])

    with tab_idx:
        st.dataframe(style_signal_rows(st.session_state.scanner_index_df), use_container_width=True)

    with tab_stocks:
        st.dataframe(style_signal_rows(st.session_state.scanner_stocks_df), use_container_width=True)

    with tab_all:
        st.write("**Indexes**")
        st.dataframe(style_signal_rows(st.session_state.scanner_index_df), use_container_width=True)
        st.write("**Nifty 500 Stocks**")
        st.dataframe(style_signal_rows(st.session_state.scanner_stocks_df), use_container_width=True)

    # ---- Download as a single Excel file with 2 separate sheets ----
    from openpyxl.styles import Font

    def color_code_sheet(worksheet, df):
        if "Signal" not in df.columns:
            return
        numeric_col_idxs = [
            i + 1 for i, col in enumerate(df.columns)
            if pd.api.types.is_numeric_dtype(df[col])
        ]
        for row_idx, signal in enumerate(df["Signal"], start=2):  # row 1 = header
            color = SIGNAL_COLORS_EXCEL.get(signal, "#000000").lstrip("#")
            for col_idx in range(1, len(df.columns) + 1):
                cell = worksheet.cell(row=row_idx, column=col_idx)
                cell.font = Font(color=color)
                if col_idx in numeric_col_idxs:
                    cell.number_format = "0.00"

    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        st.session_state.scanner_index_df.to_excel(writer, sheet_name="Indexes", index=False)
        st.session_state.scanner_stocks_df.to_excel(writer, sheet_name="Nifty500_Stocks", index=False)
        color_code_sheet(writer.sheets["Indexes"], st.session_state.scanner_index_df)
        color_code_sheet(writer.sheets["Nifty500_Stocks"], st.session_state.scanner_stocks_df)
    excel_buffer.seek(0)

    st.download_button(
        label="⬇️ Download Excel (Indexes + Nifty 500 — 2 sheets)",
        data=excel_buffer,
        file_name=(
            f"market_scan_{st.session_state.scanner_interval_used}_"
            f"{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.xlsx"
        ),
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
else:
    st.info("Click 'Run / Refresh Scanner' to load index and Nifty 500 data.")


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
