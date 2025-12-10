import altair as alt
from streamlit_searchbox import st_searchbox
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf
import pandas as pd
import requests
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --------------------------
# SET CONFIG FIRST !!!
# --------------------------
st.set_page_config(
    page_title="Stock Dashboard by SongChiTienQuan",
    layout="wide"
)

# --------------------------
# Inject HTML HEADER only
# --------------------------
def inject_html_header():
    with open("background.html", "r", encoding="utf-8") as f:
        html = f.read()

    # Chỉ chèn 1 lần ở đầu trang, KHÔNG replace body, KHÔNG xoá DOM Streamlit
    components.html(
        html,
        height=300,     # header cao bao nhiêu tuỳ bạn
        scrolling=False
    )

inject_html_header()

# --------------------------
# FINBERT loading 
# --------------------------
@st.cache_resource
def load_finbert():
    model_name = "yiyanghkust/finbert-tone"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_finbert()

def get_sentiment(text: str):
    if not text.strip():
        return "neutral", 1.0
    text = text[:2000]
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0].cpu().numpy()
        labels = ["negative", "neutral", "positive"]
        return labels[int(probs.argmax())], float(probs.max())
    except:
        return "neutral", 0.0

# -------------------------
# FETCH FUNCTIONS
# -------------------------
@st.cache_data
def fetch_stock_info(symbol: str):
    try:
        ticker = yf.Ticker(symbol)
        fast = ticker.fast_info or {}
        safe = ticker.get_info() or {}
        return {
            "longName": safe.get("longName") or safe.get("shortName") or symbol,
            "currency": fast.get("currency") or "USD",
            "marketCap": fast.get("marketCap") or 0,
            "regularMarketPrice": fast.get("lastPrice")
                or safe.get("regularMarketPrice")
                or safe.get("currentPrice"),
        }
    except:
        return {}

@st.cache_data
def fetch_quarterly_financials(symbol: str):
    try:
        df = yf.Ticker(symbol).quarterly_financials
        return df.T if df is not None else pd.DataFrame()
    except:
        return pd.DataFrame()

@st.cache_data
def fetch_annual_financials(symbol: str):
    try:
        df = yf.Ticker(symbol).financials
        return df.T if df is not None else pd.DataFrame()
    except:
        return pd.DataFrame()

@st.cache_data
def fetch_daily_price_history(symbol: str):
    try:
        df = yf.Ticker(symbol).history(period="1d", interval="1h")
        return df if df is not None else pd.DataFrame()
    except:
        return pd.DataFrame()

# -------------------------
# AUTO COMPLETE
# -------------------------
def search_wrapper(query, **kwargs):
    if not query or len(query.strip()) < 2:
        return []
    try:
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}"
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=5)
        if r.status_code != 200:
            return []
        data = r.json()
        return [
            f"{item.get('symbol')} - {item.get('shortname')}"
            for item in data.get("quotes", [])
            if item.get("symbol") and item.get("shortname")
        ][:15]
    except:
        return []

# -------------------------
# MARKET CAP
# -------------------------
def format_market_cap(value, currency="USD"):
    try: value = float(value)
    except: return f"{currency} 0"

    if value >= 1e12: return f"{currency} {value/1e12:.2f}T"
    if value >= 1e9: return f"{currency} {value/1e9:.2f}B"
    if value >= 1e6: return f"{currency} {value/1e6:.2f}M"
    return f"{currency} {value:,.0f}"

# -------------------------
# RECOMMENDATION
# -------------------------
def recommendation_from_sentiment(label, confidence):
    if label == "positive" and confidence >= 0.6: return "BUY"
    if label == "negative" and confidence >= 0.6: return "SELL"
    return "HOLD"

# -------------------------
# DASHBOARD UI
# -------------------------
st.title("Stock Dashboard by SongChiTienQuan")

if "selected_symbol" not in st.session_state:
    st.session_state.selected_symbol = ""

selected = st_searchbox(
    label="Enter a company name or stock symbol",
    search_function=search_wrapper,
    placeholder="Start typing...",
    default_value=st.session_state.selected_symbol,
    clear_on_submit=False,
)

if selected:
    st.session_state.selected_symbol = selected.split(" - ")[0].upper()

symbol = st.session_state.selected_symbol or st.text_input("Or type a symbol directly:", "").upper()

if not symbol:
    st.write("Select a stock symbol to begin.")

else:
    info = fetch_stock_info(symbol)

    if not info or info.get("regularMarketPrice") is None:
        st.warning("No information found for this symbol.")
    else:
        st.header("Company Information")
        st.subheader(f"Name: {info.get('longName', 'N/A')}")
        currency = info.get("currency", "USD")
        st.metric("Trading Currency", currency)

        market_cap = info.get("marketCap", 0)
        st.subheader(f"Market Cap: {format_market_cap(market_cap, currency)}")
        st.caption(f"Raw Value: {market_cap:,}")

        # Price Chart
        history = fetch_daily_price_history(symbol)
        if not history.empty:
            df = history.rename_axis("Date").reset_index()
            df["Date"] = pd.to_datetime(df["Date"])
            fig = go.Figure(data=[
                go.Candlestick(
                    x=df["Date"], open=df["Open"], high=df["High"],
                    low=df["Low"], close=df["Close"]
                )
            ])
            fig.update_layout(xaxis_rangeslider_visible=False)
            st.header("Daily Price Chart")
            st.plotly_chart(fig, use_container_width=True)

        # Financial Statements
        q_fin = fetch_quarterly_financials(symbol)
        a_fin = fetch_annual_financials(symbol)
        st.header("Financial Statements")

        period = st.radio("Period", ["Quarterly", "Annual"], horizontal=True)
        df = q_fin if period == "Quarterly" else a_fin

        if not df.empty:
            df_plot = df.rename_axis("Date").reset_index()
            df_plot["Date"] = df_plot["Date"].astype(str)

            revenue_col = next((c for c in ["Total Revenue", "TotalRevenue", "Revenue"] if c in df_plot.columns), None)
            net_col = next((c for c in ["Net Income", "NetIncome"] if c in df_plot.columns), None)

            if revenue_col:
                st.write("Revenue")
                st.altair_chart(
                    alt.Chart(df_plot).mark_bar().encode(x="Date:O", y=f"{revenue_col}:Q"),
                    use_container_width=True
                )

            if net_col:
                st.write("Net Income")
                st.altair_chart(
                    alt.Chart(df_plot).mark_bar().encode(x="Date:O", y=f"{net_col}:Q"),
                    use_container_width=True
                )

        # Sentiment
        st.header("Financial Sentiment Analysis (FinBERT)")
        news_text = st.text_area("Text input:", height=180)

        if st.button("Analyze Sentiment"):
            label, conf = get_sentiment(news_text)
            st.write(f"Sentiment: {label.upper()} (confidence {conf:.2f})")

            rec = recommendation_from_sentiment(label, conf)
            st.write(f"Recommendation: {rec}")
