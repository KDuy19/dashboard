import altair as alt  # Graph
from streamlit_searchbox import st_searchbox  # Search box
import plotly.graph_objects as go  # Graph
import streamlit as st  # Deploy web
import yfinance as yf  # Data collecting from Yahoo Finance
import pandas as pd  # Table support
import requests  #  autocomplete search API
import torch #Sentiment predict
from transformers import AutoTokenizer, AutoModelForSequenceClassification

st.set_page_config(page_title="Stock Dashboard by SongChiTienQuan", layout="wide")

# -------------------------
# FINBERT (pretrained)
# -------------------------
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

    text = text.strip()
    if len(text) > 2000:
        text = text[:2000]

    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0].cpu().numpy()

        labels = ["negative", "neutral", "positive"]
        label = labels[int(probs.argmax())]
        confidence = float(probs.max())
        return label, confidence

    except Exception:
        return "neutral", 0.0


# -------------------------
# FETCH / UTIL FUNCTIONS
# -------------------------
@st.cache_data
def fetch_stock_info(symbol: str):
    try:
        ticker = yf.Ticker(symbol)
        fast = ticker.fast_info or {}
        safe = ticker.get_info() or {}

        info = {
            "longName": safe.get("longName") or safe.get("shortName") or symbol,
            "currency": fast.get("currency") or safe.get("currency") or "USD",
            "marketCap": fast.get("marketCap") or safe.get("marketCap") or 0,
            "regularMarketPrice": fast.get("lastPrice")
                or safe.get("regularMarketPrice")
                or safe.get("currentPrice"),
        }

        return info
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
# AUTO-COMPLETE SEARCH
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
        results = [
            f"{item.get('symbol')} - {item.get('shortname')}"
            for item in data.get("quotes", [])
            if item.get("symbol") and item.get("shortname")
        ]
        return results[:15]

    except:
        return []

# -------------------------
# MARKET CAP FORMAT
# -------------------------
def format_market_cap(value, currency="USD"):
    try:
        value = float(value)
    except:
        return f"{currency} 0"

    if value >= 1e12:
        return f"{currency} {value/1e12:.2f}T"
    if value >= 1e9:
        return f"{currency} {value/1e9:.2f}B"
    if value >= 1e6:
        return f"{currency} {value/1e6:.2f}M"
    return f"{currency} {value:,.0f}"

# -------------------------
# RECOMMENDATION LOGIC
# -------------------------
def recommendation_from_sentiment(label, confidence):
    if label == "positive" and confidence >= 0.6:
        return "BUY"
    if label == "negative" and confidence >= 0.6:
        return "SELL"
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
    key="stock_search_box"
)

if selected:
    st.session_state.selected_symbol = selected.split(" - ")[0].upper()

symbol = st.session_state.selected_symbol or st.text_input("Or type a symbol directly:", "").upper()

if not symbol:
    st.write("Select a stock symbol to begin.")
else:
    with st.spinner("Fetching company info..."):
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

        # -------------------------
        # PRICE CHART
        # -------------------------
        with st.spinner("Fetching price history..."):
            history = fetch_daily_price_history(symbol)

        if history.empty:
            st.info("No price history available.")
        else:
            df = history.rename_axis("Date").reset_index()
            df["Date"] = pd.to_datetime(df["Date"])

            if all(col in df.columns for col in ["Open", "High", "Low", "Close"]):
                fig = go.Figure(
                    data=[go.Candlestick(
                        x=df["Date"],
                        open=df["Open"],
                        high=df["High"],
                        low=df["Low"],
                        close=df["Close"]
                    )]
                )
                fig.update_layout(xaxis_rangeslider_visible=False)
                st.header("Daily Price Chart")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough data to plot a chart.")

        # -------------------------
        # FINANCIAL STATEMENTS
        # -------------------------
        q_fin = fetch_quarterly_financials(symbol)
        a_fin = fetch_annual_financials(symbol)

        st.header("Financial Statements")

        if q_fin.empty and a_fin.empty:
            st.info("No financial statements available.")
        else:
            period = st.radio("Period", ["Quarterly", "Annual"], horizontal=True)
            df = q_fin if period == "Quarterly" else a_fin

            if df.empty:
                st.info(f"No {period} financial data.")
            else:
                df_plot = df.rename_axis("Date").reset_index()
                df_plot["Date"] = df_plot["Date"].astype(str)

                # find revenue + income columns
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

        # -------------------------
        # SENTIMENT ANALYSIS
        # -------------------------
        st.header("Financial Sentiment Analysis (FinBERT)")
        st.write("Paste news or headlines related to this company:")

        news_text = st.text_area("Text input:", height=180)

        if st.button("Analyze Sentiment"):
            if not news_text.strip():
                st.warning("Please paste some news to analyze.")
            else:
                with st.spinner("Analyzing..."):
                    label, conf = get_sentiment(news_text)
                    st.write(f"Sentiment: {label.upper()} (confidence {conf:.2f})")

                    # show probability breakdown
                    try:
                        inputs = tokenizer(news_text, return_tensors="pt", truncation=True, padding=True)
                        with torch.no_grad():
                            outs = model(**inputs)
                        probs = torch.softmax(outs.logits, dim=1)[0].cpu().numpy()
                        probs_dict = dict(zip(["negative", "neutral", "positive"], probs.round(3).tolist()))
                        st.write("Probabilities:", probs_dict)
                    except:
                        pass

                    rec = recommendation_from_sentiment(label, conf)
                    st.write(f"Recommendation(not totally right so be careful bros): {rec}")
