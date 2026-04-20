import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import feedparser
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh
import datetime
import sqlite3
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Monitor IA Hedge Fund - Bitcoin", layout="wide")
st_autorefresh(interval=180 * 1000, key="data_refresh")

if "telegram_ativado" not in st.session_state:
    st.session_state.telegram_ativado = False

# ===================== BANCO =====================
def init_db():
    conn = sqlite3.connect('sinais_bitcoin.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS sinais (
        data TEXT, ticker TEXT, preco_usd REAL, preco_brl REAL,
        veredito TEXT, motivo TEXT, expectancy REAL, sharpe REAL,
        rsi REAL, timeframe TEXT, candle_padrao TEXT
    )''')
    conn.commit()
    conn.close()

def salvar_sinal(acao):
    init_db()
    conn = sqlite3.connect('sinais_bitcoin.db')
    c = conn.cursor()
    agora = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute('''INSERT INTO sinais VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
        (agora, acao['Ticker'], acao['Preço_USD'], acao.get('Preço_BRL', 0),
         acao['Veredito'], acao['Motivo'], acao.get('ExpectancyCompra', 0),
         acao.get('SharpeCompra', 0), acao.get('RSI', 50), acao.get('Timeframe', 'Daily'),
         acao.get('CandleInfo', {}).get('padrao', '')))
    conn.commit()
    conn.close()

# ===================== RSI =====================
def calcular_rsi_series(close, window=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

# ===================== ML + BACKTEST =====================
def simular_performance_historica(hist):
    if len(hist) < 150:
        return {
            "expectancy_compra": 0,
            "sharpe_compra": 0,
            "sortino_compra": 0,
            "taxa_compra": 0,
            "score_ml": 50
        }

    df = hist.copy()

    # Features
    df["ret"] = df["Close"].pct_change()
    df["rsi"] = calcular_rsi_series(df["Close"])
    df["sma20"] = df["Close"].rolling(20).mean()
    df["sma50"] = df["Close"].rolling(50).mean()
    df["mom"] = df["Close"].pct_change(5)

    df = df.dropna()

    # Target
    df["target"] = (df["ret"].shift(-1) > 0).astype(int)

    X = df[["rsi", "mom"]]
    y = df["target"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression()
    model.fit(X_scaled, y)

    prob = model.predict_proba(X_scaled)[:, 1]
    df["prob"] = prob

    # Estratégia baseada em probabilidade
    df["signal"] = (df["prob"] > 0.55).astype(int)
    df["strategy"] = df["signal"].shift(1) * df["ret"]

    # Métricas
    ganhos = df[df["strategy"] > 0]["strategy"]
    perdas = df[df["strategy"] < 0]["strategy"]

    winrate = len(ganhos) / (len(ganhos) + len(perdas)) if len(ganhos)+len(perdas)>0 else 0

    media_gain = ganhos.mean() if len(ganhos) > 0 else 0
    media_loss = perdas.mean() if len(perdas) > 0 else 0

    expectancy = (winrate * media_gain) + ((1 - winrate) * media_loss)

    std = df["strategy"].std()
    sharpe = (df["strategy"].mean() / std) * np.sqrt(252) if std != 0 else 0

    downside = df[df["strategy"] < 0]["strategy"].std()
    sortino = (df["strategy"].mean() / downside) * np.sqrt(252) if downside != 0 else 0

    score_ml = float(df["prob"].iloc[-1] * 100)

    return {
        "expectancy_compra": float(expectancy),
        "sharpe_compra": float(sharpe),
        "sortino_compra": float(sortino),
        "taxa_compra": float(winrate),
        "score_ml": score_ml
    }

# ===================== DADOS =====================
def obter_dados():
    btc = yf.Ticker("BTC-USD")
    hist = btc.history(period="5y")
    return hist, float(btc.fast_info.last_price)

# ===================== IA =====================
def processar_bitcoin():
    hist, preco = obter_dados()

    if hist.empty:
        return None

    sim = simular_performance_historica(hist)

    score = sim["score_ml"]

    # Ensemble + IA
    if score > 70 and sim["sharpe_compra"] > 1:
        veredito = "COMPRA FORTE 🚀"
        motivo = f"Probabilidade alta ({score:.1f}%) + Sharpe forte"

    elif score < 40 and sim["sharpe_compra"] < 0.5:
        veredito = "VENDA FORTE 🚨"
        motivo = f"Probabilidade baixa ({score:.1f}%) + fraqueza estatística"

    else:
        veredito = "NEUTRO INTELIGENTE 🧠"
        motivo = f"Zona cinzenta ({score:.1f}%)"

    return {
        "Ticker": "BTC",
        "Preço_USD": preco,
        "Veredito": veredito,
        "Motivo": motivo,
        "ExpectancyCompra": sim["expectancy_compra"],
        "SharpeCompra": sim["sharpe_compra"],
        "SortinoCompra": sim["sortino_compra"],
        "ScoreML": sim["score_ml"],
        "Hist": hist
    }

# ===================== APP =====================
st.title("🤖 IA Hedge Fund Bitcoin")

resultado = processar_bitcoin()

if resultado:
    salvar_sinal(resultado)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Preço", f"${resultado['Preço_USD']:,.0f}")
    col2.metric("Sharpe", f"{resultado['SharpeCompra']:.2f}")
    col3.metric("Sortino", f"{resultado['SortinoCompra']:.2f}")
    col4.metric("Score IA", f"{resultado['ScoreML']:.1f}")

    st.success(resultado["Veredito"])
    st.write(resultado["Motivo"])

    fig = go.Figure()
    hist = resultado["Hist"]

    fig.add_trace(go.Candlestick(
        x=hist.index,
        open=hist['Open'],
        high=hist['High'],
        low=hist['Low'],
        close=hist['Close']
    ))

    st.plotly_chart(fig, use_container_width=True)
