import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time
import feedparser
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh
import datetime
import sqlite3
import requests

st.set_page_config(page_title="Monitor IA Pro - Bitcoin", layout="wide")
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

def calcular_rsi(data):
    return float(calcular_rsi_series(data).iloc[-1]) if len(data) > 14 else 50

# ===================== BACKTEST INSTITUCIONAL =====================
def simular_performance_historica(hist):
    if len(hist) < 100:
        return {"expectancy_compra": 0, "sharpe_compra": 0, "taxa_compra": 0}

    df = hist.copy()
    df["retorno"] = df["Close"].pct_change()

    # Estratégia institucional híbrida
    df["sma20"] = df["Close"].rolling(20).mean()
    df["sma50"] = df["Close"].rolling(50).mean()
    df["rsi"] = calcular_rsi_series(df["Close"])

    df["sinal"] = 0
    df.loc[
        (df["Close"] > df["sma20"]) &
        (df["sma20"] > df["sma50"]) &
        (df["rsi"] < 70),
        "sinal"
    ] = 1

    df["ret_estrat"] = df["sinal"].shift(1) * df["retorno"]

    # Equity curve
    df["equity"] = (1 + df["ret_estrat"]).cumprod()

    # Drawdown
    peak = df["equity"].cummax()
    drawdown = (df["equity"] - peak) / peak
    max_dd = drawdown.min()

    ganhos = df[df["ret_estrat"] > 0]["ret_estrat"]
    perdas = df[df["ret_estrat"] < 0]["ret_estrat"]

    winrate = len(ganhos) / (len(ganhos) + len(perdas)) if len(ganhos)+len(perdas) > 0 else 0
    media_gain = ganhos.mean() if len(ganhos) > 0 else 0
    media_loss = perdas.mean() if len(perdas) > 0 else 0

    expectancy = (winrate * media_gain) + ((1 - winrate) * media_loss)

    std = df["ret_estrat"].std()
    sharpe = (df["ret_estrat"].mean() / std) * np.sqrt(252) if std != 0 else 0

    return {
        "expectancy_compra": float(expectancy),
        "sharpe_compra": float(sharpe),
        "taxa_compra": float(winrate)
    }

# ===================== DADOS =====================
def obter_dados(timeframe="Daily"):
    interval_map = {"1h": "1h", "4h": "4h", "Daily": "1d"}
    period_map = {"1h": "10d", "4h": "30d", "Daily": "5y"}

    btc = yf.Ticker("BTC-USD")
    hist = btc.history(period=period_map[timeframe], interval=interval_map[timeframe])

    return hist, float(btc.fast_info.last_price)

# ===================== PROCESSAMENTO =====================
def processar_bitcoin(timeframe="Daily"):
    hist, preco = obter_dados(timeframe)

    if hist.empty:
        return None

    close = hist["Close"]
    rsi = calcular_rsi(close)

    sma50 = close.rolling(50).mean().iloc[-1] if len(close) > 50 else close.mean()

    sim = simular_performance_historica(hist)

    # IA institucional
    if sim["expectancy_compra"] > 0.002 and sim["sharpe_compra"] > 1.2:
        if rsi < 70 and close.iloc[-1] > sma50:
            veredito = "COMPRA FORTE 🚀"
            motivo = "Alta probabilidade estatística + tendência + Sharpe alto"

    elif sim["expectancy_compra"] < 0 and sim["sharpe_compra"] < 0.5:
        veredito = "VENDA FORTE 🚨"
        motivo = "Expectativa negativa + risco elevado"

    else:
        veredito = "CAUTELA ⚠️"
        motivo = "Mercado indefinido"

    return {
        "Ticker": "BTC",
        "Preço_USD": preco,
        "RSI": rsi,
        "Veredito": veredito,
        "Motivo": motivo,
        "ExpectancyCompra": sim["expectancy_compra"],
        "SharpeCompra": sim["sharpe_compra"],
        "TaxaCompra": sim["taxa_compra"],
        "Hist": hist
    }

# ===================== APP =====================
st.title("🤖 Monitor IA Institucional - Bitcoin")

timeframe = st.sidebar.selectbox("Timeframe", ["Daily", "4h", "1h"])

resultado = processar_bitcoin(timeframe)

if resultado:
    salvar_sinal(resultado)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Preço", f"${resultado['Preço_USD']:,.0f}")
    col2.metric("RSI", f"{resultado['RSI']:.1f}")
    col3.metric("Sharpe", f"{resultado['SharpeCompra']:.2f}")
    col4.metric("Expectancy", f"{resultado['ExpectancyCompra']:.4f}")

    st.subheader("Veredito IA")
    st.success(resultado["Veredito"])
    st.write(resultado["Motivo"])

    # gráfico
    hist = resultado["Hist"]

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=hist.index,
        open=hist['Open'],
        high=hist['High'],
        low=hist['Low'],
        close=hist['Close']
    ))

    st.plotly_chart(fig, use_container_width=True)
