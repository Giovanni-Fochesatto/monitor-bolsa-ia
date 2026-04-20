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
st_autorefresh(interval=180 * 1000, key="data_refresh")  # 3 minutos

if "telegram_ativado" not in st.session_state:
    st.session_state.telegram_ativado = False

# ===================== BANCO DE DADOS =====================
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

# ===================== TELEGRAM REAL =====================
def enviar_alerta_telegram(bot_token, chat_id, mensagem):
    if not bot_token or not chat_id:
        return False
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {"chat_id": chat_id, "text": mensagem, "parse_mode": "HTML"}
        r = requests.post(url, json=payload, timeout=10)
        return r.status_code == 200
    except:
        return False

# ===================== RSI =====================
def calcular_rsi_series(close: pd.Series, window: int = 14) -> pd.Series:
    if len(close) < window:
        return pd.Series([50.0] * len(close), index=close.index)
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
    rs = gain / loss.where(loss != 0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def calcular_rsi(data, window: int = 14):
    if len(data) < window:
        return 50.0
    return float(calcular_rsi_series(data, window).iloc[-1])

# ===================== CANDLES EXTREMAMENTE EVOLUÍDA =====================
def detectar_candlestick_patterns(hist):
    if len(hist) < 15:
        return {"padrao": "Dados insuficientes", "forca": 0, "descricao": ""}
    
    o, h, l, c, v = hist['Open'].iloc[-15:], hist['High'].iloc[-15:], hist['Low'].iloc[-15:], hist['Close'].iloc[-15:], hist['Volume'].iloc[-15:]
    body = abs(c - o)
    upper = h - c.where(c > o, o)
    lower = o.where(c > o, c) - l
    range_c = h - l + 1e-8
    avg_vol = v.mean()
    
    patterns = []
    forca = 0
    i = -1  # último candle
    
    # 1. Doji / Spinning Top
    if body.iloc[i] / range_c.iloc[i] < 0.1:
        patterns.append("Doji")
        forca += 2
    # 2. Hammer / Inverted Hammer
    if lower.iloc[i] > 2 * body.iloc[i] and body.iloc[i] / range_c.iloc[i] < 0.3 and v.iloc[i] > avg_vol * 1.2:
        patterns.append("Hammer (Bullish)")
        forca += 5
    if upper.iloc[i] > 2 * body.iloc[i] and body.iloc[i] / range_c.iloc[i] < 0.3 and v.iloc[i] > avg_vol * 1.2:
        patterns.append("Shooting Star (Bearish)")
        forca += 5
    # 3. Engulfing
    if c.iloc[i] > o.iloc[i] and c.iloc[i-1] < o.iloc[i-1] and c.iloc[i] > o.iloc[i-1] and o.iloc[i] < c.iloc[i-1]:
        patterns.append("Bullish Engulfing")
        forca += 7
    if c.iloc[i] < o.iloc[i] and c.iloc[i-1] > o.iloc[i-1] and c.iloc[i] < o.iloc[i-1] and o.iloc[i] > c.iloc[i-1]:
        patterns.append("Bearish Engulfing")
        forca += 7
    # 4. Morning / Evening Star
    if len(hist) >= 3 and c.iloc[i-2] < o.iloc[i-2] and body.iloc[i-1] < 0.3*(h.iloc[i-1]-l.iloc[i-1]) and c.iloc[i] > (o.iloc[i-2] + c.iloc[i-2])/2:
        patterns.append("Morning Star (Bullish)")
        forca += 8
    # 5. Harami
    if body.iloc[i] < body.iloc[i-1] and ((c.iloc[i] > o.iloc[i-1] and c.iloc[i] < o.iloc[i]) or (c.iloc[i] < o.iloc[i-1] and c.iloc[i] > o.iloc[i])):
        patterns.append("Harami")
        forca += 4
    # 6. Piercing Line / Dark Cloud Cover
    if c.iloc[i] > o.iloc[i] and c.iloc[i-1] < o.iloc[i-1] and c.iloc[i] > (o.iloc[i-1] + c.iloc[i-1])/2 and o.iloc[i] < c.iloc[i-1]:
        patterns.append("Piercing Line (Bullish)")
        forca += 6
    if c.iloc[i] < o.iloc[i] and c.iloc[i-1] > o.iloc[i-1] and c.iloc[i] < (o.iloc[i-1] + c.iloc[i-1])/2 and o.iloc[i] > c.iloc[i-1]:
        patterns.append("Dark Cloud Cover (Bearish)")
        forca += 6
    # 7. Three White Soldiers / Three Black Crows
    if (c.iloc[i] > o.iloc[i] and c.iloc[i-1] > o.iloc[i-1] and c.iloc[i-2] > o.iloc[i-2] and
        c.iloc[i] > c.iloc[i-1] and c.iloc[i-1] > c.iloc[i
