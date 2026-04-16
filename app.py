import streamlit as st
import yfinance as yf
import pandas as pd
import time
import feedparser
import nltk
import numpy as np
from streamlit_autorefresh import st_autorefresh

# --- CONFIGURAÇÕES E AMBIENTE ---
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

st.set_page_config(page_title="Monitor IA Pro", layout="wide")
st_autorefresh(interval=300 * 1000, key="data_refresh")

if 'filtros_ativos' not in st.session_state:
    st.session_state.filtros_ativos = False

def ativar_filtros():
    st.session_state.filtros_ativos = True

# --- FUNÇÕES TÉCNICAS ---
def calcular_graham(lpa, vpa):
    """Calcula o Preço Justo de Graham: sqrt(22.5 * LPA * VPA)"""
    if lpa > 0 and vpa > 0:
        return np.sqrt(22.5 * lpa * vpa)
    return 0

def calcular_rsi(data, window=14):
    """Calcula o RSI para identificar sobrecompra ou sobrevenda"""
    if len(data) < window: return 50.0
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1])

@st.cache_data(ttl=600)
def obter_dados_ticker(ticker, mercado):
    try:
        if mercado == "Brasil" and not ticker.endswith(".SA"):
            ticker += ".SA"
        t = yf.Ticker(ticker)
        hist = t.history(period="1y")
        return t.info, hist
    except:
        return None, None

@st.cache_data(ttl=300)
def obter_cambio():
    """Busca cotações das moedas em tempo real"""
    moedas = {'Dólar': 'USDBRL=X', 'Euro': 'EURBRL=X', 'Bitcoin': 'BTC-BRL'}
    resultados = {}
    for nome, ticker in moedas.items():
        try:
            data = yf.Ticker(ticker)
            dado = data.history(period='2d')
            if not dado.empty and len(dado) >= 2:
                atual = dado['Close'].iloc[-1]
                anterior = dado['Close'].iloc[-2]
                variacao = ((atual / anterior) - 1) * 100
                resultados[nome] = (atual, variacao)
            else:
                resultados[nome] = (data.fast_info.last_price, 0.0)
        except:
            resultados[nome] = (0.0, 0.0)
    return resultados

# --- BARRA LATERAL (SIDEBAR) ---
st.sidebar.title("🌎 Mercado e Estratégia")

# Câmbio
st.sidebar.subheader("💱 Câmbio em Tempo Real")
cambio = obter_cambio()
col_c1, col_c2 = st.sidebar.columns(2)
col_c1.metric("Dólar", f"R$ {cambio['Dólar'][0]:.2f}", f"{cambio['Dólar'][1]:.2f}%")
col_c2.metric("Euro", f"R$ {cambio['Euro'][0]:.2f}", f"{cambio['Euro'][1]:.2f}%")
st.sidebar.metric("Bitcoin", f"R$ {cambio['Bitcoin'][0]:.0f}", f"{cambio['Bitcoin'][1]:.2f}%")
st.sidebar.divider()

mercado_selecionado = st.sidebar.radio("Escolha o Mercado:", ["Brasil", "EUA"], on_change=ativar_filtros)
busca_direta = st.sidebar.text_input(f"🔍 Busca Rápida ({mercado_selecionado}):").upper()

with st.sidebar.expander("📊 Filtros de Valuation", expanded=True):
    f_pl = st.slider("P/L Máximo", 0.0, 5
