import streamlit as st
import yfinance as yf
import pandas as pd
import time
import feedparser
import nltk
import numpy as np
import random
from newspaper import Article, Config
from streamlit_autorefresh import st_autorefresh

# --- CONFIGURAÇÕES INICIAIS ---
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

st.set_page_config(page_title="IA Investidor Pro", layout="wide")
st_autorefresh(interval=300 * 1000, key="data_refresh")

st.title("🤖 Monitor IA: Automação B3 Pro")
st.caption(f"Última atualização: {time.strftime('%H:%M:%S')} (Fuso: Blumenau/SC)")

# --- BARRA LATERAL: FILTROS E LEGENDAS ---
st.sidebar.header("🎯 Filtros e Definições")

# P/L
f_pl = st.sidebar.slider("P/L Máximo", 0.0, 50.0, 15.0, step=0.5)
st.sidebar.caption("💡 P/L: Preço/Lucro. Indica em quantos anos você recuperaria o investimento.")

# P/VP
f_pvp = st.sidebar.slider("P/VP Máximo", 0.0, 10.0, 2.5, step=0.1)
st.sidebar.caption("💡 P/VP: Indica se a ação está barata em relação ao patrimônio físico.")

# DY
f_dy = st.sidebar.slider("DY Mínimo (%)", 0.0, 20.0, 4.0, step=0.5)
st.sidebar.caption("💡 DY: Rendimento em dividendos pagos nos últimos 12 meses.")

# ROE
f_roe = st.sidebar.slider("ROE Mínimo (%)", 0.0, 40.0, 10.0, step=1.0)
st.sidebar.caption("💡 ROE: Eficiência em gerar lucro com o capital dos sócios.")

# Margem Líquida
f_margem = st.sidebar.slider("Marg. Líquida Mínima (%)", 0.0, 50.0, 5.0, step=1.0)
st.sidebar.caption("💡 Margem Líq: Porcentagem de lucro real sobre a receita total.")

# EV/EBITDA
f_ev_ebitda = st.sidebar.slider("EV/EBITDA Máximo", 0.0, 30.0, 12.0, step=0.5)
st.sidebar.caption("💡 EV/EBITDA: Valor da empresa sobre o lucro operacional.")

# Dívida/EBITDA
f_div_ebitda = st.sidebar.slider("Dív.Líq/EBITDA Máximo", 0.0, 10.0, 3.5, step=0.5)
st.sidebar.caption("💡 Dív.Líq/EBITDA: Tempo para pagar a dívida com o lucro operacional.")

st.sidebar.divider()
st.sidebar.caption("📈 RSI (IFR): Acima de 70 = Caro | Abaixo de 30 = Barato.")

# --- FUNÇÕES TÉCNICAS ---
def get_random_header():
    agents = ['Mozilla/5.0 (Windows NT 10.0; Win64; x64)','Mozilla/5.0 (X11; Linux x86_64)']
    return random.choice(agents)

def calcular_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# --- BLOCOS DE INTERFACE ---
def exibir_indicadores(info):
    st.subheader("📊 Indicadores Fundamentalistas")
    try:
        pl = info.get('trailingPE', 0)
        pvp = info.get('priceToBook', 0)
        roe = info.get('returnOnEquity', 0) * 100
        dy = info.get('dividendYield', 0) * 100
        margem = info.get('profitMargins', 0) * 100
        ev_ebitda = info.get('enterpriseToEbitda', 0)
        
        div_liq = info.get('totalDebt', 0) - info.get('totalCash', 0)
        ebitda_v = info.get('ebitda', 1)
        div_ebitda = div_liq / ebitda_v if ebitda_v != 0 else 0

        c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
        c1.metric("P/L", f"{pl:.2f}" if pl else "---")
        c2.metric("P/VP", f"{pvp:.2f}" if pvp else "---")
        c3.metric("DY", f"{dy:.2f}%")
        c4.metric("ROE", f"{roe:.1f}%")
        c5.metric("Marg. Líq.", f"{margem:.1f}%")
        c6.metric("EV/EBITDA", f"{ev_ebitda:.2f}" if ev_ebitda else "---")
        c7.metric("Dív.Líq/EBITDA", f"{div_ebitda:.2f}" if ebitda_v != 1 else "---")
    except Exception:
        st.warning("Indicadores financeiros parcialmente indisponíveis.")

def analise_ia_noticias(ticker, preco, media, rsi_atual):
    st.subheader(f"🕵️‍♂️ Inteligência de Mercado: {ticker}")
    noticias = []
    texto_total = ""
    config = Config()
    config.browser_user_agent = get_random_header()
    
    try:
        url = f"https://news.google.com/rss/search?q={ticker}+when:2d&hl=pt-BR&gl=BR&ceid=BR:pt-419"
        feed = feedparser.parse(url)
        for entry in feed.entries[:5]:
            titulo = entry.title.split(' - ')[0]
            noticias.append({"t": titulo, "l": entry.link})
            texto_total += f"{titulo}. "
    except Exception:
        pass

    pos = ["alta", "lucro", "compra", "subiu", "dividendo", "positivo"]
    neg = ["queda", "prejuízo", "venda", "caiu", "risco", "negativo"]
    txt = texto_total.lower()
    score_p = sum(txt.count(w) for w in pos)
    score_n = sum(txt.count(w) for w in neg)

    cv, ci = st.columns([1, 2])
    with cv:
        if preco > media and score_p > score_n: st.success("**VEREDITO: COMPRA ✅**")
        elif preco < media and score_n > score_p: st.error("**VEREDITO: EVITAR ❌**")
        else: st.warning("**VEREDITO: NEUTRO ⚖️**")

    with ci:
        st.write(f"🧠 Sentimento IA: {score_p} Positivos | {score_n
