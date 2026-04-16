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

# --- CONFIGURAÇÕES E CACHE ---
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
def calcular_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    # Correção da lógica de retorno do RSI
    rsi = 100 - (100 / (1 + rs))
    return rsi

@st.cache_data(ttl=600)
def obter_dados_ticker(ticker):
    try:
        if not ticker.endswith(".SA"): ticker += ".SA"
        t = yf.Ticker(ticker)
        hist = t.history(period="1y")
        return t.info, hist
    except:
        return None, None

# --- BARRA LATERAL: FILTROS E BUSCA ---
st.sidebar.title("🎯 Estratégia e Busca")

# Área de Pesquisa Única
busca_direta = st.sidebar.text_input("🔍 Buscar Ação Única:", placeholder="Ex: PETR4").upper()

st.sidebar.divider()

with st.sidebar.expander("📊 Filtros B3 (Geral)", expanded=True):
    f_pl = st.slider("P/L Máximo", 0.0, 50.0, 50.0, step=0.5, on_change=ativar_filtros)
    f_pvp = st.slider("P/VP Máximo", 0.0, 10.0, 10.0, step=0.1, on_change=ativar_filtros)
    f_dy = st.slider("DY Mínimo (%)", 0.0, 20.0, 0.0, step=0.5, on_change=ativar_filtros)
    f_div_ebitda = st.slider("Dív.Líq/EBITDA Máximo", 0.0, 10.0, 10.0, step=0.5, on_change=ativar_filtros)

if st.sidebar.button("Resetar Monitor Geral"):
    st.session_state.filtros_ativos = False
    st.rerun()

# --- CABEÇALHO ---
st.title("🤖 Monitor IA")
st.caption(f"Atualização: {time.strftime('%H:%M:%S')} | Local: Blumenau/SC")

# --- LISTA AMPLIADA DA BOLSA BRASILEIRA ---
# Incluindo as principais Blue Chips e Small Caps
tickers_b3 = [
    'PETR4', 'VALE3', 'ITUB4', 'BBAS3', 'BBDC4', 'ABEV3', 'SANB11', 'EGIE3', 
    'WEGE3', 'TRPL4', 'SAPR11', 'ITSA4', 'B3SA3', 'ELET3', 'GGBR4', 'JBSS3', 
    'RENT3', 'SUZB3', 'RAIL3', 'VIVT3', 'CPLE6', 'BBSE3', 'PSSA3', 'HYPE3', 
    'CSNA3', 'GOAU4', 'UGPA3', 'PRIO3', 'LREN3', 'RDOR3', 'RADL3', 'TOTS3'
]

# --- LÓGICA DE EXIBIÇÃO ---
tickers_para_processar = [busca_direta] if busca_direta else tickers_b3
vencedoras_count = 0

for tkr in tickers_para_processar:
    info, hist = obter_dados_ticker(tkr)
    if info and not hist.empty:
        # Extração de Indicadores
        pl = info.get('trailingPE', 0) or 0
        pvp = info.get('priceToBook', 0) or 0
        dy = (info.get('dividendYield', 0) or 0) * 100
        ebitda = info.get('ebitda', 1) or 1
        div_e = (info.get('totalDebt', 0) - info.get('totalCash', 0)) / ebitda
        
        # Lógica de Filtro
        passou = True
        if not busca_direta and st.session_state.filtros_ativos:
            if not (pl <= f_pl and pvp <= f_pvp and dy >= f_dy and div_e <= f_div_ebitda):
                passou = False
        
        if passou:
            vencedoras_count += 1
            st.divider()
            # Layout clássico: Título e Gráfico
            st.header(f"🏢 {info.get('longName', tkr)}")
            st.line_chart(hist['Close'])

            # Cards de Indicadores Fundamentais
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Preço Atual", f"R$ {hist['Close'].iloc[-1]:.2f}")
            c2.metric("P/L", round(pl, 2))
            c3.metric("DY", f"{dy:.2f}%")
            c4.metric("Dív.Líq/EBITDA", round(div_e, 2))

            # Inteligência de Mercado e Links
            st.subheader(f"🕵️‍♂️ Inteligência de Mercado: {tkr}")
            
            noticias = []
            texto_sentimento = ""
            try:
                url = f"https://news.google.com/rss/search?q={tkr}+when:2d&hl=pt-BR&gl=BR&ceid=BR:pt-419"
                feed = feedparser.parse(url)
                for entry in feed.entries[:3]:
                    titulo = entry.title.split(' - ')[0]
                    noticias.append({"t": titulo, "l": entry.link})
                    texto_sentimento += f"{titulo}. "
            except: pass

            score_p = sum(texto_sentimento.lower().count(w) for w in ["alta", "lucro", "compra", "subiu"])
            score_n = sum(texto_sentimento.lower().count(w) for w in ["queda", "prejuízo", "venda", "caiu"])

            col_v, col_r = st.columns([1, 1])
            with col_v:
                if score_p > score_n: st.success("**VEREDITO: COMPRA ✅**")
                elif score_n > score_p: st.error("**VEREDITO: CAUTELA ⚠️**")
                else: st.warning("**VEREDITO: NEUTRO ⚖️**")
            
            with col_r:
                # Correção das f-strings para evitar erro de fechamento
                rsi_series = calcular_rsi(hist['Close'])
                rsi_val = float(rsi_series.iloc[-1]) if not rsi_series.empty else 50.0
                st.write(f"📈 RSI Atual: {rsi_val:.2f}")
                st.write(f"🧠 Sentimento IA: {score_p} Positivos | {score_n} Negativos")

            with st.expander("📌 Ver Manchetes Analisadas (Links Oficiais)"):
                for n in noticias:
                    st.markdown(f"• {n['t']} [[Link]({n['l']})]")

if vencedoras_count == 0:
    st.info("💡 Nenhuma ação atende aos filtros atuais na lista expandida.")
