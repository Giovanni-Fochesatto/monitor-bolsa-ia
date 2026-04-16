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
def calcular_graham(lpa, vpa):
    if lpa > 0 and vpa > 0:
        return np.sqrt(22.5 * lpa * vpa)
    return 0

def calcular_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

@st.cache_data(ttl=600)
def obter_dados_ticker(ticker):
    try:
        if not ticker.endswith(".SA"): ticker += ".SA"
        t = yf.Ticker(ticker)
        hist = t.history(period="1y")
        if hist.empty: return None, None
        return t.info, hist
    except:
        return None, None

# --- BARRA LATERAL: FILTROS ---
st.sidebar.title("🎯 Estratégia")
with st.sidebar.expander("📊 Ajustar Filtros", expanded=True):
    f_pl = st.slider("P/L Máximo", 0.0, 50.0, 50.0, step=0.5, on_change=ativar_filtros)
    f_pvp = st.slider("P/VP Máximo", 0.0, 10.0, 10.0, step=0.1, on_change=ativar_filtros)
    f_dy = st.slider("DY Mínimo (%)", 0.0, 20.0, 0.0, step=0.5, on_change=ativar_filtros)
    f_div_ebitda = st.slider("Dív.Líq/EBITDA Máximo", 0.0, 10.0, 10.0, step=0.5, on_change=ativar_filtros)

if st.sidebar.button("Resetar Filtros"):
    st.session_state.filtros_ativos = False
    st.rerun()

# --- CABEÇALHO ---
st.title("🤖 Monitor IA")

# --- NAVEGAÇÃO POR ABAS ---
tab_monitor, tab_busca, tab_comparacao = st.tabs(["🔍 Monitor Geral", "🏢 Busca Única", "⚔️ Comparação"])

# --- ABA 1: MONITOR GERAL ---
with tab_monitor:
    tickers_b3 = ['PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'BBAS3.SA', 'BBDC4.SA', 'ABEV3.SA', 'SANB11.SA', 'EGIE3.SA', 'WEGE3.SA', 'TRPL4.SA', 'SAPR11.SA']
    vencedoras = []
    
    status_txt = "✅ Filtros Ativos" if st.session_state.filtros_ativos else "⚪ Filtros Desligados"
    st.caption(f"{status_txt} | Local: Blumenau/SC")

    for tkr in tickers_b3:
        info, hist = obter_dados_ticker(tkr)
        if info and not hist.empty:
            pl = info.get('trailingPE', 0) or 0
            pvp = info.get('priceToBook', 0) or 0
            dy = (info.get('dividendYield', 0) or 0) * 100
            div_e = (info.get('totalDebt', 0) - info.get('totalCash', 0)) / (info.get('ebitda', 1) or 1)
            
            passou = True
            if st.session_state.filtros_ativos:
                if not (pl <= f_pl and pvp <= f_pvp and dy >= f_dy and div_e <= f_div_ebitda):
                    passou = False
            
            if passou:
                vencedoras.append({"Ticker": tkr, "P/L": round(pl, 2), "DY %": round(dy, 2), "Dív.EBITDA": round(div_e, 2), "info": info, "hist": hist})

    if vencedoras:
        df_rank = pd.DataFrame([{"Ticker": x["Ticker"], "P/L": x["P/L"], "DY %": x["DY %"], "Dív.EBITDA": x["Dív.EBITDA"]} for x in vencedoras])
        st.dataframe(df_rank, use_container_width=True, hide_index=True)

# --- ABA 2: BUSCA ÚNICA ---
with tab_busca:
    busca_tkr = st.text_input("Digite o Ticker da Ação (ex: PETR4, MGLU3):").upper()
    if busca_tkr:
        info, hist = obter_dados_ticker(busca_tkr)
        if info:
            st.header(f"🏢 {info.get('longName')}")
            st.line_chart(hist['Close'])
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Preço Atual", f"R$ {hist['Close'].iloc[-1]:.2f}")
            c2.metric("P/L", round(info.get('trailingPE', 0), 2))
            c3.metric("DY", f"{info.get('dividendYield', 0)*100:.2f}%")
            c4.metric("Dív.Líq/EBITDA", round((info.get('totalDebt', 0)-info.get('totalCash',0))/info.get('ebitda',1), 2))
            
            with st.expander("📌 Notícias Recentes"):
                url = f"https://news.google.com/rss/search?q={busca_tkr}+when:2d&hl=pt-BR&gl=BR&ceid=BR:pt-419"
                feed = feedparser.parse(url)
                for entry in feed.entries[:5]:
                    st.markdown(f"• {entry.title} [[Link]({entry.link})]")
        else:
            st.error("Ticker não encontrado.")

# --- ABA 3: COMPARAÇÃO ---
with tab_comparacao:
    col_a, col_b = st.columns(2)
    tkr_a = col_a.text_input("Ação A:", value="PETR4").upper()
    tkr_b = col_b.text_input("Ação B:", value="VALE3").upper()
    
    if tkr_a and tkr_b:
        info_a, hist_a = obter_dados_ticker(tkr_a)
        info_b, hist_b = obter_dados_ticker(tkr_b)
        
        if info_a and info_b:
            # Gráfico Comparativo de Performance (Normalizado)
            st.subheader("📈 Performance Relativa (1 Ano)")
            df_comp = pd.DataFrame({
                tkr_a: (hist_a['Close'] / hist_a['Close'].iloc[0]) * 100,
                tkr_b: (hist_b['Close'] / hist_b['Close'].iloc[0]) * 100
            })
            st.line_chart(df_comp)
            
            # Tabela de Duelo
            st.subheader("⚔️ Duelo de Indicadores")
            dados_comp = {
                "Indicador": ["P/L", "P/VP", "DY %", "ROE %", "Dív.Líq/EBITDA"],
                tkr_a: [
                    round(info_a.get('trailingPE', 0), 2),
                    round(info_a.get('priceToBook', 0), 2),
                    f"{info_a.get('dividendYield', 0)*100:.2f}%",
                    f"{info_a.get('returnOnEquity', 0)*100:.2f}%",
                    round((info_a.get('totalDebt', 0)-info_a.get('totalCash',0))/info_a.get('ebitda',1), 2)
                ],
                tkr_b: [
                    round(info_b.get('trailingPE', 0), 2),
                    round(info_b.get('priceToBook', 0), 2),
                    f"{info_b.get('dividendYield', 0)*100:.2f}%",
                    f"{info_b.get('returnOnEquity', 0)*100:.2f}%",
                    round((info_b.get('totalDebt', 0)-info_b.get('totalCash',0))/info_b.get('ebitda',1), 2)
                ]
            }
            st.table(pd.DataFrame(dados_comp))
