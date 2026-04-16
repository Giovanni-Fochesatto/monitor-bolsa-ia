import streamlit as st
import yfinance as yf
import pandas as pd
import time
import feedparser
import nltk
import numpy as np
import random
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
    if len(data) < window: return 50.0
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1]

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
tickers_b3 = [
    'PETR4', 'VALE3', 'ITUB4', 'BBAS3', 'BBDC4', 'ABEV3', 'SANB11', 'EGIE3', 
    'WEGE3', 'TRPL4', 'SAPR11', 'ITSA4', 'B3SA3', 'ELET3', 'GGBR4', 'JBSS3', 
    'RENT3', 'SUZB3', 'RAIL3', 'VIVT3', 'CPLE6', 'BBSE3', 'PSSA3', 'HYPE3'
]

tickers_para_processar = [busca_direta] if busca_direta else tickers_b3
dados_vencedoras = []

# --- PROCESSAMENTO INICIAL PARA O RANKING ---
for tkr in tickers_para_processar:
    info, hist = obter_dados_ticker(tkr)
    if info and not hist.empty:
        pl = info.get('trailingPE', 0) or 0
        pvp = info.get('priceToBook', 0) or 0
        dy = (info.get('dividendYield', 0) or 0) * 100
        ebitda = info.get('ebitda', 1) or 1
        div_e = (info.get('totalDebt', 0) - info.get('totalCash', 0)) / ebitda
        
        lpa = info.get('trailingEps', 0) or 0
        vpa = info.get('bookValue', 0) or 0
        p_justo = calcular_graham(lpa, vpa)
        p_atual = hist['Close'].iloc[-1]
        upside = ((p_justo / p_atual) - 1) * 100 if p_justo > 0 else 0
        
        passou = True
        if not busca_direta and st.session_state.filtros_ativos:
            if not (pl <= f_pl and pvp <= f_pvp and dy >= f_dy and div_e <= f_div_ebitda):
                passou = False
        
        if passou:
            dados_vencedoras.append({
                "Ticker": tkr,
                "Empresa": info.get('shortName', tkr),
                "Preço": round(p_atual, 2),
                "P/L": round(pl, 2),
                "DY %": round(dy, 2),
                "Dív.Líq/EBITDA": round(div_e, 2),
                "Graham": round(p_justo, 2),
                "Upside %": round(upside, 2),
                "info": info,
                "hist": hist
            })

# --- EXIBIÇÃO: TABELA DE RANKING NO TOPO ---
if dados_vencedoras:
    st.subheader("🏆 Ranking de Oportunidades")
    df_ranking = pd.DataFrame(dados_vencedoras).drop(columns=['info', 'hist'])
    st.dataframe(df_ranking.sort_values(by="Upside %", ascending=False), use_container_width=True, hide_index=True)

    # --- EXIBIÇÃO: DETALHAMENTO INDIVIDUAL ---
    for acao in dados_vencedoras:
        st.divider()
        st.header(f"🏢 {acao['Empresa']} ({acao['Ticker']})")
        st.line_chart(acao['hist']['Close'])

        # Cards de Indicadores
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Preço Atual", f"R$ {acao['Preço']:.2f}")
        c2.metric("P/L", acao['P/L'])
        c3.metric("DY", f"{acao['DY %']:.2f}%")
        c4.metric("Dív.Líq/EBITDA", acao['Dív.Líq/EBITDA'])
        c5.metric("Preço Justo (Graham)", f"R$ {acao['Graham']:.2f}", f"{acao['Upside %']:.1f}%")

        # Inteligência de Mercado
        with st.expander("📌 Notícias e Sentimento IA"):
            rsi_final = calcular_rsi(acao['hist']['Close'])
            st.write(f"📈 RSI Atual: {rsi_final:.2f}")
            
            url = f"https://news.google.com/rss/search?q={acao['Ticker']}+when:2d&hl=pt-BR&gl=BR&ceid=BR:pt-419"
            feed = feedparser.parse(url)
            for entry in feed.entries[:3]:
                st.markdown(f"• {entry.title.split(' - ')[0]} [[Link]({entry.link})]")
else:
    st.info("💡 Nenhuma ação atende aos filtros atuais.")
