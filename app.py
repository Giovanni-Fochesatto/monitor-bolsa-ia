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
        t = yf.Ticker(ticker)
        hist = t.history(period="1y")
        return t.info, hist
    except:
        return None, None

# --- BARRA LATERAL: FILTROS E LEGENDAS ---
st.sidebar.title("🎯 Estratégia")
with st.sidebar.expander("📊 Ajustar Filtros", expanded=True):
    st.write("Os filtros estão **desativados** (mostrando tudo). Eles ligam ao serem alterados.")
    
    f_pl = st.slider("P/L Máximo", 0.0, 50.0, 50.0, step=0.5, on_change=ativar_filtros)
    st.sidebar.caption("💡 P/L: Preço/Lucro. Indica em quantos anos você recuperaria o investimento.")
    
    f_pvp = st.slider("P/VP Máximo", 0.0, 10.0, 10.0, step=0.1, on_change=ativar_filtros)
    st.sidebar.caption("💡 P/VP: Indica se a ação está barata em relação ao patrimônio.")
    
    f_dy = st.slider("DY Mínimo (%)", 0.0, 20.0, 0.0, step=0.5, on_change=ativar_filtros)
    st.sidebar.caption("💡 DY: Rendimento em dividendos pagos nos últimos 12 meses.")

    f_div_ebitda = st.slider("Dív.Líq/EBITDA Máximo", 0.0, 10.0, 10.0, step=0.5, on_change=ativar_filtros)
    st.sidebar.caption("💡 Dív.Líq/EBITDA: Mede o endividamento. Indica quantos anos de lucro pagariam a dívida.")

if st.sidebar.button("Resetar Filtros"):
    st.session_state.filtros_ativos = False
    st.rerun()

# --- CABEÇALHO ---
st.title("🤖 Monitor IA")
status_txt = "✅ Filtros Ativos" if st.session_state.filtros_ativos else "⚪ Filtros Desligados (Mostrando Tudo)"
st.caption(f"{status_txt} | Local: Blumenau/SC")

# --- LISTA EXPANDIDA (Principais da B3) ---
tickers = [
    'PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'BBAS3.SA', 'BBDC4.SA', 'ABEV3.SA', 'SANB11.SA', 
    'EGIE3.SA', 'WEGE3.SA', 'RENT3.SA', 'MGLU3.SA', 'PRIO3.SA', 'VBBR3.SA', 'CSNA3.SA',
    'B3SA3.SA', 'ELET3.SA', 'GGBR4.SA', 'ITSA4.SA', 'JBSS3.SA', 'RAIL3.SA', 'SUZB3.SA',
    'VIVT3.SA', 'CPLE6.SA', 'TRPL4.SA', 'KLBN11.SA', 'SAPR11.SA', 'BBSE3.SA', 'PSSA3.SA'
]

vencedoras = []

# --- PROCESSAMENTO ---
progress_bar = st.progress(0)
for i, tkr in enumerate(tickers):
    progress_bar.progress((i + 1) / len(tickers))
    info, hist = obter_dados_ticker(tkr)
    if info and not hist.empty:
        # Indicadores base
        pl = info.get('trailingPE', 0) or 0
        pvp = info.get('priceToBook', 0) or 0
        dy = (info.get('dividendYield', 0) or 0) * 100
        
        # Cálculo da Dívida Líquida / EBITDA
        div_total = info.get('totalDebt', 0) or 0
        caixa = info.get('totalCash', 0) or 0
        ebitda = info.get('ebitda', 0) or 1 # Evitar divisão por zero
        div_e = (div_total - caixa) / ebitda if ebitda != 0 else 0
        
        # Lógica de Filtro
        passou = True
        if st.session_state.filtros_ativos:
            if not (pl <= f_pl and pvp <= f_pvp and dy >= f_dy and div_e <= f_div_ebitda):
                passou = False
        
        if passou:
            p_atual = hist['Close'].iloc[-1]
            lpa = info.get('trailingEps', 0) or 0
            vpa = info.get('bookValue', 0) or 0
            v_graham = calcular_graham(lpa, vpa)
            upside = ((v_graham / p_atual) - 1) * 100 if v_graham > 0 else 0
            
            vencedoras.append({
                "Ticker": tkr,
                "Empresa": info.get('shortName', tkr),
                "Preço": round(p_atual, 2),
                "P/L": round(pl, 2),
                "DY %": round(dy, 2),
                "Dív.Líq/EBITDA": round(div_e, 2),
                "Preço Justo": round(v_graham, 2),
                "Potencial %": round(upside, 2),
                "info": info, "hist": hist
            })

# --- INTERFACE FINAL ---
if vencedoras:
    st.subheader("🏆 Ranking de Ativos Encontrados")
    df_rank = pd.DataFrame(vencedoras).drop(columns=['info', 'hist'])
    st.dataframe(df_rank.sort_values(by="Potencial %", ascending=False), use_container_width=True, hide_index=True)
    
    for acao in vencedoras:
        st.divider()
        st.header(f"🏢 {acao['Empresa']} ({acao['Ticker']})")
        
        st.line_chart(acao['hist']['Close'])

        # Bloco de Métricas com Dívida Líquida
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("P/L", acao['P/L'])
        c2.metric("DY", f"{acao['DY %']}%")
        c3.metric("P/VP", round(acao['info'].get('priceToBook', 0), 2))
        c4.metric("Dív.Líq/EBITDA", acao['Dív.Líq/EBITDA'])
        c5.metric("Preço Justo (Graham)", f"R$ {acao['Preço Justo']}")

        # IA e Links
        with st.expander("📌 Ver Manchetes e Links Oficiais"):
            url = f"https://news.google.com/rss/search?q={acao['Ticker']}+when:2d&hl=pt-BR&gl=BR&ceid=BR:pt-419"
            feed = feedparser.parse(url)
            for entry in feed.entries[:3]:
                st.markdown(f"• {entry.title.split(' - ')[0]} [[Link]({entry.link})]")
else:
    st.warning("Nenhuma ação atende aos filtros atuais.")
