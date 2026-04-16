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

# --- ESTILIZAÇÃO CUSTOMIZADA ---
st.markdown("""
    <style>
    .metric-card { background-color: #1e2129; padding: 15px; border-radius: 10px; border: 1px solid #3d414d; }
    .stMetric { background-color: #0e1117; padding: 10px; border-radius: 5px; }
    </style>
""", unsafe_allow_html=True)

# --- BARRA LATERAL: FILTROS E LEGENDAS ---
st.sidebar.title("🎯 Estratégia")

with st.sidebar.expander("📊 Filtros de Valuation", expanded=True):
    f_pl = st.slider("P/L Máximo", 0.0, 50.0, 15.0, step=0.5)
    st.caption("P/L: Preço/Lucro. Indica o tempo de retorno do investimento.")
    
    f_pvp = st.slider("P/VP Máximo", 0.0, 10.0, 2.0, step=0.1)
    st.caption("P/VP: Preço/Valor Patrimonial. Abaixo de 1.0 pode indicar desconto.")

with st.sidebar.expander("💰 Filtros de Rentabilidade", expanded=True):
    f_dy = st.slider("DY Mínimo (%)", 0.0, 20.0, 5.0, step=0.5)
    st.caption("Dividend Yield: Rendimento em dividendos no último ano.")
    
    f_roe = st.slider("ROE Mínimo (%)", 0.0, 40.0, 12.0, step=1.0)
    st.caption("ROE: Retorno sobre o capital dos sócios.")

with st.sidebar.expander("⚖️ Filtros de Saúde Financeira", expanded=True):
    f_ev_ebitda = st.slider("EV/EBITDA Máximo", 0.0, 30.0, 10.0, step=0.5)
    f_div_ebitda = st.slider("Dív.Líq/EBITDA Máximo", 0.0, 10.0, 3.0, step=0.5)
    st.caption("Dív.Líq/EBITDA: Mede o nível de endividamento operacional.")

# --- FUNÇÕES CORE ---
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

# --- CABEÇALHO ---
st.title("🤖 Monitor IA")
st.caption(f"Última atualização: {time.strftime('%H:%M:%S')} | Local: Blumenau/SC")

# --- PROCESSAMENTO ---
tickers = ['PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'BBAS3.SA', 'ABEV3.SA', 'SANB11.SA', 'EGIE3.SA', 'WEGE3.SA']
vencedoras = []

# Primeira passada: Filtragem e Coleta para o Ranking
for tkr in tickers:
    info, hist = obter_dados_ticker(tkr)
    if info and not hist.empty:
        # Extração de dados com tratamento de erro
        pl = info.get('trailingPE', 0) or 0
        pvp = info.get('priceToBook', 0) or 0
        dy = (info.get('dividendYield', 0) or 0) * 100
        roe = (info.get('returnOnEquity', 0) or 0) * 100
        margem = (info.get('profitMargins', 0) or 0) * 100
        ev_ebitda = info.get('enterpriseToEbitda', 0) or 0
        ebitda = info.get('ebitda', 1)
        div_e = (info.get('totalDebt', 0) - info.get('totalCash', 0)) / ebitda if ebitda != 0 else 0
        
        # Lógica do Filtro
        if (pl <= f_pl and pvp <= f_pvp and dy >= f_dy and 
            roe >= f_roe and ev_ebitda <= f_ev_ebitda and div_e <= f_div_ebitda):
            
            p_atual = hist['Close'].iloc[-1]
            lpa = info.get('trailingEps', 0) or 0
            vpa = info.get('bookValue', 0) or 0
            v_graham = calcular_graham(lpa, vpa)
            upside = ((v_graham / p_atual) - 1) * 100 if v_graham > 0 else 0
            
            vencedoras.append({
                "Empresa": info.get('shortName', tkr),
                "Ticker": tkr,
                "Preço": round(p_atual, 2),
                "P/L": round(pl, 2),
                "DY %": round(dy, 2),
                "ROE %": round(roe, 2),
                "Preço Justo (Graham)": round(v_graham, 2),
                "Potencial %": round(upside, 2),
                "info_obj": info,
                "hist_obj": hist
            })

# --- INTERFACE DE RESULTADOS ---
if vencedoras:
    # 1. Ranking Comparativo
    st.subheader("🏆 Ranking de Oportunidades (Filtrado)")
    df_rank = pd.DataFrame(vencedoras).drop(columns=['info_obj', 'hist_obj'])
    st.dataframe(df_rank.sort_values(by="Potencial %", ascending=False), use_container_width=True, hide_index=True)
    
    # 2. Análise Detalhada Individual
    for acao in vencedoras:
        with st.container():
            st.divider()
            col_tit, col_val = st.columns([3, 1])
            col_tit.header(f"🏢 {acao['Empresa']} ({acao['Ticker']})")
            col_val.metric("Preço Justo (Graham)", f"R$ {acao['Preço Justo (Graham)']}", f"{acao['Potencial %']}%")

            # Gráfico com MA200
            h = acao['hist_obj']
            h['MA200'] = h['Close'].rolling(window=200).mean()
            st.line_chart(h[['Close', 'MA200']])

            # Indicadores Fundamentalistas
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            inf = acao['info_obj']
            c1.metric("P/L", acao['P/L'])
            c2.metric("P/VP", round(inf.get('priceToBook', 0), 2))
            c3.metric("DY", f"{acao['DY %']}%")
            c4.metric("ROE", f"{acao['ROE %']}%")
            c5.metric("EV/EBITDA", round(inf.get('enterpriseToEbitda', 0), 2))
            c6.metric("Marg. Líq.", f"{round(inf.get('profitMargins', 0)*100, 1)}%")

            # IA e Notícias
            st.write("---")
            st.subheader("🕵️‍♂️ Inteligência de Mercado")
            # (Aqui você mantém sua função de raspagem de notícias e sentimento atual)
            st.info(f"Análise Técnica: RSI está em {round(calcular_rsi(h['Close']).iloc[-1], 2)}")
            
else:
    st.info("💡 Nenhuma ação atende aos filtros rigorosos selecionados. Tente flexibilizar os critérios na barra lateral.")

st.caption("Engenharia de IA 2026 - Terminal de Investimentos Pro.")
