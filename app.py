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

# Auto-refresh a cada 5 minutos
st_autorefresh(interval=300 * 1000, key="data_refresh")

st.title("🤖 Monitor IA: Automação B3 Pro")
st.caption(f"Última atualização: {time.strftime('%H:%M:%S')} (Fuso: Blumenau/SC)")

# --- BARRA LATERAL: FILTROS DE ESTRATÉGIA ---
st.sidebar.header("🎯 Filtros de Estratégia")
f_pl = st.sidebar.slider("P/L Máximo", 0.0, 50.0, 15.0, step=0.5)
f_pvp = st.sidebar.slider("P/VP Máximo", 0.0, 10.0, 2.5, step=0.1)
f_dy = st.sidebar.slider("DY Mínimo (%)", 0.0, 20.0, 4.0, step=0.5)
f_roe = st.sidebar.slider("ROE Mínimo (%)", 0.0, 40.0, 10.0, step=1.0)
f_margem = st.sidebar.slider("Marg. Líquida Mínima (%)", 0.0, 50.0, 5.0, step=1.0)
f_ev_ebitda = st.sidebar.slider("EV/EBITDA Máximo", 0.0, 30.0, 12.0, step=0.5)
f_div_ebitda = st.sidebar.slider("Dív.Líq/EBITDA Máximo", 0.0, 10.0, 3.5, step=0.5)

# --- FUNÇÕES DE APOIO ---
def get_random_header():
    agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36'
    ]
    return random.choice(agents)

def calcular_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# --- BLOCOS DE INTERFACE ---
def analise_fundamentalista(ticker, info):
    st.subheader("📊 Indicadores Fundamentalistas")
    try:
        pl = info.get('trailingPE', 0)
        pvp = info.get('priceToBook', 0)
        roe = info.get('returnOnEquity', 0) * 100
        dy = info.get('dividendYield', 0) * 100
        margem = info.get('profitMargins', 0) * 100
        ev_ebitda = info.get('enterpriseToEbitda', 0)
        
        div_liq = info.get('totalDebt', 0) - info.get('totalCash', 0)
        ebitda_v = info.get('ebitda', 0)
        div_ebitda = div_liq / ebitda_v if ebitda_v and ebitda_v != 0 else 0

        c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
        c1.metric("P/L", f"{pl:.2f}" if pl else "---")
        c2.metric("P/VP", f"{pvp:.2f}" if pvp else "---")
        c3.metric("DY", f"{dy:.2f}%")
        c4.metric("ROE", f"{roe:.1f}%")
        c5.metric("Marg. Líq.", f"{margem:.1f}%")
        c6.metric("EV/EBITDA", f"{ev_ebitda:.2f}" if ev_ebitda else "---")
        c7.metric("Dív.Líq/EBITDA", f"{div_ebitda:.2f}" if div_ebitda != 0 else "---")

        with st.expander("🎓 O que significam esses números?"):
            st.markdown(f"""
            * **P/L:** Preço sobre Lucro. Tempo de retorno do investimento.
            * **P/VP:** Valor de mercado sobre valor patrimonial.
            * **DY:** Rendimento de dividendos nos últimos 12 meses.
            * **ROE:** Retorno sobre o patrimônio líquido.
            * **Margem Líquida:** Porcentagem de lucro sobre a receita.
            * **EV/EBITDA:** Valor da empresa sobre lucro operacional.
            * **Dívida Líquida / EBITDA:** Mede o nível de endividamento da empresa.
            * **RSI (IFR):** Indica se a ação está cara (>70) ou barata (<30).
            """)
    except Exception:
        st.error("Erro ao processar indicadores financeiros.")

def analise_ia(ticker, preco, media, rsi_atual):
    st.subheader(f"🕵️‍♂️ Inteligência de Mercado: {ticker}")
    noticias = []
    texto_analise = ""
    config = Config()
    config.browser_user_agent = get_random_header()
    
    try:
        url = f"https://news.google.com/rss/search?q={ticker}+when:2d&hl=pt-BR&gl=BR&ceid=BR:pt-419"
        feed = feedparser.parse(url)
        for entry in feed.entries[:5]:
            titulo = entry.title.split(' - ')[0]
            noticias.append({"t": titulo, "l": entry.link})
            texto_analise += f"{titulo}. "
    except Exception:
        pass

    # Sentimento simplificado
    pos = ["alta", "lucro", "compra", "subiu", "dividendo"]
    neg = ["queda", "prejuízo", "venda", "caiu", "risco"]
    txt = texto_analise.lower()
    score_p = sum(txt.count(w) for w in pos)
    score_n = sum(txt.count(w) for w in neg)

    cv, ci = st.columns([1, 2])
    with cv:
        if preco > media and score_p > score_n: st.success("**VEREDITO: COMPRA ✅**")
        elif preco < media and score_n > score_p: st.error("**VEREDITO: EVITAR ❌**")
        else: st.warning("**VEREDITO: NEUTRO ⚖️**")

    with ci:
        st.write(f"🧠 Sentimento IA: {score_p} Positivos | {score_n} Negativos")
        st.write(f"📈 RSI Atual: {rsi_atual:.2f}")

    with st.expander("📌 Ver Manchetes Analisadas"):
        for n in noticias:
            st.markdown(f"• {n['t']} [[Link]({n['l']})]")

# --- PROCESSAMENTO ---
tickers = ['PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'BBAS3.SA', 'ABEV3.SA']

for tkr in tickers:
    try:
        obj = yf.Ticker(tkr)
        hist = obj.history(period="1y")
        if hist.empty: continue
        
        info = obj.info
        # Lógica de Filtro
        pl_v = info.get('trailingPE', 0) or 0
        pvp_v = info.get('priceToBook', 0) or 0
        dy_v = (info.get('dividendYield', 0) or 0) * 100
        
        if pl_v <= f_pl and pvp_v <= f_pvp and dy_v >= f_dy:
            st.divider()
            st.header(f"🏢 {info.get('longName', tkr)}")
            
            hist['MA200'] = hist['Close'].rolling(window=200).mean()
            hist['RSI'] = calcular_rsi(hist['Close'])
            
            p_atual = float(hist['Close'].iloc[-1])
            m_200 = hist['MA200'].iloc[-1]
            m_200 = m_200 if not np.isnan(m_200) else p_atual
            r_atual = hist['RSI'].iloc[-1]
            r_atual = float(r_atual) if not np.isnan(r_atual) else 50.0

            st.line_chart(hist[['Close', 'MA200']])
            analise_fundamentalista(tkr, info)
            analise_ia(tkr, p_atual, m_200, r_atual)
    except Exception:
        continue
