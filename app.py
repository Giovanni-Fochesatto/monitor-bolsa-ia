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

def get_random_header():
    agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36'
    ]
    return random.choice(agents)

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
with st.sidebar.expander("📊 Filtros e Definições", expanded=True):
    f_pl = st.slider("P/L Máximo", 0.0, 50.0, 15.0, step=0.5)
    st.caption("P/L: Tempo de retorno do investimento via lucros.")
    
    f_pvp = st.slider("P/VP Máximo", 0.0, 10.0, 2.0, step=0.1)
    st.caption("P/VP: Valor de mercado sobre valor patrimonial.")
    
    f_dy = st.slider("DY Mínimo (%)", 0.0, 20.0, 5.0, step=0.5)
    st.caption("DY: Rendimento de dividendos no último ano.")

    f_ev_ebitda = st.slider("EV/EBITDA Máximo", 0.0, 30.0, 12.0, step=0.5)
    st.caption("EV/EBITDA: Valor da empresa sobre lucro operacional.") #

# --- CABEÇALHO ---
st.title("🤖 Monitor IA")
st.caption(f"Última atualização: {time.strftime('%H:%M:%S')} | Local: Blumenau/SC")

# --- PROCESSAMENTO ---
tickers = ['PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'BBAS3.SA', 'ABEV3.SA', 'SANB11.SA', 'EGIE3.SA']
vencedoras = []

for tkr in tickers:
    info, hist = obter_dados_ticker(tkr)
    if info and not hist.empty:
        pl = info.get('trailingPE', 0) or 0
        pvp = info.get('priceToBook', 0) or 0
        dy = (info.get('dividendYield', 0) or 0) * 100
        ev_ebitda = info.get('enterpriseToEbitda', 0) or 0
        
        if (pl <= f_pl and pvp <= f_pvp and dy >= f_dy and ev_ebitda <= f_ev_ebitda):
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
                "Preço Justo": round(v_graham, 2),
                "Potencial %": round(upside, 2),
                "info": info,
                "hist": hist
            })

# --- INTERFACE DE RESULTADOS ---
if vencedoras:
    st.subheader("🏆 Ranking de Oportunidades")
    df_rank = pd.DataFrame(vencedoras).drop(columns=['info', 'hist'])
    st.dataframe(df_rank.sort_values(by="Potencial %", ascending=False), use_container_width=True, hide_index=True)
    
    for acao in vencedoras:
        st.divider()
        st.header(f"🏢 {acao['Empresa']} ({acao['Ticker']})")
        
        # Gráfico MA200
        h = acao['hist']
        h['MA200'] = h['Close'].rolling(window=200).mean()
        st.line_chart(h[['Close', 'MA200']])

        # Indicadores
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("P/L", acao['P/L'])
        c2.metric("DY", f"{acao['DY %']}%")
        c3.metric("Preço Justo", f"R$ {acao['Preço Justo']}")
        c4.metric("Potencial", f"{acao['Potencial %']}%")

        # --- SEÇÃO DE NOTÍCIAS E LINKS ---
        st.subheader("🕵️‍♂️ Inteligência de Mercado e Links")
        noticias_detalhes = []
        texto_analise = ""
        user_config = Config()
        user_config.browser_user_agent = get_random_header()
        
        try:
            url_news = f"https://news.google.com/rss/search?q={acao['Ticker']}+when:2d&hl=pt-BR&gl=BR&ceid=BR:pt-419"
            feed = feedparser.parse(url_news)
            for entry in feed.entries[:5]: 
                titulo = entry.title.split(' - ')[0]
                noticias_detalhes.append({"titulo": titulo, "link": entry.link})
                texto_analise += f"{titulo}. "
        except:
            pass

        # Veredito de Sentimento
        pos = ["alta", "dividendo", "lucro", "compra", "subiu"]
        neg = ["queda", "risco", "prejuízo", "venda", "caiu"]
        txt = texto_analise.lower()
        score_p = sum(txt.count(p) for p in pos)
        score_n = sum(txt.count(n) for n in neg)

        col_v, col_n = st.columns([1, 2])
        with col_v:
            if score_p > score_n: st.success("**SENTIMENTO: POSITIVO 👍**")
            elif score_n > score_p: st.error("**SENTIMENTO: CAUTELA ⚠️**")
            else: st.warning("**SENTIMENTO: NEUTRO ⚖️**")

        with col_n:
            with st.expander("📌 Ver Manchetes Analisadas (Links Oficiais)"): #
                for n in noticias_detalhes:
                    st.markdown(f"• {n['titulo']} [[Link]({n['link']})]")
else:
    st.info("💡 Nenhuma ação atende aos filtros. Ajuste os sliders na barra lateral.")
