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

st.set_page_config(page_title="Monitor IA Pro", layout="wide")
st_autorefresh(interval=300 * 1000, key="data_refresh")

# Inicializar estado do filtro
if 'filtros_ativos' not in st.session_state:
    st.session_state.filtros_ativos = False

def ativar_filtros():
    st.session_state.filtros_ativos = True

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

# --- BARRA LATERAL: FILTROS COM GATILHO ---
st.sidebar.title("🎯 Estratégia")
with st.sidebar.expander("📊 Ajustar Filtros", expanded=True):
    st.write("Os filtros abaixo estão inativos até você alterá-los.")
    
    f_pl = st.slider("P/L Máximo", 0.0, 50.0, 50.0, step=0.5, on_change=ativar_filtros)
    st.sidebar.caption("P/L: Indica o tempo de retorno em anos.")
    
    f_pvp = st.slider("P/VP Máximo", 0.0, 10.0, 10.0, step=0.1, on_change=ativar_filtros)
    st.sidebar.caption("P/VP: Preço sobre Valor Patrimonial.")
    
    f_dy = st.slider("DY Mínimo (%)", 0.0, 20.0, 0.0, step=0.5, on_change=ativar_filtros)
    st.sidebar.caption("DY: Rendimento de dividendos anual.")

    f_ev_ebitda = st.slider("EV/EBITDA Máximo", 0.0, 30.0, 30.0, step=0.5, on_change=ativar_filtros)
    st.sidebar.caption("EV/EBITDA: Valor da firma/Lucro Operacional.") #

if st.sidebar.button("Resetar Filtros"):
    st.session_state.filtros_ativos = False
    st.rerun()

# --- CABEÇALHO ---
st.title("🤖 Monitor IA")
status_filtro = "✅ Filtros Ativos" if st.session_state.filtros_ativos else "⚪ Exibindo Tudo (Filtros Desligados)"
st.caption(f"{status_filtro} | Atualização: {time.strftime('%H:%M:%S')} | Local: Blumenau/SC")

# --- PROCESSAMENTO ---
tickers = ['PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'BBAS3.SA', 'ABEV3.SA', 'SANB11.SA', 'EGIE3.SA', 'WEGE3.SA']
vencedoras = []

for tkr in tickers:
    info, hist = obter_dados_ticker(tkr)
    if info and not hist.empty:
        pl = info.get('trailingPE', 0) or 0
        pvp = info.get('priceToBook', 0) or 0
        dy = (info.get('dividendYield', 0) or 0) * 100
        ev_ebitda = info.get('enterpriseToEbitda', 0) or 0
        
        # Lógica: Se filtros ativos, aplica regras. Se não, passa tudo.
        passou_filtro = True
        if st.session_state.filtros_ativos:
            if not (pl <= f_pl and pvp <= f_pvp and dy >= f_dy and ev_ebitda <= f_ev_ebitda):
                passou_filtro = False
        
        if passou_filtro:
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

# --- INTERFACE ---
if vencedoras:
    st.subheader("🏆 Ranking de Ativos")
    df_rank = pd.DataFrame(vencedoras).drop(columns=['info', 'hist'])
    st.dataframe(df_rank.sort_values(by="Potencial %", ascending=False), use_container_width=True, hide_index=True)
    
    for acao in vencedoras:
        st.divider()
        col_t, col_p = st.columns([3, 1])
        col_t.header(f"🏢 {acao['Empresa']} ({acao['Ticker']})")
        col_p.metric("Preço Justo (Graham)", f"R$ {acao['Preço Justo']}", f"{acao['Potencial %']}%")
        
        h = acao['hist']
        h['MA200'] = h['Close'].rolling(window=200).mean()
        st.line_chart(h[['Close', 'MA200']])

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("P/L", acao['P/L'])
        c2.metric("DY", f"{acao['DY %']}%")
        c3.metric("P/VP", round(acao['info'].get('priceToBook', 0), 2))
        c4.metric("EV/EBITDA", round(acao['info'].get('enterpriseToEbitda', 0), 2))

        # IA e Links Oficiais
        st.subheader("🕵️‍♂️ Inteligência e Notícias")
        noticias = []
        texto_noticias = ""
        try:
            url = f"https://news.google.com/rss/search?q={acao['Ticker']}+when:2d&hl=pt-BR&gl=BR&ceid=BR:pt-419"
            feed = feedparser.parse(url)
            for entry in feed.entries[:5]:
                titulo = entry.title.split(' - ')[0]
                noticias.append({"t": titulo, "l": entry.link})
                texto_noticias += f"{titulo}. "
        except: pass

        score_p = sum(texto_noticias.lower().count(w) for w in ["alta", "lucro", "compra", "subiu"])
        score_n = sum(texto_noticias.lower().count(w) for w in ["queda", "prejuízo", "venda", "caiu"])

        v_col, l_col = st.columns([1, 2])
        with v_col:
            if score_p > score_n: st.success("SENTIMENTO: POSITIVO 👍")
            elif score_n > score_p: st.error("SENTIMENTO: CAUTELA ⚠️")
            else: st.warning("SENTIMENTO: NEUTRO ⚖️")
        
        with l_col:
            with st.expander("📌 Ver Manchetes e Links Oficiais"):
                for n in noticias:
                    st.markdown(f"• {n['t']} [[Link]({n['l']})]")
else:
    st.info("Nenhuma ação encontrada para os critérios atuais.")
