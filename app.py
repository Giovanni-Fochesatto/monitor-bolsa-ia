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

# --- BARRA LATERAL ---
st.sidebar.title("🌎 Mercado e Estratégia")
mercado_selecionado = st.sidebar.radio("Escolha o Mercado:", ["Brasil", "EUA"], on_change=ativar_filtros)
busca_direta = st.sidebar.text_input(f"🔍 Busca Rápida ({mercado_selecionado}):").upper()

with st.sidebar.expander("📊 Filtros de Valuation", expanded=True):
    f_pl = st.slider("P/L Máximo", 0.0, 50.0, 50.0, step=0.5, on_change=ativar_filtros)
    f_pvp = st.slider("P/VP Máximo", 0.0, 10.0, 10.0, step=0.1, on_change=ativar_filtros)
    f_dy = st.slider("DY Mínimo (%)", 0.0, 20.0, 0.0, step=0.5, on_change=ativar_filtros)
    f_div_ebitda = st.slider("Dív.Líq/EBITDA Máximo", 0.0, 15.0, 15.0, step=0.5, on_change=ativar_filtros)

# --- PROCESSAMENTO ---
lista_base = ['PETR4', 'VALE3', 'ITUB4', 'BBAS3', 'ABEV3', 'EGIE3', 'WEGE3'] if mercado_selecionado == "Brasil" else ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA']
tickers_para_processar = [busca_direta] if busca_direta else lista_base
moeda = "R$" if mercado_selecionado == "Brasil" else "US$"

# --- CABEÇALHO ---
st.title(f"🤖 Monitor IA - {mercado_selecionado}")
st.caption(f"Blumenau/SC | {time.strftime('%H:%M:%S')}")

dados_vencedoras = []
for tkr in tickers_para_processar:
    info, hist = obter_dados_ticker(tkr, mercado_selecionado)
    if info and not hist.empty:
        # Indicadores
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
        
        if not busca_direta and st.session_state.filtros_ativos:
            if not (pl <= f_pl and pvp <= f_pvp and dy >= f_dy and div_e <= f_div_ebitda):
                continue
        
        # --- ANÁLISE DE SENTIMENTO IA ---
        texto_noticias = ""
        links_noticias = []
        try:
            url = f"https://news.google.com/rss/search?q={tkr}&hl={'pt-BR' if mercado_selecionado == 'Brasil' else 'en-US'}"
            feed = feedparser.parse(url)
            for entry in feed.entries[:3]:
                titulo = entry.title.lower()
                texto_noticias += titulo + " "
                links_noticias.append({"t": entry.title, "l": entry.link})
        except: pass

        score_p = sum(texto_noticias.count(w) for w in ["alta", "lucro", "compra", "subiu", "dividend", "profit", "buy"])
        score_n = sum(texto_noticias.count(w) for w in ["queda", "prejuízo", "venda", "caiu", "risk", "loss", "sell"])
        rsi_val = calcular_rsi(hist['Close'])

        # Lógica do Veredito
        veredito = "NEUTRO ⚖️"
        cor_veredito = "warning"
        if score_p > score_n and rsi_val < 70:
            veredito = "COMPRA ✅"
            cor_veredito = "success"
        elif score_n > score_p or rsi_val > 75:
            veredito = "CAUTELA ⚠️"
            cor_veredito = "error"

        dados_vencedoras.append({
            "Ticker": tkr, "Empresa": info.get('shortName', tkr), "Preço": p_atual,
            "P/L": pl, "DY": dy, "Dívida": div_e, "Graham": p_justo, "Upside": upside,
            "Veredito": veredito, "Cor": cor_veredito, "Links": links_noticias, "RSI": rsi_val, "Hist": hist
        })

# --- EXIBIÇÃO ---
if dados_vencedoras:
    st.subheader("🏆 Ranking Geral")
    df_resumo = pd.DataFrame(dados_vencedoras)[["Ticker", "Preço", "DY", "Graham", "Upside", "Veredito"]]
    st.dataframe(df_resumo.sort_values(by="Upside", ascending=False), use_container_width=True, hide_index=True)

    for acao in dados_vencedoras:
        st.divider()
        col_tit, col_ver = st.columns([3, 1])
        col_tit.header(f"🏢 {acao['Empresa']} ({acao['Ticker']})")
        
        # ONDE ESTÁ A IA: O Veredito em destaque
        if acao["Cor"] == "success": col_ver.success(f"**{acao['Veredito']}**")
        elif acao["Cor"] == "error": col_ver.error(f"**{acao['Veredito']}**")
        else: col_ver.warning(f"**{acao['Veredito']}**")

        st.line_chart(acao['Hist']['Close'])

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Preço", f"{moeda} {acao['Preço']:.2f}")
        c2.metric("P/L", round(acao['P/L'], 2))
        c3.metric("DY", f"{acao['DY']:.2f}%")
        c4.metric("Dív.Líq/EBITDA", round(acao['Dívida'], 2))
        c5.metric("Preço Justo", f"{moeda} {acao['Graham']:.2f}", f"{acao['Upside']:.1f}%")

        with st.expander("📊 Detalhes Técnicos e Notícias"):
            st.write(f"📈 RSI (14d): {acao['RSI']:.2f}")
            for n in acao['Links']:
                st.markdown(f"• {n['t']} [[Link]({n['l']})]")
else:
    st.info("Nenhuma ação atende aos critérios.")
