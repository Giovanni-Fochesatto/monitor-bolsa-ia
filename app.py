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

# --- SISTEMA DE PRIVACIDADE (HEADER ROTATION) ---
def get_random_header():
    agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0'
    ]
    return random.choice(agents)

# --- FUNÇÕES TÉCNICAS ---
def calcular_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# --- ANÁLISE FUNDAMENTALISTA (ESTILO INVESTIDOR 10) ---
def analise_fundamentalista_i10(ticker, info):
    st.subheader(f"📊 Indicadores Fundamentalistas (Estilo Investidor 10)")
    
    try:
        pl = info.get('trailingPE', 0)
        pvp = info.get('priceToBook', 0)
        roe = info.get('returnOnEquity', 0) * 100
        dy = info.get('dividendYield', 0) * 100
        margem = info.get('profitMargins', 0) * 100
        ebitda = info.get('enterpriseToEbitda', 0)

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("P/L", f"{pl:.2f}" if pl else "---")
        c2.metric("P/VP", f"{pvp:.2f}" if pvp else "---")
        c3.metric("DY", f"{dy:.2f}%")
        c4.metric("ROE", f"{roe:.1f}%")
        c5.metric("Marg. Líq.", f"{margem:.1f}%")
        c6.metric("EV/EBITDA", f"{ebitda:.2f}" if ebitda else "---")

        with st.expander("🎓 O que significam esses números? (Legenda)"):
            st.markdown("""
            * **P/L:** Indica em quantos anos você recuperaria seu investimento através do lucro.
            * **P/VP:** Preço sobre Valor Patrimonial. Abaixo de 1,00 pode indicar desconto.
            * **DY (Dividend Yield):** Rendimento em dividendos nos últimos 12 meses.
            * **ROE:** Eficiência em gerar lucro com o capital próprio. Acima de 15% é excelente.
            """)
    except Exception as e:
        st.warning(f"Erro ao carregar fundamentos: {e}")

# --- FUNÇÃO DE IA E SENTIMENTO ---
def analise_minuciosa_ia(ticker, preco, media, rsi_atual):
    st.subheader(f"🕵️‍♂️ Inteligência de Mercado: {ticker}")
    manchetes_encontradas = []
    texto_total_analise = ""
    
    user_config = Config()
    user_config.browser_user_agent = get_random_header()
    
    try:
        url_news = f"https://news.google.com/rss/search?q={ticker}+when:2d&hl=pt-BR&gl=BR&ceid=BR:pt-419"
        feed = feedparser.parse(url_news)
        
        if feed.entries:
            for entry in feed.entries[:5]: 
                titulo = entry.title.split(' - ')[0]
                manchetes_encontradas.append(titulo)
                texto_total_analise += f"{titulo}. "
                try:
                    time.sleep(random.uniform(1, 2))
                    article = Article(entry.link, config=user_config)
                    article.download()
                    article.parse()
                    texto_total_analise += article.text[:300] + " "
                except: continue
    except Exception as e:
        st.error(f"Erro no radar de notícias: {e}")

    # Motor de Sentimento
    positivas = ["alta", "dividendo", "lucro", "compra", "subiu", "recorde"]
    negativas = ["queda", "risco", "prejuízo", "venda", "caiu", "dívida"]
    
    texto_limpo = texto_total_analise.lower()
    otimismo = sum(texto_limpo.count(p) for p in positivas)
    pessimismo = sum(texto_limpo.count(p) for p in negativas)

    col_v, col_i = st.columns([1, 2])
    with col_v:
        if preco > media and otimismo > pessimismo:
            st.success("**VEREDITO: COMPRA ✅**")
        elif preco < media and pessimismo > otimismo:
            st.error("**VEREDITO: EVITAR ❌**")
        else:
            st.warning("**VEREDITO: NEUTRO ⚖️**")

    with col_i:
        st.write(f"Técnica: {'Acima' if preco > media else 'Abaixo'} da MA200 | RSI: {rsi_atual:.1f}")

# --- BUSCA DE DADOS ---
def buscar_dados(ticker):
    try:
        t = yf.Ticker(ticker)
        df = t.history(period="1y")
        if df.empty: return None, None
        
        df['MA200'] = df['Close'].rolling(window=200).mean()
        df['RSI'] = calcular_rsi(df['Close'])
        return df, t.info
    except Exception as e:
        st.error(f"Erro ao buscar {ticker}: {e}")
        return None, None

# --- LOOP PRINCIPAL ---
tickers = ['PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'BBAS3.SA']

for ticker in tickers:
    dados, info = buscar_dados(ticker)
    if dados is not None:
        st.divider()
        st.header(f"🏢 {info.get('longName', ticker)}")
        
        preco_
