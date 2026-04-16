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

# --- SISTEMA DE OCULTAÇÃO E PRIVACIDADE (USER-AGENTS) ---
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
        # Extração de métricas principais
        pl = info.get('trailingPE', 0)
        pvp = info.get('priceToBook', 0)
        roe = info.get('returnOnEquity', 0) * 100
        dy = info.get('dividendYield', 0) * 100
        margem = info.get('profitMargins', 0) * 100
        ebitda = info.get('enterpriseToEbitda', 0)

        # Layout de Cards
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("P/L", f"{pl:.2f}" if pl else "---")
        c2.metric("P/VP", f"{pvp:.2f}" if pvp else "---")
        c3.metric("DY", f"{dy:.2f}%")
        c4.metric("ROE", f"{roe:.1f}%")
        c5.metric("Marg. Líq.", f"{margem:.1f}%")
        c6.metric("EV/EBITDA", f"{ebitda:.2f}" if ebitda else "---")

        if roe > 15 and dy > 6:
            st.success("💎 Perfil 'Value Investing': Alta rentabilidade e bons dividendos.")
    except:
        st.warning("⚠️ Alguns indicadores fundamentalistas estão indisponíveis para este ativo.")

# --- FUNÇÃO DE IA E SENTIMENTO ---
def analise_minuciosa_ia(ticker, preco, media, rsi_atual):
    st.subheader(f"🕵️‍♂️ Inteligência de Mercado: {ticker}")
    manchetes_encontradas = []
    texto_total_analise = ""
    
    user_config = Config()
    user_config.browser_user_agent = get_random_header()
    user_config.request_timeout = 15

    try:
        url_news = f"https://news.google.com/rss/search?q={ticker}+when:2d&hl=pt-BR&gl=BR&ceid=BR:pt-419"
        feed = feedparser.parse(url_news)
        
        if feed.entries:
            for entry in feed.entries[:5]: 
                titulo = entry.title.split(' - ')[0]
                manchetes_encontradas.append(titulo)
                texto_total_analise += f"{titulo}. "
                try:
                    time.sleep(random.uniform(1, 2)) # Jitter aleatório para evitar detecção
                    article = Article(entry.link, config=user_config)
                    article.download()
                    article.parse()
                    texto_total_analise += article.text[:300] + " "
                except: continue
    except:
        st.error("Erro ao conectar com o radar de notícias.")

    # Lógica de Sentimento
    positivas = ["alta", "dividendo", "lucro", "compra", "crescimento", "subiu", "positivo", "recorde", "recompra"]
    negativas = ["queda", "risco", "prejuízo", "venda", "caiu", "dívida", "crise", "negativo", "corte"]
    
    texto_limpo = texto_total_analise.lower()
    otimismo = sum(texto_limpo.count(p) for p in positivas)
    pessimismo = sum(texto_limpo.count(p) for p in negativas)

    # Veredito
    status_rsi = "Caro" if rsi_atual > 70 else "Barato" if rsi_atual < 30 else "Neutro"
    col_v, col_i = st.columns([1, 2])
    
    with col_v:
        if preco > media and otimismo > pessimismo:
            st.success("**VEREDITO: COMPRA ✅**")
        elif preco < media and pessimismo > otimismo:
            st.error("**VEREDITO: EVITAR ❌**")
        else:
            st.warning("**VEREDITO: NEUTRO ⚖️**")

    with col_i:
        st.write(f"Técnica: {'Acima' if preco > media else 'Abaixo'} da MA200 | RSI: {rsi_atual:.1f} ({status_rsi})")

    with st.expander("🔍 Ver Manchetes Analisadas"):
        for m in manchetes_encontradas:
            st.write(f"• {m}")

# --- BUSCA DE DADOS ---
def buscar_dados(ticker):
    try:
        t = yf.Ticker(ticker)
        df = t.history(period="1y")
        if df.empty: return None, None
        
        df['MA200'] = df['Close'].rolling(window=200).mean()
        df['RSI'] = calcular_rsi(df['Close'])
        return df, t.info
    except:
        return None, None

# --- LOOP PRINCIPAL ---
tickers = ['PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'BBAS3.SA']

for ticker in tickers:
    dados, info = buscar_dados(ticker)
    if dados is not None:
        st.divider()
        nome_completo = info.get('longName', ticker)
        st.header(f"🏢 {nome_completo} ({ticker})")
        
        preco_atual = float(dados['Close'].iloc[-1])
        media_val = dados['MA200'].iloc[-1]
        media_atual = float(media_val) if not np.isnan(media_val) else preco_atual
        rsi_atual = float(dados['RSI'].iloc[-1]) if not np.isnan(dados['RSI'].iloc[-1]) else 50
        
        # 1. Gráfico
        st.line_chart(dados[['Close', 'MA200']])
        
        # 2. Indicadores (Investidor 10)
        analise_fundamentalista_i10(ticker, info)
        
        # 3. Inteligência de IA
        analise_minuciosa_ia(ticker, preco_atual, media_atual, rsi_atual)

st.caption("Sistema Blindado 2026 - Proteção de IP Ativa via Header Rotation.")
