import streamlit as st
import yfinance as yf
import pandas as pd
import time
import feedparser
import nltk
import numpy as np
from newspaper import Article, Config
from streamlit_autorefresh import st_autorefresh

# --- CONFIGURAÇÕES INICIAIS ---
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

st.set_page_config(page_title="IA Investidor Pro", layout="wide")

# Auto-refresh a cada 5 minutos para manter os dados de Blumenau atualizados
st_autorefresh(interval=300 * 1000, key="data_refresh")

st.title("🤖 Monitor IA: Automação B3 Pro")
st.caption(f"Última atualização: {time.strftime('%H:%M:%S')} (Blumenau, SC)")

# --- FUNÇÕES TÉCNICAS ---
def calcular_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# --- FUNÇÃO DE ANÁLISE DE NOTÍCIAS E SENTIMENTO (VERSÃO ULTRA-RESILIENTE) ---
def analise_minuciosa_ia(ticker, preco, media, rsi_atual):
    st.subheader(f"🕵️‍♂️ Inteligência de Mercado: {ticker}")
    
    # GARANTIA ANTI-TELA PRETA: Inicialização de variáveis
    manchetes_encontradas = []
    texto_total_analise = ""
    detectadas_pos = []
    detectadas_neg = []
    otimismo = 0
    pessimismo = 0

    # 1. Configuração de Bypass de Bot
    timestamp = int(time.time())
    url_news = f"https://news.google.com/rss/search?q={ticker}+when:2d&hl=pt-BR&gl=BR&ceid=BR:pt&t={timestamp}"
    
    user_config = Config()
    user_config.browser_user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36'
    user_config.request_timeout = 15
    user_config.fetch_images = False 

    try:
        # 2. Captura do Feed
        feed = feedparser.parse(url_news)
        
        if feed.entries:
            with st.spinner(f'IA vasculhando fontes para {ticker}...'):
                for entry in feed.entries[:5]: 
                    # Passo Crítico: Guardar a manchete antes de qualquer erro de download
                    titulo = entry.title.split(' - ')[0]
                    manchetes_encontradas.append(titulo)
                    texto_total_analise += f"{titulo}. "
                    
                    try:
                        time.sleep(1) # Jitter para evitar bloqueio de IP
                        article = Article(entry.link, config=user_config)
                        article.download()
                        article.parse()
                        texto_total_analise += article.text[:300] + " "
                    except:
                        continue 
        else:
            st
