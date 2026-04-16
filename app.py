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

# Auto-refresh a cada 5 minutos para manter os dados atualizados
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

# --- ANÁLISE FUNDAMENTALISTA (ESTILO INVESTIDOR 10) COM LEGENDA ---
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

        # Layout de Cards de Indicadores
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("P/L", f"{pl:.2f}" if pl else "---")
        c2.metric("P/VP", f"{pvp:.2f}" if pvp else "---")
        c3.metric("DY", f"{dy:.2f}%")
        c4.metric("ROE", f"{roe:.1f}%")
        c5.metric("Marg. Líq.", f"{margem:.1f}%")
        c6.metric("EV/EBITDA", f"{ebitda:.2f}" if ebitda else "---")

        # --- SEÇÃO: LEGENDA PARA LEIGOS ---
        with st.expander("🎓 O que significam esses números? (Legenda)"):
            st.markdown("""
            * **P/L (Preço sobre Lucro):** Indica em quantos anos você recuperaria seu investimento através do lucro da empresa. 
                * *Abaixo de 10-15 geralmente é visto como barato.*
            * **P/VP (Preço sobre Valor Patrimonial):** Indica quanto o mercado paga pelo patrimônio líquido da empresa. 
                * *Abaixo de 1,00 significa que a empresa vale na bolsa menos do que o patrimônio físico que ela possui.*
            * **DY (Dividend Yield):** O rendimento em dividendos pagos nos últimos 12 meses em relação ao preço atual. 
                * *Acima de 6% é considerado um bom rendimento para renda passiva.*
            * **ROE (Retorno sobre Patrimônio):** Mede a eficiência da empresa em gerar lucro com o dinheiro dos acionistas. 
                * *Quanto maior, melhor (acima de 15% é excelente).*
            * **Margem Líquida:** Quanto a empresa lucra de verdade para cada R$ 100 vendidos após pagar todas as despesas.
            * **EV/EBITDA:** Valor da empresa sobre o que ela gera de caixa operacional. Ajuda a ver se a empresa está cara ou barata ignorando impostos e dívidas.
            """)

        if roe > 15 and dy > 6:
            st.success("💎 **Perfil Value Investing:** Empresa rentável e boa pagadora de dividendos.")
            
    except Exception as e:
        st.warning(f"⚠️ Alguns indicadores fundamentalistas estão indisponíveis: {e}")

# --- FUNÇÃO DE IA E SENTIMENTO ---
def analise_minuciosa_ia(ticker, preco, media, rsi_atual):
    st.subheader(f"🕵️‍♂️ Inteligência de Mercado: {ticker}")
    
    # Inicialização para evitar erros de renderização
    manchetes_encontradas = []
    texto_total_analise = ""
    
    user_config = Config()
    user_config.browser_user_agent = get_random_header()
    user_config.request_timeout = 15

    try:
        url_news = f"https://news.google.com/rss/search?q={ticker}+when:2d&hl=pt-BR&gl=BR&ceid=BR:pt-419"
        feed = feedparser.parse(url_news)
        
        if feed.entries:
            with st.spinner(f'IA analisando notícias de {ticker}...'):
                for entry in feed.entries[:5]: 
                    titulo = entry.title.split(' - ')[0]
                    manchetes_encontradas.append(titulo)
                    texto_total_analise += f"{titulo}. "
                    try:
                        time.sleep(random.uniform(1, 2)) # Simula comportamento humano
                        article = Article(entry.link, config=user_config)
                        article.download()
                        article.parse()
                        texto_total_analise += article
