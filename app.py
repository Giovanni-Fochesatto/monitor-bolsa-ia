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

st.set_page_config(page_title="IA Investidor Pro", layout="wide") # Layout wide para melhor visualização

# Auto-refresh a cada 5 minutos
st_autorefresh(interval=300 * 1000, key="data_refresh")

st.title("🤖 Monitor IA: Automação B3 Pro")
st.caption(f"Última atualização: {time.strftime('%H:%M:%S')} (Blumenau, SC)")

# --- FUNÇÕES TÉCNICAS ADICIONAIS ---
def calcular_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# --- FUNÇÃO DE ANÁLISE DE NOTÍCIAS E SENTIMENTO ---
def analise_minuciosa_ia(ticker, preco, media, rsi_atual):
    st.subheader(f"🕵️‍♂️ Inteligência de Mercado: {ticker}")
    
    url_news = f"https://news.google.com/rss/search?q={ticker}+quando:2d&hl=pt-BR&gl=BR&ceid=BR:pt-419"
    manchetes_encontradas = []
    texto_total_analise = ""
    
    user_config = Config()
    user_config.browser_user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    user_config.request_timeout = 10

    try:
        feed = feedparser.parse(url_news)
        with st.spinner(f'IA processando notícias de {ticker}...'):
            for entry in feed.entries[:5]: 
                manchetes_encontradas.append(entry.title)
                texto_total_analise += f"{entry.title}. "
                try:
                    article = Article(entry.link, config=user_config)
                    article.download()
                    article.parse()
                    texto_total_analise += article.text[:500] + " "
                except:
                    continue
    except:
        st.warning("Fluxo de notícias offline.")

    # Motor de Decisão
    positivas = ["alta", "dividendo", "lucro", "compra", "crescimento", "ebitda", "subiu", "positivo", "recorde", "manter", "valorização", "superou", "recompra"]
    negativas = ["queda", "risco", "prejuízo", "venda", "caiu", "dívida", "crise", "negativo", "abaixo", "desvalorização", "corte", "inflação"]

    texto_limpo = texto_total_analise.lower()
    detectadas_pos = list(set([p for p in positivas if p in texto_limpo]))
    detectadas_neg = list(set([p for p in negativas if p in texto_limpo]))
    
    otimismo = sum(texto_limpo.count(p) for p in positivas)
    pessimismo = sum(texto_limpo.count(p) for p in negativas)

    # --- LÓGICA DE VEREDITO HÍBRIDA ---
    status_rsi = "Sobrecomprado (Caro)" if rsi_atual > 70 else "Sobrevendido (Barato)" if rsi_atual < 30 else "Neutro"
    
    col_veredito, col_info = st.columns([1, 2])

    with col_veredito:
        if preco > media and otimismo > pessimismo:
            st.success("**VEREDITO: COMPRA ✅**")
        elif preco < media and pessimismo > otimismo:
            st.error("**VEREDITO: EVITAR/VENDA ❌**")
        else:
            st.warning("**VEREDITO: NEUTRO ⚖️**")

    with col_info:
        st.write(f"**Análise Técnica:** Preço {'acima' if preco > media else 'abaixo'} da média de 200 dias.")
        st.write(f"**Indicador RSI:** {rsi_atual:.2f} ({status_rsi})")

    # --- RELATÓRIO DE INTELIGÊNCIA ---
    with st.expander("🔍 Relatório de Inteligência Detalhado"):
        c1, c2 = st.columns(2)
        c1.markdown("### 🟢 Sinais Positivos")
        c1.write(", ".join(detectadas_pos) if detectadas_pos else "Nenhum sinal claro")
        
        c2.markdown("### 🔴 Sinais de Risco")
        c2.write(", ".join(detectadas_neg) if detectadas_neg else "Nenhum risco detectado")
        
        st.markdown("---")
        st.markdown("**📌 Manchetes Recentes Analisadas:**")
        for m in manchetes_encontradas:
            st.write(f"• {m}")

# --- BUSCA DE DADOS ---
def buscar_dados_completos(ticker):
    t = yf.Ticker(ticker)
    df = t.history(period="1y")
    if df.empty: return None, None
    
    df['MA200'] = df['Close'].rolling(window=200).mean()
    df['RSI'] = calcular_rsi(df['Close'])
    
    info = t.info
    fundamentos = {
        "dy": info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0,
        "pl": info.get('trailingPE', 0) if info.get('trailingPE') else 0,
        "nome": info.get('longName', ticker),
        "volume": info.get('averageVolume', 0)
    }
    return df, fundamentos

# --- LOOP PRINCIPAL ---
tickers = ['PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'BBAS3.SA'] # Adicionado Banco do Brasil para diversificar

for ticker in tickers:
    try:
        dados, fund = buscar_dados_completos(ticker)
        if dados is None: continue
        
        preco_atual = float(dados['Close'].iloc[-1])
        media_atual = float(dados['MA200'].iloc[-1])
        rsi_atual = float(dados['RSI'].iloc[-1])
        
        st.divider()
        st.header(f"🏢 {fund['nome']} ({ticker})")
        
        # Dashboard de Métricas
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Preço Atual", f"R${preco_atual:.2f}")
        m2.metric("Dividend Yield", f"{fund['dy']:.2f}%")
        m3.metric("P/L", f"{fund['pl']:.2f}")
        m4.metric("Vol. Médio", f"{fund['volume']:,}")

        # Gráfico
        st.line_chart(dados[['Close', 'MA200']])
        
        # IA e Sentimento
        analise_minuciosa_ia(ticker, preco_atual, media_atual, rsi_atual)
        
    except Exception as e:
        st.error(f"Erro ao analisar {ticker}. O sistema tentará novamente em 5 min.")

st.caption("Sistema de Monitoramento Autônomo - Engenharia de IA 2026")
