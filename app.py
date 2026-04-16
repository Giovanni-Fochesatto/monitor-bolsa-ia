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

# Auto-refresh a cada 5 minutos
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

# --- FUNÇÃO DE ANÁLISE DE NOTÍCIAS E SENTIMENTO (VERSÃO FORÇA BRUTA) ---
def analise_minuciosa_ia(ticker, preco, media, rsi_atual):
    st.subheader(f"🕵️‍♂️ Inteligência de Mercado: {ticker}")
    
    # URL otimizada para maior estabilidade
    url_news = f"https://news.google.com/rss/search?q={ticker}+quando:2d&hl=pt-BR&gl=BR&ceid=BR:pt"
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
        st.warning("📡 Radar de notícias offline ou bloqueado temporariamente.")

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
            st.warning("**VEREDITO: NEUTRO / OBSERVAR ⚖️**")

    with col_info:
        st.write(f"**Análise Técnica:** Preço {'acima' if preco > media else 'abaixo'} da média de 200 dias.")
        st.write(f"**Indicador RSI:** {rsi_atual:.2f} ({status_rsi})")

    # --- RELATÓRIO DE INTELIGÊNCIA COM FORÇA BRUTA (EXIBIÇÃO) ---
    with st.expander("🔍 Relatório de Inteligência Detalhado", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### 🟢 Sinais Positivos")
            if detectadas_pos:
                st.write(", ".join(detectadas_pos))
            else:
                st.write("Nenhum termo otimista identificado.")
        
        with c2:
            st.markdown("### 🔴 Sinais de Risco")
            if detectadas_neg:
                st.write(", ".join(detectadas_neg))
            else:
                st.write("Nenhum termo de risco identificado.")
        
        st.markdown("---")
        st.markdown("### 📌 Manchetes Recentes Analisadas")
        
        # Lógica de Força Bruta para garantir que algo apareça
        if manchetes_encontradas:
            for m in manchetes_encontradas:
                st.write(f"✅ {m}")
        elif len(texto_total_analise) > 10:
            # Plano B: Se a lista falhou mas o texto existe, extrai do texto
            fallback = texto_total_analise.split('. ')
            for item in fallback[:5]:
                if len(item) > 10: st.write(f"👉 {item}")
        else:
            st.error("⚠️ Não foi possível recuperar manchetes. O Google News pode estar limitando as requisições.")
            
        if len(texto_total_analise) > 300:
            st.toast(f"IA: Análise profunda concluída para {ticker}")

# --- BUSCA DE DADOS FINANCEIROS ---
def buscar_dados_completos(ticker):
    t = yf.Ticker(ticker)
    df = t.history(period="1y")
    if df.empty: return None, None
    
    df['MA200'] = df['Close'].rolling(window=200).mean()
    df['RSI'] = calcular_rsi(df['Close'])
    
    try:
        info = t.info
        fundamentos = {
            "dy": info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0,
            "pl": info.get('trailingPE', 0) if info.get('trailingPE') else 0,
            "nome": info.get('longName', ticker),
            "volume": info.get('averageVolume', 0)
        }
    except:
        fundamentos = {"dy": 0, "pl": 0, "nome": ticker, "volume": 0}
        
    return df, fundamentos

# --- LOOP PRINCIPAL ---
tickers = ['PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'BBAS3.SA']

for ticker in tickers:
    try:
        dados, fund = buscar_dados_completos(ticker)
        if dados is None: continue
        
        preco_atual = float(dados['Close'].iloc[-1])
        media_atual = float(dados['MA200'].iloc[-1])
        rsi_atual = float(dados['RSI'].iloc[-1]) if not np.isnan(dados['RSI'].iloc[-1]) else 50
        
        st.divider()
        st.header(f"🏢 {fund['nome']} ({ticker})")
        
        # Dashboard de Métricas
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Preço Atual", f"R${preco_atual:.2f}")
        m2.metric("Dividend Yield", f"{fund['dy']:.2f}%")
        m3.metric("P/L", f"{fund['pl']:.2f}")
        m4.metric("Vol. Médio", f"{fund['volume']:,}")

        # Gráfico de Tendência
        st.line_chart(dados[['Close', 'MA200']])
        
        # Execução da IA
        analise_minuciosa_ia(ticker, preco_atual, media_atual, rsi_atual)
        
    except Exception as e:
        st.error(f"Erro ao processar {ticker}. Avançando para o próximo...")

st.caption("Engenharia de IA 2026 - Blumenau, SC. Dados em tempo real via YFinance & Google RSS.")
