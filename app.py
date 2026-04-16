import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import requests
import time
import feedparser
from bs4 import BeautifulSoup
from newspaper import Article
from streamlit_autorefresh import st_autorefresh

# 1. Configuração da Página
st.set_page_config(page_title="IA Investidor Pro", layout="centered")

# AUTO-REFRESH: Atualiza o app sozinho a cada 5 minutos (300 segundos)
st_autorefresh(interval=300 * 1000, key="data_refresh")

st.title("🤖 Monitor IA: Automação B3")
st.caption(f"Última atualização: {time.strftime('%H:%M:%S')} (Atualiza sozinho a cada 5 min)")

# --- FUNÇÃO DE ANÁLISE MINUCIOSA (VERSÃO DEFINITIVA COM NEWSPAPER3K) ---
def analise_minuciosa_ia(ticker, preco, media):
    st.subheader(f"🕵️‍♂️ Análise Profunda IA: {ticker}")
    
    url_news = f"https://news.google.com/rss/search?q={ticker}+quando:2d&hl=pt-BR&gl=BR&ceid=BR:pt-419"
    resumo_noticias = ""
    
    try:
        feed = feedparser.parse(url_news)
        with st.spinner(f'IA utilizando Newspaper3k para analisar {ticker}...'):
            for entry in feed.entries[:3]: 
                # Passo 1: Captura imediata do título (Garante dados para o motor de decisão)
                resumo_noticias += f"{entry.title}. "
                
                try:
                    # Passo 2: Extração inteligente do corpo da notícia
                    article = Article(entry.link)
                    article.download()
                    article.parse()
                    # Limitamos o texto para não estourar o contexto da análise
                    resumo_noticias += article.text[:400] + " "
                except:
                    # Se falhar a extração profunda, mantemos o título e seguimos
                    continue
    except:
        st.warning("Erro ao acessar o radar de notícias.")

    st.write("---")
    
    # Motor de Decisão Aprimorado
    positivas = ["alta", "dividendo", "lucro", "compra", "crescimento", "ebitda", "subiu", "positivo", "recorde", "manter", "valorização"]
    negativas = ["queda", "risco", "prejuízo", "venda", "caiu", "dívida", "crise", "negativo", "abaixo", "desvalorização"]

    texto_analise = resumo_noticias.lower()
    otimismo = sum(texto_analise.count(p) for p in positivas)
    pessimismo = sum(texto_analise.count(p) for p in negativas)

    # Lógica de Veredito (Técnica + Sentimento)
    if preco > media and otimismo > pessimismo:
        st.success("**VEREDITO: COMPRA ✅**")
        st.info("Tendência de alta confirmada: Preço acima da média e notícias otimistas.")
    elif preco < media and pessimismo > otimismo:
        st.error("**VEREDITO: EVITAR/VENDA ❌**")
        st.info("Sinal de alerta: Preço abaixo da média e sentimento de mercado negativo.")
    else:
        st.warning("**VEREDITO: NEUTRO / OBSERVAR ⚖️**")
        st.info("Sinais mistos ou dados insuficientes para uma recomendação clara.")

    with st.expander("🔍 Relatório de Varredura (Conteúdo Extraído)"):
        if len(resumo_noticias.strip()) > 10:
            st.write(resumo_noticias)
        else:
            st.write("A IA baseou a decisão nos títulos recentes, pois o conteúdo denso estava protegido.")

# --- BUSCA DE DADOS FINANCEIROS ---
def buscar_dados_completos(ticker):
    t = yf.Ticker(ticker)
    df = t.history(period="1y")
    # Média Móvel de 200 dias (MA200)
    df['MA200'] = df['Close'].rolling(window=200).mean()
    
    info = t.info
    fundamentos = {
        "dy": info.get('dividendYield', 0) * 100,
        "pl": info.get('trailingPE', 0),
        "nome": info.get('longName', ticker)
    }
    return df, fundamentos

# --- INTERFACE E LOOP PRINCIPAL ---
tickers = ['PETR4.SA', 'VALE3.SA', 'ITUB4.SA']

for ticker in tickers:
    try:
        dados, fund = buscar_dados_completos(ticker)
        preco_atual = float(dados['Close'].iloc[-1])
        media_atual = float(dados['MA200'].iloc[-1])
        
        st.header(f"🏢 {fund['nome']}")
        
        # Dashboard de Indicadores Rápidos
        c1, c2, c3 = st.columns(3)
        c1.metric("Preço", f"R${preco_atual:.2f}")
        c2.metric("Div. Yield", f"{fund['dy']:.2f}%")
        c3.metric("P/L", f"{fund['pl']:.2f}")

        # Gráfico de Preço vs Média Móvel
        st.line_chart(dados[['Close', 'MA200']])
        
        # Execução da Análise de IA
        analise_minuciosa_ia(ticker, preco_atual, media_atual)
        st.divider()
        
    except Exception as e:
        st.error(f"Erro ao processar {ticker}. Verifique a conexão ou o código do ativo.")

st.caption("Desenvolvido para Engenharia de IA - 2026. Monitoramento em tempo real ativado.")
