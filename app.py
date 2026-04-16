import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import feedparser

# 1. Configuração da Página
st.set_page_config(page_title="IA Investidor Pro", layout="centered")

st.title("🤖 Monitor IA: Análise Técnica + Fundamentalista")
st.caption("Dados em tempo real: B3, Google News e Indicadores de Valor")

# --- FUNÇÃO DE ANÁLISE MINUCIOSA (COM O SEU CÓDIGO INTEGRADO) ---
def analise_minuciosa_ia(ticker, preco, media):
    st.subheader(f"🕵️‍♂️ Análise Profunda IA: {ticker}")
    
    url_news = f"https://news.google.com/rss/search?q={ticker}+stock+analysis+quando:2d&hl=pt-BR&gl=BR&ceid=BR:pt-419"
    resumo_noticias = ""
    
    try:
        feed = feedparser.parse(url_news)
        with st.spinner(f'IA vasculhando portais financeiros para {ticker}...'):
            # SEU CÓDIGO INTEGRADO AQUI:
            for entry in feed.entries[:3]: 
                try:
                    # Plano A: Resumo do Google
                    resumo_noticias += entry.summary + " "
                    
                    # Plano B: Detalhes do site
                    headers = {'User-Agent': 'Mozilla/5.0'}
                    response = requests.get(entry.link, timeout=5, headers=headers)
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.text, 'html.parser')
                        paragrafos = soup.find_all('p')
                        texto_extra = " ".join([p.get_text() for p in paragrafos[:2]])
                        resumo_noticias += texto_extra + " "
                except:
                    continue
    except Exception as e:
        st.warning("Falha ao acessar feed de notícias.")

    st.write("---")
    
    # Motor de Decisão (Aprimorado)
    palavras_positivas = ["alta", "dividendo", "lucro", "compra", "crescimento", "ebitda", "subiu"]
    palavras_negativas = ["queda", "risco", "prejuízo", "venda", "caiu", "dívida", "crise"]

    otimismo = sum(resumo_noticias.lower().count(p) for p in palavras_positivas)
    pessimismo = sum(resumo_noticias.lower().count(p) for p in palavras_negativas)

    if preco > media and otimismo > pessimismo:
        st.success("**VEREDITO: COMPRA ✅**")
        st.write("Cenário de alta técnica confirmado por otimismo nas notícias.")
    elif preco < media and pessimismo > otimismo:
        st.error("**VEREDITO: EVITAR ❌**")
        st.write("Tendência de baixa com suporte de notícias negativas.")
    else:
        st.warning("**VEREDITO: NEUTRO / OBSERVAR ⚖️**")
        st.write("Falta de consenso entre gráfico e notícias.")

    with st.expander("🔍 Detalhes da Varredura"):
        st.write(resumo_noticias if resumo_noticias else "Nenhum texto adicional encontrado.")

# --- FUNÇÃO DE BUSCA DE DADOS E FUNDAMENTOS ---
def buscar_dados_completos(ticker):
    t = yf.Ticker(ticker)
    df = t.history(period="1y") # 1 ano de dados
    df['MA200'] = df['Close'].rolling(window=200).mean()
    
    # Pegando indicadores reais de mercado
    info = t.info
    fundamentos = {
        "dy": info.get('dividendYield', 0) * 100,
        "pl": info.get('trailingPE', 0),
        "nome": info.get('longName', ticker)
    }
    return df, fundamentos

# --- INTERFACE PRINCIPAL ---
tickers = ['PETR4.SA', 'VALE3.SA', 'ITUB4.SA']

for ticker in tickers:
    try:
        dados, fund = buscar_dados_completos(ticker)
        preco_atual = float(dados['Close'].iloc[-1])
        media_atual = float(dados['MA200'].iloc[-1])
        
        st.header(f"🏢 {fund['nome']}")
        
        # Exibindo Indicadores Reais (Dashboard)
        c1, c2, c3 = st.columns(3)
        c1.metric("Preço", f"R${preco_atual:.2f}")
        c2.metric("Div. Yield", f"{fund['dy']:.2f}%")
        c3.metric("P/L", f"{fund['pl']:.2f}")

        # Gráfico
        st.line_chart(dados[['Close', 'MA200']])
        
        # Análise de IA
        analise_minuciosa_ia(ticker, preco_atual, media_atual)
        st.divider()
        
    except Exception as e:
        st.error(f"Erro em {ticker}: {e}")

st.caption("Desenvolvido para estudos de Engenharia de IA - Blumenau, SC.")
