import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import requests
from bs4 import BeautifulSoup
from googlesearch import search

# 1. Configuração da Página para Celular
st.set_page_config(page_title="IA Investidor Pro", layout="centered")

st.title("🤖 Monitor IA: Análise Técnica + Fundamentalista")
st.write("Varredura minuciosa de dados e notícias em tempo real.")

# --- FUNÇÃO DE ANÁLISE MINUCIOSA (A que você enviou) ---
def analise_minuciosa_ia(ticker, preco, media):
    st.subheader(f"🕵️‍♂️ Análise Profunda IA: {ticker}")
    
    query = f"análise completa recomendação {ticker} 2026 dividendos lucro"
    links = []
    try:
        # Busca os 3 melhores resultados recentes
        for j in search(query, num=3, stop=3, lang='pt'):
            links.append(j)
    except:
        st.warning("IA em modo offline: Falha na busca em tempo real.")
        return

    resumo_noticias = ""
    for link in links:
        try:
            response = requests.get(link, timeout=5, headers={'User-Agent': 'Mozilla/5.0'})
            soup = BeautifulSoup(response.text, 'html.parser')
            paragrafos = soup.find_all('p')
            texto = " ".join([p.get_text() for p in paragrafos[:3]])
            resumo_noticias += texto + " "
        except:
            continue

    st.write("---")
    
    # Lógica de análise de palavras-chave
    otimismo = resumo_noticias.lower().count("alta") + resumo_noticias.lower().count("dividendo") + resumo_noticias.lower().count("lucro")
    pessimismo = resumo_noticias.lower().count("queda") + resumo_noticias.lower().count("risco") + resumo_noticias.lower().count("prejuízo")

    if preco > media and otimismo > pessimismo:
        st.success("**VEREDITO: COMPRA MINUCIOSA ✅**")
        st.write("Gráfico em alta + Notícias positivas (Lucros/Dividendos).")
    elif preco < media and pessimismo > otimismo:
        st.error("**VEREDITO: VENDA / EVITAR ❌**")
        st.write("Gráfico em queda + Notícias negativas (Riscos/Quedas).")
    else:
        st.warning("**VEREDITO: NEUTRO / OBSERVAR ⚖️**")
        st.write("Sinais mistos. Mercado indeciso ou notícias divergentes.")

    with st.expander("Ver detalhes da varredura (O que a IA leu)"):
        st.write(resumo_noticias[:1000] + "...")

# --- FUNÇÃO DE BUSCA DE DADOS ---
def buscar_dados(ticker):
    ticker_obj = yf.Ticker(ticker)
    df = ticker_obj.history(start='2024-01-01')
    if df.empty:
        df = yf.download(ticker, start='2024-01-01', progress=False)
    df['MA200'] = df['Close'].rolling(window=200).mean()
    return df

# --- LOOP PRINCIPAL DO APP ---
tickers = ['PETR4.SA', 'VALE3.SA', 'ITUB4.SA']

for ticker in tickers:
    try:
        dados = buscar_dados(ticker)
        preco_atual = float(dados['Close'].iloc[-1])
        media_atual = float(dados['MA200'].iloc[-1])
        
        # Parte Visual (Gráfico)
        st.header(f"📊 {ticker}")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(dados['Close'], label='Preço', color='#1f77b4')
        ax.plot(dados['MA200'], label='Média 200d', color='#ff7f0e', linestyle='--')
        ax.legend()
        st.pyplot(fig)
        
        # Chamada da sua Função Minuciosa
        analise_minuciosa_ia(ticker, preco_atual, media_atual)
        st.divider()
        
    except Exception as e:
        st.error(f"Erro ao processar {ticker}")

st.caption("Desenvolvido para estudos de Engenharia de IA - 2026")
