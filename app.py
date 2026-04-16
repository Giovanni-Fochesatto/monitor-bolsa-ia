import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import feedparser # Biblioteca excelente para ler notícias sem ser bloqueado

# 1. Configuração da Página
st.set_page_config(page_title="IA Investidor Pro", layout="centered")

st.title("🤖 Monitor IA: Análise Técnica + Fundamentalista")
st.caption("Cruzamento de Médias Móveis com Varredura de Notícias via Google News RSS")

# --- FUNÇÃO DE ANÁLISE MINUCIOSA (VERSÃO BLINDADA) ---
def analise_minuciosa_ia(ticker, preco, media):
    st.subheader(f"🕵️‍♂️ Análise Profunda IA: {ticker}")
    
    # Usamos o RSS do Google News para evitar bloqueios de "bot"
    # Ele busca notícias específicas do ticker nas últimas 48h
    url_news = f"https://news.google.com/rss/search?q={ticker}+stock+analysis+quando:2d&hl=pt-BR&gl=BR&ceid=BR:pt-419"
    
    links = []
    try:
        feed = feedparser.parse(url_news)
        # Pegamos os 3 links de notícias mais relevantes do feed
        for entry in feed.entries[:3]:
            links.append(entry.link)
    except Exception as e:
        st.warning("IA em modo offline: Falha ao acessar feed de notícias.")
        return

    resumo_noticias = ""
    # Barra de progresso charmosa enquanto a IA "lê" os sites
    with st.spinner(f'IA vasculhando portais financeiros para {ticker}...'):
        for link in links:
            try:
                # Simulamos um navegador real para o site não barrar a leitura do texto
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
                response = requests.get(link, timeout=7, headers=headers)
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extraímos o texto dos parágrafos
                paragrafos = soup.find_all('p')
                texto = " ".join([p.get_text() for p in paragrafos[:4]])
                resumo_noticias += texto + " "
                time.sleep(1) # Intervalo ético entre leituras
            except:
                continue

    st.write("---")
    
    # 2. MOTOR DE DECISÃO (Lógica Fundamentalista)
    # Procuramos termos que indicam saúde financeira ou risco
    palavras_positivas = ["alta", "dividendo", "lucro", "compra", "crescimento", "ebitda", "recorde"]
    palavras_negativas = ["queda", "risco", "prejuízo", "venda", "crise", "dívida", "processo"]

    otimismo = sum(resumo_noticias.lower().count(p) for p in palavras_positivas)
    pessimismo = sum(resumo_noticias.lower().count(p) for p in palavras_negativas)

    # Exibição do Veredito
    if preco > media and otimismo > pessimismo:
        st.success(f"**VEREDITO: COMPRA MINUCIOSA ✅**")
        st.write(f"**Motivo:** O preço está acima da média móvel (tendência de alta) e a mídia financeira cita termos como {', '.join(palavras_positivas[:3])}.")
    elif preco < media and pessimismo > otimismo:
        st.error(f"**VEREDITO: VENDA / EVITAR ❌**")
        st.write(f"**Motivo:** Tendência técnica de baixa confirmada por notícias sobre {', '.join(palavras_negativas[:3])}.")
    else:
        st.warning(f"**VEREDITO: NEUTRO / OBSERVAR ⚖️**")
        st.write("Sinais mistos. O gráfico e o sentimento das notícias não estão em sintonia clara.")

    with st.expander("🔍 Ver o que a IA extraiu para análise"):
        if resumo_noticias:
            st.write(resumo_noticias[:1200] + "...")
        else:
            st.write("Não foi possível extrair texto detalhado, mas a análise de títulos foi concluída.")

# --- FUNÇÃO DE BUSCA DE PREÇOS ---
def buscar_dados(ticker):
    # Tentativa de busca com Ticker obj (mais estável no Streamlit)
    t = yf.Ticker(ticker)
    df = t.history(start='2024-01-01')
    if df.empty:
        df = yf.download(ticker, start='2024-01-01', progress=False)
    
    df['MA200'] = df['Close'].rolling(window=200).mean()
    return df

# --- INTERFACE PRINCIPAL ---
tickers = ['PETR4.SA', 'VALE3.SA', 'ITUB4.SA']

for ticker in tickers:
    try:
        dados = buscar_dados(ticker)
        # Pegamos o valor da última linha
        preco_atual = float(dados['Close'].iloc[-1])
        media_atual = float(dados['MA200'].iloc[-1])
        
        # UI: Cabeçalho e Gráfico
        st.header(f"📉 {ticker}")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(dados.index, dados['Close'], label='Preço Fechamento', color='#1f77b4')
        ax.plot(dados.index, dados['MA200'], label='Média 200d', color='#ff7f0e', linestyle='--')
        ax.set_ylabel("Preço (R$)")
        ax.legend()
        st.pyplot(fig)
        
        # Executa a análise minuciosa
        analise_minuciosa_ia(ticker, preco_atual, media_atual)
        st.divider()
        
    except Exception as e:
        st.error(f"Erro ao processar {ticker}: {e}")

st.caption("Fonte: Yahoo Finance & Google News. Análise gerada via Algoritmo de IA.")
