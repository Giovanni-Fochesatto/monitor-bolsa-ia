import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd

# Configuração da página para visualização mobile
st.set_page_config(page_title="Monitor IA - Bolsa", layout="centered")

st.title("🤖 Monitor de Tendências B3")
st.write("Análise de médias móveis para suporte à decisão.")

# Lista de ações para monitorar
tickers = ['PETR4.SA', 'VALE3.SA', 'ITUB4.SA']

def buscar_dados(ticker):
    # Buscamos desde 2024 para ter base para a média de 200 dias
    df = yf.download(ticker, start='2024-01-01')
    df['MA200'] = df['Close'].rolling(window=200).mean()
    return df

# Loop para criar a análise de cada empresa
for ticker in tickers:
    try:
        dados = buscar_dados(ticker)
        
        # Extraindo os valores mais recentes de forma segura
        preco_atual = float(dados['Close'].iloc[-1])
        media_atual = float(dados['MA200'].iloc[-1])
        
        st.header(f"📊 {ticker.split('.')[0]}")
        
        # Lógica de indicação de tendência
        if preco_atual > media_atual:
            st.success(f"TENDÊNCIA: ALTA 🚀")
            delta_cor = "normal"
        else:
            st.error(f"TENDÊNCIA: BAIXA ⚠️")
            delta_cor = "inverse"

        # Exibe o preço e a distância da média
        st.metric(
            label="Preço Atual", 
            value=f"R$ {preco_atual:.2f}", 
            delta=f"{preco_atual - media_atual:.2f} vs Média 200d",
            delta_color=delta_cor
        )

        # Criando o gráfico
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(dados['Close'], label='Preço Fechamento', color='#1f77b4')
        ax.plot(dados['MA200'], label='Média 200d', color='#ff7f0e', linestyle='--')
        ax.set_title(f"Histórico e Média - {ticker}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Mostra o gráfico no app
        st.pyplot(fig)
        st.divider()
        
    except Exception as e:
        st.warning(f"Não foi possível carregar dados de {ticker}. Aguarde o mercado abrir ou verifique a conexão.")

st.caption("Dados fornecidos via Yahoo Finance. Desenvolvido para estudos de Engenharia de IA.")
