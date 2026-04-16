import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import feedparser
from streamlit_autorefresh import st_autorefresh

# 1. Configuração da Página
st.set_page_config(page_title="IA Investidor Pro", layout="centered")

# AUTO-REFRESH: Atualiza o app sozinho a cada 5 minutos (300 segundos)
st_autorefresh(interval=300 * 1000, key="data_refresh")

st.title("🤖 Monitor IA: Automação B3")
st.caption(f"Última atualização: {time.strftime('%H:%M:%S')} (Atualiza sozinho a cada 5 min)")

# --- FUNÇÃO DE ANÁLISE MINUCIOSA (VERSÃO INTEGRADA E SEGURA) ---
def analise_minuciosa_ia(ticker, preco, media):
    st.subheader(f"🕵️‍♂️ Análise Profunda IA: {ticker}")
    
    # URL do Feed RSS do Google News
    url_news = f"https://news.google.com/rss/search?q={ticker}+quando:2d&hl=pt-BR&gl=BR&ceid=BR:pt-419"
    resumo_noticias = ""
    
    try:
        feed = feedparser.parse(url_news)
        with st.spinner(f'IA vasculhando portais para {ticker}...'):
            for entry in feed.entries[:3]: 
                # ESTRATÉGIA DE SEGURANÇA: Captura título + resumo antes de tentar acesso externo
                informacao_base = f"{entry.title}. {getattr(entry, 'summary', '')} "
                resumo_noticias += informacao_base
                
                try:
                    # PLANO B: Tentativa de busca profunda no site original
                    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
                    response = requests.get(entry.link, timeout=5, headers=headers)
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.text, 'html.parser')
                        # Extrai os primeiros parágrafos
                        paragrafos = soup.find_all('p')
                        texto_extra = " ".join([p.get_text() for p in paragrafos[:2]])
                        resumo_noticias += texto_extra + " "
                except:
                    # Se o site bloquear, continuamos com o que já temos do RSS
                    continue
    except Exception as e:
        st.warning("Falha ao acessar feed de notícias.")

    st.write("---")
    
    # Motor de Decisão (Sensibilidade Aumentada)
    palavras_positivas = ["alta", "dividendo", "lucro", "compra", "crescimento", "ebitda", "subiu", "valorização", "positivo", "recorde"]
    palavras_negativas = ["queda", "risco", "prejuízo", "venda", "caiu", "dívida", "crise", "desvalorização", "negativo"]

    texto_para_analise = resumo_noticias.lower()
    otimismo = sum(texto_para_analise.count(p) for p in palavras_positivas)
    pessimismo = sum(texto_para_analise.count(p) for p in palavras_negativas)

    # Veredito Final
    if preco > media and otimismo > pessimismo:
        st.success("**VEREDITO: COMPRA ✅**")
        st.write("Sinais técnicos (preço > média) e notícias otimistas em consenso.")
    elif preco < media and pessimismo > otimismo:
        st.error("**VEREDITO: EVITAR ❌**")
        st.write("Tendência técnica de baixa confirmada por sentimento negativo nas notícias.")
    else:
        st.warning("**VEREDITO: NEUTRO / OBSERVAR ⚖️**")
        st.write("Falta de clareza: Gráfico e notícias em direções opostas ou volume de dados insuficiente.")

    with st.expander("🔍 Detalhes da Varredura (Fontes Analisadas)"):
        if len(resumo_noticias.strip()) > 15:
            # Limpa possíveis tags HTML remanescentes do RSS
            texto_exibicao = BeautifulSoup(resumo_noticias, "html.parser").get_text()
            st.write(texto_exibicao[:1500] + "...")
        else:
            st.write("A IA processou os títulos, mas os portais restringiram a leitura detalhada. Prossiga com cautela.")

# --- BUSCA DE DADOS ---
def buscar_dados_completos(ticker):
    t = yf.Ticker(ticker)
    df = t.history(period="1y")
    # Média Móvel de 200 dias
    df['MA200'] = df['Close'].rolling(window=200).mean()
    
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
        # Valores atuais para a lógica do veredito
        preco_atual = float(dados['Close'].iloc[-1])
        media_atual = float(dados['MA200'].iloc[-1])
        
        st.header(f"🏢 {fund['nome']}")
        
        # Dashboard de Indicadores
        c1, c2, c3 = st.columns(3)
        c1.metric("Preço Atual", f"R${preco_atual:.2f}")
        c2.metric("Div. Yield", f"{fund['dy']:.2f}%")
        c3.metric("P/L (Valuation)", f"{fund['pl']:.2f}")

        # Gráfico Interativo
        st.line_chart(dados[['Close', 'MA200']])
        
        # Execução da Análise Profunda
        analise_minuciosa_ia(ticker, preco_atual, media_atual)
        st.divider()
        
    except Exception as e:
        st.error(f"Erro ao processar dados de {ticker}. Verifique o ticker ou a conexão.")

st.caption("Desenvolvido para estudos de Engenharia de IA. Os dados são processados a cada 5 minutos.")
