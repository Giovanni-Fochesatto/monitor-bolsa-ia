import streamlit as st
import yfinance as yf
import pandas as pd
import time
import feedparser
import nltk
from newspaper import Article, Config
from streamlit_autorefresh import st_autorefresh

# --- CONFIGURAÇÕES INICIAIS ---
# Necessário para o processamento de texto do Newspaper3k
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

st.set_page_config(page_title="IA Investidor Pro", layout="centered")

# Auto-refresh a cada 5 minutos
st_autorefresh(interval=300 * 1000, key="data_refresh")

st.title("🤖 Monitor IA: Automação B3")
st.caption(f"Última atualização: {time.strftime('%H:%M:%S')} (Atualiza sozinho a cada 5 min)")

# --- FUNÇÃO DE ANÁLISE DE NOTÍCIAS ---
def analise_minuciosa_ia(ticker, preco, media):
    st.subheader(f"🕵️‍♂️ Análise Profunda IA: {ticker}")
    
    url_news = f"https://news.google.com/rss/search?q={ticker}+quando:2d&hl=pt-BR&gl=BR&ceid=BR:pt-419"
    resumo_noticias = ""
    
    # Configuração de Navegador Real para evitar bloqueios
    user_config = Config()
    user_config.browser_user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    user_config.request_timeout = 10

    try:
        feed = feedparser.parse(url_news)
        with st.spinner(f'IA vasculhando mercado para {ticker}...'):
            for entry in feed.entries[:3]: 
                # Passo 1: Captura imediata do título (Garante dado mínimo)
                resumo_noticias += f"{entry.title}. "
                
                try:
                    # Passo 2: Extração profunda com User-Agent real
                    article = Article(entry.link, config=user_config)
                    article.download()
                    article.parse()
                    resumo_noticias += article.text[:500] + " "
                except:
                    continue
    except:
        st.warning("Radar de notícias temporariamente indisponível.")

    st.write("---")
    
    # Motor de Decisão Aprimorado
    positivas = ["alta", "dividendo", "lucro", "compra", "crescimento", "ebitda", "subiu", "positivo", "recorde", "manter", "valorização", "superou"]
    negativas = ["queda", "risco", "prejuízo", "venda", "caiu", "dívida", "crise", "negativo", "abaixo", "desvalorização", "corte"]

    texto_analise = resumo_noticias.lower()
    otimismo = sum(texto_analise.count(p) for p in positivas)
    pessimismo = sum(texto_analise.count(p) for p in negativas)

    # --- LÓGICA DE VEREDITO HÍBRIDA (TÉCNICA + SENTIMENTO) ---
    # Se o texto for curto (só títulos), o limite de decisão é menor
    tem_noticias_claras = abs(otimismo - pessimismo) > 0

    if preco > media:
        if otimismo > pessimismo:
            st.success("**VEREDITO: COMPRA ✅**")
            st.info("Consenso Forte: Gráfico em alta e notícias positivas.")
        elif tem_noticias_claras and pessimismo > otimismo:
            st.warning("**VEREDITO: OBSERVAR ⚖️**")
            st.write("Divergência: Gráfico em alta, mas notícias recentes são negativas.")
        else:
            st.success("**VEREDITO: ALTA TÉCNICA 📈**")
            st.write("Tendência de alta pelo gráfico. Notícias neutras ou protegidas.")
            
    elif preco < media:
        if pessimismo > otimismo:
            st.error("**VEREDITO: EVITAR/VENDA ❌**")
            st.info("Risco Máximo: Gráfico em queda e notícias negativas.")
        elif tem_noticias_claras and otimismo > pessimismo:
            st.warning("**VEREDITO: OBSERVAR ⚖️**")
            st.write("Recuperação? Gráfico ainda em queda, mas surgem notícias positivas.")
        else:
            st.error("**VEREDITO: BAIXA TÉCNICA 📉**")
            st.write("Tendência de baixa pelo gráfico. Sem fatos novos nas notícias.")

    with st.expander("🔍 Relatório de Inteligência"):
        if len(resumo_noticias.strip()) > 20:
            st.write(resumo_noticias)
        else:
            st.write("Conteúdo detalhado protegido. Análise baseada em dados técnicos e manchetes.")

# --- BUSCA DE DADOS ---
def buscar_dados_completos(ticker):
    t = yf.Ticker(ticker)
    df = t.history(period="1y")
    if df.empty: return None, None
    
    # Média Móvel de 200 dias
    df['MA200'] = df['Close'].rolling(window=200).mean()
    
    info = t.info
    fundamentos = {
        "dy": info.get('dividendYield', 0) * 100,
        "pl": info.get('trailingPE', 0),
        "nome": info.get('longName', ticker)
    }
    return df, fundamentos

# --- LOOP PRINCIPAL ---
tickers = ['PETR4.SA', 'VALE3.SA', 'ITUB4.SA']

for ticker in tickers:
    try:
        dados, fund = buscar_dados_completos(ticker)
        if dados is None: continue
        
        preco_atual = float(dados['Close'].iloc[-1])
        media_atual = float(dados['MA200'].iloc[-1])
        
        st.header(f"🏢 {fund['nome']}")
        
        # Dashboard
        col1, col2, col3 = st.columns(3)
        col1.metric("Preço", f"R${preco_atual:.2f}")
        col2.metric("Div. Yield", f"{fund['dy']:.2f}%")
        col3.metric("P/L", f"{fund['pl']:.2f}")

        # Gráfico
        st.line_chart(dados[['Close', 'MA200']])
        
        # Análise de IA
        analise_minuciosa_ia(ticker, preco_atual, media_atual)
        st.divider()
        
    except Exception as e:
        st.error(f"Erro ao analisar {ticker}. Tentando reconectar...")

st.caption("Engenharia de IA - Blumenau, SC. Dados em tempo real.")
