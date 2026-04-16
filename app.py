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

# --- FUNÇÃO DE ANÁLISE DE NOTÍCIAS (BLINDADA CONTRA TELA PRETA) ---
def analise_minuciosa_ia(ticker, preco, media, rsi_atual):
    st.subheader(f"🕵️‍♂️ Inteligência de Mercado: {ticker}")
    
    # Inicialização obrigatória para evitar erros de renderização
    manchetes_encontradas = []
    texto_total_analise = ""
    detectadas_pos = []
    detectadas_neg = []
    otimismo = 0
    pessimismo = 0

    timestamp = int(time.time())
    url_news = f"https://news.google.com/rss/search?q={ticker}+when:2d&hl=pt-BR&gl=BR&ceid=BR:pt&t={timestamp}"
    
    user_config = Config()
    user_config.browser_user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'
    user_config.request_timeout = 15
    user_config.fetch_images = False 

    try:
        feed = feedparser.parse(url_news)
        if feed.entries:
            with st.spinner(f'IA analisando notícias de {ticker}...'):
                for entry in feed.entries[:5]: 
                    titulo = entry.title.split(' - ')[0]
                    manchetes_encontradas.append(titulo)
                    texto_total_analise += f"{titulo}. "
                    
                    try:
                        time.sleep(1) 
                        article = Article(entry.link, config=user_config)
                        article.download()
                        article.parse()
                        texto_total_analise += article.text[:300] + " "
                    except:
                        continue 
    except Exception as e:
        st.error(f"Erro no radar de notícias: {e}")

    # Motor de Sentimento
    positivas = ["alta", "dividendo", "lucro", "compra", "crescimento", "ebitda", "subiu", "positivo", "recorde", "valorização", "superou", "recompra"]
    negativas = ["queda", "risco", "prejuízo", "venda", "caiu", "dívida", "crise", "negativo", "abaixo", "desvalorização", "corte", "inflação"]

    texto_limpo = texto_total_analise.lower()
    detectadas_pos = list(set([p for p in positivas if p in texto_limpo]))
    detectadas_neg = list(set([p for p in negativas if p in texto_limpo]))
    
    otimismo = sum(texto_limpo.count(p) for p in positivas)
    pessimismo = sum(texto_limpo.count(p) for p in negativas)

    # Exibição do Veredito
    status_rsi = "Caro" if rsi_atual > 70 else "Barato" if rsi_atual < 30 else "Neutro"
    
    col_v, col_i = st.columns([1, 2])
    with col_v:
        if preco > media and otimismo > pessimismo:
            st.success("**VEREDITO: COMPRA ✅**")
        elif preco < media and pessimismo > otimismo:
            st.error("**VEREDITO: EVITAR/VENDA ❌**")
        else:
            st.warning("**VEREDITO: NEUTRO / OBSERVAR ⚖️**")

    with col_i:
        st.write(f"**Técnica:** {'Acima' if preco > media else 'Abaixo'} da média de 200 dias.")
        st.write(f"**Indicador RSI:** {rsi_atual:.2f} ({status_rsi})")

    with st.expander("🔍 Detalhes da Varredura de IA", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### 🟢 Sinais Positivos")
            st.write(", ".join(detectadas_pos) if detectadas_pos else "Nenhum sinal claro.")
        with c2:
            st.markdown("### 🔴 Sinais de Risco")
            st.write(", ".join(detectadas_neg) if detectadas_neg else "Nenhum risco detectado.")
        
        st.write("---")
        st.markdown("### 📌 Manchetes Processadas")
        if manchetes_encontradas:
            for m in manchetes_encontradas:
                st.write(f"✅ {m}")
        else:
            st.info("Aguardando novas manchetes ou acesso limitado pelo portal.")

# --- BUSCA DE DADOS FINANCEIROS (CORREÇÃO DA LINHA 131) ---
def buscar_dados_completos(ticker):
    try:
        t = yf.Ticker(ticker)
        df = t.history(period="1y")
        
        if df.empty: 
            return None, None
        
        # O cálculo deve estar fora de um try vazio ou devidamente fechado
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
    except Exception as e:
        st.error(f"Erro crítico no ticker {ticker}: {e}")
        return None, None

# --- LOOP PRINCIPAL ---
tickers = ['PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'BBAS3.SA']

for ticker in tickers:
    dados, fund = buscar_dados_completos(ticker)
    
    if dados is not None:
        try:
            preco_atual = float(dados['Close'].iloc[-1])
            media_val = dados['MA200'].iloc[-1]
            media_atual = float(media_val) if not np.isnan(media_val) else preco_atual
            
            rsi_val = dados['RSI'].iloc[-1]
            rsi_atual = float(rsi_val) if not np.isnan(rsi_val) else 50
            
            st.divider()
            st.header(f"🏢 {fund['nome']} ({ticker})")
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Preço Atual", f"R${preco_atual:.2f}")
            m2.metric("Dividend Yield", f"{fund['dy']:.2f}%")
            m3.metric("P/L", f"{fund['pl']:.2f}")
            m4.metric("Vol. Médio", f"{fund['volume']:,}")

            st.line_chart(dados[['Close', 'MA200']])
            analise_minuciosa_ia(ticker, preco_atual, media_atual, rsi_atual)
            
        except Exception as e:
            st.error(f"Erro ao processar dados visuais: {e}")

st.caption("Engenharia de IA 2026 - Blumenau, SC.")
