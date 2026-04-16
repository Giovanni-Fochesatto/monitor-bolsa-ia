import streamlit as st
import yfinance as yf
import pandas as pd
import time
import feedparser
import nltk
import numpy as np
import random
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
st.caption(f"Última atualização: {time.strftime('%H:%M:%S')} (Fuso: Blumenau/SC)")

# --- SISTEMA DE PRIVACIDADE (HEADER ROTATION) ---
def get_random_header():
    agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0'
    ]
    return random.choice(agents)

# --- FUNÇÕES TÉCNICAS ---
def calcular_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# --- ANÁLISE FUNDAMENTALISTA (ESTILO INVESTIDOR 10) ---
def analise_fundamentalista_i10(ticker, info):
    st.subheader(f"📊 Indicadores Fundamentalistas (Estilo Investidor 10)")
    try:
        # Extração de métricas
        pl = info.get('trailingPE', 0)
        pvp = info.get('priceToBook', 0)
        roe = info.get('returnOnEquity', 0) * 100
        dy = info.get('dividendYield', 0) * 100
        margem = info.get('profitMargins', 0) * 100
        ev_ebitda = info.get('enterpriseToEbitda', 0)
        
        # Cálculo Dívida Líquida / EBITDA
        # Algumas empresas podem não ter o dado de dívida disponível via API gratuita
        divida_liquida = info.get('totalDebt', 0) - info.get('totalCash', 0)
        ebitda_valor = info.get('ebitda', 0)
        div_ebitda = divida_liquida / ebitda_valor if ebitda_valor and ebitda_valor != 0 else 0

        # Layout de Cards de Indicadores (7 colunas agora)
        c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
        c1.metric("P/L", f"{pl:.2f}" if pl else "---")
        c2.metric("P/VP", f"{pvp:.2f}" if pvp else "---")
        c3.metric("DY", f"{dy:.2f}%")
        c4.metric("ROE", f"{roe:.1f}%")
        c5.metric("Marg. Líq.", f"{margem:.1f}%")
        c6.metric("EV/EBITDA", f"{ev_ebitda:.2f}" if ev_ebitda else "---")
        c7.metric("Dív.Líq/EBITDA", f"{div_ebitda:.2f}" if div_ebitda != 0 else "---")

        with st.expander("🎓 O que significam esses números? (Legenda)"):
            st.markdown("""
            * **P/L:** Preço sobre Lucro. Indica em quantos anos você recuperaria o investimento através dos lucros.
            * **P/VP:** Preço sobre Valor Patrimonial. Abaixo de 1,00 pode indicar que a ação está barata.
            * **DY (Dividend Yield):** Rendimento em dividendos pagos nos últimos 12 meses.
            * **ROE:** Retorno sobre Patrimônio. Mede a eficiência em gerar lucro com o capital dos sócios.
            * **Margem Líquida:** Porcentagem de lucro real sobre a receita total.
            * **EV/EBITDA:** Valor da empresa (com dívidas) dividido pelo lucro operacional.
            * **Dívida Líquida / EBITDA:** Indica quanto tempo a empresa levaria para pagar sua dívida usando apenas sua geração de caixa operacional. 
                * *Valores abaixo de 2.0 são considerados saudáveis. Acima de 3.0 acende um sinal de alerta sobre o endividamento.*
            """)
    except Exception:
        st.warning("Alguns indicadores fundamentalistas estão indisponíveis para este ativo.")

# --- FUNÇÃO DE IA E SENTIMENTO ---
def analise_minuciosa_ia(ticker, preco, media, rsi_atual):
    st.subheader(f"🕵️‍♂️ Inteligência de Mercado: {ticker}")
    manchetes_encontradas = []
    texto_total_analise = ""
    
    user_config = Config()
    user_config.browser_user_agent = get_random_header()
    
    try:
        url_news = f"https://news.google.com/rss/search?q={ticker}+when:2d&hl=pt-BR&gl=BR&ceid=BR:pt-419"
        feed = feedparser.parse(url_news)
        
        if feed.entries:
            for entry in feed.entries[:6]: 
                titulo = entry.title.split(' - ')[0]
                manchetes_encontradas.append(titulo)
                try:
                    time.sleep(random.uniform(0.5, 1.2))
                    article = Article(entry.link, config=user_config)
                    article.download()
                    article.parse()
                    texto_total_analise += f"{titulo}. {article.text[:300]} "
                except:
                    texto_total_analise += f"{titulo}. "
        else:
            st.info("Nenhuma notícia recente encontrada.")
    except Exception:
        st.error("Erro no sistema de notícias.")

    positivas = ["alta", "dividendo", "lucro", "compra", "crescimento", "subiu", "positivo", "recorde", "recompra"]
    negativas = ["queda", "risco", "prejuízo", "venda", "caiu", "dívida", "crise", "negativo", "corte"]
    
    texto_limpo = texto_total_analise.lower()
    otimismo = sum(texto_limpo.count(p) for p in positivas)
    pessimismo = sum(texto_limpo.count(n) for n in negativas)

    col_v, col_i = st.columns([1, 2])
    with col_v:
        if preco > media and otimismo > pessimismo:
            st.success("**VEREDITO: COMPRA ✅**")
        elif preco < media and pessimismo > otimismo:
            st.error("**VEREDITO: EVITAR ❌**")
        else:
            st.warning("**VEREDITO: NEUTRO ⚖️**")

    with col_i:
        st.write(f"📊 Técnica: {'Acima' if preco > media else 'Abaixo'} da MA200 | RSI: {rsi_atual:.2f}")
        st.write(f"🧠 Sentimento: {otimismo} Positivos | {pessimismo} Negativos")

    with st.expander("📌 Ver Manchetes Analisadas"):
        for m in manchetes_encontradas:
            st.write(f"• {m}")

# --- BUSCA DE DADOS ---
def buscar_dados(ticker):
    try:
        t = yf.Ticker(ticker)
        df = t.history(period="1y")
        if df.empty: return None, None
        
        df['MA200'] = df['Close'].rolling(window=200).mean()
        df['RSI'] = calcular_rsi(df['Close'])
        return df, t.info
    except Exception:
        return None, None

# --- LOOP PRINCIPAL ---
tickers = ['PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'BBAS3.SA']

for ticker in tickers:
    dados, info = buscar_dados(ticker)
    if dados is not None:
        st.divider()
        st.header(f"🏢 {info.get('longName', ticker)}")
        
        preco_atual = float(dados['Close'].iloc[-1])
        media_val = dados['MA200'].iloc[-1]
        media_atual = float(media_val) if not np.isnan(media_val) else preco_atual
        rsi_atual = float(dados['RSI'].iloc[-1]) if not np.isnan(dados['RSI'].iloc[-1]) else 50
        
        st.line_chart(dados[['Close', 'MA200']])
        analise_fundamentalista_i10(ticker, info)
        analise_minuciosa_ia(ticker, preco_atual, media_atual, rsi_atual)

st.caption("Engenharia de IA 2026 - Gestão de Risco e Endividamento Atualizada.")
