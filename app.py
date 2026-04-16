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

# --- BARRA LATERAL: FILTROS DE ESTRATÉGIA ---
st.sidebar.header("🎯 Filtros de Estratégia")
st.sidebar.write("Defina os limites máximos ou mínimos para exibir as ações:")

f_pl = st.sidebar.slider("P/L Máximo", 0.0, 50.0, 15.0, step=0.5)
f_pvp = st.sidebar.slider("P/VP Máximo", 0.0, 10.0, 2.5, step=0.1)
f_dy = st.sidebar.slider("DY Mínimo (%)", 0.0, 20.0, 4.0, step=0.5)
f_roe = st.sidebar.slider("ROE Mínimo (%)", 0.0, 40.0, 10.0, step=1.0)
f_margem = st.sidebar.slider("Marg. Líquida Mínima (%)", 0.0, 50.0, 5.0, step=1.0)
f_ev_ebitda = st.sidebar.slider("EV/EBITDA Máximo", 0.0, 30.0, 12.0, step=0.5)
f_div_ebitda = st.sidebar.slider("Dív.Líq/EBITDA Máximo", 0.0, 10.0, 3.5, step=0.5)

# --- SISTEMA DE PRIVACIDADE ---
def get_random_header():
    agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0'
    ]
    return random.choice(agents)

def calcular_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# --- ANÁLISE FUNDAMENTALISTA COM LEGENDA AMPLIADA ---
def analise_fundamentalista_i10(ticker, info):
    st.subheader(f"📊 Indicadores Fundamentalistas (Estilo Investidor 10)")
    try:
        pl = info.get('trailingPE', 0)
        pvp = info.get('priceToBook', 0)
        roe = info.get('returnOnEquity', 0) * 100
        dy = info.get('dividendYield', 0) * 100
        margem = info.get('profitMargins', 0) * 100
        ev_ebitda = info.get('enterpriseToEbitda', 0)
        divida_liquida = info.get('totalDebt', 0) - info.get('totalCash', 0)
        ebitda_valor = info.get('ebitda', 0)
        div_ebitda = divida_liquida / ebitda_valor if ebitda_valor and ebitda_valor != 0 else 0

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
            * **P/VP:** Preço sobre Valor Patrimonial. Indica se a ação está barata em relação ao patrimônio físico.
            * **DY (Dividend Yield):** Rendimento em dividendos pagos nos últimos 12 meses.
            * **ROE:** Eficiência da empresa em gerar lucro com o capital dos acionistas.
            * **Margem Líquida:** Porcentagem de lucro real sobre a receita total.
            * **EV/EBITDA:** Valor da empresa dividido pelo lucro operacional. Ajuda a ver o preço real do negócio incluindo dívidas.
            * **Dívida Líquida / EBITDA:** Mede o endividamento. Indica quantos anos de lucro operacional seriam necessários para pagar a dívida total.
            * **RSI (IFR):** Mede se uma ação está "comprada demais" (acima de 70 = cara) ou "vendida demais" (abaixo de 30 = barata).
            """)
    except Exception:
        st.warning("Erro ao carregar indicadores.")

# --- FUNÇÃO DE IA E SENTIMENTO COM LINKS ---
def analise_minuciosa_ia(ticker, preco, media, rsi_atual):
    st.subheader(f"🕵️‍♂️ Inteligência de Mercado: {ticker}")
    noticias_detalhes = []
    texto_total_analise = ""
    user_config = Config()
    user_config.browser_user_agent = get_random_header()
    
    try:
        url_news = f"https://news.google.com/rss/search?q={ticker}+when:2d&hl=pt-BR&gl=BR&ceid=BR:pt-419"
        feed = feedparser.parse(url_news)
        if feed.entries:
            for entry in feed.entries[:6]: 
                titulo = entry.title.split(' - ')[0]
                link = entry.link
                noticias_detalhes.append({"titulo": titulo, "link": link})
                try:
                    time.sleep(random.uniform(0.3, 0.8))
                    article = Article(link, config=user_config)
                    article.download()
                    article.parse()
                    texto_total_analise += f"{titulo}. {article.text[:300]} "
                except:
                    texto_total_analise += f"{titulo}. "
    except Exception:
        pass

    positivas = ["alta", "dividendo", "lucro", "compra", "crescimento", "subiu", "positivo", "recorde"]
    negativas = ["queda", "risco", "prejuízo", "venda", "caiu", "dívida", "crise", "negativo"]
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
        st.write(f"🧠 Sentimento IA: {otimismo} Sinais Positivos | {pessimismo} Sinais de Risco")
        st.write(f"📈 RSI Atual: {rsi_atual:.2f}")

    with st.expander("📌 Ver Manchetes Analisadas (Links Oficiais)"):
        for n in noticias_detalhes:
            st.markdown(f"• {n['titulo']} [[Link]({n['link']})]")

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

# --- LOOP PRINCIPAL COM LÓGICA DE FILTRO ---
tickers = ['PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'BBAS3.SA', 'SANB11.SA', 'ABEV3.SA']
encontrou_alguma = False

for ticker in tickers:
    dados, info = buscar_dados(ticker)
    if dados is not None:
        # Extração de valores para o filtro
        pl = info.get('trailingPE', 0) or 0
        pvp = info.get('priceToBook', 0) or 0
        dy = (info.get('dividendYield', 0) or 0) * 100
        roe = (info.get('returnOnEquity', 0) or 0) * 100
        margem = (info.get('profitMargins', 0) or 0) * 100
        ev_ebitda = info.get('enterpriseToEbitda', 0) or 0
        divida_liquida = info.get('totalDebt', 0) - info.get('totalCash', 0)
        ebitda_valor = info.get('ebitda', 1) # evita divisão por zero
        div_ebitda = divida_liquida / ebitda_valor

        # LÓGICA DO FILTRO: Verifica se a ação atende a TODOS os critérios da barra lateral
        if (pl <= f_pl and pvp <= f_pvp and dy >= f_dy and roe >= f_roe and 
            margem >= f_margem and ev_ebitda <= f_ev_ebitda and div_ebitda <= f_div_ebitda):
            
            encontrou_alguma = True
            st.divider()
            st.header(f"🏢 {info.get('longName', ticker)}")
            
            preco_atual = float(dados['Close'].iloc[-1])
            media_val = dados['MA200'].iloc[-1]
            media_atual = float(media_val) if not np.isnan(media_val) else preco_atual
            rsi_atual = float(dados['RSI'].iloc[-1]) if not np.isnan(dados['RSI'].iloc[-1]) else 50
            
            st.line_chart(dados[['Close', 'MA200']])
            analise_fundamentalista_i10(ticker, info)
            analise_minuciosa_ia(ticker, preco_atual, media_atual, rsi_atual)

if not encontrou_alguma:
    st.info("💡 Nenhuma ação da lista atende aos filtros selecionados no momento. Tente flexibilizar os critérios na barra lateral.")

st.caption("Engenharia de IA 2026 - Sistema de Triagem Fundamentalista Pro.")
