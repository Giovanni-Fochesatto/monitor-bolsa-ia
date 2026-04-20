import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time
import feedparser
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh
import datetime
import sqlite3

# ===================== CONFIGURAÇÕES =====================
st.set_page_config(page_title="Monitor IA Pro - Bitcoin", layout="wide")
st_autorefresh(interval=300 * 1000, key="data_refresh") # atualiza a cada 5 minutos

if "telegram_ativado" not in st.session_state:
    st.session_state.telegram_ativado = False
if "ultimos_alertas" not in st.session_state:
    st.session_state.ultimos_alertas = set()
if "usar_tradingview" not in st.session_state:
    st.session_state.usar_tradingview = True

# ===================== BANCO DE DADOS (adaptado para BTC) =====================
def init_db():
    conn = sqlite3.connect('sinais_bitcoin.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS sinais (
                    data TEXT,
                    ticker TEXT,
                    preco REAL,
                    veredito TEXT,
                    motivo TEXT,
                    expectancy REAL,
                    sharpe REAL
                )''')
    conn.commit()
    conn.close()

def salvar_sinal(acao):
    init_db()
    conn = sqlite3.connect('sinais_bitcoin.db')
    c = conn.cursor()
    agora = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute('''INSERT INTO sinais
                 (data, ticker, preco, veredito, motivo, expectancy, sharpe)
                 VALUES (?, ?, ?, ?, ?, ?, ?)''',
              (agora, acao['Ticker'], acao['Preço'],
               acao['Veredito'], acao['Motivo'],
               acao.get('ExpectancyCompra', 0), acao.get('SharpeCompra', 0)))
    conn.commit()
    conn.close()

# ===================== FUNÇÕES TÉCNICAS =====================
def calcular_rsi_series(close: pd.Series, window: int = 14) -> pd.Series:
    if len(close) < window:
        return pd.Series([50.0] * len(close), index=close.index)
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
    rs = gain / loss.where(loss != 0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def calcular_rsi(data, window: int = 14):
    if len(data) < window:
        return 50.0
    return float(calcular_rsi_series(data, window).iloc[-1])

def simular_performance_historica(hist):
    try:
        if hist is None or len(hist) < 50:
            raise ValueError("Histórico insuficiente")
        df = hist.copy()
        df["retorno"] = df["Close"].pct_change()
        df["sinal"] = 0
        df.loc[df["Close"] > df["Close"].rolling(20).mean(), "sinal"] = 1
        df["retorno_estrategia"] = df["sinal"].shift(1) * df["retorno"]
        trades = df[df["sinal"].diff() == 1]
        ganhos = df[df["retorno_estrategia"] > 0]["retorno_estrategia"]
        perdas = df[df["retorno_estrategia"] < 0]["retorno_estrategia"]
        total_trades = len(trades)
        winrate = len(ganhos) / (len(ganhos) + len(perdas)) if (len(ganhos) + len(perdas)) > 0 else 0
        media_gain = ganhos.mean() if len(ganhos) > 0 else 0
        media_loss = perdas.mean() if len(perdas) > 0 else 0
        expectancy = (winrate * media_gain) + ((1 - winrate) * media_loss)
        std = df["retorno_estrategia"].std()
        sharpe = (df["retorno_estrategia"].mean() / std) * np.sqrt(252) if std != 0 else 0
        # Sortino
        downside = df[df["retorno_estrategia"] < 0]["retorno_estrategia"]
        downside_std = downside.std()
        sortino = (df["retorno_estrategia"].mean() / downside_std) * np.sqrt(252) if downside_std != 0 else 0
        # Drawdown
        df["equity"] = (1 + df["retorno_estrategia"].fillna(0)).cumprod()
        df["peak"] = df["equity"].cummax()
        df["drawdown"] = (df["equity"] - df["peak"]) / df["peak"]
        max_dd = df["drawdown"].min()
        return {
            "taxa_compra": float(winrate * 100),
            "expectancy_compra": float(expectancy * 100),
            "sharpe_compra": float(sharpe),
            "sortino_compra": float(sortino),
            "max_drawdown": float(max_dd * 100),
            "qtd_compra": int(total_trades)
        }
    except Exception as e:
        # 🔒 fallback institucional (NUNCA quebra app)
        return {
            "taxa_compra": 0,
            "expectancy_compra": 0,
            "sharpe_compra": 0,
            "sortino_compra": 0,
            "max_drawdown": 0,
            "qtd_compra": 0
        }
    # ===================== MÉTRICAS =====================
    winrate = len(ganhos) / (len(ganhos) + len(perdas)) if (len(ganhos) + len(perdas)) > 0 else 0
    media_gain = ganhos.mean() if len(ganhos) > 0 else 0
    media_loss = perdas.mean() if len(perdas) > 0 else 0
    expectancy = (winrate * media_gain) + ((1 - winrate) * media_loss)
    std = df["retorno_estrategia"].std()
    sharpe = (df["retorno_estrategia"].mean() / std) * np.sqrt(252) if std != 0 else 0
    # ===================== SORTINO =====================
    downside = df[df["retorno_estrategia"] < 0]["retorno_estrategia"]
    downside_std = downside.std()
    sortino = (df["retorno_estrategia"].mean() / downside_std) * np.sqrt(252) if downside_std != 0 else 0
    # ===================== MAX DRAWDOWN =====================
    df["equity"] = (1 + df["retorno_estrategia"].fillna(0)).cumprod()
    df["peak"] = df["equity"].cummax()
    df["drawdown"] = (df["equity"] - df["peak"]) / df["peak"]
    max_dd = df["drawdown"].min()
    return {
        "expectancy_compra": float(expectancy * 100),
        "sharpe_compra": float(sharpe),
        "taxa_compra": float(winrate * 100),
        "sortino_compra": float(sortino),
        "max_drawdown": float(max_dd * 100),
        "qtd_compra": int(total_trades)
    }

# ===================== FUNÇÃO ENSEMBLE SCORE (ADICIONADA PARA RESOLVER O ERRO) =====================
def modelo_ensemble_score(hist, rsi_val, macd, sinal_macd, score_p, score_n):
    """Score IA Ensemble - combina RSI, MACD, tendência e sentimento de notícias"""
    score = 50.0  # base neutra
    
    # 1. RSI (peso forte)
    if rsi_val < 35:
        score += 28
    elif rsi_val < 45:
        score += 12
    elif rsi_val > 70:
        score -= 28
    elif rsi_val > 65:
        score -= 15
    
    # 2. MACD (cruzamento e direção)
    if len(macd) > 1 and len(sinal_macd) > 1:
        if macd.iloc[-1] > sinal_macd.iloc[-1] and macd.iloc[-2] <= sinal_macd.iloc[-2]:
            score += 18  # cruzamento bullish
        elif macd.iloc[-1] < sinal_macd.iloc[-1] and macd.iloc[-2] >= sinal_macd.iloc[-2]:
            score -= 18  # cruzamento bearish
        if macd.iloc[-1] > 0:
            score += 8
    
    # 3. Tendência de longo prazo (SMA200)
    if len(hist) > 200:
        sma200 = hist["Close"].rolling(window=200).mean().iloc[-1]
        if hist["Close"].iloc[-1] > sma200:
            score += 12
    
    # 4. Sentimento de notícias
    sentiment = score_p - score_n
    score += sentiment * 4
    score = max(0, min(100, score))  # limita entre 0 e 100
    
    return round(score, 1)

# ===================== CACHE =====================
@st.cache_data(ttl=300, show_spinner=False)
def obter_dados_bitcoin():
    ticker = yf.Ticker("BTC-USD")
    info = ticker.info
    hist = ticker.history(period="5y", auto_adjust=True)
    return info, hist

@st.cache_data(ttl=1800, show_spinner=False)
def obter_macro():
    macro = {}
    try:
        macro["Dolar"] = yf.Ticker("USDBRL=X").fast_info.last_price
    except:
        macro["Dolar"] = 5.70
    macro["Bitcoin_USD"] = yf.Ticker("BTC-USD").fast_info.last_price
    return macro

# ===================== PROCESSAMENTO CENTRAL (adaptado para BTC) =====================
def processar_bitcoin():
    info, hist = obter_dados_bitcoin()
    if hist.empty:
        return None
    hist = hist.copy()
    close = hist["Close"]
    rsi_val = calcular_rsi(close)
    sma200 = close.rolling(window=200).mean()
    exp12 = close.ewm(span=12, adjust=False).mean()
    exp26 = close.ewm(span=26, adjust=False).mean()
    macd = exp12 - exp26
    sinal_macd = macd.ewm(span=9, adjust=False).mean()
    # ===================== NOTÍCIAS =====================
    noticias_texto = ""
    lista_links = []
    try:
        feed = feedparser.parse("https://news.google.com/rss/search?q=bitcoin&hl=pt-BR")
        for entry in feed.entries[:6]:
            titulo = entry.title.lower()
            noticias_texto += titulo + " "
            lista_links.append({
                "titulo": entry.title,
                "link": entry.link
            })
    except:
        pass
    score_p = sum(noticias_texto.count(w) for w in ["alta", "bull", "compra", "subiu", "rally", "high", "buy", "halving"])
    score_n = sum(noticias_texto.count(w) for w in ["queda", "bear", "venda", "caiu", "crash", "low", "sell", "dump"])
    # ===================== BACKTEST =====================
    sim = simular_performance_historica(hist) or {}
    p_atual = float(close.iloc[-1]) if not close.empty else 0
    # ===================== IA ENSEMBLE =====================
    score_ia = modelo_ensemble_score(hist, rsi_val, macd, sinal_macd, score_p, score_n)
    # ===================== VEREDITO INTELIGENTE =====================
    if score_ia > 75:
        veredito, cor = "COMPRA FORTE 🟢", "success"
        motivo_detalhe = f"Score IA elevado ({score_ia}/100) com confluência técnica e sentimento positivo."
    elif score_ia > 60:
        veredito, cor = "COMPRA 🟢", "success"
        motivo_detalhe = f"Score IA favorável ({score_ia}/100), tendência consistente."
    elif score_ia < 25:
        veredito, cor = "VENDA FORTE 🔴", "error"
        motivo_detalhe = f"Score IA muito baixo ({score_ia}/100), pressão vendedora dominante."
    elif score_ia < 40:
        veredito, cor = "VENDA 🔴", "error"
        motivo_detalhe = f"Score IA fraco ({score_ia}/100), risco de queda."
    else:
        veredito, cor = "NEUTRO ⚖️", "warning"
        motivo_detalhe = f"Score IA neutro ({score_ia}/100), mercado indefinido."
    # ===================== RETORNO =====================
    return {
        "Ticker": "BTC-USD",
        "Preço": p_atual,
        "Veredito": veredito,
        "Cor": cor,
        "Motivo": motivo_detalhe,
        "RSI": rsi_val,
        "Hist": hist,
        "Links": lista_links,
        "ScoreIA": score_ia,
        "TaxaCompra": sim.get("taxa_compra", 0),
        "ExpectancyCompra": sim.get("expectancy_compra", 0),
        "SharpeCompra": sim.get("sharpe_compra", 0),
        "SortinoCompra": sim.get("sortino_compra", 0),
        "MaxDrawdown": sim.get("max_drawdown", 0),
        "QtdCompra": sim.get("qtd_compra", 0),
    }

# ===================== SIDEBAR =====================
st.sidebar.title("₿ Monitor IA Pro - Bitcoin")
st.sidebar.subheader("📊 Câmbio em Tempo Real")
cambio = obter_macro()
st.sidebar.metric("Bitcoin (USD)", f"${cambio['Bitcoin_USD']:,.0f}")
st.sidebar.metric("Dólar (R$)", f"R$ {cambio['Dolar']:.2f}")
st.sidebar.divider()
st.sidebar.subheader("📺 TradingView")
st.session_state.usar_tradingview = st.sidebar.checkbox("Usar gráfico TradingView", value=True)
estrategia_ativa = st.sidebar.selectbox(
    "Estratégia:",
    ["Análise Técnica (Trader)", "Momentum + Sentimento", "Position Trading"]
)

# ===================== CABEÇALHO =====================
st.title("🤖 Monitor IA Pro - Bitcoin")
st.caption(f"Atualização: {time.strftime('%H:%M:%S')} | Local: Porto Alegre/RS")

# ===================== PROCESSAMENTO =====================
with st.spinner("📡 Baixando dados do Bitcoin..."):
    resultado = processar_bitcoin()
dados_vencedoras = [resultado] if resultado else []

# ===================== TABS =====================
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Gráfico Técnico", "📉 Análise", "📜 Backtest & Histórico"])

# ===================== TAB 1 - OVERVIEW =====================
with tab1:
    if dados_vencedoras:
        st.subheader(f"🏆 Análise Atual - Estratégia: {estrategia_ativa}")
        acao = dados_vencedoras[0]
        col1, col2, col3 = st.columns(3)
        col1.metric("Preço Atual (USD)", f"${acao['Preço']:,.0f}")
        col2.metric("RSI (14)", f"{acao['RSI']:.1f}")
        col3.metric("Veredito IA", acao['Veredito'])
        st.markdown(f"**Motivo:** {acao['Motivo']}")
        df_resumo = pd.DataFrame([acao])
        st.dataframe(
            df_resumo[["Preço", "TaxaCompra", "ExpectancyCompra", "SharpeCompra", "QtdCompra"]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "TaxaCompra": st.column_config.NumberColumn("Win Rate %", format="%.1f%%"),
                "ExpectancyCompra": st.column_config.NumberColumn("Expectancy %", format="%.2f%%"),
                "SharpeCompra": st.column_config.NumberColumn("Sharpe", format="%.2f"),
                "QtdCompra": st.column_config.NumberColumn("Sinais Históricos"),
            }
        )
        salvar_sinal(acao)
    else:
        st.error("Erro ao carregar dados do Bitcoin.")

# ===================== TAB 2 - GRÁFICO TÉCNICO =====================
with tab2:
    if dados_vencedoras:
        acao = dados_vencedoras[0]
        hist = acao["Hist"].copy()
        if not hist.empty:
            hist['SMA20'] = hist['Close'].rolling(window=20).mean()
            hist['SMA200'] = hist['Close'].rolling(window=200).mean()
            hist['BB_Mid'] = hist['Close'].rolling(window=20).mean()
            hist['BB_Std'] = hist['Close'].rolling(window=20).std()
            hist['BB_Upper'] = hist['BB_Mid'] + 2 * hist['BB_Std']
            hist['BB_Lower'] = hist['BB_Mid'] - 2 * hist['BB_Std']
            rsi_series = calcular_rsi_series(hist['Close'])
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                                row_heights=[0.65, 0.20, 0.15],
                                subplot_titles=("Candlestick + Bollinger + SMAs", "Volume", "RSI (14)"))
            fig.add_trace(go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'],
                                         low=hist['Low'], close=hist['Close'], name="Preço"), row=1, col=1)
            fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA20'], name="SMA 20", line=dict(color="yellow")), row=1, col=1)
            fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA200'], name="SMA 200", line=dict(color="orange")), row=1, col=1)
            fig.add_trace(go.Scatter(x=hist.index, y=hist['BB_Upper'], name="BB Upper", line=dict(color="lime")), row=1, col=1)
            fig.add_trace(go.Scatter(x=hist.index, y=hist['BB_Lower'], name="BB Lower", line=dict(color="red")), row=1, col=1)
            fig.add_trace(go.Bar(x=hist.index, y=hist['Volume'], name="Volume", marker_color="gray"), row=2, col=1)
            fig.add_trace(go.Scatter(x=hist.index, y=rsi_series, name="RSI", line=dict(color="purple")), row=3, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="lime", row=3, col=1)
            fig.update_layout(height=750, template="plotly_dark", showlegend=True,
                              title_text=f"Bitcoin (BTC-USD) - Análise Técnica")
            st.plotly_chart(fig, use_container_width=True)
        if st.session_state.usar_tradingview:
            tv_html = """
            <iframe src="https://www.tradingview.com/widgetembed/?symbol=BTCUSD&interval=D&theme=dark&style=1"
                    width="100%" height="650" frameborder="0" allowfullscreen></iframe>
            """
            st.components.v1.html(tv_html, height=680)
    else:
        st.info("Carregando gráfico...")

# ===================== TAB 3 - ANÁLISE =====================
with tab3:
    if dados_vencedoras:
        acao = dados_vencedoras[0]
        st.subheader("📉 Análise Detalhada")
        st.metric("Preço Atual", f"${acao['Preço']:,.0f}")
        st.markdown(f"**Veredito da IA:** {acao['Veredito']}")
        st.markdown(f"**Motivo:** {acao['Motivo']}")
        if acao.get("Links"):
            st.markdown("**Últimas Manchetes sobre Bitcoin:**")
            for n in acao["Links"][:5]:
                st.markdown(f"• [{n['titulo']}]({n['link']})")
    else:
        st.info("Nenhum dado disponível.")

# ===================== TAB 4 - BACKTEST =====================
with tab4:
    st.subheader("📜 Backtest & Histórico de Sinais")
    if dados_vencedoras:
        acao = dados_vencedoras[0]
        col1, col2, col3 = st.columns(3)
        col1.metric("Expectancy Compra", f"{acao['ExpectancyCompra']:.2f}%")
        col2.metric("Sharpe Ratio", f"{acao['SharpeCompra']:.2f}")
        col3.metric("Sinais de Compra Históricos", acao['QtdCompra'])
    # Histórico salvo
    try:
        conn = sqlite3.connect('sinais_bitcoin.db')
        df_historico = pd.read_sql("SELECT * FROM sinais ORDER BY data DESC", conn)
        conn.close()
        if not df_historico.empty:
            st.dataframe(df_historico, use_container_width=True, hide_index=True)
            if st.button("Exportar Histórico para CSV"):
                csv = df_historico.to_csv(index=False)
                st.download_button("Baixar CSV", csv, "historico_sinais_bitcoin.csv", "text/csv")
        else:
            st.info("Ainda não há sinais salvos.")
    except:
        st.info("Ainda não há sinais salvos.")

st.success("✅ Monitor Bitcoin configurado com sucesso! Atualiza automaticamente a cada 5 minutos.")
