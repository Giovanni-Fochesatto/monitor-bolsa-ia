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

# Tentativa de importar statsmodels (com fallback seguro)
try:
    import statsmodels.api as sm
    STATS_MODELS_AVAILABLE = True
except ImportError:
    STATS_MODELS_AVAILABLE = False
    sm = None

st.set_page_config(page_title="Monitor IA Pro - Bitcoin & Ethereum", layout="wide")
st_autorefresh(interval=300 * 1000, key="data_refresh")

if "telegram_ativado" not in st.session_state:
    st.session_state.telegram_ativado = False
if "ultimos_alertas" not in st.session_state:
    st.session_state.ultimos_alertas = set()
if "usar_tradingview" not in st.session_state:
    st.session_state.usar_tradingview = True

# ===================== BANCO DE DADOS =====================
def init_db():
    conn = sqlite3.connect('sinais_crypto.db')
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
    conn = sqlite3.connect('sinais_crypto.db')
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

# ===================== FUNÇÕES TÉCNICAS (todas as suas linhas originais mantidas) =====================
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
        downside = df[df["retorno_estrategia"] < 0]["retorno_estrategia"]
        downside_std = downside.std()
        sortino = (df["retorno_estrategia"].mean() / downside_std) * np.sqrt(252) if downside_std != 0 else 0
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
        return {
            "taxa_compra": 0,
            "expectancy_compra": 0,
            "sharpe_compra": 0,
            "sortino_compra": 0,
            "max_drawdown": 0,
            "qtd_compra": 0
        }

# ===================== REGRESSÃO LOGÍSTICA REAL (IA que aprende) =====================
def treinar_modelo_logistico(hist):
    if not STATS_MODELS_AVAILABLE or len(hist) < 300:
        return 50.0, 50.0
    df = hist.copy()
    df["Target"] = (df["Close"].shift(-15) > df["Close"]).astype(int)
    df = df.dropna()
    df["RSI"] = calcular_rsi_series(df["Close"])
    df["SMA200"] = df["Close"].rolling(200).mean()
    df["MACD"] = df["Close"].ewm(span=12).mean() - df["Close"].ewm(span=26).mean()
    df["MACD_signal"] = df["MACD"].ewm(span=9).mean()
    df["Vol_Change"] = df["Volume"].pct_change()
    df["Above_SMA200"] = (df["Close"] > df["SMA200"]).astype(int)
    
    features = ["RSI", "MACD", "MACD_signal", "Vol_Change", "Above_SMA200"]
    X = df[features].dropna()
    y = df["Target"].loc[X.index]
    X = sm.add_constant(X)
    model = sm.Logit(y, X).fit(disp=0)
    last_row = X.iloc[-1:]
    prob_alta = model.predict(last_row)[0] * 100
    preds = (model.predict(X) > 0.5).astype(int)
    acuracia = (preds == y).mean() * 100
    return round(prob_alta, 1), round(acuracia, 1)

# ===================== CACHE =====================
@st.cache_data(ttl=300, show_spinner=False)
def obter_dados_crypto(ticker):
    info = yf.Ticker(ticker).info
    hist = yf.Ticker(ticker).history(period="5y", auto_adjust=True)
    return info, hist

@st.cache_data(ttl=1800, show_spinner=False)
def obter_macro():
    macro = {}
    try:
        macro["Dolar"] = yf.Ticker("USDBRL=X").fast_info.last_price
    except:
        macro["Dolar"] = 5.70
    macro["Bitcoin_USD"] = yf.Ticker("BTC-USD").fast_info.last_price
    macro["Ethereum_USD"] = yf.Ticker("ETH-USD").fast_info.last_price
    return macro

# ===================== PROCESSAMENTO CENTRAL =====================
def processar_crypto(ticker):
    info, hist = obter_dados_crypto(ticker)
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

    noticias_texto = ""
    lista_links = []
    try:
        feed = feedparser.parse(f"https://news.google.com/rss/search?q={ticker.replace('-USD','')}&hl=pt-BR")
        for entry in feed.entries[:6]:
            titulo = entry.title.lower()
            noticias_texto += titulo + " "
            lista_links.append({"titulo": entry.title, "link": entry.link})
    except:
        pass
    score_p = sum(noticias_texto.count(w) for w in ["alta", "bull", "compra", "subiu", "rally", "high", "buy", "halving"])
    score_n = sum(noticias_texto.count(w) for w in ["queda", "bear", "venda", "caiu", "crash", "low", "sell", "dump"])

    sim = simular_performance_historica(hist)
    p_atual = float(close.iloc[-1]) if not close.empty else 0

    prob_alta, acuracia_ia = treinar_modelo_logistico(hist)

    if prob_alta > 75:
        veredito, cor = "COMPRA FORTE 🟢", "success"
        motivo_detalhe = f"Probabilidade real de alta em 15 dias: {prob_alta}% (IA treinada em 5 anos)"
    elif prob_alta > 60:
        veredito, cor = "COMPRA 🟢", "success"
        motivo_detalhe = f"Probabilidade moderada de alta: {prob_alta}%"
    elif prob_alta < 30:
        veredito, cor = "VENDA FORTE 🔴", "error"
        motivo_detalhe = f"Probabilidade de queda: {100 - prob_alta}%"
    elif prob_alta < 45:
        veredito, cor = "VENDA 🔴", "error"
        motivo_detalhe = f"Probabilidade baixa de alta: {prob_alta}%"
    else:
        veredito, cor = "NEUTRO ⚖️", "warning"
        motivo_detalhe = f"Probabilidade neutra: {prob_alta}%"

    return {
        "Ticker": ticker,
        "Preço": p_atual,
        "Veredito": veredito,
        "Cor": cor,
        "Motivo": motivo_detalhe,
        "RSI": rsi_val,
        "Hist": hist,
        "Links": lista_links,
        "ProbabilidadeAlta": prob_alta,
        "AcuraciaIA": acuracia_ia,
        "TaxaCompra": sim.get("taxa_compra", 0),
        "ExpectancyCompra": sim.get("expectancy_compra", 0),
        "SharpeCompra": sim.get("sharpe_compra", 0),
        "SortinoCompra": sim.get("sortino_compra", 0),
        "MaxDrawdown": sim.get("max_drawdown", 0),
        "QtdCompra": sim.get("qtd_compra", 0),
    }

# ===================== LAYOUT =====================
st.sidebar.title("₿ Monitor IA Pro - Crypto")
crypto_selecionado = st.sidebar.selectbox("Escolha o ativo:", ["BTC-USD", "ETH-USD"])

st.sidebar.subheader("📊 Câmbio em Tempo Real")
cambio = obter_macro()
st.sidebar.metric("Bitcoin (USD)", f"${cambio['Bitcoin_USD']:,.0f}")
st.sidebar.metric("Ethereum (USD)", f"${cambio['Ethereum_USD']:,.0f}")
st.sidebar.metric("Dólar (R$)", f"R$ {cambio['Dolar']:.2f}")

st.title(f"🤖 Monitor IA Pro - {crypto_selecionado}")
st.caption(f"Atualização: {time.strftime('%H:%M:%S')} | Local: Porto Alegre/RS | IA com Regressão Logística")

with st.spinner("📡 Baixando dados e treinando IA..."):
    resultado = processar_crypto(crypto_selecionado)
dados_vencedoras = [resultado] if resultado else []

# ===================== TABS (seu código original mantido integralmente) =====================
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Gráfico Técnico", "📉 Análise", "📜 Backtest & Histórico"])

with tab1:
    if dados_vencedoras:
        acao = dados_vencedoras[0]
        st.subheader(f"🏆 Análise Atual - {acao['Ticker']}")
        col1, col2, col3 = st.columns(3)
        col1.metric("Preço Atual (USD)", f"${acao['Preço']:,.0f}")
        col2.metric("RSI (14)", f"{acao['RSI']:.1f}")
        col3.metric("Veredito IA", acao['Veredito'])
        st.markdown(f"**Motivo:** {acao['Motivo']}")
        
        if acao.get('ProbabilidadeAlta', 50) > 70:
            st.success(f"🚨 **ALERTA BLOOMBERG** — {acao['ProbabilidadeAlta']}% de probabilidade de alta nos próximos 15 dias (Acurácia da IA: {acao.get('AcuraciaIA', 0)}%)")
        
        df_resumo = pd.DataFrame([acao])
        st.dataframe(df_resumo[["Preço", "TaxaCompra", "ExpectancyCompra", "SharpeCompra", "QtdCompra"]], use_container_width=True, hide_index=True)
        salvar_sinal(acao)

# As demais tabs (Gráfico Técnico, Análise, Backtest) permanecem exatamente como no seu código original
# (copiei todas as linhas para não perder nada)

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
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.65, 0.20, 0.15])
            fig.add_trace(go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'], name="Preço"), row=1, col=1)
            fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA20'], name="SMA 20", line=dict(color="yellow")), row=1, col=1)
            fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA200'], name="SMA 200", line=dict(color="orange")), row=1, col=1)
            fig.add_trace(go.Scatter(x=hist.index, y=hist['BB_Upper'], name="BB Upper", line=dict(color="lime")), row=1, col=1)
            fig.add_trace(go.Scatter(x=hist.index, y=hist['BB_Lower'], name="BB Lower", line=dict(color="red")), row=1, col=1)
            fig.add_trace(go.Bar(x=hist.index, y=hist['Volume'], name="Volume", marker_color="gray"), row=2, col=1)
            fig.add_trace(go.Scatter(x=hist.index, y=rsi_series, name="RSI", line=dict(color="purple")), row=3, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="lime", row=3, col=1)
            fig.update_layout(height=750, template="plotly_dark", showlegend=True, title_text=f"{acao['Ticker']} - Análise Técnica")
            st.plotly_chart(fig, use_container_width=True)
        if st.session_state.usar_tradingview:
            tv_html = f"""
            <iframe src="https://www.tradingview.com/widgetembed/?symbol={acao['Ticker']}&interval=D&theme=dark&style=1" 
                    width="100%" height="650" frameborder="0" allowfullscreen></iframe>
            """
            st.components.v1.html(tv_html, height=680)

# As tabs 3 e 4 permanecem idênticas ao seu código original (mantive todas as linhas)

with tab3:
    if dados_vencedoras:
        acao = dados_vencedoras[0]
        st.subheader("📉 Análise Detalhada")
        st.metric("Preço Atual", f"${acao['Preço']:,.0f}")
        st.markdown(f"**Veredito da IA:** {acao['Veredito']}")
        st.markdown(f"**Motivo:** {acao['Motivo']}")
        if acao.get("Links"):
            st.markdown("**Últimas Manchetes:**")
            for n in acao["Links"][:5]:
                st.markdown(f"• [{n['titulo']}]({n['link']})")

with tab4:
    st.subheader("📜 Backtest & Histórico de Sinais")
    if dados_vencedoras:
        acao = dados_vencedoras[0]
        col1, col2, col3 = st.columns(3)
        col1.metric("Expectancy Compra", f"{acao['ExpectancyCompra']:.2f}%")
        col2.metric("Sharpe Ratio", f"{acao['SharpeCompra']:.2f}")
        col3.metric("Sinais Históricos", acao['QtdCompra'])
    try:
        conn = sqlite3.connect('sinais_crypto.db')
        df_historico = pd.read_sql("SELECT * FROM sinais ORDER BY data DESC", conn)
        conn.close()
        if not df_historico.empty:
            st.dataframe(df_historico, use_container_width=True, hide_index=True)
            if st.button("Exportar Histórico para CSV"):
                csv = df_historico.to_csv(index=False)
                st.download_button("Baixar CSV", csv, "historico_sinais_crypto.csv", "text/csv")
        else:
            st.info("Ainda não há sinais salvos.")
    except:
        st.info("Ainda não há sinais salvos.")

st.success("✅ Monitor Bitcoin + Ethereum configurado com IA que aprende (Regressão Logística)!")
