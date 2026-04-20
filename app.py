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
st_autorefresh(interval=300 * 1000, key="data_refresh")  # atualiza a cada 5 minutos

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

def simular_performance_historica(hist, min_volume=50000):
    if len(hist) < 300:
        return {"taxa_compra": 0.0, "taxa_venda": 0.0, "retorno_medio_compra": 0.0, "retorno_medio_venda": 0.0,
                "expectancy_compra": 0.0, "expectancy_venda": 0.0, "sharpe_compra": 0.0, "sortino_compra": 0.0,
                "max_drawdown": 0.0, "qtd_compra": 0, "qtd_venda": 0}
    
    close = hist["Close"].copy()
    volume = hist.get("Volume", pd.Series(0, index=close.index))
    rsi = calcular_rsi_series(close)
    sma200 = close.rolling(window=200).mean()
    exp12 = close.ewm(span=12, adjust=False).mean()
    exp26 = close.ewm(span=26, adjust=False).mean()
    macd = exp12 - exp26
    sinal_macd = macd.ewm(span=9, adjust=False).mean()
    retorno_15d = close.shift(-15) / close - 1
    liquid_mask = volume > min_volume

    buy_mask = ((rsi < 35) & (close > sma200) & (macd > sinal_macd) & retorno_15d.notna() & liquid_mask)
    sell_mask = ((rsi > 70) & ((close < sma200) | (macd < sinal_macd)) & retorno_15d.notna() & liquid_mask)

    # Cálculo de métricas de compra (mantido o resto da função original)
    if buy_mask.any():
        ret_buy = retorno_15d[buy_mask]
        qtd_c = int(buy_mask.sum())
        taxa_c = (ret_buy > 0).mean() * 100
        ret_med_c = ret_buy.mean() * 100
        wins = ret_buy[ret_buy > 0]
        losses = ret_buy[ret_buy < 0]
        avg_win = wins.mean() if not wins.empty else 0
        avg_loss = abs(losses.mean()) if not losses.empty else 0
        expectancy_c = (taxa_c/100 * avg_win) - ((1 - taxa_c/100) * avg_loss) * 100
        returns = ret_buy.dropna()
        sharpe_c = returns.mean() / returns.std() * np.sqrt(252) if len(returns) > 5 and returns.std() != 0 else 0
        downside = returns[returns < 0]
        sortino_c = returns.mean() / downside.std() * np.sqrt(252) if len(downside) > 5 and downside.std() != 0 else 0
    else:
        taxa_c = ret_med_c = expectancy_c = sharpe_c = sortino_c = 0.0
        qtd_c = 0

    # Cálculo de venda (mantido)
    if sell_mask.any():
        ret_sell = retorno_15d[sell_mask]
        qtd_v = int(sell_mask.sum())
        taxa_v = (ret_sell < 0).mean() * 100
        ret_med_v = ret_sell.mean() * 100
        # ... (lógica de sell mantida igual)
        wins_v = ret_sell[ret_sell < 0]
        losses_v = ret_sell[ret_sell > 0]
        avg_win_v = abs(wins_v.mean()) if not wins_v.empty else 0
        avg_loss_v = losses_v.mean() if not losses_v.empty else 0
        expectancy_v = (taxa_v/100 * avg_win_v) - ((1 - taxa_v/100) * avg_loss_v) * 100
        returns_v = ret_sell.dropna()
        sharpe_v = returns_v.mean() / returns_v.std() * np.sqrt(252) if len(returns_v) > 5 and returns_v.std() != 0 else 0
        downside_v = returns_v[returns_v > 0]  # downside para sell é quando sobe (perda)
        sortino_v = returns_v.mean() / downside_v.std() * np.sqrt(252) if len(downside_v) > 5 and downside_v.std() != 0 else 0
    else:
        taxa_v = ret_med_v = expectancy_v = sharpe_v = sortino_v = 0.0
        qtd_v = 0

    # Max Drawdown
    if len(close) > 10:
        cum_ret = close.pct_change().cumsum()
        peak = cum_ret.cummax()
        drawdown = (cum_ret - peak) / peak
        max_dd = drawdown.min() * 100
    else:
        max_dd = 0.0

    return {
        "taxa_compra": taxa_c, "taxa_venda": taxa_v,
        "retorno_medio_compra": ret_med_c, "retorno_medio_venda": ret_med_v,
        "expectancy_compra": expectancy_c, "expectancy_venda": expectancy_v,
        "sharpe_compra": sharpe_c, "sortino_compra": sortino_c,
        "max_drawdown": max_dd,
        "qtd_compra": qtd_c, "qtd_venda": qtd_v
    }

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

    # Notícias
    noticias_texto = ""
    lista_links = []
    try:
        feed = feedparser.parse("https://news.google.com/rss/search?q=bitcoin&hl=pt-BR")
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

    # Lógica de veredito adaptada para BTC
    if rsi_val > 75 and score_n > score_p:
        veredito, cor = "VENDA FORTE 🚨", "error"
        motivo_detalhe = f"Sobrecompra extrema (RSI {rsi_val:.1f}) + sentimento negativo."
    elif rsi_val > 70:
        veredito, cor = "VENDA 🚨", "error"
        motivo_detalhe = f"RSI elevado ({rsi_val:.1f}) - possível correção."
    elif (close.iloc[-1] > sma200.iloc[-1] if not sma200.empty else True) and macd.iloc[-1] > sinal_macd.iloc[-1] and score_p > score_n + 3 and rsi_val < 60:
        veredito, cor = "COMPRA FORTE ✅", "success"
        motivo_detalhe = f"Trend de alta + sentimento positivo forte + RSI saudável."
    elif (close.iloc[-1] > sma200.iloc[-1] if not sma200.empty else True) and score_p > score_n and rsi_val < 65:
        veredito, cor = "COMPRA ✅", "success"
        motivo_detalhe = f"Momentum técnico favorável e notícias positivas."
    else:
        veredito, cor = "CAUTELA ⚠️", "warning"
        motivo_detalhe = "Sem sinal claro. Aguardar confirmação de tendência."

    return {
        "Ticker": "BTC-USD",
        "Preço": p_atual,
        "Veredito": veredito,
        "Cor": cor,
        "Motivo": motivo_detalhe,
        "RSI": rsi_val,
        "Hist": hist,
        "Links": lista_links,
        "TaxaCompra": sim["taxa_compra"],
        "ExpectancyCompra": sim["expectancy_compra"],
        "SharpeCompra": sim["sharpe_compra"],
        "SortinoCompra": sim["sortino_compra"],
        "MaxDrawdown": sim["max_drawdown"],
        "QtdCompra": sim["qtd_compra"]
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
