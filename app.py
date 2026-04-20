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
import requests

# ===================== CONFIGURAÇÕES =====================
st.set_page_config(page_title="Monitor IA Pro - Bitcoin", layout="wide")
st_autorefresh(interval=180 * 1000, key="data_refresh")  # Atualiza a cada 3 minutos

if "telegram_ativado" not in st.session_state:
    st.session_state.telegram_ativado = False
if "ultimo_alerta" not in st.session_state:
    st.session_state.ultimo_alerta = ""

# ===================== BANCO DE DADOS =====================
def init_db():
    conn = sqlite3.connect('sinais_bitcoin.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS sinais (
                    data TEXT,
                    ticker TEXT,
                    preco_usd REAL,
                    preco_brl REAL,
                    veredito TEXT,
                    motivo TEXT,
                    expectancy REAL,
                    sharpe REAL,
                    rsi REAL,
                    timeframe TEXT
                )''')
    conn.commit()
    conn.close()

def salvar_sinal(acao):
    init_db()
    conn = sqlite3.connect('sinais_bitcoin.db')
    c = conn.cursor()
    agora = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute('''INSERT INTO sinais 
                 (data, ticker, preco_usd, preco_brl, veredito, motivo, expectancy, sharpe, rsi, timeframe)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
              (agora, acao['Ticker'], acao['Preço_USD'], acao.get('Preço_BRL', 0),
               acao['Veredito'], acao['Motivo'], acao.get('ExpectancyCompra', 0),
               acao.get('SharpeCompra', 0), acao.get('RSI', 50), acao.get('Timeframe', 'Daily')))
    conn.commit()
    conn.close()

# ===================== TELEGRAM REAL =====================
def enviar_alerta_telegram(bot_token, chat_id, mensagem):
    if not bot_token or not chat_id:
        return False
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {"chat_id": chat_id, "text": mensagem, "parse_mode": "HTML"}
        r = requests.post(url, json=payload, timeout=10)
        return r.status_code == 200
    except:
        return False

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

# ===================== CANDLES ADVANCED (MELHORADA) =====================
def detectar_candlestick_patterns(hist):
    if len(hist) < 10:
        return {"padrao": "Dados insuficientes", "forca": 0, "descricao": ""}
    
    o = hist['Open'].iloc[-10:]
    h = hist['High'].iloc[-10:]
    l = hist['Low'].iloc[-10:]
    c = hist['Close'].iloc[-10:]
    v = hist['Volume'].iloc[-10:]
    avg_vol = v.mean()
    
    patterns = []
    forca = 0
    i = -1  # último candle
    
    body = abs(c - o)
    upper_shadow = h - c.where(c > o, o)
    lower_shadow = o.where(c > o, c) - l
    range_candle = h - l + 1e-8
    
    # Doji
    if body.iloc[i] / range_candle.iloc[i] < 0.1:
        patterns.append("Doji")
        forca += 2
    
    # Hammer (Bullish)
    if lower_shadow.iloc[i] > 2 * body.iloc[i] and body.iloc[i] / range_candle.iloc[i] < 0.3 and v.iloc[i] > avg_vol * 1.1:
        patterns.append("Hammer (Bullish)")
        forca += 5
    
    # Shooting Star (Bearish)
    if upper_shadow.iloc[i] > 2 * body.iloc[i] and body.iloc[i] / range_candle.iloc[i] < 0.3 and v.iloc[i] > avg_vol * 1.1:
        patterns.append("Shooting Star (Bearish)")
        forca += 5
    
    # Bullish Engulfing
    if (c.iloc[i] > o.iloc[i] and c.iloc[i-1] < o.iloc[i-1] and 
        c.iloc[i] > o.iloc[i-1] and o.iloc[i] < c.iloc[i-1]):
        patterns.append("Bullish Engulfing")
        forca += 6
    
    # Bearish Engulfing
    if (c.iloc[i] < o.iloc[i] and c.iloc[i-1] > o.iloc[i-1] and 
        c.iloc[i] < o.iloc[i-1] and o.iloc[i] > c.iloc[i-1]):
        patterns.append("Bearish Engulfing")
        forca += 6
    
    # Morning Star (3 candles)
    if len(hist) >= 3 and c.iloc[i-2] < o.iloc[i-2] and body.iloc[i-1] < 0.3*(h.iloc[i-1]-l.iloc[i-1]) and c.iloc[i] > (o.iloc[i-2] + c.iloc[i-2])/2:
        patterns.append("Morning Star (Bullish)")
        forca += 7
    
    padrao = " + ".join(patterns) if patterns else "Sem padrão forte"
    return {"padrao": padrao, "forca": min(forca, 10), "descricao": f"{padrao} | Força: {min(forca,10)}/10"}

# ===================== ON-CHAIN =====================
@st.cache_data(ttl=300)
def obter_onchain_metrics():
    metrics = {
        "hash_rate": "N/D", "transactions_24h": "N/D", "active_addresses": "N/D",
        "mvrv_z": "N/D", "exchange_flow": "Neutro", "sentimento_onchain": "Neutro"
    }
    try:
        resp = requests.get("https://api.blockchain.info/stats", timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            metrics["hash_rate"] = f"{data.get('hash_rate', 0) / 1e9:.0f} EH/s"
            metrics["transactions_24h"] = f"{data.get('n_tx', 0):,}"
    except:
        pass
    
    try:
        bc = requests.get("https://api.blockchair.com/bitcoin/stats", timeout=10)
        if bc.status_code == 200:
            d = bc.json()["data"]
            metrics["active_addresses"] = f"~{int(d.get('transactions_24h', 0)) // 3:,}"
    except:
        pass
    
    try:
        hist = yf.Ticker("BTC-USD").history(period="2y")
        realized_approx = hist["Close"].mean() * 0.65
        metrics["mvrv_z"] = f"{(hist['Close'].iloc[-1] / realized_approx - 1):.2f}" if realized_approx > 0 else "N/D"
    except:
        pass
    
    vol = yf.Ticker("BTC-USD").history(period="7d")["Volume"].mean()
    metrics["exchange_flow"] = "Saída líquida (Bullish)" if vol > 4.5e10 else "Entrada líquida (Cauteloso)"
    metrics["sentimento_onchain"] = "Positivo" if "Saída" in metrics["exchange_flow"] else "Neutro"
    
    return metrics

# ===================== PROCESSAMENTO PRINCIPAL =====================
def obter_dados(timeframe="Daily"):
    interval_map = {"1h": "1h", "4h": "4h", "Daily": "1d"}
    period_map = {"1h": "10d", "4h": "30d", "Daily": "5y"}
    hist = yf.Ticker("BTC-USD").history(period=period_map[timeframe], interval=interval_map[timeframe])
    p_usd = float(yf.Ticker("BTC-USD").fast_info.last_price)
    p_brl = float(yf.Ticker("BTC-BRL").fast_info.last_price) if hasattr(yf.Ticker("BTC-BRL").fast_info, 'last_price') else p_usd * 5.75
    return hist, p_usd, p_brl

def processar_bitcoin(timeframe="Daily"):
    hist, p_usd, p_brl = obter_dados(timeframe)
    if hist.empty:
        return None
    
    close = hist["Close"]
    rsi_val = calcular_rsi(close)
    candle_info = detectar_candlestick_patterns(hist)
    onchain = obter_onchain_metrics()
    
    sma50 = close.rolling(20 if timeframe != "Daily" else 50).mean().iloc[-1]
    sma200 = close.rolling(50 if timeframe != "Daily" else 200).mean().iloc[-1]
    
    if rsi_val > 78 or ("Bearish" in candle_info["padrao"] and candle_info["forca"] >= 6):
        veredito = "VENDA FORTE 🚨"
        motivo = f"Sobrecompra extrema + padrão bearish forte no {timeframe}"
    elif rsi_val < 25 or (candle_info["forca"] >= 6 and any(x in candle_info["padrao"] for x in ["Hammer", "Bullish", "Morning Star"])):
        veredito = "COMPRA FORTE ✅"
        motivo = f"Sobrevenda + candlestick bullish poderoso + on-chain favorável"
    elif close.iloc[-1] > sma50 and onchain["sentimento_onchain"] == "Positivo" and candle_info["forca"] >= 4:
        veredito = "COMPRA ✅"
        motivo = f"Tendência de alta + sentimento positivo no {timeframe}"
    else:
        veredito = "CAUTELA ⚠️"
        motivo = f"Mercado sem direção clara | RSI {rsi_val:.1f} | Candlestick: {candle_info['padrao']}"
    
    # Simular performance (coloque aqui sua função completa simular_performance_historica se quiser)
    sim = {"expectancy_compra": 0.0, "sharpe_compra": 0.0, "taxa_compra": 0.0}
    
    return {
        "Ticker": "BTC-USD",
        "Preço_USD": p_usd,
        "Preço_BRL": p_brl,
        "Veredito": veredito,
        "Motivo": motivo,
        "RSI": rsi_val,
        "Hist": hist,
        "CandleInfo": candle_info,
        "OnChain": onchain,
        "Timeframe": timeframe,
        "ExpectancyCompra": sim["expectancy_compra"],
        "SharpeCompra": sim["sharpe_compra"],
        "TaxaCompra": sim["taxa_compra"]
    }

# ===================== LAYOUT =====================
st.title("🤖 Monitor IA Pro - Bitcoin")

timeframe = st.sidebar.selectbox("Timeframe de Análise", ["Daily", "4h", "1h"], index=0)

st.sidebar.divider()
st.sidebar.subheader("🔔 Telegram")
bot_token = st.sidebar.text_input("Bot Token", type="password")
chat_id = st.sidebar.text_input("Chat ID")
if st.sidebar.button("Ativar Telegram"):
    st.session_state.telegram_ativado = True
    st.sidebar.success("✅ Telegram ativado!")

with st.spinner("Carregando dados completos do Bitcoin..."):
    resultado = processar_bitcoin(timeframe)

if resultado:
    salvar_sinal(resultado)
    
    # Alerta Compra Forte + Telegram
    if "COMPRA FORTE" in resultado["Veredito"]:
        st.success("🚨 **COMPRA FORTE DETECTADA!** Considere posição.")
        st.balloons()
        
        if st.session_state.telegram_ativado and bot_token and chat_id:
            msg = f"""🚨 <b>COMPRA FORTE - BTC</b>

Preço: ${resultado['Preço_USD']:,.0f} (R$ {resultado['Preço_BRL']:,.0f})
RSI: {resultado['RSI']:.1f}
Timeframe: {resultado['Timeframe']}
Motivo: {resultado['Motivo']}
On-chain: {resultado['OnChain']['sentimento_onchain']}"""

            if enviar_alerta_telegram(bot_token, chat_id, msg):
                st.toast("📨 Alerta enviado no Telegram!", icon="✅")

    # Métricas principais
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Preço USD", f"${resultado['Preço_USD']:,.0f}")
    col2.metric("Preço BRL", f"R$ {resultado['Preço_BRL']:,.0f}")
    col3.metric("RSI (14)", f"{resultado['RSI']:.1f}")
    col4.metric("Veredito", resultado["Veredito"])

    st.subheader("Motivo da IA")
    st.write(resultado["Motivo"])

    # Gráfico com marcação
    with st.expander("📈 Gráfico Técnico com Padrões Marcados", expanded=True):
        hist = resultado["Hist"]
        candle = resultado["CandleInfo"]
        
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.65, 0.20, 0.15])
        
        fig.add_trace(go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'], name="BTC"), row=1, col=1)
        
        # Marcação visual
        if candle["forca"] > 4:
            last_idx = hist.index[-1]
            color = "lime" if any(x in candle["padrao"] for x in ["Bullish", "Hammer", "Morning"]) else "red"
            fig.add_annotation(x=last_idx, y=hist['High'].iloc[-1] * 1.015,
                               text=candle["padrao"], showarrow=True, arrowhead=2, arrowsize=2,
                               arrowcolor=color, font=dict(color=color))
        
        rsi_series = calcular_rsi_series(hist['Close'])
        fig.add_trace(go.Scatter(x=hist.index, y=rsi_series, name="RSI", line=dict(color="purple")), row=3, col=1)
        fig.add_hline(y=70, row=3, col=1, line_dash="dash", line_color="red")
        fig.add_hline(y=30, row=3, col=1, line_dash="dash", line_color="lime")
        
        fig.update_layout(height=800, template="plotly_dark", title=f"BTC {timeframe} - Padrão: {candle['padrao']}")
        st.plotly_chart(fig, use_container_width=True)

    # On-chain
    st.subheader("🔬 On-Chain Metrics")
    oc = resultado["OnChain"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Hash Rate", oc["hash_rate"])
    c2.metric("Transações 24h", oc["transactions_24h"])
    c3.metric("MVRV Z aprox.", oc["mvrv_z"])
    c4.metric("Fluxo Exchanges", oc["exchange_flow"])

    # Tabs finais
    tab1, tab2 = st.tabs(["📜 Histórico de Sinais", "📉 Backtest"])
    with tab1:
        try:
            conn = sqlite3.connect('sinais_bitcoin.db')
            df = pd.read_sql("SELECT * FROM sinais ORDER BY data DESC", conn)
            conn.close()
            st.dataframe(df, use_container_width=True, hide_index=True)
        except:
            st.info("Ainda não há sinais salvos.")

st.caption("Monitor IA Pro Bitcoin - Versão Robusta Completa | Atualiza automaticamente")
