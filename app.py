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
from typing import Dict, Optional, Tuple

# ===================== CONFIGURAÇÕES =====================
st.set_page_config(page_title="Monitor IA Quant Pro - Crypto", layout="wide")
st_autorefresh(interval=300 * 1000, key="data_refresh")  # 5 minutos

if "telegram_ativado" not in st.session_state:
    st.session_state.telegram_ativado = False
if "ultimos_alertas" not in st.session_state:
    st.session_state.ultimos_alertas = set()
if "usar_tradingview" not in st.session_state:
    st.session_state.usar_tradingview = True

DB_NAME = "sinais_crypto.db"
HORIZON_LABEL = 3  # horizonte do alvo supervisionado (3 barras)
TRAIN_WINDOW = 252
TEST_WINDOW = 63
DEFAULT_FEE = 0.0007       # ~0.07%
DEFAULT_SLIPPAGE = 0.0008  # ~0.08%
TARGET_DAILY_VOL = 0.02    # 2% vol alvo
STOP_ATR_MULT = 1.8
TAKE_ATR_MULT = 3.0

# ===================== BANCO DE DADOS =====================
def get_conn():
    return sqlite3.connect(DB_NAME, check_same_thread=False)

def init_db():
    conn = get_conn()
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS sinais (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            data TEXT,
            ticker TEXT,
            timeframe TEXT,
            preco REAL,
            veredito TEXT,
            motivo TEXT,
            score_ia REAL,
            prob_alta REAL,
            regime TEXT,
            expectancy REAL,
            sharpe REAL,
            sortino REAL,
            max_drawdown REAL,
            winrate REAL,
            profit_factor REAL
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS signal_monitor (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            signal_time TEXT,
            ticker TEXT,
            timeframe TEXT,
            price_at_signal REAL,
            score_ia REAL,
            prob_alta REAL,
            regime TEXT,
            veredito TEXT,
            rsi REAL,
            sentiment REAL,
            candle_score REAL,
            onchain_score REAL,
            realized_return_h3 REAL,
            outcome_h3 INTEGER,
            resolved INTEGER DEFAULT 0
        )
    """)

    conn.commit()
    conn.close()

def salvar_sinal(acao: Dict):
    init_db()
    conn = get_conn()
    c = conn.cursor()
    agora = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    c.execute("""
        INSERT INTO sinais (
            data, ticker, timeframe, preco, veredito, motivo, score_ia, prob_alta,
            regime, expectancy, sharpe, sortino, max_drawdown, winrate, profit_factor
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        agora,
        acao["Ticker"],
        acao.get("Timeframe", "1d"),
        acao["Preço"],
        acao["Veredito"],
        acao["Motivo"],
        acao.get("ScoreIA", 0),
        acao.get("ProbAlta", 0),
        acao.get("Regime", "indefinido"),
        acao.get("ExpectancyCompra", 0),
        acao.get("SharpeCompra", 0),
        acao.get("SortinoCompra", 0),
        acao.get("MaxDrawdown", 0),
        acao.get("TaxaCompra", 0),
        acao.get("ProfitFactor", 0),
    ))
    conn.commit()
    conn.close()

def salvar_monitoramento(acao: Dict):
    init_db()
    conn = get_conn()
    c = conn.cursor()
    agora = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    c.execute("""
        INSERT INTO signal_monitor (
            signal_time, ticker, timeframe, price_at_signal, score_ia, prob_alta,
            regime, veredito, rsi, sentiment, candle_score, onchain_score,
            realized_return_h3, outcome_h3, resolved
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, 0)
    """, (
        agora,
        acao["Ticker"],
        acao.get("Timeframe", "1d"),
        acao["Preço"],
        acao.get("ScoreIA", 0),
        acao.get("ProbAlta", 0),
        acao.get("Regime", "indefinido"),
        acao.get("Veredito", ""),
        acao.get("RSI", 50),
        acao.get("SentimentoNoticias", 0),
        acao.get("CandleInfo", {}).get("score", 0),
        acao.get("OnChain", {}).get("score_total", 0),
    ))
    conn.commit()
    conn.close()

def atualizar_monitoramento_com_resultados(ticker: str, hist: pd.DataFrame):
    if hist is None or hist.empty:
        return

    conn = get_conn()
    try:
        df_mon = pd.read_sql(
            "SELECT * FROM signal_monitor WHERE ticker = ? AND resolved = 0 ORDER BY signal_time ASC",
            conn,
            params=(ticker,)
        )
        if df_mon.empty:
            conn.close()
            return

        hist_local = hist.copy()
        hist_local.index = pd.to_datetime(hist_local.index)
        close = hist_local["Close"].copy()

        updates = []
        for _, row in df_mon.iterrows():
            signal_time = pd.to_datetime(row["signal_time"])
            future_slice = close[close.index >= signal_time]

            if len(future_slice) > HORIZON_LABEL:
                entry = float(row["price_at_signal"])
                future_price = float(future_slice.iloc[HORIZON_LABEL])
                realized = ((future_price / entry) - 1.0) * 100.0
                outcome = 1 if realized > 0 else 0
                updates.append((realized, outcome, 1, int(row["id"])))

        if updates:
            cur = conn.cursor()
            cur.executemany("""
                UPDATE signal_monitor
                SET realized_return_h3 = ?, outcome_h3 = ?, resolved = ?
                WHERE id = ?
            """, updates)
            conn.commit()
    finally:
        conn.close()

# ===================== UTILITÁRIAS =====================
def safe_float(value, default=0.0):
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default

def sigmoid(x):
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))

def annualization_factor(interval: str) -> float:
    if interval == "1h":
        return 24 * 365
    if interval == "4h":
        return 6 * 365
    return 252

def format_percent(x: float) -> str:
    return f"{x:.2f}%"

# ===================== TELEGRAM =====================
def enviar_alerta_telegram(bot_token: str, chat_id: str, mensagem: str) -> bool:
    if not bot_token or not chat_id:
        return False
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {"chat_id": chat_id, "text": mensagem, "parse_mode": "HTML"}
        r = requests.post(url, json=payload, timeout=12)
        return r.status_code == 200
    except Exception:
        return False

# ===================== INDICADORES =====================
def calcular_rsi_series(close: pd.Series, window: int = 14) -> pd.Series:
    if len(close) < window:
        return pd.Series([50.0] * len(close), index=close.index)

    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)

def calcular_rsi(data: pd.Series, window: int = 14):
    if len(data) < window:
        return 50.0
    return safe_float(calcular_rsi_series(data, window).iloc[-1], 50.0)

def calcular_atr(hist: pd.DataFrame, window: int = 14) -> pd.Series:
    if hist is None or hist.empty:
        return pd.Series(dtype=float)

    high = hist["High"]
    low = hist["Low"]
    close = hist["Close"]

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    return atr.bfill().fillna(0.0)

def detectar_candlestick_patterns(hist: pd.DataFrame) -> dict:
    if hist is None or hist.empty or len(hist) < 5:
        return {"padrao": "Dados insuficientes", "forca": 0, "direcao": "neutro", "score": 0}

    df = hist.tail(5).copy()
    o = df["Open"]
    h = df["High"]
    l = df["Low"]
    c = df["Close"]

    i = -1
    body = abs(c.iloc[i] - o.iloc[i])
    candle_range = max(h.iloc[i] - l.iloc[i], 1e-9)
    upper_shadow = h.iloc[i] - max(c.iloc[i], o.iloc[i])
    lower_shadow = min(c.iloc[i], o.iloc[i]) - l.iloc[i]

    patterns = []
    score = 0
    direction = "neutro"

    if lower_shadow > 2.0 * max(body, 1e-9) and upper_shadow < max(body, 1e-9) and body / candle_range < 0.4:
        patterns.append("Hammer")
        score += 16
        direction = "bullish"

    if upper_shadow > 2.0 * max(body, 1e-9) and lower_shadow < max(body, 1e-9) and body / candle_range < 0.4:
        patterns.append("Shooting Star")
        score -= 16
        direction = "bearish"

    if len(df) >= 2:
        prev_o = o.iloc[i - 1]
        prev_c = c.iloc[i - 1]

        bullish_engulfing = (
            prev_c < prev_o and
            c.iloc[i] > o.iloc[i] and
            c.iloc[i] >= prev_o and
            o.iloc[i] <= prev_c
        )
        bearish_engulfing = (
            prev_c > prev_o and
            c.iloc[i] < o.iloc[i] and
            c.iloc[i] <= prev_o and
            o.iloc[i] >= prev_c
        )

        if bullish_engulfing:
            patterns.append("Bullish Engulfing")
            score += 22
            direction = "bullish"

        if bearish_engulfing:
            patterns.append("Bearish Engulfing")
            score -= 22
            direction = "bearish"

    if len(df) >= 3:
        last3 = df.tail(3)
        if (last3["Close"] > last3["Open"]).all() and last3["Close"].is_monotonic_increasing:
            patterns.append("Three Bullish Candles")
            score += 10
            direction = "bullish"
        elif (last3["Close"] < last3["Open"]).all() and last3["Close"].is_monotonic_decreasing:
            patterns.append("Three Bearish Candles")
            score -= 10
            direction = "bearish"

    if not patterns:
        patterns = ["Sem padrão forte"]

    return {
        "padrao": " + ".join(patterns),
        "forca": min(abs(score), 30),
        "direcao": direction,
        "score": score
    }

def classificar_regime(hist: pd.DataFrame) -> Tuple[str, float]:
    if hist is None or hist.empty or len(hist) < 220:
        return "indefinido", 0.0

    close = hist["Close"]
    sma50 = close.rolling(50).mean().iloc[-1]
    sma200 = close.rolling(200).mean().iloc[-1]
    price = close.iloc[-1]
    ret_20 = close.pct_change(20).iloc[-1]
    vol_20 = close.pct_change().rolling(20).std().iloc[-1]

    regime_score = 0.0
    regime_score += 1.0 if price > sma200 else -1.0
    regime_score += 1.0 if sma50 > sma200 else -1.0

    if ret_20 > 0.05:
        regime_score += 0.5
    elif ret_20 < -0.05:
        regime_score -= 0.5

    if vol_20 > 0.06:
        regime_score *= 0.8

    if regime_score >= 1.5:
        return "bull", regime_score
    if regime_score <= -1.5:
        return "bear", regime_score
    return "sideways", regime_score

# ===================== ON-CHAIN =====================
@st.cache_data(ttl=600, show_spinner=False)
def obter_onchain_publico(ticker: str) -> dict:
    base = {
        "score": 0.0,
        "sentimento": "Neutro",
        "hash_rate": "N/D",
        "transactions_24h": "N/D",
        "fonte": "Pública"
    }

    if ticker != "BTC-USD":
        base["fonte"] = "Neutro para ETH (sem API pública robusta configurada)"
        return base

    try:
        r = requests.get("https://api.blockchain.info/stats", timeout=10)
        if r.status_code != 200:
            return base

        data = r.json()
        hash_rate = safe_float(data.get("hash_rate"), 0.0) / 1e9
        tx_24h = safe_float(data.get("n_tx"), 0.0)

        score = 0.0
        if hash_rate > 400:
            score += 7
        elif hash_rate > 250:
            score += 3
        else:
            score -= 1

        if tx_24h > 300000:
            score += 7
        elif tx_24h > 200000:
            score += 3
        else:
            score -= 1

        sentimento = "Positivo" if score >= 7 else "Neutro"
        if score <= -1:
            sentimento = "Cauteloso"

        return {
            "score": score,
            "sentimento": sentimento,
            "hash_rate": f"{hash_rate:.0f} EH/s",
            "transactions_24h": f"{int(tx_24h):,}".replace(",", "."),
            "fonte": "Blockchain.com"
        }
    except Exception:
        return base

@st.cache_data(ttl=1800, show_spinner=False)
def obter_glassnode_basico(ticker: str, api_key: str = "") -> dict:
    base = {
        "score": 0.0,
        "sentimento": "Neutro",
        "active_addresses": "N/D",
        "sopr": "N/D",
        "mvrv": "N/D",
        "fonte": "Glassnode (indisponível ou sem acesso API)"
    }

    if not api_key:
        return base

    asset = "BTC" if ticker == "BTC-USD" else "ETH"
    headers = {"X-Api-Key": api_key}
    score = 0.0

    try:
        r1 = requests.get(
            "https://api.glassnode.com/v1/metrics/addresses/active_count",
            params={"a": asset, "i": "24h"},
            headers=headers,
            timeout=12
        )
        if r1.status_code == 200 and r1.text:
            data = r1.json()
            if isinstance(data, list) and data:
                aa = safe_float(data[-1].get("v"), 0.0)
                base["active_addresses"] = f"{int(aa):,}".replace(",", ".")
                if aa > 0:
                    score += 4

        r2 = requests.get(
            "https://api.glassnode.com/v1/metrics/indicators/sopr",
            params={"a": asset, "i": "24h"},
            headers=headers,
            timeout=12
        )
        if r2.status_code == 200 and r2.text:
            data = r2.json()
            if isinstance(data, list) and data:
                sopr = safe_float(data[-1].get("v"), 1.0)
                base["sopr"] = f"{sopr:.3f}"
                if sopr > 1.0:
                    score += 4
                else:
                    score -= 3

        r3 = requests.get(
            "https://api.glassnode.com/v1/metrics/market/mvrv",
            params={"a": asset, "i": "24h"},
            headers=headers,
            timeout=12
        )
        if r3.status_code == 200 and r3.text:
            data = r3.json()
            if isinstance(data, list) and data:
                mvrv = safe_float(data[-1].get("v"), 1.0)
                base["mvrv"] = f"{mvrv:.3f}"
                if mvrv < 1.2:
                    score += 3
                elif mvrv > 2.4:
                    score -= 4

        base["score"] = score
        base["sentimento"] = "Positivo" if score > 3 else ("Cauteloso" if score < -2 else "Neutro")
        base["fonte"] = "Glassnode"
        return base
    except Exception:
        return base

# ===================== NOTÍCIAS =====================
def pontuar_noticias(ticker: str):
    noticias_texto = ""
    lista_links = []

    queries = {
        "BTC-USD": "bitcoin OR btc OR etf bitcoin OR halving",
        "ETH-USD": "ethereum OR eth OR ether OR etf ethereum"
    }

    try:
        feed = feedparser.parse(f"https://news.google.com/rss/search?q={queries.get(ticker, ticker)}&hl=pt-BR")
        for entry in feed.entries[:8]:
            titulo = getattr(entry, "title", "")
            link = getattr(entry, "link", "")
            if titulo:
                noticias_texto += titulo.lower() + " "
                lista_links.append({"titulo": titulo, "link": link})
    except Exception:
        pass

    pos = [
        "alta", "bull", "compra", "subiu", "rally", "high", "buy", "aprovação",
        "recorde", "inflow", "institucional", "acumulação"
    ]
    neg = [
        "queda", "bear", "venda", "caiu", "crash", "sell", "hack",
        "ban", "outflow", "fraude", "liquidação"
    ]

    score_p = sum(noticias_texto.count(w) for w in pos)
    score_n = sum(noticias_texto.count(w) for w in neg)

    return score_p, score_n, lista_links

# ===================== DADOS =====================
@st.cache_data(ttl=300, show_spinner=False)
def obter_dados_crypto(ticker: str, timeframe: str):
    interval_map = {"1d": "1d", "4h": "4h", "1h": "1h"}
    period_map = {"1d": "5y", "4h": "730d", "1h": "180d"}

    ativo = yf.Ticker(ticker)
    hist = ativo.history(
        period=period_map[timeframe],
        interval=interval_map[timeframe],
        auto_adjust=True
    )

    try:
        info = ativo.info
    except Exception:
        info = {}

    return info, hist

@st.cache_data(ttl=1800, show_spinner=False)
def obter_macro():
    macro = {"Dolar": 5.70, "Bitcoin_USD": 0.0, "Ethereum_USD": 0.0}
    try:
        macro["Dolar"] = safe_float(yf.Ticker("USDBRL=X").fast_info.get("last_price"), 5.70)
    except Exception:
        pass
    try:
        macro["Bitcoin_USD"] = safe_float(yf.Ticker("BTC-USD").fast_info.get("last_price"), 0.0)
    except Exception:
        pass
    try:
        macro["Ethereum_USD"] = safe_float(yf.Ticker("ETH-USD").fast_info.get("last_price"), 0.0)
    except Exception:
        pass
    return macro

# ===================== FEATURES =====================
def construir_features(hist: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    df = hist.copy()
    df["retorno"] = df["Close"].pct_change()
    df["rsi14"] = calcular_rsi_series(df["Close"], 14)

    df["sma20"] = df["Close"].rolling(20).mean()
    df["sma50"] = df["Close"].rolling(50).mean()
    df["sma200"] = df["Close"].rolling(200).mean()

    exp12 = df["Close"].ewm(span=12, adjust=False).mean()
    exp26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["macd"] = exp12 - exp26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    df["atr14"] = calcular_atr(df, 14)
    df["atr_pct"] = df["atr14"] / df["Close"].replace(0, np.nan)

    df["mom_3"] = df["Close"].pct_change(3)
    df["mom_7"] = df["Close"].pct_change(7)
    df["mom_14"] = df["Close"].pct_change(14)
    df["dist_sma20"] = (df["Close"] / df["sma20"]) - 1
    df["dist_sma50"] = (df["Close"] / df["sma50"]) - 1
    df["dist_sma200"] = (df["Close"] / df["sma200"]) - 1

    df["vol_20"] = df["retorno"].rolling(20).std()
    df["future_return_h"] = df["Close"].shift(-HORIZON_LABEL) / df["Close"] - 1
    df["target_up"] = (df["future_return_h"] > 0).astype(int)

    return df

# ===================== MODELO SUPERVISIONADO =====================
def treinar_regressao_logistica_numpy(X: np.ndarray, y: np.ndarray, lr: float = 0.05, epochs: int = 300):
    if X is None or y is None or len(X) == 0 or len(y) == 0:
        return None

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(np.unique(y)) < 2:
        return None

    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma == 0] = 1.0

    Xn = (X - mu) / sigma
    Xb = np.column_stack([np.ones(len(Xn)), Xn])
    w = np.zeros(Xb.shape[1], dtype=float)

    for _ in range(epochs):
        preds = sigmoid(Xb @ w)
        grad = (Xb.T @ (preds - y)) / len(y)
        w -= lr * grad

    return {"weights": w, "mu": mu, "sigma": sigma}

def prever_prob_regressao(modelo: dict, x_row: np.ndarray) -> float:
    if not modelo:
        return 0.5
    x_row = np.asarray(x_row, dtype=float)
    xn = (x_row - modelo["mu"]) / modelo["sigma"]
    xb = np.concatenate([[1.0], xn])
    return float(np.clip(sigmoid(xb @ modelo["weights"]), 0.0, 1.0))

def probabilidade_modelo_ml(df_feat: pd.DataFrame) -> dict:
    result = {"prob_up": 0.5, "confianca_ml": 0.0}

    cols = [
        "rsi14", "macd_hist", "dist_sma20", "dist_sma50", "dist_sma200",
        "mom_3", "mom_7", "mom_14", "atr_pct", "vol_20"
    ]

    df = df_feat.dropna(subset=cols + ["target_up"]).copy()
    if len(df) < 260:
        return result

    X = df[cols].values
    y = df["target_up"].values.astype(float)

    train_X = X[:-1]
    train_y = y[:-1]
    last_x = X[-1]

    modelo = treinar_regressao_logistica_numpy(train_X, train_y, lr=0.05, epochs=300)
    if not modelo:
        return result

    prob = prever_prob_regressao(modelo, last_x)
    confianca = abs(prob - 0.5) * 200.0

    result["prob_up"] = float(prob)
    result["confianca_ml"] = float(np.clip(confianca, 0.0, 100.0))
    return result

# ===================== WALK-FORWARD BACKTEST =====================
def walk_forward_backtest(df_feat: pd.DataFrame, timeframe: str) -> dict:
    fallback = {
        "taxa_compra": 0.0,
        "expectancy_compra": 0.0,
        "sharpe_compra": 0.0,
        "sortino_compra": 0.0,
        "max_drawdown": 0.0,
        "qtd_compra": 0,
        "profit_factor": 0.0,
        "retorno_acumulado": 0.0,
        "equity_curve": pd.Series(dtype=float),
        "hit_prob_bucket": 0.0,
        "avg_prob": 0.5
    }

    cols = [
        "rsi14", "macd_hist", "dist_sma20", "dist_sma50", "dist_sma200",
        "mom_3", "mom_7", "mom_14", "atr_pct", "vol_20"
    ]

    df = df_feat.dropna(subset=cols + ["target_up", "retorno", "atr14"]).copy()
    if len(df) < TRAIN_WINDOW + TEST_WINDOW + 20:
        return fallback

    ann = annualization_factor(timeframe)
    records = []
    n = len(df)

    start = TRAIN_WINDOW
    while start < n - 1:
        train_start = max(0, start - TRAIN_WINDOW)
        test_end = min(start + TEST_WINDOW, n - 1)

        train = df.iloc[train_start:start].copy()
        test = df.iloc[start:test_end].copy()

        if len(train) < 120 or len(test) == 0:
            start += TEST_WINDOW
            continue

        X_train = train[cols].values
        y_train = train["target_up"].values.astype(float)
        model = treinar_regressao_logistica_numpy(X_train, y_train, lr=0.05, epochs=250)

        prev_position = 0.0

        for idx, row in test.iterrows():
            x = row[cols].values
            prob = prever_prob_regressao(model, x) if model else 0.5

            hist_until = df.loc[:idx].copy()
            regime, _ = classificar_regime(hist_until)

            long_signal = False
            if regime == "bull" and prob >= 0.56 and 40 <= row["rsi14"] <= 72:
                long_signal = True
            elif regime == "sideways" and prob >= 0.60 and row["rsi14"] < 65 and row["macd_hist"] > 0:
                long_signal = True
            elif regime == "bear":
                long_signal = prob >= 0.64 and row["rsi14"] < 40 and row["dist_sma20"] > 0

            realized_vol = safe_float(row["vol_20"], 0.01)
            target_pos = 0.0
            if long_signal:
                raw_size = TARGET_DAILY_VOL / max(realized_vol, 1e-4)
                target_pos = float(np.clip(raw_size, 0.0, 1.0))

            turnover = abs(target_pos - prev_position)
            trading_cost = turnover * (DEFAULT_FEE + DEFAULT_SLIPPAGE)

            ret = safe_float(row["retorno"], 0.0)
            atr_pct = safe_float(row["atr14"] / row["Close"], 0.0)
            stop_cap = STOP_ATR_MULT * atr_pct
            take_cap = TAKE_ATR_MULT * atr_pct

            gross = prev_position * ret
            if stop_cap > 0:
                gross = max(gross, -stop_cap)
            if take_cap > 0:
                gross = min(gross, take_cap)

            net = gross - trading_cost

            records.append({
                "time": idx,
                "prob": prob,
                "regime": regime,
                "position": prev_position,
                "target_position": target_pos,
                "retorno_estrategia": net,
                "trade": 1 if turnover > 0 else 0
            })

            prev_position = target_pos

        start += TEST_WINDOW

    if not records:
        return fallback

    bt = pd.DataFrame(records).set_index("time")
    bt["equity"] = (1.0 + bt["retorno_estrategia"].fillna(0.0)).cumprod()
    bt["peak"] = bt["equity"].cummax()
    bt["drawdown"] = (bt["equity"] - bt["peak"]) / bt["peak"].replace(0, np.nan)

    ganhos = bt.loc[bt["retorno_estrategia"] > 0, "retorno_estrategia"]
    perdas = bt.loc[bt["retorno_estrategia"] < 0, "retorno_estrategia"]

    winrate = len(ganhos) / max(len(ganhos) + len(perdas), 1)
    media_gain = ganhos.mean() if len(ganhos) > 0 else 0.0
    media_loss = perdas.mean() if len(perdas) > 0 else 0.0
    expectancy = (winrate * media_gain) + ((1 - winrate) * media_loss)

    std = bt["retorno_estrategia"].std()
    sharpe = (bt["retorno_estrategia"].mean() / std) * np.sqrt(ann) if pd.notna(std) and std > 0 else 0.0

    downside = bt.loc[bt["retorno_estrategia"] < 0, "retorno_estrategia"]
    downside_std = downside.std()
    sortino = (bt["retorno_estrategia"].mean() / downside_std) * np.sqrt(ann) if pd.notna(downside_std) and downside_std > 0 else 0.0

    gross_profit = ganhos.sum() if len(ganhos) > 0 else 0.0
    gross_loss = abs(perdas.sum()) if len(perdas) > 0 else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0.0)

    retorno_acumulado = (bt["equity"].iloc[-1] - 1.0) * 100.0
    max_dd = bt["drawdown"].min() * 100.0 if not bt.empty else 0.0
    qtd_compra = int((bt["target_position"] > 0).astype(int).diff().fillna(0).gt(0).sum())
    avg_prob = float(bt["prob"].mean()) if "prob" in bt else 0.5

    high_prob = bt[bt["prob"] >= 0.60]
    hit_prob_bucket = 0.0
    if not high_prob.empty:
        hit_prob_bucket = float((high_prob["retorno_estrategia"] > 0).mean() * 100.0)

    return {
        "taxa_compra": float(winrate * 100.0),
        "expectancy_compra": float(expectancy * 100.0),
        "sharpe_compra": float(sharpe),
        "sortino_compra": float(sortino),
        "max_drawdown": float(max_dd),
        "qtd_compra": int(qtd_compra),
        "profit_factor": float(profit_factor),
        "retorno_acumulado": float(retorno_acumulado),
        "equity_curve": bt["equity"].copy(),
        "hit_prob_bucket": float(hit_prob_bucket),
        "avg_prob": float(avg_prob)
    }

# ===================== ENSEMBLE =====================
def modelo_ensemble_score(
    ticker: str,
    hist: pd.DataFrame,
    rsi_val: float,
    macd: pd.Series,
    sinal_macd: pd.Series,
    score_p: int,
    score_n: int,
    candle_info: dict,
    onchain_public: dict,
    onchain_glassnode: dict,
    prob_ml: float,
    regime: str,
    sim: dict
) -> Tuple[float, list]:
    score = 50.0
    componentes = []

    close = hist["Close"]
    price = close.iloc[-1]
    sma20 = close.rolling(20).mean().iloc[-1] if len(close) >= 20 else price
    sma50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else price
    sma200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else price

    ml_score = (prob_ml - 0.5) * 90.0
    score += ml_score
    componentes.append(f"ML {prob_ml*100:.1f}%")

    if regime == "bull":
        score += 8
        componentes.append("Regime bull")
    elif regime == "bear":
        score -= 10
        componentes.append("Regime bear")
    else:
        componentes.append("Regime lateral")

    if rsi_val < 30:
        score += 10
        componentes.append("RSI sobrevenda")
    elif rsi_val < 40:
        score += 5
    elif rsi_val > 78:
        score -= 12
        componentes.append("RSI sobrecompra extrema")
    elif rsi_val > 68:
        score -= 6

    if len(macd) > 1 and len(sinal_macd) > 1:
        if macd.iloc[-1] > sinal_macd.iloc[-1] and macd.iloc[-2] <= sinal_macd.iloc[-2]:
            score += 9
            componentes.append("MACD cruzou para cima")
        elif macd.iloc[-1] < sinal_macd.iloc[-1] and macd.iloc[-2] >= sinal_macd.iloc[-2]:
            score -= 9
            componentes.append("MACD cruzou para baixo")
        elif macd.iloc[-1] > sinal_macd.iloc[-1]:
            score += 4
        else:
            score -= 4

    if price > sma20 > sma50:
        score += 7
        componentes.append("Tendência curta positiva")
    elif price < sma20 < sma50:
        score -= 7
        componentes.append("Tendência curta negativa")

    if price > sma200:
        score += 6
        componentes.append("Acima da SMA200")
    else:
        score -= 6
        componentes.append("Abaixo da SMA200")

    score += candle_info.get("score", 0) * 0.45
    if candle_info.get("padrao") != "Sem padrão forte":
        componentes.append(candle_info.get("padrao", ""))

    sentimento = score_p - score_n
    score += np.clip(sentimento * 2.8, -10, 10)
    if sentimento > 0:
        componentes.append("Notícias positivas")
    elif sentimento < 0:
        componentes.append("Notícias negativas")

    score += np.clip(onchain_public.get("score", 0.0), -8, 8)
    if ticker == "BTC-USD":
        componentes.append(f"On-chain público {onchain_public.get('sentimento', 'Neutro')}")

    score += np.clip(onchain_glassnode.get("score", 0.0), -8, 8)
    if onchain_glassnode.get("fonte") == "Glassnode":
        componentes.append(f"Glassnode {onchain_glassnode.get('sentimento', 'Neutro')}")

    sharpe = sim.get("sharpe_compra", 0.0)
    expectancy = sim.get("expectancy_compra", 0.0)
    profit_factor = sim.get("profit_factor", 0.0)
    max_dd = sim.get("max_drawdown", 0.0)

    if sharpe > 1.0:
        score += 5
        componentes.append("Sharpe forte")
    elif sharpe < 0:
        score -= 5

    if expectancy > 0.10:
        score += 4
        componentes.append("Expectancy positiva")
    elif expectancy < 0:
        score -= 4

    if profit_factor > 1.20:
        score += 4
    elif 0 < profit_factor < 0.95:
        score -= 4

    if max_dd < -35:
        score -= 5
        componentes.append("Drawdown alto")

    score = float(np.clip(score, 0.0, 100.0))
    return round(score, 1), componentes

def classificar_veredito(score_ia: float, prob_up: float, regime: str, rsi: float):
    if score_ia >= 82 and prob_up >= 0.62 and regime != "bear" and rsi < 74:
        return "COMPRA FORTE 🟢", "success"
    if score_ia >= 68 and prob_up >= 0.56 and regime != "bear":
        return "COMPRA 🟢", "success"
    if score_ia <= 18 and prob_up <= 0.38 and regime == "bear":
        return "VENDA FORTE 🔴", "error"
    if score_ia <= 32 and prob_up <= 0.44:
        return "VENDA 🔴", "error"
    return "NEUTRO ⚖️", "warning"

def gerar_motivo(score_ia: float, componentes: list, candle_info: dict, prob_up: float, regime: str):
    top = ", ".join(componentes[:5]) if componentes else "Sem confluência forte"
    return (
        f"Score IA {score_ia}/100 | Probabilidade estimada de alta: {prob_up*100:.1f}% | "
        f"Regime: {regime}. Confluências: {top}. Candle: {candle_info.get('padrao', 'Sem padrão forte')}."
    )

def gerar_alerta_inteligente(acao: Dict) -> Optional[dict]:
    score = acao.get("ScoreIA", 50)
    prob = acao.get("ProbAlta", 0.5)
    veredito = acao.get("Veredito", "NEUTRO ⚖️")
    ticker = acao.get("Ticker", "")
    regime = acao.get("Regime", "indefinido")

    if veredito == "COMPRA FORTE 🟢" and prob >= 0.65 and regime == "bull":
        return {
            "tipo": "entrada_agressiva",
            "titulo": f"🚨 Entrada agressiva detectada em {ticker}",
            "mensagem": f"Score {score}/100 | Prob. alta {prob*100:.1f}% | Regime {regime}."
        }

    if veredito == "COMPRA 🟢" and score >= 70:
        return {
            "tipo": "entrada_moderada",
            "titulo": f"📈 Entrada moderada em {ticker}",
            "mensagem": f"Confluência favorável com score {score}/100."
        }

    if veredito == "VENDA FORTE 🔴":
        return {
            "tipo": "risco_alto",
            "titulo": f"🛑 Risco elevado em {ticker}",
            "mensagem": f"Sinal defensivo com score {score}/100."
        }

    if 45 <= score <= 55:
        return {
            "tipo": "mercado_neutro",
            "titulo": f"⏸️ Mercado sem vantagem estatística clara em {ticker}",
            "mensagem": "Melhor aguardar confirmação antes de agir."
        }

    return None

# ===================== MONITORAMENTO =====================
def resumo_monitoramento_modelo(ticker: str):
    init_db()
    conn = get_conn()
    try:
        df = pd.read_sql("""
            SELECT *
            FROM signal_monitor
            WHERE ticker = ? AND resolved = 1
            ORDER BY signal_time DESC
        """, conn, params=(ticker,))
    finally:
        conn.close()

    if df.empty:
        return None, None

    df["score_bucket"] = pd.cut(
        df["score_ia"],
        bins=[0, 40, 55, 70, 85, 100],
        labels=["0-40", "41-55", "56-70", "71-85", "86-100"],
        include_lowest=True
    )

    calib = df.groupby("score_bucket", observed=False).agg(
        sinais=("id", "count"),
        taxa_acerto=("outcome_h3", "mean"),
        retorno_medio=("realized_return_h3", "mean")
    ).reset_index()

    calib["taxa_acerto"] = calib["taxa_acerto"].fillna(0) * 100.0
    calib["retorno_medio"] = calib["retorno_medio"].fillna(0.0)

    return df, calib

# ===================== PROCESSAMENTO CENTRAL =====================
def processar_crypto(ticker: str, timeframe: str, glassnode_api_key: str = ""):
    info, hist = obter_dados_crypto(ticker, timeframe)
    if hist is None or hist.empty:
        return None

    hist = hist.copy()
    close = hist["Close"]

    df_feat = construir_features(hist, timeframe)
    rsi_val = calcular_rsi(close)
    candle_info = detectar_candlestick_patterns(hist)
    regime, regime_score = classificar_regime(hist)

    exp12 = close.ewm(span=12, adjust=False).mean()
    exp26 = close.ewm(span=26, adjust=False).mean()
    macd = exp12 - exp26
    sinal_macd = macd.ewm(span=9, adjust=False).mean()

    score_p, score_n, lista_links = pontuar_noticias(ticker)
    onchain_public = obter_onchain_publico(ticker)
    onchain_glassnode = obter_glassnode_basico(ticker, glassnode_api_key)

    sim = walk_forward_backtest(df_feat, timeframe)
    ml_info = probabilidade_modelo_ml(df_feat)

    prob_up = ml_info.get("prob_up", 0.5)
    p_atual = safe_float(close.iloc[-1], 0.0)

    score_ia, componentes = modelo_ensemble_score(
        ticker=ticker,
        hist=hist,
        rsi_val=rsi_val,
        macd=macd,
        sinal_macd=sinal_macd,
        score_p=score_p,
        score_n=score_n,
        candle_info=candle_info,
        onchain_public=onchain_public,
        onchain_glassnode=onchain_glassnode,
        prob_ml=prob_up,
        regime=regime,
        sim=sim
    )

    veredito, cor = classificar_veredito(score_ia, prob_up, regime, rsi_val)
    motivo = gerar_motivo(score_ia, componentes, candle_info, prob_up, regime)
    alerta = gerar_alerta_inteligente({
        "Ticker": ticker,
        "ScoreIA": score_ia,
        "ProbAlta": prob_up,
        "Veredito": veredito,
        "Regime": regime
    })

    score_total_onchain = onchain_public.get("score", 0.0) + onchain_glassnode.get("score", 0.0)

    return {
        "Ticker": ticker,
        "Timeframe": timeframe,
        "Preço": p_atual,
        "Veredito": veredito,
        "Cor": cor,
        "Motivo": motivo,
        "RSI": rsi_val,
        "Hist": hist,
        "Features": df_feat,
        "Links": lista_links,
        "ScoreIA": score_ia,
        "ProbAlta": prob_up,
        "ConfiancaML": ml_info.get("confianca_ml", 0.0),
        "CandleInfo": candle_info,
        "OnChain": {
            "score_total": score_total_onchain,
            "publico": onchain_public,
            "glassnode": onchain_glassnode
        },
        "Info": info,
        "Regime": regime,
        "RegimeScore": regime_score,
        "TaxaCompra": sim.get("taxa_compra", 0.0),
        "ExpectancyCompra": sim.get("expectancy_compra", 0.0),
        "SharpeCompra": sim.get("sharpe_compra", 0.0),
        "SortinoCompra": sim.get("sortino_compra", 0.0),
        "MaxDrawdown": sim.get("max_drawdown", 0.0),
        "QtdCompra": sim.get("qtd_compra", 0),
        "ProfitFactor": sim.get("profit_factor", 0.0),
        "RetornoAcumulado": sim.get("retorno_acumulado", 0.0),
        "EquityCurve": sim.get("equity_curve", pd.Series(dtype=float)),
        "HitProbBucket": sim.get("hit_prob_bucket", 0.0),
        "AvgProbBacktest": sim.get("avg_prob", 0.5),
        "Alerta": alerta,
        "SentimentoNoticias": score_p - score_n
    }

# ===================== SIDEBAR =====================
init_db()

st.sidebar.title("₿ Monitor IA Quant Pro")
crypto_selecionado = st.sidebar.selectbox("Escolha o ativo:", ["BTC-USD", "ETH-USD"])
timeframe = st.sidebar.selectbox("Timeframe:", ["1d", "4h", "1h"], index=0)

st.sidebar.subheader("📊 Câmbio em Tempo Real")
cambio = obter_macro()
st.sidebar.metric("Bitcoin (USD)", f"${cambio['Bitcoin_USD']:,.0f}" if cambio["Bitcoin_USD"] else "N/D")
st.sidebar.metric("Ethereum (USD)", f"${cambio['Ethereum_USD']:,.0f}" if cambio["Ethereum_USD"] else "N/D")
st.sidebar.metric("Dólar (R$)", f"R$ {cambio['Dolar']:.2f}")

st.sidebar.divider()
st.sidebar.subheader("📺 TradingView")
st.session_state.usar_tradingview = st.sidebar.checkbox("Usar gráfico TradingView", value=True)

st.sidebar.divider()
st.sidebar.subheader("🔬 Glassnode (opcional)")
glassnode_api_key = st.sidebar.text_input(
    "Glassnode API Key",
    type="password",
    help="Opcional. Se não houver acesso API, o sistema usa fallback neutro."
)

st.sidebar.divider()
st.sidebar.subheader("📨 Telegram")
bot_token = st.sidebar.text_input("Bot Token", type="password")
chat_id = st.sidebar.text_input("Chat ID")
if st.sidebar.button("Ativar alertas Telegram"):
    st.session_state.telegram_ativado = True
    st.sidebar.success("Telegram ativado!")

estrategia_ativa = st.sidebar.selectbox(
    "Modo:",
    [
        "Quant Walk-Forward + ML",
        "Momentum + Sentimento",
        "Position Trading"
    ]
)

# ===================== CABEÇALHO =====================
st.title(f"🤖 Monitor IA Quant Pro - {crypto_selecionado}")
st.caption(f"Atualização: {time.strftime('%H:%M:%S')} | Local: Porto Alegre/RS | Timeframe: {timeframe}")

# ===================== PROCESSAMENTO =====================
with st.spinner("📡 Baixando dados e calculando modelo..."):
    resultado = processar_crypto(crypto_selecionado, timeframe, glassnode_api_key)

if resultado:
    atualizar_monitoramento_com_resultados(resultado["Ticker"], resultado["Hist"])

dados_vencedoras = [resultado] if resultado else []

# ===================== ALERTA ÚNICO =====================
if resultado and resultado.get("Alerta"):
    chave_alerta = (
        resultado["Ticker"],
        resultado["Timeframe"],
        resultado["Alerta"]["tipo"],
        round(resultado["ScoreIA"], 1),
        round(resultado["ProbAlta"], 2),
    )
    if chave_alerta not in st.session_state.ultimos_alertas:
        st.session_state.ultimos_alertas.add(chave_alerta)
        st.toast(resultado["Alerta"]["titulo"], icon="🚨")

        if st.session_state.telegram_ativado and bot_token and chat_id:
            msg = (
                f"<b>{resultado['Alerta']['titulo']}</b>\n\n"
                f"Ativo: {resultado['Ticker']}\n"
                f"Timeframe: {resultado['Timeframe']}\n"
                f"Preço: ${resultado['Preço']:,.2f}\n"
                f"Veredito: {resultado['Veredito']}\n"
                f"Score IA: {resultado['ScoreIA']}/100\n"
                f"Prob. Alta: {resultado['ProbAlta']*100:.1f}%\n"
                f"Regime: {resultado['Regime']}\n"
                f"RSI: {resultado['RSI']:.1f}\n"
                f"Motivo: {resultado['Motivo']}"
            )
            ok = enviar_alerta_telegram(bot_token, chat_id, msg)
            if ok:
                st.toast("Alerta enviado ao Telegram", icon="✅")

        salvar_monitoramento(resultado)
        salvar_sinal(resultado)

# ===================== TABS =====================
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 Overview", "📈 Gráfico Técnico", "📉 Análise", "📜 Backtest", "🧠 Monitoramento Modelo"]
)

# ===================== TAB 1 - OVERVIEW =====================
with tab1:
    if dados_vencedoras:
        acao = dados_vencedoras[0]
        st.subheader(f"🏆 Análise Atual - Estratégia: {estrategia_ativa}")

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Preço Atual (USD)", f"${acao['Preço']:,.2f}")
        col2.metric("RSI (14)", f"{acao['RSI']:.1f}")
        col3.metric("Score IA", f"{acao['ScoreIA']:.1f}/100")
        col4.metric("Prob. Alta", f"{acao['ProbAlta']*100:.1f}%")
        col5.metric("Veredito IA", acao["Veredito"])

        st.markdown(f"**Motivo:** {acao['Motivo']}")

        if acao.get("Alerta"):
            st.info(f"**{acao['Alerta']['titulo']}** — {acao['Alerta']['mensagem']}")

        resumo = pd.DataFrame([{
            "Preço": acao["Preço"],
            "WinRate": acao["TaxaCompra"],
            "Expectancy": acao["ExpectancyCompra"],
            "Sharpe": acao["SharpeCompra"],
            "Sortino": acao["SortinoCompra"],
            "MaxDD": acao["MaxDrawdown"],
            "QtdSinais": acao["QtdCompra"],
            "ProfitFactor": acao["ProfitFactor"],
            "RetornoBacktest": acao["RetornoAcumulado"],
            "HitProb>60": acao["HitProbBucket"]
        }])

        st.dataframe(
            resumo,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Preço": st.column_config.NumberColumn("Preço", format="$ %.2f"),
                "WinRate": st.column_config.NumberColumn("Win Rate %", format="%.2f%%"),
                "Expectancy": st.column_config.NumberColumn("Expectancy %", format="%.2f%%"),
                "Sharpe": st.column_config.NumberColumn("Sharpe", format="%.2f"),
                "Sortino": st.column_config.NumberColumn("Sortino", format="%.2f"),
                "MaxDD": st.column_config.NumberColumn("Max DD %", format="%.2f%%"),
                "QtdSinais": st.column_config.NumberColumn("Qtd. Sinais"),
                "ProfitFactor": st.column_config.NumberColumn("Profit Factor", format="%.2f"),
                "RetornoBacktest": st.column_config.NumberColumn("Retorno Backtest %", format="%.2f%%"),
                "HitProb>60": st.column_config.NumberColumn("Hit Rate Prob>60", format="%.2f%%"),
            }
        )
    else:
        st.error("Erro ao carregar dados do ativo.")

# ===================== TAB 2 - GRÁFICO TÉCNICO =====================
with tab2:
    if dados_vencedoras:
        acao = dados_vencedoras[0]
        hist = acao["Hist"].copy()

        if not hist.empty:
            hist["SMA20"] = hist["Close"].rolling(window=20).mean()
            hist["SMA200"] = hist["Close"].rolling(window=200).mean()
            hist["BB_Mid"] = hist["Close"].rolling(window=20).mean()
            hist["BB_Std"] = hist["Close"].rolling(window=20).std()
            hist["BB_Upper"] = hist["BB_Mid"] + 2 * hist["BB_Std"]
            hist["BB_Lower"] = hist["BB_Mid"] - 2 * hist["BB_Std"]
            rsi_series = calcular_rsi_series(hist["Close"])

            fig = make_subplots(
                rows=3,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.08,
                row_heights=[0.65, 0.20, 0.15],
                subplot_titles=("Candlestick + Bollinger + SMAs", "Volume", "RSI (14)")
            )

            fig.add_trace(
                go.Candlestick(
                    x=hist.index,
                    open=hist["Open"],
                    high=hist["High"],
                    low=hist["Low"],
                    close=hist["Close"],
                    name="Preço"
                ),
                row=1, col=1
            )
            fig.add_trace(go.Scatter(x=hist.index, y=hist["SMA20"], name="SMA 20", line=dict(color="yellow")), row=1, col=1)
            fig.add_trace(go.Scatter(x=hist.index, y=hist["SMA200"], name="SMA 200", line=dict(color="orange")), row=1, col=1)
            fig.add_trace(go.Scatter(x=hist.index, y=hist["BB_Upper"], name="BB Upper", line=dict(color="lime")), row=1, col=1)
            fig.add_trace(go.Scatter(x=hist.index, y=hist["BB_Lower"], name="BB Lower", line=dict(color="red")), row=1, col=1)
            fig.add_trace(go.Bar(x=hist.index, y=hist["Volume"], name="Volume", marker_color="gray"), row=2, col=1)
            fig.add_trace(go.Scatter(x=hist.index, y=rsi_series, name="RSI", line=dict(color="purple")), row=3, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="lime", row=3, col=1)

            if acao["CandleInfo"]["padrao"] != "Sem padrão forte":
                fig.add_annotation(
                    x=hist.index[-1],
                    y=hist["High"].iloc[-1] * 1.02,
                    text=acao["CandleInfo"]["padrao"],
                    showarrow=True,
                    arrowhead=2,
                    font=dict(color="white", size=12)
                )

            fig.update_layout(
                height=750,
                template="plotly_dark",
                showlegend=True,
                title_text=f"{acao['Ticker']} - Análise Técnica"
            )
            st.plotly_chart(fig, use_container_width=True)

        if st.session_state.usar_tradingview:
            tv_symbol = "BITSTAMP:BTCUSD" if acao["Ticker"] == "BTC-USD" else "BITSTAMP:ETHUSD"
            tv_html = f"""
            <iframe src="https://www.tradingview.com/widgetembed/?symbol={tv_symbol}&interval=D&theme=dark&style=1"
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

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Preço Atual", f"${acao['Preço']:,.2f}")
        col2.metric("Prob. Alta", f"{acao['ProbAlta']*100:.1f}%")
        col3.metric("Confiabilidade ML", f"{acao['ConfiancaML']:.1f}%")
        col4.metric("Regime", acao["Regime"])

        st.markdown(f"**Veredito da IA:** {acao['Veredito']}")
        st.markdown(f"**Motivo:** {acao['Motivo']}")
        st.markdown(f"**Padrão de candle:** {acao['CandleInfo']['padrao']}")
        st.markdown(f"**Força do candle:** {acao['CandleInfo']['forca']}/30")
        st.markdown(f"**Sentimento das notícias:** {acao['SentimentoNoticias']}")

        st.markdown("### On-chain")
        oc_pub = acao["OnChain"]["publico"]
        oc_glass = acao["OnChain"]["glassnode"]

        c1, c2, c3 = st.columns(3)
        c1.metric("Score On-chain Público", f"{oc_pub.get('score', 0):.1f}")
        c2.metric("Sentimento Público", oc_pub.get("sentimento", "Neutro"))
        c3.metric("Score Glassnode", f"{oc_glass.get('score', 0):.1f}")

        if acao["Ticker"] == "BTC-USD":
            d1, d2, d3 = st.columns(3)
            d1.metric("Hash Rate", oc_pub.get("hash_rate", "N/D"))
            d2.metric("Transações 24h", oc_pub.get("transactions_24h", "N/D"))
            d3.metric("Fonte Pública", oc_pub.get("fonte", "N/D"))

        g1, g2, g3 = st.columns(3)
        g1.metric("Active Addresses", oc_glass.get("active_addresses", "N/D"))
        g2.metric("SOPR", oc_glass.get("sopr", "N/D"))
        g3.metric("MVRV", oc_glass.get("mvrv", "N/D"))
        st.caption(f"Fonte Glassnode: {oc_glass.get('fonte', 'N/D')}")

        if acao.get("Links"):
            st.markdown(f"**Últimas manchetes sobre {acao['Ticker']}:**")
            for n in acao["Links"][:5]:
                st.markdown(f"• [{n['titulo']}]({n['link']})")
    else:
        st.info("Nenhum dado disponível.")

# ===================== TAB 4 - BACKTEST =====================
with tab4:
    st.subheader("📜 Backtest Walk-Forward")

    if dados_vencedoras:
        acao = dados_vencedoras[0]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Expectancy Compra", f"{acao['ExpectancyCompra']:.2f}%")
        c2.metric("Sharpe Ratio", f"{acao['SharpeCompra']:.2f}")
        c3.metric("Sortino Ratio", f"{acao['SortinoCompra']:.2f}")
        c4.metric("Sinais de Compra", acao["QtdCompra"])

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Max Drawdown", f"{acao['MaxDrawdown']:.2f}%")
        c6.metric("Profit Factor", f"{acao['ProfitFactor']:.2f}")
        c7.metric("Retorno Backtest", f"{acao['RetornoAcumulado']:.2f}%")
        c8.metric("Hit Rate Prob>60", f"{acao['HitProbBucket']:.2f}%")

        equity_curve = acao.get("EquityCurve", pd.Series(dtype=float))
        if equity_curve is not None and len(equity_curve) > 5:
            fig_eq = go.Figure()
            fig_eq.add_trace(go.Scatter(
                x=equity_curve.index,
                y=equity_curve.values,
                mode="lines",
                name="Equity Curve"
            ))
            fig_eq.update_layout(
                height=320,
                template="plotly_dark",
                title=f"Equity Curve - {acao['Ticker']}",
                margin=dict(l=20, r=20, t=50, b=20)
            )
            st.plotly_chart(fig_eq, use_container_width=True)

    try:
        conn = get_conn()
        df_hist = pd.read_sql("SELECT * FROM sinais ORDER BY data DESC", conn)
        conn.close()

        if not df_hist.empty:
            st.markdown("### Histórico de Sinais")
            st.dataframe(df_hist, use_container_width=True, hide_index=True)

            csv = df_hist.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Baixar CSV",
                data=csv,
                file_name="historico_sinais_crypto.csv",
                mime="text/csv"
            )
        else:
            st.info("Ainda não há sinais salvos.")
    except Exception:
        st.info("Ainda não há sinais salvos.")

# ===================== TAB 5 - MONITORAMENTO MODELO =====================
with tab5:
    st.subheader("🧠 Monitoramento da Performance do Modelo")

    df_mon, calib = resumo_monitoramento_modelo(crypto_selecionado)

    if df_mon is not None and not df_mon.empty:
        c1, c2, c3 = st.columns(3)
        c1.metric("Sinais Resolvidos", int(len(df_mon)))
        c2.metric("Taxa de Acerto", f"{df_mon['outcome_h3'].mean()*100:.2f}%")
        c3.metric("Retorno Médio H+3", f"{df_mon['realized_return_h3'].mean():.2f}%")

        st.markdown("### Calibração por Faixa de Score")
        st.dataframe(calib, use_container_width=True, hide_index=True)

        fig_ret = go.Figure()
        fig_ret.add_trace(go.Histogram(x=df_mon["realized_return_h3"], nbinsx=40, name="Retornos H+3"))
        fig_ret.update_layout(
            height=320,
            template="plotly_dark",
            title="Distribuição dos Retornos Realizados dos Sinais"
        )
        st.plotly_chart(fig_ret, use_container_width=True)

        st.markdown("### Últimos sinais monitorados")
        st.dataframe(
            df_mon.tail(50).sort_values("signal_time", ascending=False),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("Ainda não há sinais resolvidos suficientes para monitoramento.")

st.success("✅ Monitor IA Quant Pro carregado com sucesso. Atualização automática a cada 5 minutos.")
