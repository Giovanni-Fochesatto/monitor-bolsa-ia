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
st.set_page_config(page_title="Monitor IA Pro - Bitcoin & Ethereum", layout="wide")
st_autorefresh(interval=300 * 1000, key="data_refresh")  # atualiza a cada 5 minutos

if "telegram_ativado" not in st.session_state:
    st.session_state.telegram_ativado = False
if "ultimos_alertas" not in st.session_state:
    st.session_state.ultimos_alertas = set()
if "usar_tradingview" not in st.session_state:
    st.session_state.usar_tradingview = True

# ===================== BANCO DE DADOS =====================
def init_db():
    conn = sqlite3.connect("sinais_bitcoin.db")
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS sinais (
            data TEXT,
            ticker TEXT,
            preco REAL,
            veredito TEXT,
            motivo TEXT,
            expectancy REAL,
            sharpe REAL
        )
        """
    )
    conn.commit()
    conn.close()


def salvar_sinal(acao):
    init_db()
    conn = sqlite3.connect("sinais_bitcoin.db")
    c = conn.cursor()
    agora = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute(
        """
        INSERT INTO sinais
        (data, ticker, preco, veredito, motivo, expectancy, sharpe)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            agora,
            acao["Ticker"],
            acao["Preço"],
            acao["Veredito"],
            acao["Motivo"],
            acao.get("ExpectancyCompra", 0),
            acao.get("SharpeCompra", 0),
        ),
    )
    conn.commit()
    conn.close()


# ===================== FUNÇÕES UTILITÁRIAS =====================
def safe_float(value, default=0.0):
    try:
        if value is None:
            return default
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def sigmoid(x):
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


def annualization_factor():
    return 252.0


# ===================== FUNÇÕES TÉCNICAS =====================
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


def calcular_rsi(data: pd.Series, window: int = 14) -> float:
    if len(data) < window:
        return 50.0
    return safe_float(calcular_rsi_series(data, window).iloc[-1], 50.0)


def calcular_atr(hist: pd.DataFrame, window: int = 14) -> pd.Series:
    if hist is None or hist.empty or len(hist) < 2:
        return pd.Series([0.0] * len(hist), index=hist.index if hist is not None else None)

    high = hist["High"]
    low = hist["Low"]
    close = hist["Close"]

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    return atr.fillna(method="bfill").fillna(0.0)


def detectar_candlestick_patterns(hist: pd.DataFrame) -> dict:
    if hist is None or hist.empty or len(hist) < 5:
        return {
            "padrao": "Dados insuficientes",
            "forca": 0,
            "direcao": "neutro",
            "score": 0,
        }

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
    direcao = "neutro"

    # Hammer
    if lower_shadow > 2.0 * body and upper_shadow < body and body / candle_range < 0.4:
        patterns.append("Hammer")
        score += 18
        direcao = "bullish"

    # Shooting star
    if upper_shadow > 2.0 * body and lower_shadow < body and body / candle_range < 0.4:
        patterns.append("Shooting Star")
        score -= 18
        direcao = "bearish"

    # Bullish / Bearish engulfing
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
            direcao = "bullish"

        if bearish_engulfing:
            patterns.append("Bearish Engulfing")
            score -= 22
            direcao = "bearish"

    # 3 candles direction
    if len(df) >= 3:
        last3 = df.tail(3)
        if (last3["Close"] > last3["Open"]).all() and last3["Close"].is_monotonic_increasing:
            patterns.append("Three Bullish Candles")
            score += 12
            direcao = "bullish"
        elif (last3["Close"] < last3["Open"]).all() and last3["Close"].is_monotonic_decreasing:
            patterns.append("Three Bearish Candles")
            score -= 12
            direcao = "bearish"

    if not patterns:
        patterns = ["Sem padrão forte"]

    return {
        "padrao": " + ".join(patterns),
        "forca": min(abs(score), 30),
        "direcao": direcao,
        "score": score,
    }


# ===================== ON-CHAIN =====================
@st.cache_data(ttl=600, show_spinner=False)
def obter_onchain_score(ticker: str) -> dict:
    """
    BTC: usa dados públicos da blockchain.info.
    ETH: sem API pública robusta aqui -> neutro por design, para não inventar sinal.
    """
    base = {
        "score": 0.0,
        "sentimento": "Neutro",
        "hash_rate": "N/D",
        "transactions_24h": "N/D",
        "fonte": "N/D",
    }

    if ticker != "BTC-USD":
        base["fonte"] = "Neutro (sem API on-chain robusta configurada para ETH)"
        return base

    try:
        r = requests.get("https://api.blockchain.info/stats", timeout=10)
        if r.status_code != 200:
            return base

        data = r.json()
        hash_rate = safe_float(data.get("hash_rate"), 0.0) / 1e9  # ~ EH/s aproximado
        tx_24h = safe_float(data.get("n_tx"), 0.0)
        trade_volume_btc = safe_float(data.get("trade_volume_btc"), 0.0)
        market_price_usd = safe_float(data.get("market_price_usd"), 0.0)

        score = 0.0

        if hash_rate > 400:
            score += 8
        elif hash_rate > 250:
            score += 4
        else:
            score -= 2

        if tx_24h > 300000:
            score += 8
        elif tx_24h > 200000:
            score += 4
        else:
            score -= 2

        if trade_volume_btc > 100000:
            score += 4

        sentimento = "Positivo" if score >= 8 else "Neutro"
        if score <= -2:
            sentimento = "Cauteloso"

        return {
            "score": score,
            "sentimento": sentimento,
            "hash_rate": f"{hash_rate:.0f} EH/s",
            "transactions_24h": f"{int(tx_24h):,}".replace(",", "."),
            "market_price_usd": market_price_usd,
            "fonte": "Blockchain.com / blockchain.info",
        }
    except Exception:
        return base


# ===================== REGRESSÃO LOGÍSTICA LEVE (NUMPY) =====================
def treinar_regressao_logistica_numpy(X: np.ndarray, y: np.ndarray, lr: float = 0.05, epochs: int = 250):
    if X is None or y is None or len(X) == 0 or len(y) == 0:
        return None

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    if X.ndim != 2 or y.ndim != 1:
        return None

    if len(X) != len(y):
        return None

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
    mu = modelo["mu"]
    sigma = modelo["sigma"]
    w = modelo["weights"]

    xn = (x_row - mu) / sigma
    xb = np.concatenate([[1.0], xn])
    prob = sigmoid(xb @ w)
    return float(np.clip(prob, 0.0, 1.0))


# ===================== ENGENHARIA DE FEATURES =====================
def construir_features(hist: pd.DataFrame) -> pd.DataFrame:
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
    df["target_up_next"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    return df


# ===================== BACKTEST / PERFORMANCE =====================
def simular_performance_historica(hist: pd.DataFrame) -> dict:
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
    }

    try:
        if hist is None or hist.empty or len(hist) < 260:
            return fallback

        df = construir_features(hist)
        df = df.dropna().copy()

        if len(df) < 180:
            return fallback

        # Regras históricas mais rígidas para reduzir falso positivo
        bullish_entry = (
            (df["dist_sma20"] > 0) &
            (df["dist_sma50"] > 0) &
            (df["macd_hist"] > 0) &
            (df["rsi14"].between(45, 68)) &
            (df["mom_7"] > 0)
        )

        bearish_exit = (
            (df["macd_hist"] < 0) |
            (df["rsi14"] > 74) |
            (df["dist_sma20"] < -0.01)
        )

        df["sinal"] = 0
        em_posicao = False
        sinais = []

        for _, row in df.iterrows():
            if not em_posicao and bullish_entry.loc[row.name]:
                em_posicao = True
            elif em_posicao and bearish_exit.loc[row.name]:
                em_posicao = False
            sinais.append(1 if em_posicao else 0)

        df["sinal"] = sinais
        df["retorno_estrategia"] = df["sinal"].shift(1).fillna(0) * df["retorno"].fillna(0)

        entradas = ((df["sinal"] == 1) & (df["sinal"].shift(1).fillna(0) == 0)).sum()

        ganhos = df.loc[df["retorno_estrategia"] > 0, "retorno_estrategia"]
        perdas = df.loc[df["retorno_estrategia"] < 0, "retorno_estrategia"]

        winrate = (len(ganhos) / max(len(ganhos) + len(perdas), 1)) * 100.0
        media_gain = ganhos.mean() if len(ganhos) > 0 else 0.0
        media_loss = perdas.mean() if len(perdas) > 0 else 0.0
        expectancy = ((winrate / 100.0) * media_gain) + ((1 - winrate / 100.0) * media_loss)

        std = df["retorno_estrategia"].std()
        sharpe = (
            (df["retorno_estrategia"].mean() / std) * np.sqrt(annualization_factor())
            if std not in [0, np.nan] and pd.notna(std) and std > 0
            else 0.0
        )

        downside = df.loc[df["retorno_estrategia"] < 0, "retorno_estrategia"]
        downside_std = downside.std()
        sortino = (
            (df["retorno_estrategia"].mean() / downside_std) * np.sqrt(annualization_factor())
            if downside_std not in [0, np.nan] and pd.notna(downside_std) and downside_std > 0
            else 0.0
        )

        df["equity"] = (1 + df["retorno_estrategia"].fillna(0)).cumprod()
        df["peak"] = df["equity"].cummax()
        df["drawdown"] = (df["equity"] - df["peak"]) / df["peak"].replace(0, np.nan)
        max_dd = df["drawdown"].min() * 100.0 if not df["drawdown"].empty else 0.0

        gross_profit = ganhos.sum() if len(ganhos) > 0 else 0.0
        gross_loss = abs(perdas.sum()) if len(perdas) > 0 else 0.0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0.0)

        retorno_acumulado = (df["equity"].iloc[-1] - 1) * 100.0 if not df["equity"].empty else 0.0

        return {
            "taxa_compra": float(winrate),
            "expectancy_compra": float(expectancy * 100.0),
            "sharpe_compra": float(sharpe),
            "sortino_compra": float(sortino),
            "max_drawdown": float(max_dd),
            "qtd_compra": int(entradas),
            "profit_factor": float(profit_factor),
            "retorno_acumulado": float(retorno_acumulado),
            "equity_curve": df["equity"].copy(),
        }

    except Exception:
        return fallback


# ===================== SCORE DE NOTÍCIAS =====================
def pontuar_noticias(ticker: str):
    noticias_texto = ""
    lista_links = []

    termos_ativos = {
        "BTC-USD": ["bitcoin", "btc", "etf bitcoin", "halving"],
        "ETH-USD": ["ethereum", "eth", "ether", "ethereum etf"],
    }

    try:
        query = " OR ".join(termos_ativos.get(ticker, [ticker.replace("-USD", "")]))
        feed = feedparser.parse(f"https://news.google.com/rss/search?q={query}&hl=pt-BR")

        for entry in feed.entries[:8]:
            titulo = getattr(entry, "title", "")
            link = getattr(entry, "link", "")
            if not titulo:
                continue
            noticias_texto += titulo.lower() + " "
            lista_links.append({"titulo": titulo, "link": link})
    except Exception:
        pass

    palavras_pos = [
        "alta", "bull", "compra", "subiu", "rally", "high", "buy", "adoption",
        "aprovação", "recorde", "flows", "inflow", "institucional"
    ]
    palavras_neg = [
        "queda", "bear", "venda", "caiu", "crash", "low", "sell", "dump",
        "hack", "banimento", "processo", "outflow", "fraude"
    ]

    score_p = sum(noticias_texto.count(w) for w in palavras_pos)
    score_n = sum(noticias_texto.count(w) for w in palavras_neg)

    return score_p, score_n, lista_links


# ===================== MODELO IA / ENSEMBLE =====================
def probabilidade_modelo_ml(hist: pd.DataFrame) -> dict:
    result = {"prob_up": 0.5, "confianca_ml": 0.0}

    try:
        if hist is None or hist.empty or len(hist) < 260:
            return result

        df = construir_features(hist)
        cols = ["rsi14", "macd_hist", "dist_sma20", "dist_sma50", "dist_sma200", "mom_3", "mom_7", "mom_14", "atr_pct", "vol_20"]
        df = df.dropna(subset=cols + ["target_up_next"]).copy()

        if len(df) < 220:
            return result

        X = df[cols].values
        y = df["target_up_next"].values.astype(float)

        # treino histórico e previsão apenas do último ponto
        train_X = X[:-1]
        train_y = y[:-1]
        last_x = X[-1]

        modelo = treinar_regressao_logistica_numpy(train_X, train_y, lr=0.05, epochs=300)
        if not modelo:
            return result

        prob = prever_prob_regressao(modelo, last_x)

        # confiança aproximada pela distância para 50%
        confianca = abs(prob - 0.5) * 200.0

        result["prob_up"] = float(prob)
        result["confianca_ml"] = float(np.clip(confianca, 0.0, 100.0))
        return result

    except Exception:
        return result


def modelo_ensemble_score(
    ticker: str,
    hist: pd.DataFrame,
    rsi_val: float,
    macd: pd.Series,
    sinal_macd: pd.Series,
    score_p: int,
    score_n: int,
    candle_info: dict,
    onchain_info: dict,
    prob_ml: float,
    sim: dict,
):
    score = 50.0
    componentes = []

    close = hist["Close"]
    preco = close.iloc[-1]
    sma20 = close.rolling(20).mean().iloc[-1] if len(close) >= 20 else preco
    sma50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else preco
    sma200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else preco

    # 1. ML
    ml_score = (prob_ml - 0.5) * 80.0
    score += ml_score
    componentes.append(f"ML {prob_ml*100:.1f}%")

    # 2. RSI
    if rsi_val < 28:
        score += 14
        componentes.append("RSI sobrevenda")
    elif rsi_val < 38:
        score += 8
        componentes.append("RSI favorável")
    elif rsi_val > 78:
        score -= 16
        componentes.append("RSI sobrecompra extrema")
    elif rsi_val > 68:
        score -= 8
        componentes.append("RSI esticado")

    # 3. MACD
    if len(macd) > 1 and len(sinal_macd) > 1:
        if macd.iloc[-1] > sinal_macd.iloc[-1] and macd.iloc[-2] <= sinal_macd.iloc[-2]:
            score += 10
            componentes.append("MACD cruzou para cima")
        elif macd.iloc[-1] < sinal_macd.iloc[-1] and macd.iloc[-2] >= sinal_macd.iloc[-2]:
            score -= 10
            componentes.append("MACD cruzou para baixo")
        elif macd.iloc[-1] > sinal_macd.iloc[-1]:
            score += 5
        else:
            score -= 5

    # 4. Tendência
    if preco > sma20 > sma50:
        score += 8
        componentes.append("Tendência curta positiva")
    elif preco < sma20 < sma50:
        score -= 8
        componentes.append("Tendência curta negativa")

    if preco > sma200:
        score += 8
        componentes.append("Acima da SMA200")
    else:
        score -= 8
        componentes.append("Abaixo da SMA200")

    # 5. Candle
    score += candle_info.get("score", 0) * 0.55
    if candle_info.get("padrao") != "Sem padrão forte":
        componentes.append(candle_info.get("padrao", ""))

    # 6. Notícias
    sentimento = score_p - score_n
    score += np.clip(sentimento * 3.0, -12, 12)
    if sentimento > 0:
        componentes.append("Notícias positivas")
    elif sentimento < 0:
        componentes.append("Notícias negativas")

    # 7. On-chain
    score += np.clip(onchain_info.get("score", 0.0), -10, 10)
    if ticker == "BTC-USD":
        componentes.append(f"On-chain {onchain_info.get('sentimento', 'Neutro')}")

    # 8. Qualidade histórica da estratégia
    sharpe = sim.get("sharpe_compra", 0.0)
    expectancy = sim.get("expectancy_compra", 0.0)
    profit_factor = sim.get("profit_factor", 0.0)
    max_dd = sim.get("max_drawdown", 0.0)

    if sharpe > 1.2:
        score += 6
        componentes.append("Sharpe forte")
    elif sharpe < 0:
        score -= 6

    if expectancy > 0.15:
        score += 5
        componentes.append("Expectancy positiva")
    elif expectancy < 0:
        score -= 5

    if profit_factor > 1.25:
        score += 5
    elif 0 < profit_factor < 0.95:
        score -= 5

    if max_dd < -35:
        score -= 5
        componentes.append("Drawdown histórico alto")

    score = float(np.clip(score, 0.0, 100.0))
    return round(score, 1), componentes


def classificar_veredito(score_ia: float, prob_up: float, rsi: float):
    if score_ia >= 82 and prob_up >= 0.62 and rsi < 72:
        return "COMPRA FORTE 🟢", "success"
    if score_ia >= 68 and prob_up >= 0.56:
        return "COMPRA 🟢", "success"
    if score_ia <= 18 and prob_up <= 0.38 and rsi > 30:
        return "VENDA FORTE 🔴", "error"
    if score_ia <= 32 and prob_up <= 0.44:
        return "VENDA 🔴", "error"
    return "NEUTRO ⚖️", "warning"


def gerar_motivo(score_ia: float, componentes: list, candle_info: dict, prob_up: float):
    base = f"Score IA {score_ia}/100 | Probabilidade estimada de alta: {prob_up*100:.1f}%."
    top = ", ".join(componentes[:4]) if componentes else "Sem confluência forte."
    candle_txt = candle_info.get("padrao", "Sem padrão forte")
    return f"{base} Confluências: {top}. Candle: {candle_txt}."


def gerar_alerta_inteligente(acao: dict):
    score = acao.get("ScoreIA", 50)
    prob = acao.get("ProbAlta", 0.5)
    veredito = acao.get("Veredito", "NEUTRO ⚖️")
    ticker = acao.get("Ticker", "")

    if veredito == "COMPRA FORTE 🟢" and prob >= 0.65:
        return {
            "tipo": "entrada_agressiva",
            "titulo": f"🚨 Entrada agressiva detectada em {ticker}",
            "mensagem": f"Score {score}/100 e probabilidade de alta {prob*100:.1f}%.",
        }

    if veredito == "COMPRA 🟢" and score >= 70:
        return {
            "tipo": "entrada_moderada",
            "titulo": f"📈 Entrada moderada em {ticker}",
            "mensagem": f"Confluência favorável com score {score}/100.",
        }

    if veredito == "VENDA FORTE 🔴":
        return {
            "tipo": "risco_alto",
            "titulo": f"🛑 Risco elevado em {ticker}",
            "mensagem": f"Sinal defensivo com score {score}/100.",
        }

    if 45 <= score <= 55:
        return {
            "tipo": "mercado_neutro",
            "titulo": f"⏸️ Mercado sem vantagem estatística clara em {ticker}",
            "mensagem": "Melhor aguardar confirmação antes de agir.",
        }

    return None


# ===================== CACHE =====================
@st.cache_data(ttl=300, show_spinner=False)
def obter_dados_crypto(ticker: str):
    ativo = yf.Ticker(ticker)
    hist = ativo.history(period="5y", auto_adjust=True)
    info = {}
    try:
        info = ativo.info
    except Exception:
        info = {}
    return info, hist


@st.cache_data(ttl=1800, show_spinner=False)
def obter_macro():
    macro = {}
    try:
        macro["Dolar"] = safe_float(yf.Ticker("USDBRL=X").fast_info.get("last_price"), 5.70)
    except Exception:
        macro["Dolar"] = 5.70

    try:
        macro["Bitcoin_USD"] = safe_float(yf.Ticker("BTC-USD").fast_info.get("last_price"), 0)
    except Exception:
        macro["Bitcoin_USD"] = 0

    try:
        macro["Ethereum_USD"] = safe_float(yf.Ticker("ETH-USD").fast_info.get("last_price"), 0)
    except Exception:
        macro["Ethereum_USD"] = 0

    return macro


# ===================== PROCESSAMENTO CENTRAL =====================
def processar_crypto(ticker: str):
    info, hist = obter_dados_crypto(ticker)
    if hist is None or hist.empty:
        return None

    hist = hist.copy()
    close = hist["Close"]

    rsi_val = calcular_rsi(close)
    sma200 = close.rolling(window=200).mean()

    exp12 = close.ewm(span=12, adjust=False).mean()
    exp26 = close.ewm(span=26, adjust=False).mean()
    macd = exp12 - exp26
    sinal_macd = macd.ewm(span=9, adjust=False).mean()

    score_p, score_n, lista_links = pontuar_noticias(ticker)
    candle_info = detectar_candlestick_patterns(hist)
    onchain_info = obter_onchain_score(ticker)
    sim = simular_performance_historica(hist) or {}

    ml_info = probabilidade_modelo_ml(hist)
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
        onchain_info=onchain_info,
        prob_ml=prob_up,
        sim=sim,
    )

    veredito, cor = classificar_veredito(score_ia, prob_up, rsi_val)
    motivo_detalhe = gerar_motivo(score_ia, componentes, candle_info, prob_up)

    alerta = gerar_alerta_inteligente(
        {
            "Ticker": ticker,
            "ScoreIA": score_ia,
            "ProbAlta": prob_up,
            "Veredito": veredito,
        }
    )

    return {
        "Ticker": ticker,
        "Preço": p_atual,
        "Veredito": veredito,
        "Cor": cor,
        "Motivo": motivo_detalhe,
        "RSI": rsi_val,
        "Hist": hist,
        "Links": lista_links,
        "ScoreIA": score_ia,
        "ProbAlta": prob_up,
        "ConfiancaML": ml_info.get("confianca_ml", 0.0),
        "CandleInfo": candle_info,
        "OnChain": onchain_info,
        "Info": info,
        "SMA200": safe_float(sma200.iloc[-1], p_atual) if not sma200.empty else p_atual,
        "TaxaCompra": sim.get("taxa_compra", 0.0),
        "ExpectancyCompra": sim.get("expectancy_compra", 0.0),
        "SharpeCompra": sim.get("sharpe_compra", 0.0),
        "SortinoCompra": sim.get("sortino_compra", 0.0),
        "MaxDrawdown": sim.get("max_drawdown", 0.0),
        "QtdCompra": sim.get("qtd_compra", 0),
        "ProfitFactor": sim.get("profit_factor", 0.0),
        "RetornoAcumulado": sim.get("retorno_acumulado", 0.0),
        "EquityCurve": sim.get("equity_curve", pd.Series(dtype=float)),
        "Alerta": alerta,
        "SentimentoNoticias": score_p - score_n,
    }


# ===================== SIDEBAR =====================
st.sidebar.title("₿ Monitor IA Pro - Crypto")
crypto_selecionado = st.sidebar.selectbox("Escolha o ativo:", ["BTC-USD", "ETH-USD"])

st.sidebar.subheader("📊 Câmbio em Tempo Real")
cambio = obter_macro()
st.sidebar.metric("Bitcoin (USD)", f"${cambio['Bitcoin_USD']:,.0f}" if cambio["Bitcoin_USD"] else "N/D")
st.sidebar.metric("Ethereum (USD)", f"${cambio['Ethereum_USD']:,.0f}" if cambio["Ethereum_USD"] else "N/D")
st.sidebar.metric("Dólar (R$)", f"R$ {cambio['Dolar']:.2f}")

st.sidebar.divider()
st.sidebar.subheader("📺 TradingView")
st.session_state.usar_tradingview = st.sidebar.checkbox("Usar gráfico TradingView", value=True)

estrategia_ativa = st.sidebar.selectbox(
    "Estratégia:",
    [
        "Análise Técnica + Ensemble",
        "Momentum + Sentimento",
        "Position Trading",
    ],
)

# ===================== CABEÇALHO =====================
st.title(f"🤖 Monitor IA Pro - {crypto_selecionado}")
st.caption(f"Atualização: {time.strftime('%H:%M:%S')} | Local: Porto Alegre/RS")

# ===================== PROCESSAMENTO =====================
with st.spinner("📡 Baixando dados do ativo e calculando score institucional..."):
    resultado = processar_crypto(crypto_selecionado)

dados_vencedoras = [resultado] if resultado else []

# ===================== ALERTA ÚNICO POR ESTADO =====================
if resultado and resultado.get("Alerta"):
    chave_alerta = (
        resultado["Ticker"],
        resultado["Alerta"]["tipo"],
        round(resultado["ScoreIA"], 1),
        round(resultado["ProbAlta"], 2),
    )
    if chave_alerta not in st.session_state.ultimos_alertas:
        st.session_state.ultimos_alertas.add(chave_alerta)
        st.toast(resultado["Alerta"]["titulo"], icon="🚨")

# ===================== TABS =====================
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Overview", "📈 Gráfico Técnico", "📉 Análise", "📜 Backtest & Histórico"]
)

# ===================== TAB 1 - OVERVIEW =====================
with tab1:
    if dados_vencedoras:
        st.subheader(f"🏆 Análise Atual - Estratégia: {estrategia_ativa}")
        acao = dados_vencedoras[0]

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Preço Atual (USD)", f"${acao['Preço']:,.0f}")
        col2.metric("RSI (14)", f"{acao['RSI']:.1f}")
        col3.metric("Score IA", f"{acao['ScoreIA']:.1f}/100")
        col4.metric("Prob. Alta", f"{acao['ProbAlta']*100:.1f}%")
        col5.metric("Veredito IA", acao["Veredito"])

        st.markdown(f"**Motivo:** {acao['Motivo']}")

        if acao.get("Alerta"):
            st.info(f"**{acao['Alerta']['titulo']}** — {acao['Alerta']['mensagem']}")

        df_resumo = pd.DataFrame(
            [
                {
                    "Preço": acao["Preço"],
                    "TaxaCompra": acao["TaxaCompra"],
                    "ExpectancyCompra": acao["ExpectancyCompra"],
                    "SharpeCompra": acao["SharpeCompra"],
                    "SortinoCompra": acao["SortinoCompra"],
                    "MaxDrawdown": acao["MaxDrawdown"],
                    "QtdCompra": acao["QtdCompra"],
                    "ProfitFactor": acao["ProfitFactor"],
                }
            ]
        )

        st.dataframe(
            df_resumo,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Preço": st.column_config.NumberColumn("Preço", format="$ %.2f"),
                "TaxaCompra": st.column_config.NumberColumn("Win Rate %", format="%.1f%%"),
                "ExpectancyCompra": st.column_config.NumberColumn("Expectancy %", format="%.2f%%"),
                "SharpeCompra": st.column_config.NumberColumn("Sharpe", format="%.2f"),
                "SortinoCompra": st.column_config.NumberColumn("Sortino", format="%.2f"),
                "MaxDrawdown": st.column_config.NumberColumn("Max DD %", format="%.2f%%"),
                "QtdCompra": st.column_config.NumberColumn("Sinais Históricos"),
                "ProfitFactor": st.column_config.NumberColumn("Profit Factor", format="%.2f"),
            },
        )

        salvar_sinal(acao)
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
                subplot_titles=("Candlestick + Bollinger + SMAs", "Volume", "RSI (14)"),
            )

            fig.add_trace(
                go.Candlestick(
                    x=hist.index,
                    open=hist["Open"],
                    high=hist["High"],
                    low=hist["Low"],
                    close=hist["Close"],
                    name="Preço",
                ),
                row=1,
                col=1,
            )

            fig.add_trace(
                go.Scatter(x=hist.index, y=hist["SMA20"], name="SMA 20", line=dict(color="yellow")),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(x=hist.index, y=hist["SMA200"], name="SMA 200", line=dict(color="orange")),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(x=hist.index, y=hist["BB_Upper"], name="BB Upper", line=dict(color="lime")),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(x=hist.index, y=hist["BB_Lower"], name="BB Lower", line=dict(color="red")),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Bar(x=hist.index, y=hist["Volume"], name="Volume", marker_color="gray"),
                row=2,
                col=1,
            )
            fig.add_trace(
                go.Scatter(x=hist.index, y=rsi_series, name="RSI", line=dict(color="purple")),
                row=3,
                col=1,
            )
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="lime", row=3, col=1)

            if acao["CandleInfo"]["padrao"] != "Sem padrão forte":
                fig.add_annotation(
                    x=hist.index[-1],
                    y=hist["High"].iloc[-1] * 1.02,
                    text=acao["CandleInfo"]["padrao"],
                    showarrow=True,
                    arrowhead=2,
                    font=dict(color="white", size=12),
                )

            fig.update_layout(
                height=750,
                template="plotly_dark",
                showlegend=True,
                title_text=f"{acao['Ticker']} - Análise Técnica",
            )
            st.plotly_chart(fig, use_container_width=True)

        if st.session_state.usar_tradingview:
            tv_html = f"""
            <iframe src="https://www.tradingview.com/widgetembed/?symbol={acao['Ticker']}&interval=D&theme=dark&style=1"
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

        col1, col2, col3 = st.columns(3)
        col1.metric("Preço Atual", f"${acao['Preço']:,.0f}")
        col2.metric("Prob. Alta", f"{acao['ProbAlta']*100:.1f}%")
        col3.metric("Confiabilidade ML", f"{acao['ConfiancaML']:.1f}%")

        st.markdown(f"**Veredito da IA:** {acao['Veredito']}")
        st.markdown(f"**Motivo:** {acao['Motivo']}")
        st.markdown(f"**Padrão de candle:** {acao['CandleInfo']['padrao']}")
        st.markdown(f"**Força do candle:** {acao['CandleInfo']['forca']}/30")
        st.markdown(f"**Sentimento das notícias:** {acao['SentimentoNoticias']}")

        if acao["Ticker"] == "BTC-USD":
            st.markdown("**On-chain BTC:**")
            oc1, oc2, oc3 = st.columns(3)
            oc1.metric("Hash Rate", acao["OnChain"].get("hash_rate", "N/D"))
            oc2.metric("Transações 24h", acao["OnChain"].get("transactions_24h", "N/D"))
            oc3.metric("Sentimento On-chain", acao["OnChain"].get("sentimento", "Neutro"))
            st.caption(f"Fonte: {acao['OnChain'].get('fonte', 'N/D')}")
        else:
            st.caption("On-chain avançado para ETH foi mantido neutro por falta de API pública robusta configurada.")

        if acao.get("Links"):
            st.markdown(f"**Últimas Manchetes sobre {acao['Ticker']}:**")
            for n in acao["Links"][:5]:
                st.markdown(f"• [{n['titulo']}]({n['link']})")
    else:
        st.info("Nenhum dado disponível.")

# ===================== TAB 4 - BACKTEST =====================
with tab4:
    st.subheader("📜 Backtest & Histórico de Sinais")

    if dados_vencedoras:
        acao = dados_vencedoras[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Expectancy Compra", f"{acao['ExpectancyCompra']:.2f}%")
        c2.metric("Sharpe Ratio", f"{acao['SharpeCompra']:.2f}")
        c3.metric("Sortino Ratio", f"{acao['SortinoCompra']:.2f}")
        c4.metric("Sinais de Compra Históricos", acao["QtdCompra"])

        c5, c6, c7 = st.columns(3)
        c5.metric("Max Drawdown", f"{acao['MaxDrawdown']:.2f}%")
        c6.metric("Profit Factor", f"{acao['ProfitFactor']:.2f}")
        c7.metric("Retorno Backtest", f"{acao['RetornoAcumulado']:.2f}%")

        equity_curve = acao.get("EquityCurve", pd.Series(dtype=float))
        if equity_curve is not None and len(equity_curve) > 5:
            fig_eq = go.Figure()
            fig_eq.add_trace(
                go.Scatter(
                    x=equity_curve.index,
                    y=equity_curve.values,
                    mode="lines",
                    name="Equity Curve",
                )
            )
            fig_eq.update_layout(
                height=320,
                template="plotly_dark",
                title=f"Equity Curve - {acao['Ticker']}",
                margin=dict(l=20, r=20, t=50, b=20),
            )
            st.plotly_chart(fig_eq, use_container_width=True)

    try:
        conn = sqlite3.connect("sinais_bitcoin.db")
        df_historico = pd.read_sql("SELECT * FROM sinais ORDER BY data DESC", conn)
        conn.close()

        if not df_historico.empty:
            st.dataframe(df_historico, use_container_width=True, hide_index=True)

            csv = df_historico.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Baixar CSV",
                data=csv,
                file_name="historico_sinais_crypto.csv",
                mime="text/csv",
            )
        else:
            st.info("Ainda não há sinais salvos.")
    except Exception:
        st.info("Ainda não há sinais salvos.")

st.success("✅ Monitor Bitcoin + Ethereum configurado com sucesso! Atualiza automaticamente a cada 5 minutos.")
