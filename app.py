import streamlit as st
import yfinance as yf
import pandas as pd
import time
import feedparser
import nltk
import numpy as np
from streamlit_autorefresh import st_autorefresh

# --- CONFIGURAÇÕES E AMBIENTE ---
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

st.set_page_config(page_title="Monitor IA Pro", layout="wide")
st_autorefresh(interval=300 * 1000, key="data_refresh")

if 'filtros_ativos' not in st.session_state:
    st.session_state.filtros_ativos = False

def ativar_filtros():
    st.session_state.filtros_ativos = True

# --- FUNÇÕES TÉCNICAS ---
def calcular_graham(lpa, vpa):
    if lpa > 0 and vpa > 0:
        return np.sqrt(22.5 * lpa * vpa)
    return 0

def calcular_rsi(data, window=14):
    if len(data) < window: return 50.0
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1])

# --- ESTRATÉGIA VALUE INVESTING ---
def calcular_score_value(info):
    """Avalia a empresa com base em métricas de valor (Value Investing)"""
    score = 0
    if 0 < info.get('trailingPE', 99) < 15: score += 1
    if 0 < info.get('priceToBook', 99) < 1.5: score += 1
    if (info.get('dividendYield', 0) or 0) * 100 > 5: score += 1
    if (info.get('operatingMargins', 0) or 0) > 0.1: score += 1
    return score

def simular_performance_historica(hist):
    """Analisa RSI + Média Móvel 200 (Tendência) + MACD nos últimos 5 anos"""
    if len(hist) < 250: return 0, 0, 0, 0, 0
    
    precos = hist['Close']
    sma200 = precos.rolling(window=200).mean()
    exp12 = precos.ewm(span=12, adjust=False).mean()
    exp26 = precos.ewm(span=26, adjust=False).mean()
    macd = exp12 - exp26
    sinal_macd = macd.ewm(span=9, adjust=False).mean()

    acertos_c, total_c = 0, 0
    acertos_v, total_v = 0, 0
    retornos = []
    
    for i in range(200, len(precos) - 15):
        window_rsi = precos.iloc[i-14:i]
        rsi_p = calcular_rsi(window_rsi)
        p_entrada = precos.iloc[i]
        p_saida = precos.iloc[i + 15]
        m200 = sma200.iloc[i]
        
        retorno = (p_saida / p_entrada) - 1
        
        if rsi_p < 35 and p_entrada > m200 and macd.iloc[i] > sinal_macd.iloc[i]:
            total_c += 1
            retornos.append(retorno)
            if retorno > 0: acertos_c += 1
                
        elif rsi_p > 70 and (p_entrada < m200 or macd.iloc[i] < sinal_macd.iloc[i]):
            total_v += 1
            if retorno < 0: acertos_v += 1 
    
    taxa_compra = (acertos_c / total_c * 100) if total_c > 0 else 0
    taxa_venda = (acertos_v / total_v * 100) if total_v > 0 else 0
    retorno_medio = (np.mean(retornos) * 100) if retornos else 0
    
    return taxa_compra, taxa_venda, retorno_medio, total_c, total_v

@st.cache_data(ttl=300)
def obter_indices():
    indices = {'Ibovespa': '^BVSP', 'Nasdaq': '^IXIC', 'Dow Jones': '^DJI'}
    resultados = {}
    for nome, ticker in indices.items():
        try:
            data = yf.Ticker(ticker)
            dado = data.history(period='2d')
            if not dado.empty and len(dado) >= 2:
                atual = dado['Close'].iloc[-1]
                anterior = dado['Close'].iloc[-2]
                variacao = ((atual / anterior) - 1) * 100
                resultados[nome] = (atual, variacao)
        except:
            resultados[nome] = (0.0, 0.0)
    return resultados

@st.cache_data(ttl=300)
def obter_cambio():
    # Adicionada Libra e Tickers de Câmbio
    moedas = {'Dólar': 'USDBRL=X', 'Euro': 'EURBRL=X', 'Libra': 'GBPBRL=X', 'Bitcoin': 'BTC-BRL'}
    resultados = {}
    for nome, ticker in moedas.items():
        try:
            data = yf.Ticker(ticker)
            dado = data.history(period='2d')
            if not dado.empty and len(dado) >= 2:
                atual = dado['Close'].iloc[-1]
                anterior = dado['Close'].iloc[-2]
                variacao = ((atual / anterior) - 1) * 100
                resultados[nome] = (atual, variacao)
            else:
                # [CORREÇÃO BITCOIN] Fallback para preço atual se o histórico falhar
                val_atual = data.fast_info.last_price
                resultados[nome] = (val_atual if val_atual is not None else 0.0, 0.0)
        except:
            resultados[nome] = (0.0, 0.0)
    return resultados

@st.cache_data(ttl=600)
def obter_dados_ticker(ticker, mercado):
    try:
        if mercado == "Brasil" and not ticker.endswith(".SA"):
            ticker += ".SA"
        t = yf.Ticker(ticker)
        hist = t.history(period="5y")
        return t.info, hist
    except:
        return None, None

# --- BARRA LATERAL (SIDEBAR) ---
st.sidebar.title("🌎 Mercado e Estratégia")

st.sidebar.subheader("📈 Índices Mundiais")
indices_data = obter_indices()
for nome, (valor, var) in indices_data.items():
    st.sidebar.metric(nome, f"{valor:,.0f} pts", f"{var:.2f}%")
st.sidebar.divider()

st.sidebar.subheader("💱 Câmbio em Tempo Real")
cambio = obter_cambio()
col_c1, col_c2 = st.sidebar.columns(2)
col_c1.metric("Dólar", f"R$ {cambio['Dólar'][0]:.2f}", f"{cambio['Dólar'][1]:.2f}%")
col_c2.metric("Euro", f"R$ {cambio['Euro'][0]:.2f}", f"{cambio['Euro'][1]:.2f}%")

col_c3, col_c4 = st.sidebar.columns(2)
col_c3.metric("Libra", f"R$ {cambio['Libra'][0]:.2f}", f"{cambio['Libra'][1]:.2f}%")
# Bitcoin com formatação de milhares
btc_val = cambio['Bitcoin'][0]
col_c4.metric("Bitcoin", f"R$ {btc_val:,.0f}" if btc_val > 0 else "---", f"{cambio['Bitcoin'][1]:.2f}%")
st.sidebar.divider()

mercado_selecionado = st.sidebar.radio("Escolha o Mercado:", ["Brasil", "EUA"], on_change=ativar_filtros)
estrategia_ativa = st.sidebar.selectbox("Foco da Análise:", ["Análise Técnica + Notícias", "Value Investing (Graham/Buffett)"])
busca_direta = st.sidebar.text_input(f"🔍 Busca Rápida ({mercado_selecionado}):").upper()

with st.sidebar.expander("📊 Filtros de Valuation", expanded=True):
    f_pl = st.slider("P/L Máximo", 0.0, 50.0, 50.0, step=0.5, on_change=ativar_filtros)
    f_pvp = st.slider("P/VP Máximo", 0.0, 10.0, 10.0, step=0.1, on_change=ativar_filtros)
    f_dy = st.slider("DY Mínimo (%)", 0.0, 20.0, 0.0, step=0.5, on_change=ativar_filtros)
    f_div_ebitda = st.slider("Dív.Líq/EBITDA Máximo", 0.0, 15.0, 15.0, step=0.5, on_change=ativar_filtros)

if st.sidebar.button("Resetar Filtros"):
    st.session_state.filtros_ativos = False
    st.rerun()

if mercado_selecionado == "Brasil":
    lista_base = ['PETR4', 'VALE3', 'ITUB4', 'BBAS3', 'BBDC4', 'SANB11', 'B3SA3', 'EGIE3', 'TRPL4', 'TAEE11', 'SAPR11', 'CPLE6', 'ELET3', 'CMIG4', 'SBSP3', 'ABEV3', 'WEGE3', 'RADL3', 'RENT3', 'MGLU3', 'LREN3', 'RAIZ4', 'VBBR3', 'SUZB3', 'KLBN11', 'GOAU4', 'CSNA3', 'PRIO3', 'JBSS3', 'BRFS3', 'GGBR4', 'HAPV3', 'RDOR3']
    moeda_simbolo = "R$"
else:
    lista_base = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'DIS', 'KO', 'PEP', 'MCD', 'NKE', 'WMT', 'JPM', 'V', 'MA', 'BAC', 'PYPL', 'PFE', 'JNJ', 'PG', 'COST', 'ORCL']
    moeda_simbolo = "US$"

# --- CABEÇALHO ---
st.title(f"🤖 Monitor IA - {mercado_selecionado}")
st.caption(f"Atualização: {time.strftime('%H:%M:%S')} | Local: Blumenau/SC")

# --- PROCESSAMENTO ---
tickers_para_processar = [busca_direta] if busca_direta else lista_base
dados_vencedoras = []

for tkr in tickers_para_processar:
    info, hist = obter_dados_ticker(tkr, mercado_selecionado)
    if info and not hist.empty:
        pl = info.get('trailingPE', 0) or 0
        pvp = info.get('priceToBook', 0) or 0
        dy = (info.get('dividendYield', 0) or 0) * 100
        ebitda = info.get('ebitda', 1) or 1
        div_e = (info.get('totalDebt', 0) - info.get('totalCash', 0)) / ebitda
        lpa = info.get('trailingEps', 0) or 0
        vpa = info.get('bookValue', 0) or 0
        p_justo = calcular_graham(lpa, vpa)
        p_atual = hist['Close'].iloc[-1]
        upside = ((p_justo / p_atual) - 1) * 100 if p_justo > 0 else 0
        
        if not busca_direta and st.session_state.filtros_ativos:
            if not (pl <= f_pl and pvp <= f_pvp and dy >= f_dy and div_e <= f_div_ebitda):
                continue

        noticias_texto = ""
        lista_links = []
        try:
            url = f"https://news.google.com/rss/search?q={tkr}&hl={'pt-BR' if mercado_selecionado == 'Brasil' else 'en-US'}"
            feed = feedparser.parse(url)
            for entry in feed.entries[:5]:
                titulo = entry.title.lower()
                noticias_texto += titulo + " "
                lista_links.append({"titulo": entry.title, "link": entry.link})
        except: pass

        score_p = sum(noticias_texto.count(w) for w in ["alta", "lucro", "compra", "subiu", "dividend", "profit", "buy"])
        score_n = sum(noticias_texto.count(w) for w in ["queda", "prejuízo", "venda", "caiu", "risk", "loss", "sell"])
        rsi_val = calcular_rsi(hist['Close'])
        score_value = calcular_score_value(info)
        taxa_compra, taxa_venda, retorno_medio, total_c, total_v = simular_performance_historica(hist)

        motivo_detalhe = ""
        if estrategia_ativa == "Value Investing (Graham/Buffett)":
            if upside > 20 and score_value >= 3:
                veredito, cor = "VALOR ✅", "success"
                motivo_detalhe = f"Forte margem de segurança ({upside:.1f}%) e bons fundamentos (Score: {score_value}/4)."
            elif upside < 0:
                veredito, cor = "CARO 🚨", "error"
                motivo_detalhe = "Preço acima do valor intrínseco de Graham."
            else:
                veredito, cor = "NEUTRO ⚖️", "warning"
                motivo_detalhe = "Ativo próximo ao preço justo, sem margem de segurança clara."
        else:
            if rsi_val > 70 and score_n > score_p:
                veredito, cor = "VENDA 🚨", "error"
                motivo_detalhe = f"RSI alto ({rsi_val:.1f}) e notícias negativas."
            elif rsi_val > 75:
                veredito, cor = "VENDA 🚨", "error"
                motivo_detalhe = f"RSI em nível extremo ({rsi_val:.1f}). Sobrecomprado."
            elif score_p > score_n and rsi_val < 65:
                veredito, cor = "COMPRA ✅", "success"
                motivo_detalhe = f"Notícias positivas e RSI saudável ({rsi_val:.1f})."
            elif score_n > score_p or rsi_val > 70:
                veredito, cor = "CAUTELA ⚠️", "error"
                motivo_detalhe = f"RSI alto ({rsi_val:.1f}) ou sentimento negativo."
            else:
                veredito, cor = "NEUTRO ⚖️", "warning"
                motivo_detalhe = "Indicadores técnicos e notícias em equilíbrio."

        dados_vencedoras.append({
            "Ticker": tkr, "Empresa": info.get('shortName', tkr), "Preço": p_atual,
            "P/L": pl, "DY %": dy, "Dívida": div_e, "Graham": p_justo, "Upside %": upside,
            "Veredito": veredito, "Cor": cor, "Motivo": motivo_detalhe, "RSI": rsi_val, "Hist": hist, "Links": lista_links,
            "TaxaCompra": taxa_compra, "TaxaVenda": taxa_venda, "RetornoMedio": retorno_medio,
            "QtdCompra": total_c, "QtdVenda": total_v, "ValueScore": score_value
        })

# --- INTERFACE ---
if dados_vencedoras:
    st.subheader(f"🏆 Ranking de Oportunidades - Estratégia: {estrategia_ativa}")
    with st.expander("📌 Legenda de Sinais e Vereditos"):
        st.markdown("""
        * **VALOR ✅**: Ativo com desconto frente ao preço de Graham e bons fundamentos.
        * **COMPRA ✅**: RSI baixo/médio com notícias favoráveis.
        * **VENDA / CARO 🚨**: RSI extremo ou preço acima do valor justo.
        * **CAUTELA ⚠️**: Divergência técnica ou sentimental.
        """)

    df_resumo = pd.DataFrame(dados_vencedoras)[["Ticker", "Preço", "DY %", "Upside %", "Veredito", "Motivo", "TaxaCompra", "TaxaVenda"]]
    st.dataframe(
        df_resumo.sort_values(by="Upside %", ascending=False), 
        use_container_width=True, hide_index=True,
        column_config={
            "TaxaCompra": st.column_config.NumberColumn("Assert. Compra (5y)", format="%.1f%%"),
            "TaxaVenda": st.column_config.NumberColumn("Assert. Venda (5y)", format="%.1f%%")
        }
    )

    for acao in dados_vencedoras:
        st.divider()
        col_tit, col_ver, col_acc_c, col_acc_v = st.columns([3, 1, 1, 1])
        col_tit.header(f"🏢 {acao['Empresa']} ({acao['Ticker']})")
        col_acc_c.metric("Assertividade Compra", f"{acao['TaxaCompra']:.1f}%", f"{acao['QtdCompra']} sinais")
        col_acc_v.metric("Assertividade Venda", f"{acao['TaxaVenda']:.1f}%", f"{acao['QtdVenda']} sinais")
        
        if acao["Cor"] == "success": col_ver.success(f"**{acao['Veredito']}**")
        elif acao["Cor"] == "error": col_ver.error(f"**{acao['Veredito']}**")
        else: col_ver.warning(f"**{acao['Veredito']}**")

        st.line_chart(acao['Hist']['Close'])
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Preço Atual", f"{moeda_simbolo} {acao['Preço']:.2f}")
        c2.metric("P/L", round(acao['P/L'], 2))
        c3.metric("DY", f"{acao['DY %']:.2f}%")
        c4.metric("Dív.Líq/EBITDA", round(acao['Dívida'], 2))
        c5.metric("Graham", f"{moeda_simbolo} {acao['Graham']:.2f}", f"{acao['Upside %']:.1f}%")

        with st.expander(f"📊 Detalhes: {acao['Ticker']}"):
            st.write(f"**Fundamentos (Value Score):** {acao['ValueScore']}/4")
            st.progress(acao['ValueScore'] / 4)
            st.write(f"📈 RSI: {acao['RSI']:.2f} | 📝 Motivo: {acao['Motivo']}")
            st.markdown("**Últimas Manchetes:**")
            for n in acao['Links']: st.markdown(f"• [{n['titulo']}]({n['link']})")
else:
    st.info("💡 Use os filtros ou faça uma busca direta.")
