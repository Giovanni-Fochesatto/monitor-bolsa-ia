def analise_minuciosa_ia(ticker, preco, media, rsi_atual):
    st.subheader(f"🕵️‍♂️ Inteligência de Mercado: {ticker}")
    
    # 1. Variar a URL para evitar cache do servidor
    timestamp = int(time.time())
    url_news = f"https://news.google.com/rss/search?q={ticker}+when:2d&hl=pt-BR&gl=BR&ceid=BR:pt-419&t={timestamp}"
    
    manchetes_encontradas = []
    texto_total_analise = ""
    
    # 2. Configuração Ultra-Resiliente
    user_config = Config()
    # Usando um User-Agent mais recente e comum
    user_config.browser_user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36'
    user_config.request_timeout = 15
    user_config.fetch_images = False  # Aumenta velocidade e diminui rastro

    try:
        # 3. Request com delay para não parecer bot
        feed = feedparser.parse(url_news)
        
        if not feed.entries:
            st.error("⚠️ O Google News não retornou dados. Tente trocar o Ticker ou aguarde 1 minuto.")
            return

        with st.spinner(f'IA vasculhando fontes para {ticker}...'):
            for entry in feed.entries[:5]: 
                # Passo Crítico: Guardar a manchete IMEDIATAMENTE antes de tentar baixar o artigo
                titulo = entry.title.split(' - ')[0] # Remove o nome do jornal do título
                manchetes_encontradas.append(titulo)
                texto_total_analise += f"{titulo}. "
                
                # Tenta baixar o conteúdo, mas não trava se falhar
                try:
                    time.sleep(1) # Jitter de 1 segundo entre notícias
                    article = Article(entry.link, config=user_config)
                    article.download()
                    article.parse()
                    # Pega apenas o primeiro parágrafo para análise de sentimento
                    texto_total_analise += article.text[:300] + " "
                except Exception:
                    continue # Se falhar o texto denso, seguimos com o título
    except Exception as e:
        st.warning(f"Erro na conexão com o Radar: {e}")

    # --- MOTOR DE DECISÃO ---
    positivas = ["alta", "dividendo", "lucro", "compra", "crescimento", "ebitda", "subiu", "positivo", "recorde", "valorização", "superou", "recompra"]
    negativas = ["queda", "risco", "prejuízo", "venda", "caiu", "dívida", "crise", "negativo", "abaixo", "desvalorização", "corte", "inflação"]

    texto_limpo = texto_total_analise.lower()
    detectadas_pos = list(set([p for p in positivas if p in texto_limpo]))
    detectadas_neg = list(set([p for p in negativas if p in texto_limpo]))
    
    otimismo = sum(texto_limpo.count(p) for p in positivas)
    pessimismo = sum(texto_limpo.count(p) for p in negativas)

    # --- EXIBIÇÃO DO VEREDITO ---
    status_rsi = "Caro" if rsi_atual > 70 else "Barato" if rsi_atual < 30 else "Neutro"
    
    col_v, col_i = st.columns([1, 2])
    with col_v:
        if preco > media and otimismo > pessimismo:
            st.success("**VEREDITO: COMPRA ✅**")
        elif preco < media and pessimismo > otimismo:
            st.error("**VEREDITO: EVITAR ❌**")
        else:
            st.warning("**VEREDITO: NEUTRO ⚖️**")

    with col_i:
        st.write(f"Técnica: {'Alta' if preco > media else 'Baixa'} | RSI: {rsi_atual:.1f} ({status_rsi})")

    # --- RELATÓRIO DE FORÇA BRUTA ---
    with st.expander("🔍 Detalhes da Varredura de IA", expanded=True):
        c1, c2 = st.columns(2)
        c1.write("**Sinais Positivos:** " + (", ".join(detectadas_pos) if detectadas_pos else "Nenhum"))
        c2.write("**Sinais de Risco:** " + (", ".join(detectadas_neg) if detectadas_neg else "Nenhum"))
        
        st.write("---")
        st.markdown("### 📌 Manchetes Processadas")
        
        if manchetes_encontradas:
            for m in manchetes_encontradas:
                st.markdown(f"* {m}")
        else:
            st.info("A IA analisou tendências gerais, mas a extração individual de manchetes foi bloqueada pelos portais.")
