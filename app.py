import json
import re
import feedparser
import google.generativeai as genai
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(
    page_title="XAU/USD AI Radar Pro", page_icon="🥇", layout="wide"
)

st.title("🥇 Radar Macro, Técnico & Opciones XAU/USD")
st.caption(
    "Noticias (24h) + Oro + DXY + Bonos US02Y + EMAs + Delta & Fuerza +"
    " Pivot Points + Opciones (OI / Put-Call)"
)

api_key = st.secrets.get("GEMINI_API_KEY")

if not api_key:
    st.error("⚠️ No se encontró la API Key en los Secrets de Streamlit.")
else:

    # --- 1. DATOS TÉCNICOS Y MERCADO EXPANDIDO ---
    @st.cache_data(ttl=300)
    def obtener_datos_mercado_completos(ticker_symbol):
        try:
            ticker = yf.Ticker(ticker_symbol)
            df = ticker.history(period="1y").dropna(subset=["Close"])

            # Eliminar filas duplicadas de cierre consecutivas para evitar falsos deltas 0.00 en fuera de horario
            df = df[df["Close"].diff() != 0] if len(df) > 200 else df

            if len(df) >= 200:
                # RSI & Momentum
                delta = df["Close"].diff()
                gain = delta.where(delta > 0, 0).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / loss
                df["RSI"] = 100 - (100 / (1 + rs))
                df["Momentum"] = df["Close"] - df["Close"].shift(14)

                # EMAs
                df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
                df["EMA200"] = df["Close"].ewm(span=200, adjust=False).mean()

                # Volumen Relativo
                df["Vol_MA20"] = df["Volume"].rolling(20).mean()
                vol_actual = (
                    float(df["Volume"].iloc[-1])
                    if "Volume" in df and df["Volume"].iloc[-1] > 0
                    else 0
                )
                vol_ma20 = (
                    float(df["Vol_MA20"].iloc[-1])
                    if "Vol_MA20" in df and df["Vol_MA20"].iloc[-1] > 0
                    else 1
                )
                vol_ratio = (
                    (vol_actual / vol_ma20) if vol_ma20 > 0 else 1.0
                )

                # Precios y Deltas
                precio_actual = float(df["Close"].iloc[-1])
                precio_previo = float(df["Close"].iloc[-2])
                delta_usd = precio_actual - precio_previo
                var_pct = (delta_usd / precio_previo) * 100

                # ATR 14
                df["TR"] = (
                    df[["High", "Close"]].max(axis=1)
                    - df[["Low", "Close"]].min(axis=1)
                )
                df["ATR14"] = df["TR"].rolling(14).mean()
                rango_hoy = float(df["High"].iloc[-1] - df["Low"].iloc[-1])
                atr14 = float(df["ATR14"].iloc[-1])

                rsi_val = float(df["RSI"].iloc[-1])
                mom_val = float(df["Momentum"].iloc[-1])

                ema50_val = float(df["EMA50"].iloc[-1])
                ema200_val = float(df["EMA200"].iloc[-1])

                sobre_ema50 = precio_actual > ema50_val
                sobre_ema200 = precio_actual > ema200_val

                metricas = {
                    "precio": precio_actual,
                    "delta_usd": delta_usd,
                    "var_pct": var_pct,
                    "rsi": rsi_val,
                    "mom": mom_val,
                    "sobre_ema50": sobre_ema50,
                    "sobre_ema200": sobre_ema200,
                    "vol_ratio": vol_ratio,
                    "rango_hoy": rango_hoy,
                    "atr14": atr14,
                    "df": df.tail(120),
                }
                return metricas
        except Exception:
            pass
        return None

    # --- 2. CÁLCULO DE PIVOT POINTS COMPLETO ---
    @st.cache_data(ttl=300)
    def calcular_pivot_points():
        tickers_oro = ["GC=F", "XAUUSD=X"]
        df = None

        for t in tickers_oro:
            try:
                gold = yf.Ticker(t)
                data = gold.history(period="5d", interval="1d").dropna()
                if len(data) >= 2:
                    df = data
                    break
            except Exception:
                continue

        if df is None or len(df) < 2:
            return [], 0.0

        prev_session = df.iloc[-2]
        high = float(prev_session["High"])
        low = float(prev_session["Low"])
        close = float(prev_session["Close"])
        precio_ref = float(df["Close"].iloc[-1])

        rango = high - low
        pp = (high + low + close) / 3.0

        niveles_lista = [
            ("Extensión Fib 1.618 (R6)", round(close + (rango * 1.618), 2), 6),
            ("Extensión Fib 1.168 (R5)", round(close + (rango * 1.168), 2), 5),
            ("Breakout Camarilla (R4)", round(close + (rango * 1.1 / 2), 2), 4),
            ("Resistencia 3 (R3)", round(high + 2 * (pp - low), 2), 3),
            ("Resistencia 2 (R2)", round(pp + rango, 2), 2),
            ("Resistencia 1 (R1)", round((2 * pp) - low, 2), 1),
            ("Punto Pivote (PP)", round(pp, 2), 0),
            ("Soporte 1 (S1)", round((2 * pp) - high, 2), 1),
            ("Soporte 2 (S2)", round(pp - rango, 2), 2),
            ("Soporte 3 (S3)", round(low - 2 * (high - pp), 2), 3),
            ("Breakout Camarilla (S4)", round(close - (rango * 1.1 / 2), 2), 4),
            ("Soporte Fib 1.168 (S5)", round(close - (rango * 1.168), 2), 5),
            ("Soporte Fib 1.618 (S6)", round(close - (rango * 1.618), 2), 6),
        ]

        return niveles_lista, round(precio_ref, 2)

    # --- 3. OPEN INTEREST & PUT/CALL RATIO EN OPCIONES ---
    @st.cache_data(ttl=1800)
    def obtener_open_interest_opciones(precio_oro_ref, rango_pct=0.02):
        try:
            gld = yf.Ticker("GLD")
            df_gld = gld.history(period="1d")

            if df_gld.empty:
                return None, None, None, None

            gld_price = float(df_gld["Close"].iloc[-1])

            if precio_oro_ref <= 0:
                precio_oro_ref = gld_price * 10.8

            ratio = precio_oro_ref / gld_price

            expiraciones = gld.options
            if not expiraciones:
                return None, None, None, None

            prox_exp = expiraciones[0]
            opt = gld.option_chain(prox_exp)

            total_call_oi = opt.calls["openInterest"].sum()
            total_put_oi = opt.puts["openInterest"].sum()
            pc_ratio = (
                round(total_put_oi / total_call_oi, 2)
                if total_call_oi > 0
                else 1.0
            )

            limite_sup = gld_price * (1 + rango_pct)
            limite_inf = gld_price * (1 - rango_pct)

            otm_calls = opt.calls[
                (opt.calls["strike"] > gld_price)
                & (opt.calls["strike"] <= limite_sup)
            ]
            otm_puts = opt.puts[
                (opt.puts["strike"] < gld_price)
                & (opt.puts["strike"] >= limite_inf)
            ]

            if otm_calls.empty:
                otm_calls = opt.calls[opt.calls["strike"] > gld_price]
            if otm_puts.empty:
                otm_puts = opt.puts[opt.puts["strike"] < gld_price]

            max_call_row = otm_calls.loc[otm_calls["openInterest"].idxmax()]
            max_put_row = otm_puts.loc[otm_puts["openInterest"].idxmax()]

            strike_call_gold = round(max_call_row["strike"] * ratio, 0)
            strike_put_gold = round(max_put_row["strike"] * ratio, 0)

            return strike_call_gold, strike_put_gold, prox_exp, pc_ratio
        except Exception:
            return None, None, None, None

    # --- 4. CONSULTA A GEMINI ---
    @st.cache_data(ttl=1800)
    def consultar_gemini(prompt_text, key):
        genai.configure(api_key=key)

        modelos_candidatos = [
            "gemini-2.5-flash",
            "gemini-3.6-flash",
            "gemini-2.5-pro",
        ]
        try:
            modelos_api = [
                m.name.replace("models/", "")
                for m in genai.list_models()
                if "generateContent" in m.supported_generation_methods
            ]
            if modelos_api:
                flash_models = [m for m in modelos_api if "flash" in m]
                modelos_candidatos = (
                    flash_models + modelos_api + modelos_candidatos
                )
        except Exception:
            pass

        modelos_unicos = list(dict.fromkeys(modelos_candidatos))

        for mod in modelos_unicos:
            try:
                model = genai.GenerativeModel(mod)
                response = model.generate_content(
                    prompt_text,
                    generation_config={
                        "response_mime_type": "application/json"
                    },
                )
                if response and response.text:
                    return response.text
            except Exception:
                continue

        raise Exception("Error al conectar con la API de IA.")

    # Cargar datos
    oro_m = obtener_datos_mercado_completos("GC=F")
    dxy_m = obtener_datos_mercado_completos("DX-Y.NYB")
    us02_m = obtener_datos_mercado_completos("2YY=F")

    niveles_pivots, precio_ref = calcular_pivot_points()

    # --- BARRA LATERAL ---
    st.sidebar.header("⚙️ Configuración del Radar")
    sensibilidad_opc = (
        st.sidebar.slider(
            "Sensibilidad Muros OTM (% spot)",
            min_value=0.5,
            max_value=5.0,
            value=2.0,
            step=0.5,
        )
        / 100.0
    )

    max_niveles_pivots = st.sidebar.slider(
        "Niveles Pivote a mostrar (R/S)",
        min_value=3,
        max_value=6,
        value=4,
        step=1,
    )

    max_call_strike, max_put_strike, fecha_exp, pc_ratio = (
        obtener_open_interest_opciones(
            precio_ref, rango_pct=sensibilidad_opc
        )
    )

    def txt_ema(sobre_50, sobre_200):
        t50 = "🟢 >EMA50" if sobre_50 else "🔴 <EMA50"
        t200 = "🟢 >EMA200" if sobre_200 else "🔴 <EMA200"
        return f"{t50} | {t200}"

    # --- BLOQUE VISUAL SUPERIOR ---
    col1, col2, col3 = st.columns(3)

    if oro_m:
        with col1:
            st.metric(
                "Oro (GC Futures)",
                f"${oro_m['precio']:.2f}",
                f"{oro_m['delta_usd']:+.2f} ({oro_m['var_pct']:+.2f}%)",
            )
            st.caption(
                f"RSI: **{oro_m['rsi']:.1f}** | Mom: **{oro_m['mom']:.2f}**"
            )
            st.caption(txt_ema(oro_m["sobre_ema50"], oro_m["sobre_ema200"]))

    if dxy_m:
        with col2:
            st.metric(
                "DXY (Dólar)",
                f"{dxy_m['precio']:.2f}",
                f"{dxy_m['delta_usd']:+.2f} ({dxy_m['var_pct']:+.2f}%)",
                delta_color="inverse",
            )
            st.caption(
                f"RSI: **{dxy_m['rsi']:.1f}** | Mom: **{dxy_m['mom']:.2f}**"
            )
            st.caption(txt_ema(dxy_m["sobre_ema50"], dxy_m["sobre_ema200"]))

    if us02_m:
        with col3:
            st.metric(
                "US02Y (Bono 2Y)",
                f"{us02_m['precio']:.2f}%",
                f"{us02_m['delta_usd']:+.2f}% ({us02_m['var_pct']:+.2f}%)",
                delta_color="inverse",
            )
            st.caption(
                f"RSI: **{us02_m['rsi']:.1f}** | Mom: **{us02_m['mom']:.2f}**"
            )
            st.caption(txt_ema(us02_m["sobre_ema50"], us02_m["sobre_ema200"]))

    st.divider()

    # --- SECCIÓN CORREGIDA: MEDIDOR DE FUERZA, DELTA & BARRAS COLORIADAS ---
    st.subheader("⚡ Medidor de Fuerza, Delta Diaria & Volumen")

    f_col1, f_col2, f_col3 = st.columns(3)

    def render_barra_fuerza(titulo, m_data):
        if not m_data:
            return
        st.markdown(f"### {titulo}")

        # Corrección del Estado del Delta
        val_delta = m_data["var_pct"]
        if abs(val_delta) < 0.001:
            color_delta = "⚪ Neutral"
        elif val_delta > 0:
            color_delta = "🟢 Alcista"
        else:
            color_delta = "🔴 Bajista"

        st.write(
            f"- **Delta Diario:** `{m_data['delta_usd']:+.2f}`"
            f" ({val_delta:+.2f}%) {color_delta}"
        )

        # Volumen Relativo
        v_rat = m_data["vol_ratio"]
        txt_vol = (
            "🔥 Alto Vol."
            if v_rat > 1.2
            else ("⚠️ Bajo Vol." if v_rat < 0.8 else "Normal")
        )
        st.write(
            f"- **Volumen vs Media 20:** `{v_rat:.2f}x` media ({txt_vol})"
        )

        # Recorrido ATR
        rango = m_data["rango_hoy"]
        atr = m_data["atr14"]
        pct_atr = (rango / atr * 100) if atr > 0 else 100
        st.write(
            f"- **Recorrido Hoy:** `${rango:.2f}` ({pct_atr:.0f}% del ATR 14d)"
        )

        # Barra Dinámica Verde (Compradora) vs Roja (Vendedora)
        rsi_val = m_data["rsi"]
        if rsi_val >= 50:
            color_hex = "#22c55e"  # Verde brillante
            label_tipo = "Compradora 🟢"
        else:
            color_hex = "#ef4444"  # Rojo brillante
            label_tipo = "Vendedora 🔴"

        st.markdown(f"**Presión {label_tipo} ({rsi_val:.0f}/100)**")

        # HTML personalizado para forzar el color de la barra
        html_barra = f"""
        <div style="background-color: #262730; border-radius: 6px; width: 100%; height: 12px; margin-top: 4px; margin-bottom: 12px;">
            <div style="background-color: {color_hex}; width: {min(max(rsi_val, 0), 100)}%; height: 100%; border-radius: 6px; transition: width 0.5s;"></div>
        </div>
        """
        st.markdown(html_barra, unsafe_allow_html=True)

    with f_col1:
        render_barra_fuerza("🥇 Oro (GC)", oro_m)
    with f_col2:
        render_barra_fuerza("💵 DXY (Dólar)", dxy_m)
    with f_col3:
        render_barra_fuerza("🏛️ US02Y (Bono 2Y)", us02_m)

    st.divider()

    # --- PIVOTS Y OPEN INTEREST ---
    st.subheader("🎯 Niveles Clave & Muros Tácticos")
    if precio_ref > 0:
        st.caption(f"Precio Actual de Referencia: **${precio_ref}**")

    p_col1, p_col2 = st.columns(2)

    with p_col1:
        st.markdown(
            f"**Niveles Pivote (Hasta R{max_niveles_pivots}/S{max_niveles_pivots})**"
        )
        if niveles_pivots:
            niveles_filtrados = [
                item
                for item in niveles_pivots
                if item[2] <= max_niveles_pivots
            ]
            por_encima = [
                item for item in niveles_filtrados if item[1] > precio_ref
            ]
            por_debajo = [
                item for item in niveles_filtrados if item[1] <= precio_ref
            ]

            if por_encima:
                st.write("🔴 **Resistencias / Objetivos:**")
                for nombre, valor, _ in sorted(
                    por_encima, key=lambda x: x[1]
                ):
                    st.write(f"- {nombre}: `${valor}`")

            if por_debajo:
                st.write("🟢 **Soportes / Retesteos:**")
                for nombre, valor, _ in sorted(
                    por_debajo, key=lambda x: x[1], reverse=True
                ):
                    st.write(f"- {nombre}: `${valor}`")

    with p_col2:
        st.markdown(
            f"**Muros Tácticos Cercanos (±{sensibilidad_opc*100:.1f}%)**"
        )
        if max_call_strike and max_put_strike:
            st.write(
                "- **Resistencia Táctica (Call):**"
                f" `~${max_call_strike:.0f}`"
            )
            st.write(
                f"- **Soporte Táctico (Put):** `~${max_put_strike:.0f}`"
            )
            if pc_ratio is not None:
                bias_pc = (
                    "🟢 Alcista (Acumulación Calls)"
                    if pc_ratio < 0.8
                    else ("🔴 Bajista / Cobertura Puts" if pc_ratio > 1.1 else "⚪ Neutral")
                )
                st.write(f"- **Put/Call OI Ratio:** `{pc_ratio}` ({bias_pc})")
            st.caption(f"Vencimiento analizado: {fecha_exp}")
        else:
            st.info("No se pudieron calcular muros OTM en este momento.")

    st.divider()

    # --- GRÁFICOS INTERACTIVOS ---
    st.subheader("📈 Gráficos de Tendencia & EMAs (50 y 200)")

    tab_gold, tab_dxy, tab_us02 = st.tabs(
        ["🥇 Oro (GC)", "💵 DXY (Dólar)", "🏛️ US02Y (Bono 2Y)"]
    )

    def crear_grafico_interactivo(ticker_name, data_m):
        if not data_m or "df" not in data_m:
            st.info("Gráfico no disponible.")
            return

        df_chart = data_m["df"]
        fig_chart = go.Figure()

        fig_chart.add_trace(
            go.Candlestick(
                x=df_chart.index,
                open=df_chart["Open"],
                high=df_chart["High"],
                low=df_chart["Low"],
                close=df_chart["Close"],
                name="Precio",
            )
        )

        fig_chart.add_trace(
            go.Scatter(
                x=df_chart.index,
                y=df_chart["EMA50"],
                line=dict(color="#FFD700", width=1.5),
                name="EMA 50",
            )
        )

        fig_chart.add_trace(
            go.Scatter(
                x=df_chart.index,
                y=df_chart["EMA200"],
                line=dict(color="#FF4500", width=2),
                name="EMA 200",
            )
        )

        fig_chart.update_layout(
            title=f"Estructura Técnica {ticker_name} (Diario)",
            template="plotly_dark",
            xaxis_rangeslider_visible=False,
            height=380,
            margin=dict(l=10, r=10, t=40, b=10),
        )

        st.plotly_chart(fig_chart, use_container_width=True)

    with tab_gold:
        crear_grafico_interactivo("Oro (GC Futures)", oro_m)

    with tab_dxy:
        crear_grafico_interactivo("Índice Dólar (DXY)", dxy_m)

    with tab_us02:
        crear_grafico_interactivo("Bono EEUU 2 Años (US02Y)", us02_m)

    st.divider()

    # --- NOTICIAS Y GEMINI IA ---
    rss_url = "https://news.google.com/rss/search?q=gold+price+XAUUSD+when:1d&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(rss_url)

    if st.button("🔄 Actualizar Análisis Completo"):
        st.cache_data.clear()
        st.rerun()

    noticias_ordenadas = sorted(
        feed.entries,
        key=lambda x: getattr(x, "published_parsed", 0),
        reverse=True,
    )
    noticias = noticias_ordenadas[:5]

    texto_titulares = ""
    for i, entry in enumerate(noticias, 1):
        texto_titulares += (
            f"\n{i}. Titular: {entry.title}\n   Enlace: {entry.link}\n"
        )

    prompt = f"""
    Eres un analista macroeconómico, técnico y de flujo de opciones senior de XAU/USD (Oro).

    DATOS EN VIVO DEL MERCADO:
    1. Oro (GC): ${oro_m['precio']:.2f} | Delta: {oro_m['var_pct']:+.2f}% | Vol.Rel: {oro_m['vol_ratio']:.2f}x | RSI: {oro_m['rsi']:.1f} | >EMA50: {oro_m['sobre_ema50']} | >EMA200: {oro_m['sobre_ema200']}
    2. DXY (Dólar): {dxy_m['precio']:.2f} | Delta: {dxy_m['var_pct']:+.2f}% | RSI: {dxy_m['rsi']:.1f} | >EMA50: {dxy_m['sobre_ema50']} | >EMA200: {dxy_m['sobre_ema200']}
    3. US02Y (Bono 2Y): {us02_m['precio']:.2f}% | Delta: {us02_m['var_pct']:+.2f}% | RSI: {us02_m['rsi']:.1f} | >EMA50: {us02_m['sobre_ema50']} | >EMA200: {us02_m['sobre_ema200']}
    4. PIVOTS MOSTRADOS EN UI: {niveles_filtrados if 'niveles_filtrados' in locals() else niveles_pivots}
    5. OPCIONES TÁCTICAS: Call Muro: ~${max_call_strike} | Put Muro: ~${max_put_strike} | Put/Call Ratio: {pc_ratio}

    NOTICIAS MACRO RECIENTES (24h):
    {texto_titulares}

    ANÁLISIS REQUERIDO:
    Evalúa la fuerza del Oro combinando noticias, Delta/Volumen de DXY y Bonos, EMAs 50/200, Put/Call Ratio y Muros de Opciones.

    Responde en formato JSON strictly válido con esta estructura:
    {{
        "score_global": 75,
        "estado": "Fortaleza Alcista Institutional",
        "factores": {{
            "Presión DXY (Dólar)": 80,
            "Rendimiento Bono 2Y": 70,
            "Impulso y Delta del Oro": 85,
            "Sesgo Opciones (Put/Call Ratio)": 75
        }},
        "noticias": [
            {{
                "titulo": "Título traducido al español",
                "sesgo": "Alcista 🟢",
                "explicacion": "Explicación breve.",
                "url": "URL original"
            }}
        ]
    }}
    """

    with st.spinner("Analizando Macro, Deltas, Gráficos e IA..."):
        try:
            raw_response = consultar_gemini(prompt, api_key)

            match = re.search(r"\{.*\}", raw_response, re.DOTALL)
            json_str = match.group(0) if match else raw_response
            data = json.loads(json_str)

            score = data["score_global"]
            estado = data["estado"]

            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=score,
                    title={
                        "text": (
                            "<b>Sentimiento Global Macro + Delta +"
                            f" Opciones</b><br><span style='font-size:0.8em;color:gray'>{estado}</span>"
                        )
                    },
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#FFFFFF"},
                        "steps": [
                            {"range": [0, 35], "color": "#FF4B4B"},
                            {"range": [35, 65], "color": "#FFA500"},
                            {"range": [65, 100], "color": "#00CC96"},
                        ],
                    },
                )
            )
            fig.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("📊 Desglose de Factores")
            for factor, valor in data["factores"].items():
                st.write(f"**{factor}:** {valor}/100")
                st.progress(valor / 100)

            st.divider()

            st.subheader("Últimos titulares procesados (24h)")
            for item in data["noticias"]:
                st.markdown(f"### {item['titulo']}")
                st.write(f"- **Sesgo:** {item['sesgo']}")
                st.write(f"- **Explicación:** {item['explicacion']}")
                st.markdown(f"- [Leer noticia original]({item['url']})")
                st.write("---")

        except Exception as e:
            st.error(f"Error procesando la información con la IA: {e}")
