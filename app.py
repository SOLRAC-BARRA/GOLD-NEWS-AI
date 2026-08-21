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
    "Noticias (24h)"
)

api_key = st.secrets.get("GEMINI_API_KEY")

if not api_key:
    st.error("⚠️ No se encontró la API Key en los Secrets de Streamlit.")
else:

    # --- 1. DATOS TÉCNICOS Y MERCADO (Con EMA 50 y EMA 200) ---
    @st.cache_data(ttl=300)
    def obtener_datos_mercado(ticker_symbol):
        try:
            ticker = yf.Ticker(ticker_symbol)
            # Se requiere mínimo 1 año para calcular EMA 200 con precisión
            df = ticker.history(period="1y")
            if len(df) >= 200:
                delta = df["Close"].diff()
                gain = delta.where(delta > 0, 0).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / loss
                df["RSI"] = 100 - (100 / (1 + rs))
                df["Momentum"] = df["Close"] - df["Close"].shift(14)

                # Cálculo de EMAs
                df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
                df["EMA200"] = df["Close"].ewm(span=200, adjust=False).mean()

                precio_actual = float(df["Close"].iloc[-1])
                precio_previo = float(df["Close"].iloc[-2])
                var_pct = ((precio_actual - precio_previo) / precio_previo) * 100
                rsi_val = float(df["RSI"].iloc[-1])
                mom_val = float(df["Momentum"].iloc[-1])

                ema50_val = float(df["EMA50"].iloc[-1])
                ema200_val = float(df["EMA200"].iloc[-1])

                sobre_ema50 = precio_actual > ema50_val
                sobre_ema200 = precio_actual > ema200_val

                return (
                    precio_actual,
                    var_pct,
                    rsi_val,
                    mom_val,
                    sobre_ema50,
                    sobre_ema200,
                )
        except Exception:
            pass
        return 0.0, 0.0, 50.0, 0.0, True, True

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

    # --- 3. OPEN INTEREST EN OPCIONES ---
    @st.cache_data(ttl=1800)
    def obtener_open_interest_opciones(precio_oro_ref, rango_pct=0.02):
        try:
            gld = yf.Ticker("GLD")
            df_gld = gld.history(period="1d")

            if df_gld.empty:
                return None, None, None

            gld_price = float(df_gld["Close"].iloc[-1])

            if precio_oro_ref <= 0:
                precio_oro_ref = gld_price * 10.8

            ratio = precio_oro_ref / gld_price

            expiraciones = gld.options
            if not expiraciones:
                return None, None, None

            prox_exp = expiraciones[0]
            opt = gld.option_chain(prox_exp)

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

            return strike_call_gold, strike_put_gold, prox_exp
        except Exception:
            return None, None, None

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

    # Cargar datos de mercado
    oro_p, oro_v, oro_rsi, oro_mom, oro_e50, oro_e200 = obtener_datos_mercado(
        "GC=F"
    )
    dxy_p, dxy_v, dxy_rsi, dxy_mom, dxy_e50, dxy_e200 = obtener_datos_mercado(
        "DX-Y.NYB"
    )
    us02_p, us02_v, us02_rsi, us02_mom, us02_e50, us02_e200 = (
        obtener_datos_mercado("2YY=F")
    )

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

    max_call_strike, max_put_strike, fecha_exp = (
        obtener_open_interest_opciones(
            precio_ref, rango_pct=sensibilidad_opc
        )
    )

    # --- BLOQUE VISUAL SUPERIOR (3 COLUMNAS CON EMAs) ---
    col1, col2, col3 = st.columns(3)

    # Helper para renderizar formato de EMAs
    def txt_ema(sobre_50, sobre_200):
        t50 = "🟢 >EMA50" if sobre_50 else "🔴 <EMA50"
        t200 = "🟢 >EMA200" if sobre_200 else "🔴 <EMA200"
        return f"{t50} | {t200}"

    with col1:
        st.metric("Oro (GC Futures)", f"${oro_p:.2f}", f"{oro_v:.2f}%")
        st.caption(f"RSI: **{oro_rsi:.1f}** | Mom: **{oro_mom:.2f}**")
        st.caption(txt_ema(oro_e50, oro_e200))

    with col2:
        st.metric(
            "DXY (Dólar)",
            f"{dxy_p:.2f}",
            f"{dxy_v:.2f}%",
            delta_color="inverse",
        )
        st.caption(f"RSI: **{dxy_rsi:.1f}** | Mom: **{dxy_mom:.2f}**")
        st.caption(txt_ema(dxy_e50, dxy_e200))

    with col3:
        st.metric(
            "US02Y (Bono 2Y)",
            f"{us02_p:.2f}%",
            f"{us02_v:.2f}%",
            delta_color="inverse",
        )
        st.caption(f"RSI: **{us02_rsi:.1f}** | Mom: **{us02_mom:.2f}**")
        st.caption(txt_ema(us02_e50, us02_e200))

    st.divider()

    # --- BLOQUE PIVOTS Y OPEN INTEREST ---
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
            st.caption(f"Vencimiento analizado: {fecha_exp}")
        else:
            st.info("No se pudieron calcular muros OTM en este momento.")

    st.divider()

    # Noticias RSS
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
    1. Oro (GC): ${oro_p:.2f} | RSI: {oro_rsi:.1f} | >EMA50: {oro_e50} | >EMA200: {oro_e200}
    2. DXY (Dólar): {dxy_p:.2f} | RSI: {dxy_rsi:.1f} | >EMA50: {dxy_e50} | >EMA200: {dxy_e200}
    3. US02Y (Bono 2Y): {us02_p:.2f}% | RSI: {us02_rsi:.1f} | >EMA50: {us02_e50} | >EMA200: {us02_e200}
    4. PIVOTS MOSTRADOS EN UI: {niveles_filtrados if 'niveles_filtrados' in locals() else niveles_pivots}
    5. OPCIONES TÁCTICAS (OPEN INTEREST): Muro Resistencia Call: ~${max_call_strike} | Muro Soporte Put: ~${max_put_strike}

    NOTICIAS MACRO RECIENTES (24h):
    {texto_titulares}

    ANÁLISIS REQUERIDO:
    Evalúa la fuerza del Oro combinando noticias, DXY, Bonos, su posición respecto a las EMAs de 50/200, la estructura de niveles y los muros de opciones cercanos.

    Responde en formato JSON estrictamente válido con esta estructura:
    {{
        "score_global": 75,
        "estado": "Fortaleza Alcista",
        "factores": {{
            "Presión DXY (Dólar)": 80,
            "Rendimiento Bono 2Y": 70,
            "Alineación Estructura EMAs": 85,
            "Barrera de Opciones Tácticas": 65
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

    with st.spinner("Analizando Macro, Niveles de Opciones e IA..."):
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
                            "<b>Sentimiento Global Macro + Técnico +"
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
