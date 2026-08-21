import json
import re
import feedparser
import google.generativeai as genai
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(
    page_title="XAU/USD AI Radar Pro", page_icon="🥇", layout="centered"
)

st.title("🥇 Radar Macro, Técnico & Opciones XAU/USD")
st.caption(
    "Noticias (24h) + DXY + Bonos US02Y + Pivot Points + Open Interest (Opciones)"
)

api_key = st.secrets.get("GEMINI_API_KEY")

if not api_key:
    st.error("⚠️ No se encontró la API Key en los Secrets de Streamlit.")
else:

    # --- 1. DATOS TÉCNICOS Y MERCADO ---
    @st.cache_data(ttl=600)
    def obtener_datos_mercado(ticker_symbol):
        try:
            ticker = yf.Ticker(ticker_symbol)
            df = ticker.history(period="1mo")
            if len(df) >= 15:
                delta = df["Close"].diff()
                gain = delta.where(delta > 0, 0).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / loss
                df["RSI"] = 100 - (100 / (1 + rs))
                df["Momentum"] = df["Close"] - df["Close"].shift(14)

                precio_actual = float(df["Close"].iloc[-1])
                precio_previo = float(df["Close"].iloc[-2])
                var_pct = (
                    (precio_actual - precio_previo) / precio_previo
                ) * 100
                rsi_val = float(df["RSI"].iloc[-1])
                mom_val = float(df["Momentum"].iloc[-1])

                return precio_actual, var_pct, rsi_val, mom_val
        except Exception:
            pass
        return 0.0, 0.0, 50.0, 0.0

    # --- 2. CÁLCULO DE PIVOT POINTS (Futuros GC=F) ---
    @st.cache_data(ttl=1800)
    def calcular_pivot_points():
        try:
            gold = yf.Ticker("GC=F")
            df = gold.history(period="5d")
            if len(df) < 2:
                return {
                    "PP": 0,
                    "R1": 0,
                    "S1": 0,
                    "R2": 0,
                    "S2": 0,
                    "Precio_Ref": 0,
                }

            high = float(df["High"].iloc[-2])
            low = float(df["Low"].iloc[-2])
            close = float(df["Close"].iloc[-2])
            precio_ref = float(df["Close"].iloc[-1])

            pp = (high + low + close) / 3
            r1 = (2 * pp) - low
            s1 = (2 * pp) - high
            r2 = pp + (high - low)
            s2 = pp - (high - low)

            return {
                "PP": round(pp, 2),
                "R1": round(r1, 2),
                "S1": round(s1, 2),
                "R2": round(r2, 2),
                "S2": round(s2, 2),
                "Precio_Ref": round(precio_ref, 2),
            }
        except Exception:
            return {
                "PP": 0,
                "R1": 0,
                "S1": 0,
                "R2": 0,
                "S2": 0,
                "Precio_Ref": 0,
            }

    # --- 3. OPEN INTEREST EN OPCIONES (Muros OTM reales) ---
    @st.cache_data(ttl=3600)
    def obtener_open_interest_opciones(precio_oro_ref):
        try:
            gld = yf.Ticker("GLD")
            df_gld = gld.history(period="1d")
            if df_gld.empty or precio_oro_ref <= 0:
                return None, None, None

            gld_price = float(df_gld["Close"].iloc[-1])

            # Ratio de conversión dinámico
            ratio = precio_oro_ref / gld_price

            expiraciones = gld.options
            if not expiraciones:
                return None, None, None

            prox_exp = expiraciones[0]
            opt = gld.option_chain(prox_exp)

            # Filtrar estrictamente Out-Of-The-Money (OTM)
            otm_calls = opt.calls[opt.calls["strike"] >= gld_price]
            otm_puts = opt.puts[opt.puts["strike"] <= gld_price]

            if otm_calls.empty or otm_puts.empty:
                return None, None, None

            max_call_row = otm_calls.loc[otm_calls["openInterest"].idxmax()]
            max_put_row = otm_puts.loc[otm_puts["openInterest"].idxmax()]

            strike_call_gold = round(max_call_row["strike"] * ratio, 0)
            strike_put_gold = round(max_put_row["strike"] * ratio, 0)

            return strike_call_gold, strike_put_gold, prox_exp
        except Exception:
            return None, None, None

    # --- 4. CONSULTA DINÁMICA A GEMINI ---
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
    dxy_precio, dxy_var, dxy_rsi, dxy_mom = obtener_datos_mercado("DX-Y.NYB")
    us02_precio, us02_var, us02_rsi, us02_mom = obtener_datos_mercado("2YY=F")
    pivots = calcular_pivot_points()
    max_call_strike, max_put_strike, fecha_exp = (
        obtener_open_interest_opciones(pivots["Precio_Ref"])
    )

    # --- BLOQUE VISUAL SUPERIOR ---
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            "DXY (Dólar)",
            f"{dxy_precio:.2f}",
            f"{dxy_var:.2f}%",
            delta_color="inverse",
        )
        st.caption(f"RSI: **{dxy_rsi:.1f}** | Mom: **{dxy_mom:.2f}**")
    with col2:
        st.metric(
            "US02Y (Bono 2Y)",
            f"{us02_precio:.2f}%",
            f"{us02_var:.2f}%",
            delta_color="inverse",
        )
        st.caption(f"RSI: **{us02_rsi:.1f}** | Mom: **{us02_mom:.2f}**")

    st.divider()

    # --- BLOQUE PIVOTS Y OPEN INTEREST ---
    st.subheader("🎯 Niveles Clave & Muros de Opciones")
    if pivots["Precio_Ref"] > 0:
        st.caption(
            f"Precio de referencia de Futuros (GC=F): **${pivots['Precio_Ref']}**"
        )

    p_col1, p_col2 = st.columns(2)

    with p_col1:
        st.markdown("**Pivot Points (Rango Esperado)**")
        st.write(f"- **Resistencia 2 (R2):** `${pivots['R2']}`")
        st.write(f"- **Resistencia 1 (R1):** `${pivots['R1']}`")
        st.write(f"- **Punto Pivote (PP):** `${pivots['PP']}`")
        st.write(f"- **Soporte 1 (S1):** `${pivots['S1']}`")
        st.write(f"- **Soporte 2 (S2):** `${pivots['S2']}`")

    with p_col2:
        st.markdown("**Muros de Open Interest (Opciones GLD)**")
        if max_call_strike and max_put_strike:
            st.write(
                f"- **Resistencia OTM (Max Call OI):** `~${max_call_strike:.0f}`"
            )
            st.write(
                f"- **Soporte OTM (Max Put OI):** `~${max_put_strike:.0f}`"
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
    1. Precio Futuro Referencia (GC=F): ${pivots['Precio_Ref']}
    2. DXY (Dólar): {dxy_precio:.2f} | RSI: {dxy_rsi:.1f}
    3. US02Y (Bono 2Y): {us02_precio:.2f}% | RSI: {us02_rsi:.1f}
    4. PIVOT POINTS DIARIOS: Rango S1 ({pivots['S1']}) a R1 ({pivots['R1']}) | Pivote Central: {pivots['PP']}
    5. OPCIONES (OPEN INTEREST OTM): Resistencia Call: ~${max_call_strike} | Soporte Put: ~${max_put_strike}

    NOTICIAS MACRO RECIENTES (24h):
    {texto_titulares}

    ANÁLISIS REQUERIDO:
    Evalúa la fuerza del Oro combinando las noticias, DXY, Bonos, el rango de Pivots y los muros de opciones OTM.

    Responde en formato JSON estrictamente válido con esta estructura:
    {{
        "score_global": 75,
        "estado": "Fortaleza Alcista",
        "factores": {{
            "Presión DXY (Dólar)": 80,
            "Rendimiento Bono 2Y": 70,
            "Respeto a Pivot Points": 85,
            "Barrera de Opciones OTM": 65
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
