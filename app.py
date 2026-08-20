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

st.title("🥇 Radar Macro & Técnico XAU/USD")
st.caption("Noticias en vivo (24h) + DXY + Bonos a 2 Años (US02Y) + RSI/Momentum")

api_key = st.secrets.get("GEMINI_API_KEY")

if not api_key:
    st.error("⚠️ No se encontró la API Key en los Secrets de Streamlit.")
else:

    # --- DATOS DE MERCADO ---
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
                var_pct = ((precio_actual - precio_previo) / precio_previo) * 100
                rsi_val = float(df["RSI"].iloc[-1])
                mom_val = float(df["Momentum"].iloc[-1])

                return precio_actual, var_pct, rsi_val, mom_val
        except Exception:
            pass
        return 0.0, 0.0, 50.0, 0.0

    # --- CONSULTA A GEMINI CON AUTODETECCIÓN ---
    @st.cache_data(ttl=1800)
    def consultar_gemini(prompt_text, key):
        genai.configure(api_key=key)

        # Buscar modelos disponibles activados en la clave
        modelos_disponibles = []
        try:
            for m in genai.list_models():
                if "generateContent" in m.supported_generation_methods:
                    modelos_disponibles.append(m.name)
        except Exception as e:
            raise Exception(f"La API Key no tiene la API activada en Google Cloud: {e}")

        if not modelos_disponibles:
            raise Exception("No se encontraron modelos disponibles para esta clave API.")

        # Seleccionar el primer modelo disponible (dando preferencia a flash)
        modelo_elegido = modelos_disponibles[0]
        for m in modelos_disponibles:
            if "flash" in m:
                modelo_elegido = m
                break

        model = genai.GenerativeModel(modelo_elegido)
        
        try:
            response = model.generate_content(
                prompt_text,
                generation_config={"response_mime_type": "application/json"}
            )
        except Exception:
            response = model.generate_content(prompt_text)

        return response.text

    # Cargar datos
    dxy_precio, dxy_var, dxy_rsi, dxy_mom = obtener_datos_mercado("DX-Y.NYB")
    us02_precio, us02_var, us02_rsi, us02_mom = obtener_datos_mercado("2YY=F")

    # Métrica visual superior
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            "DXY (Dólar)",
            f"{dxy_precio:.2f}",
            f"{dxy_var:.2f}%",
            delta_color="inverse",
        )
        st.caption(f"RSI(14): **{dxy_rsi:.1f}** | Momentum: **{dxy_mom:.2f}**")
    with col2:
        st.metric(
            "US02Y (Bono 2 Años)",
            f"{us02_precio:.2f}%",
            f"{us02_var:.2f}%",
            delta_color="inverse",
        )
        st.caption(f"RSI(14): **{us02_rsi:.1f}** | Momentum: **{us02_mom:.2f}**")

    # Obtener Noticias
    rss_url = "https://news.google.com/rss/search?q=gold+price+XAUUSD+when:1d&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(rss_url)

    if st.button("🔄 Actualizar Análisis"):
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
    Eres un analista macroeconómico y técnico senior de XAU/USD (Oro).

    DATOS EN VIVO DEL MERCADO:
    1. DXY (Dólar): Cotización = {dxy_precio:.2f} | Var = {dxy_var:.2f}% | RSI(14) = {dxy_rsi:.1f} | Momentum = {dxy_mom:.2f}
    2. US02Y (Bono 2 años): Rendimiento = {us02_precio:.2f}% | Var = {us02_var:.2f}% | RSI(14) = {us02_rsi:.1f} | Momentum = {us02_mom:.2f}

    NOTICIAS MACRO RECIENTES (Últimas 24h):
    {texto_titulares}

    ANÁLISIS REQUERIDO:
    Evalúa la fuerza actual del Oro combinando las noticias, la tendencia del Dólar y el Bono a 2 años con sus indicadores técnicos.

    Responde en formato JSON válido con esta estructura:
    {{
        "score_global": 75,
        "estado": "Fortaleza Alcista",
        "factores": {{
            "Presión DXY (Dólar + RSI)": 80,
            "Rendimiento Bono 2Y (Fed)": 70,
            "Demanda Refugio Seguro": 85,
            "Presión Inflacionaria": 60
        }},
        "noticias": [
            {{
                "titulo": "Título traducido al español",
                "sesgo": "Alcista 🟢",
                "explicacion": "Explicación breve de 1 frase.",
                "url": "URL original"
            }}
        ]
    }}
    """

    with st.spinner("Procesando datos con IA..."):
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
                            "<b>Sentimiento Combinado Macro + Técnico"
                            f" XAU/USD</b><br><span style='font-size:0.8em;color:gray'>{estado}</span>"
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

            st.subheader("📊 Factores Clave (Macro & Técnico)")
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
