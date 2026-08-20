import streamlit as st
import feedparser
import google.generativeai as genai
import plotly.graph_objects as go
import yfinance as yf
import json

st.set_page_config(page_title="XAU/USD AI Radar", page_icon="🥇", layout="centered")

st.title("🥇 Radar de Noticias XAU/USD con IA")
st.caption("Análisis macroeconómico + Cotización DXY en tiempo real")

api_key = st.secrets.get("GEMINI_API_KEY")

if not api_key:
    st.error("⚠️ No se encontró la API Key en los Secrets de Streamlit.")
else:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-3.6-flash')

    # --- OBTENER DATOS REALES DEL DXY DESDE EL GRÁFICO ---
    try:
        dxy_ticker = yf.Ticker("DX-Y.NYB")
        dxy_hist = dxy_ticker.history(period="2d")
        if len(dxy_hist) >= 2:
            precio_actual = dxy_hist['Close'].iloc[-1]
            precio_anterior = dxy_hist['Close'].iloc[-2]
            var_dxy = ((precio_actual - precio_anterior) / precio_anterior) * 100
        else:
            precio_actual, var_dxy = 100.0, 0.0
    except Exception:
        precio_actual, var_dxy = 100.0, 0.0

    # --- OBTENER NOTICIAS RSS ---
    rss_url = "https://news.google.com/rss/search?q=gold+price+XAUUSD+Fed+inflation&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(rss_url)

    if st.button("🔄 Actualizar Datos"):
        st.rerun()

    noticias = feed.entries[:5]
    texto_titulares = ""
    for i, entry in enumerate(noticias, 1):
        texto_titulares += f"\n{i}. Titular: {entry.title}\n   Enlace: {entry.link}\n"

    # Prompteamos con el dato real del DXY incorporado
    prompt = f"""
    Eres un analista macroeconómico experto en XAU/USD (Oro).
    
    DATOS REALES DEL MERCADO HOY:
    - Cotización actual del DXY (Índice Dólar): {precio_actual:.2f}
    - Variación diaria del DXY: {var_dxy:.2f}% (Si es negativo es Alcista para el Oro; si es positivo es Bajista para el Oro).

    NOTICIAS RECIENTES:
    {texto_titulares}

    Calcula la fuerza del Oro considerando obligatoriamente la variación real del DXY actual.
    Responde EXCLUSIVAMENTE en formato JSON válido (sin etiquetas markdown ```json) con esta estructura:
    {{
        "score_global": 75,
        "estado": "Fortaleza Alcista",
        "factores": {{
            "Impacto DXY (Dólar Real)": 80,
            "Expectativas Tipos Fed": 70,
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

    with st.spinner("Conectando con el gráfico del DXY y analizando noticias..."):
        try:
            response = model.generate_content(prompt)
            clean_json = response.text.replace("```json", "").replace("```", "").strip()
            data = json.loads(clean_json)

            # Muestra el dato en vivo del DXY en la cabecera
            st.metric(
                label="DXY (Índice Dólar en vivo)",
                value=f"{precio_actual:.2f}",
                delta=f"{var_dxy:.2f}%",
                delta_color="inverse" # En el Oro, que el DXY baje (rojo) es una métrica positiva (verde)
            )

            # --- DIBUJAR TERMÓMETRO ---
            score = data["score_global"]
            estado = data["estado"]

            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = score,
                title = {'text': f"<b>Sentimiento Combinado XAU/USD</b><br><span style='font-size:0.8em;color:gray'>{estado}</span>"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#FFFFFF"},
                    'steps': [
                        {'range': [0, 35], 'color': "#FF4B4B"},   # Debilidad
                        {'range': [35, 65], 'color': "#FFA500"},  # Neutral
                        {'range': [65, 100], 'color': "#00CC96"}  # Fortaleza
                    ]
                }
            ))
            fig.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)

            # --- DESGROSE DE FACTORES ---
            st.subheader("📊 Desglose de Factores")
            for factor, valor in data["factores"].items():
                st.write(f"**{factor}:** {valor}/100")
                st.progress(valor / 100)

            st.divider()

            # --- LISTA DE NOTICIAS ---
            st.subheader("Últimos titulares procesados")
            for item in data["noticias"]:
                st.markdown(f"### {item['titulo']}")
                st.write(f"- **Sesgo:** {item['sesgo']}")
                st.write(f"- **Explicación:** {item['explicacion']}")
                st.markdown(f"- [Leer noticia original]({item['url']})")
                st.write("---")

        except Exception as e:
            st.error(f"Error al procesar los datos: {e}")
