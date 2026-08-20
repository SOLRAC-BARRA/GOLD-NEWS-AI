import streamlit as st
import feedparser
import google.generativeai as genai
import plotly.graph_objects as go
import json

st.set_page_config(page_title="XAU/USD AI Radar", page_icon="🥇", layout="centered")

st.title("🥇 Radar de Noticias XAU/USD con IA")
st.caption("Análisis macroeconómico en tiempo real sin coste")

api_key = st.secrets.get("GEMINI_API_KEY")

if not api_key:
    st.error("⚠️ No se encontró la API Key en los Secrets de Streamlit.")
else:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-3.6-flash')

    rss_url = "https://news.google.com/rss/search?q=gold+price+XAUUSD+Fed+inflation&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(rss_url)

    if st.button("🔄 Actualizar Noticias"):
        st.rerun()

    noticias = feed.entries[:5]
    texto_titulares = ""
    for i, entry in enumerate(noticias, 1):
        texto_titulares += f"\n{i}. Titular: {entry.title}\n   Enlace: {entry.link}\n"

    prompt = f"""
    Eres un analista macroeconómico experto en XAU/USD (Oro).
    Analiza este conjunto de noticias recientes:
    {texto_titulares}

    Responde EXCLUSIVAMENTE en formato JSON válido (sin etiquetas markdown ```json) con esta estructura:
    {{
        "score_global": 76,
        "estado": "Fortaleza Alcista",
        "factores": {{
            "Impacto DXY (Dólar)": 80,
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

    with st.spinner("Calculando sentimiento macro y generando indicador..."):
        try:
            response = model.generate_content(prompt)
            clean_json = response.text.replace("```json", "").replace("```", "").strip()
            data = json.loads(clean_json)

            # --- DIBUJAR TERMÓMETRO / VELOCÍMETRO ---
            score = data["score_global"]
            estado = data["estado"]

            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = score,
                title = {'text': f"<b>Sentimiento Macro XAU/USD</b><br><span style='font-size:0.8em;color:gray'>{estado}</span>"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#FFFFFF"},
                    'steps': [
                        {'range': [0, 35], 'color': "#FF4B4B"},   # Debilidad (Rojo)
                        {'range': [35, 65], 'color': "#FFA500"},  # Neutral (Naranja)
                        {'range': [65, 100], 'color': "#00CC96"}  # Fortaleza (Verde)
                    ]
                }
            ))
            fig.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)

            # --- BARRAS DE FACTORES MACRO ---
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
