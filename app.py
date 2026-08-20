import streamlit as st
import feedparser
import google.generativeai as genai

st.set_page_config(page_title="XAU/USD AI Radar", page_icon="🥇", layout="centered")

st.title("🥇 Radar de Noticias XAU/USD con IA")
st.caption("Análisis macroeconómico en tiempo real sin coste")

api_key = st.sidebar.text_input("Introduce tu Gemini API Key:", type="password")

if not api_key:
    st.info("👈 Por favor, ingresa tu API Key en la barra lateral para empezar.")
else:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-3.6-flash')

    rss_url = "https://news.google.com/rss/search?q=gold+price+XAUUSD+Fed+inflation&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(rss_url)

    if st.button("🔄 Actualizar Noticias"):
        st.rerun()

    st.subheader("Últimos titulares procesados")

    # Extraemos los 5 mejores titulares
    noticias = feed.entries[:5]
    
    # Construimos un único bloque de texto con todas las noticias
    texto_titulares = ""
    for i, entry in enumerate(noticias, 1):
        texto_titulares += f"\n{i}. Titular: {entry.title}\n   Enlace: {entry.link}\n"

    prompt = f"""
    Eres un analista macroeconómico experto en trading de XAU/USD (Oro).
    Analiza la siguiente lista de titulares de noticias:

    {texto_titulares}

    Para CADA titular de la lista, genera un análisis breve con este formato exactamente:
    ### [Número]. [Título de la noticia en español]
    - **Sesgo:** [Alcista 🟢 / Bajista 🔴 / Neutral ⚪]
    - **Explicación:** (1 sola frase explicando el impacto en el Oro y el DXY)
    - [Leer noticia original](URL correspondiente)
    ---
    """

    with st.spinner("Analizando mercado con IA..."):
        try:
            response = model.generate_content(prompt)
            st.markdown(response.text)
        except Exception as e:
            st.error(f"Error al analizar las noticias: {e}")
