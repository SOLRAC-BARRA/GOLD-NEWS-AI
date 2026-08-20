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
    
    for entry in feed.entries[:6]:
        titular = entry.title
        link = entry.link
        
        prompt = f"""
        Eres un analista macroeconómico experto en trading de XAU/USD.
        Analiza este titular: '{titular}'
        
        Responde strictly en este formato breve en español:
        - **Sesgo:** [Alcista 🟢 / Bajista 🔴 / Neutral ⚪]
        - **Explicación:** (1 sola frase explicando la relación entre la noticia, el DXY y el Oro)
        """
        
        try:
            response = model.generate_content(prompt)
            
            with st.expander(f"📌 {titular}"):
                st.write(response.text)
                st.markdown(f"[Leer noticia original]({link})")
        except Exception as e:
            st.error(f"Error al analizar la noticia: {e}")
