import json
import feedparser
import google.generativeai as genai
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(
    page_title="XAU/USD AI Radar Pro", page_icon="🥇", layout="centered"
)

st.title("🥇 Radar Macro & Técnico XAU/USD")
st.caption("Noticias + DXY + Bonos a 2 Años (US02Y) + RSI/Momentum")

api_key = st.secrets.get("GEMINI_API_KEY")

if not api_key:
  st.error("⚠️ No se encontró la API Key en los Secrets de Streamlit.")
else:

  # --- FUNCIÓN CON CACHÉ PARA MERCADOS (guarda datos 5 min) ---
  @st.cache_data(ttl=300)
  def obtener_datos_mercado(ticker_symbol):
    try:
      ticker = yf.Ticker(ticker_symbol)
      df = ticker.history(period="1mo")
      if len(df) >= 15:
        # RSI 14
        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))

        # Momentum 14
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

  # --- FUNCIÓN CON CACHÉ PARA LA IA (guarda respuesta 10 min) ---
  @st.cache_data(ttl=600)
  def consultar_gemini(prompt_text, key):
    genai.configure(api_key=key)
    model = genai.GenerativeModel("gemini-3.6-flash")
    response = model.generate_content(prompt_text)
    return response.text

  # Obtener datos de mercado
  dxy_precio, dxy_var, dxy_rsi, dxy_mom = obtener_datos_mercado("DX-Y.NYB")
  us02_precio, us02_var, us02_rsi, us02_mom = obtener_datos_mercado("2YY=F")

  # --- MÉTRICAS VISUALES EN CABECERA ---
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

  # --- OBTENER NOTICIAS RSS ---
  rss_url = "https://news.google.com/rss/search?q=gold+price+XAUUSD+Fed+inflation&hl=en-US&gl=US&ceid=US:en"
  feed = feedparser.parse(rss_url)

  if st.button("🔄 Actualizar Análisis"):
    st.cache_data.clear()
    st.rerun()

  noticias = feed.entries[:5]
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

    NOTICIAS MACRO RECIENTES:
    {texto_titulares}

    ANÁLISIS REQUERIDO:
    Evalúa la fuerza actual del Oro combinando las noticias, la tendencia del Dólar y el Bono a 2 años con sus indicadores técnicos.

    Responde EXCLUSIVAMENTE en formato JSON válido (sin etiquetas markdown ```json) con esta estructura:
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

  with st.spinner("Procesando datos con caché inteligente..."):
    try:
      raw_response = consultar_gemini(prompt, api_key)
      clean_json = (
          raw_response.replace("```json", "").replace("```", "").strip()
      )
      data = json.loads(clean_json)

      # --- DIBUJAR TERMÓMETRO ---
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
                      {"range": [0, 35], "color": "#FF4B4B"},  # Debilidad
                      {"range": [35, 65], "color": "#FFA500"},  # Neutral
                      {"range": [65, 100], "color": "#00CC96"},  # Fortaleza
                  ],
              },
          )
      )
      fig.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20))
      st.plotly_chart(fig, use_container_width=True)

      # --- DESGLOSE DE FACTORES ---
      st.subheader("📊 Factores Clave (Macro & Técnico)")
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
