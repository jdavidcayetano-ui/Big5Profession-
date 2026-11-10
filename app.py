import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Big Five Cluster Predictor", layout="centered")

# Título de la Aplicación
st.title("🧠 Big Five Personality Cluster Predictor")
st.write("Ingresa tus puntajes del test Big Five para descubrir a qué clúster perteneces y compararte con famosos de tu grupo.")

# Cargar datos, modelo y escalador de forma segura
@st.cache_resource
def load_assets():
    model = joblib.load('modelo_kmeans_big5.joblib')
    scaler = joblib.load('scaler_big5.joblib')
    # Descargar el dataset original de famosos para la consulta
    df_famous = pd.read_csv('https://raw.githubusercontent.com/gerrymay/fcaData/refs/heads/master/oceanbig5.csv')
    # Asignarle las etiquetas del clúster con las que fue entrenado (n_clusters=5)
    # Asegurar el orden de features correcto para la predicción de todo el dataset
    features = ['op', 'co', 'ex', 'ag', 'ne']
    scaled_data = scaler.transform(df_famous[features])
    df_famous['cluster'] = model.predict(scaled_data)
    return model, scaler, df_famous

try:
    modelo_cargado, scaler_cargado, df_famosos = load_assets()
except Exception as e:
    st.error(f"Error al cargar los componentes: {e}. Asegúrate de haber guardado 'modelo_kmeans_big5.joblib' y 'scaler_big5.joblib' en el mismo directorio.")
    st.stop()

# Entradas del usuario utilizando Sliders
st.sidebar.header("Tus Resultados de Personalidad")
op_val = st.sidebar.slider("Apertura Mental (op)", 0.0, 100.0, 50.0, 0.1)
co_val = st.sidebar.slider("Conciencia/Organización (co)", 0.0, 100.0, 50.0, 0.1)
ex_val = st.sidebar.slider("Extroversión (ex)", 0.0, 100.0, 50.0, 0.1)
ag_val = st.sidebar.slider("Amabilidad (ag)", 0.0, 100.0, 50.0, 0.1)
ne_val = st.sidebar.slider("Neuroticismo (ne)", 0.0, 100.0, 50.0, 0.1)

# Crear array con los valores ingresados
input_data = np.array([[op_val, co_val, ex_val, ag_val, ne_val]])

# Escalar y Predecir
input_scaled = scaler_cargado.transform(input_data)
cluster_predicho = modelo_cargado.predict(input_scaled)[0]

# Mostrar Resultados Principales
st.markdown(f"### 🎉 ¡Has sido asignado al **Clúster {cluster_predicho}**!")

# Mostrar 5 famosos del mismo clúster
st.subheader("🌟 Famosos en tu Clúster")
famosos_cluster = df_famosos[df_famosos['cluster'] == cluster_predicho]
if not famosos_cluster.empty:
    st.dataframe(famosos_cluster[['usuario', 'op', 'co', 'ex', 'ag', 'ne']].head(5))
else:
    st.info("No se encontraron famosos registrados en este clúster.")

# Gráfico de Radar Comparativo
st.subheader("📊 Comparación en Gráfico de Radar")

# Calcular promedios del clúster asignado
features = ['op', 'co', 'ex', 'ag', 'ne']
valores_promedio_cluster = famosos_cluster[features].mean().values.tolist()
valores_usuario = [op_val, co_val, ex_val, ag_val, ne_val]

# Cerrar el círculo del radar
valores_promedio_cluster += valores_promedio_cluster[:1]
valores_usuario += valores_usuario[:1]

num_vars = len(features)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

# Dibujar los ejes de cada dimensión
plt.xticks(angles[:-1], features, color='grey', size=11)

# Graficar el promedio del clúster
ax.plot(angles, valores_promedio_cluster, color='#1aafaf', linewidth=2, linestyle='solid', label=f'Promedio Clúster {cluster_predicho}')
ax.fill(angles, valores_promedio_cluster, color='#1aafaf', alpha=0.25)

# Graficar los resultados del usuario
ax.plot(angles, valores_usuario, color='#ff5722', linewidth=2, linestyle='--', label='Tus Resultados')
ax.fill(angles, valores_usuario, color='#ff5722', alpha=0.15)

plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
ax.set_title("Tus rasgos vs El Promedio de tu Clúster", size=14, y=1.1)

st.pyplot(fig)
