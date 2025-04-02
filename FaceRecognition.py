import streamlit as st
import cv2
import numpy as np
import os
from deepface import DeepFace
from st_social_media_links import SocialMediaIcons
import time

# --- Configuración ---
DIRECTORIO_IMAGENES_REFERENCIA = 'Face_Detection/Directorio de imagenes'
MODELO_VERIFICACION = 'VGG-Face'
ACCIONES_ANALISIS = ['age', 'gender', 'race', 'emotion']

st.set_page_config(page_title="Reconocimiento facial", layout="wide")

# --- Logo ---
st.markdown("""
    <style>
    .logo-container { display: flex; align-items: center; }
    .logo { width: 50px; margin-right: 15px; }
    .center-text { text-align: center; }
    </style>
    <div class="logo-container">
        <img src="https://previews.123rf.com/images/allismagic/allismagic1710/allismagic171000018/87699325-identificaci%C3%B3n-biom%C3%A9trica-concepto-de-sistema-de-reconocimiento-facial-reconocimiento-facial-icono.jpg" class="logo">
    </div>
    """, unsafe_allow_html=True)

# --- Función Auxiliar ---
def cargar_imagen(bytes_data):
    nparr = np.frombuffer(bytes_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image is not None else None

# --- Identificación de rostro ---
def identificar_rostro(imagen_path):
    if not os.path.exists(DIRECTORIO_IMAGENES_REFERENCIA):
        st.error(f"El directorio '{DIRECTORIO_IMAGENES_REFERENCIA}' no existe.")
        return None, None

    archivos = [f for f in os.listdir(DIRECTORIO_IMAGENES_REFERENCIA) if os.path.isfile(os.path.join(DIRECTORIO_IMAGENES_REFERENCIA, f))]
    if not archivos:
        st.warning("No hay imágenes de referencia.")
        return None, None

    progress_bar = st.progress(0)
    for i, archivo in enumerate(archivos):
        try:
            resultado = DeepFace.verify(imagen_path, os.path.join(DIRECTORIO_IMAGENES_REFERENCIA, archivo), model_name=MODELO_VERIFICACION, enforce_detection=False, silent=True)
            if resultado.get("verified", False):
                return archivo, resultado.get("distance")
        except:
            pass
        progress_bar.progress((i + 1) / len(archivos))

    return None, None

# --- Análisis Facial ---
def analizar_imagen(imagen, columna):
    try:
        resultados = DeepFace.analyze(img_path=imagen, actions=ACCIONES_ANALISIS, enforce_detection=False, silent=True)
        if resultados:
            st.write(f"Edad: {resultados[0].get('age', 'N/A')}, Género: {resultados[0].get('dominant_gender', 'N/A')}, Raza: {resultados[0].get('dominant_race', 'N/A')}, Emoción: {resultados[0].get('dominant_emotion', 'N/A')}")
    except Exception as e:
        st.error(f"Error en el análisis facial: {e}")

# --- Procesamiento Principal ---
def procesar_imagen(image_rgb, origen):
    st.subheader(f'Procesando imagen desde: {origen}')
    col1, col2, col3 = st.columns(3)
    col1.image(image_rgb, caption="Imagen de Entrada", use_column_width=True)
    
    temp_path = f"temp_{int(time.time())}.jpg"
    cv2.imwrite(temp_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    nombre_match, distancia_match = identificar_rostro(temp_path)
    
    if nombre_match:
        st.success(f"Rostro encontrado: {nombre_match} (Distancia: {distancia_match:.4f})")
        ref_path = os.path.join(DIRECTORIO_IMAGENES_REFERENCIA, nombre_match)
        if os.path.exists(ref_path):
            col2.image(cv2.imread(ref_path), caption=f"Referencia: {nombre_match}", use_column_width=True)
    else:
        st.warning("No se encontraron coincidencias.")
    
    os.remove(temp_path)
    analizar_imagen(image_rgb, col3)

# --- Interfaz ---
st.markdown("<h1 style='text-align: center;'>¿IDENTIFICA LA IMAGEN?</h1>", unsafe_allow_html=True)
st.subheader("Elige una fuente de imagen:")

if st.button("Limpiar / Empezar de Nuevo"):
    st.rerun()

archivo_subido = st.file_uploader("Cargar una imagen:", type=['jpg', 'png', 'jpeg'])
captura_webcam = st.camera_input("O tomar una foto con la webcam:")

if archivo_subido:
    procesar_imagen(cargar_imagen(archivo_subido.getvalue()), f"Archivo: {archivo_subido.name}")
elif captura_webcam:
    procesar_imagen(cargar_imagen(captura_webcam.getvalue()), "Webcam")

# --- Redes Sociales ---
st.markdown("---")
st.markdown("""
**Desarrollador:** Edwin Quintero Alzate<br>
**Email:** egqa1975@gmail.com<br>
""")
social_links = ["https://www.facebook.com/edwin.quinteroalzate", "https://www.linkedin.com/in/edwinquintero0329/", "https://github.com/Edwin1719"]
SocialMediaIcons(social_links).render()
