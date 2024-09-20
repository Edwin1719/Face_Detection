import streamlit as st
import cv2
import numpy as np
import os
from deepface import DeepFace
from st_social_media_links import SocialMediaIcons

# Configuración de la aplicación
st.set_page_config(page_title="Reconocimiento facial", layout="wide", initial_sidebar_state="expanded")

# Logo en la parte superior izquierda
st.markdown("""
    <style>
    .logo-container {
        display: flex;
        align-items: center;
    }
    .logo {
        width: 50px; /* Ajusta el tamaño del logo */
        margin-right: 15px;
    }
    </style>
    <div class="logo-container">
        <img src="https://previews.123rf.com/images/allismagic/allismagic1710/allismagic171000018/87699325-identificaci%C3%B3n-biom%C3%A9trica-concepto-de-sistema-de-reconocimiento-facial-reconocimiento-facial-icono.jpg" class="logo">
    </div>
    """, unsafe_allow_html=True)

def cargar_imagen(archivo):
    """Carga y convierte una imagen desde un archivo cargado a un formato usable por OpenCV."""
    bytes_data = archivo.getvalue()
    imageBGR = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), 1)
    return cv2.cvtColor(imageBGR, cv2.COLOR_BGR2RGB)

def identificar_rostro(imagen_buscada, directorio='FACE_DETECT/Directorio de imagenes'):
    """Compara la imagen entregada con imágenes de un directorio."""
    if not os.path.exists(directorio):
        st.error(f"La ruta {directorio} no existe.")
        return None
    
    for filename in os.listdir(directorio):
        file_path = os.path.join(directorio, filename)
        if not os.path.isfile(file_path):
            continue
        
        try:
            result = DeepFace.verify(img1_path=imagen_buscada, img2_path=file_path, model_name='VGG-Face')
            if result["verified"]:
                return filename
        except ValueError as e:
            st.error(f"Error procesando {file_path}: {e}")
    return None

def mostrar_resultados_analisis(analysis, image):
    """Muestra los resultados del análisis de DeepFace sobre la imagen."""
    for face in analysis:
        x, y, w, h = face['region']['x'], face['region']['y'], face['region']['w'], face['region']['h']
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, face['dominant_emotion'], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    st.image(image)
    st.write(f"Edad: {analysis[0]['age']}")
    st.write(f"Género: {analysis[0]['gender']}")
    st.write(f"Raza: {analysis[0]['dominant_race']}")
    st.write(f"Emoción: {analysis[0]['dominant_emotion']}")

# Interfaz de la aplicación Usamos HTML y CSS para centrar el texto
st.markdown("""
    <style>
    .center-text {
        text-align: center;
    }
    </style>
    <h1 class="center-text">¿IDENTIFICA LA IMAGEN?</h1>
    <h3 class="center-text">Uso de la Libreria DeepFace</h3>
    """, unsafe_allow_html=True)

archivo_cargado = st.file_uploader("Elige un archivo", type='jpg')

if archivo_cargado is not None:
    image = cargar_imagen(archivo_cargado)
    
    st.subheader(f'Archivo: {archivo_cargado.name}')
    st.image(image)
    
    temp_image_path = "temp_image.jpg"
    cv2.imwrite(temp_image_path, image)
    
    # Identificar rostro en directorio
    resultado_identificacion = identificar_rostro(temp_image_path)
    
    if resultado_identificacion:
        st.success(f"Rostro encontrado: {resultado_identificacion}")
        st.balloons()
    else:
        st.error("Celebridad no encontrada")
    
    # Análisis facial
    analysis = DeepFace.analyze(img_path=temp_image_path, actions=['age', 'gender', 'race', 'emotion'])
    
    mostrar_resultados_analisis(analysis, image)

# Pie de página con información del desarrollador y logos de redes sociales
st.markdown("""
---
**Desarrollador:** Edwin Quintero Alzate<br>
**Email:** egqa1975@gmail.com<br>
""")

social_media_links = [
    "https://www.facebook.com/edwin.quinteroalzate",
    "https://www.linkedin.com/in/edwinquintero0329/",
    "https://github.com/Edwin1719"]

social_media_icons = SocialMediaIcons(social_media_links)
social_media_icons.render()