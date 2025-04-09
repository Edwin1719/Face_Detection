import streamlit as st
import face_recognition
import cv2
import numpy as np
from PIL import Image, ImageDraw
import os
from st_social_media_links import SocialMediaIcons

st.set_page_config(
    page_title="Reconocimiento facial",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def identificarRostro(imagen_buscada):
    directorioBase = 'Directorio de imagenes'
    encoding_a_buscar = face_recognition.face_encodings(imagen_buscada)[0]
    imagen_encontrada = None
    nombre_archivo_encontrado = None

    for filename in os.listdir(directorioBase):
        file_path = os.path.join(directorioBase, filename)
        if not filename.lower().endswith(('.jpg', '.jpeg')) or os.path.isdir(file_path):
            continue

        try:
            imagen_comparacion = face_recognition.load_image_file(file_path)
            encoding_comparacion = face_recognition.face_encodings(imagen_comparacion)[0]
        except Exception as e:
            st.warning(f"Error al procesar {filename}: {e}")
            continue

        if face_recognition.compare_faces([encoding_a_buscar], encoding_comparacion, tolerance=0.6)[0]:
            imagen_encontrada = imagen_comparacion
            nombre_archivo_encontrado = filename
            break

    if imagen_encontrada is not None:
        st.success(f"Encontrado: {nombre_archivo_encontrado}")
        st.balloons()
        with st.columns([1,1,1,2])[3]:
            st.markdown("""
            <div style='text-align: center; font-size: 1.4em;'>
                <strong>Coincidencia</strong>
            </div>
            """, unsafe_allow_html=True)
            st.image(imagen_encontrada, caption=nombre_archivo_encontrado)
    else:
        st.error("Celebridad no encontrada")

st.markdown("""
    <div style='text-align: center;'>
        <h1>RECONOCIMIENTO FACIAL</h1>
        <h3>Uso de <a href='https://github.com/ageitgey/face_recognition'>face_recognition</a></h3>
    </div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    archivo_cargado = st.file_uploader("Sube una imagen", type=['jpg', 'jpeg'])
with col2:
    camara = st.camera_input("O toma una foto con la cámara")

imagen_bytes = archivo_cargado.getvalue() if archivo_cargado else camara.getvalue() if camara else None

if imagen_bytes:
    c1, c2, c3, c4 = st.columns([5,2,4,4])
    with c1:
        st.subheader("Imagen seleccionada")
        st.image(imagen_bytes)

    image = cv2.cvtColor(cv2.imdecode(np.frombuffer(imagen_bytes, np.uint8), 1), cv2.COLOR_BGR2RGB)

    try:
        encoding = face_recognition.face_encodings(image)[0]
        landmarks = face_recognition.face_landmarks(image)
        locations = face_recognition.face_locations(image)

        for (top, right, bottom, left) in locations:
            rostro = image[top:bottom, left:right]
            with c2:
                st.subheader("Rostro ")
                st.image(Image.fromarray(rostro))

        for face_landmarks in landmarks:
            pil_image = Image.fromarray(image)
            d = ImageDraw.Draw(pil_image)
            for key in face_landmarks:
                d.line(face_landmarks[key], fill=(255,255,255), width=2)
            d.rectangle([(left, top), (right, bottom)], outline="yellow", width=2)

            with c3:
                tabimagen, tabencoding = st.tabs(["Imagen", "Encoding"])
                with tabimagen:
                    st.subheader("Puntos clave del rostro")
                    st.write(", ".join(face_landmarks.keys()))
                    st.image(pil_image)
                with tabencoding:
                    st.code(encoding)

        with c4:
            identificarRostro(image)

    except IndexError:
        st.error("No se detectó ningún rostro en la imagen. Intenta de nuevo.")

st.markdown("""
    <hr>
    <div style='text-align: center;'>
        <strong>Desarrollador:</strong> Edwin Quintero Alzate<br>
        <strong>Email:</strong> egqa1975@gmail.com
    </div>
""", unsafe_allow_html=True)

SocialMediaIcons([
    "https://www.facebook.com/edwin.quinteroalzate",
    "https://www.linkedin.com/in/edwinquintero0329/",
    "https://github.com/Edwin1719"
]).render()
