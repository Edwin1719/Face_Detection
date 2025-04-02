import streamlit as st
import cv2
import numpy as np
import os
from deepface import DeepFace
from st_social_media_links import SocialMediaIcons
import time

# --- Constantes de Configuración ---
DIRECTORIO_IMAGENES_REFERENCIA = 'Face_Detection/Directorio de imagenes'
MODELO_VERIFICACION = 'VGG-Face'
ACCIONES_ANALISIS = ['age', 'gender', 'race', 'emotion']

# --- Configuración de la página de Streamlit ---
st.set_page_config(page_title="Reconocimiento facial", layout="wide", initial_sidebar_state="expanded")

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

# --- Funciones Auxiliares (Sin cambios) ---
def cargar_imagen_desde_bytes(bytes_data):
    nparr = np.frombuffer(bytes_data, np.uint8)
    image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image_bgr is None:
        st.error("No se pudo decodificar la imagen.")
        return None
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_rgb

def identificar_rostro(imagen_buscada_path, directorio=DIRECTORIO_IMAGENES_REFERENCIA, modelo=MODELO_VERIFICACION):
    if not os.path.exists(directorio):
        st.error(f"La ruta de referencia '{directorio}' no existe.")
        return None, None
    if not os.path.isfile(imagen_buscada_path):
         st.error(f"La imagen a buscar '{imagen_buscada_path}' no se encontró.")
         return None, None
    try:
        archivos_en_directorio = [f for f in os.listdir(directorio) if os.path.isfile(os.path.join(directorio, f))]
    except Exception as e:
        st.error(f"Error al listar archivos en '{directorio}': {e}")
        return None, None
    if not archivos_en_directorio:
        st.warning(f"El directorio de referencia '{directorio}' está vacío.")
        return None, None

    total_files = len(archivos_en_directorio)
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text(f"Comparando con {total_files} imágenes...")
    rostro_encontrado_nombre = None
    rostro_encontrado_distancia = None

    for i, filename in enumerate(archivos_en_directorio):
        file_path = os.path.join(directorio, filename)
        status_text.text(f"Comparando con: {filename} ({i+1}/{total_files})")
        try:
            result = DeepFace.verify(img1_path=imagen_buscada_path, img2_path=file_path,
                                     model_name=modelo, enforce_detection=False, silent=True)
            if result.get("verified", False):
                rostro_encontrado_nombre = filename
                rostro_encontrado_distancia = result.get("distance", None)
                status_text.text(f"¡Coincidencia encontrada con {filename}!")
                progress_bar.progress(100)
                break
        except Exception as e:
            pass
        progress_bar.progress(int(((i + 1) / total_files) * 100))

    if not rostro_encontrado_nombre:
        status_text.text("Comparación completada. No se encontraron coincidencias.")
        progress_bar.progress(100)

    time.sleep(0.5) # Reducir pausa
    status_text.empty()
    progress_bar.empty()
    return rostro_encontrado_nombre, rostro_encontrado_distancia

def mostrar_resultados_analisis(analysis_results, image_rgb, columna_destino):
    if not analysis_results or not isinstance(analysis_results, list) or len(analysis_results) == 0:
        columna_destino.warning("No se detectaron rostros o el formato del análisis es inesperado.")
        return

    image_con_anotaciones = image_rgb.copy()
    primer_rostro = analysis_results[0]
    region = primer_rostro.get('region')

    if region:
        x, y, w, h = region['x'], region['y'], region['w'], region['h']
        cv2.rectangle(image_con_anotaciones, (x, y), (x + w, y + h), (0, 255, 0), 2)
        emocion = primer_rostro.get('dominant_emotion', 'N/A')
        cv2.putText(image_con_anotaciones, emocion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        columna_destino.image(image_con_anotaciones, caption="Imagen con Análisis Facial", use_column_width=True)
    else:
        columna_destino.warning("No se pudo obtener la región del rostro para dibujar.")
        columna_destino.image(image_rgb, caption="Imagen Original (Sin región detectada)", use_column_width=True)

    with st.expander("Ver detalles del análisis"):
        st.write(f"Edad estimada: {primer_rostro.get('age', 'N/A')}")
        st.write(f"Género estimado: {primer_rostro.get('dominant_gender', primer_rostro.get('gender', 'N/A'))}")
        st.write(f"Raza estimada: {primer_rostro.get('dominant_race', 'N/A')}")
        st.write(f"Emoción dominante: {primer_rostro.get('dominant_emotion', 'N/A')}")

def procesar_imagen(image_rgb, source_name):
    # (Función sin cambios significativos, sigue igual que la versión anterior)
    temp_image_path = f"temp_{source_name.replace(':', '').replace('/', '')}_{int(time.time())}.jpg"
    nombre_match = None
    distancia_match = None
    img_referencia_rgb = None

    st.subheader(f'Procesando imagen desde: {source_name}')
    col_input, col_ref, col_analysis = st.columns(3)
    col_input.image(image_rgb, caption="Imagen de Entrada", use_column_width=True)

    try:
        st.markdown("---")
        image_bgr_temp = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(temp_image_path, image_bgr_temp)
        if os.path.exists(temp_image_path):
            st.subheader("Resultados de la Identificación")
            nombre_match, distancia_match = identificar_rostro(temp_image_path)
            if nombre_match:
                dist_str = f"{distancia_match:.4f}" if distancia_match is not None else "N/A"
                st.success(f"Rostro encontrado: Coincide con **{nombre_match}** (Distancia: {dist_str})")
                st.balloons()
                path_ref = os.path.join(DIRECTORIO_IMAGENES_REFERENCIA, nombre_match)
                if os.path.exists(path_ref):
                    img_referencia_bgr = cv2.imread(path_ref)
                    if img_referencia_bgr is not None:
                        img_referencia_rgb = cv2.cvtColor(img_referencia_bgr, cv2.COLOR_BGR2RGB)
                        col_ref.image(img_referencia_rgb, caption=f"Referencia: {nombre_match}", use_column_width=True)
                    else: col_ref.warning(f"No se pudo cargar la imagen de referencia: {nombre_match}")
                else: col_ref.warning(f"Archivo de referencia no encontrado: {nombre_match}")
            else:
                 col_ref.info("Sin coincidencia encontrada.")
        else: st.error("No se pudo guardar la imagen temporal.")
    except Exception as e: st.error(f"Error durante la identificación: {e}")
    finally:
        if os.path.exists(temp_image_path):
            try: os.remove(temp_image_path)
            except OSError as e: st.warning(f"No se pudo eliminar el archivo temporal '{temp_image_path}': {e}")

    st.markdown("---")
    st.subheader("Análisis Facial")
    analysis_placeholder = col_analysis
    try:
        with st.spinner('Realizando análisis facial...'):
            analysis_results = DeepFace.analyze(img_path = image_rgb, actions = ACCIONES_ANALISIS,
                                                enforce_detection=False, silent=True)
        mostrar_resultados_analisis(analysis_results, image_rgb, analysis_placeholder)
    except Exception as e:
        st.error(f"Error durante el análisis facial: {e}")
        analysis_placeholder.warning("El análisis facial falló.")


# --- Interfaz Principal de la Aplicación ---

st.markdown("<h1 class='center-text'>¿IDENTIFICA LA IMAGEN?</h1>", unsafe_allow_html=True)
st.markdown("<h3 class='center-text'>Uso de la Librería DeepFace</h3>", unsafe_allow_html=True)

# --- Botón Limpiar ---
# Colocar el botón antes de los inputs
clear_button_pressed = st.button("Limpiar / Empezar de Nuevo")

# --- Widgets de Entrada (Siempre visibles) ---
st.subheader("Elige una fuente de imagen:")
# Usar claves únicas para ayudar a Streamlit a gestionar el estado
uploaded_file = st.file_uploader("1. Cargar un archivo de imagen:", type=['jpg', 'png', 'jpeg'], key="file_uploader_key")
img_file_buffer = st.camera_input("2. O tomar una foto con la webcam:", key="camera_input_key")

# --- Lógica de Procesamiento ---
image_to_process = None
source_info = ""
process_this_run = False # Flag para decidir si procesar

# Determinar si hay una nueva imagen de entrada *en esta ejecución*
if img_file_buffer is not None:
    source_info = "Webcam"
    bytes_data = img_file_buffer.getvalue()
    image_to_process = cargar_imagen_desde_bytes(bytes_data)
    if image_to_process is not None:
        process_this_run = True
elif uploaded_file is not None:
    source_info = f"Archivo: {uploaded_file.name}"
    bytes_data = uploaded_file.getvalue()
    image_to_process = cargar_imagen_desde_bytes(bytes_data)
    if image_to_process is not None:
        process_this_run = True

# Decidir qué mostrar
if clear_button_pressed:
    # Si se presionó el botón, mostrar mensaje y NO procesar
    st.info("Listo para una nueva imagen. Carga un archivo o usa la webcam.")
    # Nota: La imagen anterior podría seguir mostrándose si estaba en un estado
    # anterior, pero no se *reprocesará*. Los widgets de entrada están listos.
elif process_this_run:
    # Si no se presionó Limpiar Y hay una imagen válida *nueva*, procesarla
    procesar_imagen(image_to_process, source_info)
else:
    # Si no se presionó Limpiar y NO hay imagen nueva, mostrar mensaje inicial
    st.info("Por favor, carga una imagen o toma una foto para iniciar el análisis.")


# --- Pie de página ---
st.markdown("---")
st.markdown("""
**Desarrollador:** Edwin Quintero Alzate<br>
**Email:** egqa1975@gmail.com<br>
""")

social_media_links = [
    "https://www.facebook.com/edwin.quinteroalzate",
    "https://www.linkedin.com/in/edwinquintero0329/",
    "https://github.com/Edwin1719"
]
try:
    social_media_icons = SocialMediaIcons(social_media_links)
    social_media_icons.render()
except Exception as e:
    st.warning(f"No se pudieron mostrar los iconos de redes sociales: {e}")
