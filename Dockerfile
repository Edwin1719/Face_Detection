FROM python:3.10-slim

# Instalar dependencias del sistema necesarias para dlib y cmake
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libboost-all-dev \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk2.0-dev \
    libgl1-mesa-glx \
    git \
    && rm -rf /var/lib/apt/lists/*

# Crear directorio de trabajo
WORKDIR /app

# Copiar los archivos del proyecto
COPY . /app

# Instalar dependencias de Python
RUN pip install --upgrade pip
RUN pip install streamlit opencv-python numpy face_recognition Pillow st_social_media_links

# Exponer el puerto de Streamlit
EXPOSE 8501

# Comando para correr tu aplicación (ajusta el nombre del script si es diferente)
CMD ["streamlit", "run", "app_ml.py", "--server.port=8501", "--server.enableCORS=false"]
