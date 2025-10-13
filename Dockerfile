# Imagen base con Python 3.10
FROM python:3.10-slim

# Instalar dependencias de sistema necesarias para OpenCV/mediapipe
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Crear directorio de trabajo
WORKDIR /app

# Copiar dependencias e instalarlas
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar todo el código del repo
COPY . .

# Comando para arrancar Streamlit en Render
CMD ["streamlit", "run", "app_streamlit.py", "--server.port=10000", "--server.address=0.0.0.0"]
