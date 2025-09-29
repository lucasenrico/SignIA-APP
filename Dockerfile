# Python 3.10 + sistema mínimo
FROM python:3.10-slim

# Dependencias del sistema necesarias para OpenCV/Mediapipe
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Directorio de trabajo
WORKDIR /app

# Instalar dependencias Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del repo (código + modelos .joblib)
COPY . .

# Exponer puerto (Render ignora EXPOSE pero no molesta)
EXPOSE 10000

# Comando de arranque para Streamlit en Render
CMD ["streamlit", "run", "app_streamlit.py", "--server.port=10000", "--server.address=0.0.0.0"]
