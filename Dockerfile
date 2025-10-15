FROM python:3.10-slim

# Dependencias de sistema para OpenCV/MediaPipe
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Limitar hilos (evita saturar 0.1 CPU del plan Free)
ENV OMP_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1

WORKDIR /app

# 1) Instalar PIP al día
RUN python -m pip install --no-cache-dir --upgrade pip

# 2) FORZAR paquetes críticos con versión
#    (si fallara requirements, esto asegura que streamlit-webrtc exista)
RUN pip install --no-cache-dir \
    streamlit==1.50.0 \
    streamlit-webrtc==0.47.6 \
    scikit-learn==1.5.1 \
    av>=10.0.0 \
    aiortc>=1.6.0

# 3) Resto de dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar la app
COPY . .

EXPOSE 10000
CMD ["streamlit", "run", "app_streamlit.py", "--server.port=10000", "--server.address=0.0.0.0"]
