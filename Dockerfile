FROM python:3.10-slim

# libs que necesitan OpenCV/Mediapipe
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# limitar hilos para no sobrecargar 0.1 CPU
ENV OMP_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# podés ajustar por env en Render:
# ENV MAX_SESS=2
# ENV SKIP_N=3
# ENV DRAW_EVERY=3

EXPOSE 10000
CMD ["streamlit", "run", "app_streamlit.py", "--server.port=10000", "--server.address=0.0.0.0"]
