import streamlit as st
from joblib import load
import numpy as np
import mediapipe as mp
import cv2
from collections import Counter

st.set_page_config(page_title="SIGNIA - LSA en tiempo real", layout="wide")
st.title("SIGNIA – Reconocimiento de señas (demo web)")

# ---- Cargar modelos ----
@st.cache_resource
def load_models():
    m_izq = load("modelo_letras_izq_rf.joblib")["model"]
    m_der = load("modelo_letras_der_rf.joblib")["model"]
    return m_izq, m_der

MODEL_IZQ, MODEL_DER = load_models()

# ---- Normalización ----
def normalize_seq_xy(seq_xyz):
    seq = seq_xyz.copy().astype(float)
    xy = seq[:, :2]
    xy -= xy.mean(axis=0, keepdims=True)
    max_abs = np.abs(xy).max() or 1.0
    xy /= max_abs
    seq[:, :2] = xy
    return seq

# ---- Sidebar ----
with st.sidebar:
    st.header("Ajustes")
    modo = st.radio("Elegí tu mano", ["Diestro", "Zurdo"], index=0)

st.info("📷 Usa la cámara para capturar una imagen y obtener la predicción.")

# ---- Cámara ----
img_file = st.camera_input("Sacá una foto de tu seña")

if img_file is not None:
    # Leer la imagen como array
    bytes_data = img_file.getvalue()
    nparr = np.frombuffer(bytes_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Procesar con mediapipe
    with mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.8
    ) as hands:
        result = hands.process(rgb)
        if result.multi_hand_landmarks:
            pts = np.array([[lm.x, lm.y, lm.z] for lm in result.multi_hand_landmarks[0].landmark], dtype=float)
            seq = normalize_seq_xy(pts)
            vec = seq.reshape(-1)

            if modo == "Zurdo":
                pred = MODEL_IZQ.predict([vec])[0]
            else:
                pred = MODEL_DER.predict([vec])[0]

            st.success(f"✅ Predicción: **{pred}**")
        else:
            st.error("❌ No se detectó mano en la imagen.")
