import os
import streamlit as st
from joblib import load
import numpy as np
import cv2
import mediapipe as mp
from collections import deque, Counter
from streamlit_webrtc import (
    webrtc_streamer,
    VideoProcessorBase,
    RTCConfiguration,
    WebRtcMode,
)
import os, time, streamlit_webrtc

BUILD_TAG = os.getenv("RENDER_GIT_COMMIT", "sin_sha")[:7]  # si Render lo expone
FILE_MTIME = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(__file__)))

st.caption(f"🧱 Build: {BUILD_TAG} | 📄 app_streamlit.py mtime: {FILE_MTIME} | webrtc {streamlit_webrtc.__version__}")

st.set_page_config(page_title="SIGNIA - LSA en tiempo real", layout="wide")
st.title("SIGNIA – Reconocimiento de señas (tiempo real)")

# =========================
# Carga de modelos (cache)
# =========================
@st.cache_resource
def load_models():
    m_izq = load("modelo_letras_izq_rf.joblib")["model"]
    m_der = load("modelo_letras_der_rf.joblib")["model"]
    return m_izq, m_der

MODEL_IZQ, MODEL_DER = load_models()

# =========================
# Normalización de landmarks
# =========================
def normalize_seq_xy(seq_xyz):
    """
    seq_xyz: np.array (21,3) con (x,y,z) de MediaPipe.
    Centra y escala XY a [-1,1] aprox. Z se deja relativa.
    """
    seq = seq_xyz.copy().astype(float)
    xy = seq[:, :2]
    xy -= xy.mean(axis=0, keepdims=True)
    max_abs = float(np.abs(xy).max()) or 1.0
    xy /= max_abs
    seq[:, :2] = xy
    return seq

# =========================
# Sidebar
# =========================
with st.sidebar:
    st.header("Ajustes")
    modo = st.radio("Elegí tu mano", ["Diestro", "Zurdo"], index=0)
    corregir_espejo = st.checkbox("Corregir espejo de cámara", value=True)
    st.caption("Si tu cámara se ve como 'selfie', dejá habilitado el espejo.")

# =========================
# Config WebRTC (STUN + TURN)
# Lee TURN desde env vars de Render y usa OpenRelay por defecto
# =========================
TURN_URL = os.getenv("TURN_URL", "turn:openrelay.metered.ca:80")
TURN_USERNAME = os.getenv("TURN_USERNAME", "openrelayproject")
TURN_CREDENTIAL = os.getenv("TURN_CREDENTIAL", "openrelayproject")

ice_servers = [
    {"urls": [
        "stun:stun.l.google.com:19302",
        "stun:stun1.l.google.com:19302",
        "stun:stun2.l.google.com:19302",
        "stun:stun3.l.google.com:19302",
        "stun:stun4.l.google.com:19302",
    ]},
    {
        "urls": [TURN_URL],
        "username": TURN_USERNAME,
        "credential": TURN_CREDENTIAL,
    },
]
rtc_cfg = RTCConfiguration({"iceServers": ice_servers})

with st.sidebar:
    st.subheader("Conectividad WebRTC")
    st.write("TURN:", TURN_URL)

st.info(
    "Dale permiso a la cámara. Procesamos cada cuadro con MediaPipe + tu modelo "
    "y mostramos la predicción suavizada (ventana de 5 frames)."
)

# =========================
# Procesador de video (nueva API)
# =========================
class HandSignProcessor(VideoProcessorBase):
    def __init__(self):
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            model_complexity=1,
            max_num_hands=1,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8,
        )
        self.last_preds = deque(maxlen=5)
        self.current_pred = "…"

    def recv(self, frame):
        """
        Nueva API: recibimos av.VideoFrame, devolvemos av.VideoFrame.
        """
        import av  # import local para tiempos de carga

        img = frame.to_ndarray(format="bgr24")

        # Corregir espejo si está activado
        if corregir_espejo:
            img = cv2.flip(img, 1)

        # MediaPipe
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)

        if result.multi_hand_landmarks:
            lms = result.multi_hand_landmarks[0]
            pts = np.array([[lm.x, lm.y, lm.z] for lm in lms.landmark], dtype=float)
            seq = normalize_seq_xy(pts)
            vec = seq.reshape(-1)

            if modo == "Zurdo":
                pred = MODEL_IZQ.predict([vec])[0]
            else:
                pred = MODEL_DER.predict([vec])[0]

            self.last_preds.append(pred)
            vote = Counter(self.last_preds).most_common(1)[0][0]
            self.current_pred = str(vote)

            mp.solutions.drawing_utils.draw_landmarks(
                img,
                lms,
                mp.solutions.hands.HAND_CONNECTIONS,
                mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                mp.solutions.drawing_styles.get_default_hand_connections_style(),
            )
        else:
            self.last_preds.clear()
            self.current_pred = "…"

        # Overlay con predicción
        cv2.rectangle(img, (10, 10), (420, 70), (0, 0, 0), -1)
        cv2.putText(
            img,
            f"Predicción: {self.current_pred}",
            (20, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
        )

        # Devolver como av.VideoFrame
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# =========================
# Lanzar WebRTC (nueva API)
# =========================
webrtc_ctx = webrtc_streamer(
    key="signia-rtc",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=rtc_cfg,
    media_stream_constraints={
        "video": {"facingMode": "user", "width": {"ideal": 1280}, "height": {"ideal": 720}},
        "audio": False,
    },
    video_processor_factory=HandSignProcessor,  # <-- nueva API
    async_processing=True,                      # evita bloqueos si el procesamiento tarda
    video_html_attrs={"playsinline": True, "autoPlay": True, "muted": True, "controls": False},
)

# Estado en la UI (útil para debug)
if webrtc_ctx is not None:
    with st.sidebar:
        st.write("Estado:", "playing ✅" if webrtc_ctx.state.playing else "stopped ⛔")


