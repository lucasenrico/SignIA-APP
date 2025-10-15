import os, time
import streamlit as st
import numpy as np
import cv2
import mediapipe as mp
from joblib import load
from collections import Counter, deque

st.set_page_config(page_title="SIGNIA - LSA", layout="wide")

# ======= Config general =======
LIVE_MODE = os.getenv("LIVE_MODE", "0") in ("1", "true", "True")
BUILD_TAG = os.getenv("RENDER_GIT_COMMIT", "local")[:7]
st.caption(f"🧱 Build: {BUILD_TAG} | Live mode: {LIVE_MODE}")

# ======= Modelos =======
@st.cache_resource
def load_models():
    m_izq = load("modelo_letras_izq_rf.joblib")["model"]
    m_der = load("modelo_letras_der_rf.joblib")["model"]
    return m_izq, m_der

MODEL_IZQ, MODEL_DER = load_models()

def normalize_seq_xy(seq_xyz):
    seq = seq_xyz.copy().astype(float)
    xy = seq[:, :2]
    xy -= xy.mean(axis=0, keepdims=True)
    max_abs = float(np.abs(xy).max()) or 1.0
    xy /= max_abs
    seq[:, :2] = xy
    return seq

# ======= Sidebar =======
with st.sidebar:
    st.header("Ajustes")
    mano = st.radio("Elegí tu mano:", ["Diestro", "Zurdo"], index=0)
    st.write("Modo:")
    if LIVE_MODE:
        st.success("LIVE (cámara en tiempo real)")
        st.caption("Uso recomendado: local / Hugging Face Spaces.")
    else:
        st.info("LITE (sin streaming)")
        st.caption("Uso recomendado: Render Free.")
    st.divider()
    st.caption("SIGNIA · Reconocimiento de señas con MediaPipe + RandomForest")

st.title("SIGNIA — Traductor de LSA")

# ==============================
# MODO LITE (sin WebRTC) — Render
# ==============================
if not LIVE_MODE:
    st.subheader("Demo sin streaming (ideal para Render)")

    col1, col2 = st.columns(2)
    with col1:
        img_cam = st.camera_input("Tomá una foto de tu seña")
    with col2:
        img_up = st.file_uploader("…o subí una imagen", type=["jpg", "jpeg", "png"])

    img_bytes = None
    origen = None
    if img_cam is not None:
        img_bytes = img_cam.getvalue()
        origen = "cámara"
    elif img_up is not None:
        img_bytes = img_up.read()
        origen = "archivo"

    if img_bytes is None:
        st.warning("Esperando una imagen…")
        st.stop()

    # decodificar y procesar
    nparr = np.frombuffer(img_bytes, np.uint8)
    bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if bgr is None:
        st.error("No se pudo leer la imagen.")
        st.stop()

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    with mp.solutions.hands.Hands(
        static_image_mode=True, max_num_hands=1, min_detection_confidence=0.8
    ) as hands:
        res = hands.process(rgb)

    if not res.multi_hand_landmarks:
        st.error("No se detectó mano. Probá otra toma (fondo claro, mano completa).")
        st.image(rgb, caption=f"Vista {origen}", use_container_width=True)
        st.stop()

    lms = res.multi_hand_landmarks[0]
    pts = np.array([[lm.x, lm.y, lm.z] for lm in lms.landmark], dtype=float)
    vec = normalize_seq_xy(pts).reshape(-1)
    pred = (MODEL_IZQ if mano == "Zurdo" else MODEL_DER).predict([vec])[0]

    # dibujar
    mp.solutions.drawing_utils.draw_landmarks(
        rgb, lms, mp.solutions.hands.HAND_CONNECTIONS,
        mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
        mp.solutions.drawing_styles.get_default_hand_connections_style(),
    )
    st.success(f"✅ Predicción: **{pred}**")
    st.image(rgb, caption=f"Procesada desde {origen}", use_container_width=True)

    st.info("Tip: si querés streaming en vivo, corré la app local con LIVE_MODE=1 o en Hugging Face Spaces.")
    st.stop()

# ===================================
# MODO LIVE (WebRTC) — local / HF only
# ===================================
# Solo importamos streamlit_webrtc si estamos en Live mode
from streamlit_webrtc import (
    webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode
)

# TURN público (fallback). Para producción real, usá uno propio.
TURN_URL = os.getenv("TURN_URL", "turn:openrelay.metered.ca:80")
TURN_USERNAME = os.getenv("TURN_USERNAME", "openrelayproject")
TURN_CREDENTIAL = os.getenv("TURN_CREDENTIAL", "openrelayproject")

ice_servers = [
    {"urls": [
        "stun:stun.l.google.com:19302",
        "stun:stun1.l.google.com:19302",
        "stun:stun2.l.google.com:19302",
    ]},
    {"urls": [f"{TURN_URL}?transport=udp"], "username": TURN_USERNAME, "credential": TURN_CREDENTIAL},
    {"urls": ["turn:openrelay.metered.ca:443?transport=tcp"], "username": "openrelayproject", "credential": "openrelayproject"},
]
rtc_cfg = RTCConfiguration({"iceServers": ice_servers})

# Optimizaciones básicas para CPU baja
SKIP_N = int(os.getenv("SKIP_N", "3"))
TARGET_W, TARGET_H = 640, 480

class LiveProcessor(VideoProcessorBase):
    def __init__(self):
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False, model_complexity=0,  # liviano
            max_num_hands=1, min_detection_confidence=0.8, min_tracking_confidence=0.8
        )
        self.buf = deque(maxlen=5)
        self.pred = "…"
        self._f = 0

    def recv(self, frame):
        import av
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)  # selfie natural en vivo

        self._f += 1
        small = cv2.resize(img, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)

        if self._f % SKIP_N == 0:
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            r = self.hands.process(rgb)
            if r.multi_hand_landmarks:
                lms = r.multi_hand_landmarks[0]
                pts = np.array([[lm.x, lm.y, lm.z] for lm in lms.landmark], dtype=float)
                vec = normalize_seq_xy(pts).reshape(-1)
                p = (MODEL_IZQ if mano == "Zurdo" else MODEL_DER).predict([vec])[0]
                self.buf.append(p)
                self.pred = Counter(self.buf).most_common(1)[0][0]

                mp.solutions.drawing_utils.draw_landmarks(
                    small, lms, mp.solutions.hands.HAND_CONNECTIONS,
                    mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                    mp.solutions.drawing_styles.get_default_hand_connections_style(),
                )
            else:
                self.buf.clear()
                self.pred = "…"

        cv2.rectangle(small, (10, 10), (420, 70), (0, 0, 0), -1)
        cv2.putText(small, f"Predicción: {self.pred}", (20, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
        return av.VideoFrame.from_ndarray(small, format="bgr24")

st.subheader("Demo en vivo (requiere red compatible con WebRTC)")
webrtc_streamer(
    key="signia-live",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=rtc_cfg,
    media_stream_constraints={"video": True, "audio": False},
    video_processor_factory=LiveProcessor,
    async_processing=True,
    video_html_attrs={"playsinline": True, "autoPlay": True, "muted": True, "controls": False},
)
