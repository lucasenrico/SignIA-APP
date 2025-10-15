# app_streamlit.py
import os, time
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

# ---------- LIMITES Y OPTIMIZACIONES ----------
MAX_SESS = int(os.getenv("MAX_SESS", "2"))   # cuántas cámaras simultáneas permitís
SKIP_N   = int(os.getenv("SKIP_N", "3"))     # procesar 1 de cada N frames (p.ej. 3)
DRAW_EVERY = int(os.getenv("DRAW_EVERY", "3"))  # dibujar landmarks cada N frames
TARGET_W, TARGET_H = 640, 480                # resolución “barata” para MediaPipe

# Reducir hilos (por si el Dockerfile no lo hizo)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
try:
    cv2.setNumThreads(0)  # o 1
except Exception:
    pass

# ---------- INFO DE BUILD (para verificar que se actualizó) ----------
try:
    import streamlit_webrtc as _sw
    SWV = _sw.__version__
except Exception:
    SWV = "unknown"
BUILD_TAG = os.getenv("RENDER_GIT_COMMIT", "sin_sha")[:7]
st.set_page_config(page_title="SIGNIA - LSA en tiempo real", layout="wide")
st.caption(f"🧱 Build: {BUILD_TAG} | webrtc {SWV} | {time.strftime('%H:%M:%S')}")

st.title("SIGNIA – Reconocimiento de señas (tiempo real)")

# ---------- MODELOS ----------
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

# ---------- UI ----------
with st.sidebar:
    st.header("Ajustes")
    modo = st.radio("Elegí tu mano", ["Diestro", "Zurdo"], index=0)
    corregir_espejo = st.checkbox("Corregir espejo de cámara", value=True)
    st.caption("Si tu cámara se ve como 'selfie', dejá habilitado el espejo.")

# ---------- WebRTC (STUN + TURN público de fallback) ----------
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
    {"urls": [f"{TURN_URL}?transport=udp"], "username": TURN_USERNAME, "credential": TURN_CREDENTIAL},
    {"urls": ["turn:openrelay.metered.ca:443?transport=tcp"], "username": "openrelayproject", "credential": "openrelayproject"},
    {"urls": ["turns:openrelay.metered.ca:443?transport=tcp"], "username": "openrelayproject", "credential": "openrelayproject"},
]
rtc_cfg = RTCConfiguration({"iceServers": ice_servers})

with st.sidebar:
    st.subheader("Conectividad WebRTC")
    st.write("TURN:", TURN_URL)
    st.write("Sesiones máximas:", MAX_SESS)

st.info(
    f"Optimizaciones activas → 1 de cada {SKIP_N} frames, "
    f"resolución {TARGET_W}×{TARGET_H}, landmarks cada {DRAW_EVERY} frames."
)

# ---------- CONTROL DE SESIONES ----------
ACTIVE = set()

# ---------- PROCESADOR DE VIDEO ----------
class HandSignProcessor(VideoProcessorBase):
    def __init__(self):
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            model_complexity=0,   # <= baja carga (antes 1)
            max_num_hands=1,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8,
        )
        self.last_preds = deque(maxlen=5)
        self.current_pred = "…"
        self._f = 0  # contador de frames

        # marcar esta sesión como activa
        ACTIVE.add(id(self))

    def __del__(self):
        # liberar al finalizar
        ACTIVE.discard(id(self))

    def recv(self, frame):
        import av
        img = frame.to_ndarray(format="bgr24")

        if corregir_espejo:
            img = cv2.flip(img, 1)

        # >>> SKIP DE FRAMES <<<
        self._f += 1
        do_compute = (self._f % SKIP_N == 0)

        # reducimos resolución para MediaPipe
        small = cv2.resize(img, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)

        if do_compute:
            rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            result = self.hands.process(rgb_small)

            if result.multi_hand_landmarks:
                lms = result.multi_hand_landmarks[0]
                pts = np.array([[lm.x, lm.y, lm.z] for lm in lms.landmark], dtype=float)  # normalizado 0..1

                vec = normalize_seq_xy(pts).reshape(-1)
                pred = (MODEL_IZQ if modo == "Zurdo" else MODEL_DER).predict([vec])[0]
                self.last_preds.append(pred)
                vote = Counter(self.last_preds).most_common(1)[0][0]
                self.current_pred = str(vote)

                # dibujar menos seguido
                if (self._f % DRAW_EVERY) == 0:
                    mp.solutions.drawing_utils.draw_landmarks(
                        small, lms, mp.solutions.hands.HAND_CONNECTIONS,
                        mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                        mp.solutions.drawing_styles.get_default_hand_connections_style(),
                    )
            else:
                self.last_preds.clear()
                self.current_pred = "…"

        # Mostrar overlay y devolver frame "small" (el navegador lo escala)
        cv2.rectangle(small, (10, 10), (450, 70), (0, 0, 0), -1)
        cv2.putText(small, f"Predicción: {self.current_pred}", (20, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)

        return av.VideoFrame.from_ndarray(small, format="bgr24")

# ---------- GATE DE SESIONES ----------
if len(ACTIVE) >= MAX_SESS:
    st.warning("Servidor ocupado (límite de cámaras alcanzado). Intentá en un rato 🙏")
else:
    webrtc_ctx = webrtc_streamer(
        key="signia-rtc",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_cfg,
        media_stream_constraints={
            "video": {"facingMode": "user", "width": {"ideal": 1280}, "height": {"ideal": 720}},
            "audio": False,
        },
        video_processor_factory=HandSignProcessor,
        async_processing=True,
        video_html_attrs={"playsinline": True, "autoPlay": True, "muted": True, "controls": False},
    )

    if webrtc_ctx is not None:
        with st.sidebar:
            st.write("Sesiones activas:", len(ACTIVE))
            st.write("Estado:", "playing ✅" if webrtc_ctx.state.playing else "stopped ⛔")
