# app.py
import os
import base64
import pathlib
from collections import Counter, deque

import numpy as np
import cv2
import streamlit as st
from joblib import load
import mediapipe as mp

# =========================
# CONFIG & CONSTANTES
# =========================
st.set_page_config(
    page_title="SIGNIA",
    page_icon="assets/logo.png",   # favicon
    layout="wide"
)

# Forzar preview de cámara NO ESPEJO en st.camera_input (solo la vista previa)
st.markdown("""
<style>
/* Vista previa de cámara en el widget de Streamlit (LITE) */
[data-testid="stCameraInput"] video,
[data-testid="stCameraInput"] canvas {
    transform: none !important;   /* sin espejo */
}
</style>
""", unsafe_allow_html=True)

LIVE_MODE = os.getenv("LIVE_MODE", "0") in ("1", "true", "True")
BUILD_TAG = os.getenv("RENDER_GIT_COMMIT", "local")[:7]

ASSET_LOGO = "assets/logo.png"
tutorial = "docs/tutorial.pdf"

TURN_URL = os.getenv("TURN_URL", "turn:openrelay.metered.ca:80")
TURN_USERNAME = os.getenv("TURN_USERNAME", "openrelayproject")
TURN_CREDENTIAL = os.getenv("TURN_CREDENTIAL", "openrelayproject")

SKIP_N = int(os.getenv("SKIP_N", "3"))
TARGET_W, TARGET_H = 640, 480

# =========================
# UTILS
# =========================
def file_exists(path: str) -> bool:
    try:
        return pathlib.Path(path).exists()
    except Exception:
        return False

def show_pdf(path: str, height: int = 800):
    """Embed un PDF dentro de la app (sin hosting extra)."""
    if not file_exists(path):
        st.warning("No se encontró el archivo del tutorial. Asegurate de subir `docs/tutorial.pdf`.")
        return
    with open(path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode("utf-8")
    pdf_iframe = f"""
    <iframe
        src="data:application/pdf;base64,{base64_pdf}"
        width="100%" height="{height}"
        style="border:none; border-radius: 8px;"
    ></iframe>
    """
    st.components.v1.html(pdf_iframe, height=height + 20)

def normalize_seq_xy(seq_xyz: np.ndarray) -> np.ndarray:
    seq = seq_xyz.copy().astype(float)
    xy = seq[:, :2]
    xy -= xy.mean(axis=0, keepdims=True)
    max_abs = float(np.abs(xy).max()) or 1.0
    xy /= max_abs
    seq[:, :2] = xy
    return seq

@st.cache_resource
def load_models():
    """Carga segura de modelos (maneja ausencia de archivos)."""
    m_izq = m_der = None
    try:
        m_izq = load("modelo_letras_izq_rf.joblib")["model"]
    except Exception as e:
        st.warning(f"No se pudo cargar modelo izquierdo: {e}")
    try:
        m_der = load("modelo_letras_der_rf.joblib")["model"]
    except Exception as e:
        st.warning(f"No se pudo cargar modelo derecho: {e}")
    return m_izq, m_der

MODEL_IZQ, MODEL_DER = load_models()

def predict_letter(vec: np.ndarray, mano: str) -> str:
    """Predice letra según mano elegida. Devuelve '¿?' si no hay modelo."""
    model = MODEL_IZQ if mano == "Zurdo" else MODEL_DER
    if model is None:
        return "¿?"
    return model.predict([vec])[0]

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    if file_exists(ASSET_LOGO):
        st.image(ASSET_LOGO, use_container_width=True)
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
    st.caption(f"🧱 Build: {BUILD_TAG} · Live: {LIVE_MODE}")
    st.caption("SIGNIA · MediaPipe + RandomForest")

# =========================
# HEADER CON LOGO + TABS
# =========================
col_logo, col_title = st.columns([1, 6], vertical_alignment="center")
with col_logo:
    if file_exists(ASSET_LOGO):
        st.image(ASSET_LOGO, use_container_width=True)
with col_title:
    st.title("Bienvenid@ a SIGNIA")

tab_demo, tab_tutorial = st.tabs(["🎥 Demo", "📘 Tutorial"])

# =========================
# TAB: TUTORIAL
# =========================
with tab_tutorial:
    st.subheader("Cómo usar SIGNIA")
    st.write(
        "‼ Recomendaciones: fondo claro, una sola mano que se vea completa y bien iluminada.\n"
        "1- Elegí tu mano (diestro/zurdo) para calibrar el modelo.\n"
        "2- Tomá la foto, o subí una desde tus archivos. ¡Listo!"
    )
    show_pdf(tutorial, height=820)

    if file_exists(tutorial):
        with open(tutorial, "rb") as f:
            st.download_button(
                "⬇️ Descargar tutorial (PDF)",
                data=f,
                file_name="tutorial.pdf",
                mime="application/pdf",
                use_container_width=True
            )

# =========================
# TAB: DEMO (LIVE o LITE)
# =========================
with tab_demo:
    st.caption("Elegí modo LIVE/LITE por variable de entorno LIVE_MODE (1/0).")

    # -------- LITE (sin WebRTC): camera_input / upload --------
    if not LIVE_MODE:
        col1, col2 = st.columns(2)
        with col1:
            img_cam = st.camera_input("Tomá una foto de tu seña (una sola mano)")
        with col2:
            img_up = st.file_uploader("…o subí una imagen", type=["jpg", "jpeg", "png"])

        img_bytes, origen = None, None
        if img_cam is not None:
            img_bytes = img_cam.getvalue()
            origen = "cámara"
        elif img_up is not None:
            img_bytes = img_up.read()
            origen = "archivo"

        if img_bytes is None:
            st.info("Esperando una imagen…")
            st.stop()

        # Decodificar
        nparr = np.frombuffer(img_bytes, np.uint8)
        bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if bgr is None:
            st.error("No se pudo leer la imagen.")
            st.stop()

        # Procesar SIN espejo (coherencia L/R) y SIN mostrar imagen
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        with mp.solutions.hands.Hands(
            static_image_mode=True, max_num_hands=1, min_detection_confidence=0.8
        ) as hands:
            res = hands.process(rgb)

        if not res.multi_hand_landmarks:
            st.error("No se detectó mano. Probá otra toma (fondo claro, mano completa).")
            st.stop()

        # Extraer landmarks
        lms = res.multi_hand_landmarks[0]
        pts = np.array([[lm.x, lm.y, lm.z] for lm in lms.landmark], dtype=float)
        vec = normalize_seq_xy(pts).reshape(-1)
        pred = predict_letter(vec, mano)

        # (Opcional) podrías dibujar en 'rgb' pero NO mostramos imagen.
        # mp.solutions.drawing_utils.draw_landmarks(...)

        # Mostrar SOLO la predicción (sin imágenes)
        st.success(f"✅ Predicción: **{pred}**")
        st.stop()

    # -------- LIVE (WebRTC): streaming en tiempo real --------
    else:
        try:
            from streamlit_webrtc import (
                webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode
            )
        except Exception:
            st.error("Falta `streamlit-webrtc`. Instalalo con `pip install streamlit-webrtc`.")
            st.stop()

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
                # SIN espejo: mostramos la cámara tal cual llega

                self._f += 1
                small = cv2.resize(img, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)

                if self._f % SKIP_N == 0:
                    # Procesar tal cual (sin flips)
                    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                    r = self.hands.process(rgb)
                    if r.multi_hand_landmarks:
                        lms = r.multi_hand_landmarks[0]
                        pts = np.array([[lm.x, lm.y, lm.z] for lm in lms.landmark], dtype=float)
                        vec = normalize_seq_xy(pts).reshape(-1)
                        p = predict_letter(vec, mano)
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

                cv2.rectangle(small, (10, 10), (520, 70), (0, 0, 0), -1)
                cv2.putText(small, f"Predicción: {self.pred}", (20, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
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
