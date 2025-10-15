# app.py
import os
import base64
import pathlib
from collections import Counter, deque
import gc

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

# FORZAR MODO SELFIE (preview espejada)
# - LITE: st.camera_input
# - LIVE: streamlit-webrtc <video>
st.markdown("""
<style>
/* LITE: espejar preview de cámara */
[data-testid="stCameraInput"] video,
[data-testid="stCameraInput"] canvas {
    transform: scaleX(-1) !important;
}
/* LIVE: espejar video */
.st-webrtc video, video#streamlit-webrtc-video,
video[playsinline][autoplay] {
    transform: scaleX(-1) !important;
}
</style>
""", unsafe_allow_html=True)

LIVE_MODE = os.getenv("LIVE_MODE", "0") in ("1", "true", "True")
BUILD_TAG = os.getenv("RENDER_GIT_COMMIT", "local")[:7]

ASSET_LOGO = "assets/logo.png"
TUTORIAL_PDF = "docs/tutorial.pdf"

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

def predict_with_model(vec: np.ndarray, mano_sel: str) -> str:
    """Predice con el modelo elegido manualmente (Diestro/Zurdo)."""
    model = MODEL_IZQ if mano_sel == "Zurdo" else MODEL_DER
    if model is None:
        return "¿?"
    return model.predict([vec])[0]

def swap_lr(label: str) -> str:
    """Invierte etiqueta Left/Right (útil cuando espejamos)."""
    if label == "Left":
        return "Right"
    if label == "Right":
        return "Left"
    return label

def free_vars(*vars_):
    """Libera memoria de arrays grandes para Render Free."""
    for v in vars_:
        try:
            del v
        except Exception:
            pass
    gc.collect()

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    if file_exists(ASSET_LOGO):
        st.image(ASSET_LOGO, use_container_width=True)
    st.header("Ajustes")
    mano = st.radio("Elegí tu mano (para el modelo):", ["Diestro", "Zurdo"], index=0)
    auto_hand = st.toggle("Detectar mano automáticamente (ajustada a selfie)", False)
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
        "Dataset y procesamiento en **modo selfie** (espejado) para que la vista previa coincida con lo que se procesa.\n"
        "1) Elegí tu mano (o activá la detección auto ajustada a selfie).\n"
        "2) Tomá la foto o subí una imagen. ¡Listo!"
    )
    show_pdf(TUTORIAL_PDF, height=820)

    if file_exists(TUTORIAL_PDF):
        with open(TUTORIAL_PDF, "rb") as f:
            st.download_button(
                "⬇️ Descargar tutorial (PDF)",
                data=f,
                file_name="SIGNIA_Tutorial.pdf",
                mime="application/pdf",
                use_container_width=True
            )

# =========================
# TAB: DEMO (LIVE o LITE)
# =========================
with tab_demo:
    st.caption("Elegí modo LIVE/LITE por variable de entorno LIVE_MODE (1/0).")

    # -------- LITE (sin WebRTC) --------
    if not LIVE_MODE:
        col1, col2 = st.columns(2)
        with col1:
            img_cam = st.camera_input("Tomá una foto de tu seña (modo selfie)")
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
            free_vars(img_bytes, nparr, bgr)
            st.stop()

        # ===== CLAVE: ESPEJAR PARA QUE COINCIDA CON LA PREVIEW (modo selfie) =====
        bgr = cv2.flip(bgr, 1)

        # Opcional: limitar ancho para ahorrar recursos
        max_w = 960
        if bgr.shape[1] > max_w:
            scale = max_w / bgr.shape[1]
            bgr = cv2.resize(bgr, (int(bgr.shape[1]*scale), int(bgr.shape[0]*scale)),
                             interpolation=cv2.INTER_AREA)

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        with mp.solutions.hands.Hands(
            static_image_mode=True, max_num_hands=1, min_detection_confidence=0.8
        ) as hands:
            res = hands.process(rgb)

        if not res.multi_hand_landmarks:
            st.error("No se detectó mano. Probá otra toma (fondo claro, mano completa).")
            free_vars(img_bytes, nparr, bgr, rgb, res)
            st.stop()

        lms = res.multi_hand_landmarks[0]
        pts = np.array([[lm.x, lm.y, lm.z] for lm in lms.landmark], dtype=float)
        vec = normalize_seq_xy(pts).reshape(-1)

        # Auto-hand ajustado a selfie: MediaPipe devuelve L/R del frame espejado -> invertimos
        mano_base = mano
        if auto_hand and res.multi_handedness:
            label = res.multi_handedness[0].classification[0].label  # 'Left' o 'Right'
            label_corr = swap_lr(label)  # corregir por selfie
            mano_base = "Zurdo" if label_corr == "Left" else "Diestro"

        pred = predict_with_model(vec, mano_base)

        # SOLO la predicción (no mostramos imágenes para ahorrar recursos)
        st.success(f"✅ Predicción: **{pred}** · Mano usada: **{mano_base}**")

        # Limpieza de memoria
        free_vars(img_bytes, nparr, bgr, rgb, res, lms, pts, vec)
        st.stop()

    # -------- LIVE (WebRTC) --------
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
                    static_image_mode=False, model_complexity=0,
                    max_num_hands=1, min_detection_confidence=0.8, min_tracking_confidence=0.8
                )
                self.buf = deque(maxlen=5)
                self.pred = "…"
                self._f = 0
                self.mano_usada = "?"

            def recv(self, frame):
                import av
                img = frame.to_ndarray(format="bgr24")

                # ===== CLAVE: ESPEJAR PARA QUE COINCIDA CON LA PREVIEW (modo selfie) =====
                img = cv2.flip(img, 1)

                self._f += 1
                small = cv2.resize(img, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)

                if self._f % SKIP_N == 0:
                    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                    r = self.hands.process(rgb)
                    if r.multi_hand_landmarks:
                        lms = r.multi_hand_landmarks[0]
                        pts = np.array([[lm.x, lm.y, lm.z] for lm in lms.landmark], dtype=float)
                        vec = normalize_seq_xy(pts).reshape(-1)

                        # Auto-hand ajustado a selfie
                        mano_base = mano
                        if auto_hand and r.multi_handedness:
                            label = r.multi_handedness[0].classification[0].label
                            label_corr = swap_lr(label)  # corregimos por selfie
                            mano_base = "Zurdo" if label_corr == "Left" else "Diestro"

                        p = predict_with_model(vec, mano_base)
                        self.buf.append(p)
                        self.pred = Counter(self.buf).most_common(1)[0][0]
                        self.mano_usada = mano_base

                        mp.solutions.drawing_utils.draw_landmarks(
                            small, lms, mp.solutions.hands.HAND_CONNECTIONS,
                            mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                            mp.solutions.drawing_styles.get_default_hand_connections_style(),
                        )
                    else:
                        self.buf.clear()
                        self.pred = "…"
                        self.mano_usada = "?"

                cv2.rectangle(small, (10, 10), (720, 70), (0, 0, 0), -1)
                txt = f"Predicción: {self.pred} · Mano: {self.mano_usada}"
                cv2.putText(small, txt, (20, 55),
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
