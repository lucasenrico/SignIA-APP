import streamlit as st
from joblib import load
import numpy as np
import cv2
import mediapipe as mp
from collections import deque, Counter
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

st.set_page_config(page_title="SIGNIA - LSA en tiempo real", layout="wide")
st.title("SIGNIA – Reconocimiento de señas (tiempo real)")

# ---- Modelos ----
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
    corregir_espejo = st.checkbox("Corregir espejo de cámara", value=True)
    st.caption("Si tu cámara se ve como selfie, dejá activado 'Corregir espejo'.")

# ---- WebRTC (stun público) ----
rtc_cfg = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

class HandSignTransformer(VideoTransformerBase):
    def __init__(self):
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            model_complexity=1,
            max_num_hands=1,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8
        )
        self.last_preds = deque(maxlen=5)
        self.current_pred = "…"

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        if corregir_espejo:
            img = cv2.flip(img, 1)

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
                img, lms, mp.solutions.hands.HAND_CONNECTIONS,
                mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                mp.solutions.drawing_styles.get_default_hand_connections_style()
            )
        else:
            self.last_preds.clear()
            self.current_pred = "…"

        cv2.rectangle(img, (10, 10), (360, 70), (0, 0, 0), -1)
        cv2.putText(img, f"Predicción: {self.current_pred}", (20, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        return img

st.info("Dale permiso a la cámara. La predicción aparece sobre el video (5 frames suavizados).")
webrtc_streamer(
    key="signia-rtc",
    video_transformer_factory=HandSignTransformer,
    media_stream_constraints={"video": True, "audio": False},
    rtc_configuration=rtc_cfg,
)
