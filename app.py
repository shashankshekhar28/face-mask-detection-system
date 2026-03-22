import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
#to run--- "python -m streamlit run app.py"
# -----------------------------
# Load YOUR trained model
# -----------------------------
@st.cache_resource
def load_mask_model():
    return load_model(
        r"C:\Users\Shashank\projects\mask detection system\mask_detector_model.keras"
    )

model = load_mask_model()

# -----------------------------
# Load face detector
# -----------------------------
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# -----------------------------
# Video Processor (FAST)
# -----------------------------
class MaskDetector(VideoProcessorBase):
    def __init__(self):
        self.frame_count = 0

    def recv(self, frame):
        self.frame_count += 1

        img = frame.to_ndarray(format="bgr24")

        # 🔹 Resize frame for speed
        img_small = cv2.resize(img, (640, 480))
        gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=5
        )

        # 🔹 Run prediction every 5 frames only
        if self.frame_count % 5 == 0:
            for (x, y, w, h) in faces:
                face = img_small[y:y+h, x:x+w]
                face = cv2.resize(face, (224, 224))
                face = face / 255.0
                face = np.expand_dims(face, axis=0)

                pred = model.predict(face, verbose=0)

                if pred.shape[-1] == 1:
                    mask = pred[0][0] < 0.5
                    confidence = float(pred[0][0])
                else:
                    class_id = np.argmax(pred[0])
                    mask = class_id == 0
                    confidence = float(np.max(pred[0]))

                label = (
                    f"Mask 😷 ({confidence:.2f})"
                    if mask else f"No Mask ❌ ({confidence:.2f})"
                )
                color = (0, 255, 0) if mask else (0, 0, 255)

                cv2.rectangle(img_small, (x, y), (x+w, y+h), color, 2)
                cv2.putText(
                    img_small, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
                )

        return av.VideoFrame.from_ndarray(img_small, format="bgr24")

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Real-Time Mask Detection", layout="centered")
st.title("😷 Real-Time Face Mask Detection")
st.write("Optimized real-time detection using your VGG16 model")

webrtc_streamer(
    key="mask-detection",
    video_processor_factory=MaskDetector,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,   # 🔥 CRITICAL for smooth video
)

