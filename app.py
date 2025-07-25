import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2
import numpy as np

# Page config
st.set_page_config(page_title="🎯 Moving Object Detection", layout="wide")

# Sidebar
st.sidebar.title("🛠️ Detection Settings")
area_threshold = st.sidebar.slider("🎯 Minimum Object Area", 100, 3000, 500, step=50)
gray_view_toggle = st.sidebar.checkbox("👓 Show Grayscale View", value=False)

# Header
st.markdown("""
    <style>
        .title {
            text-align: center;
            color: #3399ff;
            font-size: 45px;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .subtitle {
            text-align: center;
            color: #ccc;
            font-size: 22px;
            font-weight: bold;
            margin-top: 5px;
            margin-bottom: 30px;
        }
    </style>

    <div class="title">🎯 Moving Object Detection using OpenCV</div>
    <div class="subtitle">⚡ Real-time motion detection using webcam feed</div>
""", unsafe_allow_html=True)



st.markdown("### ▶️ Click on **Start** to activate webcam and detect motion")

# Motion Detection Processor
class MotionDetector(VideoProcessorBase):
    def __init__(self):
        self.first_frame = None
        self.text = "No Motion Detected"
        self.gray_view = gray_view_toggle

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.resize(img, (800, 600))

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (21, 21), 0)

        if self.first_frame is None:
            self.first_frame = blur
            return av.VideoFrame.from_ndarray(cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR), format="bgr24")

        delta = cv2.absdiff(self.first_frame, blur)
        thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        motion_detected = False

        for contour in contours:
            if cv2.contourArea(contour) < area_threshold:
                continue
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            motion_detected = True

        self.text = "Motion Detected!" if motion_detected else "No Motion Detected"
        text_color = (0, 0, 255) if motion_detected else (0, 255, 0)  # Red for motion, green otherwise
        cv2.putText(img, self.text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, text_color, 3)

        if self.gray_view:
            # ✅ Convert the full image (with boxes/text) to grayscale for display
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Start Webcam Stream
webrtc_streamer(
    key="motion-detection",
    video_processor_factory=MotionDetector,
    media_stream_constraints={"video": True, "audio": False},  # ✅ Audio removed
    async_processing=True,
)
