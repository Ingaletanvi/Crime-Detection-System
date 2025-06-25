import streamlit as st
from ultralytics import YOLO
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import tempfile
import os
import smtplib
from email.message import EmailMessage

# --------------------- Configuration ---------------------
EMAIL_SENDER = 'itanvi258@gmail.com'
EMAIL_PASSWORD = 'twvx wrof nzoz cvjd'  # App Password from Gmail
EMAIL_RECEIVER = 'princess.tanvi2508@gmail.com'

# --------------------- Login Credentials ---------------------
USER_CREDENTIALS = {
    "tanvi": "password",
    "user": "crime123"
}

@st.cache_resource(show_spinner=False)
def load_yolo_model():
    model = YOLO(r"C:\Users\itanv\Desktop\CRIME youtube dataset - Copy\crime_detection_system\runs\detect\train\weights\best.pt")
    return model

@st.cache_resource(show_spinner=False)
def load_3dcnn_model():
    model = load_model(r"C:\Users\itanv\Desktop\CRIME youtube dataset - Copy\crime_detection_system\models\3d_cnn_model.h5")
    return model

def extract_frames(video_path, frame_count=16):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // frame_count)

    for i in range(frame_count):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return frames

def prepare_frames_for_3dcnn(frames, target_size=(112, 112)):
    processed = [cv2.resize(f, target_size) / 255.0 for f in frames]
    processed = np.expand_dims(np.array(processed), axis=0)
    return processed

def run_yolo_detection(yolo_model, video_path):
    results = yolo_model(video_path)
    return results[0].plot()

def run_3dcnn_prediction(cnn_model, frames):
    input_data = prepare_frames_for_3dcnn(frames)
    preds = cnn_model.predict(input_data)
    return preds

def send_email_alert():
    msg = EmailMessage()
    msg['Subject'] = 'Crime Detected Alert'
    msg['From'] = EMAIL_SENDER
    msg['To'] = EMAIL_RECEIVER
    msg.set_content("Crime detected by the YOLO + 3D CNN surveillance system. Immediate attention required.")

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(EMAIL_SENDER, EMAIL_PASSWORD)
            smtp.send_message(msg)
        return True
    except Exception as e:
        st.error(f"Failed to send alert email: {e}")
        return False

# --------------------- UI Starts Here ---------------------
st.set_page_config(page_title="Crime Detection System", layout="wide")

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# --------------------- Login Page ---------------------
if not st.session_state.logged_in:
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
            st.session_state.logged_in = True
            st.success("Login successful!")
        else:
            st.error("Invalid credentials. Try again.")
    st.stop()

# --------------------- Main App ---------------------
st.title("Crime Detection System - YOLOv8 + 3D CNN")

video_source = st.radio("Choose video source:", ["Upload Video", "Use CCTV Stream"])
temp_video_path = None

if video_source == "Upload Video":
    uploaded_file = st.file_uploader("Upload a video file for crime detection", type=["mp4", "avi", "mov"])
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            temp_video.write(uploaded_file.read())
            temp_video_path = temp_video.name
            st.video(temp_video_path)

elif video_source == "Use CCTV Stream":
    stream_url = st.text_input("Enter CCTV stream URL (e.g. rtsp:// or http://):")
    if stream_url:
        temp_video_path = stream_url

# --------------------- Detect Button ---------------------
if temp_video_path and st.button("Detect Crime"):
    yolo_model = load_yolo_model()
    cnn_model = load_3dcnn_model()

    st.write("Running YOLOv8 Object Detection...")
    yolo_img = run_yolo_detection(yolo_model, temp_video_path)
    st.image(yolo_img, caption="YOLOv8 Detection Result", use_column_width=True)

    st.write("Running 3D CNN Action Recognition...")
    frames = extract_frames(temp_video_path)
    preds = run_3dcnn_prediction(cnn_model, frames)

    class_names = ['Normal Activity', 'Criminal Activity']
    pred_class = class_names[np.argmax(preds)]
    pred_prob = np.max(preds)

    st.write(f"3D CNN Prediction: **{pred_class}** with confidence {pred_prob:.2f}")

    if pred_class == "Criminal Activity":
        st.error("🚨 Alert! Crime detected in the video.")
        if send_email_alert():
            st.success("📧 Alert email sent to authorities.")
    else:
        st.success("✅ No crime detected in the video.")

    if video_source == "Upload Video" and os.path.exists(temp_video_path):
        os.remove(temp_video_path)
