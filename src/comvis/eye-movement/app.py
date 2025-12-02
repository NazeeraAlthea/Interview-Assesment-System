import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pickle
from PIL import Image

st.set_page_config(page_title="Real-Time Eye Gaze Tracking", layout="wide")

# ---- Load model ----
model = pickle.load(open("src/comvis/eye-movement/unityeyes_eye_model.pkl", "rb"))

# ---- Mediapipe FaceMesh ----
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
drawing = mp.solutions.drawing_utils

# ---- Helper: get iris position ----
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

def get_iris_center(landmarks, idx_list, img_w, img_h):
    pts = []
    for idx in idx_list:
        x = landmarks[idx].x * img_w
        y = landmarks[idx].y * img_h
        pts.append([x, y])
    pts = np.array(pts)
    cx, cy = pts.mean(axis=0)
    return cx, cy

def normalize_feature(iris_x, iris_y, eye_left, eye_right):
    # midpoint of eye corners
    mid_x = (eye_left[0] + eye_right[0]) / 2
    mid_y = (eye_left[1] + eye_right[1]) / 2

    # eye width and height
    eye_width = abs(eye_right[0] - eye_left[0])

    # normalize to -1..+1 range
    gx = (iris_x - mid_x) / (eye_width / 2)
    gx *= 2
    gy = (iris_y - mid_y) / (eye_width / 2)
    gy *= 4.0

    return gx, gy


# ---- Streamlit UI ----
st.title("Real-Time Eye Gaze Detection")
st.write("Model: UnityEyes + MLPClassifier")

run = st.checkbox("Start Webcam")

FRAME_WINDOW = st.image([])

cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()
    if not ret:
        st.write("Tidak bisa membaca webcam!")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, c = frame_rgb.shape

    results = face_mesh.process(frame_rgb)

    gaze_label = "No Face"

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            
            # Ambil iris kiri & kanan
            left_iris_x, left_iris_y = get_iris_center(face_landmarks.landmark, LEFT_IRIS, w, h)
            right_iris_x, right_iris_y = get_iris_center(face_landmarks.landmark, RIGHT_IRIS, w, h)

            # Eye corner indices: kiri = 33, kanan = 263
            eye_left_corner = (
                face_landmarks.landmark[33].x * w,
                face_landmarks.landmark[33].y * h
            )
            eye_right_corner = (
                face_landmarks.landmark[263].x * w,
                face_landmarks.landmark[263].y * h
            )

            # NORMALIZE gaze input
            gx_left, gy_left = normalize_feature(left_iris_x, left_iris_y, eye_left_corner, eye_right_corner)
            gx_right, gy_right = normalize_feature(right_iris_x, right_iris_y, eye_left_corner, eye_right_corner)

            # Kombinasi kedua mata
            gx = (gx_left + gx_right) / 2
            gy = (gy_left + gy_right) / 2

            # ----- MODEL PREDICTION -----
            pred = model.predict([[gx, gy]])[0]
            gaze_label = pred

            # Tampilkan text di kamera
            cv2.putText(frame, f"Gaze: {pred}", (30, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 3)

    FRAME_WINDOW.image(frame, channels="BGR")

cap.release()
