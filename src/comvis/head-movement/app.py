import streamlit as st
import cv2
import mediapipe as mp
import numpy as np

st.set_page_config(page_title="Industrial Head Movement Detection", layout="wide")

# --------------------------
# INDUSTRY-GRADE HEAD POSE ESTIMATION
# --------------------------

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# 3D facial model points used by solvePnP (standard industry template)
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),            # Nose tip
    (0.0, -330.0, -65.0),       # Chin
    (-225.0, 170.0, -135.0),    # Left eye left corner
    (225.0, 170.0, -135.0),     # Right eye right corner
    (-150.0, -150.0, -125.0),   # Left mouth corner
    (150.0, -150.0, -125.0)     # Right mouth corner
])

LANDMARK_IDX = [1, 199, 33, 263, 61, 291]  # nose, chin, eyes, mouth corners


def rotationMatrixToEulerAngles(R):
    sy = np.sqrt(R[0,0] ** 2 + R[1,0] ** 2)
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(R[2,1], R[2,2])
        y = np.arctan2(-R[2,0], sy)
        z = np.arctan2(R[1,0], R[0,0])
    else:
        x = np.arctan2(-R[1,2], R[1,1])
        y = np.arctan2(-R[2,0], sy)
        z = 0
    
    return np.degrees(y), np.degrees(x), np.degrees(z)  # yaw, pitch, roll


def interpret_head(yaw, pitch, roll):
    """
    Threshold used in industry (driver monitoring, proctoring, attention tracking).
    """
    if yaw > 20:
        return "LOOKING RIGHT"
    elif yaw < -20:
        return "LOOKING LEFT"
    elif pitch > 15:
        return "LOOKING UP"
    elif pitch < -15:
        return "LOOKING DOWN"
    else:
        return "CENTER / FACING FORWARD"


# --------------------------
# STREAMLIT UI
# --------------------------

st.title("🎯 Industrial Head Movement Tracking (No Training Needed)")
st.write("Uses head-pose estimation (solvePnP), same method used in automotive & proctoring systems.")

run = st.checkbox("Start Camera")

FRAME_WINDOW = st.image([])

cap = None

if run:
    cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()
    if not ret:
        st.write("Camera not detected.")
        break

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)

    yaw = pitch = roll = None

    if res.multi_face_landmarks:
        face = res.multi_face_landmarks[0]
        image_points = []

        for i in LANDMARK_IDX:
            lm = face.landmark[i]
            x, y = int(lm.x*w), int(lm.y*h)
            image_points.append((x, y))

        image_points = np.array(image_points, dtype="double")

        focal_length = w
        center = (w/2, h/2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype="double")

        dist_coeffs = np.zeros((4,1))

        success, rot_vec, trans_vec = cv2.solvePnP(
            MODEL_POINTS, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )

        if success:
            rmat, _ = cv2.Rodrigues(rot_vec)
            yaw, pitch, roll = rotationMatrixToEulerAngles(rmat)

            # show angles
            cv2.putText(frame, f"Yaw: {yaw:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.putText(frame, f"Pitch: {pitch:.1f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.putText(frame, f"Roll: {roll:.1f}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            label = interpret_head(yaw, pitch, roll)
            cv2.putText(frame, f"HEAD: {label}", (10, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

    FRAME_WINDOW.image(frame, channels="BGR")

if cap:
    cap.release()
