import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import pickle

# =========================================================
# STREAMLIT CONFIG
# =========================================================
st.set_page_config(page_title="Eye Gaze Detection", layout="wide")
st.title("👁 Real-Time Eye Gaze Detection — Full Pipeline (Stable Version)")

# =========================================================
# LOAD MODEL + ENCODER
# =========================================================
MODEL_PATH = r"D:\05_Personal\Asah by Dicoding\capstone-project\src\comvis\fine-tunning\gaze_high_accuracy.keras"
LABEL_PATH = r"D:\05_Personal\Asah by Dicoding\capstone-project\src\comvis\fine-tunning\label_encoder.pkl"

model = tf.keras.models.load_model(MODEL_PATH)

with open(LABEL_PATH, "rb") as f:
    enc = pickle.load(f)

IMG_SIZE = 128


# =========================================================
# MEDIAPIPE SETUP
# =========================================================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# =========================================================
# PREPROCESSING FUNCTIONS
# =========================================================

def preprocess_eye(img):
    """ Denoise + Sharpen + Gamma Correction """
    
    # 1) DENOISE
    img = cv2.bilateralFilter(img, d=7, sigmaColor=75, sigmaSpace=75)

    # 2) SHARPEN (Unsharp Mask)
    blur = cv2.GaussianBlur(img, (0, 0), 3)
    img = cv2.addWeighted(img, 1.8, blur, -0.8, 0)

    # 3) GAMMA CORRECTION
    gamma = 1.2
    inv = 1.0 / gamma
    table = np.array([(i/255.0)**inv * 255 for i in np.arange(256)]).astype("uint8")
    img = cv2.LUT(img, table)

    # 4) FINAL RESIZE
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    return img


# =========================================================
# CROP EYE FUNCTION (STABLE VERSION)
# =========================================================

def crop_eye(image, landmarks, side="left"):
    h, w, _ = image.shape

    # IRIS + EYELID LANDMARKS (VERY PRECISE)
    if side == "left":
        ids = [33, 133, 160, 159, 158, 157, 173, 468, 469, 470, 471]
    else:
        ids = [263, 362, 387, 386, 385, 384, 398, 473, 474, 475, 476]

    pts = np.array([(int(landmarks[i].x*w), int(landmarks[i].y*h)) for i in ids])

    x, y, ww, hh = cv2.boundingRect(pts)

    # AUTO ZOOM (BIGGER CROP)
    margin = int(max(ww, hh) * 0.6)

    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(w, x + ww + margin)
    y2 = min(h, y + hh + margin)

    crop = image[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    crop = preprocess_eye(crop)
    return crop


# =========================================================
# SMOOTHING (EMA)
# =========================================================
smooth_pred = None
alpha = 0.3

def smooth(prev, new):
    if prev is None:
        return new
    return prev * (1-alpha) + new * alpha


# =========================================================
# STREAMLIT CAMERA UI
# =========================================================

run = st.checkbox("Start Camera")
frame_view = st.image([])
eye_view = st.image([])

if run:
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            st.write("⚠️ Cannot access camera")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(frame_rgb)

        label_text = "Detecting..."

        if result.multi_face_landmarks:
            lm = result.multi_face_landmarks[0].landmark

            # Try left eye (better for right-handed webcam setups)
            eye = crop_eye(frame_rgb, lm, side="left")

            if eye is not None:
                eye_view.image(eye, caption="Eye Input", width=200)

                # PREPROCESS FOR MODEL
                inp = eye.astype(np.float32)
                inp = np.expand_dims(inp, 0)
                inp = tf.keras.applications.mobilenet_v2.preprocess_input(inp)

                pred = model.predict(inp)[0]
                pred_s = smooth(smooth_pred, pred)
                smooth_pred = pred_s

                cls_id = np.argmax(pred_s)
                cls_name = enc.inverse_transform([cls_id])[0]
                conf = float(pred_s[cls_id])

                label_text = f"{cls_name}  ({conf:.2f})"

                cv2.putText(frame_rgb, label_text, (30,40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,255,0), 2)

        frame_view.image(frame_rgb)

    cap.release()
