# src/comvis/eye_movement.py
import cv2
import mediapipe as mp
import numpy as np
import pickle

LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

class EyeGazeEstimator:
    def __init__(self, model_path="src/comvis/eye-movement/unityeyes_eye_model.pkl", refine_landmarks=True):
        # load small classifier (MLP / RF / etc)
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

        # mediapipe face mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(refine_landmarks=refine_landmarks)
        # drawing utils kept out (UI will draw)
    
    def _get_iris_center(self, landmarks, idx_list, img_w, img_h):
        pts = []
        for idx in idx_list:
            x = landmarks[idx].x * img_w
            y = landmarks[idx].y * img_h
            pts.append([x, y])
        pts = np.array(pts)
        cx, cy = pts.mean(axis=0)
        return float(cx), float(cy)

    def _normalize_feature(self, iris_x, iris_y, eye_left, eye_right):
        # midpoint of eye corners
        mid_x = (eye_left[0] + eye_right[0]) / 2.0
        mid_y = (eye_left[1] + eye_right[1]) / 2.0

        # eye width (pixel)
        eye_width = max(abs(eye_right[0] - eye_left[0]), 1.0)  # avoid div0

        # normalize to roughly -1..+1 but tuned
        gx = (iris_x - mid_x) / (eye_width / 2.0)
        gx *= 2.0
        gy = (iris_y - mid_y) / (eye_width / 2.0)
        gy *= 4.0

        return float(gx), float(gy)

    def predict_from_frame(self, frame_bgr):
        """
        Input: BGR frame (numpy array)
        Output: dict {
            'gaze_label': str,
            'gaze_probs': dict or None,
            'gx': float,
            'gy': float,
            'left_iris': (x,y),
            'right_iris': (x,y),
            'status': 'OK' or 'NO_FACE'
        }
        """
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.face_mesh.process(rgb)

        out = {
            "gaze_label": None,
            "gaze_probs": None,
            "gx": None,
            "gy": None,
            "left_iris": None,
            "right_iris": None,
            "status": "NO_FACE"
        }

        if not res.multi_face_landmarks:
            return out

        # use first face detected
        face_landmarks = res.multi_face_landmarks[0].landmark

        # iris centers
        lx, ly = self._get_iris_center(face_landmarks, LEFT_IRIS, w, h)
        rx, ry = self._get_iris_center(face_landmarks, RIGHT_IRIS, w, h)

        # eye corner indices (left=33, right=263)
        eye_left_corner = (face_landmarks[33].x * w, face_landmarks[33].y * h)
        eye_right_corner = (face_landmarks[263].x * w, face_landmarks[263].y * h)

        gx_left, gy_left = self._normalize_feature(lx, ly, eye_left_corner, eye_right_corner)
        gx_right, gy_right = self._normalize_feature(rx, ry, eye_left_corner, eye_right_corner)

        gx = (gx_left + gx_right) / 2.0
        gy = (gy_left + gy_right) / 2.0

        # predict with your model
        try:
            label = self.model.predict([[gx, gy]])[0]
        except Exception as e:
            # fallback: if model fails, return raw vector
            out.update({
                "gaze_label": None,
                "gaze_probs": None,
                "gx": gx,
                "gy": gy,
                "left_iris": (lx, ly),
                "right_iris": (rx, ry),
                "status": f"MODEL_ERROR: {e}"
            })
            return out

        # try to get probabilities if available
        probs = None
        try:
            if hasattr(self.model, "predict_proba"):
                p = self.model.predict_proba([[gx, gy]])[0]
                classes = getattr(self.model, "classes_", None)
                if classes is not None:
                    probs = {str(classes[i]): float(p[i]) for i in range(len(p))}
                else:
                    probs = {str(i): float(p[i]) for i in range(len(p))}
        except Exception:
            probs = None

        out.update({
            "gaze_label": str(label),
            "gaze_probs": probs,
            "gx": float(gx),
            "gy": float(gy),
            "left_iris": (float(lx), float(ly)),
            "right_iris": (float(rx), float(ry)),
            "status": "OK"
        })

        return out
