# src/comvis/gaze_estimator.py
import cv2
import numpy as np
import mediapipe as mp
import math
from typing import Tuple, Dict, Optional

# Try to import openvino runtime
try:
    from openvino.runtime import Core
    _HAS_OPENVINO = True
except Exception:
    _HAS_OPENVINO = False

# -----------------------
# Helpers
# -----------------------
def compute_iris_quality_from_landmarks(iris_points: np.ndarray) -> float:
    """
    iris_points: Nx2 array of iris landmarks (pixel coords)
    returns quality in [0..1]
    """
    if iris_points is None or len(iris_points) == 0:
        return 0.0
    center = iris_points.mean(axis=0)
    dists = np.linalg.norm(iris_points - center, axis=1)
    std = float(np.std(dists))
    # tune divisor depending on video resolution; 5 works for 720p/1080p typical
    quality = max(0.0, 1.0 - (std / 5.0))
    return float(np.clip(quality, 0.0, 1.0))

def smooth_value(prev: float, new: float, alpha: float = 0.6) -> float:
    if prev is None:
        return new
    return prev * (1 - alpha) + new * alpha

# -----------------------
# Gaze Estimator class
# -----------------------
class GazeEstimator:
    def __init__(
        self,
        openvino_model_xml: Optional[str] = None,
        iris_threshold: float = 0.45,
        smoothing_alpha: float = 0.6,
    ):
        """
        openvino_model_xml: path to OpenVINO IR model (.xml). If None, OpenVINO disabled.
        iris_threshold: if iris_quality < threshold -> fallback to model
        smoothing_alpha: EMA smoothing for output values
        """
        # mediapipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(refine_landmarks=True)
        self.iris_left_idx = [474, 475, 476, 477]
        self.iris_right_idx = [469, 470, 471, 472]
        self.eye_left_corner_idx = 33
        self.eye_right_corner_idx = 263

        # openvino
        self.ov_core = None
        self.ov_model = None
        self.ov_compiled = None
        self.ov_input_name = None
        self.ov_output_names = None
        if openvino_model_xml and _HAS_OPENVINO:
            self._load_openvino_model(openvino_model_xml)
        else:
            if openvino_model_xml:
                print("[GazeEstimator] Warning: OpenVINO requested but runtime not available.")

        self.iris_threshold = iris_threshold
        self.smoothing_alpha = smoothing_alpha

        # smoothing state
        self.prev_gx = None
        self.prev_gy = None
        self.prev_source = None

    # -----------------------
    # OpenVINO helper
    # -----------------------
    def _load_openvino_model(self, model_xml_path: str):
        print(f"[GazeEstimator] Loading OpenVINO model: {model_xml_path}")
        self.ov_core = Core()
        model = self.ov_core.read_model(model=model_xml_path)
        self.ov_compiled = self.ov_core.compile_model(model, device_name="CPU")
        # auto-detect input and outputs
        inputs = list(self.ov_compiled.inputs)
        outputs = list(self.ov_compiled.outputs)
        if len(inputs) == 0:
            raise RuntimeError("OpenVINO model has no inputs")
        self.ov_input_name = inputs[0].any_name
        self.ov_output_names = [o.any_name for o in outputs]
        print("[GazeEstimator] OpenVINO model loaded. Inputs:", self.ov_input_name, "Outputs:", self.ov_output_names)

    # -----------------------
    # Eye crop helpers
    # -----------------------
    def _crop_eye(self, frame: np.ndarray, eye_landmarks: np.ndarray, pad: float = 0.6) -> np.ndarray:
        """
        Crop a square region around eye landmarks.
        eye_landmarks: Nx2 array (x,y) pixel coords of relevant eye landmarks (corners or iris)
        pad: fraction of box size to add
        Returns resized crop (as BGR).
        """
        if eye_landmarks is None or len(eye_landmarks) == 0:
            return None
        x_min = np.min(eye_landmarks[:, 0])
        x_max = np.max(eye_landmarks[:, 0])
        y_min = np.min(eye_landmarks[:, 1])
        y_max = np.max(eye_landmarks[:, 1])
        w = x_max - x_min
        h = y_max - y_min
        size = max(w, h)
        if size <= 0:
            return None
        cx = (x_min + x_max) / 2.0
        cy = (y_min + y_max) / 2.0
        pad_px = size * pad
        x0 = int(max(0, cx - size/2 - pad_px))
        y0 = int(max(0, cy - size/2 - pad_px))
        x1 = int(min(frame.shape[1], cx + size/2 + pad_px))
        y1 = int(min(frame.shape[0], cy + size/2 + pad_px))
        crop = frame[y0:y1, x0:x1].copy()
        if crop.size == 0:
            return None
        return crop

    # -----------------------
    # OpenVINO inference wrapper (model expected to take eye images + head pose)
    # The exact input format depends on the model you use. This example assumes:
    # - one input: concatenated eye images (or single eye) preprocessed, or adapt as needed.
    # - outputs: gaze vector [x,y,z] or yaw/pitch.
    # You must adapt preprocess to the actual model.
    # -----------------------
    def _run_openvino_inference(self, left_eye_img: np.ndarray, right_eye_img: np.ndarray, head_pose: Tuple[float,float,float]):
        """
        Simple wrapper. Preprocessing highly depends on the model. This is a template.
        Returns dict of outputs.
        """
        if self.ov_compiled is None:
            return None

        # Example preprocessing: convert left eye to grayscale 60x60 and normalize
        # NOTE: adapt this to the model's expected input.
        def prep(img, size=(60,60)):
            if img is None:
                return np.zeros((size[1], size[0], 3), dtype=np.uint8)
            im = cv2.resize(img, size)
            im = im.astype(np.float32) / 255.0
            # change layout to NCHW if model expects it
            return im.transpose(2,0,1)
        left_in = prep(left_eye_img)
        right_in = prep(right_eye_img)
        # stack or concat depending on model
        inp = np.concatenate([left_in, right_in], axis=0)[np.newaxis, ...]  # shape (1, C, H, W)
        # run
        res = self.ov_compiled([inp])[self.ov_compiled.outputs[0]]
        # Interpret result (this depends on model): assume [1x2] yaw/pitch
        out = np.array(res).squeeze()
        # example transform: yaw, pitch
        if out.size >= 2:
            yaw, pitch = float(out[0]), float(out[1])
            return {"yaw": yaw, "pitch": pitch}
        return None

    # -----------------------
    # Public API: estimate_from_frame
    # -----------------------
    def estimate_from_frame(self, frame_bgr: np.ndarray) -> Dict:
        """
        Input: frame BGR
        Output: dict {
            'gx','gy' (if iris used),
            'yaw','pitch' (if head/model used),
            'gaze_label': e.g. 'LEFT','RIGHT','CENTER',
            'source': 'iris' or 'model',
            'iris_quality': float,
            'confidence': float (0..1)
        }
        """
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.face_mesh.process(rgb)

        out = {
            "gx": None, "gy": None,
            "yaw": None, "pitch": None, "roll": None,
            "gaze_label": "UNKNOWN",
            "source": None,
            "iris_quality": 0.0,
            "confidence": 0.0
        }

        if not res.multi_face_landmarks:
            out["gaze_label"] = "NO_FACE"
            return out

        face = res.multi_face_landmarks[0]
        # collect iris points
        lpts = []
        rpts = []
        for i in self.iris_left_idx:
            lm = face.landmark[i]
            lpts.append([lm.x * w, lm.y * h])
        for i in self.iris_right_idx:
            lm = face.landmark[i]
            rpts.append([lm.x * w, lm.y * h])

        lpts = np.array(lpts)
        rpts = np.array(rpts)

        # compute iris quality (take min of two eyes)
        ql = compute_iris_quality_from_landmarks(lpts)
        qr = compute_iris_quality_from_landmarks(rpts)
        iq = min(ql, qr)
        out["iris_quality"] = float(iq)

        # compute eye corners
        leye_corner = face.landmark[self.eye_left_corner_idx]
        reye_corner = face.landmark[self.eye_right_corner_idx]
        eye_left_corner = (leye_corner.x * w, leye_corner.y * h)
        eye_right_corner = (reye_corner.x * w, reye_corner.y * h)

        # compute normalized gx,gy (same as your existing normalize_feature)
        def normalize_feature(iris_x, iris_y, eye_left, eye_right):
            mid_x = (eye_left[0] + eye_right[0]) / 2
            eye_width = max(abs(eye_right[0] - eye_left[0]), 1.0)
            gx = (iris_x - mid_x) / (eye_width / 2.0) * 2.0
            gy = (iris_y - eye_left[1]) / (eye_width / 2.0) * 4.0
            return gx, gy

        # compute centers
        lx, ly = lpts.mean(axis=0)
        rx, ry = rpts.mean(axis=0)
        gx_left, gy_left = normalize_feature(lx, ly, eye_left_corner, eye_right_corner)
        gx_right, gy_right = normalize_feature(rx, ry, eye_left_corner, eye_right_corner)
        gx = float((gx_left + gx_right) / 2.0)
        gy = float((gy_left + gy_right) / 2.0)

        # head-pose (solvePnP) - reuse method like your head_pose module (simple)
        # We'll compute a simple head yaw/pitch estimate via landmarks (coarse)
        # For robust head-pose use your head_pose.Estimator instead; here minimal
        # compute eye vector for coarse yaw
        eye_dx = eye_right_corner[0] - eye_left_corner[0]
        eye_dy = eye_right_corner[1] - eye_left_corner[1]
        # estimate roll rough
        roll = math.degrees(math.atan2(eye_dy, eye_dx))  
        out["roll"] = roll

        # Decide source: iris if quality >= threshold and gx/gy sensible
        if iq >= self.iris_threshold:
            # use iris-based gaze
            label = "CENTER"
            if gx < -0.5:
                label = "LEFT"
            elif gx > 0.5:
                label = "RIGHT"
            # simple up/down
            if gy < -1.5:
                label = "UP"
            elif gy > 1.5:
                label = "DOWN"

            out.update({
                "gx": gx, "gy": gy,
                "gaze_label": label,
                "source": "iris",
                "confidence": float(np.clip(iq, 0.0, 1.0))
            })
            # smoothing
            out["gx"] = smooth_value(self.prev_gx, out["gx"], alpha=self.smoothing_alpha)
            out["gy"] = smooth_value(self.prev_gy, out["gy"], alpha=self.smoothing_alpha)
            self.prev_gx = out["gx"]
            self.prev_gy = out["gy"]
            self.prev_source = "iris"
            return out

        # -----------------------
        # Fallback: model (OpenVINO)
        # -----------------------
        # crop eyes
        left_crop = self._crop_eye(frame_bgr, lpts)
        right_crop = self._crop_eye(frame_bgr, rpts)

        # If openvino available, run model
        model_out = None
        if _HAS_OPENVINO and self.ov_compiled is not None:
            # use head pose rough yaw/pitch (placeholder zeros if not available)
            model_out = self._run_openvino_inference(left_crop, right_crop, (0.0, 0.0, 0.0))

        # If model returned yaw/pitch, convert to gaze label
        if model_out is not None and "yaw" in model_out:
            yaw = model_out["yaw"]
            pitch = model_out.get("pitch", 0.0)
            # simple mapping thresholds (tune empirically)
            label = "CENTER"
            if yaw < -10: label = "LEFT"
            elif yaw > 10: label = "RIGHT"
            if pitch < -10: label = "UP"
            elif pitch > 10: label = "DOWN"

            conf_model = 0.7  # placeholder confidence for model outputs
            out.update({
                "yaw": float(yaw),
                "pitch": float(pitch),
                "gaze_label": label,
                "source": "model",
                "confidence": float(np.clip(conf_model, 0.0, 1.0)),
                "gx": None, "gy": None
            })
            self.prev_source = "model"
            return out

        # If no model, fallback to head pose coarse
        # Use sign of eye positions: if left eye x < center -> looking left etc.
        if gx is not None:
            label = "CENTER"
            if gx < -0.6: label = "LEFT"
            elif gx > 0.6: label = "RIGHT"
            out.update({
                "gx": gx, "gy": gy, "gaze_label": label,
                "source": "head_pose_fallback", "confidence": float(iq * 0.6)
            })
            return out

        # final fallback
        out["gaze_label"] = "UNKNOWN"
        return out
