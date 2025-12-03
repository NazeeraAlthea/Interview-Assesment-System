import cv2
import numpy as np
import mediapipe as mp

class HeadPoseEstimator:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(refine_landmarks=True)

        # model 3D points
        self.MODEL_POINTS = np.array([
            (0.0, 0.0, 0.0),          # Nose tip
            (0.0, -330.0, -65.0),     # Chin
            (-225.0, 170.0, -135.0),  # Left eye left corner
            (225.0, 170.0, -135.0),   # Right eye right corner
            (-150.0, -150.0, -125.0), # Left mouth corner
            (150.0, -150.0, -125.0)   # Right mouth corner
        ])

        self.LANDMARK_IDX = [1, 199, 33, 263, 61, 291]

    def rotationMatrixToEulerAngles(self, R):
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

    def interpret_head(self, yaw, pitch, roll):
        if yaw > 20:
            return "LOOKING RIGHT"
        elif yaw < -20:
            return "LOOKING LEFT"
        elif pitch > 15:
            return "LOOKING UP"
        elif pitch < -15:
            return "LOOKING DOWN"
        return "CENTER"

    def estimate_from_frame(self, frame):
        """
        Input: frame BGR
        Output: yaw, pitch, roll, label
        """
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.face_mesh.process(rgb)

        if not res.multi_face_landmarks:
            return None, None, None, "NO_FACE"

        face = res.multi_face_landmarks[0]
        image_points = []

        for i in self.LANDMARK_IDX:
            lm = face.landmark[i]
            image_points.append((int(lm.x*w), int(lm.y*h)))

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
            self.MODEL_POINTS, image_points, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return None, None, None, "NO_FACE"

        rmat, _ = cv2.Rodrigues(rot_vec)
        yaw, pitch, roll = self.rotationMatrixToEulerAngles(rmat)
        label = self.interpret_head(yaw, pitch, roll)

        return yaw, pitch, roll, label
