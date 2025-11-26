import cv2
import mediapipe as mp
import numpy as np
import math
import time

mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection

cap = cv2.VideoCapture(0)

# helper: rata-rata koordinat landmark index tertentu (pixel coords)
def mean_landmark_xy(landmarks, indices, shape):
    h, w = shape
    xs = [landmarks[i].x * w for i in indices]
    ys = [landmarks[i].y * h for i in indices]
    return int(np.mean(xs)), int(np.mean(ys))

# helper: hitung euler angles (deg) dari rotation matrix
def rotationMatrixToEulerAngles(R):
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else:
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    # convert to degrees: roll (x), pitch (y), yaw (z)
    return np.degrees(x), np.degrees(y), np.degrees(z)

with mp_face_mesh.FaceMesh(
    max_num_faces=5,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh, mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_det:

    last_time = time.time()
    fps = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]

        face_result = face_det.process(rgb)
        mesh_result = face_mesh.process(rgb)

        face_boxes = []
        if face_result.detections:
            for det in face_result.detections:
                bbox = det.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                bw = int(bbox.width * w)
                bh = int(bbox.height * h)
                face_boxes.append((x, y, bw, bh))

        # if multiple face meshes detected, iterate
        if mesh_result.multi_face_landmarks:
            for i, face_landmarks in enumerate(mesh_result.multi_face_landmarks):
                # ensure we have matching face box; if not, continue
                if i >= len(face_boxes):
                    continue
                x, y, bw, bh = face_boxes[i]

                # draw bounding box (thicker)
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), (20, 200, 255), 3)

                lm = face_landmarks.landmark

                # ----- compute iris centers (pixel coords) -----
                # Mediapipe iris indices
                iris_left = mean_landmark_xy(lm, [473, 474, 475, 476], (h, w))
                iris_right = mean_landmark_xy(lm, [468, 469, 470, 471], (h, w))

                # eye corner centers (approx) - used to compute gaze vector in image plane
                left_eye_corner = mean_landmark_xy(lm, [33, 133], (h, w))    # left eye outer/inner approx
                right_eye_corner = mean_landmark_xy(lm, [362, 263], (h, w))  # right eye approx

                # ----- draw gaze vector lines (image-plane approximation) -----
                # left eye gaze vector: from iris_left in direction (iris_left - eye_center)
                def draw_gaze_line(iris, eye_corner, color=(0,255,0)):
                    ix, iy = iris
                    ex, ey = eye_corner
                    vx = ix - ex
                    vy = iy - ey
                    norm = math.hypot(vx, vy) + 1e-6
                    vx_n, vy_n = vx / norm, vy / norm
                    length = int(min(w, h) * 0.25)  # panjang line relatif frame
                    x2 = int(ix + vx_n * length)
                    y2 = int(iy + vy_n * length)
                    # line
                    cv2.line(frame, (ix, iy), (x2, y2), color, 2, cv2.LINE_AA)
                    # arrow head
                    cv2.circle(frame, (x2, y2), 4, color, -1)

                draw_gaze_line(iris_left, left_eye_corner, color=(0,255,0))
                draw_gaze_line(iris_right, right_eye_corner, color=(0,255,0))

                # ----- HEAD POSE ESTIMATION using solvePnP -----
                # use set of 2D image points mapped to 3D model points (approximate)
                # Mediapipe landmark indices commonly used:
                # nose tip: 1, left eye outer: 33, right eye outer: 263, mouth left: 61, mouth right: 291, chin: 199
                image_points = np.array([
                    (lm[1].x * w, lm[1].y * h),   # nose tip
                    (lm[33].x * w, lm[33].y * h), # left eye outer
                    (lm[263].x * w, lm[263].y * h), # right eye outer
                    (lm[61].x * w, lm[61].y * h), # mouth left
                    (lm[291].x * w, lm[291].y * h), # mouth right
                    (lm[199].x * w, lm[199].y * h)  # chin
                ], dtype="double")

                # approximate 3D model points of a generic face (in mm)
                model_points = np.array([
                    (0.0, 0.0, 0.0),         # nose tip
                    (-30.0, -125.0, -30.0),  # left eye
                    (30.0, -125.0, -30.0),   # right eye
                    (-60.0, -170.0, -40.0),  # mouth left
                    (60.0, -170.0, -40.0),   # mouth right
                    (0.0, -250.0, -60.0)     # chin
                ])

                # camera internals
                focal_length = w
                center = (w/2, h/2)
                camera_matrix = np.array(
                    [[focal_length, 0, center[0]],
                     [0, focal_length, center[1]],
                     [0, 0, 1]], dtype="double"
                )
                dist_coeffs = np.zeros((4,1))  # assume no lens distortion

                # solvePnP
                success, rotation_vector, translation_vector = cv2.solvePnP(
                    model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
                )

                # project a 3D point (0,0,1000) forward to get a line to draw for head orientation
                (nose_end_point2D, jacobian) = cv2.projectPoints(
                    np.array([(0.0, 0.0, 1000.0)]),
                    rotation_vector, translation_vector, camera_matrix, dist_coeffs
                )
                p1 = (int(image_points[0][0]), int(image_points[0][1]))  # nose tip in image
                p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
                cv2.line(frame, p1, p2, (255, 0, 0), 2)  # blue line = head orientation

                # compute Euler angles
                rot_mat, _ = cv2.Rodrigues(rotation_vector)
                roll, pitch, yaw = rotationMatrixToEulerAngles(rot_mat)  # degrees
                # note: roll=x, pitch=y, yaw=z in our function mapping

                # ----- Combine gaze + head pose into cheat logic -----
                # thresholds (tunable)
                yaw_thresh = 20.0   # degrees
                pitch_thresh = 20.0 # degrees
                gaze_x_thresh = 0.25  # normalized image-plane gaze x magnitude
                gaze_y_thresh = 0.35  # normalized y

                # compute normalized gaze (image-plane) using left eye for example:
                # normalized vector from eye center to iris (range approx -1..1)
                lx, ly = iris_left
                ex, ey = left_eye_corner
                gx = (lx - ex) / float(bw + 1e-6)   # normalized by face box width
                gy = (ly - ey) / float(bh + 1e-6)

                # decide status using combination
                status = "NORMAL"
                color = (0,255,0)

                # if head turned too much OR gaze vector large -> suspicious
                if abs(yaw) > yaw_thresh:
                    status = "HEAD TURN"
                    color = (0,0,255)
                elif abs(pitch) > pitch_thresh:
                    status = "HEAD PITCH"
                    color = (0,0,255)
                elif gx < -gaze_x_thresh:
                    status = "LOOKING LEFT"
                    color = (0,0,255)
                elif gx > gaze_x_thresh:
                    status = "LOOKING RIGHT"
                    color = (0,0,255)
                elif gy > gaze_y_thresh:
                    status = "LOOKING DOWN"
                    color = (0,0,255)

                # ----- panel info above the face box -----
                panel_h = 90
                panel_y0 = max(y - panel_h - 12, 8)
                panel_y1 = panel_y0 + panel_h

                overlay = frame.copy()
                cv2.rectangle(overlay, (x, panel_y0), (x + bw, panel_y1), (12,12,12), -1)
                frame = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)

                # status big
                cv2.putText(frame, f"{status}", (x + 8, panel_y0 + 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
                # pixel matrix
                cv2.putText(frame, f"L: {iris_left}", (x + 8, panel_y0 + 52),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230,230,230), 1, cv2.LINE_AA)
                cv2.putText(frame, f"R: {iris_right}", (x + 8, panel_y0 + 72),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230,230,230), 1, cv2.LINE_AA)
                # head pose numbers
                cv2.putText(frame, f"yaw:{yaw:.1f} pitch:{pitch:.1f} roll:{roll:.1f}",
                            (x + bw - 220, panel_y0 + 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1, cv2.LINE_AA)

                # label person
                cv2.putText(frame, f"PERSON {i+1}", (x, y - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20,200,255), 2, cv2.LINE_AA)

        # FPS counter
        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / (now - last_time)) if last_time != 0 else 0
        last_time = now
        cv2.putText(frame, f"FPS: {int(fps)}", (10, frame.shape[0]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

        cv2.imshow("Multi-Person Gaze + HeadPose PRO", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
