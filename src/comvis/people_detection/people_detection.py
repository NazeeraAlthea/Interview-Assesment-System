from ultralytics import YOLO
import cv2

class PeopleDetector:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)

    def count_persons(self, frame):
        results = self.model(frame)[0]
        person_count = sum(1 for box in results.boxes if int(box.cls[0]) == 0)
        return person_count

    def detect_in_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        frame_id = 0

        summary = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_id += 1

            count = self.count_persons(frame)
            timestamp = frame_id / frame_rate

            summary.append({
                "frame": frame_id,
                "timestamp_sec": timestamp,
                "person_count": count
            })

        cap.release()
        return summary
