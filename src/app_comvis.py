import cv2
import os
import matplotlib.pyplot as plt

from comvis.eye_movement.eye_movement import EyeGazeEstimator
from comvis.head_movement.head_movement import HeadPoseEstimator
from comvis.people_detection.people_detection import PeopleDetector


# ============================================================
# EVENT DETECTION (dipakai di industri proctoring)
# ============================================================

def extract_intervals(data, condition_fn, min_duration=0.5):
    """
    Mengubah per-frame detection menjadi interval cheating.
    """
    intervals = []
    start = None
    end = None

    for row in data:
        if condition_fn(row):
            if start is None:
                start = row["timestamp"]
            end = row["timestamp"]
        else:
            if start is not None:
                # cek durasi minimum event cheating
                if end - start >= min_duration:
                    intervals.append({"start": start, "end": end})
                start = None

    return intervals


# ============================================================
# VISUALISASI TIMELINE CHEATING
# ============================================================

def plot_cheating_timeline(events, video_duration):
    """
    Membuat visual timeline cheat (mirip industry dashboards).
    """
    fig, ax = plt.subplots(figsize=(12, 4))

    colors = {
        "multiple_person": "red",
        "eye_cheating": "orange",
        "head_cheating": "blue"
    }

    y_positions = {
        "multiple_person": 3,
        "eye_cheating": 2,
        "head_cheating": 1
    }

    for event_type, event_list in events.items():
        for ev in event_list:
            ax.plot(
                [ev["start"], ev["end"]],
                [y_positions[event_type]] * 2,
                color=colors[event_type],
                linewidth=10,
                solid_capstyle="butt",
            )

    ax.set_ylim(0.5, 3.5)
    ax.set_xlim(0, video_duration)

    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels(["Head", "Eye", "People"])

    ax.set_xlabel("Time (seconds)")
    ax.set_title("Cheating Event Timeline")

    plt.grid(axis="x", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================
# PIPELINE UTAMA
# ============================================================

def run_full_pipeline(video_path):
    if not os.path.exists(video_path):
        print(f"Video tidak ditemukan: {video_path}")
        return None

    print(f"\n=== Running COMVIS Pipeline ===")
    print(f"Video: {video_path}\n")

    # Load models
    eye_model = EyeGazeEstimator(model_path="src/comvis/eye_movement/unityeyes_eye_model.pkl")
    head_model = HeadPoseEstimator()
    people_model = PeopleDetector(model_path="yolov8n.pt")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_id = 0

    eye_results = []
    head_results = []
    people_results = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        timestamp = frame_id / fps

        # Eye
        eye_out = eye_model.predict_from_frame(frame)
        eye_results.append({
            "frame": frame_id,
            "timestamp": timestamp,
            "gaze": eye_out["gaze_label"],
            "gx": eye_out["gx"],
            "gy": eye_out["gy"],
            "status": eye_out["status"],
        })

        # Head
        yaw, pitch, roll, label = head_model.estimate_from_frame(frame)
        head_results.append({
            "frame": frame_id,
            "timestamp": timestamp,
            "yaw": yaw,
            "pitch": pitch,
            "roll": roll,
            "label": label,
        })

        # People
        pcount = people_model.count_persons(frame)
        people_results.append({
            "frame": frame_id,
            "timestamp": timestamp,
            "person_count": pcount,
        })

    cap.release()
    print("=== Pipeline Selesai ===\n")

    return eye_results, head_results, people_results, frame_id / fps



# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    VIDEO = "data/interview_question_2.webm"   # sesuaikan path

    # 1) Jalankan pipeline
    outputs = run_full_pipeline(VIDEO)

    if outputs is None:
        exit()

    eye, head, people, duration = outputs

    # 2) EVENT DETECTION
    multi_person = extract_intervals(
        people,
        condition_fn=lambda r: r["person_count"] > 1,
        min_duration=0.2
    )

    eye_cheat = extract_intervals(
        eye,
        condition_fn=lambda r: r["gaze"] != "center" and r["status"] == "OK",
        min_duration=1.0
    )

    head_cheat = extract_intervals(
        head,
        condition_fn=lambda r: abs(r["yaw"]) > 25 or abs(r["pitch"]) > 20,
        min_duration=1.0
    )

    events = {
        "multiple_person": multi_person,
        "eye_cheating": eye_cheat,
        "head_cheating": head_cheat
    }

    print("\n=== Detected Events ===")
    print(events)

    # 3) VISUALISASI
    print("\n=== Generating Cheating Timeline ===")
    plot_cheating_timeline(events, video_duration=duration)
