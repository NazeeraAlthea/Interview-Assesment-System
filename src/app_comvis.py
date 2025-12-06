# src/app_comvis.py
import os
import time
import json
import csv
from datetime import datetime, timedelta
from typing import List, Dict, Callable

import cv2
import numpy as np

# imports modul inference (pastikan path package benar)
from comvis.eye_movement.eye_movement import EyeGazeEstimator
from comvis.eye_movement.gaze_estimator import GazeEstimator
from comvis.head_movement.head_movement import HeadPoseEstimator
from comvis.people_detection.people_detection import PeopleDetector


# -----------------------
# CONFIG
# -----------------------
VIDEO_PATH = "data/interview_question_2.webm"

# model paths (sesuaikan jika letak model berbeda)
EYE_MODEL_PATH = "src/comvis/eye_movement/unityeyes_eye_model.pkl"
YOLO_MODEL_PATH = "yolov8n.pt"
OPENVINO_XML = None  # set path ke xml jika pakai OpenVINO fallback

# thresholds (tweakable)
EYE_CHEAT_MIN_DURATION = 1.0   # seconds
HEAD_CHEAT_MIN_DURATION = 1.0  # seconds
PEOPLE_MIN_DURATION = 0.3      # seconds
FALLBACK_MIN_DURATION = 0.5    # seconds

HEAD_YAW_THRESHOLD = 25.0     # degrees
HEAD_PITCH_THRESHOLD = 20.0   # degrees

# output directories
RESULT_DIR = "result/comvis_output"
RAW_DIR = os.path.join(RESULT_DIR, "raw")
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)


# -----------------------
# Helpers
# -----------------------
def format_hhmmss(seconds: float) -> str:
    """Return HH:MM:SS.mmm formatted string for seconds offset."""
    td = timedelta(seconds=float(seconds))
    # Extract hours:minutes:seconds.milliseconds
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = (total_seconds % 60)
    ms = int((td.total_seconds() - total_seconds) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"


def now_ts() -> float:
    return time.time()


def csv_dump(path: str, fieldnames: List[str], rows: List[Dict]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


# -----------------------
# Event extraction utilities
# -----------------------
def extract_intervals_from_boolean_series(series: List[Dict], cond_fn: Callable[[Dict], bool], min_duration: float) -> List[Dict]:
    """
    Given list of per-frame dicts including 'timestamp', return aggregated intervals
    where cond_fn(frame) is True for at least min_duration seconds.
    Returns list of dicts: {start: , end: } (seconds offset).
    """
    intervals = []
    start = None
    end = None

    for row in series:
        if cond_fn(row):
            if start is None:
                start = row["timestamp"]
            end = row["timestamp"]
        else:
            if start is not None:
                # include the end timestamp of the last positive frame
                if end - start >= min_duration:
                    intervals.append({"start": start, "end": end})
                start = None
                end = None

    # close final open interval
    if start is not None and end is not None and (end - start >= min_duration):
        intervals.append({"start": start, "end": end})

    return intervals


def build_segments_from_events(events: Dict[str, List[Dict]], session_start_unix: float) -> List[Dict]:
    """
    Convert named event intervals to segment rows similar to example.
    Each segment will include absolute start_ts/end_ts (unix float), durations, and hhmmss strings.
    Assign track_id = 1 (single candidate) by default.
    """
    segments = []
    track_id = 1
    for reason, intervals in events.items():
        for iv in intervals:
            start_offset = float(iv["start"])
            end_offset = float(iv["end"])
            duration = end_offset - start_offset
            start_ts = session_start_unix + start_offset
            end_ts = session_start_unix + end_offset
            segments.append({
                "track_id": track_id,
                "reason": reason,
                "start_ts": float(start_ts),
                "end_ts": float(end_ts),
                "duration_sec": float(duration),
                "start_hhmmss": format_hhmmss(start_offset),
                "end_hhmmss": format_hhmmss(end_offset)
            })
    # sort by start_ts
    segments = sorted(segments, key=lambda x: x["start_ts"])
    return segments


# -----------------------
# Main pipeline
# -----------------------
def run_full_pipeline(video_path: str):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    print("Loading models...")
    eye_model = EyeGazeEstimator(model_path=EYE_MODEL_PATH)
    gaze_fallback = GazeEstimator(openvino_model_xml=OPENVINO_XML, iris_threshold=0.45)
    head_model = HeadPoseEstimator()
    people_model = PeopleDetector(model_path=YOLO_MODEL_PATH)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_id = 0

    # per-frame results
    eye_frames = []
    fallback_frames = []
    head_frames = []
    people_frames = []

    print("Starting inference loop...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        timestamp = frame_id / fps  # seconds offset from video start

        # Eye (iris MLP)
        eye_out = eye_model.predict_from_frame(frame)
        eye_row = {
            "frame": frame_id,
            "timestamp": timestamp,
            "gaze": eye_out.get("gaze_label", "UNKNOWN"),
            "gx": eye_out.get("gx"),
            "gy": eye_out.get("gy"),
            "status": eye_out.get("status", "N/A"),
            "iris_quality": eye_out.get("iris_quality", None)
        }
        eye_frames.append(eye_row)

        # Gaze fallback
        fb = gaze_fallback.estimate_from_frame(frame)
        fb_row = {
            "frame": frame_id,
            "timestamp": timestamp,
            "source": fb.get("source", None),
            "gaze_label": fb.get("gaze_label", None),
            "confidence": fb.get("confidence", 0.0),
            "yaw": fb.get("yaw"),
            "pitch": fb.get("pitch"),
            "gx": fb.get("gx"),
            "gy": fb.get("gy")
        }
        fallback_frames.append(fb_row)

        # Head pose
        yaw, pitch, roll, label = head_model.estimate_from_frame(frame)
        head_row = {
            "frame": frame_id,
            "timestamp": timestamp,
            "yaw": yaw,
            "pitch": pitch,
            "roll": roll,
            "label": label
        }
        head_frames.append(head_row)

        # People detection
        person_count = people_model.count_persons(frame)
        people_row = {
            "frame": frame_id,
            "timestamp": timestamp,
            "person_count": int(person_count)
        }
        people_frames.append(people_row)

        # optional progress
        if frame_id % int(fps * 5) == 0:
            print(f"Processed {frame_id} frames (~{timestamp:.1f}s)")

    cap.release()
    print("Inference finished.")

    return eye_frames, fallback_frames, head_frames, people_frames, fps, frame_id


# -----------------------
# Orchestration: postprocessing + export
# -----------------------
def postprocess_and_export(eye_frames, fallback_frames, head_frames, people_frames, fps, total_frames):
    # create session timestamps
    session_start_unix = now_ts()
    session_start_readable = datetime.fromtimestamp(session_start_unix).strftime("%Y-%m-%d %H:%M:%S")

    print("Building events...")

    # event definitions
    multi_person_intervals = extract_intervals_from_boolean_series(
        people_frames,
        cond_fn=lambda r: r["person_count"] > 1,
        min_duration=PEOPLE_MIN_DURATION
    )

    eye_cheat_intervals = extract_intervals_from_boolean_series(
        eye_frames,
        cond_fn=lambda r: (r["gaze"].lower() != "center" and r.get("status") in ("OK", "ok", None)),
        min_duration=EYE_CHEAT_MIN_DURATION
    )

    head_cheat_intervals = extract_intervals_from_boolean_series(
        head_frames,
        cond_fn=lambda r: (abs(r.get("yaw", 0.0)) > HEAD_YAW_THRESHOLD) or (abs(r.get("pitch", 0.0)) > HEAD_PITCH_THRESHOLD),
        min_duration=HEAD_CHEAT_MIN_DURATION
    )

    fallback_intervals = extract_intervals_from_boolean_series(
        fallback_frames,
        cond_fn=lambda r: (r.get("source") == "model" and float(r.get("confidence", 0.0)) > 0.5),
        min_duration=FALLBACK_MIN_DURATION
    )

    # map to reason strings (match example)
    events = {
        "HEAD_POSE_OFF": head_cheat_intervals,
        "EYES_MOVING": eye_cheat_intervals,
        "EYES_OFF": [],  # example separates moving vs off; we can infer eyes off if status not OK
        "MULTI_PERSON": multi_person_intervals,
        "FALLBACK_GAZE": fallback_intervals
    }

    # detect EYES_OFF small events where status != OK for some duration
    eyes_off_intervals = extract_intervals_from_boolean_series(
        eye_frames,
        cond_fn=lambda r: (r.get("status") not in ("OK", "ok", None)),
        min_duration=0.1
    )
    events["EYES_OFF"] = eyes_off_intervals

    # build segments list (convert offsets -> absolute unix timestamps)
    segments = build_segments_from_events(events, session_start_unix)

    # build output structure
    output = {
        "session_start_unix": session_start_unix,
        "session_start_readable": session_start_readable,
        "video_path": VIDEO_PATH,
        "fps": fps,
        "total_frames": total_frames,
        "segments": segments
    }

    # filenames
    ts_str = datetime.fromtimestamp(session_start_unix).strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(RESULT_DIR, f"{ts_str}_segments.json")
    csv_path = os.path.join(RESULT_DIR, f"{ts_str}_segments.csv")
    raw_csv_path = os.path.join(RAW_DIR, f"{ts_str}_raw_eye_frames.csv")

    print(f"Saving JSON -> {json_path}")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # create CSV rows that match example (flatten)
    csv_fieldnames = ["track_id", "reason", "start_ts", "end_ts", "duration_sec", "start_hhmmss", "end_hhmmss"]
    csv_rows = []
    for s in segments:
        csv_rows.append({
            "track_id": s["track_id"],
            "reason": s["reason"],
            "start_ts": s["start_ts"],
            "end_ts": s["end_ts"],
            "duration_sec": s["duration_sec"],
            "start_hhmmss": s["start_hhmmss"],
            "end_hhmmss": s["end_hhmmss"]
        })
    csv_dump(csv_path, csv_fieldnames, csv_rows)

    # save raw eye frames for debugging (optional)
    raw_fieldnames = ["frame", "timestamp", "gaze", "gx", "gy", "status", "iris_quality"]
    csv_dump(raw_csv_path, raw_fieldnames, eye_frames)

    print("Export done.")
    return output, json_path, csv_path, raw_csv_path


# -----------------------
# CLI
# -----------------------
def main():
    print("COMVIS - production runner")
    print("Video:", VIDEO_PATH)
    start = time.time()
    eye_frames, fallback_frames, head_frames, people_frames, fps, total_frames = run_full_pipeline(VIDEO_PATH)
    output, jpath, cpath, rawpath = postprocess_and_export(eye_frames, fallback_frames, head_frames, people_frames, fps, total_frames)
    end = time.time()
    print(f"All done in {end - start:.1f}s")
    print("JSON saved at:", jpath)
    print("CSV saved at:", cpath)
    print("Raw frames saved at:", rawpath)


if __name__ == "__main__":
    main()
