# src/asr/utils.py
import os
import subprocess
from datetime import datetime

def format_time(seconds: float) -> str:
    """Konversi detik ke format HH:MM:SS,ms (00:00:00,000)."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def ensure_wav_16k_mono(input_path: str) -> str:
    """
    Pastikan file WAV 16kHz mono ada.
    Jika input_path sudah WAV 16k mono, akan mengembalikan path yang sama.
    Kalau belum ada, akan membuat <inputname>.wav dengan ffmpeg.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File tidak ditemukan: {input_path}")

    base, ext = os.path.splitext(input_path)
    wav_path = base + ".wav"

    # Quick check: if wav exists and likely 16k, return it
    if os.path.exists(wav_path):
        return wav_path

    # Convert with ffmpeg
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-ar", "16000", "-ac", "1", wav_path
    ]
    subprocess.run(cmd, check=True)
    return wav_path

def build_json_filename(original_file):
    """
    Generate dynamic JSON filename based on original media filename + timestamp.
    Example: interview_question_1_2025-12-04_15-30-55.json
    """
    base = os.path.splitext(os.path.basename(original_file))[0]
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"{base}_{timestamp}.json"