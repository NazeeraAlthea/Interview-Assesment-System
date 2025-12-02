import whisper
import os
import time
import pandas as pd
from pyannote.audio import Pipeline
import torch
import subprocess
from huggingface_hub import snapshot_download
from jiwer import wer

# =============================================================================
#                            KONFIGURASI UTAMA
# =============================================================================

FILE_PATH = "data/wawancara2.mp4"
WAV_PATH = FILE_PATH.replace(".mp4", ".wav")
REF_PATH = FILE_PATH.replace(".mp4", ".txt")

MODEL_SIZE = "base.en"
DEVICE = "cpu"

LOCAL_MODEL_DIR = "models"
WHISPER_DIR = os.path.join(LOCAL_MODEL_DIR, "whisper")
PYANNOTE_DIR = os.path.join(LOCAL_MODEL_DIR, "pyannote")

os.makedirs(WHISPER_DIR, exist_ok=True)
os.makedirs(PYANNOTE_DIR, exist_ok=True)

# =============================================================================
#                        DOWNLOAD MODEL JIKA BELUM ADA
# =============================================================================

def ensure_whisper_model():
    model_path = os.path.join(WHISPER_DIR, MODEL_SIZE)
    if not os.path.exists(model_path):
        print(f"Downloading Whisper model to {WHISPER_DIR} ...")
        whisper.load_model(MODEL_SIZE, download_root=WHISPER_DIR)
    else:
        print("Whisper model already exists. Skipping download.")

def ensure_pyannote_model():
    model_repo = "pyannote/speaker-diarization-3.1"
    print(f"Mengecek model Pyannote...")

    # cache_dir isi model hasil download HF
    if not os.path.exists(os.path.join(PYANNOTE_DIR, "config.yaml")):
        print(f"Downloading Pyannote model into {PYANNOTE_DIR} ...")
        snapshot_download(
            repo_id=model_repo,
            local_dir=PYANNOTE_DIR,
            local_dir_use_symlinks=False
        )
    else:
        print("Pyannote model sudah ada. Skip download.")

# =============================================================================
#                        KONVERSI FILE KE WAV
# =============================================================================

if not os.path.exists(WAV_PATH):
    print("Mengonversi file ke WAV...")
    subprocess.run([
        "ffmpeg", "-i", FILE_PATH, "-ar", "16000", "-ac", "1", WAV_PATH
    ], check=True)

# =============================================================================
#                        FUNGSI FORMAT TIME
# =============================================================================

def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

# =============================================================================
#                        FUNGSI PROSES UTAMA
# =============================================================================

def run_full_process():

    print("--- MEMULAI PROSES ---")

    # Pastikan model tersedia
    ensure_whisper_model()
    ensure_pyannote_model()

    # Load Whisper dari folder lokal
    print("\nMemuat Whisper dari folder lokal...")
    whisper_model = whisper.load_model(MODEL_SIZE, device=DEVICE, download_root=WHISPER_DIR)
    print("Whisper loaded.")

    # Load Pyannote dari folder lokal
    print("\nMemuat Pyannote diarization dari cache lokal...")
    diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        cache_dir=PYANNOTE_DIR,
        use_auth_token=None  # atau token jika model dilindungi
    )
    diarization_pipeline.to(torch.device(DEVICE))
    print("Pyannote loaded.")

    diarization_pipeline.to(torch.device(DEVICE))
    print("Pyannote loaded.")

    # Diarisasi
    print("\nMenjalankan diarization...")
    diarization_result = diarization_pipeline(WAV_PATH)

    # Transkripsi
    print("\nMenjalankan transkripsi Whisper...")
    transcription_result = whisper_model.transcribe(FILE_PATH, language="en", word_timestamps=True)

    # Gabung diarization + transkripsi
    print("\nMenggabungkan hasil")

    speaker_turns = []
    for turn, _, speaker in diarization_result.itertracks(yield_label=True):
        speaker_turns.append({"start": turn.start, "end": turn.end, "speaker": speaker})

    speaker_df = pd.DataFrame(speaker_turns)

    all_words = []
    for segment in transcription_result["segments"]:
        all_words.extend(segment["words"])

    final_transcript = []
    current = None

    for word in all_words:
        word_start = word["start"]
        spk_row = speaker_df[(speaker_df["start"] <= word_start) & (speaker_df["end"] >= word_start)]
        speaker = spk_row.iloc[0]["speaker"] if not spk_row.empty else "UNKNOWN"

        if current is None:
            current = {"start": word_start, "end": word["end"], "speaker": speaker, "text": word["word"]}
        else:
            if speaker == current["speaker"]:
                current["text"] += word["word"]
                current["end"] = word["end"]
            else:
                final_transcript.append(current)
                current = {"start": word_start, "end": word["end"], "speaker": speaker, "text": word["word"]}

    if current:
        final_transcript.append(current)

    # Output hasil
    print("\n==================== HASIL ====================")
    for seg in final_transcript:
        print(f"[{format_time(seg['start'])} --> {format_time(seg['end'])}] {seg['speaker']}: {seg['text']}")
    print("=================================================")

    # ===================== HITUNG WER =====================
    print("\nMenghitung WER...")

    if not os.path.exists(REF_PATH):
        print(f"Gagal: file referensi tidak ditemukan: {REF_PATH}")
    else:
        asr_text = " ".join([seg["text"] for seg in final_transcript])
        ref_text = open(REF_PATH, "r", encoding="utf-8").read().strip()
        error_rate = wer(ref_text, asr_text)

        print("\n=========== HASIL WER ===========")
        print(f"WER: {error_rate * 100:.2f}%")
        print("=================================\n")


if __name__ == "__main__":
    run_full_process()
