import os
os.environ["HF_DATASETS_DISABLE_TORCHCODEC"] = "1"
os.environ["HF_DATASETS_AUDIO_ALLOW_UNSUPPORTED"] = "1"

import re
from pathlib import Path
import torch
import whisper
import pandas as pd
from jiwer import wer
from datasets import load_dataset, Audio
Audio.decode_example = lambda self, x: x
from pyannote.audio import Pipeline
from huggingface_hub import snapshot_download


# =============================================================================
#                             KONFIGURASI
# =============================================================================

DATASET_NAME = "rakshya34/filtered_english_female_voice_v1"
DATASET_SPLIT = "train"

MODEL_SIZE = "base.en"
DEVICE = "cpu"

LOCAL_MODEL_DIR = "models"
WHISPER_DIR = os.path.join(LOCAL_MODEL_DIR, "whisper")
PYANNOTE_DIR = os.path.join(LOCAL_MODEL_DIR, "pyannote")

os.makedirs(WHISPER_DIR, exist_ok=True)
os.makedirs(PYANNOTE_DIR, exist_ok=True)


# =============================================================================
#                            NORMALIZER
# =============================================================================

def normalize_text(s: str):
    s = s.lower()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# =============================================================================
#                            DOWNLOAD MODEL
# =============================================================================

def ensure_whisper_model():
    model_path = os.path.join(WHISPER_DIR, MODEL_SIZE)
    if not os.path.exists(model_path):
        print(f"Downloading Whisper model to {WHISPER_DIR} ...")
        whisper.load_model(MODEL_SIZE, download_root=WHISPER_DIR)
    else:
        print("Whisper model already exists. Skip.")


def ensure_pyannote_model():
    print("Checking Pyannote model...")
    config_path = os.path.join(PYANNOTE_DIR, "config.yaml")

    if not os.path.exists(config_path):
        print("Downloading Pyannote model...")
        snapshot_download(
            repo_id="pyannote/speaker-diarization-3.1",
            local_dir=PYANNOTE_DIR,
            local_dir_use_symlinks=False
        )
    else:
        print("Pyannote model exists. Skip.")


# =============================================================================
#                             LOAD DATASET
# =============================================================================

def load_audio_dataset():
    print(f"Loading dataset: {DATASET_NAME}")
    ds = load_dataset(DATASET_NAME, split=DATASET_SPLIT)

    def extract_path(example):
        # gunakan path asli dari huggingface cache
        example["audio_path"] = example["audio"]["path"]
        return example

    ds = ds.map(extract_path)

    return ds


# =============================================================================
#                           PROSES UTAMA (ASR + DIARIZATION + WER)
# =============================================================================

def process_one_audio(item, whisper_model, diarization_pipeline):
    file_path = item["audio_path"]       # path asli HF
    ref_text = item["text"]

    # 1. Transkripsi (Whisper)
    trans_result = whisper_model.transcribe(
        file_path,
        language="en",
        word_timestamps=True
    )

    # 2. Diarization (Pyannote)
    diar_result = diarization_pipeline(file_path)

    speaker_turns = []
    for turn, _, speaker in diar_result.itertracks(yield_label=True):
        speaker_turns.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker
        })

    speaker_df = pd.DataFrame(speaker_turns)

    # 3. Ambil kata per kata
    words = []
    for seg in trans_result["segments"]:
        words.extend(seg["words"])

    # 4. Gabungkan berdasarkan speaker
    final_transcript = []
    current = None

    for w in words:
        start = w["start"]
        end = w["end"]
        word = w["word"]

        # cari speaker berdasarkan timestamp
        match = speaker_df[
            (speaker_df["start"] <= start) &
            (speaker_df["end"] >= start)
        ]

        speaker = match.iloc[0]["speaker"] if not match.empty else "UNKNOWN"

        if current is None:
            current = {
                "speaker": speaker,
                "start": start,
                "end": end,
                "text": word
            }
        else:
            if speaker == current["speaker"]:
                current["text"] += word
                current["end"] = end
            else:
                final_transcript.append(current)
                current = {
                    "speaker": speaker,
                    "start": start,
                    "end": end,
                    "text": word
                }

    if current:
        final_transcript.append(current)

    # join teks
    hyp_text = " ".join([seg["text"] for seg in final_transcript])

    # WER
    ref_n = normalize_text(ref_text)
    hyp_n = normalize_text(hyp_text)
    error = wer(ref_n, hyp_n)

    return {
        "ref": ref_text,
        "hyp": hyp_text,
        "wer": error,
        "segments": final_transcript
    }


# =============================================================================
#                                  MAIN
# =============================================================================

def run_full_process():
    print("\n=== MEMULAI PROSES ===")

    ensure_whisper_model()
    ensure_pyannote_model()

    whisper_model = whisper.load_model(
        MODEL_SIZE,
        device=DEVICE,
        download_root=WHISPER_DIR
    )

    diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        cache_dir=PYANNOTE_DIR
    )
    diarization_pipeline.to(torch.device(DEVICE))

    ds = load_audio_dataset()

    results = []
    total_wer = 0

    for i, item in enumerate(ds, 1):
        print(f"\n[{i}/{len(ds)}] Memproses audio...")

        res = process_one_audio(item, whisper_model, diarization_pipeline)

        results.append(res)
        total_wer += res["wer"]

        print(f"WER: {res['wer'] * 100:.2f}%")

    avg_wer = total_wer / len(ds)

    print("\n==================== HASIL AKHIR ====================")
    print(f"AVERAGE WER: {avg_wer * 100:.2f}%")
    print("=====================================================")

    return results


if __name__ == "__main__":
    run_full_process()
