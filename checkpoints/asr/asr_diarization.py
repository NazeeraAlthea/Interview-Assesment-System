import whisper
import os
import time
import pandas as pd
from pyannote.audio import Pipeline
import torch
from huggingface_hub import HfFolder
import subprocess
from jiwer import wer

# =============================================================================
#                           KONFIGURASI UTAMA
# =============================================================================

FILE_PATH = "data/wawancara2.mp4"
WAV_PATH = FILE_PATH.replace(".mp4", ".wav")
MODEL_SIZE = "base.en"
DEVICE = "cpu"

# =============================================================================
#                           LOGIKA SCRIPT
# =============================================================================

# Ubah file ke .wav pakai ffmpeg
if not os.path.exists(WAV_PATH):
    print("Mengonversi file ke format WAV...")
    subprocess.run([
        "ffmpeg", "-i", FILE_PATH, "-ar", "16000", "-ac", "1", WAV_PATH
    ], check=True)

def format_time(seconds):
    """Konversi detik ke format H:M:S,ms"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def run_full_process():
    """
    Fungsi utama untuk menjalankan proses transkripsi dan diarisasi.
    """
    print("--- Memulai Proses Transkripsi & Diarisasi ---")
    
    # 1. Validasi File
    print(f"Mengecek file di: {FILE_PATH}")
    if not os.path.exists(FILE_PATH):
        print("\n" + "!"*50 + f"\n  ERROR: FILE '{FILE_PATH}' TIDAK DITEMUKAN!\n" + "!"*50 + "\n")
        return

    print("File ditemukan. Lanjut...")

    # 2. Muat Model Diarisasi (pyannote.audio)
    try:
        print(f"\nMemuat pipeline diarisasi 'pyannote/speaker-diarization-3.1'...")
        diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
        )
        diarization_pipeline.to(torch.device(DEVICE))
        print("Pipeline diarisasi berhasil dimuat.")
    except Exception as e:
        print(f"\n!!! ERROR saat memuat pipeline diarisasi: {e}")
        print("Pastikan Anda sudah menyetujui syarat model di Hugging Face dan token Anda benar.")
        return
        
    # 3. Muat Model Transkripsi (Whisper)
    try:
        print(f"\nMemuat model Whisper '{MODEL_SIZE}' ke '{DEVICE}'...")
        whisper_model = whisper.load_model(MODEL_SIZE, device=DEVICE)
        print("Model Whisper berhasil dimuat.")
    except Exception as e:
        print(f"\n!!! ERROR saat memuat model Whisper: {e}")
        return

    # 4. Proses Diarisasi
    print("\nMemulai diarisasi (mendeteksi siapa bicara kapan)...")
    start_diarize_time = time.time()
    diarization_result = diarization_pipeline(WAV_PATH)
    end_diarize_time = time.time()
    print(f"Diarisasi selesai dalam {end_diarize_time - start_diarize_time:.2f} detik.")

    # 5. Proses Transkripsi dengan Word Timestamps
    print("\nMemulai transkripsi (mengubah audio ke teks)...")
    start_transcribe_time = time.time()
    # word_timestamps=True adalah kunci untuk menggabungkan hasilnya
    transcription_result = whisper_model.transcribe(FILE_PATH, language="en", word_timestamps=True)
    end_transcribe_time = time.time()
    print(f"Transkripsi selesai dalam {end_transcribe_time - start_transcribe_time:.2f} detik.")
    
    # 6. Menggabungkan Hasil Diarisasi dan Transkripsi
    print("\nMenggabungkan hasil transkripsi dan diarisasi...")
    
    speaker_turns = []
    for turn, _, speaker in diarization_result.itertracks(yield_label=True):
        speaker_turns.append({'start': turn.start, 'end': turn.end, 'speaker': speaker})
    
    # Buat DataFrame untuk memudahkan pencarian
    speaker_df = pd.DataFrame(speaker_turns)

    # Dapatkan semua kata dari hasil Whisper
    all_words = []
    for segment in transcription_result['segments']:
        all_words.extend(segment['words'])

    # Assign speaker ke setiap kata
    word_speaker_mapping = []
    for word in all_words:
        word_start = word['start']
        # Cari giliran pembicara yang paling cocok untuk kata ini
        speaker = speaker_df[(speaker_df['start'] <= word_start) & (speaker_df['end'] >= word_start)]
        if not speaker.empty:
            word['speaker'] = speaker.iloc[0]['speaker']
        else:
            word['speaker'] = "UNKNOWN" # Jika tidak ada yang cocok
        word_speaker_mapping.append(word)

    # Gabungkan kata-kata menjadi kalimat berdasarkan pembicara
    final_transcript = []
    current_segment = None

    for word in word_speaker_mapping:
        if current_segment is None:
            # Mulai segmen baru
            current_segment = {
                'start': word['start'],
                'end': word['end'],
                'speaker': word['speaker'],
                'text': word['word']
            }
        else:
            # Jika pembicara sama, lanjutkan segmen
            if word['speaker'] == current_segment['speaker']:
                current_segment['text'] += word['word']
                current_segment['end'] = word['end']
            else:
                # Jika pembicara beda, simpan segmen lama dan mulai yang baru
                final_transcript.append(current_segment)
                current_segment = {
                    'start': word['start'],
                    'end': word['end'],
                    'speaker': word['speaker'],
                    'text': word['word']
                }

    # Jangan lupa simpan segmen terakhir
    if current_segment is not None:
        final_transcript.append(current_segment)

    # 7. Tampilkan Hasil Akhir
    print("\n" + "="*80)
    print("                      HASIL TRANSKRIPSI DENGAN DIARISASI")
    print("="*80 + "\n")
    
    for segment in final_transcript:
        start_time = format_time(segment['start'])
        end_time = format_time(segment['end'])
        speaker = segment['speaker']
        text = segment['text'].strip()
        print(f"[{start_time} --> {end_time}] {speaker}: {text}")

    print("\n" + "="*80)
    print("--- Proses Selesai ---")
    

if __name__ == "__main__":
    run_full_process()