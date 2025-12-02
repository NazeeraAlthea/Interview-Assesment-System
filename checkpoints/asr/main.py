import whisper
import os
import time

# =============================================================================
#                             KONFIGURASI UTAMA
# =============================================================================

# GANTI INI!
# Ini adalah path LOKAL di komputer Anda setelah Anda download dari Drive.
# Contoh: "D:/Data_Project/interview_kandidat_01.webm"
#          "C:/Users/Anda/Downloads/my_video.webm"
#          "/home/user/project/data/test.mp4"
FILE_PATH = "data/interview_question_5.webm"

# Ukuran model. ".en" berarti khusus Bahasa Inggris.
# Pilihan: "tiny.en", "base.en", "small.en", "medium.en"
# "base.en" adalah awal yang baik (cepat dan akurat).
MODEL_SIZE = "base.en"

# "cpu" jika Anda tidak punya GPU Nvidia.
# "cuda" jika Anda punya GPU Nvidia (jauh lebih cepat).
DEVICE = "cpu"

# =============================================================================
#                             LOGIKA SCRIPT
# =============================================================================

def run_transcription():
    """
    Fungsi utama untuk menjalankan proses transkripsi.
    """
    
    print(f"--- Memulai Tes Transkripsi ---")
    
    # 1. Validasi File: Cek apakah filenya ada
    print(f"Mengecek file di: {FILE_PATH}")
    if not os.path.exists(FILE_PATH):
        print("\n" + "!"*50)
        print(f"  ERROR: FILE TIDAK DITEMUKAN!")
        print(f"  Path yang Anda masukkan: {FILE_PATH}")
        print(f"  Pastikan Anda sudah mengganti 'FILE_PATH' di dalam kode ini.")
        print("!"*50 + "\n")
        return # Menghentikan script

    print("File ditemukan. Lanjut...")

    # 2. Muat Model: Mengunduh model (jika perlu) dan memuatnya ke memori
    try:
        print(f"\nMemuat model Whisper '{MODEL_SIZE}' ke '{DEVICE}'...")
        print("(Catatan: Jika ini pertama kalinya, model akan diunduh. Mohon tunggu.)")
        
        start_load_time = time.time()
        model = whisper.load_model(MODEL_SIZE, device=DEVICE)
        end_load_time = time.time()
        
        print(f"Model berhasil dimuat dalam {end_load_time - start_load_time:.2f} detik.")
    
    except Exception as e:
        print(f"\n!!! ERROR saat memuat model: {e}")
        print("Pastikan Anda punya koneksi internet dan 'DEVICE' disetel dengan benar.")
        return

    # 3. Proses Transkripsi: Ini adalah inti kerjanya
    print(f"\nMemulai transkripsi untuk '{os.path.basename(FILE_PATH)}'...")
    print("Ini bisa memakan waktu beberapa saat tergantung durasi video...")
    
    start_transcribe_time = time.time()
    
    # Tentukan fp16 (Floating Point 16)
    # Ini membuat proses jauh lebih cepat di GPU, tapi tidak didukung di CPU
    use_fp16 = (DEVICE == "cuda")
    
    try:
        # Kita paksa 'language="en"' untuk hasil terbaik Bahasa Inggris
        result = model.transcribe(FILE_PATH, language="en", fp16=use_fp16)
        
        end_transcribe_time = time.time()
        print(f"Transkripsi selesai dalam {end_transcribe_time - start_transcribe_time:.2f} detik.")
    
    except Exception as e:
        print(f"\n!!! ERROR saat transkripsi: {e}")
        # Jika error di sini, seringkali karena ffmpeg belum terinstal
        print("Pastikan 'ffmpeg' sudah terinstal di sistem Anda.")
        return

    # 4. Tampilkan Hasil: Menampilkan apa yang ditemukan Whisper
    print("\n" + "="*50)
    print("                HASIL TRANSKRIPSI")
    print("="*50 + "\n")
    
    # 'result["text"]' adalah transkrip penuh
    print(result["text"])
    
    print("\n" + "="*50)
    print("--- Tes Transkripsi Selesai ---")


# Ini adalah 'entry point' standar Python
# Saat Anda menjalankan 'python tes_whisper.py', kode di bawah ini akan dieksekusi
if __name__ == "__main__":
    run_transcription()