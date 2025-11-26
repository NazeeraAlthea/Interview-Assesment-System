# ASR & Speaker Diarization System

A robust end‑to‑end system for **Automatic Speech Recognition (ASR)**, **Speaker Diarization**, and **ASR Quality Evaluation** built using **OpenAI Whisper** and **PyAnnote 3.x**. This project is designed with industry‑grade practices, allowing scalable experimentation, reproducible evaluation, and integration with large speech datasets such as **OpenSLR** or **LibriSpeech**.

---

## 🚀 Overview
Modern speech intelligence systems typically involve two major components:

1. **Automatic Speech Recognition (ASR)** — converting speech into text
2. **Speaker Diarization** — determining *who* spoke *when*

This repository provides:
- A full diarization + transcription pipeline
- Accurate WER/CER evaluation for ASR models
- Local dataset evaluator compatible with LibriSpeech/OpenSLR structure
- Text normalization aligned with industry standards for fair WER scoring
- Modular and maintainable code for further experimentation

---

## ✨ Key Features
### 🔊 Whisper‑based ASR
- Supports Whisper models (tiny/base/small/medium)
- FP32 inference optimized for CPU environments
- Accepts raw waveform arrays (no FFmpeg dependency)

### 🗣️ PyAnnote 3.x Speaker Diarization
- Based on `pyannote/speaker-diarization-3.1`
- Requires HuggingFace authentication
- Outputs diarized speech segments with timestamps

### 📊 ASR Evaluation (WER/CER)
- Implements **text normalization** to avoid unfair mismatch
- Computes **Word Error Rate** and **Character Error Rate**
- Can evaluate per‑sample or full dataset
- Includes optional progress tracking (via `tqdm`)

### 📂 Local Dataset Support
Compatible with the LibriSpeech/OpenSLR folder structure:
```
root/
   speaker-id/
       chapter-id/
           speaker-chapter.trans.txt
           speaker-chapter-0000.flac
           speaker-chapter-0001.flac
```

---

## 📁 Project Structure
```
project/
│
├── asr_diarization_hg.ipynb            # Main diarization + transcription pipeline + evaluation with hugging face datasets
├── asr_diarization_openlsr.ipynb       # WER/CER evaluation utilities
├── requirements.txt                    # Environment dependencies
├── README.md                           # Project documentation
│
└── data/
     └── openslr/                # Local evaluation dataset
```

---

## 🔧 Installation
### 1. Clone Repository
```bash
git clone <repository-url>
cd project
```

### 2. Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Authenticate HuggingFace (Required for PyAnnote)
```bash
huggingface-cli login
```
Ensure you have manually accepted access for:
https://huggingface.co/pyannote/speaker-diarization-3.1

---

## ▶️ Running the Pipeline
### Transcription + Diarization

This will:
- Load diarization pipeline
- Load Whisper model
- Process the audio file
- Combine diarization and ASR output
- Display final transcript with speaker labels

---

## 📊 Evaluating ASR Model Performance
### Example Output
```
=========================
         FINAL METRICS
=========================
WER: 0.0521  (5.21%)
CER: 0.0214  (2.14%)
Total samples: 2620
```

---

## 🧠 Text Normalization (Critical for Fair WER)
Raw Whisper output includes punctuation and casing, while LibriSpeech/OpenSLR transcripts are uppercase without punctuation. Without normalization, WER will be misleadingly high.

Normalization includes:
- Lowercasing
- Removing punctuation
- Collapsing whitespace

This mirrors industry practice used in academic benchmarks and commercial ASR systems.

---

### 🎉 Thank you for exploring this project!
This system is optimized for ASR research, experimentation, and academic capstone use. Feel free to modify, extend, or integrate it into your own ML workflow.

