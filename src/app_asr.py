# src/app_asr.py
import os
import json
from asr.asr_whisper import ASRWhisper
from asr.diarization import Diarizer
from asr.utils import ensure_wav_16k_mono, build_json_filename

# Konfigurasi
DEVICE = "cpu"
MODEL_SIZE = "base.en"
INPUT_VIDEO_OR_AUDIO = "data/interview_question_1.webm"
output_filename = build_json_filename(INPUT_VIDEO_OR_AUDIO)
OUTPUT_JSON = os.path.join("outputs", output_filename)


def main():
    os.makedirs("outputs", exist_ok=True)

    # -------------------------------------
    # 1) Convert to WAV
    # -------------------------------------
    wav_path = ensure_wav_16k_mono(INPUT_VIDEO_OR_AUDIO)

    # -------------------------------------
    # 2) DIARIZATION
    # -------------------------------------
    diar = Diarizer(device=DEVICE)
    diar.load()

    speaker_turns, df_turns = diar.diarize(wav_path)
    print(f"Found {len(speaker_turns)} speaker turns.")

    # -------------------------------------
    # 3) ASR
    # -------------------------------------
    asr = ASRWhisper(model_size=MODEL_SIZE, device=DEVICE)
    asr.load()

    asr_res = asr.transcribe(wav_path, language="en", word_timestamps=True)

    # -------------------------------------
    # 4) Map Words → Speakers
    # -------------------------------------
    mapped_words = asr.map_words_to_speakers(asr_res, speaker_turns)

    # -------------------------------------
    # 5) Build final segments
    # -------------------------------------
    final_segments = asr.words_to_sentences(mapped_words)
    print(f"Built {len(final_segments)} final segments.")

    # -------------------------------------
    # 6) Compute Whisper confidence per segment
    # -------------------------------------
    whisper_seg_conf = asr.compute_confidences_for_whisper_segments(asr_res)

    # -------------------------------------
    # 7) Aggregate confidence into final_segments
    # -------------------------------------
    final_segments_enriched = asr.aggregate_confidence_to_final_segments(
        final_segments, whisper_seg_conf
    )

    # -------------------------------------
    # 8) Overall transcript confidence
    # -------------------------------------
    overall_confidence = asr.compute_overall_confidence(final_segments_enriched)
    print(f"Overall ASR confidence: {overall_confidence:.3f}")

    metrics = {"overall_confidence": overall_confidence}

    # -------------------------------------
    # 9) Build final JSON output
    # -------------------------------------
    out = asr.build_output_json(
        INPUT_VIDEO_OR_AUDIO,
        MODEL_SIZE,
        DEVICE,
        final_segments_enriched,
        metrics
    )

    # -------------------------------------
    # 10) Save JSON
    # -------------------------------------
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"Saved output to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
