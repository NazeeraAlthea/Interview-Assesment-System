# src/asr/asr_whisper.py
from typing import List, Dict, Tuple, Optional
import whisper
import time
import json
import os
from .utils import format_time, ensure_wav_16k_mono
import pandas as pd
import numpy as np

class ASRWhisper:
    def __init__(self, model_size: str = "base.en", device: str = "cpu"):
        """
        model_size: e.g. "tiny", "base", "base.en", "small", "medium", "large"
        device: "cpu" or "cuda"
        """
        self.model_size = model_size
        self.device = device
        self.model = None

    def load(self):
        print(f"Loading Whisper model '{self.model_size}' on {self.device} ...")
        self.model = whisper.load_model(self.model_size, device=self.device)
        print("Whisper loaded.")

    def transcribe(self, audio_path: str, language: Optional[str] = None, word_timestamps: bool = True):
        """
        Run transcription with word timestamps.
        Returns the raw whisper result dict.
        """
        if self.model is None:
            self.load()

        # If input is not wav 16k, convert it but pass original path to whisper if needed.
        wav_path = ensure_wav_16k_mono(audio_path)

        opts = {"word_timestamps": word_timestamps}
        if language is not None:
            opts["language"] = language
            opts["task"] = "transcribe"

        start = time.time()
        res = self.model.transcribe(wav_path, **opts)
        end = time.time()
        print(f"Transcription finished in {end - start:.2f} s")
        return res

    @staticmethod
    def map_words_to_speakers(asr_result: dict, speaker_turns: List[dict]) -> List[dict]:
        """
        asr_result: raw whisper output with 'segments' -> each segment has 'words' with start/end/word
        speaker_turns: list of {'start','end','speaker'}
        Returns a list of words with speaker tag.
        """
        # flatten words
        all_words = []
        for seg in asr_result.get("segments", []):
            for w in seg.get("words", []):
                all_words.append({
                    "start": float(w["start"]),
                    "end": float(w["end"]),
                    "word": w["word"]
                })

        import pandas as pd
        df_turns = pd.DataFrame(speaker_turns)
        mapped = []
        for w in all_words:
            s = w["start"]
            matched = df_turns[(df_turns["start"] <= s) & (df_turns["end"] >= s)]
            if not matched.empty:
                spk = matched.iloc[0]["speaker"]
            else:
                spk = "UNKNOWN"
            mapped.append({
                "start": w["start"],
                "end": w["end"],
                "word": w["word"],
                "speaker": spk
            })
        return mapped

    @staticmethod
    def words_to_sentences(mapped_words: List[dict]) -> List[dict]:
        """
        Group sequential words by speaker into segments with start/end/text.
        """
        final_segments = []
        cur = None
        for w in mapped_words:
            if cur is None:
                cur = {"start": w["start"], "end": w["end"], "speaker": w["speaker"], "text": w["word"]}
            else:
                if w["speaker"] == cur["speaker"]:
                    cur["text"] += " " + w["word"]
                    cur["end"] = w["end"]
                else:
                    final_segments.append(cur)
                    cur = {"start": w["start"], "end": w["end"], "speaker": w["speaker"], "text": w["word"]}
        if cur is not None:
            final_segments.append(cur)
        return final_segments

    @staticmethod
    def build_output_json(file_path: str, model_size: str, device: str, final_segments: List[dict], metrics: dict = None):
        """
        Build final JSON structure (similar to notebook).
        """
        out = {
            "file_path": file_path,
            "model_size": model_size,
            "device": device,
            "segments": [
                {
                    "start_sec": float(seg["start"]),
                    "end_sec": float(seg["end"]),
                    "start_time": format_time(seg["start"]),
                    "end_time": format_time(seg["end"]),
                    "speaker": seg["speaker"],
                    "text": seg["text"]
                } for seg in final_segments
            ],
            "wer_cer_metrics": metrics or {}
        }
        return out

    # --- CONFIDENCE HELPERS ---
    def _confidence_from_logprob(self, avg_logprob: float) -> float:
        """
        Convert avg_logprob (usually negative) to a 0..1 confidence.
        We use a sigmoid-shaped mapping that maps typical whisper logprobs
        (e.g. -0.2 .. -3.0) to sensible confidences.
        """
        if avg_logprob is None:
            return 0.0
        # scale and shift: tune alpha for sensitivity (default 5)
        alpha = 5.0
        # shift to bring typical logprobs to sigmoid domain
        x = avg_logprob + 1.0
        conf = 1.0 / (1.0 + np.exp(-alpha * x))
        return float(np.clip(conf, 0.0, 1.0))

    def _adjust_with_compression(self, conf: float, compression_ratio: float) -> float:
        """
        If compression_ratio indicates hallucination or forced text,
        penalize the confidence.
        """
        if compression_ratio is None:
            return conf
        if compression_ratio > 2.5:
            conf *= 0.25
        elif compression_ratio > 1.8:
            conf *= 0.5
        elif compression_ratio > 1.4:
            conf *= 0.8
        return float(np.clip(conf, 0.0, 1.0))

    def compute_confidences_for_whisper_segments(self, asr_result: dict) -> List[dict]:
        """
        For each whisper segment in asr_result['segments'], compute a confidence value.
        Returns a list of segment dicts with ('start','end','text','confidence', 'avg_logprob', 'compression_ratio').
        """
        segments = []
        for seg in asr_result.get("segments", []):
            avg_logprob = seg.get("avg_logprob", None)
            compression_ratio = seg.get("compression_ratio", None)
            base = self._confidence_from_logprob(avg_logprob)
            final = self._adjust_with_compression(base, compression_ratio)
            segments.append({
                "start": float(seg.get("start", 0.0)),
                "end": float(seg.get("end", 0.0)),
                "text": seg.get("text", ""),
                "avg_logprob": avg_logprob,
                "compression_ratio": compression_ratio,
                "confidence": float(round(final, 4))
            })
        return segments

    def aggregate_confidence_to_final_segments(self, final_segments: List[dict], whisper_segments: List[dict]) -> List[dict]:
        """
        Aggregate per-whisper-segment confidences to the final speaker segments.
        For each final_segment, find overlapping whisper_segments and compute
        a duration-weighted average of confidence.
        """
        def overlap(a_start, a_end, b_start, b_end):
            return max(0.0, min(a_end, b_end) - max(a_start, b_start))

        # make sure whisper_segments sorted by start
        whisper_segments = sorted(whisper_segments, key=lambda x: x["start"])

        enriched = []
        for fs in final_segments:
            s = fs["start"]
            e = fs["end"]
            total_weight = 0.0
            weighted_conf = 0.0

            for ws in whisper_segments:
                ov = overlap(s, e, ws["start"], ws["end"])
                if ov > 0:
                    total_weight += ov
                    weighted_conf += ov * ws.get("confidence", 0.0)

            if total_weight > 0:
                agg_conf = float(weighted_conf / total_weight)
            else:
                # fallback: no overlap (rare) -> set None or 0.0
                agg_conf = 0.0

            fs_copy = fs.copy()
            fs_copy["confidence"] = float(round(agg_conf, 4))
            enriched.append(fs_copy)

        return enriched

    def compute_overall_confidence(self, final_segments_enriched: List[dict]) -> float:
        """
        Compute a single overall confidence score for the whole transcript.
        We weight by duration of each final segment.
        """
        total_dur = 0.0
        weighted_conf = 0.0
        for seg in final_segments_enriched:
            dur = max(0.0, seg["end"] - seg["start"])
            total_dur += dur
            weighted_conf += dur * seg.get("confidence", 0.0)

        if total_dur == 0:
            return 0.0
        return float(round(weighted_conf / total_dur, 4))