# src/asr/diarization.py
from typing import List, Dict
import torch
import pandas as pd
from pyannote.audio import Pipeline

class Diarizer:
    def __init__(self, model_name: str = "pyannote/speaker-diarization-3.1", device: str = "cpu"):
        """
        Requires that you have huggingface credentials set up (huggingface-cli login)
        or that the model is accessible publicly.
        """
        self.model_name = model_name
        self.device = device
        self.pipeline = None

    def load(self):
        print(f"Loading diarization pipeline '{self.model_name}' to device {self.device} ...")
        self.pipeline = Pipeline.from_pretrained(self.model_name)
        self.pipeline.to(torch.device(self.device))
        print("Diarization pipeline loaded.")

    def diarize(self, wav_path: str):
        """
        Run diarization pipeline on wav_path.
        Returns a list of dicts: [{ 'start': float, 'end': float, 'speaker': 'SPEAKER_00' }, ...]
        """
        if self.pipeline is None:
            self.load()

        diarization_result = self.pipeline(wav_path)
        turns = []
        for turn, track, speaker in diarization_result.itertracks(yield_label=True):
            turns.append({
                "start": float(turn.start),
                "end": float(turn.end),
                "speaker": str(speaker)
            })
        # Optionally return as DataFrame
        df = pd.DataFrame(turns)
        return turns, df
