"""Gemeinsame Hilfsfunktionen: Konfiguration, Pfade, Audio lesen/schreiben."""

import hashlib
import logging
import os
import subprocess
import wave
from pathlib import Path

import numpy as np
import yaml

LOG = logging.getLogger("wwtrain")

SAMPLE_RATE = 16000


class Paths:
    """Alle Verzeichnisse relativ zum Arbeitsverzeichnis (im Container /work)."""

    def __init__(self, root: Path, cfg: dict):
        self.root = root
        self.data = root / "data"
        self.downloads = self.data / "downloads"
        self.voices = self.downloads / "voices"
        self.rirs = self.downloads / "mit_rirs"
        self.audioset = self.downloads / "audioset_16k"
        self.fma = self.downloads / "fma_16k"
        self.negative_datasets = self.downloads / "negative_datasets"

        self.clips = self.data / "clips"
        self.real_positive = self.clips / "real_positive"
        self.real_negative = self.clips / "real_negative"
        self.tts_positive = self.clips / "tts_positive"
        self.tts_negative = self.clips / "tts_negative"

        self.features = self.data / "features"
        self.trained = self.data / "trained" / cfg["wake_word"]["id"]

        self.recordings_positive = root / cfg["recordings"]["positive_dir"]
        self.recordings_negative = root / cfg["recordings"]["negative_dir"]

        self.output = root / "output"
        self.preview = self.output / "preview"


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def work_root() -> Path:
    return Path(os.environ.get("WWTRAIN_ROOT", os.getcwd())).resolve()


def load_audio(path: Path) -> np.ndarray:
    """Liest eine beliebige Audiodatei (via ffmpeg) als 16 kHz mono int16."""
    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-i", str(path),
        "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "s16le", "-",
    ]
    raw = subprocess.run(cmd, check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.int16)


def save_wav(path: Path, audio: np.ndarray, sample_rate: int = SAMPLE_RATE):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if audio.dtype != np.int16:
        audio = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(audio.tobytes())


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        assert w.getframerate() == SAMPLE_RATE and w.getnchannels() == 1, path
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)


def frame_energy_db(audio: np.ndarray, frame: int = 160) -> np.ndarray:
    n = len(audio) // frame
    if n == 0:
        return np.array([-100.0])
    frames = audio[: n * frame].reshape(n, frame).astype(np.float64)
    return 20 * np.log10(np.sqrt((frames**2).mean(axis=1)) + 1.0)


def trim_silence(audio: np.ndarray, threshold_db: float = 30.0, pad_s: float = 0.08):
    """Schneidet Stille vorne/hinten weg (relativ zum lautesten Frame)."""
    energy = frame_energy_db(audio)
    active = np.where(energy > energy.max() - threshold_db)[0]
    if len(active) == 0:
        return audio
    pad = int(pad_s * SAMPLE_RATE)
    start = max(0, active[0] * 160 - pad)
    end = min(len(audio), (active[-1] + 1) * 160 + pad)
    return audio[start:end]


def stable_fraction(key: str) -> float:
    """Deterministische Zahl in [0, 1) für eine reproduzierbare Aufteilung."""
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0x100000000


def count_wavs(directory: Path) -> int:
    if not directory.exists():
        return 0
    return sum(1 for _ in directory.glob("*.wav"))
