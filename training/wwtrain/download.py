"""Lädt Stimmen, Hall-Impulsantworten, Hintergrundgeräusche und die
vorberechneten Negativ-Features von microWakeWord herunter."""

import json
import shutil
import subprocess
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .common import LOG, SAMPLE_RATE, Paths, save_wav

HF = "https://huggingface.co"
RIR_REPO = "davidscripka/MIT_environmental_impulse_responses"
AUDIOSET_PARQUETS = ["data/bal_train/00.parquet", "data/bal_train/01.parquet"]
FMA_ZIP = f"{HF}/datasets/mchl914/fma_xsmall/resolve/main/fma_xs.zip"
NEGATIVE_ROOT = f"{HF}/datasets/kahrendt/microwakeword/resolve/main/"
NEGATIVE_SETS = ["speech", "dinner_party", "dinner_party_eval", "no_speech"]


def fetch(url: str, dest: Path):
    """Lädt eine Datei herunter (atomar über eine .part-Datei)."""
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as out:
        total = int(resp.headers.get("Content-Length", 0)) or None
        with tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar:
            while chunk := resp.read(1 << 20):
                out.write(chunk)
                bar.update(len(chunk))
    tmp.rename(dest)


def decode_bytes(data: bytes) -> np.ndarray:
    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
        "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "s16le", "-",
    ]
    out = subprocess.run(cmd, input=data, capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype=np.int16)


def download_voices(paths: Paths, voices: list[str]):
    from piper.download_voices import download_voice

    paths.voices.mkdir(parents=True, exist_ok=True)
    for voice in voices:
        LOG.info("Stimme %s", voice)
        download_voice(voice, paths.voices)


def download_rirs(paths: Paths):
    if paths.rirs.exists() and any(paths.rirs.glob("*.wav")):
        LOG.info("Impulsantworten bereits vorhanden")
        return
    with urllib.request.urlopen(f"{HF}/api/datasets/{RIR_REPO}/tree/main/16khz") as r:
        files = [f["path"] for f in json.load(r) if f["path"].endswith(".wav")]
    LOG.info("Lade %d Impulsantworten", len(files))

    def one(p):
        fetch(f"{HF}/datasets/{RIR_REPO}/resolve/main/{p}", paths.rirs / Path(p).name)

    with ThreadPoolExecutor(8) as pool:
        list(tqdm(pool.map(one, files), total=len(files), desc="RIR"))


def download_audioset(paths: Paths):
    import pyarrow.parquet as pq

    done = paths.audioset / ".done"
    if done.exists():
        LOG.info("AudioSet bereits vorhanden")
        return
    paths.audioset.mkdir(parents=True, exist_ok=True)
    tmp_dir = paths.downloads / "tmp"
    for rel in AUDIOSET_PARQUETS:
        local = tmp_dir / rel.replace("/", "_")
        fetch(f"{HF}/datasets/agkphysics/AudioSet/resolve/main/{rel}", local)
        table = pq.read_table(local, columns=["video_id", "audio"])
        ids = table.column("video_id").to_pylist()
        audios = table.column("audio").to_pylist()
        for vid, audio in tqdm(zip(ids, audios), total=len(ids), desc=local.name):
            try:
                save_wav(paths.audioset / f"{vid}.wav", decode_bytes(audio["bytes"]))
            except subprocess.CalledProcessError:
                LOG.warning("AudioSet-Clip %s nicht lesbar, übersprungen", vid)
        local.unlink()
    done.touch()


def download_fma(paths: Paths):
    done = paths.fma / ".done"
    if done.exists():
        LOG.info("FMA bereits vorhanden")
        return
    paths.fma.mkdir(parents=True, exist_ok=True)
    local = paths.downloads / "tmp" / "fma_xs.zip"
    fetch(FMA_ZIP, local)
    with zipfile.ZipFile(local) as z:
        names = [n for n in z.namelist() if n.endswith(".mp3")]
        for name in tqdm(names, desc="FMA"):
            try:
                save_wav(paths.fma / (Path(name).stem + ".wav"), decode_bytes(z.read(name)))
            except subprocess.CalledProcessError:
                LOG.warning("FMA-Datei %s nicht lesbar, übersprungen", name)
    local.unlink()
    done.touch()


def download_negative_features(paths: Paths):
    for name in NEGATIVE_SETS:
        target = paths.negative_datasets / name
        if target.exists():
            LOG.info("Negativ-Features %s bereits vorhanden", name)
            continue
        local = paths.downloads / "tmp" / f"{name}.zip"
        fetch(NEGATIVE_ROOT + f"{name}.zip", local)
        LOG.info("Entpacke %s", local.name)
        with zipfile.ZipFile(local) as z:
            z.extractall(paths.negative_datasets)
        local.unlink()


def run(paths: Paths, cfg: dict, skip_negative_features: bool = False):
    download_voices(paths, list(cfg["tts"]["voices"].keys()))
    download_rirs(paths)
    download_audioset(paths)
    download_fma(paths)
    if not skip_negative_features:
        download_negative_features(paths)
    shutil.rmtree(paths.downloads / "tmp", ignore_errors=True)
    LOG.info("Downloads fertig")

