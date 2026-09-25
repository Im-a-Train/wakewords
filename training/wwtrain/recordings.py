"""Importiert eigene Aufnahmen.

Positiv: recordings/positive/<person>/<datei> enthält viele «Hey Henä» mit
Pausen dazwischen. Jede Datei wird per Sprachaktivitätserkennung (VAD) in
einzelne Schnipsel zerschnitten und reproduzierbar in train/test aufgeteilt.

Negativ: recordings/negative/<datei> enthält lange Aufnahmen ohne Wake Word.
Die letzten 20 % jeder Datei gehen in den Testsatz (Fehlauslösungen pro Stunde).
"""

import shutil
from pathlib import Path

import numpy as np
import webrtcvad

from .common import LOG, SAMPLE_RATE, Paths, load_audio, save_wav, stable_fraction

AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".aac", ".ogg", ".opus", ".flac", ".webm", ".mp4", ".3gp", ".amr"}

FRAME = 480  # 30 ms
MERGE_GAP_S = 0.45  # kürzere Pausen gehören zum selben «Hey Henä»
PAD_S = 0.15


def audio_files(directory: Path):
    return sorted(p for p in directory.rglob("*") if p.suffix.lower() in AUDIO_EXTENSIONS)


def split_utterances(audio: np.ndarray, min_s: float, max_s: float, aggressiveness: int = 2):
    vad = webrtcvad.Vad(aggressiveness)
    n = len(audio) // FRAME
    # Pegel normalisieren, damit leise Handy-Aufnahmen erkannt werden
    peak = np.abs(audio).max() or 1
    norm = (audio.astype(np.float64) * (20000 / peak)).clip(-32768, 32767).astype(np.int16)
    speech = np.array([vad.is_speech(norm[i * FRAME:(i + 1) * FRAME].tobytes(), SAMPLE_RATE) for i in range(n)])

    # Frames mit Energie klar über dem Grundrauschen zusätzlich verlangen
    frames = audio[: n * FRAME].reshape(n, FRAME).astype(np.float64)
    energy = 20 * np.log10(np.sqrt((frames**2).mean(axis=1)) + 1)
    floor = np.percentile(energy, 20)
    speech &= energy > floor + 10

    segments, start, gap = [], None, 0
    max_gap = int(MERGE_GAP_S * SAMPLE_RATE / FRAME)
    for i, s in enumerate(speech):
        if s:
            if start is None:
                start = i
            gap = 0
        elif start is not None:
            gap += 1
            if gap > max_gap:
                segments.append((start, i - gap + 1))
                start, gap = None, 0
    if start is not None:
        segments.append((start, n - gap))

    pad = int(PAD_S * SAMPLE_RATE)
    result, rejected = [], []
    for a, b in segments:
        lo, hi = max(0, a * FRAME - pad), min(len(audio), b * FRAME + pad)
        dur = (b - a) * FRAME / SAMPLE_RATE
        (result if min_s <= dur <= max_s else rejected).append((lo, hi, dur))
    return result, rejected


def import_positive(paths: Paths, cfg: dict):
    rcfg = cfg["recordings"]
    out_train = paths.real_positive / "train"
    out_test = paths.real_positive / "test"
    for d in (out_train, out_test):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)

    if not paths.recordings_positive.exists():
        LOG.warning("Keine Aufnahmen in %s gefunden", paths.recordings_positive)
        return

    summary = {}
    for f in audio_files(paths.recordings_positive):
        rel = f.relative_to(paths.recordings_positive)
        person = rel.parts[0] if len(rel.parts) > 1 else "unbekannt"
        audio = load_audio(f)
        segments, rejected = split_utterances(audio, rcfg["min_duration_s"], rcfg["max_duration_s"])
        stem = "__".join(rel.with_suffix("").parts).replace(" ", "_")
        if len(segments) == 0 and len(audio) / SAMPLE_RATE <= rcfg["max_duration_s"] + 2 * PAD_S:
            # Datei enthält nur ein einzelnes, bereits geschnittenes Beispiel
            segments = [(0, len(audio), len(audio) / SAMPLE_RATE)]
        for i, (lo, hi, _) in enumerate(segments):
            name = f"{stem}__{i:03d}"
            target = out_test if stable_fraction(name) < rcfg["test_fraction"] else out_train
            save_wav(target / f"{name}.wav", audio[lo:hi])
        s = summary.setdefault(person, [0, 0])
        s[0] += len(segments)
        s[1] += len(rejected)
        if rejected:
            LOG.info("%s: %d Schnipsel, %d verworfen (Dauern: %s)", rel, len(segments), len(rejected),
                     ", ".join(f"{d:.1f}s" for *_, d in rejected[:8]))
        else:
            LOG.info("%s: %d Schnipsel", rel, len(segments))

    LOG.info("Zusammenfassung pro Person (Schnipsel / verworfen):")
    for person, (ok, bad) in sorted(summary.items()):
        LOG.info("  %-15s %4d / %d", person, ok, bad)
    LOG.info("Training: %d, Test: %d Schnipsel. Bitte in %s kurz reinhören und Fehlschnitte löschen.",
             len(list(out_train.glob("*.wav"))), len(list(out_test.glob("*.wav"))), paths.real_positive)


def import_negative(paths: Paths, cfg: dict):
    out_train = paths.real_negative / "train"
    out_test = paths.real_negative / "test"
    for d in (out_train, out_test):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
    if not paths.recordings_negative.exists():
        return
    total = 0.0
    for f in audio_files(paths.recordings_negative):
        audio = load_audio(f)
        stem = "__".join(f.relative_to(paths.recordings_negative).with_suffix("").parts).replace(" ", "_")
        cut = int(len(audio) * 0.8)
        # In Stücke von max. 10 Minuten schneiden (Speicherbedarf)
        chunk = 600 * SAMPLE_RATE
        for part, (lo, hi, target) in enumerate([(0, cut, out_train), (cut, len(audio), out_test)]):
            for j, start in enumerate(range(lo, hi, chunk)):
                piece = audio[start:min(hi, start + chunk)]
                if len(piece) > 4 * SAMPLE_RATE:
                    save_wav(target / f"{stem}__{part}_{j:03d}.wav", piece)
        total += len(audio) / SAMPLE_RATE
        LOG.info("Negativ: %s (%.1f min)", f.name, len(audio) / SAMPLE_RATE / 60)
    LOG.info("Negativ-Aufnahmen total: %.1f min", total / 60)


def run(paths: Paths, cfg: dict):
    import_positive(paths, cfg)
    import_negative(paths, cfg)
