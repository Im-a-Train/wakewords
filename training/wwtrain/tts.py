"""Erzeugt synthetische Positiv- und Negativbeispiele mit Piper TTS."""

import multiprocessing as mp
import os
import random
import re
import shutil
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly
from tqdm import tqdm

from .common import LOG, SAMPLE_RATE, Paths, count_wavs, save_wav, trim_silence

PHONEME_RE = re.compile(r"\[\[(.*?)\]\]")

_voices = {}
_voice_dir = None


def _init_worker(voice_dir: str):
    global _voice_dir
    _voice_dir = voice_dir


def _get_voice(name: str):
    from piper import PiperVoice

    import onnxruntime

    if name not in _voices:
        path = os.path.join(_voice_dir, f"{name}.onnx")
        voice = PiperVoice.load(path)
        # Ein Thread pro Prozess, parallelisiert wird über mehrere Prozesse
        opts = onnxruntime.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        voice.session = onnxruntime.InferenceSession(path, sess_options=opts, providers=["CPUExecutionProvider"])
        _voices[name] = voice
    return _voices[name]


def phrase_supported(voice, phrase: str) -> bool:
    id_map = voice.config.phoneme_id_map
    for block in PHONEME_RE.findall(phrase):
        if any(ch not in id_map for ch in block.strip() if ch != " "):
            return False
    return True


def synthesize(task):
    """Wird im Worker ausgeführt. Gibt (index, audio) oder (index, None) zurück."""
    from piper import SynthesisConfig

    index, voice_name, phrase, seed, tcfg = task
    rnd = random.Random(seed)
    voice = _get_voice(voice_name)
    speaker = rnd.randrange(voice.config.num_speakers) if voice.config.num_speakers > 1 else None
    syn = SynthesisConfig(
        speaker_id=speaker,
        length_scale=rnd.uniform(*tcfg["length_scale"]),
        noise_scale=rnd.uniform(*tcfg["noise_scale"]),
        noise_w_scale=rnd.uniform(*tcfg["noise_w_scale"]),
    )
    chunks = [c.audio_int16_array for c in voice.synthesize(phrase, syn_config=syn)]
    if not chunks:
        return index, None
    audio = np.concatenate(chunks)
    ratio = Fraction(SAMPLE_RATE, voice.config.sample_rate)
    if ratio != 1:
        audio = resample_poly(audio.astype(np.float32), ratio.numerator, ratio.denominator)
        audio = np.clip(audio, -32768, 32767).astype(np.int16)
    audio = trim_silence(audio)
    duration = len(audio) / SAMPLE_RATE
    if not (tcfg["min_duration_s"] <= duration <= tcfg["max_duration_s"]):
        return index, None
    return index, audio


def build_combos(paths: Paths, cfg: dict, phrases: list[str]):
    """Liste von (Stimme, Gewicht, [unterstützte Phrasen])."""
    from piper import PiperVoice

    combos = []
    for name, weight in cfg["tts"]["voices"].items():
        onnx = paths.voices / f"{name}.onnx"
        if not onnx.exists():
            LOG.warning("Stimme %s fehlt, zuerst «download» ausführen", name)
            continue
        voice = PiperVoice.load(str(onnx))
        ok = [p for p in phrases if phrase_supported(voice, p)]
        skipped = set(phrases) - set(ok)
        if skipped:
            LOG.warning("%s unterstützt nicht: %s", name, ", ".join(sorted(skipped)))
        if ok:
            combos.append((name, float(weight), ok))
    if not combos:
        raise SystemExit("Keine nutzbaren Stimmen gefunden.")
    return combos


def task_stream(combos, tcfg, seed: int):
    rnd = random.Random(seed)
    names = [c[0] for c in combos]
    weights = [c[1] for c in combos]
    phrase_map = {c[0]: c[2] for c in combos}
    i = 0
    while True:
        voice = rnd.choices(names, weights)[0]
        yield (i, voice, rnd.choice(phrase_map[voice]), rnd.getrandbits(32), tcfg)
        i += 1


def generate(paths: Paths, cfg: dict, kind: str, out_dir: Path, count: int, seed: int, force: bool):
    tcfg = cfg["tts"]
    phrases = tcfg[kind]["phrases"]
    if not force and count_wavs(out_dir) >= count:
        LOG.info("%s: %d Beispiele bereits vorhanden (mit --force neu erzeugen)", out_dir.name, count_wavs(out_dir))
        return
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True)

    combos = build_combos(paths, cfg, phrases)
    worker_cfg = {k: tcfg[k] for k in ("length_scale", "noise_scale", "noise_w_scale", "min_duration_s", "max_duration_s")}
    procs = max(1, (os.cpu_count() or 2) - 1)
    LOG.info("Erzeuge %d %s-Beispiele mit %d Prozessen", count, kind, procs)

    written = rejected = 0
    with mp.get_context("spawn").Pool(procs, initializer=_init_worker, initargs=(str(paths.voices),)) as pool:
        with tqdm(total=count, desc=f"TTS {kind}") as bar:
            for _, audio in pool.imap_unordered(synthesize, task_stream(combos, worker_cfg, seed), chunksize=8):
                if audio is None:
                    rejected += 1
                    if rejected > 5 * count + 100:
                        raise SystemExit("Zu viele verworfene TTS-Beispiele, bitte Dauergrenzen prüfen.")
                    continue
                save_wav(out_dir / f"{written:06d}.wav", audio)
                written += 1
                bar.update(1)
                if written >= count:
                    break
            pool.terminate()
    LOG.info("%s: %d geschrieben, %d verworfen (zu kurz/lang)", kind, written, rejected)


def preview(paths: Paths, cfg: dict, per_phrase: int = 3):
    """Ein paar Beispiele pro Phrase zum Reinhören nach output/preview/."""
    shutil.rmtree(paths.preview, ignore_errors=True)
    tcfg = cfg["tts"]
    worker_cfg = {k: tcfg[k] for k in ("length_scale", "noise_scale", "noise_w_scale")}
    worker_cfg.update(min_duration_s=0.0, max_duration_s=99.0)
    _init_worker(str(paths.voices))
    rnd = random.Random(1)
    for kind in ("positive", "negative"):
        phrases = tcfg[kind]["phrases"]
        combos = build_combos(paths, cfg, phrases)
        for pi, phrase in enumerate(dict.fromkeys(phrases)):
            usable = [c for c in combos if phrase in c[2]]
            for j in range(per_phrase):
                voice = rnd.choices([c[0] for c in usable], [c[1] for c in usable])[0]
                _, audio = synthesize((0, voice, phrase, rnd.getrandbits(32), worker_cfg))
                label = re.sub(r"[^\w]+", "_", phrase).strip("_")[:40]
                save_wav(paths.preview / kind / f"{pi:02d}_{label}__{voice}_{j}.wav", audio)
    LOG.info("Vorschau gespeichert in %s", paths.preview)


def run(paths: Paths, cfg: dict, force: bool = False):
    tcfg = cfg["tts"]
    generate(paths, cfg, "positive", paths.tts_positive, tcfg["positive"]["count"], seed=1, force=force)
    generate(paths, cfg, "negative", paths.tts_negative, tcfg["negative"]["count"], seed=2, force=force)
