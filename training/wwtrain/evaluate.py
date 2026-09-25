"""Testet ein Modell so, wie es auf dem Voice PE läuft (Streaming, gleitender
Mittelwert), mit euren echten Test-Aufnahmen.

- Erkennungsrate: Anteil der zurückgehaltenen «Hey Henä»-Schnipsel, die erkannt werden
- Fehlauslösungen pro Stunde: auf den eigenen Negativ-Aufnahmen (Testteil) und dem
  «dinner_party_eval»-Satz von microWakeWord
"""

from pathlib import Path

import numpy as np

from .common import LOG, SAMPLE_RATE, Paths, read_wav

CUTOFFS = [round(x, 2) for x in np.arange(0.30, 0.99, 0.05)]
STEP_S = 0.03  # eine Modell-Inferenz pro 30 ms


def _model(path: Path):
    from microwakeword.inference import Model

    return Model(str(path))


def _smoothed(pred, window: int):
    pred = np.asarray(pred, dtype=np.float32)
    if len(pred) < window:
        return pred
    return np.convolve(pred, np.ones(window) / window, mode="valid")


def _spectrogram(audio: np.ndarray):
    from microwakeword.audio.audio_utils import generate_features_for_clip

    return generate_features_for_clip(audio, step_ms=10)


def positive_scores(model_path: Path, clips: list[Path], window: int):
    rng = np.random.default_rng(0)
    scores = []
    for clip in clips:
        audio = read_wav(clip)
        # Etwas Vorlauf (leises Rauschen), damit der Streaming-Zustand eingeschwungen ist
        pre = (rng.normal(0, 30, int(1.5 * SAMPLE_RATE))).astype(np.int16)
        post = (rng.normal(0, 30, int(0.5 * SAMPLE_RATE))).astype(np.int16)
        pred = _model(model_path).predict_spectrogram(_spectrogram(np.concatenate([pre, audio, post])))
        scores.append(float(_smoothed(pred, window).max()))
    return np.array(scores)


def count_activations(smoothed: np.ndarray, cutoff: float, cooldown_steps: int = 33) -> int:
    count, i = 0, 0
    above = smoothed >= cutoff
    while i < len(above):
        if above[i]:
            count += 1
            i += cooldown_steps
        else:
            i += 1
    return count


def negative_streams(paths: Paths):
    """Liefert (Name, Liste von Spektrogrammen) für lange Negativ-Aufnahmen."""
    from mmap_ninja.ragged import RaggedMmap

    own = sorted((paths.real_negative / "test").glob("*.wav"))
    if own:
        yield "eigene Negativ-Aufnahmen", [_spectrogram(read_wav(p)) for p in own]
    dp = paths.negative_datasets / "dinner_party_eval" / "testing_ambient"
    mmaps = list(dp.glob("**/*_mmap"))
    if mmaps:
        yield "dinner_party_eval", [np.asarray(s) for m in mmaps for s in RaggedMmap(str(m))]


def run(paths: Paths, cfg: dict, model_path: Path | None = None, window: int | None = None):
    model_path = model_path or paths.output / f"{cfg['wake_word']['id']}.tflite"
    window = window or cfg["manifest"]["sliding_window_size"]
    if not model_path.exists():
        raise SystemExit(f"Modell {model_path} nicht gefunden")
    LOG.info("Modell: %s (sliding_window_size=%d)", model_path, window)

    clips = sorted((paths.real_positive / "test").glob("*.wav"))
    if not clips:
        LOG.warning("Keine echten Test-Aufnahmen (data/clips/real_positive/test). Nutze TTS-Beispiele.")
        clips = sorted(paths.tts_positive.glob("*.wav"))[-200:]
    pos = positive_scores(model_path, clips, window)

    negatives = []
    for name, spectrograms in negative_streams(paths):
        smoothed = []
        hours = 0.0
        for spec in spectrograms:
            model = _model(model_path)
            pred = model.predict_spectrogram(spec)
            smoothed.append(_smoothed(pred, window))
            hours += len(pred) * STEP_S / 3600
        negatives.append((name, smoothed, hours))
        LOG.info("Negativ-Satz %s: %.2f h", name, hours)

    header = f"{'Schwelle':>8} | {'Erkennung':>9}"
    for name, _, _ in negatives:
        header += f" | {('FA/h ' + name)[:26]:>26}"
    lines = [header, "-" * len(header)]
    table = []
    for cutoff in CUTOFFS:
        recall = float((pos >= cutoff).mean())
        fa = [sum(count_activations(s, cutoff) for s in sm) / max(h, 1e-9) for _, sm, h in negatives]
        table.append((cutoff, recall, fa))
        row = f"{cutoff:>8.2f} | {recall * 100:>8.1f}%"
        for v in fa:
            row += f" | {v:>26.2f}"
        lines.append(row)

    # Empfehlung: höchste Erkennung bei höchstens ~0.5 Fehlauslösungen pro Stunde;
    # bei gleicher Erkennung die höhere (vorsichtigere) Schwelle
    target = 0.5
    ok = [r for r in table if all(v <= target for v in r[2])] if negatives else table
    best = max(ok, key=lambda r: (r[1], r[0])) if ok else table[-1]

    report = "\n".join(lines)
    report += f"\n\n{len(clips)} Positiv-Schnipsel. Empfohlener probability_cutoff: {best[0]:.2f} " \
              f"(Erkennung {best[1] * 100:.1f}%)"
    if negatives and not ok:
        report += f"\nAchtung: keine Schwelle erreicht max. {target} Fehlauslösungen/h. Fehlen die " \
                  "Negativ-Features (./run.sh download)? Sonst mehr Negativ-Aufnahmen sammeln."
    if not negatives:
        report += "\nAchtung: keine Negativ-Daten gefunden, Fehlauslösungen wurden nicht gemessen."
    if best[1] < 0.8:
        report += "\nAchtung: Erkennung unter 80 %. Mehr eigene Aufnahmen, länger trainieren " \
                  "oder negative_class_weight senken (siehe README)."
    worst = sorted(zip(pos, clips))[:10]
    report += "\n\nAm schlechtesten erkannte Schnipsel (reinhören!):\n" + \
              "\n".join(f"  {s:.2f}  {c.name}" for s, c in worst)
    print(report)
    out = paths.output / f"{model_path.stem}_evaluation.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report + "\n", encoding="utf-8")
    LOG.info("Bericht: %s", out)

    if model_path == paths.output / f"{cfg['wake_word']['id']}.tflite":
        from .train import write_manifest

        LOG.info("Manifest mit Schwelle %.2f aktualisiert: %s", best[0], write_manifest(paths, cfg, best[0]))
