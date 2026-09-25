"""Augmentiert alle Clips und speichert Spektrogramm-Features (RaggedMmap)."""

import shutil
from pathlib import Path

from .common import LOG, Paths, count_wavs


def _augmenter(paths: Paths, cfg: dict):
    from microwakeword.audio.augmentation import Augmentation

    acfg = cfg["augmentation"]
    backgrounds = [str(p) for p in (paths.fma, paths.audioset) if p.exists()]
    # Eigene Negativ-Aufnahmen (z. B. Gespräche auf Berndeutsch) auch als Hintergrund
    if count_wavs(paths.real_negative / "train"):
        backgrounds.append(str(paths.real_negative / "train"))
    return Augmentation(
        augmentation_duration_s=acfg["duration_s"],
        augmentation_probabilities=acfg["probabilities"],
        impulse_paths=[str(paths.rirs)] if paths.rirs.exists() else [],
        background_paths=backgrounds,
        background_min_snr_db=acfg["background_min_snr_db"],
        background_max_snr_db=acfg["background_max_snr_db"],
        min_jitter_s=0.195,
        max_jitter_s=0.205,
    )


def _write(out_dir: Path, generator):
    from mmap_ninja.ragged import RaggedMmap

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    RaggedMmap.from_generator(out_dir=str(out_dir), sample_generator=generator, batch_size=100, verbose=True)


def _spectrograms(clips, augmenter, slide_frames):
    from microwakeword.audio.spectrograms import SpectrogramGeneration

    return SpectrogramGeneration(clips=clips, augmenter=augmenter, slide_frames=slide_frames, step_ms=10)


def _clips(directory: Path, split: bool):
    from microwakeword.audio.clips import Clips

    return Clips(
        input_directory=str(directory),
        file_pattern="*.wav",
        max_clip_duration_s=None,
        remove_silence=False,
        random_split_seed=10 if split else None,
        split_count=0.1,
    )


def split_set(name: str, source: Path, target: Path, augmenter, train_repeat: int, slide: int | None):
    """TTS-Clips: 80/10/10 aufteilen."""
    if count_wavs(source) == 0:
        LOG.warning("%s: keine Clips in %s", name, source)
        return False
    shutil.rmtree(target, ignore_errors=True)
    clips = _clips(source, split=True)
    LOG.info("%s: Training", name)
    _write(target / "training" / "wakeword_mmap",
           _spectrograms(clips, augmenter, slide).spectrogram_generator(split="train", repeat=train_repeat))
    LOG.info("%s: Validierung", name)
    _write(target / "validation" / "wakeword_mmap",
           _spectrograms(clips, augmenter, slide).spectrogram_generator(split="validation", repeat=1))
    LOG.info("%s: Test", name)
    _write(target / "testing" / "wakeword_mmap",
           _spectrograms(clips, augmenter, 1 if slide else None).spectrogram_generator(split="test", repeat=1))
    return True


def real_positive(paths: Paths, cfg: dict, augmenter):
    train_dir, test_dir = paths.real_positive / "train", paths.real_positive / "test"
    target = paths.features / "real_positive"
    shutil.rmtree(target, ignore_errors=True)
    if count_wavs(train_dir) == 0:
        LOG.warning("Keine echten Aufnahmen importiert, Training nur mit TTS.")
        return False
    reps = cfg["recordings"]["train_repetitions"]
    LOG.info("Echte Aufnahmen: %d Clips × %d Wiederholungen", count_wavs(train_dir), reps)
    _write(target / "training" / "wakeword_mmap",
           _spectrograms(_clips(train_dir, False), augmenter, 10).spectrogram_generator(repeat=reps))
    if count_wavs(test_dir):
        # Validierung mit echten Stimmen: bestimmt, welche Gewichte als «beste» gelten
        _write(target / "validation" / "wakeword_mmap",
               _spectrograms(_clips(test_dir, False), augmenter, 10).spectrogram_generator(repeat=2))
        _write(target / "testing" / "wakeword_mmap",
               _spectrograms(_clips(test_dir, False), augmenter, 1).spectrogram_generator(repeat=2))
    return True


def real_negative(paths: Paths):
    """Lange Negativ-Aufnahmen ohne Augmentierung als Features speichern."""
    from microwakeword.audio.spectrograms import SpectrogramGeneration

    train_dir, test_dir = paths.real_negative / "train", paths.real_negative / "test"
    target = paths.features / "real_negative"
    target_eval = paths.features / "real_negative_eval"
    shutil.rmtree(target, ignore_errors=True)
    shutil.rmtree(target_eval, ignore_errors=True)
    found = False
    if count_wavs(train_dir):
        gen = SpectrogramGeneration(clips=_clips(train_dir, False), augmenter=None,
                                    step_ms=10, split_spectrogram_duration_s=3.0)
        _write(target / "training" / "wakeword_mmap", gen.spectrogram_generator())
        found = True
    if count_wavs(test_dir):
        for mode in ("validation_ambient", "testing_ambient"):
            gen = SpectrogramGeneration(clips=_clips(test_dir, False), augmenter=None, step_ms=10)
            _write(target_eval / mode / "wakeword_mmap", gen.spectrogram_generator())
    return found


def run(paths: Paths, cfg: dict):
    augmenter = _augmenter(paths, cfg)
    split_set("TTS positiv", paths.tts_positive, paths.features / "tts_positive", augmenter, train_repeat=1, slide=10)
    split_set("TTS negativ", paths.tts_negative, paths.features / "tts_negative", augmenter, train_repeat=1, slide=None)
    real_positive(paths, cfg, augmenter)
    real_negative(paths)
    LOG.info("Features fertig in %s", paths.features)
