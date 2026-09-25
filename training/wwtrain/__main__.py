"""Kommandozeile: python -m wwtrain <schritt>"""

import argparse
import logging
from pathlib import Path

from .common import Paths, load_config, work_root


def check_gpu():
    import tensorflow as tf

    print("TensorFlow", tf.__version__)
    print("Gebaut mit ROCm:", tf.test.is_built_with_rocm())
    gpus = tf.config.list_physical_devices("GPU")
    print("GPUs:", gpus or "keine – Training läuft auf der CPU (langsam)")
    if gpus:
        with tf.device("/GPU:0"):
            x = tf.random.normal((2048, 2048))
            print("Testrechnung auf GPU ok:", float(tf.reduce_sum(tf.matmul(x, x))) != 0)


def main():
    parser = argparse.ArgumentParser(prog="wwtrain", description="Wake-Word-Training für microWakeWord")
    parser.add_argument("--config", default="config.yaml")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check-gpu", help="Prüft, ob TensorFlow die GPU sieht")
    d = sub.add_parser("download", help="Stimmen, Geräusche und Negativ-Daten herunterladen")
    d.add_argument("--skip-negative-features", action="store_true", help="die ~6 GB Negativ-Features auslassen")
    sub.add_parser("import", help="Eigene Aufnahmen aus recordings/ importieren und schneiden")
    sub.add_parser("preview", help="Einige TTS-Beispiele zum Reinhören erzeugen")
    t = sub.add_parser("tts", help="Synthetische Beispiele erzeugen")
    t.add_argument("--force", action="store_true", help="vorhandene Beispiele neu erzeugen")
    sub.add_parser("features", help="Augmentieren und Spektrogramme berechnen")
    tr = sub.add_parser("train", help="Modell trainieren")
    tr.add_argument("--fresh", action="store_true", help="altes Training löschen statt fortsetzen")
    tr.add_argument("--convert-only", action="store_true", help="nur beste Gewichte konvertieren/testen")
    e = sub.add_parser("evaluate", help="Modell mit echten Aufnahmen testen")
    e.add_argument("--model", type=Path, help="anderes .tflite testen, z. B. ein altes Modell")
    e.add_argument("--window", type=int, help="sliding_window_size (Standard aus config.yaml)")
    a = sub.add_parser("all", help="import → tts → features → train → evaluate")
    a.add_argument("--fresh", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    root = work_root()
    cfg = load_config(root / args.config)
    paths = Paths(root, cfg)

    if args.cmd == "check-gpu":
        check_gpu()
    elif args.cmd == "download":
        from . import download
        download.run(paths, cfg, skip_negative_features=args.skip_negative_features)
    elif args.cmd == "import":
        from . import recordings
        recordings.run(paths, cfg)
    elif args.cmd == "preview":
        from . import tts
        tts.preview(paths, cfg)
    elif args.cmd == "tts":
        from . import tts
        tts.run(paths, cfg, force=args.force)
    elif args.cmd == "features":
        from . import features
        features.run(paths, cfg)
    elif args.cmd == "train":
        from . import train
        train.run(paths, cfg, fresh=args.fresh, convert_only=args.convert_only)
    elif args.cmd == "evaluate":
        from . import evaluate
        evaluate.run(paths, cfg, model_path=args.model, window=args.window)
    elif args.cmd == "all":
        from . import evaluate, features, recordings, train, tts
        recordings.run(paths, cfg)
        tts.run(paths, cfg)
        features.run(paths, cfg)
        train.run(paths, cfg, fresh=args.fresh)
        evaluate.run(paths, cfg)


if __name__ == "__main__":
    main()
