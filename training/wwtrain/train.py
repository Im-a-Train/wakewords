"""Schreibt die microWakeWord-Trainingskonfiguration und startet das Training."""

import json
import shutil
import subprocess
import sys

import yaml

from .common import LOG, Paths


def _feature(dir_, weight, truth, strategy, penalty=1.0):
    return {
        "features_dir": str(dir_),
        "sampling_weight": weight,
        "penalty_weight": penalty,
        "truth": truth,
        "truncation_strategy": strategy,
        "type": "mmap",
    }


def training_parameters(paths: Paths, cfg: dict) -> dict:
    t = cfg["training"]
    w = t["sampling_weights"]
    f = paths.features
    neg = paths.negative_datasets

    features = []
    candidates = [
        (f / "tts_positive", w["tts_positive"], True, "truncate_start"),
        (f / "real_positive", w["real_positive"], True, "truncate_start"),
        (f / "tts_negative", w["tts_negative"], False, "truncate_start"),
        (f / "real_negative", w["real_negative"], False, "random"),
        (neg / "speech", w["speech"], False, "random"),
        (neg / "dinner_party", w["dinner_party"], False, "random"),
        (neg / "no_speech", w["no_speech"], False, "random"),
        # nur für Validierung/Test (Fehlauslösungen pro Stunde)
        (neg / "dinner_party_eval", 0.0, False, "split"),
        (f / "real_negative_eval", 0.0, False, "split"),
    ]
    for dir_, weight, truth, strategy in candidates:
        if dir_.exists():
            features.append(_feature(dir_, weight, truth, strategy))
        else:
            LOG.warning("Datenquelle fehlt und wird ausgelassen: %s", dir_)
    if not any(x["truth"] for x in features):
        raise SystemExit("Keine Positiv-Features gefunden. Zuerst «tts» und «features» ausführen.")

    return {
        "window_step_ms": 10,
        "train_dir": str(paths.trained),
        "features": features,
        "training_steps": t["steps"],
        "positive_class_weight": t["positive_class_weight"],
        "negative_class_weight": t["negative_class_weight"],
        "learning_rates": t["learning_rates"],
        "batch_size": t["batch_size"],
        "time_mask_max_size": t["time_mask_max_size"],
        "time_mask_count": t["time_mask_count"],
        "freq_mask_max_size": t["freq_mask_max_size"],
        "freq_mask_count": t["freq_mask_count"],
        "eval_step_interval": t["eval_step_interval"],
        "clip_duration_ms": t["clip_duration_ms"],
        "target_minimization": t["target_minimization"],
        "minimization_metric": t["minimization_metric"],
        "maximization_metric": t["maximization_metric"],
    }


def write_manifest(paths: Paths, cfg: dict, cutoff: float | None = None):
    ww, m = cfg["wake_word"], cfg["manifest"]
    manifest = {
        "type": "micro",
        "wake_word": ww["display_name"],
        "author": ww["author"],
        "website": ww["website"],
        "model": f"{ww['id']}.tflite",
        "trained_languages": m["trained_languages"],
        "version": 2,
        "micro": {
            "probability_cutoff": round(cutoff if cutoff is not None else m["probability_cutoff"], 2),
            "sliding_window_size": m["sliding_window_size"],
            "feature_step_size": 10,
            "tensor_arena_size": m["tensor_arena_size"],
            "minimum_esphome_version": m["minimum_esphome_version"],
        },
    }
    out = paths.output / f"{ww['id']}.json"
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


def run(paths: Paths, cfg: dict, fresh: bool = False, convert_only: bool = False):
    if fresh and paths.trained.exists():
        LOG.info("Lösche altes Training in %s", paths.trained)
        shutil.rmtree(paths.trained)
    paths.trained.parent.mkdir(parents=True, exist_ok=True)
    params_file = paths.trained.parent / f"{cfg['wake_word']['id']}_training_parameters.yaml"
    with open(params_file, "w") as fh:
        yaml.safe_dump(training_parameters(paths, cfg), fh, sort_keys=False)
    LOG.info("Trainingsparameter: %s", params_file)

    cmd = [
        sys.executable, "-m", "wwtrain.mww_run",
        f"--training_config={params_file}",
        f"--train={0 if convert_only else 1}",
        "--restore_checkpoint=1",
        "--test_tf_nonstreaming=0",
        "--test_tflite_nonstreaming=0",
        "--test_tflite_nonstreaming_quantized=0",
        "--test_tflite_streaming=0",
        "--test_tflite_streaming_quantized=1",
        "--use_weights=best_weights",
        *cfg["training"]["model_args"],
    ]
    LOG.info("Starte: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)

    paths.output.mkdir(parents=True, exist_ok=True)
    quant_dir = paths.trained / "tflite_stream_state_internal_quant"
    model_id = cfg["wake_word"]["id"]
    shutil.copy(quant_dir / "stream_state_internal_quant.tflite", paths.output / f"{model_id}.tflite")
    roc = quant_dir / "tflite_streaming_roc.txt"
    if roc.exists():
        shutil.copy(roc, paths.output / f"{model_id}_roc.txt")
        LOG.info("Testergebnisse von microWakeWord:\n%s", roc.read_text())
    manifest = write_manifest(paths, cfg)
    LOG.info("Modell: %s, Manifest: %s", paths.output / f"{model_id}.tflite", manifest)
    LOG.info("Als nächstes: ./run.sh evaluate (empfiehlt einen Schwellwert)")
