"""Startet microwakeword.model_train_eval mit einem Workaround für den ROCm-Build.

Mit tensorflow_rocm 2.20 (rocm/tensorflow-Image) bricht der Export des Streaming-Modells ab:
  TypeError: this __dict__ descriptor does not support '_DictWrapper' objects
tensor_util.is_tf_type() wirft dort für interne Dict-Wrapper einen TypeError statt False
zu liefern (Python 3.12, TF < 2.21, ältere Keras-Version im Image). Ein Dict-Wrapper
ist nie ein Tensor, darum ist False die richtige Antwort.
"""

import runpy

from tensorflow.python.framework import tensor_util

_is_tf_type = tensor_util.is_tf_type


def _safe_is_tf_type(x):
    # Gleicher Fix wie in TensorFlow 2.21 bzw. neueren Keras-Versionen
    if type(x).__name__ in ("ObjectProxy", "_DictWrapper"):
        return False
    try:
        return _is_tf_type(x)
    except TypeError:
        return False


tensor_util.is_tf_type = _safe_is_tf_type
tensor_util.is_tensor = _safe_is_tf_type

if __name__ == "__main__":
    runpy.run_module("microwakeword.model_train_eval", run_name="__main__", alter_sys=True)
