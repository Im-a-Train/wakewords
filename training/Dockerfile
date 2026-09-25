# Standard: ROCm-Image von AMD mit TensorFlow (für Radeon RX 9070 XT / gfx1201).
# CPU-Variante: --build-arg BASE_IMAGE=python:3.12-slim-bookworm
#               --build-arg INSTALL_TENSORFLOW=tensorflow==2.20.*
ARG BASE_IMAGE=rocm/tensorflow:rocm7.2.4-py3.12-tf2.20-dev
FROM ${BASE_IMAGE}

ARG INSTALL_TENSORFLOW=""
ARG MWW_COMMIT=4665173cd35f1cff9a61e06fc427f124766c488e

ENV PIP_BREAK_SYSTEM_PACKAGES=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=1 \
    TF_FORCE_GPU_ALLOW_GROWTH=true

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg git ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Die mitgelieferte TensorFlow-Version (ROCm-Build) darf von pip nicht ersetzt werden
RUN if [ -n "$INSTALL_TENSORFLOW" ]; then pip install "$INSTALL_TENSORFLOW"; fi \
 && python3 -c "import tensorflow as tf; print('TensorFlow', tf.__version__)" \
 && pip freeze | grep -iE '^(tensorflow[a-z_-]*|numpy|keras|protobuf|h5py|ml[-_]dtypes|tensorboard[a-z_-]*)==' > /opt/constraints.txt \
 && cat /opt/constraints.txt

COPY requirements.txt /opt/requirements.txt
RUN pip install -c /opt/constraints.txt -r /opt/requirements.txt

# microWakeWord hat keine __init__.py in den Unterordnern, darum per PYTHONPATH statt pip
RUN git clone https://github.com/kahrendt/microWakeWord /opt/microWakeWord \
 && git -C /opt/microWakeWord checkout ${MWW_COMMIT}

ENV PYTHONPATH=/opt/microWakeWord:/work \
    WWTRAIN_ROOT=/work
WORKDIR /work
ENTRYPOINT ["python3", "-m", "wwtrain"]
CMD ["--help"]
