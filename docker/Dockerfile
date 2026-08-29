ARG CUDA_IMAGE=nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04
FROM ${CUDA_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ARG VERL_TAG=v0.8.0

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    CUDA_DEVICE_MAX_CONNECTIONS=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      git curl wget unzip build-essential libaio-dev \
      python3 python3-dev python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && python3 --version

WORKDIR /workspace/reasonsql
COPY . /workspace/reasonsql

# Install a CUDA 12.8 PyTorch wheel first, then project/GPU dependencies.
RUN python3 -m pip install --upgrade pip setuptools wheel \
    && python3 -m pip install torch --index-url https://download.pytorch.org/whl/cu128 \
    && python3 -m pip install -r requirements.txt \
    && python3 -m pip install -r requirements-gpu.txt \
    && git clone --branch ${VERL_TAG} --depth 1 https://github.com/verl-project/verl.git /opt/verl \
    && python3 -m pip install -e /opt/verl

ENV PYTHONPATH=/workspace/reasonsql:/opt/verl

RUN python3 -m compileall -q reasonsql common scripts eval patches tests \
    && python3 -m unittest discover -s tests

ENTRYPOINT ["/workspace/reasonsql/docker/entrypoint.sh"]
CMD ["bash"]
