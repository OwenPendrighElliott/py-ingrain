# Ingrain Python Client

[![PyPI Version](https://img.shields.io/pypi/v/ingrain)](https://pypi.org/project/ingrain/)
![Test Status](https://github.com/OwenPendrighElliott/py-ingrain/actions/workflows/test.yml/badge.svg)

This is the Python client for the Ingrain API. It provides a simple interface to interact with the Ingrain API.

## Install
    
```bash
pip install ingrain
```

## Dev Setup
```bash
uv sync --dev
```

### Testing

#### Unit tests

```bash
uv run pytest
```

#### Integration tests and unit tests

This requires that Ingrain Server is running. You can start it with Docker Compose:

```yml
services:
  ingrain-models:
    image: owenpelliott/ingrain-models:latest
    container_name: ingrain-models
    ports:
      - "8687:8687"
    environment:
      - TRITON_GRPC_URL=triton:8001
      - MAX_BATCH_SIZE=16
      - MODEL_INSTANCES=1
      - INSTANCE_KIND=KIND_GPU # Change to KIND_CPU if using a CPU
    depends_on:
      - triton
    volumes:
      - ./model_repository:/app/model_repository 
      - ${HOME}/.cache/huggingface:/app/model_cache/
  ingrain-inference:
    image: owenpelliott/ingrain-inference:latest
    container_name: ingrain-inference
    ports:
      - "8686:8686"
    environment:
      - TRITON_GRPC_URL=triton:8001
    depends_on:
      - triton
    volumes:
      - ./model_repository:/app/model_repository 
  triton:
    image: nvcr.io/nvidia/tritonserver:25.06-py3
    container_name: triton
    runtime: nvidia # Remove if using a CPU
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    shm_size: "256m"
    command: >
      tritonserver
      --model-repository=/models
      --model-control-mode=explicit
    ports:
      - "8000:8000"
      - "8001:8001"
      - "8002:8002"
    volumes:
      - ./model_repository:/models
    restart:
      unless-stopped
```

```bash
uv run pytest --integration
```