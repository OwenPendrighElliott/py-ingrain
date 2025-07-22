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

```bash
uv run pytest --integration
```