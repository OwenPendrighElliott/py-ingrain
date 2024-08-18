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
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### Testing

#### Unit tests

```bash
pytest
```

#### Integration tests and unit tests

```bash
pytest --integration
```