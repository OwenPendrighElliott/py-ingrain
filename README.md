# Ingrain Python Client

This is the Python client for the Ingrain API. It provides a simple interface to interact with the Ingrain API.

## Install
    
```bash
pip install .
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