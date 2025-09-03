import pytest
from unittest.mock import patch
import ingrain


@pytest.fixture
def mock_requestor():
    # Patch where PyCURLEngine is instantiated inside the Client class
    with patch("ingrain.client.PyCURLEngine") as MockRequestor:
        yield MockRequestor.return_value


@pytest.fixture
def client(mock_requestor):
    return ingrain.Client()


def test_mock_health(client: ingrain.Client, mock_requestor):
    mock_requestor.get.return_value = ({"message": "OK"}, 200)
    response = client.health()
    print(response)
    assert response[0].message == "OK"
    assert response[1].message == "OK"
    assert mock_requestor.get.call_count == 2


def test_mock_health_error(client: ingrain.Client, mock_requestor):
    mock_requestor.get.return_value = ("Error", 500)
    with pytest.raises(Exception):
        client.health()


def test_mock_loaded_models(client: ingrain.Client, mock_requestor):
    mock_requestor.get.return_value = (
        {
            "models": [
                {"name": "model1", "library": "lib"},
                {"name": "model2", "library": "lib"},
            ]
        },
        200,
    )
    response = client.loaded_models()
    assert [m.name for m in response.models] == ["model1", "model2"]
    mock_requestor.get.assert_called_once_with("http://localhost:8687/loaded_models")


def test_mock_repository_models(client: ingrain.Client, mock_requestor):
    mock_requestor.get.return_value = (
        {
            "models": [
                {"name": "model1", "state": "READY"},
                {"name": "model2", "state": "READY"},
            ]
        },
        200,
    )
    response = client.repository_models()
    assert [m.name for m in response.models] == ["model1", "model2"]
    mock_requestor.get.assert_called_once_with(
        "http://localhost:8687/repository_models"
    )
