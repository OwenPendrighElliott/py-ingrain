import pytest
from unittest.mock import patch, MagicMock
from ingrain.model import Model
from ingrain.ingrain_errors import error_factory
import ingrain


@pytest.fixture
def mock_requestor():
    # Patch where PyCURLEngine is instantiated inside the Client class
    with patch("ingrain.client.PyCURLEngine") as MockRequestor:
        yield MockRequestor.return_value


@pytest.fixture
def client(mock_requestor):
    return ingrain.Client()


def test_health(client: ingrain.Client, mock_requestor):
    mock_requestor.get.return_value = ("OK", 200)
    response = client.health()
    assert response == ["OK", "OK"]
    assert mock_requestor.get.call_count == 2


def test_health_error(client: ingrain.Client, mock_requestor):
    mock_requestor.get.return_value = ("Error", 500)
    with pytest.raises(Exception):
        client.health()


def test_loaded_models(client: ingrain.Client, mock_requestor):
    mock_requestor.get.return_value = (["model1", "model2"], 200)
    response = client.loaded_models()
    assert response == ["model1", "model2"]
    mock_requestor.get.assert_called_once_with("http://localhost:8687/loaded_models")


def test_repository_models(client: ingrain.Client, mock_requestor):
    mock_requestor.get.return_value = (["repo_model1", "repo_model2"], 200)
    response = client.repository_models()
    assert response == ["repo_model1", "repo_model2"]
    mock_requestor.get.assert_called_once_with(
        "http://localhost:8687/repository_models"
    )


def test_metrics(client: ingrain.Client, mock_requestor):
    mock_requestor.get.return_value = ({"metric": "value"}, 200)
    response = client.metrics()
    assert response == {"metric": "value"}
    mock_requestor.get.assert_called_once_with("http://localhost:8686/metrics")


def test_load_clip_model(client: ingrain.Client, mock_requestor):
    mock_requestor.post.return_value = ({"success": True}, 200)
    model = client.load_clip_model("clip_model_name")
    assert isinstance(model, Model)
    mock_requestor.post.assert_called_once()


def test_load_sentence_transformer_model(client: ingrain.Client, mock_requestor):
    mock_requestor.post.return_value = ({"success": True}, 200)
    model = client.load_sentence_transformer_model("sentence_transformer_model")
    assert isinstance(model, Model)
    mock_requestor.post.assert_called_once()


def test_unload_model(client: ingrain.Client, mock_requestor):
    mock_requestor.post.return_value = ({"success": True}, 200)
    response = client.unload_model("model_name")
    assert response == {"success": True}
    mock_requestor.post.assert_called_once()


def test_delete_model(client: ingrain.Client, mock_requestor):
    mock_requestor.post.return_value = ({"success": True}, 200)
    response = client.delete_model("model_name")
    assert response == {"success": True}
    mock_requestor.post.assert_called_once()


def test_infer_text(client: ingrain.Client, mock_requestor):
    mock_requestor.post.return_value = ({"result": "inference_result"}, 200)
    response = client.infer_text("text_model", text=["sample text"])
    assert response == {"result": "inference_result"}
    mock_requestor.post.assert_called_once()


def test_infer_image(client: ingrain.Client, mock_requestor):
    mock_requestor.post.return_value = ({"result": "image_inference_result"}, 200)
    response = client.infer_image("image_model", image=["image_data"])
    assert response == {"result": "image_inference_result"}
    mock_requestor.post.assert_called_once()


def test_infer(client: ingrain.Client, mock_requestor):
    mock_requestor.post.return_value = ({"result": "combined_inference_result"}, 200)
    response = client.infer("clip_model", text="text", image="image")
    assert response == {"result": "combined_inference_result"}
    mock_requestor.post.assert_called_once()
