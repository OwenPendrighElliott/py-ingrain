import ingrain.ingrain_errors
import pytest
import ingrain
import numpy as np

BASE_URL = "http://127.0.0.1:8686"

# test models
SENTENCE_TRANSFORMER_MODEL = "intfloat/e5-small-v2"
OPENCLIP_MODEL = "ViT-B-32"
OPENCLIP_PRETRAINED = "laion2b_s34b_b79k"

CLIENT = ingrain.Client(url=BASE_URL)


def check_server_running():
    response = CLIENT.health()


def load_openclip_model():
    model = CLIENT.load_clip_model(name=OPENCLIP_MODEL, pretrained=OPENCLIP_PRETRAINED)


def load_sentence_transformer_model():
    model = CLIENT.load_sentence_transformer_model(name=SENTENCE_TRANSFORMER_MODEL)


@pytest.mark.integration
def test_health():
    check_server_running()
    assert CLIENT.health() == {"message": "The server is running."}


@pytest.mark.integration
def test_load_sentence_transformer_model():
    check_server_running()
    model = CLIENT.load_sentence_transformer_model(name=SENTENCE_TRANSFORMER_MODEL)
    assert model.name == SENTENCE_TRANSFORMER_MODEL


@pytest.mark.integration
def test_load_loaded_sentence_transformer_model():
    check_server_running()
    load_sentence_transformer_model()
    model = CLIENT.load_sentence_transformer_model(name=SENTENCE_TRANSFORMER_MODEL)
    assert model.name == SENTENCE_TRANSFORMER_MODEL


@pytest.mark.integration
def test_load_clip_model():
    check_server_running()
    model = CLIENT.load_clip_model(name=OPENCLIP_MODEL, pretrained=OPENCLIP_PRETRAINED)
    assert model.name == OPENCLIP_MODEL
    assert model.pretrained == OPENCLIP_PRETRAINED


@pytest.mark.integration
def test_infer_text():
    check_server_running()
    load_sentence_transformer_model()
    test_text = "This is a test sentence."
    response = CLIENT.infer_text(name=SENTENCE_TRANSFORMER_MODEL, text=test_text)
    assert "embedding" in response
    assert "processingTimeMs" in response


@pytest.mark.integration
def test_infer_image():
    check_server_running()
    load_openclip_model()
    test_image = "data:image/jpeg;base64,iVBORw0KGgoAAAANSUhEUgAAAOAAAADgCAIAAACVT/22AAACkElEQVR4nOzUMQ0CYRgEUQ5wgwAUnA+EUKKJBkeowAHVJd/kz3sKtpjs9fH9nDjO/npOT1jKeXoA/CNQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKRtl/t7esNSbvs2PWEpHpQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIl7RcAAP//iL8GbQ2nM1wAAAAASUVORK5CYII="
    response = CLIENT.infer_image(
        name=OPENCLIP_MODEL, pretrained=OPENCLIP_PRETRAINED, image=test_image
    )
    assert "embedding" in response
    assert "processingTimeMs" in response


@pytest.mark.integration
def test_infer_text_image():
    check_server_running()
    load_openclip_model()

    # this image is pink
    test_image = "data:image/jpeg;base64,iVBORw0KGgoAAAANSUhEUgAAAOAAAADgCAIAAACVT/22AAACcklEQVR4nOzSMRHAIADAwF4PbVjFITMOWMnwryBDxp7rg6r/dQDcGJQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaScAAP//3nYDppOW6x0AAAAASUVORK5CYII="
    test_texts = ["A green image", "A pink image"]
    response = CLIENT.infer(
        name=OPENCLIP_MODEL,
        pretrained=OPENCLIP_PRETRAINED,
        text=test_texts,
        image=test_image,
    )
    assert "text_embeddings" in response
    assert "image_embeddings" in response
    assert len(response["text_embeddings"]) == 2

    image_embeddings_arr = np.array(response["image_embeddings"])
    text_embeddings_arr = np.array(response["text_embeddings"])

    image_text_similarities = np.dot(image_embeddings_arr, text_embeddings_arr.T)
    assert image_text_similarities[0, 0] < image_text_similarities[0, 1]


@pytest.mark.integration
def test_unload_model():
    check_server_running()
    load_sentence_transformer_model()
    response = CLIENT.unload_model(name=SENTENCE_TRANSFORMER_MODEL)
    assert "unloaded successfully" in response["message"]


@pytest.mark.integration
def test_delete_model():
    check_server_running()
    load_sentence_transformer_model()
    response = CLIENT.delete_model(name=SENTENCE_TRANSFORMER_MODEL)
    assert "deleted successfully" in response["message"]


@pytest.mark.integration
def test_delete_clip_model():
    check_server_running()
    load_openclip_model()
    response = CLIENT.delete_model(name=OPENCLIP_MODEL, pretrained=OPENCLIP_PRETRAINED)
    assert "deleted successfully" in response["message"]


@pytest.mark.integration
def test_loaded_models():
    check_server_running()
    load_sentence_transformer_model()
    load_openclip_model()
    assert "models" in CLIENT.loaded_models()


@pytest.mark.integration
def test_repository_models():
    check_server_running()
    assert "models" in CLIENT.repository_models()


@pytest.mark.integration
def test_metrics():
    check_server_running()
    assert "model_stats" in CLIENT.metrics()
