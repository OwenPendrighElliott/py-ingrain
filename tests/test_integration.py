import ingrain.ingrain_errors
import pytest
import ingrain
import numpy as np

INFERENCE_BASE_URL = "http://127.0.0.1:8686"
MODEL_BASE_URL = "http://127.0.0.1:8687"

# test models
SENTENCE_TRANSFORMER_MODEL = "intfloat/e5-small-v2"
OPENCLIP_MODEL = "ViT-B-32"
OPENCLIP_PRETRAINED = "laion2b_s34b_b79k"


@pytest.fixture
def client():
    return ingrain.Client(
        inference_server_url=INFERENCE_BASE_URL, model_server_url=MODEL_BASE_URL
    )


@pytest.fixture
def client_numpy():
    return ingrain.Client(
        inference_server_url=INFERENCE_BASE_URL,
        model_server_url=MODEL_BASE_URL,
        return_numpy=True,
    )


def check_server_running(client: ingrain.Client):
    response = client.health()


def load_openclip_model(client: ingrain.Client):
    model = client.load_clip_model(name=OPENCLIP_MODEL, pretrained=OPENCLIP_PRETRAINED)


def load_sentence_transformer_model(client: ingrain.Client):
    model = client.load_sentence_transformer_model(name=SENTENCE_TRANSFORMER_MODEL)


@pytest.mark.integration
def test_health(client: ingrain.Client):
    check_server_running(client)
    health_resp = client.health()
    assert len(health_resp) == 2
    assert health_resp[0] == {"message": "The inference server is running."}
    assert health_resp[1] == {"message": "The model server is running."}


@pytest.mark.integration
def test_load_sentence_transformer_model(client: ingrain.Client):
    check_server_running(client)
    model = client.load_sentence_transformer_model(name=SENTENCE_TRANSFORMER_MODEL)
    assert model.name == SENTENCE_TRANSFORMER_MODEL


@pytest.mark.integration
def test_load_loaded_sentence_transformer_model(client: ingrain.Client):
    check_server_running(client)
    load_sentence_transformer_model(client)
    model = client.load_sentence_transformer_model(name=SENTENCE_TRANSFORMER_MODEL)
    assert model.name == SENTENCE_TRANSFORMER_MODEL


@pytest.mark.integration
def test_load_clip_model(client: ingrain.Client):
    check_server_running(client)
    model = client.load_clip_model(name=OPENCLIP_MODEL, pretrained=OPENCLIP_PRETRAINED)
    assert model.name == OPENCLIP_MODEL
    assert model.pretrained == OPENCLIP_PRETRAINED


@pytest.mark.integration
def test_infer_text(client: ingrain.Client):
    check_server_running(client)
    load_sentence_transformer_model(client)
    test_text = "This is a test sentence."
    response = client.infer_text(name=SENTENCE_TRANSFORMER_MODEL, text=test_text)
    assert "embeddings" in response
    assert "processingTimeMs" in response


@pytest.mark.integration
def test_infer_text_from_model(client: ingrain.Client):
    check_server_running(client)
    load_sentence_transformer_model(client)
    test_text = "This is a test sentence."
    model = client.load_sentence_transformer_model(name=SENTENCE_TRANSFORMER_MODEL)
    response = model.infer_text(text=test_text)
    assert "embeddings" in response
    assert "processingTimeMs" in response


@pytest.mark.integration
def test_infer_image(client: ingrain.Client):
    check_server_running(client)
    load_openclip_model(client)
    test_image = "data:image/jpeg;base64,iVBORw0KGgoAAAANSUhEUgAAAOAAAADgCAIAAACVT/22AAACkElEQVR4nOzUMQ0CYRgEUQ5wgwAUnA+EUKKJBkeowAHVJd/kz3sKtpjs9fH9nDjO/npOT1jKeXoA/CNQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKRtl/t7esNSbvs2PWEpHpQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIl7RcAAP//iL8GbQ2nM1wAAAAASUVORK5CYII="
    response = client.infer_image(
        name=OPENCLIP_MODEL, pretrained=OPENCLIP_PRETRAINED, image=test_image
    )
    assert "embeddings" in response
    assert "processingTimeMs" in response


@pytest.mark.integration
def test_infer_image_from_model(client: ingrain.Client):
    check_server_running(client)
    load_openclip_model(client)
    test_image = "data:image/jpeg;base64,iVBORw0KGgoAAAANSUhEUgAAAOAAAADgCAIAAACVT/22AAACkElEQVR4nOzUMQ0CYRgEUQ5wgwAUnA+EUKKJBkeowAHVJd/kz3sKtpjs9fH9nDjO/npOT1jKeXoA/CNQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKRtl/t7esNSbvs2PWEpHpQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIl7RcAAP//iL8GbQ2nM1wAAAAASUVORK5CYII="
    model = client.load_clip_model(name=OPENCLIP_MODEL, pretrained=OPENCLIP_PRETRAINED)
    response = model.infer_image(image=test_image)
    assert "embeddings" in response
    assert "processingTimeMs" in response


@pytest.mark.integration
def test_infer_text_image(client: ingrain.Client):
    check_server_running(client)
    load_openclip_model(client)

    # this image is pink
    test_image = "data:image/jpeg;base64,iVBORw0KGgoAAAANSUhEUgAAAOAAAADgCAIAAACVT/22AAACcklEQVR4nOzSMRHAIADAwF4PbVjFITMOWMnwryBDxp7rg6r/dQDcGJQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaScAAP//3nYDppOW6x0AAAAASUVORK5CYII="
    test_texts = ["A green image", "A pink image"]
    response = client.infer(
        name=OPENCLIP_MODEL,
        pretrained=OPENCLIP_PRETRAINED,
        text=test_texts,
        image=test_image,
    )
    assert "textEmbeddings" in response
    assert "imageEmbeddings" in response
    assert len(response["textEmbeddings"]) == 2

    image_embeddings_arr = np.array(response["imageEmbeddings"])
    text_embeddings_arr = np.array(response["textEmbeddings"])

    image_text_similarities = np.dot(image_embeddings_arr, text_embeddings_arr.T)
    assert image_text_similarities[0, 0] < image_text_similarities[0, 1]


@pytest.mark.integration
def test_infer_text_image_numpy_client(client_numpy: ingrain.Client):
    check_server_running(client_numpy)
    load_openclip_model(client_numpy)

    # this image is pink
    test_image = "data:image/jpeg;base64,iVBORw0KGgoAAAANSUhEUgAAAOAAAADgCAIAAACVT/22AAACcklEQVR4nOzSMRHAIADAwF4PbVjFITMOWMnwryBDxp7rg6r/dQDcGJQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaScAAP//3nYDppOW6x0AAAAASUVORK5CYII="
    test_texts = ["A green image", "A pink image"]
    response = client_numpy.infer(
        name=OPENCLIP_MODEL,
        pretrained=OPENCLIP_PRETRAINED,
        text=test_texts,
        image=test_image,
    )
    assert "textEmbeddings" in response
    assert "imageEmbeddings" in response
    assert len(response["textEmbeddings"]) == 2

    image_embeddings_arr = response["imageEmbeddings"]
    text_embeddings_arr = response["textEmbeddings"]

    assert isinstance(image_embeddings_arr, np.ndarray)
    assert isinstance(text_embeddings_arr, np.ndarray)

    image_text_similarities = np.dot(image_embeddings_arr, text_embeddings_arr.T)
    assert image_text_similarities[0, 0] < image_text_similarities[0, 1]


@pytest.mark.integration
def compare_numpy_and_normal_client(
    client_numpy: ingrain.Client, client: ingrain.Client
):
    check_server_running(client)
    check_server_running(client_numpy)
    load_sentence_transformer_model(client)
    load_sentence_transformer_model(client_numpy)
    test_text = "This is a test sentence."
    normal_model = client.load_sentence_transformer_model(
        name=SENTENCE_TRANSFORMER_MODEL
    )
    normal_response = normal_model.infer_text(text=test_text)

    numpy_model = client_numpy.load_sentence_transformer_model(
        name=SENTENCE_TRANSFORMER_MODEL
    )
    numpy_response = numpy_model.infer_text(text=test_text)

    assert "embeddings" in normal_response
    assert "processingTimeMs" in normal_response
    assert "embeddings" in numpy_response
    assert "processingTimeMs" in numpy_response

    assert np.array_equal(
        np.array(normal_response["embeddings"]), numpy_response["embeddings"]
    )


@pytest.mark.integration
def test_unload_model(client: ingrain.Client):
    check_server_running(client)
    load_sentence_transformer_model(client)
    response = client.unload_model(name=SENTENCE_TRANSFORMER_MODEL)
    assert "unloaded successfully" in response["message"]


@pytest.mark.integration
def test_delete_model(client: ingrain.Client):
    check_server_running(client)
    load_sentence_transformer_model(client)
    response = client.delete_model(name=SENTENCE_TRANSFORMER_MODEL)
    assert "deleted successfully" in response["message"]


@pytest.mark.integration
def test_delete_clip_model(client: ingrain.Client):
    check_server_running(client)
    load_openclip_model(client)
    response = client.delete_model(name=OPENCLIP_MODEL, pretrained=OPENCLIP_PRETRAINED)
    assert "deleted successfully" in response["message"]


@pytest.mark.integration
def test_loaded_models(client: ingrain.Client):
    check_server_running(client)
    load_sentence_transformer_model(client)
    load_openclip_model(client)
    assert "models" in client.loaded_models()


@pytest.mark.integration
def test_repository_models(client: ingrain.Client):
    check_server_running(client)
    assert "models" in client.repository_models()


@pytest.mark.integration
def test_metrics(client: ingrain.Client):
    check_server_running(client)
    assert "modelStats" in client.metrics()
