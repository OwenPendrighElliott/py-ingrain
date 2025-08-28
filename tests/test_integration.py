import ingrain.ingrain_errors
import pytest
import ingrain
import numpy as np

INFERENCE_BASE_URL = "http://127.0.0.1:8686"
MODEL_BASE_URL = "http://127.0.0.1:8687"

# test models
SENTENCE_TRANSFORMER_MODEL = "intfloat/e5-small-v2"
OPENCLIP_MODEL = "hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K"


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
    _ = client.health()


def load_openclip_model(client: ingrain.Client):
    _ = client.load_model(name=OPENCLIP_MODEL, library="open_clip")


def load_sentence_transformer_model(client: ingrain.Client):
    _ = client.load_model(
        name=SENTENCE_TRANSFORMER_MODEL, library="sentence_transformers"
    )


def unload_all_models(client: ingrain.Client):
    _ = client.unload_model(name=OPENCLIP_MODEL)
    _ = client.unload_model(name=SENTENCE_TRANSFORMER_MODEL)


@pytest.mark.integration
def test_health(client: ingrain.Client):
    check_server_running(client)
    health_resp = client.health()
    assert len(health_resp) == 2
    assert health_resp[0].message == "The inference server is running."
    assert health_resp[1].message == "The model server is running."


@pytest.mark.integration
def test_load_sentence_transformer_model(client: ingrain.Client):
    check_server_running(client)
    model = client.load_model(
        name=SENTENCE_TRANSFORMER_MODEL, library="sentence_transformers"
    )
    assert model.name == SENTENCE_TRANSFORMER_MODEL


@pytest.mark.integration
def test_load_timm_model(client: ingrain.Client):
    check_server_running(client)
    model = client.load_model(
        name="hf_hub:timm/mobilenetv4_conv_medium.e250_r384_in12k_ft_in1k",
        library="timm",
    )
    assert model.name == "hf_hub:timm/mobilenetv4_conv_medium.e250_r384_in12k_ft_in1k"


@pytest.mark.integration
def test_load_loaded_sentence_transformer_model(client: ingrain.Client):
    check_server_running(client)
    load_sentence_transformer_model(client)
    model = client.load_model(
        name=SENTENCE_TRANSFORMER_MODEL, library="sentence_transformers"
    )
    assert model.name == SENTENCE_TRANSFORMER_MODEL


@pytest.mark.integration
def test_load_clip_model(client: ingrain.Client):
    check_server_running(client)
    model = client.load_model(name=OPENCLIP_MODEL, library="open_clip")
    assert model.name == OPENCLIP_MODEL


@pytest.mark.integration
def test_embed_text(client: ingrain.Client):
    check_server_running(client)
    load_sentence_transformer_model(client)
    test_text = "This is a test sentence."
    response = client.embed_text(name=SENTENCE_TRANSFORMER_MODEL, text=test_text)
    assert isinstance(response.embeddings, list)
    assert isinstance(response.embeddings[0], list)
    assert len(response.embeddings[0]) == 384
    assert isinstance(response.processing_time_ms, float)


@pytest.mark.integration
def test_embed_text_from_model(client: ingrain.Client):
    check_server_running(client)
    load_sentence_transformer_model(client)
    test_text = "This is a test sentence."
    model = client.load_model(
        name=SENTENCE_TRANSFORMER_MODEL, library="sentence_transformers"
    )
    response = model.embed_text(text=test_text)
    assert isinstance(response.embeddings, list)
    assert isinstance(response.embeddings[0], list)
    assert len(response.embeddings[0]) == 384  # e5-small-v2 has 384 dims
    assert isinstance(response.processing_time_ms, float)


@pytest.mark.integration
def test_embed_image(client: ingrain.Client):
    check_server_running(client)
    load_openclip_model(client)
    test_image = "data:image/jpeg;base64,iVBORw0KGgoAAAANSUhEUgAAAOAAAADgCAIAAACVT/22AAACkElEQVR4nOzUMQ0CYRgEUQ5wgwAUnA+EUKKJBkeowAHVJd/kz3sKtpjs9fH9nDjO/npOT1jKeXoA/CNQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKRtl/t7esNSbvs2PWEpHpQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIl7RcAAP//iL8GbQ2nM1wAAAAASUVORK5CYII="
    response = client.embed_image(name=OPENCLIP_MODEL, image=test_image)
    assert isinstance(response.embeddings, list)
    assert isinstance(response.embeddings[0], list)
    assert len(response.embeddings[0]) == 512


@pytest.mark.integration
def test_embed_image_from_model(client: ingrain.Client):
    check_server_running(client)
    load_openclip_model(client)
    test_image = "data:image/jpeg;base64,iVBORw0KGgoAAAANSUhEUgAAAOAAAADgCAIAAACVT/22AAACkElEQVR4nOzUMQ0CYRgEUQ5wgwAUnA+EUKKJBkeowAHVJd/kz3sKtpjs9fH9nDjO/npOT1jKeXoA/CNQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKRtl/t7esNSbvs2PWEpHpQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIl7RcAAP//iL8GbQ2nM1wAAAAASUVORK5CYII="
    model = client.load_model(name=OPENCLIP_MODEL, library="open_clip")
    response = model.embed_image(image=test_image)
    assert isinstance(response.embeddings, list)
    assert isinstance(response.embeddings[0], list)
    assert len(response.embeddings[0]) == 512


@pytest.mark.integration
def test_embed_text_image(client: ingrain.Client):
    check_server_running(client)
    load_openclip_model(client)

    # this image is pink
    test_image = "data:image/jpeg;base64,iVBORw0KGgoAAAANSUhEUgAAAOAAAADgCAIAAACVT/22AAACcklEQVR4nOzSMRHAIADAwF4PbVjFITMOWMnwryBDxp7rg6r/dQDcGJQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaScAAP//3nYDppOW6x0AAAAASUVORK5CYII="
    test_texts = ["A green image", "A pink image"]
    response = client.embed(
        name=OPENCLIP_MODEL,
        text=test_texts,
        image=test_image,
    )
    assert isinstance(response.text_embeddings, list)
    assert isinstance(response.image_embeddings, list)
    assert len(response.text_embeddings) == 2
    assert len(response.image_embeddings) == 1
    assert len(response.text_embeddings[0]) == 512
    assert len(response.image_embeddings[0]) == 512
    assert isinstance(response.processing_time_ms, float)

    image_embeddings_arr = np.array(response.image_embeddings)
    text_embeddings_arr = np.array(response.text_embeddings)

    image_text_similarities = np.dot(image_embeddings_arr, text_embeddings_arr.T)
    assert image_text_similarities[0, 0] < image_text_similarities[0, 1]


@pytest.mark.integration
def test_embed_text_image_numpy_client(client_numpy: ingrain.Client):
    check_server_running(client_numpy)
    load_openclip_model(client_numpy)

    # this image is pink
    test_image = "data:image/jpeg;base64,iVBORw0KGgoAAAANSUhEUgAAAOAAAADgCAIAAACVT/22AAACcklEQVR4nOzSMRHAIADAwF4PbVjFITMOWMnwryBDxp7rg6r/dQDcGJQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaScAAP//3nYDppOW6x0AAAAASUVORK5CYII="
    test_texts = ["A green image", "A pink image"]
    response = client_numpy.embed(
        name=OPENCLIP_MODEL,
        text=test_texts,
        image=test_image,
    )
    assert isinstance(response.text_embeddings, np.ndarray)
    assert isinstance(response.image_embeddings, np.ndarray)
    assert response.text_embeddings.shape == (2, 512)
    assert response.image_embeddings.shape == (1, 512)
    assert isinstance(response.processing_time_ms, float)

    image_embeddings_arr = response.image_embeddings
    text_embeddings_arr = response.text_embeddings

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
    normal_model = client.load_model(
        name=SENTENCE_TRANSFORMER_MODEL, library="sentence_transformers"
    )
    normal_response = normal_model.embed_text(text=test_text)

    numpy_model = client_numpy.load_model(
        name=SENTENCE_TRANSFORMER_MODEL, library="sentence_transformers"
    )
    numpy_response = numpy_model.embed_text(text=test_text)

    assert isinstance(normal_response.embeddings, list)
    assert isinstance(normal_response.embeddings[0], list)
    assert isinstance(numpy_response.embeddings, np.ndarray)
    assert numpy_response.embeddings.shape == (1, 384)
    assert normal_response.processing_time_ms == numpy_response.processing_time_ms

    assert np.array_equal(
        np.array(normal_response.embeddings), numpy_response.embeddings
    )


@pytest.mark.integration
def test_unload_model(client: ingrain.Client):
    check_server_running(client)
    load_sentence_transformer_model(client)
    response = client.unload_model(name=SENTENCE_TRANSFORMER_MODEL)
    assert "unloaded successfully" in response.message


@pytest.mark.integration
def test_delete_model(client: ingrain.Client):
    check_server_running(client)
    load_sentence_transformer_model(client)
    response = client.delete_model(name=SENTENCE_TRANSFORMER_MODEL)
    assert "deleted successfully" in response.message


@pytest.mark.integration
def test_delete_clip_model(client: ingrain.Client):
    check_server_running(client)
    load_openclip_model(client)
    response = client.delete_model(name=OPENCLIP_MODEL)
    assert "deleted successfully" in response.message


@pytest.mark.integration
def test_loaded_models(client: ingrain.Client):
    check_server_running(client)
    load_sentence_transformer_model(client)
    load_openclip_model(client)
    assert len(client.loaded_models().models) >= 2


@pytest.mark.integration
def test_repository_models(client: ingrain.Client):
    check_server_running(client)
    assert len(client.repository_models().models) >= 2


@pytest.mark.integration
def test_unload_all_models(client: ingrain.Client):
    check_server_running(client)
    load_sentence_transformer_model(client)
    load_openclip_model(client)
    before = len(client.loaded_models().models)
    print(client.loaded_models().models)
    unload_all_models(client)
    after = len(client.loaded_models().models)
    print(client.loaded_models().models)
    assert after == before - 2
