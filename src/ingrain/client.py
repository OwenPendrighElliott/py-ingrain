from ingrain.pycurl_engine import PyCURLEngine
from ingrain.models.request_models import (
    LoadModelRequest,
    UnloadModelRequest,
    TextEmbeddingRequest,
    ImageEmbeddingRequest,
    EmbeddingRequest,
    ImageClassificationRequest,
    ModelMetadataRequest,
)
from ingrain.models.response_models import (
    EmbeddingResponse,
    TextEmbeddingResponse,
    ImageEmbeddingResponse,
    ImageClassificationResponse,
    LoadedModelResponse,
    RepositoryModelResponse,
    GenericMessageResponse,
    MetricsResponse,
    ModelClassificationLabelsResponse,
    ModelEmbeddingDimsResponse,
)
from ingrain.model import Model
from ingrain.utils import make_response_data_numpy
from ingrain.ingrain_errors import error_factory
from typing import List, Union, Optional, Literal


class Client:
    def __init__(
        self,
        inference_server_url: str = "http://localhost:8686",
        model_server_url: str = "http://localhost:8687",
        timeout: int = 600,
        connect_timeout: int = 600,
        header: List[str] = ["Content-Type: application/json"],
        user_agent: str = "ingrain-client/1.0.0",
        return_numpy: bool = False,
    ):
        self.inference_server_url = inference_server_url
        self.model_server_url = model_server_url
        self.return_numpy = return_numpy

        self.requestor = PyCURLEngine(
            timeout=timeout,
            connect_timeout=connect_timeout,
            header=header,
            user_agent=user_agent,
        )

    def health(self) -> tuple[GenericMessageResponse, GenericMessageResponse]:
        """Check the health of the inference and model servers.

        Raises:
            error_factory: Errors raised when the inference server is not healthy.
            error_factory: Errors raised when the model server is not healthy.

        Returns:
            tuple[GenericMessageResponse, GenericMessageResponse]: Health status of the inference and model servers.
        """
        resp_inf, response_code_inf = self.requestor.get(
            f"{self.inference_server_url}/health"
        )
        resp_model, response_code_model = self.requestor.get(
            f"{self.model_server_url}/health"
        )
        if response_code_inf != 200:
            raise error_factory(response_code_inf, resp_inf)

        if response_code_model != 200:
            raise error_factory(response_code_model, resp_model)

        return (
            GenericMessageResponse.model_construct(**resp_inf),
            GenericMessageResponse.model_construct(**resp_model),
        )

    def loaded_models(self) -> LoadedModelResponse:
        """Get the list of currently loaded models.

        Raises:
            error_factory: Errors raised when the request to get loaded models fails.

        Returns:
            LoadedModelResponse: List of currently loaded models.
        """
        resp, response_code = self.requestor.get(
            f"{self.model_server_url}/loaded_models"
        )
        if response_code != 200:
            raise error_factory(response_code, resp)
        return LoadedModelResponse.model_construct(**resp)

    def repository_models(self) -> RepositoryModelResponse:
        """Get the list of models available in the model repository.

        Raises:
            error_factory: Errors raised when the request to get repository models fails.

        Returns:
            RepositoryModelResponse: List of models available in the model repository.
        """
        resp, response_code = self.requestor.get(
            f"{self.model_server_url}/repository_models"
        )
        if response_code != 200:
            raise error_factory(response_code, resp)
        return RepositoryModelResponse.model_construct(**resp)

    def metrics(self) -> MetricsResponse:
        """Get the metrics of the inference server.

        Raises:
            error_factory: Errors raised when the request to get metrics fails.

        Returns:
            MetricsResponse: Metrics of the inference server.
        """
        resp, response_code = self.requestor.get(f"{self.inference_server_url}/metrics")
        if response_code != 200:
            raise error_factory(response_code, resp)
        return MetricsResponse.model_construct(**resp)

    def load_model(
        self, name: str, library: Literal["open_clip", "sentence_transformers", "timm"]
    ) -> Model:
        """Load a model into ingrain.

        Args:
            name (str): The name of the model to load.
            library (Literal[&quot;open_clip&quot;, &quot;sentence_transformers&quot;, &quot;timm&quot;]): The library the model belongs to, this is used for loading.

        Raises:
            error_factory: Errors raised when the request to load a model fails.

        Returns:
            Model: The loaded model instance.
        """
        request = LoadModelRequest(name=name, library=library)
        resp, response_code = self.requestor.post(
            f"{self.model_server_url}/load_model", request.model_dump()
        )
        if response_code != 200:
            raise error_factory(response_code, resp)

        return Model(
            requestor=self.requestor,
            name=name,
            library=library,
            inference_server_url=self.inference_server_url,
            model_server_url=self.model_server_url,
        )

    def unload_model(self, name: str) -> GenericMessageResponse:
        """Unload a model from ingrain.

        Args:
            name (str): The name of the model to unload.

        Raises:
            error_factory: Errors raised when the request to unload a model fails.

        Returns:
            GenericMessageResponse: Response message indicating the result of the unload operation.
        """
        request = UnloadModelRequest(name=name)
        resp, response_code = self.requestor.post(
            f"{self.model_server_url}/unload_model", request.model_dump()
        )
        if response_code != 200:
            raise error_factory(response_code, resp)
        return GenericMessageResponse.model_construct(**resp)

    def delete_model(self, name: str) -> GenericMessageResponse:
        """Delete a model from the model repository.

        Args:
            name (str): The name of the model to delete.

        Raises:
            error_factory: Errors raised when the request to delete a model fails.

        Returns:
            GenericMessageResponse: Response message indicating the result of the delete operation.
        """
        request = UnloadModelRequest(name=name)
        resp, response_code = self.requestor.delete(
            f"{self.model_server_url}/delete_model", request.model_dump()
        )
        if response_code != 200:
            raise error_factory(response_code, resp)
        return GenericMessageResponse.model_construct(**resp)

    def embed_text(
        self,
        name: str,
        text: Union[List[str], str] = [],
        normalize: bool = True,
        n_dims: Optional[int] = None,
        retries: int = 0,
    ) -> TextEmbeddingResponse:
        """Get text embeddings for the given text using the specified model.

        Args:
            name (str): The name of the model to use for embedding.
            text (Union[List[str], str], optional): A single string or list of strings to embed. Defaults to [].
            normalize (bool, optional): Whether or not normalisation should be applied (unit norm on vectors), useful for downstream calculations. Defaults to True.
            n_dims (Optional[int], optional): The number of dimensions to return, only useful for MRL models. Defaults to None.
            retries (int, optional): The number of retries to perform. Defaults to 0.

        Raises:
            error_factory: Errors raised when the request to get text embeddings fails.

        Returns:
            TextEmbeddingResponse: Embeddings and processing time for the given text.
        """
        request = TextEmbeddingRequest(
            name=name, text=text, normalize=normalize, n_dims=n_dims
        )
        resp, response_code = self.requestor.post(
            f"{self.inference_server_url}/embed_text",
            request.model_dump(),
            retries=retries,
        )
        if response_code != 200:
            raise error_factory(response_code, resp)

        if self.return_numpy:
            resp = make_response_data_numpy(resp)

        return TextEmbeddingResponse.model_construct(**resp)

    def embed_image(
        self,
        name: str,
        image: Union[List[str], str] = [],
        normalize: bool = True,
        n_dims: Optional[int] = None,
        image_download_headers: Optional[dict[str, str]] = None,
        retries: int = 0,
    ) -> ImageEmbeddingResponse:
        """Get image embeddings for the given image(s) using the specified model.

        Args:
            name (str): The name of the model to use for embedding.
            image (Union[List[str], str], optional): A single image URL or list of image URLs to embed. Defaults to [].
            normalize (bool, optional): Whether or not normalisation should be applied (unit norm on vectors), useful for downstream calculations. Defaults to True.
            n_dims (Optional[int], optional): The number of dimensions to return, only useful for MRL models. Defaults to None.
            image_download_headers (Optional[dict[str, str]], optional): Optional headers to include when downloading images. Defaults to None.
            retries (int, optional): The number of retries to perform. Defaults to 0.

        Raises:
            error_factory: Errors raised when the request to get image embeddings fails.

        Returns:
            ImageEmbeddingResponse: Embeddings and processing time for the given image(s).
        """

        request = ImageEmbeddingRequest(
            name=name,
            image=image,
            normalize=normalize,
            n_dims=n_dims,
            image_download_headers=image_download_headers,
        )
        resp, response_code = self.requestor.post(
            f"{self.inference_server_url}/embed_image",
            request.model_dump(),
            retries=retries,
        )
        if response_code != 200:
            raise error_factory(response_code, resp)

        if self.return_numpy:
            resp = make_response_data_numpy(resp)
        return ImageEmbeddingResponse.model_construct(**resp)

    def embed(
        self,
        name: str,
        text: Optional[Union[List[str], str]] = None,
        image: Optional[Union[List[str], str]] = None,
        normalize: bool = True,
        n_dims: Optional[int] = None,
        image_download_headers: Optional[dict[str, str]] = None,
        retries: int = 0,
    ) -> EmbeddingResponse:
        """Get embeddings for the given text and/or image(s) using the specified model.

        Args:
            name (str): The name of the model to use for embedding.
            text (Optional[Union[List[str], str]], optional): A single string or list of strings to embed. Defaults to None.
            image (Optional[Union[List[str], str]], optional): A single image URL or list of image URLs to embed. Defaults to None.
            normalize (bool, optional): Whether or not normalisation should be applied (unit norm on vectors), useful for downstream calculations. Defaults to True.
            n_dims (Optional[int], optional): The number of dimensions to return, only useful for MRL models. Defaults to None.
            image_download_headers (Optional[dict[str, str]], optional): Optional headers to include when downloading images. Defaults to None.
            retries (int, optional): The number of retries to perform. Defaults to 0.

        Raises:
            error_factory: Errors raised when the request to get embeddings fails.

        Returns:
            EmbeddingResponse: Embeddings and processing time for the given text and/or image(s).
        """

        request = EmbeddingRequest(
            name=name,
            text=text,
            image=image,
            normalize=normalize,
            n_dims=n_dims,
            image_download_headers=image_download_headers,
        )
        resp, response_code = self.requestor.post(
            f"{self.inference_server_url}/embed", request.model_dump(), retries=retries
        )
        if response_code != 200:
            raise error_factory(response_code, resp)

        if self.return_numpy:
            resp = make_response_data_numpy(resp)
        return EmbeddingResponse.model_construct(**resp)

    def classify_image(
        self,
        name: str,
        image: Union[List[str], str] = [],
        image_download_headers: Optional[dict[str, str]] = None,
        retries: int = 0,
    ) -> ImageClassificationResponse:
        """Classify the given image(s) using the specified model.

        Args:
            name (str): The name of the model to use for classification.
            image (Union[List[str], str], optional): A single image URL or list of image URLs to classify. Defaults to [].
            image_download_headers (Optional[dict[str, str]], optional): Optional headers to include when downloading images. Defaults to None.
            retries (int, optional): The number of retries to perform. Defaults to 0.

        Raises:
            error_factory: Errors raised when the request to classify images fails.

        Returns:
            ImageClassificationResponse: Classification probabilities and processing time for the given image(s).
        """

        request = ImageClassificationRequest(
            name=name,
            image=image,
            image_download_headers=image_download_headers,
        )
        resp, response_code = self.requestor.post(
            f"{self.inference_server_url}/classify_image",
            request.model_dump(),
            retries=retries,
        )
        if response_code != 200:
            raise error_factory(response_code, resp)

        if self.return_numpy:
            resp = make_response_data_numpy(resp)
        return ImageClassificationResponse.model_construct(**resp)

    def model_classification_labels(
        self,
        name: str,
        retries: int = 0,
    ) -> ModelClassificationLabelsResponse:
        """Get the classification labels for the specified model.

        Args:
            name (str): The name of the model to get classification labels for.
            retries (int, optional): The number of retries to perform. Defaults to 0.

        Raises:
            error_factory: Errors raised when the request to get classification labels fails.

        Returns:
            ModelClassificationLabelsResponse: Classification labels for the specified model.
        """
        req = ModelMetadataRequest(name=name)
        resp, response_code = self.requestor.post(
            f"{self.model_server_url}/model_classification_labels",
            req.model_dump(),
            retries=retries,
        )

        if response_code != 200:
            raise error_factory(response_code, resp)

        return ModelClassificationLabelsResponse.model_construct(**resp)

    def model_embedding_dims(
        self,
        name: str,
        retries: int = 0,
    ) -> ModelEmbeddingDimsResponse:
        """Get the embedding dimensions for the specified model.

        Args:
            name (str): The name of the model to get embedding dimensions for.
            retries (int, optional): The number of retries to perform. Defaults to 0.

        Raises:
            error_factory: Errors raised when the request to get embedding dimensions fails.

        Returns:
            ModelEmbeddingDimsResponse: Embedding dimensions for the specified model.
        """
        req = ModelMetadataRequest(name=name)
        resp, response_code = self.requestor.post(
            f"{self.model_server_url}/model_embedding_size",
            req.model_dump(),
            retries=retries,
        )

        if response_code != 200:
            raise error_factory(response_code, resp)

        return ModelEmbeddingDimsResponse.model_construct(**resp)
