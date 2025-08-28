from ingrain.pycurl_engine import PyCURLEngine
from ingrain.ingrain_errors import error_factory
from ingrain.models.request_models import (
    EmbeddingRequest,
    TextEmbeddingRequest,
    ImageEmbeddingRequest,
    ImageClassificationRequest,
    LoadModelRequest,
    UnloadModelRequest,
)
from ingrain.models.response_models import (
    EmbeddingResponse,
    TextEmbeddingResponse,
    ImageEmbeddingResponse,
    ImageClassificationResponse,
    GenericMessageResponse,
)
from ingrain.utils import make_response_data_numpy
from typing import Optional, Union, List, Literal


class Model:
    def __init__(
        self,
        requestor: PyCURLEngine,
        name: str,
        library: Literal["open_clip", "sentence_transformers", "timm"],
        inference_server_url: str = "http://localhost:8686",
        model_server_url: str = "http://localhost:8687",
        return_numpy: bool = False,
    ):
        self.requestor = requestor
        self.inference_server_url = inference_server_url
        self.model_server_url = model_server_url
        self.name = name
        self.library: Literal["open_clip", "sentence_transformers", "timm"] = library
        self.return_numpy = return_numpy

    def __str__(self):
        return f"Model(name={self.name})"

    def __repr__(self):
        return self.__str__()

    def embed_text(
        self,
        text: Union[List[str], str] = [],
        normalize: bool = True,
        n_dims: Optional[int] = None,
        retries: int = 0,
    ) -> TextEmbeddingResponse:
        request = TextEmbeddingRequest(
            name=self.name, text=text, normalize=normalize, n_dims=n_dims
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
        image: Union[List[str], str] = [],
        normalize: bool = True,
        n_dims: Optional[int] = None,
        image_download_headers: Optional[dict[str, str]] = None,
        retries: int = 0,
    ) -> ImageEmbeddingResponse:
        request = ImageEmbeddingRequest(
            name=self.name,
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
        text: Optional[Union[List[str], str]] = None,
        image: Optional[Union[List[str], str]] = None,
        normalize: bool = True,
        n_dims: Optional[int] = None,
        image_download_headers: Optional[dict[str, str]] = None,
        retries: int = 0,
    ) -> EmbeddingResponse:
        request = EmbeddingRequest(
            name=self.name,
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
        image: Union[List[str], str] = [],
        image_download_headers: Optional[dict[str, str]] = None,
        retries: int = 0,
    ) -> ImageClassificationResponse:
        request = ImageClassificationRequest(
            name=self.name,
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

    def load_(self) -> GenericMessageResponse:
        request = LoadModelRequest(name=self.name, library=self.library)
        resp, response_code = self.requestor.post(
            f"{self.model_server_url}/load_model", request.model_dump()
        )
        if response_code != 200:
            raise error_factory(response_code, resp)

        return GenericMessageResponse.model_construct(**resp)

    def unload(self) -> GenericMessageResponse:
        request = UnloadModelRequest(name=self.name)
        resp, response_code = self.requestor.post(
            f"{self.model_server_url}/unload_model", request.model_dump()
        )
        if response_code != 200:
            raise error_factory(response_code, resp)
        return GenericMessageResponse.model_construct(**resp)

    def delete(self) -> GenericMessageResponse:
        request = UnloadModelRequest(name=self.name)
        resp, response_code = self.requestor.delete(
            f"{self.model_server_url}/delete_model", request.model_dump()
        )
        if response_code != 200:
            raise error_factory(response_code, resp)
        return GenericMessageResponse.model_construct(**resp)
