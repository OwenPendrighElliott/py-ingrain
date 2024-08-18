from ingrain.pycurl_engine import PyCURLEngine
from ingrain.models.request_models import (
    GenericModelRequest,
    SentenceTransformerModelRequest,
    OpenCLIPModelRequest,
    TextInferenceRequest,
    ImageInferenceRequest,
    InferenceRequest,
)
from ingrain.models.response_models import (
    InferenceResponse,
    TextInferenceResponse,
    ImageInferenceResponse,
    LoadedModelResponse,
    RepositoryModelResponse,
    GenericMessageResponse,
    MetricsResponse,
)
from ingrain.model import Model
from ingrain.utils import make_response_embeddings_numpy
from ingrain.ingrain_errors import error_factory
from typing import List, Union, Optional


class Client:
    def __init__(
        self,
        url="http://localhost:8686",
        timeout: int = 600,
        connect_timeout: int = 600,
        header: List[str] = ["Content-Type: application/json"],
        user_agent: str = "ingrain-client/1.0.0",
        return_numpy: bool = False,
    ):
        self.url = url
        self.return_numpy = return_numpy

        self.requestor = PyCURLEngine(
            timeout=timeout,
            connect_timeout=connect_timeout,
            header=header,
            user_agent=user_agent,
        )

    def health(self) -> GenericMessageResponse:
        resp, response_code = self.requestor.get(f"{self.url}/health")
        if response_code != 200:
            raise error_factory(resp)
        return resp

    def loaded_models(self) -> LoadedModelResponse:
        resp, response_code = self.requestor.get(f"{self.url}/loaded_models")
        if response_code != 200:
            raise error_factory(resp)
        return resp

    def repository_models(self) -> RepositoryModelResponse:
        resp, response_code = self.requestor.get(f"{self.url}/repository_models")
        if response_code != 200:
            raise error_factory(resp)
        return resp

    def metrics(self) -> MetricsResponse:
        resp, response_code = self.requestor.get(f"{self.url}/metrics")
        if response_code != 200:
            raise error_factory(resp)
        return resp

    def load_clip_model(self, name: str, pretrained: Union[str, None] = None) -> Model:
        request = OpenCLIPModelRequest(name=name, pretrained=pretrained)
        resp, response_code = self.requestor.post(
            f"{self.url}/load_clip_model", request.model_dump()
        )
        if response_code != 200:
            raise error_factory(resp)
        return Model(
            requestor=self.requestor,
            name=name,
            pretrained=pretrained,
            url=self.url,
        )

    def load_sentence_transformer_model(self, name: str) -> Model:
        request = SentenceTransformerModelRequest(name=name)
        resp, response_code = self.requestor.post(
            f"{self.url}/load_sentence_transformer_model", request.model_dump()
        )
        if response_code != 200:
            raise error_factory(resp)
        return Model(requestor=self.requestor, name=name, url=self.url)

    def unload_model(
        self, name: str, pretrained: Union[str, None] = None
    ) -> GenericMessageResponse:
        request = GenericModelRequest(name=name, pretrained=pretrained)
        resp, response_code = self.requestor.post(
            f"{self.url}/unload_model", request.model_dump()
        )
        if response_code != 200:
            raise error_factory(resp)
        return resp

    def delete_model(
        self, name: str, pretrained: Union[str, None] = None
    ) -> GenericMessageResponse:
        request = GenericModelRequest(name=name, pretrained=pretrained)
        resp, response_code = self.requestor.post(
            f"{self.url}/delete_model", request.model_dump()
        )
        if response_code != 200:
            raise error_factory(resp)
        return resp

    def infer_text(
        self,
        name: str,
        pretrained: Union[str, None] = None,
        text: Union[List[str], str] = [],
        normalize: bool = True,
    ) -> TextInferenceResponse:
        request = TextInferenceRequest(
            name=name,
            text=text,
            pretrained=pretrained,
            normalize=normalize,
        )
        resp, response_code = self.requestor.post(
            f"{self.url}/infer_text", request.model_dump()
        )
        if response_code != 200:
            raise error_factory(response_code, resp)

        if self.return_numpy:
            resp = make_response_embeddings_numpy(resp)
        return resp

    def infer_image(
        self,
        name: str,
        pretrained: Union[str, None] = None,
        image: Union[List[str], str] = [],
        normalize: bool = True,
    ) -> ImageInferenceResponse:
        request = ImageInferenceRequest(
            name=name,
            image=image,
            pretrained=pretrained,
            normalize=normalize,
        )
        resp, response_code = self.requestor.post(
            f"{self.url}/infer_image", request.model_dump()
        )
        if response_code != 200:
            raise error_factory(response_code, resp)

        if self.return_numpy:
            resp = make_response_embeddings_numpy(resp)
        return resp

    def infer(
        self,
        name: str,
        pretrained: Union[str, None] = None,
        text: Optional[Union[List[str], str]] = None,
        image: Optional[Union[List[str], str]] = None,
        normalize: bool = True,
    ) -> InferenceResponse:
        request = InferenceRequest(
            name=name,
            text=text,
            image=image,
            pretrained=pretrained,
            normalize=normalize,
        )
        resp, response_code = self.requestor.post(
            f"{self.url}/infer", request.model_dump()
        )
        if response_code != 200:
            raise error_factory(response_code, resp)

        if self.return_numpy:
            resp = make_response_embeddings_numpy(resp)
        return resp
