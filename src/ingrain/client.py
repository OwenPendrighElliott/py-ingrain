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
        inference_server_url="http://localhost:8686",
        model_server_url="http://localhost:8687",
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

    def health(self) -> GenericMessageResponse:
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
        return [resp_inf, resp_model]

    def loaded_models(self) -> LoadedModelResponse:
        resp, response_code = self.requestor.get(
            f"{self.model_server_url}/loaded_models"
        )
        if response_code != 200:
            raise error_factory(response_code, resp)
        return resp

    def repository_models(self) -> RepositoryModelResponse:
        resp, response_code = self.requestor.get(
            f"{self.model_server_url}/repository_models"
        )
        if response_code != 200:
            raise error_factory(response_code, resp)
        return resp

    def metrics(self) -> MetricsResponse:
        resp, response_code = self.requestor.get(f"{self.inference_server_url}/metrics")
        if response_code != 200:
            raise error_factory(response_code, resp)
        return resp

    def load_clip_model(self, name: str, pretrained: Union[str, None] = None) -> Model:
        request = OpenCLIPModelRequest(name=name, pretrained=pretrained)
        resp, response_code = self.requestor.post(
            f"{self.model_server_url}/load_clip_model", request.model_dump()
        )
        if response_code != 200:
            raise error_factory(response_code, resp)
        return Model(
            requestor=self.requestor,
            name=name,
            pretrained=pretrained,
            inference_server_url=self.inference_server_url,
            model_server_url=self.model_server_url,
        )

    def load_sentence_transformer_model(self, name: str) -> Model:
        request = SentenceTransformerModelRequest(name=name)
        resp, response_code = self.requestor.post(
            f"{self.model_server_url}/load_sentence_transformer_model",
            request.model_dump(),
        )
        if response_code != 200:
            raise error_factory(response_code, resp)
        return Model(
            requestor=self.requestor,
            name=name,
            inference_server_url=self.inference_server_url,
            model_server_url=self.model_server_url,
        )

    def unload_model(
        self, name: str, pretrained: Union[str, None] = None
    ) -> GenericMessageResponse:
        request = GenericModelRequest(name=name, pretrained=pretrained)
        resp, response_code = self.requestor.post(
            f"{self.model_server_url}/unload_model", request.model_dump()
        )
        if response_code != 200:
            raise error_factory(response_code, resp)
        return resp

    def delete_model(
        self, name: str, pretrained: Union[str, None] = None
    ) -> GenericMessageResponse:
        request = GenericModelRequest(name=name, pretrained=pretrained)
        resp, response_code = self.requestor.post(
            f"{self.model_server_url}/delete_model", request.model_dump()
        )
        if response_code != 200:
            raise error_factory(response_code, resp)
        return resp

    def infer_text(
        self,
        name: str,
        pretrained: Union[str, None] = None,
        text: Union[List[str], str] = [],
        normalize: bool = True,
        retries: int = 0,
    ) -> TextInferenceResponse:
        request = TextInferenceRequest(
            name=name,
            text=text,
            pretrained=pretrained,
            normalize=normalize,
        )
        resp, response_code = self.requestor.post(
            f"{self.inference_server_url}/infer_text",
            request.model_dump(),
            retries=retries,
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
        retries: int = 0,
    ) -> ImageInferenceResponse:
        request = ImageInferenceRequest(
            name=name,
            image=image,
            pretrained=pretrained,
            normalize=normalize,
        )
        resp, response_code = self.requestor.post(
            f"{self.inference_server_url}/infer_image",
            request.model_dump(),
            retries=retries,
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
        retries: int = 0,
    ) -> InferenceResponse:
        request = InferenceRequest(
            name=name,
            text=text,
            image=image,
            pretrained=pretrained,
            normalize=normalize,
        )
        resp, response_code = self.requestor.post(
            f"{self.inference_server_url}/infer", request.model_dump(), retries=retries
        )
        if response_code != 200:
            raise error_factory(response_code, resp)

        if self.return_numpy:
            resp = make_response_embeddings_numpy(resp)
        return resp
