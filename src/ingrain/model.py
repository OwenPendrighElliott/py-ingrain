from ingrain.pycurl_engine import PyCURLEngine
from ingrain.ingrain_errors import error_factory
from ingrain.models.request_models import (
    InferenceRequest,
    TextInferenceRequest,
    ImageInferenceRequest,
    UnLoadModelRequest,
    LoadModelRequest,
)
from ingrain.models.response_models import (
    InferenceResponse,
    TextInferenceResponse,
    ImageInferenceResponse,
    GenericMessageResponse,
)
from ingrain.utils import make_response_embeddings_numpy
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
        self.library = library
        self.return_numpy = return_numpy

    def __str__(self):
        return f"Model(name={self.name})"

    def __repr__(self):
        return self.__str__()

    def infer_text(
        self, text: Union[List[str], str] = [], normalize: bool = True
    ) -> TextInferenceResponse:
        request = TextInferenceRequest(
            name=self.name,
            text=text,
            normalize=normalize,
        )
        resp, response_code = self.requestor.post(
            f"{self.inference_server_url}/infer_text", request.model_dump()
        )
        if response_code != 200:
            raise error_factory(response_code, resp)

        if self.return_numpy:
            resp = make_response_embeddings_numpy(resp)
        return resp

    def infer_image(
        self, image: Union[List[str], str] = [], normalize: bool = True
    ) -> ImageInferenceResponse:
        request = ImageInferenceRequest(
            name=self.name,
            image=image,
            normalize=normalize,
        )
        resp, response_code = self.requestor.post(
            f"{self.inference_server_url}/infer_image", request.model_dump()
        )
        if response_code != 200:
            raise error_factory(response_code, resp)

        if self.return_numpy:
            resp = make_response_embeddings_numpy(resp)
        return resp

    def infer(
        self,
        text: Optional[Union[List[str], str]] = None,
        image: Optional[Union[List[str], str]] = None,
        normalize: bool = True,
    ) -> InferenceResponse:
        request = InferenceRequest(
            name=self.name,
            text=text,
            image=image,
            normalize=normalize,
        )
        resp, response_code = self.requestor.post(
            f"{self.inference_server_url}/infer", request.model_dump()
        )
        if response_code != 200:
            raise error_factory(response_code, resp)

        if self.return_numpy:
            resp = make_response_embeddings_numpy(resp)
        return resp

    def load(self) -> GenericMessageResponse:
        request = LoadModelRequest(name=self.name, library=self.library)
        resp, response_code = self.requestor.post(
            f"{self.model_server_url}/load_model", request.model_dump()
        )
        if response_code != 200:
            raise error_factory(response_code, resp)
        return resp

    def unload(self) -> GenericMessageResponse:
        request = UnLoadModelRequest(name=self.name)
        resp, response_code = self.requestor.post(
            f"{self.model_server_url}/unload_model", request.model_dump()
        )
        if response_code != 200:
            raise error_factory(response_code, resp)
        return resp

    def delete(self) -> GenericMessageResponse:
        request = UnLoadModelRequest(name=self.name)
        resp, response_code = self.requestor.delete(
            f"{self.model_server_url}/delete_model", request.model_dump()
        )
        if response_code != 200:
            raise error_factory(response_code, resp)
        return resp
