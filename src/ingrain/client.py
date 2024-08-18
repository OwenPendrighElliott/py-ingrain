from ingrain.pycurl_engine import PyCURLEngine
from ingrain.models.models import (
    GenericModelRequest,
    SentenceTransformerModelRequest,
    OpenCLIPModelRequest,
    TextInferenceRequest,
    ImageInferenceRequest,
    InferenceRequest,
)
from ingrain.model import Model
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
    ):
        self.url = url

        self.requestor = PyCURLEngine(
            timeout=timeout,
            connect_timeout=connect_timeout,
            header=header,
            user_agent=user_agent,
        )

    def health(self):
        resp, response_code = self.requestor.get(f"{self.url}/health")
        if response_code != 200:
            raise error_factory(resp)
        return resp

    def loaded_models(self):
        resp, response_code = self.requestor.get(f"{self.url}/loaded_models")
        if response_code != 200:
            raise error_factory(resp)
        return resp

    def repository_models(self):
        resp, response_code = self.requestor.get(f"{self.url}/repository_models")
        if response_code != 200:
            raise error_factory(resp)
        return resp

    def metrics(self):
        resp, response_code = self.requestor.get(f"{self.url}/metrics")
        if response_code != 200:
            raise error_factory(resp)
        return resp

    def load_clip_model(self, name: str, pretrained: Union[str, None] = None):
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

    def load_sentence_transformer_model(self, name: str):
        request = SentenceTransformerModelRequest(name=name)
        resp, response_code = self.requestor.post(
            f"{self.url}/load_sentence_transformer_model", request.model_dump()
        )
        if response_code != 200:
            raise error_factory(resp)
        return Model(requestor=self.requestor, name=name, url=self.url)

    def unload_model(self, name: str, pretrained: Union[str, None] = None):
        request = GenericModelRequest(name=name, pretrained=pretrained)
        resp, response_code = self.requestor.post(
            f"{self.url}/unload_model", request.model_dump()
        )
        if response_code != 200:
            raise error_factory(resp)
        return resp

    def delete_model(self, name: str, pretrained: Union[str, None] = None):
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
    ):
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
        return resp

    def infer_image(
        self,
        name: str,
        pretrained: Union[str, None] = None,
        image: Union[List[str], str] = [],
        normalize: bool = True,
    ):
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
        return resp

    def infer(
        self,
        name: str,
        pretrained: Union[str, None] = None,
        text: Optional[Union[List[str], str]] = None,
        image: Optional[Union[List[str], str]] = None,
        normalize: bool = True,
    ):
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
        return resp
