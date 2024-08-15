from ingrain.pycurl_engine import PyCURLEngine
from ingrain.ingrain_errors import error_factory
from ingrain.models.models import (
    InferenceRequest,
    TextInferenceRequest,
    ImageInferenceRequest,
    GenericModelRequest,
)
from typing import Optional, Union, List


class Model:
    def __init__(
        self,
        requestor: PyCURLEngine,
        name: str,
        pretrained: Optional[str] = None,
        url: str = "http://localhost:8686",
    ):
        self.requestor = requestor
        self.url = url
        self.name = name
        self.pretrained = pretrained

    def __str__(self):
        return f"Model(name={self.name}, pretrained={self.pretrained})"

    def __repr__(self):
        return self.__str__()

    def infer_text(self, text: Union[List[str], str] = [], normalize: bool = True):
        request = TextInferenceRequest(
            name=self.name,
            text=text,
            pretrained=self.pretrained,
            normalize=normalize,
        )
        resp, response_code = self.requestor.post(
            f"{self.url}/infer_text", request.model_dump()
        )
        if response_code != 200:
            raise error_factory(resp)
        return resp

    def infer_image(self, image: Union[List[str], str] = [], normalize: bool = True):
        request = ImageInferenceRequest(
            name=self.name,
            image=image,
            pretrained=self.pretrained,
            normalize=normalize,
        )
        resp, response_code = self.requestor.post(
            f"{self.url}/infer_image", request.model_dump()
        )
        if response_code != 200:
            raise error_factory(resp)
        return resp

    def infer(
        self,
        text: Optional[Union[List[str], str]] = None,
        image: Optional[Union[List[str], str]] = None,
        normalize: bool = True,
    ):
        request = InferenceRequest(
            name=self.name,
            text=text,
            image=image,
            pretrained=self.pretrained,
            normalize=normalize,
        )
        resp, response_code = self.requestor.post(
            f"{self.url}/infer", request.model_dump()
        )
        if response_code != 200:
            raise error_factory(resp)
        return resp

    def unload(self):
        request = GenericModelRequest(name=self.name, pretrained=self.pretrained)
        resp, response_code = self.requestor.post(
            f"{self.url}/unload_model", request.model_dump()
        )
        if response_code != 200:
            raise error_factory(resp)
        return resp

    def delete(self):
        request = GenericModelRequest(name=self.name, pretrained=self.pretrained)
        resp, response_code = self.requestor.post(
            f"{self.url}/delete_model", request.model_dump()
        )
        if response_code != 200:
            raise error_factory(resp)
        return resp
