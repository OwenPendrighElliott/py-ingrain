from pydantic import BaseModel

from typing import List, Optional, Union, Literal


class InferenceRequest(BaseModel):
    name: str
    text: Optional[Union[str, List[str]]] = None
    image: Optional[Union[str, List[str]]] = None
    normalize: Optional[bool] = True


class TextInferenceRequest(BaseModel):
    name: str
    text: Union[str, List[str]]
    normalize: Optional[bool] = True


class ImageInferenceRequest(BaseModel):
    name: str
    image: Union[str, List[str]]
    normalize: Optional[bool] = True


class LoadModelRequest(BaseModel):
    name: str
    library: Literal["open_clip", "sentence_transformers", "timm"]


class UnLoadModelRequest(BaseModel):
    name: str
