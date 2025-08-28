from ingrain.models.camel_model import CamelModel

from typing import List, Optional, Union, Literal


class EmbeddingRequest(CamelModel):
    name: str
    text: Optional[Union[str, List[str]]] = None
    image: Optional[Union[str, List[str]]] = None
    normalize: Optional[bool] = True
    n_dims: Optional[int] = None
    image_download_headers: Optional[dict] = None


class TextEmbeddingRequest(CamelModel):
    name: str
    text: Union[str, List[str]]
    normalize: Optional[bool] = True
    n_dims: Optional[int] = None


class ImageEmbeddingRequest(CamelModel):
    name: str
    image: Union[str, List[str]]
    normalize: Optional[bool] = True
    n_dims: Optional[int] = None
    image_download_headers: Optional[dict] = None


class ImageClassificationRequest(CamelModel):
    name: str
    image: Union[str, List[str]]
    image_download_headers: Optional[dict] = None


class LoadModelRequest(CamelModel):
    name: str
    library: Literal["open_clip", "sentence_transformers", "timm"]


class UnloadModelRequest(CamelModel):
    name: str


class ModelMetadataRequest(CamelModel):
    name: str
