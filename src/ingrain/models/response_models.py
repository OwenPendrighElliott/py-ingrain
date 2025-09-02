from ingrain.models.camel_model import CamelModel
import numpy as np

from typing import List, Optional, Dict


class EmbeddingResponse(CamelModel):
    text_embeddings: Optional[List[List[float]] | np.ndarray] = None
    image_embeddings: Optional[List[List[float]] | np.ndarray] = None
    processing_time_ms: float


class TextEmbeddingResponse(CamelModel):
    embeddings: List[List[float]] | np.ndarray
    processing_time_ms: float


class ImageEmbeddingResponse(CamelModel):
    embeddings: List[List[float]] | np.ndarray
    processing_time_ms: float


class ImageClassificationResponse(CamelModel):
    probabilities: List[List[float]] | np.ndarray
    processing_time_ms: float



class LoadedModelData(CamelModel):
    name: str
    library: str


class LoadedModelResponse(CamelModel):
    models: List[LoadedModelData]

class RepositoryModel(CamelModel):
    name: str
    state: str


class RepositoryModelResponse(CamelModel):
    models: List[RepositoryModel]


class GenericMessageResponse(CamelModel):
    message: str


class InferenceStats(CamelModel):
    count: Optional[str]
    ns: Optional[str]


class BatchStats(CamelModel):
    batch_size: str
    compute_input: InferenceStats
    compute_infer: InferenceStats
    compute_output: InferenceStats


class ModelStats(CamelModel):
    name: str
    version: str
    last_inference: Optional[str] = None
    inference_count: Optional[str] = None
    execution_count: Optional[str] = None
    inference_stats: Dict[str, InferenceStats]
    batch_stats: Optional[List[BatchStats]] = None


class MetricsResponse(CamelModel):
    model_stats: List[ModelStats]


class ModelEmbeddingDimsResponse(CamelModel):
    embedding_size: int


class ModelClassificationLabelsResponse(CamelModel):
    labels: List[str]
