from dataclasses import dataclass
import numpy as np

from typing import List, Optional, Dict


@dataclass
class EmbeddingResponse:
    processing_time_ms: float
    text_embeddings: Optional[List[List[float]] | np.ndarray] = None
    image_embeddings: Optional[List[List[float]] | np.ndarray] = None


@dataclass
class TextEmbeddingResponse:
    embeddings: List[List[float]] | np.ndarray
    processing_time_ms: float


@dataclass
class ImageEmbeddingResponse:
    embeddings: List[List[float]] | np.ndarray
    processing_time_ms: float


@dataclass
class ImageClassificationResponse:
    probabilities: List[List[float]] | np.ndarray
    processing_time_ms: float


@dataclass
class LoadedModel:
    name: str
    library: str


@dataclass
class LoadedModelResponse:
    models: List[LoadedModel]


@dataclass
class RepositoryModel:
    name: str
    state: str


@dataclass
class RepositoryModelResponse:
    models: List[RepositoryModel]


@dataclass
class GenericMessageResponse:
    message: str


@dataclass
class InferenceStats:
    count: Optional[str]
    ns: Optional[str]


@dataclass
class BatchStats:
    batch_size: str
    compute_input: InferenceStats
    compute_infer: InferenceStats
    compute_output: InferenceStats


@dataclass
class ModelStats:
    name: str
    version: str
    inference_stats: Dict[str, InferenceStats]
    last_inference: Optional[str] = None
    inference_count: Optional[str] = None
    execution_count: Optional[str] = None
    batch_stats: Optional[List[BatchStats]] = None


@dataclass
class MetricsResponse:
    model_stats: List[ModelStats]


@dataclass
class ModelEmbeddingDimsResponse:
    embedding_size: int


@dataclass
class ModelClassificationLabelsResponse:
    labels: List[str]
