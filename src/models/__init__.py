"""
Models package for the IR Image Classification System.

This package contains data models and abstract interfaces for system components.
"""

from .data_models import IRImage, Embedding, SimilarityResult, QueryResult
from .interfaces import (
    IDataAugmentation, IImageProcessor, IEmbeddingExtractor, IVectorStore,
    BaseDataAugmentation, BaseImageProcessor, BaseEmbeddingExtractor, BaseVectorStore
)

__all__ = [
    # Data models
    'IRImage',
    'Embedding', 
    'SimilarityResult',
    'QueryResult',
    # Abstract interfaces
    'IDataAugmentation',
    'IImageProcessor', 
    'IEmbeddingExtractor',
    'IVectorStore',
    # Base implementations
    'BaseDataAugmentation',
    'BaseImageProcessor',
    'BaseEmbeddingExtractor', 
    'BaseVectorStore'
]