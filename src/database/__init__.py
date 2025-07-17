"""
Vector database operations for IR image classification.

This module provides a complete vector database solution using ChromaDB
for storing and searching IR image embeddings with military-grade security
and performance requirements.
"""

from .vector_store import ChromaVectorStore
from .db_manager import DatabaseManager
from .index_manager import IndexManager, IndexConfig, IndexMetrics
from .similarity_searcher import SimilaritySearcher, SearchConfig, SearchMode, SearchMetrics

__all__ = [
    'ChromaVectorStore',
    'DatabaseManager', 
    'IndexManager',
    'IndexConfig',
    'IndexMetrics',
    'SimilaritySearcher',
    'SearchConfig',
    'SearchMode',
    'SearchMetrics'
]