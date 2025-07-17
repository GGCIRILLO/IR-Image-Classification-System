"""
Embedding extraction module for IR Image Classification System.

This module provides embedding extraction services with batch processing,
GPU acceleration, caching, and quality validation.
"""

from .extractor import EmbeddingExtractor, EmbeddingCache, ExtractionConfig

__all__ = [
    'EmbeddingExtractor',
    'EmbeddingCache', 
    'ExtractionConfig'
]