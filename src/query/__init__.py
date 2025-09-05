"""
Query processing and similarity search module.

This module provides the query processing engine for IR image similarity search,
including the main QueryProcessor class, result ranking, confidence scoring,
configuration classes, and related exceptions.
"""

from .processor import (
    QueryProcessor, 
    QueryValidationError, 
    QueryProcessingError,
    create_query_processor_for_model,
    create_resnet18_query_processor,
    create_resnet50_query_processor
)
from .config import (
    QueryProcessorConfig,
    MilitaryQueryConfig,
    DevelopmentQueryConfig,
    ValidationMode,
    CachePolicy,
    get_config_for_environment
)
from .ranker import (
    ResultRanker,
    RankingConfig,
    RankingStrategy,
    RankingMetrics,
    ConfidenceLevel
)
from .confidence import (
    ConfidenceCalculator,
    ConfidenceConfig,
    ConfidenceStrategy,
    ConfidenceAnalysis,
    ConfidenceMetrics,
    ConfidenceFactors
)

__all__ = [
    'QueryProcessor',
    'QueryValidationError', 
    'QueryProcessingError',
    'create_query_processor_for_model',
    'create_resnet18_query_processor',
    'create_resnet50_query_processor',
    'QueryProcessorConfig',
    'MilitaryQueryConfig',
    'DevelopmentQueryConfig',
    'ValidationMode',
    'CachePolicy',
    'get_config_for_environment',
    'ResultRanker',
    'RankingConfig',
    'RankingStrategy',
    'RankingMetrics',
    'ConfidenceLevel',
    'ConfidenceCalculator',
    'ConfidenceConfig',
    'ConfidenceStrategy',
    'ConfidenceAnalysis',
    'ConfidenceMetrics',
    'ConfidenceFactors'
]