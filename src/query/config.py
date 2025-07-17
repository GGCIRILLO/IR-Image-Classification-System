"""
Configuration classes for QueryProcessor.

This module provides configuration dataclasses for customizing
QueryProcessor behavior and performance parameters.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from enum import Enum


class ValidationMode(Enum):
    """Image validation modes for query processing."""
    STRICT = "strict"      # Enforce all IR format requirements
    RELAXED = "relaxed"    # Allow minor format deviations
    DISABLED = "disabled"  # Skip validation (for testing)


class CachePolicy(Enum):
    """Cache policies for query results."""
    ENABLED = "enabled"    # Cache all queries
    DISABLED = "disabled"  # No caching
    MEMORY_ONLY = "memory_only"  # Cache in memory only


@dataclass
class QueryProcessorConfig:
    """
    Configuration for QueryProcessor behavior and performance.
    
    This configuration class allows fine-tuning of query processing
    parameters to meet specific performance and accuracy requirements.
    """
    
    # Performance requirements
    max_query_time: float = 2.0  # Maximum time for query processing (seconds)
    min_confidence_threshold: float = 0.7  # Minimum confidence for results
    top_k_results: int = 5  # Number of top results to return
    
    # Query validation settings
    validation_mode: ValidationMode = ValidationMode.RELAXED
    strict_ir_format: bool = False  # Enforce strict IR format validation
    allow_color_images: bool = True  # Allow color images (convert to grayscale)
    min_image_size: int = 64  # Minimum image dimension
    max_image_size: int = 2048  # Maximum image dimension
    
    # Performance optimization
    enable_result_reranking: bool = True  # Enable result re-ranking
    enable_gpu_acceleration: bool = True  # Use GPU if available
    batch_processing: bool = False  # Enable batch processing for multiple queries
    parallel_processing: bool = False  # Enable parallel processing
    
    # Caching configuration
    cache_policy: CachePolicy = CachePolicy.ENABLED
    max_cache_size: int = 1000  # Maximum number of cached queries
    cache_ttl_seconds: int = 3600  # Cache time-to-live (1 hour)
    
    # Search configuration
    search_timeout_ms: float = 1500.0  # Maximum search time in milliseconds
    similarity_threshold: float = 0.5  # Minimum similarity for results
    enable_approximate_search: bool = True  # Use approximate search for speed
    
    # Model configuration
    model_type: str = "resnet50"  # Model type: "resnet50" or "qwen_vlm"
    embedding_dimension: int = 512  # Expected embedding dimension
    model_confidence_threshold: float = 0.8  # Model confidence threshold
    
    # Logging and monitoring
    enable_performance_logging: bool = True  # Log performance metrics
    enable_query_logging: bool = True  # Log all queries
    log_level: str = "INFO"  # Logging level
    
    # Error handling
    retry_attempts: int = 3  # Number of retry attempts on failure
    fallback_to_cpu: bool = True  # Fallback to CPU if GPU fails
    continue_on_warning: bool = True  # Continue processing on warnings
    
    # Advanced options
    custom_preprocessing: Optional[Dict[str, Any]] = None  # Custom preprocessing options
    custom_postprocessing: Optional[Dict[str, Any]] = None  # Custom postprocessing options
    debug_mode: bool = False  # Enable debug mode for troubleshooting
    
    def validate(self) -> bool:
        """
        Validate configuration parameters.
        
        Returns:
            bool: True if configuration is valid
            
        Raises:
            ValueError: If configuration is invalid
        """
        if self.max_query_time <= 0:
            raise ValueError("max_query_time must be positive")
        
        if not 0.0 <= self.min_confidence_threshold <= 1.0:
            raise ValueError("min_confidence_threshold must be between 0.0 and 1.0")
        
        if self.top_k_results <= 0:
            raise ValueError("top_k_results must be positive")
        
        if not 0.0 <= self.similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between 0.0 and 1.0")
        
        if self.embedding_dimension <= 0:
            raise ValueError("embedding_dimension must be positive")
        
        if self.max_cache_size <= 0:
            raise ValueError("max_cache_size must be positive")
        
        if self.cache_ttl_seconds <= 0:
            raise ValueError("cache_ttl_seconds must be positive")
        
        if self.retry_attempts < 0:
            raise ValueError("retry_attempts must be non-negative")
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'max_query_time': self.max_query_time,
            'min_confidence_threshold': self.min_confidence_threshold,
            'top_k_results': self.top_k_results,
            'validation_mode': self.validation_mode.value,
            'strict_ir_format': self.strict_ir_format,
            'allow_color_images': self.allow_color_images,
            'min_image_size': self.min_image_size,
            'max_image_size': self.max_image_size,
            'enable_result_reranking': self.enable_result_reranking,
            'enable_gpu_acceleration': self.enable_gpu_acceleration,
            'batch_processing': self.batch_processing,
            'parallel_processing': self.parallel_processing,
            'cache_policy': self.cache_policy.value,
            'max_cache_size': self.max_cache_size,
            'cache_ttl_seconds': self.cache_ttl_seconds,
            'search_timeout_ms': self.search_timeout_ms,
            'similarity_threshold': self.similarity_threshold,
            'enable_approximate_search': self.enable_approximate_search,
            'model_type': self.model_type,
            'embedding_dimension': self.embedding_dimension,
            'model_confidence_threshold': self.model_confidence_threshold,
            'enable_performance_logging': self.enable_performance_logging,
            'enable_query_logging': self.enable_query_logging,
            'log_level': self.log_level,
            'retry_attempts': self.retry_attempts,
            'fallback_to_cpu': self.fallback_to_cpu,
            'continue_on_warning': self.continue_on_warning,
            'custom_preprocessing': self.custom_preprocessing,
            'custom_postprocessing': self.custom_postprocessing,
            'debug_mode': self.debug_mode
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'QueryProcessorConfig':
        """Create configuration from dictionary."""
        # Handle enum conversions
        if 'validation_mode' in config_dict:
            config_dict['validation_mode'] = ValidationMode(config_dict['validation_mode'])
        if 'cache_policy' in config_dict:
            config_dict['cache_policy'] = CachePolicy(config_dict['cache_policy'])
        
        return cls(**config_dict)


@dataclass
class MilitaryQueryConfig(QueryProcessorConfig):
    """
    Military-specific configuration with enhanced security and accuracy.
    
    This configuration preset is optimized for military applications
    requiring high precision, security, and offline operation.
    """
    
    # Military-specific overrides
    max_query_time: float = 1.5  # Faster response for operational use
    min_confidence_threshold: float = 0.85  # Higher confidence requirement
    validation_mode: ValidationMode = ValidationMode.STRICT  # Strict validation
    strict_ir_format: bool = True  # Enforce IR format
    
    # Security settings
    cache_policy: CachePolicy = CachePolicy.MEMORY_ONLY  # No disk caching
    enable_query_logging: bool = False  # Disable query logging for security
    
    # Performance optimization for military hardware
    enable_gpu_acceleration: bool = False  # Assume CPU-only deployment
    fallback_to_cpu: bool = True  # Always use CPU
    
    # Higher accuracy requirements
    enable_result_reranking: bool = True  # Always re-rank for accuracy
    model_confidence_threshold: float = 0.9  # High model confidence
    similarity_threshold: float = 0.7  # Higher similarity threshold


@dataclass
class DevelopmentQueryConfig(QueryProcessorConfig):
    """
    Development configuration with debugging and relaxed validation.
    
    This configuration is optimized for development and testing,
    with relaxed validation and enhanced debugging capabilities.
    """
    
    # Development-specific overrides
    validation_mode: ValidationMode = ValidationMode.RELAXED
    strict_ir_format: bool = False  # Allow various image formats
    allow_color_images: bool = True  # Accept color images
    
    # Enhanced debugging
    debug_mode: bool = True  # Enable debug mode
    enable_performance_logging: bool = True  # Log everything
    enable_query_logging: bool = True  # Log all queries
    log_level: str = "DEBUG"  # Verbose logging
    
    # Relaxed thresholds for testing
    min_confidence_threshold: float = 0.5  # Lower confidence for testing
    similarity_threshold: float = 0.3  # Lower similarity for more results
    
    # Extended timeouts for debugging
    max_query_time: float = 5.0  # Longer timeout for debugging
    search_timeout_ms: float = 3000.0  # Longer search timeout


def get_config_for_environment(environment: str) -> QueryProcessorConfig:
    """
    Get configuration preset for specific environment.
    
    Args:
        environment: Environment name ("production", "military", "development", "testing")
        
    Returns:
        QueryProcessorConfig: Configuration for the specified environment
    """
    if environment.lower() in ("production", "prod"):
        return QueryProcessorConfig()
    elif environment.lower() in ("military", "mil", "secure"):
        return MilitaryQueryConfig()
    elif environment.lower() in ("development", "dev", "debug"):
        return DevelopmentQueryConfig()
    elif environment.lower() in ("testing", "test"):
        config = DevelopmentQueryConfig()
        config.max_query_time = 10.0  # Very long timeout for testing
        config.retry_attempts = 5  # More retries for testing
        return config
    else:
        raise ValueError(f"Unknown environment: {environment}")
