"""
Data processing module for IR Image Classification System.

This module provides image processing, loading, and validation capabilities
specifically designed for military infrared imagery.
"""

from .ir_processor import IRImageProcessor
from .image_loader import (
    IRImageLoader, 
    ImageFormatError, 
    ImageCorruptionError,
    load_ir_image,
    validate_ir_image_format,
    get_ir_image_info
)
from .augmentation import (
    DataAugmentationEngine,
    AugmentationConfig,
    AugmentationType
)

__all__ = [
    'IRImageProcessor',
    'IRImageLoader',
    'ImageFormatError',
    'ImageCorruptionError',
    'load_ir_image',
    'validate_ir_image_format',
    'get_ir_image_info',
    'DataAugmentationEngine',
    'AugmentationConfig',
    'AugmentationType'
]