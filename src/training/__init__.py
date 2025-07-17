"""
Training module for IR Image Classification System.

This module provides comprehensive training infrastructure including:
- ModelTrainer with triplet loss optimization
- TrainingMetrics for precision, recall, and embedding quality tracking
- ValidationPipeline with 95% accuracy target validation
- Model adapters for ResNet50 and Qwen VLM
"""

from .trainer import ModelTrainer, TrainingMetrics, TripletLoss, IRImageDataset
from .validation_pipeline import ValidationPipeline, ValidationResult
from .model_adapters import (
    BaseModelAdapter, IRResNet50Adapter, QwenVLMAdapter, 
    ModelAdapterFactory
)
from .model_config import (
    ModelConfig, ResNet50Config, QwenVLMConfig, 
    TrainingConfig, DeploymentConfig, ConfigManager
)

__all__ = [
    # Core training classes
    'ModelTrainer',
    'TrainingMetrics', 
    'TripletLoss',
    'IRImageDataset',
    
    # Validation classes
    'ValidationPipeline',
    'ValidationResult',
    
    # Model adapters
    'BaseModelAdapter',
    'IRResNet50Adapter', 
    'QwenVLMAdapter',
    'ModelAdapterFactory',
    
    # Configuration classes
    'ModelConfig',
    'ResNet50Config',
    'QwenVLMConfig', 
    'TrainingConfig',
    'DeploymentConfig',
    'ConfigManager'
]