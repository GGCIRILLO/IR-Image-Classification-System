"""
Model configuration management for IR Image Classification System.

This module provides configuration classes and utilities for managing
model adapter settings, training parameters, and deployment configurations.
"""

import json
import yaml
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """
    Base configuration class for model adapters.
    
    Contains common configuration parameters shared across different model types.
    """
    model_type: str  # 'resnet50' or 'qwen_vlm'
    embedding_dim: int = 512
    device: str = 'auto'  # 'auto', 'cuda', 'cpu'
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    dropout_rate: float = 0.2
    model_path: Optional[str] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate()
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.model_type not in ['resnet50', 'qwen_vlm']:
            raise ValueError(f"Invalid model_type: {self.model_type}")
        
        if self.embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {self.embedding_dim}")
        
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        
        if not 0 < self.learning_rate < 1:
            raise ValueError(f"learning_rate must be between 0 and 1, got {self.learning_rate}")
        
        if not 0 <= self.dropout_rate < 1:
            raise ValueError(f"dropout_rate must be between 0 and 1, got {self.dropout_rate}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)


@dataclass
class ResNet50Config(ModelConfig):
    """
    Configuration specific to ResNet50 model adapter.
    """
    model_type: str = 'resnet50'
    pretrained: bool = True
    freeze_backbone: bool = False
    input_size: tuple = (224, 224)
    num_classes: Optional[int] = None  # For classification head if needed
    
    # ResNet50-specific augmentation settings
    normalize_mean: List[float] = None
    normalize_std: List[float] = None
    
    def __post_init__(self):
        """Initialize ResNet50-specific defaults."""
        if self.normalize_mean is None:
            self.normalize_mean = [0.485, 0.456, 0.406]  # ImageNet defaults
        if self.normalize_std is None:
            self.normalize_std = [0.229, 0.224, 0.225]   # ImageNet defaults
        super().__post_init__()


@dataclass
class QwenVLMConfig(ModelConfig):
    """
    Configuration specific to Qwen VLM model adapter.
    """
    model_type: str = 'qwen_vlm'
    hf_model_name: str = 'Qwen/Qwen-VL-Chat'
    freeze_vision_encoder: bool = True
    vision_dim: int = 768
    max_image_size: int = 448
    
    # Qwen-specific settings
    trust_remote_code: bool = True
    use_flash_attention: bool = False
    
    def __post_init__(self):
        """Initialize Qwen VLM-specific defaults."""
        # Adjust learning rate for VLM (typically lower)
        if self.learning_rate > 1e-4:
            self.learning_rate = 5e-5
        
        # Adjust batch size for VLM (typically smaller due to memory)
        if self.batch_size > 16:
            self.batch_size = 8
        
        super().__post_init__()


@dataclass
class TrainingConfig:
    """
    Configuration for model training pipeline.
    """
    # Training parameters
    num_epochs: int = 50
    early_stopping_patience: int = 10
    validation_split: float = 0.2
    save_best_only: bool = True
    
    # Loss function settings
    loss_type: str = 'triplet'  # 'triplet', 'contrastive', 'arcface'
    triplet_margin: float = 0.3
    hard_negative_mining: bool = True
    
    # Optimization settings
    optimizer: str = 'adamw'  # 'adamw', 'adam', 'sgd'
    scheduler: str = 'cosine'  # 'cosine', 'step', 'plateau'
    warmup_epochs: int = 5
    
    # Regularization
    gradient_clip_norm: float = 1.0
    label_smoothing: float = 0.0
    
    # Logging and checkpointing
    log_interval: int = 10
    save_interval: int = 5
    checkpoint_dir: str = './checkpoints'
    
    def validate(self) -> None:
        """Validate training configuration."""
        if self.num_epochs <= 0:
            raise ValueError(f"num_epochs must be positive, got {self.num_epochs}")
        
        if not 0 < self.validation_split < 1:
            raise ValueError(f"validation_split must be between 0 and 1, got {self.validation_split}")
        
        if self.loss_type not in ['triplet', 'contrastive', 'arcface']:
            raise ValueError(f"Invalid loss_type: {self.loss_type}")
        
        if self.optimizer not in ['adamw', 'adam', 'sgd']:
            raise ValueError(f"Invalid optimizer: {self.optimizer}")


@dataclass
class DeploymentConfig:
    """
    Configuration for model deployment and inference.
    """
    # Model serving settings
    model_path: str
    device: str = 'auto'
    batch_size: int = 1
    max_batch_size: int = 32
    
    # Performance settings
    enable_tensorrt: bool = False
    enable_onnx: bool = False
    fp16_inference: bool = False
    
    # Caching settings
    enable_embedding_cache: bool = True
    cache_size: int = 1000
    cache_ttl: int = 3600  # seconds
    
    # API settings
    api_timeout: float = 30.0
    max_image_size: int = 10 * 1024 * 1024  # 10MB
    
    def validate(self) -> None:
        """Validate deployment configuration."""
        if not Path(self.model_path).exists():
            logger.warning(f"Model path does not exist: {self.model_path}")
        
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        
        if self.max_batch_size < self.batch_size:
            raise ValueError("max_batch_size must be >= batch_size")


class ConfigManager:
    """
    Manager class for loading, saving, and validating configurations.
    """
    
    def __init__(self, config_dir: str = './config'):
        """
        Initialize configuration manager.
        
        Args:
            config_dir: Directory to store configuration files
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
    
    def save_config(self, config: Union[ModelConfig, TrainingConfig, DeploymentConfig], 
                   filename: str) -> None:
        """
        Save configuration to file.
        
        Args:
            config: Configuration object to save
            filename: Name of the configuration file
        """
        config_path = self.config_dir / filename
        
        # Determine file format from extension
        if filename.endswith('.json'):
            with open(config_path, 'w') as f:
                json.dump(asdict(config), f, indent=2)
        elif filename.endswith(('.yaml', '.yml')):
            with open(config_path, 'w') as f:
                yaml.dump(asdict(config), f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported file format: {filename}")
        
        logger.info(f"Configuration saved to {config_path}")
    
    def load_config(self, filename: str, config_type: str) -> Union[ModelConfig, TrainingConfig, DeploymentConfig]:
        """
        Load configuration from file.
        
        Args:
            filename: Name of the configuration file
            config_type: Type of configuration ('model', 'training', 'deployment')
            
        Returns:
            Configuration object
        """
        config_path = self.config_dir / filename
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Load configuration data
        if filename.endswith('.json'):
            with open(config_path, 'r') as f:
                config_data = json.load(f)
        elif filename.endswith(('.yaml', '.yml')):
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported file format: {filename}")
        
        # Create appropriate configuration object
        if config_type == 'model':
            model_type = config_data.get('model_type', 'resnet50')
            if model_type == 'resnet50':
                return ResNet50Config.from_dict(config_data)
            elif model_type == 'qwen_vlm':
                return QwenVLMConfig.from_dict(config_data)
            else:
                return ModelConfig.from_dict(config_data)
        elif config_type == 'training':
            return TrainingConfig(**config_data)
        elif config_type == 'deployment':
            return DeploymentConfig(**config_data)
        else:
            raise ValueError(f"Unknown config_type: {config_type}")
    
    def create_default_configs(self) -> None:
        """Create default configuration files."""
        # Default ResNet50 config
        resnet_config = ResNet50Config(
            model_type='resnet50',
            embedding_dim=512,
            batch_size=32,
            learning_rate=1e-4
        )
        self.save_config(resnet_config, 'resnet50_default.yaml')
        
        # Default Qwen VLM config
        qwen_config = QwenVLMConfig(
            model_type='qwen_vlm',
            embedding_dim=768,
            batch_size=8,
            learning_rate=5e-5
        )
        self.save_config(qwen_config, 'qwen_vlm_default.yaml')
        
        # Default training config
        training_config = TrainingConfig(
            num_epochs=50,
            loss_type='triplet',
            optimizer='adamw',
            scheduler='cosine'
        )
        self.save_config(training_config, 'training_default.yaml')
        
        # Default deployment config
        deployment_config = DeploymentConfig(
            model_path='./models/best_model.pth',
            device='auto',
            batch_size=1
        )
        self.save_config(deployment_config, 'deployment_default.yaml')
        
        logger.info("Default configuration files created")
    
    def validate_config_compatibility(self, model_config: ModelConfig, 
                                    training_config: TrainingConfig) -> bool:
        """
        Validate compatibility between model and training configurations.
        
        Args:
            model_config: Model configuration
            training_config: Training configuration
            
        Returns:
            bool: True if configurations are compatible
        """
        # Check batch size compatibility
        if model_config.batch_size != training_config.checkpoint_dir:
            logger.warning("Model and training batch sizes don't match")
        
        # Check device compatibility
        if model_config.device not in ['auto', 'cuda', 'cpu']:
            logger.warning(f"Unknown device: {model_config.device}")
        
        # Model-specific checks
        if isinstance(model_config, QwenVLMConfig):
            if model_config.batch_size > 16:
                logger.warning("Large batch size for Qwen VLM may cause memory issues")
        
        return True
    
    def get_config_summary(self, config: Union[ModelConfig, TrainingConfig, DeploymentConfig]) -> str:
        """
        Get a human-readable summary of the configuration.
        
        Args:
            config: Configuration object
            
        Returns:
            str: Configuration summary
        """
        config_dict = asdict(config)
        summary_lines = []
        
        for key, value in config_dict.items():
            if isinstance(value, (list, dict)):
                value = str(value)
            summary_lines.append(f"{key}: {value}")
        
        return "\n".join(summary_lines)


# Utility functions for configuration management

def get_device_config() -> str:
    """
    Automatically determine the best device configuration.
    
    Returns:
        str: Device configuration ('cuda' or 'cpu')
    """
    import torch
    
    if torch.cuda.is_available():
        device = 'cuda'
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"CUDA available: {gpu_count} GPU(s), Primary: {gpu_name}")
    else:
        device = 'cpu'
        logger.info("CUDA not available, using CPU")
    
    return device


def estimate_memory_requirements(config: ModelConfig) -> Dict[str, float]:
    """
    Estimate memory requirements for model configuration.
    
    Args:
        config: Model configuration
        
    Returns:
        Dict[str, float]: Memory estimates in GB
    """
    # Rough estimates based on model type and configuration
    if config.model_type == 'resnet50':
        model_memory = 0.1  # ResNet50 base memory
        batch_memory = config.batch_size * 0.01  # Per image in batch
    elif config.model_type == 'qwen_vlm':
        model_memory = 2.0  # VLM models are larger
        batch_memory = config.batch_size * 0.05  # VLM requires more memory per image
    else:
        model_memory = 0.5
        batch_memory = config.batch_size * 0.02
    
    total_memory = model_memory + batch_memory
    
    return {
        'model_memory_gb': model_memory,
        'batch_memory_gb': batch_memory,
        'total_memory_gb': total_memory,
        'recommended_gpu_memory_gb': total_memory * 2  # Safety margin
    }