"""
Model adapter classes for IR Image Classification System.

This module provides adapter classes for integrating different pre-trained models
(ResNet50 and Qwen VLM) for IR image embedding extraction and fine-tuning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoProcessor, AutoConfig
from torchvision import models, transforms
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union, cast
from abc import ABC, abstractmethod
import logging
from pathlib import Path
import json
from PIL import Image


from src.models.interfaces import IEmbeddingExtractor, BaseEmbeddingExtractor
from src.models.data_models import IRImage, Embedding


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseModelAdapter(ABC):
    """
    Abstract base class for model adapters.
    
    Provides common functionality for model loading, configuration management,
    and embedding extraction interface.
    """
    
    def __init__(self, model_name: str, embedding_dim: int = 512, device: Optional[str] = None):
        """
        Initialize base model adapter.
        
        Args:
            model_name: Name/identifier of the model
            embedding_dim: Dimension of output embeddings
            device: Device to run model on ('cuda', 'cpu', or None for auto)
        """
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model: Optional[nn.Module] = None
        self.is_loaded = False
        self.config: Dict[str, Any] = {}
        
        logger.info(f"Initialized {model_name} adapter on device: {self.device}")
    
    @abstractmethod
    def load_model(self, model_path: Optional[str] = None, **kwargs) -> None:
        """Load the model from path or initialize pre-trained model."""
        pass
    
    @abstractmethod
    def extract_embedding(self, image: np.ndarray) -> np.ndarray:
        """Extract embedding from preprocessed IR image."""
        pass
    
    @abstractmethod
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for model input."""
        pass
    
    @abstractmethod
    def batch_extract(self, images: List[np.ndarray], batch_size: int = 32) -> List[np.ndarray]:
        """
        Extract embeddings from multiple images efficiently.
        
        Args:
            images: List of preprocessed IR images
            batch_size: Number of images to process in each batch
            
        Returns:
            List[np.ndarray]: List of embedding vectors
        """
        pass
    
    def save_model(self, save_path: str) -> None:
        """
        Save the current model state.
        
        Args:
            save_path: Path to save the model
        """
        if not self.is_loaded:
            raise RuntimeError("No model loaded to save")
        
        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model state dict and configuration
        if self.model is not None:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'config': self.config,
                'model_name': self.model_name,
                'embedding_dim': self.embedding_dim
            }, save_path_obj)
            
            logger.info(f"Model saved to {save_path_obj}")
        else:
            raise RuntimeError("Model is None, cannot save")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        parameters_count = 0
        if self.model is not None and hasattr(self.model, 'parameters'):
            try:
                parameters_count = sum(p.numel() for p in self.model.parameters())
            except Exception:
                parameters_count = 0
        
        return {
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim,
            'device': self.device,
            'is_loaded': self.is_loaded,
            'config': self.config,
            'parameters': parameters_count
        }
    
    def _validate_image_input(self, image: np.ndarray) -> None:
        """Validate input image format."""
        if not isinstance(image, np.ndarray):
            raise ValueError("Image must be numpy array")
        
        if len(image.shape) not in [2, 3]:
            raise ValueError(f"Image must be 2D or 3D array, got shape {image.shape}")
    
    def validate_embedding_quality(self, embedding: np.ndarray) -> float:
        """
        Validate the quality of an extracted embedding.
        
        This is an optional method that adapters can override to provide
        model-specific quality validation.
        
        Args:
            embedding: Feature embedding vector
            
        Returns:
            float: Quality score (0.0-1.0, higher is better)
        """
        # Default implementation - basic validation
        if not isinstance(embedding, np.ndarray):
            return 0.0
        
        # Check for invalid values
        if np.any(np.isnan(embedding)) or np.any(np.isinf(embedding)):
            return 0.0
        
        # Check embedding dimension
        if len(embedding) == 0:
            return 0.0
        
        # Return a neutral score if no specific validation is implemented
        return 0.5


class IRResNet50Adapter(BaseModelAdapter):
    """
    Adapter for ResNet50 model fine-tuned for IR image embedding extraction.
    
    This adapter modifies ResNet50 to output embeddings instead of classifications,
    with IR-specific preprocessing and fine-tuning capabilities.
    """
    
    def __init__(self, embedding_dim: int = 512, device: Optional[str] = None, pretrained: bool = True):
        """
        Initialize ResNet50 adapter for IR images.
        
        Args:
            embedding_dim: Dimension of output embeddings
            device: Device to run model on
            pretrained: Whether to use ImageNet pre-trained weights
        """
        super().__init__("resnet50_ir", embedding_dim, device)
        self.pretrained = pretrained
        self.transform = None
        self._setup_transforms()
    
    def _setup_transforms(self) -> None:
        """Setup image preprocessing transforms for ResNet50."""
        # IR-specific transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3-channel for ResNet
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
        ])
    
    def load_model(self, model_path: Optional[str] = None, **kwargs) -> None:
        """
        Load ResNet50 model with custom embedding head.
        
        Args:
            model_path: Path to saved model weights (optional)
            **kwargs: Additional configuration parameters
        """
        try:
            # Load base ResNet50
            resnet_model = models.resnet50(pretrained=self.pretrained)
            
            # Replace final fully connected layer with embedding layer
            num_features = resnet_model.fc.in_features
            # Type ignore per permettere la sostituzione del layer fc
            resnet_model.fc = nn.Sequential(  # type: ignore
                nn.Linear(num_features, self.embedding_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(self.embedding_dim, self.embedding_dim),
                L2Norm(dim=1)  # L2 normalization for better embeddings
            )
            
            self.model = resnet_model
            
            # Load fine-tuned weights if provided
            if model_path and Path(model_path).exists():
                checkpoint = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.config = checkpoint.get('config', {})
                logger.info(f"Loaded fine-tuned weights from {model_path}")
            
            # Move model to device and set to evaluation mode
            self.model.to(self.device)
            self.model.eval()
            self.is_loaded = True
            
            # Update configuration
            self.config.update({
                'architecture': 'resnet50',
                'pretrained': self.pretrained,
                'embedding_dim': self.embedding_dim,
                'input_size': (224, 224),
                'num_parameters': sum(p.numel() for p in self.model.parameters())
            })
            
            logger.info(f"ResNet50 IR adapter loaded successfully with {self.config['num_parameters']} parameters")
            
        except Exception as e:
            logger.error(f"Failed to load ResNet50 model: {str(e)}")
            raise RuntimeError(f"Model loading failed: {str(e)}")
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess IR image for ResNet50 input.
        
        Args:
            image: Input IR image as numpy array (grayscale, 0-1 normalized)
            
        Returns:
            torch.Tensor: Preprocessed image tensor ready for model input
        """
        self._validate_image_input(image)
        
        if self.transform is None:
            raise RuntimeError("Transform not initialized. Call load_model() first.")
        
        # Ensure image is in correct format (0-255 uint8)
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        # Apply transforms
        tensor = self.transform(image)
        
        # Add batch dimension
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.unsqueeze(0)
        else:
            raise RuntimeError("Transform did not return a tensor")
        
        return tensor.to(self.device)
    
    def extract_embedding(self, image: np.ndarray) -> np.ndarray:
        """
        Extract embedding from IR image using ResNet50.
        
        Args:
            image: Preprocessed IR image as numpy array
            
        Returns:
            np.ndarray: Embedding vector
        """
        if not self.is_loaded or self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Preprocess image
        input_tensor = self.preprocess_image(image)
        
        # Extract embedding
        with torch.no_grad():
            embedding = self.model(input_tensor)
            embedding = embedding.cpu().numpy().flatten()
        
        return embedding
    
    def batch_extract(self, images: List[np.ndarray], batch_size: int = 32) -> List[np.ndarray]:
        """
        Extract embeddings from multiple images efficiently.
        
        Args:
            images: List of IR images
            batch_size: Batch size for processing
            
        Returns:
            List[np.ndarray]: List of embedding vectors
        """
        if not self.is_loaded or self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        embeddings = []
        
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            
            # Preprocess batch
            batch_tensors = torch.stack([
                self.preprocess_image(img).squeeze(0) for img in batch_images
            ])
            
            # Extract embeddings
            with torch.no_grad():
                batch_embeddings = self.model(batch_tensors)
                batch_embeddings = batch_embeddings.cpu().numpy()
            
            embeddings.extend(batch_embeddings)
        
        return embeddings
    
    def validate_embedding_quality(self, embedding: np.ndarray) -> float:
        """
        Assess the quality of an extracted embedding.
        
        Args:
            embedding: Feature embedding vector
            
        Returns:
            float: Quality score (0.0-1.0, higher is better)
        """
        if not isinstance(embedding, np.ndarray):
            return 0.0
        
        # Check for invalid values
        if np.any(np.isnan(embedding)) or np.any(np.isinf(embedding)):
            return 0.0
        
        # Check embedding dimension
        if len(embedding) != self.embedding_dim:
            return 0.0
        
        # Calculate quality metrics
        # 1. L2 norm should be close to 1 (normalized embeddings)
        l2_norm = np.linalg.norm(embedding)
        norm_score = 1.0 - abs(l2_norm - 1.0)  # Closer to 1 is better
        
        # 2. Check for diversity (not all zeros or all same values)
        diversity_score = 1.0 - (np.std(embedding) < 1e-6)  # Low std indicates poor diversity
        
        # 3. Check value range (should be reasonable, not extreme)
        range_score = 1.0 if np.all(np.abs(embedding) < 10.0) else 0.5
        
        # Combine scores
        quality_score = (norm_score * 0.5 + diversity_score * 0.3 + range_score * 0.2)
        return max(0.0, min(1.0, float(quality_score)))
    
    def fine_tune_setup(self, learning_rate: float = 1e-4, freeze_backbone: bool = False) -> torch.optim.Optimizer:
        """
        Setup model for fine-tuning on IR images.
        
        Args:
            learning_rate: Learning rate for fine-tuning
            freeze_backbone: Whether to freeze ResNet backbone layers
            
        Returns:
            torch.optim.Optimizer: Configured optimizer
        """
        if not self.is_loaded or self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Freeze backbone if requested
        if freeze_backbone:
            for name, param in self.model.named_parameters():
                if 'fc' not in name:  # Don't freeze the final embedding layers
                    param.requires_grad = False
            logger.info("Backbone layers frozen for fine-tuning")
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=learning_rate,
            weight_decay=1e-4
        )
        
        self.model.train()
        return optimizer


class QwenVLMAdapter(BaseModelAdapter):
    """
    Adapter for Qwen Vision Language Model for IR image embedding extraction.
    
    This adapter uses Qwen's vision encoder to extract embeddings from IR images,
    ignoring the text components and focusing on visual feature extraction.
    """
    
    def __init__(self, model_name: str = "Qwen/Qwen-VL-Chat", embedding_dim: int = 768, device: Optional[str] = None):
        """
        Initialize Qwen VLM adapter for IR images.
        
        Args:
            model_name: Hugging Face model name for Qwen VLM
            embedding_dim: Dimension of output embeddings
            device: Device to run model on
        """
        super().__init__(f"qwen_vlm_{model_name.split('/')[-1]}", embedding_dim, device)
        self.hf_model_name = model_name
        self.processor: Optional[Any] = None
        self.vision_encoder: Optional[nn.Module] = None
        self.projection_head: Optional[nn.Module] = None
    
    def load_model(self, model_path: Optional[str] = None, **kwargs) -> None:
        """
        Load Qwen VLM model and extract vision encoder.
        
        Args:
            model_path: Path to saved model weights (optional)
            **kwargs: Additional configuration parameters
        """
        try:
            # Load processor and model
            self.processor = AutoProcessor.from_pretrained(self.hf_model_name, trust_remote_code=True)
            full_model = AutoModel.from_pretrained(self.hf_model_name, trust_remote_code=True)
            
            # Extract vision encoder
            self.vision_encoder = full_model.visual
            
            # Create projection head to match desired embedding dimension
            vision_dim = 768  # Default Qwen VLM hidden size
            # Note: We use a safe default since accessing config attributes can be complex
            
            self.projection_head = nn.Sequential(
                nn.Linear(vision_dim, self.embedding_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.embedding_dim, self.embedding_dim),
                nn.LayerNorm(self.embedding_dim)
            )
            
            # Combine vision encoder and projection head
            if self.vision_encoder is not None and self.projection_head is not None:
                self.model = nn.Sequential(self.vision_encoder, self.projection_head)
            else:
                raise RuntimeError("Failed to initialize vision encoder or projection head")
            
            # Load fine-tuned weights if provided
            if model_path and Path(model_path).exists():
                checkpoint = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.config = checkpoint.get('config', {})
                logger.info(f"Loaded fine-tuned weights from {model_path}")
            
            # Move to device and set evaluation mode
            self.model.to(self.device)
            self.model.eval()
            self.is_loaded = True
            
            # Update configuration
            self.config.update({
                'architecture': 'qwen_vlm',
                'hf_model_name': self.hf_model_name,
                'embedding_dim': self.embedding_dim,
                'vision_dim': vision_dim,
                'num_parameters': sum(p.numel() for p in self.model.parameters())
            })
            
            logger.info(f"Qwen VLM adapter loaded successfully with {self.config['num_parameters']} parameters")
            
        except Exception as e:
            logger.error(f"Failed to load Qwen VLM model: {str(e)}")
            raise RuntimeError(f"Model loading failed: {str(e)}")
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess IR image for Qwen VLM input.
        
        Args:
            image: Input IR image as numpy array
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        self._validate_image_input(image)
        
        if self.processor is None:
            raise RuntimeError("Processor not initialized. Call load_model() first.")
        
        # Ensure image is in correct format
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        # Convert grayscale to RGB for VLM
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        
        pil_image = Image.fromarray(image)
        
        # Use processor to prepare image
        inputs = self.processor(images=pil_image, return_tensors="pt")
        
        # Extract pixel values and move to device
        pixel_values = inputs['pixel_values'].to(self.device)
        
        return pixel_values
    
    def extract_embedding(self, image: np.ndarray) -> np.ndarray:
        """
        Extract embedding from IR image using Qwen VLM vision encoder.
        
        Args:
            image: Preprocessed IR image as numpy array
            
        Returns:
            np.ndarray: Embedding vector
        """
        if not self.is_loaded or self.vision_encoder is None or self.projection_head is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Preprocess image
        input_tensor = self.preprocess_image(image)
        
        # Extract embedding using vision encoder
        with torch.no_grad():
            # Get vision features
            vision_features = self.vision_encoder(input_tensor)
            
            # Apply global average pooling if needed
            if len(vision_features.shape) > 2:
                vision_features = vision_features.mean(dim=1)  # Pool over sequence dimension
            
            # Apply projection head
            embedding = self.projection_head(vision_features)
            embedding = embedding.cpu().numpy().flatten()
        
        return embedding
    
    def batch_extract(self, images: List[np.ndarray], batch_size: int = 16) -> List[np.ndarray]:
        """
        Extract embeddings from multiple images efficiently.
        
        Args:
            images: List of IR images
            batch_size: Batch size for processing (smaller for VLM due to memory)
            
        Returns:
            List[np.ndarray]: List of embedding vectors
        """
        if not self.is_loaded or self.vision_encoder is None or self.projection_head is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        embeddings = []
        
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            
            # Preprocess batch
            batch_tensors = []
            for img in batch_images:
                tensor = self.preprocess_image(img)
                batch_tensors.append(tensor)
            
            batch_input = torch.cat(batch_tensors, dim=0)
            
            # Extract embeddings
            with torch.no_grad():
                vision_features = self.vision_encoder(batch_input)
                if len(vision_features.shape) > 2:
                    vision_features = vision_features.mean(dim=1)
                
                batch_embeddings = self.projection_head(vision_features)
                batch_embeddings = batch_embeddings.cpu().numpy()
            
            embeddings.extend(batch_embeddings)
        
        return embeddings
    
    def fine_tune_setup(self, learning_rate: float = 5e-5, freeze_vision_encoder: bool = True) -> torch.optim.Optimizer:
        """
        Setup model for fine-tuning on IR images.
        
        Args:
            learning_rate: Learning rate for fine-tuning
            freeze_vision_encoder: Whether to freeze vision encoder layers
            
        Returns:
            torch.optim.Optimizer: Configured optimizer
        """
        if not self.is_loaded or self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Freeze vision encoder if requested (recommended for stability)
        if freeze_vision_encoder and self.vision_encoder is not None:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
            logger.info("Vision encoder layers frozen for fine-tuning")
        
        # Setup optimizer (only for projection head if vision encoder is frozen)
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=learning_rate,
            weight_decay=1e-5
        )
        
        self.model.train()
        return optimizer


class ModelAdapterFactory:
    """
    Factory class for creating model adapters based on configuration.
    """
    
    @staticmethod
    def create_adapter(model_type: str, **kwargs) -> BaseModelAdapter:
        """
        Create a model adapter based on type.
        
        Args:
            model_type: Type of model adapter ('resnet50' or 'qwen_vlm')
            **kwargs: Additional configuration parameters
            
        Returns:
            BaseModelAdapter: Configured model adapter
        """
        if model_type.lower() == 'resnet50':
            return IRResNet50Adapter(**kwargs)
        elif model_type.lower() == 'qwen_vlm':
            return QwenVLMAdapter(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}. Supported types: 'resnet50', 'qwen_vlm'")
    
    @staticmethod
    def get_available_adapters() -> List[str]:
        """Get list of available model adapter types."""
        return ['resnet50', 'qwen_vlm']


# Custom L2 normalization layer
class L2Norm(nn.Module):
    """L2 normalization layer for embedding vectors."""
    
    def __init__(self, dim: int = 1):
        super(L2Norm, self).__init__()
        self.dim = dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, p=2, dim=self.dim)