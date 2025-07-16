"""
Abstract interfaces for the IR Image Classification System.

This module defines the core interfaces that system components must implement,
providing clear contracts for data augmentation, image processing, embedding
extraction, and vector storage operations.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from .data_models import IRImage, Embedding, SimilarityResult, QueryResult


class IDataAugmentation(ABC):
    """
    Abstract interface for data augmentation operations.
    
    Implementations should provide methods for augmenting IR images while
    preserving their military-relevant characteristics.
    """
    
    @abstractmethod
    def augment_batch(self, images: List[np.ndarray], target_count: int) -> List[np.ndarray]:
        """
        Augment a batch of images to reach target count.
        
        Args:
            images: List of input images as numpy arrays
            target_count: Desired total number of images after augmentation
            
        Returns:
            List[np.ndarray]: Augmented images including originals
        """
        pass
    
    @abstractmethod
    def preserve_ir_characteristics(self, image: np.ndarray) -> np.ndarray:
        """
        Apply augmentation while preserving IR characteristics.
        
        Ensures that white objects remain on black background and
        military-relevant features are maintained.
        
        Args:
            image: Input IR image as numpy array
            
        Returns:
            np.ndarray: Augmented image with preserved IR characteristics
        """
        pass
    
    @abstractmethod
    def rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """
        Rotate image by specified angle while maintaining IR properties.
        
        Args:
            image: Input image
            angle: Rotation angle in degrees
            
        Returns:
            np.ndarray: Rotated image
        """
        pass
    
    @abstractmethod
    def scale_image(self, image: np.ndarray, scale_factor: float) -> np.ndarray:
        """
        Scale image by specified factor.
        
        Args:
            image: Input image
            scale_factor: Scaling factor (e.g., 0.8 for 80% size)
            
        Returns:
            np.ndarray: Scaled image
        """
        pass
    
    @abstractmethod
    def add_noise(self, image: np.ndarray, noise_level: float = 0.1) -> np.ndarray:
        """
        Add realistic noise to simulate IR sensor conditions.
        
        Args:
            image: Input image
            noise_level: Intensity of noise to add (0.0-1.0)
            
        Returns:
            np.ndarray: Image with added noise
        """
        pass


class IImageProcessor(ABC):
    """
    Abstract interface for IR image processing operations.
    
    Implementations should handle IR-specific preprocessing, validation,
    and format conversion operations.
    """
    
    @abstractmethod
    def preprocess_ir_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply IR-specific preprocessing to enhance image quality.
        
        Args:
            image: Raw IR image as numpy array
            
        Returns:
            np.ndarray: Preprocessed IR image
        """
        pass
    
    @abstractmethod
    def validate_ir_format(self, image: np.ndarray) -> bool:
        """
        Validate that image meets IR format requirements.
        
        Checks for white objects on black background, proper dimensions,
        and other IR-specific characteristics.
        
        Args:
            image: Image to validate
            
        Returns:
            bool: True if image meets IR format requirements
        """
        pass
    
    @abstractmethod
    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance contrast for better object visibility in IR images.
        
        Args:
            image: Input IR image
            
        Returns:
            np.ndarray: Contrast-enhanced image
        """
        pass
    
    @abstractmethod
    def reduce_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Apply noise reduction techniques suitable for IR imagery.
        
        Args:
            image: Noisy IR image
            
        Returns:
            np.ndarray: Denoised image
        """
        pass
    
    @abstractmethod
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image values for consistent processing.
        
        Args:
            image: Input image
            
        Returns:
            np.ndarray: Normalized image (typically 0-1 range)
        """
        pass
    
    @abstractmethod
    def resize_to_standard(self, image: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """
        Resize image to standard dimensions for model input.
        
        Args:
            image: Input image
            target_size: Target dimensions (height, width)
            
        Returns:
            np.ndarray: Resized image
        """
        pass


class IEmbeddingExtractor(ABC):
    """
    Abstract interface for embedding extraction from IR images.
    
    Implementations should provide methods for extracting high-quality
    feature embeddings using fine-tuned models.
    """
    
    @abstractmethod
    def extract_embedding(self, image: np.ndarray) -> np.ndarray:
        """
        Extract feature embedding from a single IR image.
        
        Args:
            image: Preprocessed IR image as numpy array
            
        Returns:
            np.ndarray: Feature embedding vector
        """
        pass
    
    @abstractmethod
    def batch_extract(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """
        Extract embeddings from multiple images efficiently.
        
        Args:
            images: List of preprocessed IR images
            
        Returns:
            List[np.ndarray]: List of embedding vectors
        """
        pass
    
    @abstractmethod
    def validate_embedding_quality(self, embedding: np.ndarray) -> float:
        """
        Assess the quality of an extracted embedding.
        
        Args:
            embedding: Feature embedding vector
            
        Returns:
            float: Quality score (0.0-1.0, higher is better)
        """
        pass
    
    @abstractmethod
    def load_model(self, model_path: str) -> None:
        """
        Load the fine-tuned model for embedding extraction.
        
        Args:
            model_path: Path to the trained model file
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the currently loaded model.
        
        Returns:
            Dict[str, Any]: Model metadata including version, architecture, etc.
        """
        pass


class IVectorStore(ABC):
    """
    Abstract interface for vector database operations.
    
    Implementations should provide efficient storage, indexing, and
    similarity search capabilities for embedding vectors.
    """
    
    @abstractmethod
    def store_embedding(self, embedding: Embedding) -> bool:
        """
        Store a single embedding in the vector database.
        
        Args:
            embedding: Embedding object to store
            
        Returns:
            bool: True if storage was successful
        """
        pass
    
    @abstractmethod
    def store_embeddings_batch(self, embeddings: List[Embedding]) -> bool:
        """
        Store multiple embeddings efficiently.
        
        Args:
            embeddings: List of embedding objects to store
            
        Returns:
            bool: True if all embeddings were stored successfully
        """
        pass
    
    @abstractmethod
    def search_similar(self, query_embedding: np.ndarray, k: int = 5) -> List[SimilarityResult]:
        """
        Find the k most similar embeddings to the query.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of similar results to return
            
        Returns:
            List[SimilarityResult]: Top-k similar embeddings with scores
        """
        pass
    
    @abstractmethod
    def get_embedding(self, embedding_id: str) -> Optional[Embedding]:
        """
        Retrieve a specific embedding by ID.
        
        Args:
            embedding_id: Unique identifier of the embedding
            
        Returns:
            Optional[Embedding]: The embedding if found, None otherwise
        """
        pass
    
    @abstractmethod
    def delete_embedding(self, embedding_id: str) -> bool:
        """
        Delete an embedding from the database.
        
        Args:
            embedding_id: Unique identifier of the embedding to delete
            
        Returns:
            bool: True if deletion was successful
        """
        pass
    
    @abstractmethod
    def create_index(self, index_type: str = "hnsw") -> bool:
        """
        Create or rebuild the similarity search index.
        
        Args:
            index_type: Type of index to create (e.g., "hnsw", "ivf")
            
        Returns:
            bool: True if index creation was successful
        """
        pass
    
    @abstractmethod
    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector database.
        
        Returns:
            Dict[str, Any]: Database statistics (count, size, index info, etc.)
        """
        pass
    
    @abstractmethod
    def initialize_database(self, config: Dict[str, Any]) -> bool:
        """
        Initialize the vector database with configuration.
        
        Args:
            config: Database configuration parameters
            
        Returns:
            bool: True if initialization was successful
        """
        pass


# Base classes with common functionality

class BaseDataAugmentation(IDataAugmentation):
    """
    Base implementation providing common augmentation functionality.
    
    Subclasses can override specific methods while inheriting common utilities.
    """
    
    def __init__(self, preserve_ir_properties: bool = True):
        """
        Initialize base augmentation with IR preservation settings.
        
        Args:
            preserve_ir_properties: Whether to maintain IR characteristics
        """
        self.preserve_ir_properties = preserve_ir_properties
        self.supported_augmentations = [
            'rotation', 'scaling', 'noise', 'brightness', 'contrast'
        ]
    
    def _validate_image(self, image: np.ndarray) -> None:
        """
        Validate input image format.
        
        Args:
            image: Image to validate
            
        Raises:
            ValueError: If image format is invalid
        """
        if not isinstance(image, np.ndarray):
            raise ValueError("Image must be a numpy array")
        
        if len(image.shape) not in [2, 3]:
            raise ValueError(f"Image must be 2D or 3D array, got shape {image.shape}")
    
    def _ensure_ir_properties(self, image: np.ndarray) -> np.ndarray:
        """
        Ensure image maintains IR properties after augmentation.
        
        Args:
            image: Augmented image
            
        Returns:
            np.ndarray: Image with preserved IR properties
        """
        if not self.preserve_ir_properties:
            return image
        
        # Ensure values are in valid range
        image = np.clip(image, 0.0, 1.0)
        
        # Additional IR-specific processing can be added here
        return image


class BaseImageProcessor(IImageProcessor):
    """
    Base implementation providing common image processing functionality.
    """
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        """
        Initialize base processor with target dimensions.
        
        Args:
            target_size: Target image dimensions (height, width)
        """
        self.target_size = target_size
        self.supported_formats = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
    
    def _validate_dimensions(self, image: np.ndarray) -> None:
        """
        Validate image dimensions.
        
        Args:
            image: Image to validate
            
        Raises:
            ValueError: If dimensions are invalid
        """
        if len(image.shape) < 2:
            raise ValueError(f"Image must have at least 2 dimensions, got {len(image.shape)}")


class BaseEmbeddingExtractor(IEmbeddingExtractor):
    """
    Base implementation providing common embedding extraction functionality.
    """
    
    def __init__(self, model_name: str = "resnet50"):
        """
        Initialize base extractor with model configuration.
        
        Args:
            model_name: Name of the model architecture
        """
        self.model_name = model_name
        self.model = None
        self.model_info = {
            'name': model_name,
            'version': '1.0.0',
            'loaded': False
        }
    
    def _validate_model_loaded(self) -> None:
        """
        Ensure model is loaded before extraction.
        
        Raises:
            RuntimeError: If model is not loaded
        """
        if self.model is None:
            raise RuntimeError("Model must be loaded before extracting embeddings")


class BaseVectorStore(IVectorStore):
    """
    Base implementation providing common vector store functionality.
    """
    
    def __init__(self, database_path: str = "./vector_db"):
        """
        Initialize base vector store with database path.
        
        Args:
            database_path: Path to the vector database
        """
        self.database_path = database_path
        self.is_initialized = False
        self.embedding_count = 0
        self.index_type = "hnsw"
    
    def _validate_embedding_vector(self, vector: np.ndarray) -> None:
        """
        Validate embedding vector format.
        
        Args:
            vector: Embedding vector to validate
            
        Raises:
            ValueError: If vector format is invalid
        """
        if not isinstance(vector, np.ndarray):
            raise ValueError("Embedding vector must be a numpy array")
        
        if len(vector.shape) != 1:
            raise ValueError(f"Embedding vector must be 1-dimensional, got shape {vector.shape}")
        
        if np.any(np.isnan(vector)) or np.any(np.isinf(vector)):
            raise ValueError("Embedding vector contains NaN or infinite values")