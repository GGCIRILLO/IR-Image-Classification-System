"""
Core data models for the IR Image Classification System.

This module contains the primary data structures used throughout the system,
including IRImage, Embedding, SimilarityResult, and QueryResult dataclasses.
All models include validation methods and proper type hints.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from PIL import Image
import io

# Import object classification system
from .object_classes import ObjectClass, ObjectCategory, OBJECT_REGISTRY


@dataclass
class IRImage:
    """
    Represents an infrared image with metadata and validation.
    
    Attributes:
        id: Unique identifier for the image
        image_data: Raw image data as numpy array
        metadata: Additional information about the image
        object_class: Classification label for the object
        confidence_score: Confidence in the classification (0.0-1.0)
        created_at: Timestamp when the image was created/processed
    """
    id: str
    image_data: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)
    object_class: str = ""
    confidence_score: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate the IR image after initialization."""
        self.validate()
    
    def validate(self) -> bool:
        """
        Validate IR image format requirements.
        
        Requirements:
        - Image must be resized to 224x224
        - Image must be in grayscale
        - Confidence score must be between 0.0 and 1.0
        
        Returns:
            bool: True if validation passes
            
        Raises:
            ValueError: If validation fails
        """
        # Check if image_data is valid numpy array
        if not isinstance(self.image_data, np.ndarray):
            raise ValueError("image_data must be a numpy array")
        
        # Check image dimensions (should be 224x224 for grayscale or 224x224x1)
        if len(self.image_data.shape) == 2:
            height, width = self.image_data.shape
            if height != 224 or width != 224:
                raise ValueError(f"Image must be 224x224 pixels, got {height}x{width}")
        elif len(self.image_data.shape) == 3:
            height, width, channels = self.image_data.shape
            if height != 224 or width != 224:
                raise ValueError(f"Image must be 224x224 pixels, got {height}x{width}")
            if channels != 1:
                raise ValueError(f"Image must be grayscale (1 channel), got {channels} channels")
        else:
            raise ValueError(f"Invalid image shape: {self.image_data.shape}")
        
        # Check confidence score range
        if not 0.0 <= self.confidence_score <= 1.0:
            raise ValueError(f"Confidence score must be between 0.0 and 1.0, got {self.confidence_score}")
        
        return True
    
    @classmethod
    def from_file_path(cls, file_path: str, image_id: str, object_class: str = "") -> 'IRImage':
        """
        Create IRImage from file path with format validation.
        
        Args:
            file_path: Path to the image file
            image_id: Unique identifier for the image
            object_class: Classification label
            
        Returns:
            IRImage: Validated IRImage instance
            
        Raises:
            ValueError: If file format is not supported or validation fails
        """
        # Validate file extension
        supported_formats = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
        file_ext = file_path.lower().split('.')[-1]
        if f'.{file_ext}' not in supported_formats:
            raise ValueError(f"Unsupported image format: {file_ext}. "
                           f"Supported formats: {', '.join(supported_formats)}")
        
        # Load and process image
        try:
            with Image.open(file_path) as img:
                # Convert to grayscale if needed
                if img.mode != 'L':
                    img = img.convert('L')
                
                # Resize to 224x224
                img = img.resize((224, 224), Image.Resampling.LANCZOS)
                
                # Convert to numpy array
                image_data = np.array(img, dtype=np.float32)
                
                # Normalize to 0-1 range
                image_data = image_data / 255.0
                
        except Exception as e:
            raise ValueError(f"Failed to load image from {file_path}: {str(e)}")
        
        metadata = {
            'file_path': file_path,
            'original_format': file_ext,
            'processed_at': datetime.now().isoformat()
        }
        
        return cls(
            id=image_id,
            image_data=image_data,
            metadata=metadata,
            object_class=object_class,
            created_at=datetime.now()
        )
    
    def to_pil_image(self) -> Image.Image:
        """Convert the numpy array back to PIL Image for display."""
        # Denormalize from 0-1 to 0-255
        image_data_uint8 = (self.image_data * 255).astype(np.uint8)
        return Image.fromarray(image_data_uint8, mode='L')


@dataclass
class Embedding:
    """
    Represents a feature embedding extracted from an IR image.
    
    Attributes:
        id: Unique identifier for the embedding
        vector: The embedding vector as numpy array
        image_id: Reference to the source image
        model_version: Version of the model used for extraction
        extraction_timestamp: When the embedding was created
    """
    id: str
    vector: np.ndarray
    image_id: str
    model_version: str
    extraction_timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate the embedding after initialization."""
        self.validate()
    
    def validate(self) -> bool:
        """
        Validate embedding format and properties.
        
        Returns:
            bool: True if validation passes
            
        Raises:
            ValueError: If validation fails
        """
        # Check if vector is valid numpy array
        if not isinstance(self.vector, np.ndarray):
            raise ValueError("vector must be a numpy array")
        
        # Check if vector is 1-dimensional
        if len(self.vector.shape) != 1:
            raise ValueError(f"Embedding vector must be 1-dimensional, got shape {self.vector.shape}")
        
        # Check if vector has reasonable size (typically 512, 768, or 1024 for common models)
        if self.vector.shape[0] < 64 or self.vector.shape[0] > 4096:
            raise ValueError(f"Embedding vector size seems unusual: {self.vector.shape[0]}")
        
        # Check for NaN or infinite values
        if np.any(np.isnan(self.vector)) or np.any(np.isinf(self.vector)):
            raise ValueError("Embedding vector contains NaN or infinite values")
        
        return True
    
    def normalize(self) -> 'Embedding':
        """
        Return a new Embedding with L2-normalized vector.
        
        Returns:
            Embedding: New embedding with normalized vector
        """
        norm = np.linalg.norm(self.vector)
        if norm == 0:
            raise ValueError("Cannot normalize zero vector")
        
        normalized_vector = self.vector / norm
        
        return Embedding(
            id=f"{self.id}_normalized",
            vector=normalized_vector,
            image_id=self.image_id,
            model_version=self.model_version,
            extraction_timestamp=self.extraction_timestamp
        )
    
    def cosine_similarity(self, other: 'Embedding') -> float:
        """
        Calculate cosine similarity with another embedding.
        
        Args:
            other: Another embedding to compare with
            
        Returns:
            float: Cosine similarity score (-1 to 1)
        """
        if self.vector.shape != other.vector.shape:
            raise ValueError("Embedding vectors must have the same shape")
        
        # Normalize vectors
        norm_a = np.linalg.norm(self.vector)
        norm_b = np.linalg.norm(other.vector)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return np.dot(self.vector, other.vector) / (norm_a * norm_b)


@dataclass
class SimilarityResult:
    """
    Represents a single similarity search result.
    
    Attributes:
        image_id: ID of the similar image found
        similarity_score: Similarity score (0.0-1.0, higher is more similar)
        confidence: Confidence in the result (0.0-1.0)
        object_class: Classification of the similar object
        metadata: Additional information about the result
    """
    image_id: str
    similarity_score: float
    confidence: float
    object_class: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate the similarity result after initialization."""
        self.validate()
    
    def validate(self) -> bool:
        """
        Validate similarity result values.
        
        Returns:
            bool: True if validation passes
            
        Raises:
            ValueError: If validation fails
        """
        # Check similarity score range
        if not 0.0 <= self.similarity_score <= 1.0:
            raise ValueError(f"Similarity score must be between 0.0 and 1.0, got {self.similarity_score}")
        
        # Check confidence range
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")
        
        return True
    
    def get_object_class_info(self) -> Optional[ObjectClass]:
        """
        Get detailed object class information from the registry.
        
        Returns:
            ObjectClass: Object class details if found, None otherwise
        """
        # Try to get by standardized name first
        obj_class = OBJECT_REGISTRY.get_class_by_name(self.object_class)
        if obj_class:
            return obj_class
        
        # Try to get by folder name
        obj_class = OBJECT_REGISTRY.get_class_by_folder(self.object_class)
        if obj_class:
            return obj_class
        
        # Try fuzzy matching - extract the main object type from the class name
        # Handle cases like "Jeep jeep (20) orig" -> "Jeep"
        cleaned_class = self._extract_main_object_type(self.object_class)
        if cleaned_class != self.object_class:
            # Try again with cleaned name
            obj_class = OBJECT_REGISTRY.get_class_by_folder(cleaned_class)
            if obj_class:
                return obj_class
        
        # Try partial matching against all folder names
        for folder_name in OBJECT_REGISTRY.get_folder_names():
            if folder_name.lower() in self.object_class.lower() or self.object_class.lower() in folder_name.lower():
                return OBJECT_REGISTRY.get_class_by_folder(folder_name)
        
        return None
    
    def _extract_main_object_type(self, object_class: str) -> str:
        """
        Extract the main object type from a complex object class name.
        
        Examples:
            "Jeep jeep (20) orig" -> "Jeep"
            "M-981 Mobile Air Defense m981 mobile air defense (30) orig" -> "M-981 Mobile Air Defense"
            "BMP-2 APC Tank bmp2 apc tank (11) orig" -> "BMP-2 APC Tank"
        
        Args:
            object_class: Original object class string
            
        Returns:
            str: Cleaned object class name
        """
        # Remove common suffixes
        cleaned = object_class
        
        # Remove " orig" and " aug" suffixes
        for suffix in [" orig", " aug"]:
            if cleaned.endswith(suffix):
                cleaned = cleaned[:-len(suffix)]
        
        # Remove parenthetical numbers like " (20)"
        import re
        cleaned = re.sub(r'\s*\(\d+\)', '', cleaned)
        
        # Split by spaces and look for the pattern where the first part is repeated
        words = cleaned.split()
        if len(words) > 1:
            # Find where the repetition starts
            first_part = []
            seen_words = set()
            
            for word in words:
                word_lower = word.lower()
                # If we've seen this word before (case-insensitive), we've hit the repetition
                if word_lower in seen_words:
                    break
                seen_words.add(word_lower)
                first_part.append(word)
            
            if first_part:
                return ' '.join(first_part)
        
        return cleaned
    
    def get_object_category(self) -> Optional[ObjectCategory]:
        """
        Get the high-level category for this object class.
        
        Returns:
            ObjectCategory: Category if object class is recognized, None otherwise
        """
        obj_class = self.get_object_class_info()
        return obj_class.category if obj_class else None
    
    def is_military_asset(self) -> bool:
        """
        Check if this result represents a military asset.
        
        Returns:
            bool: True if the object is a military asset
        """
        category = self.get_object_category()
        if not category:
            return False
        
        military_categories = {
            ObjectCategory.MILITARY_VEHICLE,
            ObjectCategory.AIR_DEFENSE,
            ObjectCategory.MISSILE_SYSTEM
        }
        return category in military_categories
    
    def is_critical_asset(self) -> bool:
        """
        Check if this result represents a critical/high-value asset.
        
        Returns:
            bool: True if the object is considered critical
        """
        obj_class = self.get_object_class_info()
        if not obj_class:
            return False
        
        # Define critical asset keywords
        critical_keywords = [
            'tank', 'missile', 'radar', 'launcher', 'defense', 'aircraft',
            'artillery', 'scud', 'patriot', 'thaad', 'tomahawk'
        ]
        
        class_name_lower = obj_class.name.lower()
        return any(keyword in class_name_lower for keyword in critical_keywords)
    
    def get_threat_level(self) -> str:
        """
        Assess threat level based on object type and confidence.
        
        Returns:
            str: Threat level ('CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'UNKNOWN')
        """
        if not self.get_object_class_info():
            return 'UNKNOWN'
        
        # Critical assets with high confidence
        if self.is_critical_asset() and self.confidence >= 0.8:
            return 'CRITICAL'
        
        # Military assets with high confidence
        if self.is_military_asset() and self.confidence >= 0.7:
            return 'HIGH'
        
        # Any military asset or critical with medium confidence
        if (self.is_military_asset() or self.is_critical_asset()) and self.confidence >= 0.5:
            return 'MEDIUM'
        
        # Low confidence or civilian assets
        return 'LOW'


@dataclass
class QueryResult:
    """
    Represents the complete result of a similarity query.
    
    Attributes:
        query_id: Unique identifier for the query
        results: List of similarity results (top-K matches)
        processing_time: Time taken to process the query in seconds
        model_version: Version of the model used
        timestamp: When the query was executed
    """
    query_id: str
    results: List[SimilarityResult]
    processing_time: float
    model_version: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate the query result after initialization."""
        self.validate()
    
    def validate(self) -> bool:
        """
        Validate query result structure and values.
        
        Returns:
            bool: True if validation passes
            
        Raises:
            ValueError: If validation fails
        """
        # Check processing time is non-negative
        if self.processing_time < 0:
            raise ValueError(f"Processing time must be non-negative, got {self.processing_time}")
        
        # Check that results list is not empty for successful queries
        if not isinstance(self.results, list):
            raise ValueError("Results must be a list")
        
        # Validate each result
        for i, result in enumerate(self.results):
            if not isinstance(result, SimilarityResult):
                raise ValueError(f"Result {i} must be a SimilarityResult instance")
            result.validate()
        
        return True
    
    def get_top_k(self, k: int = 5) -> List[SimilarityResult]:
        """
        Get the top-K results sorted by similarity score.
        
        Args:
            k: Number of top results to return
            
        Returns:
            List[SimilarityResult]: Top-K results sorted by similarity (descending)
        """
        sorted_results = sorted(self.results, key=lambda x: x.similarity_score, reverse=True)
        return sorted_results[:k]
    
    def get_high_confidence_results(self, min_confidence: float = 0.8) -> List[SimilarityResult]:
        """
        Get results with confidence above threshold.
        
        Args:
            min_confidence: Minimum confidence threshold
            
        Returns:
            List[SimilarityResult]: Results with high confidence
        """
        return [result for result in self.results if result.confidence >= min_confidence]