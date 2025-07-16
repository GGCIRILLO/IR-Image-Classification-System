"""
Image loading and format validation for IR imagery.

This module provides comprehensive image loading capabilities with support for
common image formats, metadata extraction, and robust error handling for
corrupted or invalid images.
"""

import numpy as np
from PIL import Image, ExifTags
from PIL.ExifTags import TAGS
import cv2
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
import logging
from datetime import datetime
import hashlib
import mimetypes

from ..models.data_models import IRImage


class ImageFormatError(Exception):
    """Custom exception for image format-related errors."""
    pass


class ImageCorruptionError(Exception):
    """Custom exception for corrupted image files."""
    pass


class IRImageLoader:
    """
    Comprehensive image loader with format validation and metadata extraction.
    
    Supports common image formats (PNG, JPEG, TIFF, BMP) with robust error
    handling for corrupted files and comprehensive metadata extraction.
    """
    
    def __init__(self):
        """Initialize the image loader with configuration."""
        self.logger = logging.getLogger(__name__)
        
        # Supported image formats and their MIME types
        self.supported_formats = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.bmp': 'image/bmp',
            '.tiff': 'image/tiff',
            '.tif': 'image/tiff'
        }
        
        # Maximum file size (100MB) to prevent memory issues
        self.max_file_size = 100 * 1024 * 1024
        
        # Minimum and maximum image dimensions
        self.min_dimension = 32
        self.max_dimension = 8192
        
        self.logger.info("IRImageLoader initialized")
    
    def load_image(self, file_path: Union[str, Path], 
                   image_id: Optional[str] = None,
                   object_class: str = "") -> IRImage:
        """
        Load and validate an image file, returning an IRImage object.
        
        Args:
            file_path: Path to the image file
            image_id: Optional unique identifier (generated if not provided)
            object_class: Classification label for the image
            
        Returns:
            IRImage: Validated IRImage object
            
        Raises:
            ImageFormatError: If image format is not supported
            ImageCorruptionError: If image file is corrupted
            FileNotFoundError: If file doesn't exist
            ValueError: If image dimensions are invalid
        """
        file_path = Path(file_path)
        
        # Generate image ID if not provided
        if image_id is None:
            image_id = self._generate_image_id(file_path)
        
        self.logger.info(f"Loading image: {file_path}")
        
        # Step 1: Basic file validation
        self._validate_file_exists(file_path)
        self._validate_file_size(file_path)
        self._validate_file_format(file_path)
        
        # Step 2: Load and validate image content
        image_data = self._load_image_data(file_path)
        
        # Step 3: Extract metadata
        metadata = self._extract_metadata(file_path, image_data)
        
        # Step 4: Create IRImage object
        ir_image = IRImage(
            id=image_id,
            image_data=image_data,
            metadata=metadata,
            object_class=object_class,
            created_at=datetime.now()
        )
        
        self.logger.info(f"Successfully loaded image {image_id} from {file_path}")
        return ir_image
    
    def load_images_batch(self, file_paths: List[Union[str, Path]], 
                         object_classes: Optional[List[str]] = None) -> List[IRImage]:
        """
        Load multiple images in batch with error handling.
        
        Args:
            file_paths: List of paths to image files
            object_classes: Optional list of classification labels
            
        Returns:
            List[IRImage]: List of successfully loaded images
        """
        if object_classes is None:
            object_classes = [""] * len(file_paths)
        elif len(object_classes) != len(file_paths):
            raise ValueError("Number of object classes must match number of file paths")
        
        loaded_images = []
        failed_loads = []
        
        for i, (file_path, object_class) in enumerate(zip(file_paths, object_classes)):
            try:
                image = self.load_image(file_path, object_class=object_class)
                loaded_images.append(image)
            except Exception as e:
                failed_loads.append((file_path, str(e)))
                self.logger.error(f"Failed to load {file_path}: {str(e)}")
        
        if failed_loads:
            self.logger.warning(f"Failed to load {len(failed_loads)} out of {len(file_paths)} images")
        
        return loaded_images
    
    def validate_image_format(self, file_path: Union[str, Path]) -> bool:
        """
        Validate image format without loading the full image.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            bool: True if format is valid and supported
        """
        try:
            file_path = Path(file_path)
            
            # Check file extension
            if not self._is_supported_format(file_path):
                return False
            
            # Check MIME type
            mime_type, _ = mimetypes.guess_type(str(file_path))
            expected_mime = self.supported_formats.get(file_path.suffix.lower())
            
            if mime_type != expected_mime:
                self.logger.warning(f"MIME type mismatch for {file_path}: {mime_type} vs {expected_mime}")
            
            # Try to open image header to validate format
            with Image.open(file_path) as img:
                # Just access basic properties to validate format
                _ = img.size
                _ = img.mode
            
            return True
            
        except Exception as e:
            self.logger.error(f"Format validation failed for {file_path}: {str(e)}")
            return False
    
    def get_image_info(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Get basic information about an image without loading full data.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            Dict[str, Any]: Image information
        """
        file_path = Path(file_path)
        
        try:
            with Image.open(file_path) as img:
                info = {
                    'file_path': str(file_path),
                    'file_size': file_path.stat().st_size,
                    'format': img.format,
                    'mode': img.mode,
                    'size': img.size,
                    'width': img.width,
                    'height': img.height,
                    'has_transparency': img.mode in ('RGBA', 'LA') or 'transparency' in img.info,
                    'is_animated': getattr(img, 'is_animated', False)
                }
                
                # Add format-specific information
                if hasattr(img, 'info') and img.info:
                    info['pil_info'] = dict(img.info)
                
                return info
                
        except Exception as e:
            self.logger.error(f"Failed to get image info for {file_path}: {str(e)}")
            return {'error': str(e)}
    
    # Private methods
    
    def _validate_file_exists(self, file_path: Path) -> None:
        """Validate that file exists and is readable."""
        if not file_path.exists():
            raise FileNotFoundError(f"Image file not found: {file_path}")
        
        if not file_path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")
        
        if not file_path.stat().st_size > 0:
            raise ValueError(f"File is empty: {file_path}")
    
    def _validate_file_size(self, file_path: Path) -> None:
        """Validate file size is within acceptable limits."""
        file_size = file_path.stat().st_size
        
        if file_size > self.max_file_size:
            raise ValueError(f"File too large: {file_size} bytes (max: {self.max_file_size})")
    
    def _validate_file_format(self, file_path: Path) -> None:
        """Validate file format is supported."""
        if not self._is_supported_format(file_path):
            supported = ', '.join(self.supported_formats.keys())
            raise ImageFormatError(f"Unsupported format {file_path.suffix}. "
                                 f"Supported formats: {supported}")
    
    def _is_supported_format(self, file_path: Path) -> bool:
        """Check if file format is supported."""
        return file_path.suffix.lower() in self.supported_formats
    
    def _load_image_data(self, file_path: Path) -> np.ndarray:
        """
        Load image data with corruption detection and format conversion.
        
        Returns:
            np.ndarray: Image data as float32 array (0-1 range, grayscale)
        """
        try:
            # Try loading with PIL first (handles most formats well)
            with Image.open(file_path) as img:
                # Validate image is not corrupted
                img.verify()
            
            # Reload image for actual processing (verify() closes the image)
            with Image.open(file_path) as img:
                # Convert to grayscale if needed
                if img.mode != 'L':
                    if img.mode == 'RGBA':
                        # Handle transparency by compositing on black background
                        background = Image.new('RGB', img.size, (0, 0, 0))
                        background.paste(img, mask=img.split()[-1])  # Use alpha channel as mask
                        img = background.convert('L')
                    else:
                        img = img.convert('L')
                
                # Validate dimensions
                width, height = img.size
                if width < self.min_dimension or height < self.min_dimension:
                    raise ValueError(f"Image too small: {width}x{height} "
                                   f"(minimum: {self.min_dimension}x{self.min_dimension})")
                
                if width > self.max_dimension or height > self.max_dimension:
                    raise ValueError(f"Image too large: {width}x{height} "
                                   f"(maximum: {self.max_dimension}x{self.max_dimension})")
                
                # Convert to numpy array
                image_array = np.array(img, dtype=np.float32)
                
                # Normalize to 0-1 range
                if image_array.max() > 1.0:
                    image_array = image_array / 255.0
                
                # Validate array properties
                if np.any(np.isnan(image_array)) or np.any(np.isinf(image_array)):
                    raise ImageCorruptionError("Image contains NaN or infinite values")
                
                return image_array
                
        except Image.UnidentifiedImageError:
            raise ImageCorruptionError(f"Cannot identify image format: {file_path}")
        except Image.DecompressionBombError:
            raise ImageCorruptionError(f"Image too large (decompression bomb): {file_path}")
        except OSError as e:
            if "cannot identify image file" in str(e).lower():
                raise ImageCorruptionError(f"Corrupted or invalid image file: {file_path}")
            else:
                raise ImageCorruptionError(f"Error reading image file: {file_path} - {str(e)}")
        except Exception as e:
            # Try fallback with OpenCV for some edge cases
            try:
                return self._load_with_opencv_fallback(file_path)
            except Exception:
                raise ImageCorruptionError(f"Failed to load image with both PIL and OpenCV: {file_path} - {str(e)}")
    
    def _load_with_opencv_fallback(self, file_path: Path) -> np.ndarray:
        """Fallback image loading using OpenCV."""
        self.logger.warning(f"Using OpenCV fallback for {file_path}")
        
        # Load with OpenCV (automatically converts to grayscale if needed)
        image_array = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
        
        if image_array is None:
            raise ImageCorruptionError(f"OpenCV failed to load image: {file_path}")
        
        # Convert to float32 and normalize
        image_array = image_array.astype(np.float32) / 255.0
        
        return image_array
    
    def _extract_metadata(self, file_path: Path, image_data: np.ndarray) -> Dict[str, Any]:
        """Extract comprehensive metadata from image file and data."""
        metadata = {
            'file_path': str(file_path),
            'file_name': file_path.name,
            'file_size': file_path.stat().st_size,
            'file_extension': file_path.suffix.lower(),
            'created_at': datetime.now().isoformat(),
            'image_shape': image_data.shape,
            'image_dtype': str(image_data.dtype),
            'pixel_count': image_data.size,
            'file_hash': self._calculate_file_hash(file_path)
        }
        
        # Add file timestamps
        stat = file_path.stat()
        metadata.update({
            'file_modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
            'file_created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
        })
        
        # Extract EXIF data if available
        try:
            with Image.open(file_path) as img:
                metadata.update({
                    'pil_format': img.format,
                    'pil_mode': img.mode,
                    'pil_size': img.size
                })
                
                # Extract EXIF data
                exif_data = self._extract_exif_data(img)
                if exif_data:
                    metadata['exif'] = exif_data
                
                # Extract other PIL info
                if hasattr(img, 'info') and img.info:
                    metadata['pil_info'] = {k: v for k, v in img.info.items() 
                                          if isinstance(v, (str, int, float, bool))}
        
        except Exception as e:
            self.logger.warning(f"Failed to extract PIL metadata from {file_path}: {str(e)}")
        
        # Add image statistics
        metadata.update({
            'pixel_min': float(image_data.min()),
            'pixel_max': float(image_data.max()),
            'pixel_mean': float(image_data.mean()),
            'pixel_std': float(image_data.std()),
            'pixel_median': float(np.median(image_data))
        })
        
        return metadata
    
    def _extract_exif_data(self, img: Image.Image) -> Optional[Dict[str, Any]]:
        """Extract EXIF data from PIL Image."""
        try:
            exif_dict = {}
            
            if hasattr(img, '_getexif') and img._getexif() is not None:
                exif = img._getexif()
                
                for tag_id, value in exif.items():
                    tag = TAGS.get(tag_id, tag_id)
                    
                    # Convert bytes to string if needed
                    if isinstance(value, bytes):
                        try:
                            value = value.decode('utf-8')
                        except UnicodeDecodeError:
                            value = str(value)
                    
                    # Only include serializable values
                    if isinstance(value, (str, int, float, bool)):
                        exif_dict[tag] = value
                
                return exif_dict if exif_dict else None
            
        except Exception as e:
            self.logger.warning(f"Failed to extract EXIF data: {str(e)}")
        
        return None
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of the file for integrity checking."""
        try:
            hash_sha256 = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            self.logger.warning(f"Failed to calculate file hash for {file_path}: {str(e)}")
            return ""
    
    def _generate_image_id(self, file_path: Path) -> str:
        """Generate unique image ID based on file path and timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_stem = file_path.stem
        return f"{file_stem}_{timestamp}"


# Convenience functions

def load_ir_image(file_path: Union[str, Path], 
                  image_id: Optional[str] = None,
                  object_class: str = "") -> IRImage:
    """
    Convenience function to load a single IR image.
    
    Args:
        file_path: Path to the image file
        image_id: Optional unique identifier
        object_class: Classification label
        
    Returns:
        IRImage: Loaded and validated IR image
    """
    loader = IRImageLoader()
    return loader.load_image(file_path, image_id, object_class)


def validate_ir_image_format(file_path: Union[str, Path]) -> bool:
    """
    Convenience function to validate image format.
    
    Args:
        file_path: Path to the image file
        
    Returns:
        bool: True if format is valid
    """
    loader = IRImageLoader()
    return loader.validate_image_format(file_path)


def get_ir_image_info(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Convenience function to get image information.
    
    Args:
        file_path: Path to the image file
        
    Returns:
        Dict[str, Any]: Image information
    """
    loader = IRImageLoader()
    return loader.get_image_info(file_path)