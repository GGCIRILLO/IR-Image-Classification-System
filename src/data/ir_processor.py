"""
IR Image Processor implementation for military infrared imagery.

This module provides concrete implementation of IR-specific image preprocessing,
validation, and enhancement operations optimized for military applications.
"""

import numpy as np
import cv2
from typing import Tuple, Optional, Dict, Any
from PIL import Image, ImageEnhance, ImageFilter
import logging
from pathlib import Path

from ..models.interfaces import IImageProcessor
from ..models.data_models import IRImage


class IRImageProcessor(IImageProcessor):
    """
    Concrete implementation of IR image processing for military applications.
    
    Handles IR-specific preprocessing including contrast enhancement, noise reduction,
    validation for IR format (white objects on black background), and standardization.
    """
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224), 
                 preserve_aspect_ratio: bool = False):
        """
        Initialize IR image processor with configuration.
        
        Args:
            target_size: Target dimensions for processed images (height, width)
            preserve_aspect_ratio: Whether to maintain aspect ratio during resize
        """
        super().__init__()  # Call ABC's __init__ without parameters
        self.target_size = target_size
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.logger = logging.getLogger(__name__)
        
        # IR-specific processing parameters
        self.contrast_enhancement_factor = 1.5
        self.noise_reduction_kernel_size = 3
        self.brightness_threshold = 0.3  # For IR validation (white objects)
        self.background_threshold = 0.1  # For IR validation (black background)
        
        # Supported image formats for IR imagery
        self.supported_formats = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
        
        self.logger.info(f"IRImageProcessor initialized with target size {target_size}")
    
    def preprocess_ir_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply comprehensive IR-specific preprocessing pipeline.
        
        Pipeline includes:
        1. Contrast enhancement for better object visibility
        2. Noise reduction to clean sensor artifacts
        3. Normalization for consistent processing
        4. Resize to standard dimensions
        
        Args:
            image: Raw IR image as numpy array (0-255 or 0-1 range)
            
        Returns:
            np.ndarray: Preprocessed IR image (0-1 range, target dimensions)
        """
        self.logger.debug(f"Preprocessing IR image with shape {image.shape}")
        
        # Ensure image is in proper format
        if image.dtype != np.float32:
            if image.max() > 1.0:
                # Convert from 0-255 to 0-1 range
                image = image.astype(np.float32) / 255.0
            else:
                image = image.astype(np.float32)
        
        # Step 1: Enhance contrast for better object visibility
        enhanced_image = self.enhance_contrast(image)
        
        # Step 2: Apply noise reduction
        denoised_image = self.reduce_noise(enhanced_image)
        
        # Step 3: Normalize the image
        normalized_image = self.normalize_image(denoised_image)
        
        # Step 4: Resize to standard dimensions
        resized_image = self.resize_to_standard(normalized_image, self.target_size)
        
        self.logger.debug(f"Preprocessing complete. Output shape: {resized_image.shape}")
        return resized_image
    
    def validate_ir_format(self, image: np.ndarray) -> bool:
        """
        Validate that image meets IR format requirements.
        
        IR format requirements:
        - White objects on black background (high contrast)
        - Proper intensity distribution (bright objects, dark background)
        - Reasonable signal-to-noise ratio
        
        Args:
            image: Image to validate (any range)
            
        Returns:
            bool: True if image meets IR format requirements
        """
        try:
            # Normalize image to 0-1 range for consistent analysis
            if image.max() > 1.0:
                normalized = image.astype(np.float32) / 255.0
            else:
                normalized = image.astype(np.float32)
            
            # Check 1: Image should have reasonable contrast
            contrast_ratio = self._calculate_contrast_ratio(normalized)
            if contrast_ratio < 0.3:  # Minimum contrast for IR imagery
                self.logger.warning(f"Low contrast ratio: {contrast_ratio:.3f}")
                return False
            
            # Check 2: Should have bright objects (white) on dark background
            bright_pixel_ratio = self._calculate_bright_pixel_ratio(normalized)
            if bright_pixel_ratio < 0.05 or bright_pixel_ratio > 0.7:
                # Too few bright pixels (no objects) or too many (not IR-like)
                self.logger.warning(f"Invalid bright pixel ratio: {bright_pixel_ratio:.3f}")
                return False
            
            # Check 3: Background should be predominantly dark
            background_darkness = self._calculate_background_darkness(normalized)
            if background_darkness < 0.6:
                self.logger.warning(f"Background not dark enough: {background_darkness:.3f}")
                return False
            
            # Check 4: Image should not be too noisy
            noise_level = self._estimate_noise_level(normalized)
            if noise_level > 0.15:  # Maximum acceptable noise level
                self.logger.warning(f"High noise level: {noise_level:.3f}")
                return False
            
            self.logger.debug("IR format validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Error during IR format validation: {str(e)}")
            return False
    
    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance contrast using adaptive histogram equalization and gamma correction.
        
        Uses CLAHE (Contrast Limited Adaptive Histogram Equalization) which is
        particularly effective for IR imagery with varying local contrast.
        
        Args:
            image: Input IR image (0-1 range)
            
        Returns:
            np.ndarray: Contrast-enhanced image
        """
        # Convert to uint8 for OpenCV processing
        image_uint8 = (image * 255).astype(np.uint8)
        
        # Apply CLAHE for adaptive contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_uint8 = clahe.apply(image_uint8)
        
        # Convert back to float32 and 0-1 range
        enhanced = enhanced_uint8.astype(np.float32) / 255.0
        
        # Apply gamma correction for IR imagery (gamma < 1 brightens mid-tones)
        gamma = 0.8
        enhanced = np.power(enhanced, gamma)
        
        # Ensure values stay in valid range
        enhanced = np.clip(enhanced, 0.0, 1.0)
        
        return enhanced
    
    def reduce_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Apply noise reduction techniques suitable for IR imagery.
        
        Uses bilateral filtering which preserves edges while reducing noise,
        making it ideal for IR images where object boundaries are critical.
        
        Args:
            image: Noisy IR image (0-1 range)
            
        Returns:
            np.ndarray: Denoised image
        """
        # Convert to uint8 for OpenCV processing
        image_uint8 = (image * 255).astype(np.uint8)
        
        # Apply bilateral filter - preserves edges while reducing noise
        # Parameters: d=9 (neighborhood diameter), sigmaColor=75, sigmaSpace=75
        denoised_uint8 = cv2.bilateralFilter(image_uint8, 9, 75, 75)
        
        # Convert back to float32 and 0-1 range
        denoised = denoised_uint8.astype(np.float32) / 255.0
        
        return denoised
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image values for consistent processing.
        
        Applies min-max normalization to ensure consistent 0-1 range
        while preserving the relative intensity relationships.
        
        Args:
            image: Input image (any range)
            
        Returns:
            np.ndarray: Normalized image (0-1 range)
        """
        # Handle edge case of constant image
        if image.min() == image.max():
            return np.zeros_like(image, dtype=np.float32)
        
        # Min-max normalization
        normalized = (image - image.min()) / (image.max() - image.min())
        
        return normalized.astype(np.float32)
    
    def resize_to_standard(self, image: np.ndarray, 
                          target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """
        Resize image to standard dimensions for model input.
        
        Uses high-quality resampling (Lanczos) to preserve image quality
        during resize operations.
        
        Args:
            image: Input image (0-1 range)
            target_size: Target dimensions (height, width)
            
        Returns:
            np.ndarray: Resized image
        """
        current_height, current_width = image.shape[:2]
        target_height, target_width = target_size
        
        # Skip resize if already correct size
        if current_height == target_height and current_width == target_width:
            return image
        
        # Convert to PIL Image for high-quality resize
        if image.max() <= 1.0:
            pil_image = Image.fromarray((image * 255).astype(np.uint8), mode='L')
        else:
            pil_image = Image.fromarray(image.astype(np.uint8), mode='L')
        
        # Resize using high-quality Lanczos resampling
        if self.preserve_aspect_ratio:
            # Calculate new size maintaining aspect ratio
            aspect_ratio = current_width / current_height
            if aspect_ratio > 1:  # Wider than tall
                new_width = target_width
                new_height = int(target_width / aspect_ratio)
            else:  # Taller than wide
                new_height = target_height
                new_width = int(target_height * aspect_ratio)
            
            resized_pil = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Pad to target size with black (appropriate for IR background)
            padded_pil = Image.new('L', (target_width, target_height), 0)
            paste_x = (target_width - new_width) // 2
            paste_y = (target_height - new_height) // 2
            padded_pil.paste(resized_pil, (paste_x, paste_y))
            
            resized_array = np.array(padded_pil, dtype=np.float32) / 255.0
        else:
            # Direct resize to target dimensions
            resized_pil = pil_image.resize((target_width, target_height), Image.Resampling.LANCZOS)
            resized_array = np.array(resized_pil, dtype=np.float32) / 255.0
        
        return resized_array
    
    def load_and_validate_image(self, file_path: str) -> Optional[np.ndarray]:
        """
        Load image from file and validate format.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            Optional[np.ndarray]: Loaded and validated image, None if invalid
        """
        try:
            # Check file extension
            file_ext = Path(file_path).suffix.lower()
            if file_ext not in self.supported_formats:
                self.logger.error(f"Unsupported format {file_ext} for file {file_path}")
                return None
            
            # Load image
            with Image.open(file_path) as img:
                # Convert to grayscale if needed
                if img.mode != 'L':
                    img = img.convert('L')
                
                # Convert to numpy array
                image_array = np.array(img, dtype=np.float32)
                
                # Normalize to 0-1 range
                if image_array.max() > 1.0:
                    image_array = image_array / 255.0
                
                # Validate IR format
                if not self.validate_ir_format(image_array):
                    self.logger.warning(f"Image {file_path} failed IR format validation")
                    return None
                
                self.logger.info(f"Successfully loaded and validated image {file_path}")
                return image_array
                
        except Exception as e:
            self.logger.error(f"Failed to load image {file_path}: {str(e)}")
            return None
    
    def get_processing_stats(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Get detailed statistics about image processing.
        
        Args:
            image: Processed image
            
        Returns:
            Dict[str, Any]: Processing statistics
        """
        return {
            'shape': image.shape,
            'dtype': str(image.dtype),
            'min_value': float(image.min()),
            'max_value': float(image.max()),
            'mean_value': float(image.mean()),
            'std_value': float(image.std()),
            'contrast_ratio': self._calculate_contrast_ratio(image),
            'bright_pixel_ratio': self._calculate_bright_pixel_ratio(image),
            'background_darkness': self._calculate_background_darkness(image),
            'estimated_noise': self._estimate_noise_level(image)
        }
    
    # Private helper methods
    
    def _calculate_contrast_ratio(self, image: np.ndarray) -> float:
        """Calculate contrast ratio (max - min) / (max + min)."""
        min_val, max_val = image.min(), image.max()
        if max_val + min_val == 0:
            return 0.0
        return (max_val - min_val) / (max_val + min_val)
    
    def _calculate_bright_pixel_ratio(self, image: np.ndarray) -> float:
        """Calculate ratio of bright pixels (above threshold)."""
        bright_pixels = np.sum(image > self.brightness_threshold)
        total_pixels = image.size
        return bright_pixels / total_pixels
    
    def _calculate_background_darkness(self, image: np.ndarray) -> float:
        """Calculate ratio of dark background pixels."""
        dark_pixels = np.sum(image < self.background_threshold)
        total_pixels = image.size
        return dark_pixels / total_pixels
    
    def _estimate_noise_level(self, image: np.ndarray) -> float:
        """Estimate noise level using Laplacian variance."""
        # Convert to uint8 for OpenCV
        image_uint8 = (image * 255).astype(np.uint8)
        
        # Calculate Laplacian variance (measure of image sharpness/noise)
        laplacian_var = cv2.Laplacian(image_uint8, cv2.CV_64F).var()
        
        # Normalize to 0-1 range (empirically determined scaling)
        normalized_noise = min(laplacian_var / 1000.0, 1.0)
        
        return normalized_noise