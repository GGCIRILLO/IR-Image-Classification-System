"""
Data augmentation engine for IR image classification.

This module implements the DataAugmentationEngine class with configurable strategies
for augmenting military IR imagery while preserving IR characteristics.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Callable
import cv2
from scipy import ndimage
from scipy.ndimage import gaussian_filter
import random
from dataclasses import dataclass
from enum import Enum

from ..models.interfaces import IDataAugmentation
from ..models.data_models import IRImage


class AugmentationType(Enum):
    """Enumeration of available augmentation types."""
    ROTATION = "rotation"
    SCALING = "scaling"
    NOISE = "noise"
    BRIGHTNESS = "brightness"
    CONTRAST = "contrast"
    HORIZONTAL_FLIP = "horizontal_flip"
    ELASTIC_DEFORMATION = "elastic_deformation"


@dataclass
class AugmentationConfig:
    """Configuration for augmentation parameters."""
    rotation_range: Tuple[float, float] = (-15.0, 15.0)  # degrees
    scale_range: Tuple[float, float] = (0.8, 1.2)  # scale factors
    noise_range: Tuple[float, float] = (0.0, 0.1)  # noise levels
    brightness_range: Tuple[float, float] = (0.8, 1.2)  # brightness multipliers
    contrast_range: Tuple[float, float] = (0.8, 1.2)  # contrast multipliers
    flip_probability: float = 0.3  # probability of horizontal flip
    elastic_alpha_range: Tuple[float, float] = (0.5, 2.0)  # elastic deformation strength
    elastic_sigma: float = 0.5  # elastic deformation smoothness
    preserve_ir_properties: bool = True
    target_size: Tuple[int, int] = (224, 224)


class DataAugmentationEngine(IDataAugmentation):
    """
    Configurable data augmentation engine for IR military imagery.
    
    Implements various augmentation strategies while preserving IR characteristics
    such as white objects on black backgrounds and military-relevant features.
    """
    
    def __init__(self, config: Optional[AugmentationConfig] = None):
        """
        Initialize the augmentation engine with configuration.
        
        Args:
            config: Augmentation configuration parameters
        """
        self.config = config or AugmentationConfig()
        self.augmentation_strategies: Dict[AugmentationType, Callable] = {
            AugmentationType.ROTATION: self._apply_rotation,
            AugmentationType.SCALING: self._apply_scaling,
            AugmentationType.NOISE: self._apply_noise,
            AugmentationType.BRIGHTNESS: self._apply_brightness,
            AugmentationType.CONTRAST: self._apply_contrast,
            AugmentationType.HORIZONTAL_FLIP: self._apply_horizontal_flip,
            AugmentationType.ELASTIC_DEFORMATION: self._apply_elastic_deformation,
        }
        
        # Set random seed for reproducibility during testing
        self._random_state = np.random.RandomState(42)
        
        # Military-specific augmentation weights (higher probability for critical augmentations)
        self.military_augmentation_weights = {
            AugmentationType.BRIGHTNESS: 0.8,  # High priority for IR sensor variations
            AugmentationType.CONTRAST: 0.8,   # High priority for IR sensor variations
            AugmentationType.NOISE: 0.7,      # Important for sensor noise simulation
            AugmentationType.ROTATION: 0.6,   # Moderate for different viewing angles
            AugmentationType.SCALING: 0.5,    # Moderate for distance variations
            AugmentationType.ELASTIC_DEFORMATION: 0.4,  # Lower for object integrity
            AugmentationType.HORIZONTAL_FLIP: 0.3,      # Lower for military context
        }
    
    def augment_batch(self, images: List[np.ndarray], target_count: int) -> List[np.ndarray]:
        """
        Augment a batch of images to reach target count.
        
        Args:
            images: List of input images as numpy arrays
            target_count: Desired total number of images after augmentation
            
        Returns:
            List[np.ndarray]: Augmented images including originals
        """
        if not images:
            raise ValueError("Input images list cannot be empty")
        
        if target_count < len(images):
            raise ValueError(f"Target count ({target_count}) must be >= input count ({len(images)})")
        
        # Start with original images
        augmented_images = images.copy()
        
        # Calculate how many augmented images we need to generate
        augmentations_needed = target_count - len(images)
        
        # Generate augmented images
        for _ in range(augmentations_needed):
            # Randomly select a source image index
            source_index = self._random_state.randint(0, len(images))
            source_image = images[source_index]
            
            # Apply random augmentation strategy
            augmented_image = self._apply_random_augmentation(source_image)
            
            # Ensure IR characteristics are preserved
            if self.config.preserve_ir_properties:
                augmented_image = self.preserve_ir_characteristics(augmented_image)
            
            augmented_images.append(augmented_image)
        
        return augmented_images
    
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
        # Ensure image is in valid range [0, 1]
        image = np.clip(image, 0.0, 1.0)
        
        # Preserve IR contrast characteristics
        # IR images typically have white/bright objects on dark backgrounds
        # Ensure the dynamic range is maintained
        
        # Calculate image statistics
        mean_intensity = np.mean(image)
        std_intensity = np.std(image)
        
        # If the image has become too uniform, enhance contrast
        if std_intensity < 0.1:
            # Apply histogram stretching to restore contrast
            min_val = np.min(image)
            max_val = np.max(image)
            if max_val > min_val:
                image = (image - min_val) / (max_val - min_val)
        
        # Ensure proper IR characteristics: objects should be brighter than background
        # Apply a slight bias to maintain the IR appearance
        if mean_intensity > 0.5:
            # Image might be inverted, check and correct if needed
            # For IR images, we expect more dark pixels (background) than bright pixels (objects)
            bright_pixel_ratio = np.sum(image > 0.5) / image.size
            if bright_pixel_ratio > 0.6:
                # Too many bright pixels, might need adjustment
                image = np.clip(image * 0.9, 0.0, 1.0)
        
        return image
    
    def rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """
        Rotate image by specified angle while maintaining IR properties.
        
        Args:
            image: Input image
            angle: Rotation angle in degrees
            
        Returns:
            np.ndarray: Rotated image
        """
        return self._apply_rotation(image, angle)
    
    def scale_image(self, image: np.ndarray, scale_factor: float) -> np.ndarray:
        """
        Scale image by specified factor.
        
        Args:
            image: Input image
            scale_factor: Scaling factor (e.g., 0.8 for 80% size)
            
        Returns:
            np.ndarray: Scaled image
        """
        return self._apply_scaling(image, scale_factor)
    
    def add_noise(self, image: np.ndarray, noise_level: float = 0.1) -> np.ndarray:
        """
        Add realistic noise to simulate IR sensor conditions.
        
        Args:
            image: Input image
            noise_level: Intensity of noise to add (0.0-1.0)
            
        Returns:
            np.ndarray: Image with added noise
        """
        return self._apply_noise(image, noise_level)
    
    def _apply_random_augmentation(self, image: np.ndarray) -> np.ndarray:
        """
        Apply a random combination of augmentation strategies.
        
        Args:
            image: Input image
            
        Returns:
            np.ndarray: Augmented image
        """
        # Randomly select 1-3 augmentation strategies
        num_augmentations = self._random_state.randint(1, 4)
        selected_strategies = self._random_state.choice(
            list(self.augmentation_strategies.keys()),
            size=num_augmentations,
            replace=False
        )
        
        augmented_image = image.copy()
        
        for strategy in selected_strategies:
            augmentation_func = self.augmentation_strategies[strategy]
            augmented_image = augmentation_func(augmented_image)
        
        return augmented_image
    
    def _apply_rotation(self, image: np.ndarray, angle: Optional[float] = None) -> np.ndarray:
        """Apply rotation augmentation."""
        if angle is None:
            angle = self._random_state.uniform(*self.config.rotation_range)
        
        # Use scipy's rotate function which handles edge cases well
        rotated = ndimage.rotate(
            image, 
            angle, 
            reshape=False, 
            mode='constant', 
            cval=0.0,  # Fill with black (background)
            order=1  # Linear interpolation
        )
        
        return rotated
    
    def _apply_scaling(self, image: np.ndarray, scale_factor: Optional[float] = None) -> np.ndarray:
        """Apply scaling augmentation."""
        if scale_factor is None:
            scale_factor = self._random_state.uniform(*self.config.scale_range)
        
        # Calculate new dimensions
        h, w = image.shape[:2]
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        
        # Resize image
        if len(image.shape) == 2:
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Crop or pad to maintain target size
        target_h, target_w = self.config.target_size
        
        if new_h >= target_h and new_w >= target_w:
            # Crop from center
            start_h = (new_h - target_h) // 2
            start_w = (new_w - target_w) // 2
            result = resized[start_h:start_h + target_h, start_w:start_w + target_w]
        else:
            # Pad with zeros (black background)
            result = np.zeros((target_h, target_w), dtype=image.dtype)
            start_h = (target_h - new_h) // 2
            start_w = (target_w - new_w) // 2
            result[start_h:start_h + new_h, start_w:start_w + new_w] = resized
        
        return result
    
    def _apply_noise(self, image: np.ndarray, noise_level: Optional[float] = None) -> np.ndarray:
        """Apply noise augmentation to simulate IR sensor conditions."""
        if noise_level is None:
            noise_level = self._random_state.uniform(*self.config.noise_range)
        
        # Generate Gaussian noise with same dtype as input image
        noise = self._random_state.normal(0, noise_level, image.shape).astype(image.dtype)
        
        # Add noise and clip to valid range
        noisy_image = image + noise
        noisy_image = np.clip(noisy_image, 0.0, 1.0)
        
        return noisy_image.astype(image.dtype)
    
    def _apply_brightness(self, image: np.ndarray, brightness_factor: Optional[float] = None) -> np.ndarray:
        """Apply brightness adjustment for IR imagery."""
        if brightness_factor is None:
            brightness_factor = self._random_state.uniform(*self.config.brightness_range)
        
        # Apply brightness adjustment
        bright_image = image * brightness_factor
        bright_image = np.clip(bright_image, 0.0, 1.0)
        
        return bright_image
    
    def _apply_contrast(self, image: np.ndarray, contrast_factor: Optional[float] = None) -> np.ndarray:
        """Apply contrast adjustment for IR imagery."""
        if contrast_factor is None:
            contrast_factor = self._random_state.uniform(*self.config.contrast_range)
        
        # Apply contrast adjustment around the mean
        mean_intensity = np.mean(image)
        contrast_image = (image - mean_intensity) * contrast_factor + mean_intensity
        contrast_image = np.clip(contrast_image, 0.0, 1.0)
        
        return contrast_image
    
    def _apply_horizontal_flip(self, image: np.ndarray) -> np.ndarray:
        """Apply horizontal flip augmentation."""
        if self._random_state.random() < self.config.flip_probability:
            return np.fliplr(image)
        return image
    
    def _apply_elastic_deformation(self, image: np.ndarray) -> np.ndarray:
        """Apply elastic deformation while preserving object integrity."""
        alpha = self._random_state.uniform(*self.config.elastic_alpha_range)
        sigma = self.config.elastic_sigma
        
        # Generate random displacement fields
        h, w = image.shape[:2]
        dx = gaussian_filter(
            self._random_state.randn(h, w) * alpha, 
            sigma, 
            mode='constant', 
            cval=0
        )
        dy = gaussian_filter(
            self._random_state.randn(h, w) * alpha, 
            sigma, 
            mode='constant', 
            cval=0
        )
        
        # Create coordinate grids
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
        
        # Apply deformation
        deformed = ndimage.map_coordinates(
            image, 
            indices, 
            order=1, 
            mode='constant', 
            cval=0.0
        ).reshape(image.shape)
        
        return deformed
    
    def get_augmentation_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the augmentation configuration.
        
        Returns:
            Dict[str, Any]: Configuration and capability information
        """
        return {
            'available_strategies': [strategy.value for strategy in AugmentationType],
            'rotation_range': self.config.rotation_range,
            'scale_range': self.config.scale_range,
            'noise_range': self.config.noise_range,
            'brightness_range': self.config.brightness_range,
            'contrast_range': self.config.contrast_range,
            'flip_probability': self.config.flip_probability,
            'preserve_ir_properties': self.config.preserve_ir_properties,
            'target_size': self.config.target_size
        }
    
    def create_military_augmentation_pipeline(self, base_images: List[np.ndarray], target_count: int = 15000) -> List[np.ndarray]:
        """
        Create a comprehensive augmentation pipeline specifically for military IR imagery.
        
        Expands 3k base images to 15-18k images using military-relevant augmentation strategies.
        
        Args:
            base_images: List of base IR images (typically ~3k images)
            target_count: Target number of augmented images (default 15k)
            
        Returns:
            List[np.ndarray]: Augmented dataset with military-specific variations
        """
        if len(base_images) == 0:
            raise ValueError("Base images list cannot be empty")
        
        if target_count < len(base_images):
            raise ValueError(f"Target count ({target_count}) must be >= base images count ({len(base_images)})")
        
        print(f"Starting military augmentation pipeline: {len(base_images)} -> {target_count} images")
        
        # Start with original images
        augmented_dataset = base_images.copy()
        
        # Calculate augmentation multiplier
        augmentation_factor = target_count / len(base_images)
        augmentations_per_image = int(augmentation_factor) - 1  # Subtract 1 for original
        remaining_augmentations = target_count - len(base_images) - (augmentations_per_image * len(base_images))
        
        # Apply systematic augmentation for each base image
        for i, base_image in enumerate(base_images):
            # Apply standard augmentations
            for aug_idx in range(augmentations_per_image):
                augmented = self._apply_military_specific_augmentation(base_image, aug_idx)
                augmented_dataset.append(augmented)
            
            # Add extra augmentations for remaining count
            if i < remaining_augmentations:
                extra_augmented = self._apply_military_specific_augmentation(base_image, augmentations_per_image)
                augmented_dataset.append(extra_augmented)
            
            # Progress reporting
            if (i + 1) % 500 == 0:
                print(f"Processed {i + 1}/{len(base_images)} base images")
        
        print(f"Military augmentation pipeline completed: {len(augmented_dataset)} total images")
        return augmented_dataset
    
    def _apply_military_specific_augmentation(self, image: np.ndarray, variation_index: int) -> np.ndarray:
        """
        Apply military-specific augmentation strategies with controlled variation.
        
        Args:
            image: Input IR image
            variation_index: Index to control augmentation variation
            
        Returns:
            np.ndarray: Augmented image with military-specific characteristics
        """
        # Use variation index to ensure diverse but controlled augmentations
        local_random = np.random.RandomState(self._random_state.randint(0, 10000) + variation_index)
        
        augmented = image.copy()
        
        # Apply military-priority augmentations with higher probability
        augmentation_sequence = []
        
        # IR sensor variations (high priority)
        if local_random.random() < self.military_augmentation_weights[AugmentationType.BRIGHTNESS]:
            augmentation_sequence.append(AugmentationType.BRIGHTNESS)
        
        if local_random.random() < self.military_augmentation_weights[AugmentationType.CONTRAST]:
            augmentation_sequence.append(AugmentationType.CONTRAST)
        
        # Environmental noise simulation (high priority)
        if local_random.random() < self.military_augmentation_weights[AugmentationType.NOISE]:
            augmentation_sequence.append(AugmentationType.NOISE)
        
        # Viewing angle variations (moderate priority)
        if local_random.random() < self.military_augmentation_weights[AugmentationType.ROTATION]:
            augmentation_sequence.append(AugmentationType.ROTATION)
        
        # Distance variations (moderate priority)
        if local_random.random() < self.military_augmentation_weights[AugmentationType.SCALING]:
            augmentation_sequence.append(AugmentationType.SCALING)
        
        # Subtle deformations (lower priority, preserve object integrity)
        if local_random.random() < self.military_augmentation_weights[AugmentationType.ELASTIC_DEFORMATION]:
            augmentation_sequence.append(AugmentationType.ELASTIC_DEFORMATION)
        
        # Horizontal flip (lowest priority for military context)
        if local_random.random() < self.military_augmentation_weights[AugmentationType.HORIZONTAL_FLIP]:
            augmentation_sequence.append(AugmentationType.HORIZONTAL_FLIP)
        
        # Apply selected augmentations in sequence
        for aug_type in augmentation_sequence:
            augmentation_func = self.augmentation_strategies[aug_type]
            augmented = augmentation_func(augmented)
        
        # Apply military-specific IR enhancements
        augmented = self._apply_military_ir_enhancements(augmented, local_random)
        
        # Ensure IR characteristics are preserved
        if self.config.preserve_ir_properties:
            augmented = self.preserve_ir_characteristics(augmented)
        
        return augmented
    
    def _apply_military_ir_enhancements(self, image: np.ndarray, random_state: np.random.RandomState) -> np.ndarray:
        """
        Apply military-specific IR enhancements to simulate real-world conditions.
        
        Args:
            image: Input IR image
            random_state: Random state for reproducible results
            
        Returns:
            np.ndarray: Enhanced IR image
        """
        enhanced = image.copy()
        
        # Simulate atmospheric effects (heat shimmer, humidity)
        if random_state.random() < 0.3:
            enhanced = self._simulate_atmospheric_effects(enhanced, random_state)
        
        # Simulate thermal crossover effects (temperature inversion)
        if random_state.random() < 0.2:
            enhanced = self._simulate_thermal_crossover(enhanced, random_state)
        
        # Simulate sensor saturation in hot spots
        if random_state.random() < 0.25:
            enhanced = self._simulate_sensor_saturation(enhanced, random_state)
        
        # Simulate range-dependent noise
        if random_state.random() < 0.4:
            enhanced = self._simulate_range_dependent_noise(enhanced, random_state)
        
        return enhanced
    
    def _simulate_atmospheric_effects(self, image: np.ndarray, random_state: np.random.RandomState) -> np.ndarray:
        """Simulate atmospheric effects like heat shimmer and humidity."""
        # Add subtle blur to simulate atmospheric distortion
        blur_strength = random_state.uniform(0.3, 0.8)
        blurred = gaussian_filter(image, sigma=blur_strength)
        
        # Blend with original to maintain object definition
        blend_factor = random_state.uniform(0.1, 0.3)
        atmospheric = (1 - blend_factor) * image + blend_factor * blurred
        
        return atmospheric.astype(image.dtype)
    
    def _simulate_thermal_crossover(self, image: np.ndarray, random_state: np.random.RandomState) -> np.ndarray:
        """Simulate thermal crossover effects where hot objects appear cooler."""
        # Apply subtle inversion to random regions
        crossover_strength = random_state.uniform(0.05, 0.15)
        
        # Create random mask for crossover regions
        mask = random_state.random(image.shape) < 0.1  # 10% of pixels affected
        
        crossover_image = image.copy()
        crossover_image[mask] = 1.0 - crossover_image[mask]
        
        # Blend with original
        result = (1 - crossover_strength) * image + crossover_strength * crossover_image
        
        return np.clip(result, 0.0, 1.0)
    
    def _simulate_sensor_saturation(self, image: np.ndarray, random_state: np.random.RandomState) -> np.ndarray:
        """Simulate sensor saturation in very hot regions."""
        # Find bright regions that might saturate
        saturation_threshold = random_state.uniform(0.8, 0.95)
        saturation_mask = image > saturation_threshold
        
        # Apply saturation effect
        saturated = image.copy()
        saturated[saturation_mask] = 1.0  # Full saturation
        
        # Add slight blooming effect around saturated regions
        if np.any(saturation_mask):
            # Dilate the saturation mask slightly
            from scipy.ndimage import binary_dilation
            blooming_mask = binary_dilation(saturation_mask, iterations=1)
            blooming_strength = random_state.uniform(0.1, 0.3)
            saturated[blooming_mask] = np.clip(
                saturated[blooming_mask] + blooming_strength, 0.0, 1.0
            )
        
        return saturated
    
    def _simulate_range_dependent_noise(self, image: np.ndarray, random_state: np.random.RandomState) -> np.ndarray:
        """Simulate noise that increases with range/distance."""
        # Create distance-dependent noise pattern
        h, w = image.shape[:2]
        
        # Create radial distance from center (simulating range)
        center_y, center_x = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        distance_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # Normalize distance
        max_distance = np.sqrt(center_x**2 + center_y**2)
        normalized_distance = distance_from_center / max_distance
        
        # Generate range-dependent noise
        base_noise_level = random_state.uniform(0.01, 0.05)
        range_noise_factor = random_state.uniform(0.02, 0.08)
        
        noise_level = base_noise_level + range_noise_factor * normalized_distance
        noise = random_state.normal(0, noise_level, image.shape).astype(image.dtype)
        
        # Apply noise
        noisy_image = image + noise
        
        return np.clip(noisy_image, 0.0, 1.0).astype(image.dtype)
    
    def get_military_augmentation_stats(self) -> Dict[str, Any]:
        """
        Get statistics about military-specific augmentation capabilities.
        
        Returns:
            Dict[str, Any]: Military augmentation configuration and weights
        """
        stats = self.get_augmentation_stats()
        stats.update({
            'military_augmentation_weights': self.military_augmentation_weights,
            'military_enhancements': [
                'atmospheric_effects',
                'thermal_crossover',
                'sensor_saturation',
                'range_dependent_noise'
            ],
            'typical_expansion_ratio': '3k -> 15-18k images (5-6x expansion)',
            'military_priority_augmentations': [
                'brightness_contrast_variations',
                'sensor_noise_simulation',
                'environmental_effects'
            ]
        })
        return stats
    
    def set_random_seed(self, seed: int) -> None:
        """
        Set random seed for reproducible augmentations.
        
        Args:
            seed: Random seed value
        """
        self._random_state = np.random.RandomState(seed)
        random.seed(seed)
        np.random.seed(seed)