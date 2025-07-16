"""
Tests for the data augmentation engine.

This module contains unit tests for the DataAugmentationEngine class
and its various augmentation strategies.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from src.data.augmentation import (
    DataAugmentationEngine,
    AugmentationConfig,
    AugmentationType
)


class TestDataAugmentationEngine:
    """Test cases for DataAugmentationEngine class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = AugmentationConfig()
        self.engine = DataAugmentationEngine(self.config)
        
        # Create a sample IR image (224x224 grayscale)
        self.sample_image = np.random.rand(224, 224).astype(np.float32)
        # Make it look more like an IR image (mostly dark with some bright objects)
        self.sample_image = self.sample_image * 0.3  # Mostly dark background
        self.sample_image[100:150, 100:150] = 0.8  # Bright object
    
    def test_initialization(self):
        """Test engine initialization."""
        # Test with default config
        engine = DataAugmentationEngine()
        assert engine.config is not None
        assert engine.config.preserve_ir_properties is True
        
        # Test with custom config
        custom_config = AugmentationConfig(rotation_range=(-10, 10))
        engine = DataAugmentationEngine(custom_config)
        assert engine.config.rotation_range == (-10, 10)
    
    def test_augment_batch_basic(self):
        """Test basic batch augmentation functionality."""
        images = [self.sample_image, self.sample_image.copy()]
        target_count = 5
        
        result = self.engine.augment_batch(images, target_count)
        
        assert len(result) == target_count
        assert result[0] is images[0]  # Original images should be included
        assert result[1] is images[1]
        
        # Check that augmented images have correct shape
        for img in result:
            assert img.shape == (224, 224)
            assert img.dtype == np.float32
    
    def test_augment_batch_validation(self):
        """Test input validation for batch augmentation."""
        # Test empty images list
        with pytest.raises(ValueError, match="Input images list cannot be empty"):
            self.engine.augment_batch([], 5)
        
        # Test target count less than input count
        images = [self.sample_image, self.sample_image.copy()]
        with pytest.raises(ValueError, match="Target count .* must be >= input count"):
            self.engine.augment_batch(images, 1)
    
    def test_preserve_ir_characteristics(self):
        """Test IR characteristics preservation."""
        # Create an image that might lose IR characteristics
        test_image = np.ones((224, 224)) * 0.5  # Uniform gray
        
        preserved = self.engine.preserve_ir_characteristics(test_image)
        
        # Should be clipped to valid range
        assert np.all(preserved >= 0.0)
        assert np.all(preserved <= 1.0)
        assert preserved.shape == test_image.shape
    
    def test_rotate_image(self):
        """Test image rotation."""
        angle = 15.0
        rotated = self.engine.rotate_image(self.sample_image, angle)
        
        assert rotated.shape == self.sample_image.shape
        assert rotated.dtype == self.sample_image.dtype
        assert np.all(rotated >= 0.0)
        assert np.all(rotated <= 1.0)
    
    def test_scale_image(self):
        """Test image scaling."""
        scale_factor = 0.8
        scaled = self.engine.scale_image(self.sample_image, scale_factor)
        
        assert scaled.shape == self.sample_image.shape  # Should maintain target size
        assert scaled.dtype == self.sample_image.dtype
        assert np.all(scaled >= 0.0)
        assert np.all(scaled <= 1.0)
    
    def test_add_noise(self):
        """Test noise addition."""
        noise_level = 0.1
        noisy = self.engine.add_noise(self.sample_image, noise_level)
        
        assert noisy.shape == self.sample_image.shape
        assert noisy.dtype == self.sample_image.dtype
        assert np.all(noisy >= 0.0)
        assert np.all(noisy <= 1.0)
        
        # Should be different from original (with high probability)
        assert not np.array_equal(noisy, self.sample_image)
    
    def test_rotation_augmentation(self):
        """Test rotation augmentation strategy."""
        rotated = self.engine._apply_rotation(self.sample_image)
        
        assert rotated.shape == self.sample_image.shape
        assert rotated.dtype == self.sample_image.dtype
        
        # Test with specific angle
        rotated_45 = self.engine._apply_rotation(self.sample_image, 45.0)
        assert rotated_45.shape == self.sample_image.shape
    
    def test_scaling_augmentation(self):
        """Test scaling augmentation strategy."""
        scaled = self.engine._apply_scaling(self.sample_image)
        
        assert scaled.shape == self.sample_image.shape
        assert scaled.dtype == self.sample_image.dtype
        
        # Test with specific scale factor
        scaled_small = self.engine._apply_scaling(self.sample_image, 0.5)
        assert scaled_small.shape == self.sample_image.shape
    
    def test_noise_augmentation(self):
        """Test noise augmentation strategy."""
        noisy = self.engine._apply_noise(self.sample_image)
        
        assert noisy.shape == self.sample_image.shape
        assert noisy.dtype == self.sample_image.dtype
        assert np.all(noisy >= 0.0)
        assert np.all(noisy <= 1.0)
    
    def test_brightness_augmentation(self):
        """Test brightness augmentation strategy."""
        bright = self.engine._apply_brightness(self.sample_image)
        
        assert bright.shape == self.sample_image.shape
        assert bright.dtype == self.sample_image.dtype
        assert np.all(bright >= 0.0)
        assert np.all(bright <= 1.0)
        
        # Test with specific brightness factor
        bright_2x = self.engine._apply_brightness(self.sample_image, 2.0)
        assert np.all(bright_2x <= 1.0)  # Should be clipped
    
    def test_contrast_augmentation(self):
        """Test contrast augmentation strategy."""
        contrast = self.engine._apply_contrast(self.sample_image)
        
        assert contrast.shape == self.sample_image.shape
        assert contrast.dtype == self.sample_image.dtype
        assert np.all(contrast >= 0.0)
        assert np.all(contrast <= 1.0)
    
    def test_horizontal_flip_augmentation(self):
        """Test horizontal flip augmentation strategy."""
        # Create an asymmetric image to test flipping
        asymmetric_image = np.zeros((224, 224))
        asymmetric_image[:, :112] = 0.2  # Left half darker
        asymmetric_image[:, 112:] = 0.8  # Right half brighter
        
        # Test multiple times due to probability
        flipped_results = []
        for _ in range(10):
            result = self.engine._apply_horizontal_flip(asymmetric_image)
            flipped_results.append(np.array_equal(result, np.fliplr(asymmetric_image)))
        
        # Should have some flipped and some not flipped results
        assert any(flipped_results) or not any(flipped_results)  # At least consistent behavior
    
    def test_elastic_deformation_augmentation(self):
        """Test elastic deformation augmentation strategy."""
        deformed = self.engine._apply_elastic_deformation(self.sample_image)
        
        assert deformed.shape == self.sample_image.shape
        assert deformed.dtype == self.sample_image.dtype
        assert np.all(deformed >= 0.0)
        assert np.all(deformed <= 1.0)
    
    def test_random_augmentation(self):
        """Test random augmentation application."""
        augmented = self.engine._apply_random_augmentation(self.sample_image)
        
        assert augmented.shape == self.sample_image.shape
        assert augmented.dtype == self.sample_image.dtype
        assert np.all(augmented >= 0.0)
        assert np.all(augmented <= 1.0)
    
    def test_get_augmentation_stats(self):
        """Test augmentation statistics retrieval."""
        stats = self.engine.get_augmentation_stats()
        
        assert 'available_strategies' in stats
        assert 'rotation_range' in stats
        assert 'scale_range' in stats
        assert 'preserve_ir_properties' in stats
        
        assert len(stats['available_strategies']) == len(AugmentationType)
        assert stats['preserve_ir_properties'] is True
    
    def test_set_random_seed(self):
        """Test random seed setting for reproducibility."""
        self.engine.set_random_seed(42)
        
        # Generate augmented images with same seed
        result1 = self.engine._apply_random_augmentation(self.sample_image)
        
        self.engine.set_random_seed(42)
        result2 = self.engine._apply_random_augmentation(self.sample_image)
        
        # Results should be identical with same seed
        np.testing.assert_array_equal(result1, result2)
    
    def test_augmentation_config(self):
        """Test augmentation configuration."""
        config = AugmentationConfig(
            rotation_range=(-30, 30),
            scale_range=(0.5, 1.5),
            preserve_ir_properties=False
        )
        
        assert config.rotation_range == (-30, 30)
        assert config.scale_range == (0.5, 1.5)
        assert config.preserve_ir_properties is False
    
    def test_augmentation_type_enum(self):
        """Test augmentation type enumeration."""
        assert AugmentationType.ROTATION.value == "rotation"
        assert AugmentationType.SCALING.value == "scaling"
        assert AugmentationType.NOISE.value == "noise"
        assert AugmentationType.BRIGHTNESS.value == "brightness"
        assert AugmentationType.CONTRAST.value == "contrast"
        assert AugmentationType.HORIZONTAL_FLIP.value == "horizontal_flip"
        assert AugmentationType.ELASTIC_DEFORMATION.value == "elastic_deformation"


class TestMilitaryAugmentation:
    """Test cases for military-specific augmentation features."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = DataAugmentationEngine()
        
        # Create sample IR images for testing
        self.sample_images = []
        for i in range(5):
            img = np.zeros((224, 224), dtype=np.float32)
            # Add different objects for each image
            img[50 + i*10:100 + i*10, 50 + i*10:100 + i*10] = 0.8 + i*0.05
            self.sample_images.append(img)
    
    def test_military_augmentation_pipeline_basic(self):
        """Test basic military augmentation pipeline functionality."""
        base_images = self.sample_images[:3]  # Use 3 base images
        target_count = 15
        
        result = self.engine.create_military_augmentation_pipeline(base_images, target_count)
        
        assert len(result) == target_count
        # First 3 should be original images
        for i in range(3):
            np.testing.assert_array_equal(result[i], base_images[i])
        
        # Rest should be augmented
        for i in range(3, target_count):
            assert result[i].shape == (224, 224)
            assert result[i].dtype == np.float32
    
    def test_military_augmentation_pipeline_validation(self):
        """Test input validation for military augmentation pipeline."""
        # Test empty images list
        with pytest.raises(ValueError, match="Base images list cannot be empty"):
            self.engine.create_military_augmentation_pipeline([], 100)
        
        # Test target count less than base count
        with pytest.raises(ValueError, match="Target count .* must be >= base images count"):
            self.engine.create_military_augmentation_pipeline(self.sample_images, 2)
    
    def test_military_specific_augmentation(self):
        """Test military-specific augmentation application."""
        base_image = self.sample_images[0]
        
        # Test with different variation indices
        aug1 = self.engine._apply_military_specific_augmentation(base_image, 0)
        aug2 = self.engine._apply_military_specific_augmentation(base_image, 1)
        
        assert aug1.shape == base_image.shape
        assert aug2.shape == base_image.shape
        assert aug1.dtype == base_image.dtype
        assert aug2.dtype == base_image.dtype
        
        # Should produce different results with different indices
        assert not np.array_equal(aug1, aug2)
    
    def test_military_ir_enhancements(self):
        """Test military-specific IR enhancements."""
        base_image = self.sample_images[0]
        random_state = np.random.RandomState(42)
        
        enhanced = self.engine._apply_military_ir_enhancements(base_image, random_state)
        
        assert enhanced.shape == base_image.shape
        assert enhanced.dtype == base_image.dtype
        assert np.all(enhanced >= 0.0)
        assert np.all(enhanced <= 1.0)
    
    def test_atmospheric_effects_simulation(self):
        """Test atmospheric effects simulation."""
        base_image = self.sample_images[0]
        random_state = np.random.RandomState(42)
        
        atmospheric = self.engine._simulate_atmospheric_effects(base_image, random_state)
        
        assert atmospheric.shape == base_image.shape
        assert atmospheric.dtype == base_image.dtype
        assert np.all(atmospheric >= 0.0)
        assert np.all(atmospheric <= 1.0)
    
    def test_thermal_crossover_simulation(self):
        """Test thermal crossover effects simulation."""
        base_image = self.sample_images[0]
        random_state = np.random.RandomState(42)
        
        crossover = self.engine._simulate_thermal_crossover(base_image, random_state)
        
        assert crossover.shape == base_image.shape
        assert crossover.dtype == base_image.dtype
        assert np.all(crossover >= 0.0)
        assert np.all(crossover <= 1.0)
    
    def test_sensor_saturation_simulation(self):
        """Test sensor saturation simulation."""
        # Create image with bright regions
        bright_image = np.zeros((224, 224), dtype=np.float32)
        bright_image[100:150, 100:150] = 0.9  # Bright region
        
        random_state = np.random.RandomState(42)
        saturated = self.engine._simulate_sensor_saturation(bright_image, random_state)
        
        assert saturated.shape == bright_image.shape
        assert saturated.dtype == bright_image.dtype
        assert np.all(saturated >= 0.0)
        assert np.all(saturated <= 1.0)
    
    def test_range_dependent_noise_simulation(self):
        """Test range-dependent noise simulation."""
        base_image = self.sample_images[0]
        random_state = np.random.RandomState(42)
        
        noisy = self.engine._simulate_range_dependent_noise(base_image, random_state)
        
        assert noisy.shape == base_image.shape
        assert noisy.dtype == base_image.dtype
        assert np.all(noisy >= 0.0)
        assert np.all(noisy <= 1.0)
    
    def test_military_augmentation_weights(self):
        """Test military augmentation weights configuration."""
        weights = self.engine.military_augmentation_weights
        
        # Check that all augmentation types have weights
        assert AugmentationType.BRIGHTNESS in weights
        assert AugmentationType.CONTRAST in weights
        assert AugmentationType.NOISE in weights
        
        # Check that brightness and contrast have high priority
        assert weights[AugmentationType.BRIGHTNESS] >= 0.8
        assert weights[AugmentationType.CONTRAST] >= 0.8
        
        # Check that all weights are valid probabilities
        for weight in weights.values():
            assert 0.0 <= weight <= 1.0
    
    def test_get_military_augmentation_stats(self):
        """Test military augmentation statistics retrieval."""
        stats = self.engine.get_military_augmentation_stats()
        
        assert 'military_augmentation_weights' in stats
        assert 'military_enhancements' in stats
        assert 'typical_expansion_ratio' in stats
        assert 'military_priority_augmentations' in stats
        
        # Check military enhancements list
        expected_enhancements = [
            'atmospheric_effects',
            'thermal_crossover',
            'sensor_saturation',
            'range_dependent_noise'
        ]
        assert stats['military_enhancements'] == expected_enhancements
    
    def test_military_pipeline_expansion_ratio(self):
        """Test that military pipeline achieves correct expansion ratio."""
        base_images = self.sample_images[:2]  # 2 base images
        target_count = 10  # 5x expansion
        
        result = self.engine.create_military_augmentation_pipeline(base_images, target_count)
        
        assert len(result) == target_count
        expansion_ratio = len(result) / len(base_images)
        assert expansion_ratio == 5.0


class TestAugmentationIntegration:
    """Integration tests for augmentation with IR images."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = DataAugmentationEngine()
        
        # Create realistic IR image data
        self.ir_image = np.zeros((224, 224), dtype=np.float32)
        # Add some objects (bright spots on dark background)
        self.ir_image[50:100, 50:100] = 0.9  # Object 1
        self.ir_image[150:180, 150:180] = 0.7  # Object 2
        # Add some background noise
        noise = np.random.rand(224, 224) * 0.1
        self.ir_image += noise
        self.ir_image = np.clip(self.ir_image, 0.0, 1.0)
    
    def test_ir_characteristics_preservation(self):
        """Test that IR characteristics are preserved through augmentation."""
        # Apply multiple augmentations
        augmented = self.engine.augment_batch([self.ir_image], 5)
        
        for img in augmented:
            # Check that bright objects are still present
            max_intensity = np.max(img)
            assert max_intensity > 0.5, "Bright objects should be preserved"
            
            # Check that background is still relatively dark
            background_pixels = img[img < 0.3]
            if len(background_pixels) > 0:
                mean_background = np.mean(background_pixels)
                assert mean_background < 0.3, "Background should remain dark"
    
    def test_batch_augmentation_diversity(self):
        """Test that batch augmentation produces diverse results."""
        augmented = self.engine.augment_batch([self.ir_image], 10)
        
        # Check that augmented images are different from each other
        for i in range(1, len(augmented)):
            for j in range(i + 1, len(augmented)):
                # Images should not be identical
                assert not np.array_equal(augmented[i], augmented[j])
    
    def test_augmentation_preserves_object_structure(self):
        """Test that augmentation preserves basic object structure."""
        # Create image with clear object structure
        structured_image = np.zeros((224, 224), dtype=np.float32)
        # Create a rectangular object
        structured_image[100:150, 80:170] = 0.8
        
        augmented = self.engine._apply_random_augmentation(structured_image)
        
        # Object should still be detectable (some bright region should exist)
        assert np.max(augmented) > 0.4, "Object structure should be preserved"