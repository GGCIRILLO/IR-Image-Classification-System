"""
Tests for IR image processing and loading functionality.
"""

import pytest
import numpy as np
from PIL import Image
import tempfile
import os
from pathlib import Path

from src.data import IRImageProcessor, IRImageLoader, ImageFormatError, ImageCorruptionError
from src.models.data_models import IRImage


class TestIRImageProcessor:
    """Test cases for IRImageProcessor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = IRImageProcessor()
        
        # Create a sample IR-like image (white objects on black background)
        self.sample_ir_image = self._create_sample_ir_image()
    
    def _create_sample_ir_image(self, size=(224, 224)):
        """Create a sample IR image for testing."""
        # Create black background
        image = np.zeros(size, dtype=np.float32)
        
        # Add some white objects (rectangles and circles)
        # Rectangle in upper left
        image[50:100, 50:120] = 0.9
        
        # Circle in lower right
        center_y, center_x = 150, 150
        radius = 30
        y, x = np.ogrid[:size[0], :size[1]]
        mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
        image[mask] = 0.8
        
        # Add some noise to make it more realistic
        noise = np.random.normal(0, 0.05, size)
        image = np.clip(image + noise, 0, 1)
        
        return image
    
    def test_preprocess_ir_image(self):
        """Test IR image preprocessing pipeline."""
        processed = self.processor.preprocess_ir_image(self.sample_ir_image)
        
        # Check output properties
        assert processed.shape == (224, 224)
        assert processed.dtype == np.float32
        assert 0.0 <= processed.min() <= processed.max() <= 1.0
    
    def test_validate_ir_format_valid(self):
        """Test IR format validation with valid image."""
        is_valid = self.processor.validate_ir_format(self.sample_ir_image)
        assert is_valid is True
    
    def test_validate_ir_format_invalid_low_contrast(self):
        """Test IR format validation with low contrast image."""
        # Create uniform gray image (low contrast)
        low_contrast_image = np.full((224, 224), 0.5, dtype=np.float32)
        
        is_valid = self.processor.validate_ir_format(low_contrast_image)
        assert is_valid is False
    
    def test_enhance_contrast(self):
        """Test contrast enhancement."""
        enhanced = self.processor.enhance_contrast(self.sample_ir_image)
        
        # Enhanced image should have better contrast
        original_contrast = self.sample_ir_image.max() - self.sample_ir_image.min()
        enhanced_contrast = enhanced.max() - enhanced.min()
        
        assert enhanced_contrast >= original_contrast * 0.9  # Allow some tolerance
        assert enhanced.shape == self.sample_ir_image.shape
    
    def test_reduce_noise(self):
        """Test noise reduction."""
        # Add noise to image
        noisy_image = self.sample_ir_image + np.random.normal(0, 0.1, self.sample_ir_image.shape)
        noisy_image = np.clip(noisy_image, 0, 1)
        
        denoised = self.processor.reduce_noise(noisy_image)
        
        # Denoised image should be smoother (lower standard deviation)
        assert denoised.std() <= noisy_image.std()
        assert denoised.shape == noisy_image.shape
    
    def test_normalize_image(self):
        """Test image normalization."""
        # Create image with arbitrary range
        test_image = self.sample_ir_image * 100 + 50
        
        normalized = self.processor.normalize_image(test_image)
        
        assert normalized.min() == 0.0
        assert normalized.max() == 1.0
        assert normalized.shape == test_image.shape
    
    def test_resize_to_standard(self):
        """Test image resizing."""
        # Create image with different size
        large_image = np.random.rand(512, 512).astype(np.float32)
        
        resized = self.processor.resize_to_standard(large_image, (224, 224))
        
        assert resized.shape == (224, 224)
        assert resized.dtype == np.float32


class TestIRImageLoader:
    """Test cases for IRImageLoader class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.loader = IRImageLoader()
        
        # Create temporary directory for test images
        self.temp_dir = tempfile.mkdtemp()
        
        # Create sample test images
        self.test_image_paths = self._create_test_images()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        # Clean up temporary files
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_test_images(self):
        """Create test image files in various formats."""
        paths = {}
        
        # Create a sample IR-like image
        sample_image = self._create_sample_ir_image()
        pil_image = Image.fromarray((sample_image * 255).astype(np.uint8), mode='L')
        
        # Save in different formats
        formats = ['.png', '.jpg', '.bmp', '.tiff']
        for fmt in formats:
            path = Path(self.temp_dir) / f"test_image{fmt}"
            pil_image.save(path)
            paths[fmt] = path
        
        # Create a corrupted file
        corrupted_path = Path(self.temp_dir) / "corrupted.png"
        with open(corrupted_path, 'wb') as f:
            f.write(b"not an image")
        paths['corrupted'] = corrupted_path
        
        return paths
    
    def _create_sample_ir_image(self, size=(224, 224)):
        """Create a sample IR image for testing."""
        # Create black background with white objects
        image = np.zeros(size, dtype=np.float32)
        image[50:100, 50:120] = 0.9  # Rectangle
        
        # Add circle
        center_y, center_x = 150, 150
        radius = 30
        y, x = np.ogrid[:size[0], :size[1]]
        mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
        image[mask] = 0.8
        
        return image
    
    def test_load_image_png(self):
        """Test loading PNG image."""
        ir_image = self.loader.load_image(self.test_image_paths['.png'])
        
        assert isinstance(ir_image, IRImage)
        assert ir_image.image_data.shape == (224, 224)
        assert ir_image.image_data.dtype == np.float32
        assert 0.0 <= ir_image.image_data.min() <= ir_image.image_data.max() <= 1.0
    
    def test_load_image_jpg(self):
        """Test loading JPEG image."""
        ir_image = self.loader.load_image(self.test_image_paths['.jpg'])
        
        assert isinstance(ir_image, IRImage)
        assert ir_image.image_data.shape == (224, 224)
    
    def test_validate_image_format_valid(self):
        """Test format validation with valid image."""
        is_valid = self.loader.validate_image_format(self.test_image_paths['.png'])
        assert is_valid is True
    
    def test_validate_image_format_corrupted(self):
        """Test format validation with corrupted file."""
        is_valid = self.loader.validate_image_format(self.test_image_paths['corrupted'])
        assert is_valid is False
    
    def test_load_image_unsupported_format(self):
        """Test loading unsupported format raises error."""
        # Create file with unsupported extension
        unsupported_path = Path(self.temp_dir) / "test.xyz"
        with open(unsupported_path, 'wb') as f:
            f.write(b"fake content")
        
        with pytest.raises(ImageFormatError):
            self.loader.load_image(unsupported_path)
    
    def test_load_image_corrupted(self):
        """Test loading corrupted file raises error."""
        with pytest.raises(ImageCorruptionError):
            self.loader.load_image(self.test_image_paths['corrupted'])
    
    def test_get_image_info(self):
        """Test getting image information."""
        info = self.loader.get_image_info(self.test_image_paths['.png'])
        
        assert 'file_path' in info
        assert 'file_size' in info
        assert 'format' in info
        assert 'size' in info
        assert info['size'] == (224, 224)
    
    def test_load_images_batch(self):
        """Test batch loading of images."""
        valid_paths = [self.test_image_paths['.png'], self.test_image_paths['.jpg']]
        
        images = self.loader.load_images_batch(valid_paths)
        
        assert len(images) == 2
        assert all(isinstance(img, IRImage) for img in images)


if __name__ == "__main__":
    # Run basic tests
    processor_test = TestIRImageProcessor()
    processor_test.setup_method()
    
    print("Testing IR Image Processor...")
    try:
        processor_test.test_preprocess_ir_image()
        processor_test.test_validate_ir_format_valid()
        processor_test.test_enhance_contrast()
        processor_test.test_normalize_image()
        print("✓ IR Image Processor tests passed")
    except Exception as e:
        print(f"✗ IR Image Processor tests failed: {e}")
    
    loader_test = TestIRImageLoader()
    loader_test.setup_method()
    
    print("Testing IR Image Loader...")
    try:
        loader_test.test_validate_image_format_valid()
        loader_test.test_get_image_info()
        print("✓ IR Image Loader tests passed")
    except Exception as e:
        print(f"✗ IR Image Loader tests failed: {e}")
    finally:
        loader_test.teardown_method()
    
    print("All tests completed!")