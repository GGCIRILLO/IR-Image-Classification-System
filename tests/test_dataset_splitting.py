#!/usr/bin/env python3
"""
Unit tests for dataset splitting functionality.
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

# Add scripts directory to path
scripts_dir = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(scripts_dir))

from split_dataset import create_test_set_structure, get_image_files, save_metadata
from validate_split import validate_split


class TestDatasetSplitting:
    """Test cases for dataset splitting functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.processed_dir = self.temp_dir / "processed"
        self.test_dir = self.temp_dir / "test_set"
        self.metadata_file = self.temp_dir / "metadata.json"
        
        # Create test directory structure with sample images
        self.processed_dir.mkdir(parents=True)
        
        # Create sample classes with images
        for class_name in ["ClassA", "ClassB", "ClassC"]:
            class_dir = self.processed_dir / class_name
            class_dir.mkdir()
            
            # Create 10 sample images per class
            for i in range(1, 11):
                (class_dir / f"{class_name.lower()}_image_{i}.png").touch()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_get_image_files(self):
        """Test getting image files from directory."""
        class_dir = self.processed_dir / "ClassA"
        images = get_image_files(class_dir)
        
        assert len(images) == 10
        assert all(img.suffix == '.png' for img in images)
        assert all('classa_image_' in img.name for img in images)
    
    def test_create_test_set_structure(self):
        """Test creating test set structure."""
        # Set random seed for reproducible results
        import random
        random.seed(42)
        
        metadata = create_test_set_structure(
            self.processed_dir, 
            self.test_dir, 
            images_per_class=3
        )
        
        # Check metadata structure
        assert metadata['images_per_class'] == 3
        assert metadata['total_test_images'] == 9  # 3 classes * 3 images
        assert metadata['total_training_images'] == 21  # 3 classes * 7 remaining
        assert len(metadata['classes']) == 3
        
        # Check test directory was created
        assert self.test_dir.exists()
        
        # Check each class has correct number of test images
        for class_name in ["ClassA", "ClassB", "ClassC"]:
            test_class_dir = self.test_dir / class_name
            assert test_class_dir.exists()
            
            test_images = list(test_class_dir.glob('*.png'))
            assert len(test_images) == 3
            
            # Check metadata for this class
            class_data = metadata['classes'][class_name]
            assert class_data['test_images'] == 3
            assert class_data['training_images'] == 7
            assert len(class_data['test_files']) == 3
    
    def test_save_metadata(self):
        """Test saving metadata to file."""
        sample_metadata = {
            'split_date': '2025-01-01T00:00:00',
            'total_test_images': 5,
            'total_training_images': 15
        }
        
        save_metadata(sample_metadata, self.metadata_file)
        
        # Verify file was created and contains correct data
        assert self.metadata_file.exists()
        
        with open(self.metadata_file, 'r') as f:
            loaded_metadata = json.load(f)
        
        assert loaded_metadata['total_test_images'] == 5
        assert loaded_metadata['total_training_images'] == 15
        assert loaded_metadata['split_date'] == '2025-01-01T00:00:00'
    
    def test_insufficient_images_handling(self):
        """Test handling of classes with insufficient images."""
        # Create a class with only 2 images
        small_class_dir = self.processed_dir / "SmallClass"
        small_class_dir.mkdir()
        (small_class_dir / "image1.png").touch()
        (small_class_dir / "image2.png").touch()
        
        import random
        random.seed(42)
        
        metadata = create_test_set_structure(
            self.processed_dir,
            self.test_dir,
            images_per_class=5  # More than available
        )
        
        # Check that SmallClass only moved 2 images (all available)
        small_class_data = metadata['classes']['SmallClass']
        assert small_class_data['test_images'] == 2
        assert small_class_data['training_images'] == 0
        assert len(small_class_data['test_files']) == 2


class TestSplitValidation:
    """Test cases for split validation functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.processed_dir = self.temp_dir / "processed"
        self.test_dir = self.temp_dir / "test_set"
        self.metadata_file = self.temp_dir / "metadata.json"
        
        # Create directory structure
        self.processed_dir.mkdir(parents=True)
        self.test_dir.mkdir(parents=True)
        
        # Create sample class directories
        (self.processed_dir / "ClassA").mkdir()
        (self.test_dir / "ClassA").mkdir()
        
        # Create sample images
        for i in range(1, 6):  # 5 training images
            (self.processed_dir / "ClassA" / f"training_{i}.png").touch()
        
        for i in range(1, 3):  # 2 test images
            (self.test_dir / "ClassA" / f"test_{i}.png").touch()
        
        # Create metadata
        metadata = {
            "classes": {
                "ClassA": {
                    "test_images": 2,
                    "training_images": 5,
                    "test_files": ["test_1.png", "test_2.png"]
                }
            },
            "total_test_images": 2,
            "total_training_images": 5
        }
        
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_validate_split_success(self):
        """Test successful split validation."""
        success = validate_split(
            self.processed_dir,
            self.test_dir,
            self.metadata_file
        )
        
        assert success is True
    
    def test_validate_split_missing_metadata(self):
        """Test validation with missing metadata file."""
        missing_metadata = self.temp_dir / "missing.json"
        
        success = validate_split(
            self.processed_dir,
            self.test_dir,
            missing_metadata
        )
        
        assert success is False


if __name__ == "__main__":
    pytest.main([__file__])