#!/usr/bin/env python3
"""
Unit tests for test set management utilities.
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

from test_set_manager import TestSetManager
from training_validator import TrainingValidator


class TestTestSetManager:
    """Test cases for TestSetManager class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.processed_dir = self.temp_dir / "processed"
        self.test_dir = self.temp_dir / "test_set"
        self.metadata_file = self.temp_dir / "metadata.json"
        
        # Create test directory structure
        self.processed_dir.mkdir(parents=True)
        self.test_dir.mkdir(parents=True)
        
        # Create sample metadata
        self.sample_metadata = {
            "split_date": "2025-01-01T00:00:00",
            "images_per_class": 2,
            "classes": {
                "ClassA": {
                    "total_original": 10,
                    "test_images": 2,
                    "training_images": 8,
                    "test_files": ["imageA1.png", "imageA2.png"],
                    "test_ratio": 0.2
                },
                "ClassB": {
                    "total_original": 8,
                    "test_images": 2,
                    "training_images": 6,
                    "test_files": ["imageB1.png", "imageB2.png"],
                    "test_ratio": 0.25
                }
            },
            "total_test_images": 4,
            "total_training_images": 14,
            "split_ratio": {
                "test_percentage": 22.2,
                "training_percentage": 77.8
            }
        }
        
        # Create sample files
        (self.test_dir / "ClassA").mkdir()
        (self.test_dir / "ClassB").mkdir()
        (self.test_dir / "ClassA" / "imageA1.png").touch()
        (self.test_dir / "ClassA" / "imageA2.png").touch()
        (self.test_dir / "ClassB" / "imageB1.png").touch()
        (self.test_dir / "ClassB" / "imageB2.png").touch()
        
        (self.processed_dir / "ClassA").mkdir()
        (self.processed_dir / "ClassB").mkdir()
        for i in range(3, 11):  # Training images
            (self.processed_dir / "ClassA" / f"imageA{i}.png").touch()
        for i in range(3, 9):   # Training images
            (self.processed_dir / "ClassB" / f"imageB{i}.png").touch()
        
        # Save metadata
        with open(self.metadata_file, 'w') as f:
            json.dump(self.sample_metadata, f)
        
        self.manager = TestSetManager(
            str(self.processed_dir),
            str(self.test_dir),
            str(self.metadata_file)
        )
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_load_metadata(self):
        """Test metadata loading."""
        metadata = self.manager.load_metadata()
        assert metadata is not None
        assert metadata["total_test_images"] == 4
        assert metadata["total_training_images"] == 14
        assert len(metadata["classes"]) == 2
    
    def test_get_test_image_paths(self):
        """Test getting test image paths."""
        test_paths = self.manager.get_test_image_paths()
        assert len(test_paths) == 4
        assert any("imageA1.png" in path for path in test_paths)
        assert any("imageB2.png" in path for path in test_paths)
    
    def test_get_training_image_paths(self):
        """Test getting training image paths."""
        training_paths = self.manager.get_training_image_paths()
        assert len(training_paths) == 14  # 8 + 6 training images
    
    def test_validate_no_overlap(self):
        """Test validation of no overlap between test and training."""
        is_valid, overlapping = self.manager.validate_no_overlap()
        assert is_valid is True
        assert len(overlapping) == 0
    
    def test_get_status(self):
        """Test status reporting."""
        status = self.manager.get_status()
        assert status["test_set_exists"] is True
        assert status["metadata_exists"] is True
        assert status["test_images"] == 4
        assert status["training_images"] == 14
        assert status["classes"] == 2
        assert status["validation_passed"] is True
    
    def test_track_image_sources(self):
        """Test image source tracking."""
        tracking = self.manager.track_image_sources()
        assert len(tracking) == 4
        assert "imageA1.png" in tracking
        assert tracking["imageA1.png"]["class"] == "ClassA"
        assert tracking["imageA1.png"]["in_test_set"] is True


class TestTrainingValidator:
    """Test cases for TrainingValidator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.metadata_file = self.temp_dir / "metadata.json"
        
        # Create sample metadata
        self.sample_metadata = {
            "classes": {
                "ClassA": {
                    "test_files": ["test1.png", "test2.png"]
                },
                "ClassB": {
                    "test_files": ["test3.png", "test4.png"]
                }
            }
        }
        
        with open(self.metadata_file, 'w') as f:
            json.dump(self.sample_metadata, f)
        
        self.validator = TrainingValidator(str(self.metadata_file))
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_load_test_filenames(self):
        """Test loading test filenames."""
        test_filenames = self.validator._load_test_filenames()
        assert len(test_filenames) == 4
        assert "test1.png" in test_filenames
        assert "test4.png" in test_filenames
    
    def test_validate_file_list_clean(self):
        """Test validation of clean file list."""
        clean_files = ["training1.png", "training2.png"]
        is_valid, violations = self.validator.validate_file_list(clean_files)
        assert is_valid is True
        assert len(violations) == 0
    
    def test_validate_file_list_contaminated(self):
        """Test validation of contaminated file list."""
        contaminated_files = ["training1.png", "test1.png", "training2.png"]
        is_valid, violations = self.validator.validate_file_list(contaminated_files)
        assert is_valid is False
        assert len(violations) == 1
        assert "test1.png" in violations[0]
    
    def test_validate_directory_nonexistent(self):
        """Test validation of non-existent directory."""
        is_valid, violations = self.validator.validate_directory("/nonexistent/path")
        assert is_valid is True
        assert len(violations) == 0


class TestIntegration:
    """Integration tests for test set management."""
    
    def test_manager_validator_integration(self):
        """Test integration between manager and validator."""
        # This would test the full workflow in a real scenario
        # For now, just verify both classes can be instantiated
        manager = TestSetManager()
        validator = TrainingValidator()
        
        assert manager is not None
        assert validator is not None


if __name__ == "__main__":
    pytest.main([__file__])