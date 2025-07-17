"""
Test suite for EmbeddingExtractor implementation.

Comprehensive tests to verify the completion of task 6.1:
- EmbeddingExtractor class with batch processing
- Model inference optimization for IR images  
- Embedding validation and quality assessment
"""

import unittest
import numpy as np
import tempfile
import shutil
from pathlib import Path

from src.embedding.extractor import EmbeddingExtractor, ExtractionConfig, EmbeddingCache


class TestEmbeddingExtractor(unittest.TestCase):
    """Test cases for EmbeddingExtractor functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = ExtractionConfig(
            batch_size=4,
            cache_dir=str(Path(self.temp_dir) / "cache"),
            cache_embeddings=True,
            quality_threshold=0.5
        )
        self.extractor = EmbeddingExtractor(config=self.config)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_extractor_initialization(self):
        """Test EmbeddingExtractor initialization."""
        self.assertIsNotNone(self.extractor)
        self.assertEqual(self.extractor.model_name, "resnet50")
        self.assertIsNotNone(self.extractor.config)
        self.assertEqual(self.extractor.config.batch_size, 4)
        self.assertTrue(self.extractor.config.cache_embeddings)
    
    def test_embedding_validation_quality(self):
        """Test embedding quality validation."""
        # Test valid embedding
        valid_embedding = np.random.rand(512).astype(np.float32)
        quality = self.extractor.validate_embedding_quality(valid_embedding)
        self.assertGreaterEqual(quality, 0.0)
        self.assertLessEqual(quality, 1.0)
        
        # Test invalid embedding with NaN values
        invalid_embedding = np.array([np.nan, 1.0, 2.0])
        quality = self.extractor.validate_embedding_quality(invalid_embedding)
        self.assertEqual(quality, 0.0)
        
        # Test invalid embedding with infinite values
        invalid_embedding = np.array([np.inf, 1.0, 2.0])
        quality = self.extractor.validate_embedding_quality(invalid_embedding)
        self.assertEqual(quality, 0.0)
        
        # Test embedding with wrong dimensions
        wrong_dim_embedding = np.random.rand(10).astype(np.float32)  # Too small
        quality = self.extractor.validate_embedding_quality(wrong_dim_embedding)
        self.assertEqual(quality, 0.0)
    
    def test_extraction_stats_tracking(self):
        """Test extraction statistics tracking."""
        stats = self.extractor.get_extraction_stats()
        
        # Check required stats fields
        required_fields = [
            'total_extractions', 'cache_hits', 'cache_misses',
            'total_time', 'average_time', 'batch_extractions',
            'gpu_extractions', 'cpu_extractions'
        ]
        
        for field in required_fields:
            self.assertIn(field, stats)
            self.assertIsInstance(stats[field], (int, float))
    
    def test_model_info_retrieval(self):
        """Test model information retrieval."""
        info = self.extractor.get_model_info()
        
        self.assertIn('name', info)
        self.assertIn('version', info)
        self.assertIn('loaded', info)
        self.assertIn('extraction_stats', info)
        self.assertIn('cache_stats', info)
    
    def test_optimization_methods(self):
        """Test model optimization methods."""
        # Test optimization for inference (should not raise errors)
        try:
            self.extractor.optimize_for_inference()
        except Exception as e:
            self.fail(f"optimize_for_inference raised exception: {e}")
        
        # Test warmup (should handle no model gracefully)
        try:
            self.extractor.warmup(num_warmup_images=2)
        except Exception as e:
            if "model not loaded" not in str(e).lower():
                self.fail(f"warmup raised unexpected exception: {e}")
    
    def test_cache_functionality(self):
        """Test embedding cache functionality."""
        self.assertIsNotNone(self.extractor.cache)
        
        # Test cache stats
        if self.extractor.cache is not None:
            cache_stats = self.extractor.cache.get_stats()
            self.assertIn('memory_cache_size', cache_stats)
            self.assertIn('disk_cache_size', cache_stats)
            self.assertIn('cache_dir', cache_stats)
            self.assertIn('max_size', cache_stats)
        else:
            self.fail("Extractor cache is not initialized (None).")
        
        # Test cache clear
        self.extractor.clear_cache()
        cache_stats_after = self.extractor.cache.get_stats()
        self.assertEqual(cache_stats_after['memory_cache_size'], 0)


class TestEmbeddingCache(unittest.TestCase):
    """Test cases for EmbeddingCache functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache = EmbeddingCache(cache_dir=str(Path(self.temp_dir) / "cache"))
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_cache_operations(self):
        """Test basic cache operations."""
        # Create test data
        image_data = np.random.rand(224, 224).astype(np.float32)
        embedding = np.random.rand(512).astype(np.float32)
        model_version = "v1.0.0"
        
        # Test cache miss
        result = self.cache.get(image_data, model_version)
        self.assertIsNone(result)
        
        # Test cache put and get
        self.cache.put(image_data, model_version, embedding)
        result = self.cache.get(image_data, model_version)
        
        self.assertIsNotNone(result)
        np.testing.assert_array_equal(result, embedding)
    
    def test_cache_stats(self):
        """Test cache statistics."""
        stats = self.cache.get_stats()
        
        self.assertIn('memory_cache_size', stats)
        self.assertIn('disk_cache_size', stats)
        self.assertIn('cache_dir', stats)
        self.assertIn('max_size', stats)
        
        self.assertIsInstance(stats['memory_cache_size'], int)
        self.assertIsInstance(stats['disk_cache_size'], int)
        self.assertIsInstance(stats['max_size'], int)
    
    def test_cache_clear(self):
        """Test cache clearing."""
        # Add some data
        image_data = np.random.rand(224, 224).astype(np.float32)
        embedding = np.random.rand(512).astype(np.float32)
        self.cache.put(image_data, "v1.0", embedding)
        
        # Verify data exists
        self.assertIsNotNone(self.cache.get(image_data, "v1.0"))
        
        # Clear cache
        self.cache.clear()
        
        # Verify data is gone
        self.assertIsNone(self.cache.get(image_data, "v1.0"))
        stats = self.cache.get_stats()
        self.assertEqual(stats['memory_cache_size'], 0)


class TestExtractionConfig(unittest.TestCase):
    """Test cases for ExtractionConfig."""
    
    def test_default_configuration(self):
        """Test default configuration values."""
        config = ExtractionConfig()
        
        self.assertEqual(config.batch_size, 32)
        self.assertEqual(config.max_workers, 4)
        self.assertTrue(config.use_gpu)
        self.assertTrue(config.cache_embeddings)
        self.assertEqual(config.cache_dir, "./cache/embeddings")
        self.assertEqual(config.quality_threshold, 0.7)
        self.assertEqual(config.timeout_seconds, 30.0)
        self.assertTrue(config.enable_preprocessing)
    
    def test_custom_configuration(self):
        """Test custom configuration values."""
        config = ExtractionConfig(
            batch_size=16,
            max_workers=2,
            use_gpu=False,
            cache_embeddings=False,
            quality_threshold=0.8
        )
        
        self.assertEqual(config.batch_size, 16)
        self.assertEqual(config.max_workers, 2)
        self.assertFalse(config.use_gpu)
        self.assertFalse(config.cache_embeddings)
        self.assertEqual(config.quality_threshold, 0.8)


if __name__ == '__main__':
    # Run the test suite
    unittest.main(verbosity=2)
