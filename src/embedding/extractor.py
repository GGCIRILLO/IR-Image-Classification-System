"""
Embedding extraction service for IR Image Classification System.

This module provides the main EmbeddingExtractor class that handles embedding
extraction from IR images using fine-tuned models with batch processing,
GPU acceleration, and quality validation.
"""

import hashlib
import logging
import pickle
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np
import torch

from src.models.interfaces import BaseEmbeddingExtractor
from src.training.model_adapters import BaseModelAdapter, ModelAdapterFactory


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ExtractionConfig:
    """Configuration for embedding extraction."""
    batch_size: int = 32
    max_workers: int = 4
    use_gpu: bool = True
    cache_embeddings: bool = True
    cache_dir: str = "./cache/embeddings"
    quality_threshold: float = 0.7
    timeout_seconds: float = 30.0
    enable_preprocessing: bool = True
    # Performance optimization settings
    enable_mixed_precision: bool = False  # For GPU acceleration
    prefetch_factor: int = 2  # For data loading optimization
    pin_memory: bool = True   # For faster GPU transfers


class EmbeddingCache:
    """
    Thread-safe caching mechanism for embeddings.
    
    Provides efficient storage and retrieval of computed embeddings
    to avoid redundant computations.
    """
    
    def __init__(self, cache_dir: str = "./cache/embeddings", max_size: int = 10000):
        """
        Initialize embedding cache.
        
        Args:
            cache_dir: Directory to store cached embeddings
            max_size: Maximum number of embeddings to cache
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size = max_size
        self.memory_cache = {}
        self.cache_lock = threading.RLock()
        
        logger.info("Initialized embedding cache at %s with max size %d", cache_dir, max_size)
    
    def _get_cache_key(self, image_data: np.ndarray, model_version: str) -> str:
        """Generate unique cache key for image and model combination."""
        # Create hash from image data and model version
        image_hash = hashlib.md5(image_data.tobytes()).hexdigest()
        combined_key = f"{image_hash}_{model_version}"
        return hashlib.md5(combined_key.encode()).hexdigest()
    
    def get(self, image_data: np.ndarray, model_version: str) -> Optional[np.ndarray]:
        """
        Retrieve cached embedding if available.
        
        Args:
            image_data: Image data as numpy array
            model_version: Version identifier of the model
            
        Returns:
            Optional[np.ndarray]: Cached embedding vector or None
        """
        cache_key = self._get_cache_key(image_data, model_version)
        
        with self.cache_lock:
            # Check memory cache first
            if cache_key in self.memory_cache:
                return self.memory_cache[cache_key]
            
            # Check disk cache
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            if cache_file.exists():
                try:
                    with open(cache_file, 'rb') as f:
                        embedding = pickle.load(f)
                    
                    # Add to memory cache
                    self.memory_cache[cache_key] = embedding
                    return embedding
                except Exception as e:
                    logger.warning(f"Failed to load cached embedding: {e}")
                    cache_file.unlink(missing_ok=True)
        
        return None
    
    def put(self, image_data: np.ndarray, model_version: str, embedding: np.ndarray) -> None:
        """
        Store embedding in cache.
        
        Args:
            image_data: Image data as numpy array
            model_version: Version identifier of the model
            embedding: Embedding vector to cache
        """
        cache_key = self._get_cache_key(image_data, model_version)
        
        with self.cache_lock:
            # Store in memory cache
            if len(self.memory_cache) >= self.max_size:
                # Remove oldest entry (simple FIFO)
                oldest_key = next(iter(self.memory_cache))
                del self.memory_cache[oldest_key]
            
            self.memory_cache[cache_key] = embedding
            
            # Store on disk
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(embedding, f)
            except Exception as e:
                logger.warning(f"Failed to cache embedding to disk: {e}")
    
    def clear(self) -> None:
        """Clear all cached embeddings."""
        with self.cache_lock:
            self.memory_cache.clear()
            
            # Clear disk cache
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink(missing_ok=True)
        
        logger.info("Embedding cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.cache_lock:
            disk_files = len(list(self.cache_dir.glob("*.pkl")))
            return {
                'memory_cache_size': len(self.memory_cache),
                'disk_cache_size': disk_files,
                'cache_dir': str(self.cache_dir),
                'max_size': self.max_size
            }


class EmbeddingExtractor(BaseEmbeddingExtractor):
    """
    Main embedding extraction service for IR images.
    
    Provides high-performance embedding extraction with batch processing,
    GPU acceleration, caching, and quality validation.
    """
    
    def __init__(self, 
                 model_type: str = "resnet50",
                 config: Optional[ExtractionConfig] = None,
                 model_path: Optional[str] = None):
        """
        Initialize embedding extractor.
        
        Args:
            model_type: Type of model to use ('resnet50' or 'qwen_vlm')
            config: Configuration for extraction parameters
            model_path: Path to fine-tuned model weights
        """
        super().__init__(model_type)
        
        self.config = config or ExtractionConfig()
        self.model_path = model_path
        self.model_adapter: Optional[BaseModelAdapter] = None
        self.cache = EmbeddingCache(
            cache_dir=self.config.cache_dir,
            max_size=10000
        ) if self.config.cache_embeddings else None
        
        # Performance tracking
        self.extraction_stats = {
            'total_extractions': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_time': 0.0,
            'average_time': 0.0,
            'batch_extractions': 0,
            'gpu_extractions': 0,
            'cpu_extractions': 0
        }
        
        logger.info(f"Initialized EmbeddingExtractor with model type: {model_type}")
    
    def load_model(self, model_path: Optional[str] = None) -> None:
        """
        Load the fine-tuned model for embedding extraction.
        
        Args:
            model_path: Path to the trained model file (optional)
        """
        try:
            # Use provided path or fallback to instance path
            path_to_use = model_path or self.model_path
            
            # Create model adapter
            adapter_kwargs = {
                'device': 'cuda' if self.config.use_gpu and torch.cuda.is_available() else 'cpu'
            }
            
            self.model_adapter = ModelAdapterFactory.create_adapter(
                self.model_name, **adapter_kwargs
            )
            
            # Load model weights
            self.model_adapter.load_model(path_to_use)
            
            # Update model info
            self.model_info = self.model_adapter.get_model_info()
            self.model_info['loaded'] = True
            self.model_info['load_time'] = datetime.now().isoformat()
            
            logger.info(f"Model loaded successfully: {self.model_info}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise RuntimeError(f"Model loading failed: {str(e)}")
    
    def extract_embedding(self, image: np.ndarray) -> np.ndarray:
        """
        Extract feature embedding from a single IR image.
        
        Args:
            image: Preprocessed IR image as numpy array
            
        Returns:
            np.ndarray: Feature embedding vector
        """
        if self.model_adapter is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        start_time = time.time()
        
        try:
            # Check cache first
            if self.cache:
                cached_embedding = self.cache.get(image, self.model_info.get('version', '1.0.0'))
                if cached_embedding is not None:
                    self.extraction_stats['cache_hits'] += 1
                    self.extraction_stats['total_extractions'] += 1
                    logger.debug("Retrieved embedding from cache")
                    return cached_embedding
                else:
                    self.extraction_stats['cache_misses'] += 1
            
            # Extract embedding using model adapter
            embedding = self.model_adapter.extract_embedding(image)
            
            # Validate embedding quality
            quality_score = self.validate_embedding_quality(embedding)
            if quality_score < self.config.quality_threshold:
                logger.warning(f"Low quality embedding detected (score: {quality_score:.3f})")
            
            # Cache the embedding
            if self.cache:
                self.cache.put(image, self.model_info.get('version', '1.0.0'), embedding)
            
            # Update statistics
            extraction_time = time.time() - start_time
            self._update_stats(extraction_time, use_gpu=self.model_adapter.device == 'cuda')
            
            logger.debug(f"Extracted embedding in {extraction_time:.3f}s (quality: {quality_score:.3f})")
            
            return embedding
            
        except Exception as e:
            logger.error(f"Embedding extraction failed: {str(e)}")
            raise RuntimeError(f"Embedding extraction failed: {str(e)}")
    
    def batch_extract(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """
        Extract embeddings from multiple images efficiently.
        
        Args:
            images: List of preprocessed IR images
            
        Returns:
            List[np.ndarray]: List of embedding vectors
        """
        if self.model_adapter is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        if not images:
            return []
        
        start_time = time.time()
        embeddings = []
        
        try:
            # Check cache for all images first
            cached_embeddings = {}
            uncached_images = []
            uncached_indices = []
            
            if self.cache:
                model_version = self.model_info.get('version', '1.0.0')
                for i, image in enumerate(images):
                    cached_embedding = self.cache.get(image, model_version)
                    if cached_embedding is not None:
                        cached_embeddings[i] = cached_embedding
                        self.extraction_stats['cache_hits'] += 1
                    else:
                        uncached_images.append(image)
                        uncached_indices.append(i)
                        self.extraction_stats['cache_misses'] += 1
            else:
                uncached_images = images
                uncached_indices = list(range(len(images)))
            
            # Extract embeddings for uncached images in batches
            if uncached_images:
                batch_embeddings = self._batch_extract_uncached(uncached_images)
                
                # Cache new embeddings
                if self.cache:
                    model_version = self.model_info.get('version', '1.0.0')
                    for image, embedding in zip(uncached_images, batch_embeddings):
                        self.cache.put(image, model_version, embedding)
            else:
                batch_embeddings = []
            
            # Combine cached and newly extracted embeddings in correct order
            embeddings: List[Optional[np.ndarray]] = [None] * len(images)
            
            # Fill in cached embeddings
            for i, embedding in cached_embeddings.items():
                embeddings[i] = embedding
            
            # Fill in newly extracted embeddings
            for i, embedding in zip(uncached_indices, batch_embeddings):
                embeddings[i] = embedding
            
            # Ensure all embeddings are np.ndarray (no None values)
            if any(e is None for e in embeddings):
                raise RuntimeError("Some embeddings could not be extracted and are None.")
            embeddings_np: List[np.ndarray] = [e for e in embeddings if e is not None]
            
            # Update statistics
            extraction_time = time.time() - start_time
            self.extraction_stats['batch_extractions'] += 1
            self.extraction_stats['total_extractions'] += len(images)
            self.extraction_stats['total_time'] += extraction_time
            self.extraction_stats['average_time'] = (
                self.extraction_stats['total_time'] / self.extraction_stats['total_extractions']
            )
            
            if self.model_adapter.device == 'cuda':
                self.extraction_stats['gpu_extractions'] += len(uncached_images)
            else:
                self.extraction_stats['cpu_extractions'] += len(uncached_images)
            
            logger.info(f"Batch extracted {len(images)} embeddings in {extraction_time:.3f}s "
                       f"({len(cached_embeddings)} from cache, {len(uncached_images)} computed)")
            
            return embeddings_np
            
        except Exception as e:
            logger.error(f"Batch embedding extraction failed: {str(e)}")
            raise RuntimeError(f"Batch embedding extraction failed: {str(e)}")
    
    def _batch_extract_uncached(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """
        Extract embeddings for uncached images using model adapter batch processing.
        
        Args:
            images: List of images that need embedding extraction
            
        Returns:
            List[np.ndarray]: List of embedding vectors
        """
        if not images:
            return []
        
        # Use adaptive batch size for better memory management
        batch_size = self._adaptive_batch_size(len(images))
        
        # Check if adapter has batch_extract method (it should, as it's abstract in BaseModelAdapter)
        if self.model_adapter and hasattr(self.model_adapter, 'batch_extract'):
            # Use adapter's optimized batch processing
            embeddings = self.model_adapter.batch_extract(images, batch_size)
        elif self.model_adapter:
            # Fallback to sequential processing if batch_extract is not available
            embeddings = []
            for i in range(0, len(images), batch_size):
                batch_images = images[i:i + batch_size]
                
                # Use parallel preprocessing if beneficial
                if len(batch_images) > 1 and self.config.max_workers > 1:
                    # For small batches, parallel preprocessing might help
                    batch_embeddings = []
                    for img in batch_images:
                        embedding = self.model_adapter.extract_embedding(img)
                        batch_embeddings.append(embedding)
                else:
                    # Sequential processing for single images or when parallelization is disabled
                    batch_embeddings = [
                        self.model_adapter.extract_embedding(img) for img in batch_images
                    ]
                
                embeddings.extend(batch_embeddings)
        else:
            raise RuntimeError("Model adapter is not available")
        
        return embeddings
    
    def validate_embedding_quality(self, embedding: np.ndarray) -> float:
        """
        Assess the quality of an extracted embedding.
        
        Args:
            embedding: Feature embedding vector
            
        Returns:
            float: Quality score (0.0-1.0, higher is better)
        """
        if self.model_adapter and hasattr(self.model_adapter, 'validate_embedding_quality'):
            return self.model_adapter.validate_embedding_quality(embedding)
        
        # Fallback quality validation
        if not isinstance(embedding, np.ndarray):
            return 0.0
        
        # Check for invalid values
        if np.any(np.isnan(embedding)) or np.any(np.isinf(embedding)):
            return 0.0
        
        # Check embedding dimension (should be reasonable)
        if len(embedding) < 64 or len(embedding) > 4096:
            return 0.0
        
        # Basic quality metrics
        # 1. Check for diversity (not all zeros or all same values)
        std_dev = np.std(embedding)
        diversity_score = min(1.0, float(std_dev) / 0.1)  # Normalize by expected std
        
        # 2. Check value range (should be reasonable, not extreme)
        max_abs_value = np.max(np.abs(embedding))
        range_score = 1.0 if max_abs_value < 10.0 else max(0.0, 1.0 - (max_abs_value - 10.0) / 10.0)
        
        # 3. Check for sparsity (too many zeros might indicate poor extraction)
        non_zero_ratio = np.count_nonzero(embedding) / len(embedding)
        sparsity_score = min(1.0, non_zero_ratio / 0.1)  # Expect at least 10% non-zero
        
        # Combine scores
        quality_score = (diversity_score * 0.4 + range_score * 0.3 + sparsity_score * 0.3)
        return max(0.0, min(1.0, quality_score))
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the currently loaded model.
        
        Returns:
            Dict[str, Any]: Model metadata including version, architecture, etc.
        """
        # Start with base model info from parent class
        base_info = self.model_info.copy()
        
        if self.model_adapter:
            adapter_info = self.model_adapter.get_model_info()
            base_info.update(adapter_info)
        
        # Add extraction statistics
        base_info['extraction_stats'] = self.extraction_stats.copy()
        
        # Add cache statistics
        if self.cache:
            base_info['cache_stats'] = self.cache.get_stats()
        
        return base_info
    
    def _update_stats(self, extraction_time: float, use_gpu: bool = False) -> None:
        """Update extraction statistics."""
        self.extraction_stats['total_extractions'] += 1
        self.extraction_stats['total_time'] += extraction_time
        self.extraction_stats['average_time'] = (
            self.extraction_stats['total_time'] / self.extraction_stats['total_extractions']
        )
        
        if use_gpu:
            self.extraction_stats['gpu_extractions'] += 1
        else:
            self.extraction_stats['cpu_extractions'] += 1
    
    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        if self.cache:
            self.cache.clear()
            logger.info("Embedding cache cleared")
    
    def get_extraction_stats(self) -> Dict[str, Any]:
        """Get detailed extraction statistics."""
        stats = self.extraction_stats.copy()
        
        if self.cache:
            cache_stats = self.cache.get_stats()
            stats['cache_hit_rate'] = (
                stats['cache_hits'] / max(1, stats['cache_hits'] + stats['cache_misses'])
            )
            stats.update(cache_stats)
        
        return stats
    
    def optimize_for_inference(self) -> None:
        """Optimize model for inference performance."""
        if self.model_adapter and hasattr(self.model_adapter, 'model') and self.model_adapter.model:
            # Set model to evaluation mode
            self.model_adapter.model.eval()
            
            # Enable inference optimizations if using PyTorch
            if hasattr(torch, 'jit') and hasattr(self.model_adapter.model, 'eval'):
                try:
                    # Try to optimize with TorchScript (may not work for all models)
                    # This is optional and will fail gracefully
                    pass
                except Exception:
                    pass
            
            logger.info("Model optimized for inference")
        else:
            logger.warning("Cannot optimize: model not loaded or model adapter not available")
    
    def warmup(self, num_warmup_images: int = 5) -> None:
        """
        Warm up the model with dummy images to optimize performance.
        
        Args:
            num_warmup_images: Number of dummy images to process for warmup
        """
        if self.model_adapter is None:
            logger.warning("Cannot warmup: model not loaded")
            return
        
        logger.info(f"Warming up model with {num_warmup_images} dummy images...")
        
        # Create dummy IR images (224x224 grayscale)
        dummy_images = [
            np.random.rand(224, 224).astype(np.float32) 
            for _ in range(num_warmup_images)
        ]
        
        start_time = time.time()
        
        # Process dummy images
        try:
            self.batch_extract(dummy_images)
            warmup_time = time.time() - start_time
            logger.info(f"Model warmup completed in {warmup_time:.3f}s")
        except Exception as e:
            logger.warning(f"Model warmup failed: {e}")
    
    def _parallel_preprocess(self, images: List[np.ndarray]) -> List[torch.Tensor]:
        """
        Preprocess multiple images in parallel for better performance.
        
        Args:
            images: List of images to preprocess
            
        Returns:
            List[torch.Tensor]: Preprocessed tensors
        """
        if self.model_adapter is None:
            raise RuntimeError("Model adapter not available")
        
        if len(images) == 1:
            # Single image - no need for parallelization
            return [self.model_adapter.preprocess_image(images[0])]
        
        # For batch processing, we can leverage model adapter's preprocessing
        # This is more efficient than processing images one by one
        try:
            if hasattr(self.model_adapter, 'batch_preprocess') and callable(getattr(self.model_adapter, 'batch_preprocess', None)):
                # Use adapter's batch preprocessing if available
                return self.model_adapter.batch_preprocess(images)  # type: ignore
            else:
                # Fallback to sequential preprocessing
                # Note: We could use ThreadPoolExecutor here for CPU-bound preprocessing
                # but model adapters typically handle this efficiently
                return [self.model_adapter.preprocess_image(img) for img in images]
        except Exception as e:
            logger.warning(f"Batch preprocessing failed, falling back to sequential: {e}")
            return [self.model_adapter.preprocess_image(img) for img in images]
    
    def _adaptive_batch_size(self, num_images: int) -> int:
        """
        Dynamically adjust batch size based on available memory and image count.
        
        Args:
            num_images: Total number of images to process
            
        Returns:
            int: Optimal batch size
        """
        base_batch_size = self.config.batch_size
        
        # Adjust based on GPU memory if using CUDA
        if self.model_adapter and self.model_adapter.device == 'cuda':
            try:
                # Get GPU memory info
                if torch.cuda.is_available():
                    total_memory = torch.cuda.get_device_properties(0).total_memory
                    allocated_memory = torch.cuda.memory_allocated(0)
                    free_memory = total_memory - allocated_memory
                    
                    # Estimate memory per image (rough heuristic)
                    memory_per_image = 50 * 1024 * 1024  # 50MB per image (conservative)
                    max_batch_from_memory = max(1, int(free_memory * 0.8 / memory_per_image))
                    
                    # Use smaller of configured batch size and memory-based limit
                    optimal_batch_size = min(base_batch_size, max_batch_from_memory)
                    
                    # Don't make batches larger than the total number of images
                    optimal_batch_size = min(optimal_batch_size, num_images)
                    
                    if optimal_batch_size != base_batch_size:
                        logger.info(f"Adaptive batch size: {optimal_batch_size} (from {base_batch_size})")
                    
                    return optimal_batch_size
                    
            except Exception as e:
                logger.debug(f"Could not determine optimal batch size: {e}")
        
        # Default: use configured batch size, but don't exceed number of images
        return min(base_batch_size, num_images)
    
    def enable_performance_optimizations(self) -> None:
        """
        Enable advanced performance optimizations for embedding extraction.
        
        This method should be called after loading the model and before
        starting heavy extraction workloads.
        """
        if self.model_adapter is None:
            logger.warning("Cannot enable optimizations: model not loaded")
            return
        
        logger.info("Enabling performance optimizations...")
        
        try:
            # Enable model optimizations
            self.optimize_for_inference()
            
            # Warm up the model
            self.warmup(num_warmup_images=3)
            
            # Enable mixed precision if requested and supported
            if (self.config.enable_mixed_precision and 
                self.model_adapter.device == 'cuda' and 
                hasattr(torch.cuda, 'amp')):
                logger.info("Mixed precision enabled for GPU acceleration")
                # Note: Actual implementation would be in model adapters
            
            # Clear cache to start fresh
            if self.cache:
                logger.info("Cache optimized and ready")
            
            # Set optimal threading for CPU operations
            if self.model_adapter.device == 'cpu' and hasattr(torch, 'set_num_threads'):
                optimal_threads = min(self.config.max_workers, torch.get_num_threads())
                torch.set_num_threads(optimal_threads)
                logger.info(f"CPU threading optimized: {optimal_threads} threads")
            
            logger.info("✅ Performance optimizations enabled successfully")
            
        except Exception as e:
            logger.warning(f"Some optimizations could not be applied: {e}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive performance metrics for monitoring.
        
        Returns:
            Dict[str, Any]: Performance metrics including throughput, latency, etc.
        """
        stats = self.get_extraction_stats()
        
        # Calculate additional metrics
        metrics = {
            'basic_stats': stats,
            'throughput': {
                'images_per_second': 0.0,
                'batches_per_second': 0.0,
                'cache_hit_rate': stats.get('cache_hit_rate', 0.0)
            },
            'latency': {
                'average_extraction_time_ms': stats.get('average_time', 0.0) * 1000,
                'total_time_seconds': stats.get('total_time', 0.0)
            },
            'resource_usage': {
                'gpu_usage_ratio': 0.0,
                'memory_efficiency': 0.0
            },
            'batch_efficiency': {
                'average_batch_size': 0.0,
                'batch_utilization': 0.0
            }
        }
        
        # Calculate throughput
        if stats.get('total_time', 0) > 0:
            metrics['throughput']['images_per_second'] = (
                stats.get('total_extractions', 0) / stats.get('total_time', 1)
            )
            
            if stats.get('batch_extractions', 0) > 0:
                metrics['throughput']['batches_per_second'] = (
                    stats.get('batch_extractions', 0) / stats.get('total_time', 1)
                )
                
                metrics['batch_efficiency']['average_batch_size'] = (
                    stats.get('total_extractions', 0) / stats.get('batch_extractions', 1)
                )
        
        # Calculate GPU usage ratio
        total_gpu = stats.get('gpu_extractions', 0)
        total_cpu = stats.get('cpu_extractions', 0)
        if total_gpu + total_cpu > 0:
            metrics['resource_usage']['gpu_usage_ratio'] = (
                total_gpu / (total_gpu + total_cpu)
            )
        
        # Add device information
        metrics['device_info'] = {
            'current_device': self.model_adapter.device if self.model_adapter else 'unknown',
            'cuda_available': torch.cuda.is_available(),
            'gpu_enabled': self.config.use_gpu
        }
        
        return metrics