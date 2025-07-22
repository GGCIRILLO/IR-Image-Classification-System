# Embedding Module Documentation

The `embedding` module is responsible for extracting feature embeddings from IR images. These embeddings are vector representations that capture the semantic content of images and enable similarity-based search and classification.

## Files and Classes

### `extractor.py`

This file contains classes for extracting embeddings from IR images and caching them for improved performance.

#### `ExtractionConfig`

Configuration class for embedding extraction settings.

#### `EmbeddingExtractor`

The main class responsible for extracting embeddings from IR images using various deep learning models.

**Key Methods:**

- `__init__(self, model_type: str = "resnet50", config: Optional[ExtractionConfig] = None, model_path: Optional[str] = None)`: Initializes the extractor with a specified model type and configuration.
- `load_model(self, model_path: Optional[str] = None)`: Loads a pre-trained model for embedding extraction.
- `extract_embedding(self, image: np.ndarray)`: Extracts an embedding vector from a single image.
- `batch_extract(self, images: List[np.ndarray])`: Extracts embeddings from multiple images in a batch.
- `validate_embedding_quality(self, embedding: np.ndarray)`: Validates the quality of an extracted embedding.
- `get_model_info(self)`: Returns information about the loaded model.
- `get_extraction_stats(self)`: Returns statistics about the extraction process.
- `get_performance_metrics(self)`: Returns performance metrics for the extraction process.
- `clear_cache(self)`: Clears the embedding cache.
- `optimize_for_inference(self)`: Optimizes the model for inference.
- `enable_performance_optimizations(self)`: Enables various performance optimizations.
- `warmup(self, num_warmup_images: int = 5)`: Performs warmup inference to initialize the model.

**Internal Methods:**

- `_batch_extract_uncached(self, images: List[np.ndarray])`: Extracts embeddings for a batch of images not in cache.
- `_parallel_preprocess(self, images: List[np.ndarray])`: Preprocesses multiple images in parallel.
- `_adaptive_batch_size(self, num_images: int)`: Determines the optimal batch size based on the number of images.
- `_update_stats(self, extraction_time: float, use_gpu: bool = False)`: Updates extraction performance statistics.

#### `EmbeddingCache`

A cache for storing and retrieving embeddings to avoid redundant extraction.

**Key Methods:**

- `__init__(self, cache_dir: str = "./cache/embeddings", max_size: int = 10000)`: Initializes the cache with directory and size limit.
- `get(self, image_data: np.ndarray, model_version: str)`: Retrieves an embedding from the cache.
- `put(self, image_data: np.ndarray, model_version: str, embedding: np.ndarray)`: Stores an embedding in the cache.
- `clear(self)`: Clears all cached embeddings.
- `get_stats(self)`: Returns statistics about the cache usage.
- `_get_cache_key(self, image_data: np.ndarray, model_version: str)`: Generates a unique key for caching.

The embedding module is a critical component of the IR image classification system, as it transforms raw image data into feature vectors that can be efficiently compared and searched. The quality and efficiency of embedding extraction directly impacts the overall performance of the system.
