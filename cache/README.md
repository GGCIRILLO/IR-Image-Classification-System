# Cache Directory

This directory stores cached data for the IR Image Classification System to improve performance and reduce redundant computations.

## Structure

### `embeddings/`

Contains cached embedding vectors extracted from IR images:

- Organized by hash keys based on image content and model version
- Stored in pickle format for efficient loading
- Includes metadata for cache management

## Caching System

The system implements a two-level caching mechanism:

1. **Memory Cache**: Fast in-memory cache for frequently accessed embeddings
2. **Disk Cache**: Persistent storage for embeddings in this directory

The `EmbeddingCache` class in `src/embedding/extractor.py` manages this caching system with features like:

- Thread-safe operations for concurrent access
- Cache size management with eviction policies
- Cache statistics and monitoring
- Hash-based lookup for efficient retrieval

## Cache Benefits

Caching provides significant performance improvements:

- Eliminates redundant embedding extraction for the same images
- Reduces GPU/CPU load for repeated queries
- Improves query response time by up to 10x for cached images
- Enables efficient batch processing with partial cache hits

## Cache Management

The cache is automatically managed by the system, but can also be controlled manually:

```python
# Clear the embedding cache
from src.embedding.extractor import EmbeddingExtractor

extractor = EmbeddingExtractor()
extractor.clear_cache()
```

Or using the command line:

```bash
# Run mission with cache disabled
python scripts/run_mission.py --image query.png --database data/vector_db --disable-cache
```

## Cache Statistics

The system tracks cache performance metrics:

- Hit rate: Percentage of cache hits vs. misses
- Memory usage: Size of in-memory cache
- Disk usage: Size of on-disk cache
- Time savings: Processing time saved through caching

These statistics can be accessed through the `get_stats()` method of the `EmbeddingCache` class or through the `get_extraction_stats()` method of the `EmbeddingExtractor` class.

## Cache Configuration

Cache behavior can be configured in `config/settings.py`:

- `CACHE_EMBEDDINGS`: Enable/disable caching (default: True)
- `CACHE_DIR`: Path to cache directory (default: "./cache/embeddings")
- `MAX_CACHE_SIZE`: Maximum number of embeddings in memory cache (default: 10000)
- `CACHE_POLICY`: Caching policy (FIFO, LRU, etc.)
