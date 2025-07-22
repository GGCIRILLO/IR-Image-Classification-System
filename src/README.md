# Source Code Directory

This directory contains the core source code for the IR Image Classification System, organized into modular components.

## Package Structure

### `data/`

Data processing and manipulation modules:

- `ir_processor.py`: IR image preprocessing and validation
- `augmentation.py`: Data augmentation techniques for IR images
- `dataset.py`: Dataset classes for training and evaluation
- `transforms.py`: Custom transformations for IR images

### `database/`

Vector database interface and similarity search:

- `similarity_searcher.py`: High-performance similarity search with exact and approximate methods
- `chroma_client.py`: ChromaDB client wrapper with enhanced functionality
- `index_manager.py`: Vector index management and optimization
- `metadata_manager.py`: Metadata storage and retrieval

### `embedding/`

Embedding extraction and model inference:

- `extractor.py`: Feature embedding extraction with caching and batch processing
- `quality.py`: Embedding quality validation and metrics
- `normalization.py`: IR-specific embedding normalization techniques
- `cache.py`: Thread-safe embedding cache implementation

### `models/`

Data models and interfaces:

- `data_models.py`: Core data structures (IRImage, Embedding, SimilarityResult, QueryResult)
- `object_classes.py`: Object classification registry with military categories
- `interfaces.py`: Abstract interfaces for system components
- `config_models.py`: Configuration data models

### `query/`

Query processing and similarity search:

- `processor.py`: Main query processing engine
- `ranker.py`: Result ranking with multiple strategies
- `confidence.py`: Confidence calculation and calibration
- `validation.py`: Query validation and error handling

### `training/`

Model training and fine-tuning:

- `trainer.py`: Model training pipeline
- `losses.py`: Custom loss functions for IR image embeddings
- `metrics.py`: Training and evaluation metrics
- `model_adapters.py`: Adapters for different model architectures

### `ui/`

User interface components:

- `app.py`: Streamlit web interface
- `visualizer.py`: Result visualization utilities
- `components.py`: Reusable UI components
- `military_ui.py`: Military-specific UI elements

## Key Classes

### QueryProcessor

The central component that orchestrates the entire query workflow:

- Validates and preprocesses query images
- Extracts embeddings using the embedding extractor
- Performs similarity search against the vector database
- Ranks and filters results based on confidence and similarity scores

### EmbeddingExtractor

Handles the extraction of feature embeddings from IR images:

- Supports multiple model architectures (ResNet50, Qwen VLM)
- Provides single and batch embedding extraction
- Implements caching for performance optimization
- Validates embedding quality

### SimilaritySearcher

Provides efficient vector similarity search capabilities:

- Implements exact, approximate, and hybrid search modes
- Enhances similarity calculation for IR images
- Integrates with ChromaDB for persistent storage
- Includes performance optimization features

### ObjectClassRegistry

Manages the comprehensive object classification system:

- Defines hierarchical classification (categories and classes)
- Provides military intelligence features
- Implements lookup capabilities by name and folder

## Usage Example

```python
from src.query import QueryProcessor
from src.models.data_models import QueryResult

# Initialize processor
processor = QueryProcessor(
    database_path="data/vector_db",
    model_path="checkpoints/fine_tuned_model.pth"
)
processor.initialize()

# Process query
result = processor.process_query("path/to/ir_image.png")

# Display results
for i, match in enumerate(result.results, 1):
    print(f"{i}. {match.object_class}: {match.confidence:.3f}")
```
