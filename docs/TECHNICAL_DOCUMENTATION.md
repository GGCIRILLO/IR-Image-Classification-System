# IR Image Classification System - Technical Documentation

This document provides a comprehensive technical overview of the IR Image Classification System, detailing its architecture, components, workflows, and implementation details.

## Table of Contents

1. [System Overview](#system-overview)
2. [Directory Structure](#directory-structure)
3. [Core Components](#core-components)
   - [Data Models](#data-models)
   - [Query Processing](#query-processing)
   - [Embedding Extraction](#embedding-extraction)
   - [Similarity Search](#similarity-search)
   - [Object Classification](#object-classification)
4. [Workflows](#workflows)
   - [Database Population](#database-population)
   - [Query Processing](#query-processing-workflow)
   - [Mission Execution](#mission-execution)
   - [Performance Improvement](#performance-improvement)
5. [Configuration](#configuration)
6. [Scripts](#scripts)
   - [Command Examples](#command-examples)
7. [Performance Optimization](#performance-optimization)
8. [Military-Specific Features](#military-specific-features)
9. [Troubleshooting](#troubleshooting)
10. [Advanced Usage](#advanced-usage)

## System Overview

The IR Image Classification System is a specialized computer vision solution designed for infrared (IR) image classification and similarity search. The system is optimized for military applications, providing high-accuracy object identification in thermal imagery with confidence scoring and detailed reporting capabilities.

Key features include:

- IR-specific image preprocessing and embedding extraction
- Vector similarity search with multiple ranking strategies
- Confidence scoring calibrated for military applications
- Comprehensive mission execution capabilities
- Performance optimization tools for deployment scenarios

The system follows a modular architecture with clear separation of concerns between data processing, embedding extraction, similarity search, and result ranking components.

### Technical Architecture

The system is built on a multi-layered architecture:

1. **Presentation Layer**

   - Command-line interfaces for mission execution and database management
   - Reporting formats (table, JSON, detailed, military)
   - Performance monitoring dashboards

2. **Application Layer**

   - Query processing engine
   - Mission execution workflows
   - Performance optimization tools
   - Database population utilities

3. **Domain Layer**

   - Object classification models
   - Embedding extraction services
   - Similarity search algorithms
   - Confidence calculation strategies

4. **Infrastructure Layer**
   - Vector database (ChromaDB)
   - Deep learning models (PyTorch)
   - File system storage
   - Caching mechanisms

### System Requirements

- **Hardware Requirements**

  - CPU: 4+ cores recommended (8+ for production)
  - RAM: 8GB minimum (16GB+ recommended)
  - GPU: CUDA-compatible GPU recommended for production
  - Storage: 10GB minimum for database and models

- **Software Requirements**
  - Python 3.8+
  - PyTorch 1.8+
  - ChromaDB
  - NumPy, PIL, and other dependencies
  - CUDA toolkit (for GPU acceleration)

## Directory Structure

```bash
/
├── cache/                  # Caching directory for embeddings
├── checkpoints/            # Model checkpoints and fine-tuning results
├── config/                 # Configuration files
├── data/                   # Data storage
│   ├── processed/          # Processed IR images
│   └── vector_db/          # Vector database storage
├── examples/               # Example query images
├── results/                # Output results
├── scripts/                # Utility and operational scripts
├── src/                    # Core source code
│   ├── data/               # Data processing modules
│   ├── database/           # Vector database interface
│   ├── embedding/          # Embedding extraction
│   ├── models/             # Data models and object classes
│   ├── query/              # Query processing engine
│   └── training/           # Model training utilities
└── tests/                  # Test suite
```

## Core Components

### Data Models

The system uses a set of well-defined data models to represent IR images, embeddings, and query results.

#### Key Data Models (`src/models/data_models.py`)

1. **IRImage**

   - Represents an infrared image with metadata
   - Handles validation of IR image format requirements
   - Provides conversion utilities between formats

2. **Embedding**

   - Represents a feature embedding extracted from an IR image
   - Includes validation for embedding quality
   - Provides normalization and similarity calculation methods

3. **SimilarityResult**

   - Represents a single similarity search result
   - Includes similarity score, confidence, and object classification
   - Provides methods for threat assessment and categorization

4. **QueryResult**
   - Represents the complete result of a similarity query
   - Contains a list of similarity results and processing metadata
   - Provides filtering and sorting methods for result analysis

#### Object Classification System (`src/models/object_classes.py`)

The system includes a comprehensive object classification registry with detailed metadata for each object class:

- **ObjectCategory**: Enum defining high-level categories (MILITARY_VEHICLE, CIVILIAN_VEHICLE, AIR_DEFENSE, etc.)
- **ObjectClass**: Dataclass representing a specific object class with metadata
- **ObjectClassRegistry**: Registry managing all object classes with lookup methods

The registry contains over 80 object classes organized into categories like:

- Military Vehicles (tanks, APCs, etc.)
- Civilian Vehicles
- Air Defense Systems
- Missile Systems
- Launch Pads
- Buildings
- Infrastructure
- Communication Systems

### Query Processing

The query processing engine (`src/query/processor.py`) handles the complete pipeline from image input to ranked similarity results.

#### QueryProcessor

The `QueryProcessor` class is the central component that orchestrates the entire query workflow:

1. **Initialization**

   - Loads configuration parameters
   - Initializes image processor, embedding extractor, and similarity searcher
   - Sets up ranking and confidence calculation components

2. **Query Processing Pipeline**

   - Validates and preprocesses query images
   - Extracts embeddings using the embedding extractor
   - Performs similarity search against the vector database
   - Ranks and filters results based on confidence and similarity scores
   - Applies military-specific ranking strategies when configured

3. **Performance Tracking**

   - Monitors query processing time
   - Tracks component-level performance metrics
   - Validates system performance against requirements

4. **Caching**
   - Implements query result caching for improved performance
   - Manages cache size and eviction policies

### Embedding Extraction

The embedding extraction service (`src/embedding/extractor.py`) handles the extraction of feature embeddings from IR images.

#### EmbeddingExtractor

The `EmbeddingExtractor` class provides high-performance embedding extraction with:

1. **Model Management**

   - Supports multiple model architectures (ResNet50, Qwen VLM)
   - Handles model loading and initialization
   - Provides model information and version tracking

2. **Extraction Features**

   - Single and batch embedding extraction
   - GPU acceleration with CUDA support
   - Mixed precision for performance optimization
   - Embedding quality validation

3. **Caching System**

   - Thread-safe embedding cache
   - Memory and disk-based caching
   - Cache statistics and management

4. **Performance Optimization**
   - Adaptive batch sizing based on available memory
   - Parallel preprocessing for improved throughput
   - Warmup procedures for consistent performance

### Similarity Search

The similarity search component (`src/database/similarity_searcher.py`) provides efficient vector similarity search capabilities.

#### SimilaritySearcher

The `SimilaritySearcher` class implements:

1. **Search Modes**

   - EXACT: Precise but slower search for critical applications
   - APPROXIMATE: Fast approximate search using HNSW algorithm
   - HYBRID: Starts with approximate search, falls back to exact if needed

2. **Enhanced Similarity Calculation**

   - IR-specific similarity score computation
   - Confidence calculation based on similarity and rank
   - Distance metric optimization for thermal imagery

3. **Performance Features**

   - Query result caching
   - Search performance metrics
   - Configurable search parameters

4. **Integration with ChromaDB**
   - Persistent vector database storage
   - Metadata filtering capabilities
   - Efficient vector indexing

### Object Classification

The object classification system provides a comprehensive registry of object classes with detailed metadata.

#### Key Features

1. **Hierarchical Classification**

   - High-level categories (ObjectCategory)
   - Specific object classes with detailed metadata
   - Relationships between classes and categories

2. **Military Intelligence Features**

   - Methods to identify military assets
   - Critical asset identification
   - Threat level assessment

3. **Lookup Capabilities**
   - Lookup by standardized name
   - Lookup by folder name
   - Category-based filtering

## Workflows

### Database Population

The database population workflow prepares the vector database with embeddings extracted from processed IR images.

#### Process Flow (`scripts/populate_database.py`)

1. **Image Collection**

   - Scans processed image directory
   - Organizes images by object class
   - Applies sampling strategies (max per class, max total)

2. **Embedding Extraction**

   - Loads and initializes embedding extractor
   - Processes images to extract feature embeddings
   - Validates embedding quality

3. **Database Storage**
   - Initializes vector database
   - Stores embeddings with metadata
   - Verifies database integrity

### Query Processing Workflow

The query processing workflow handles the end-to-end process of classifying an IR image.

#### Process Flow

1. **Query Validation**

   - Validates input image format
   - Checks for IR-specific requirements
   - Applies preprocessing for optimal results

2. **Embedding Extraction**

   - Extracts feature embedding from query image
   - Validates embedding quality
   - Applies normalization for better similarity

3. **Similarity Search**

   - Performs vector similarity search against database
   - Applies filters and thresholds
   - Returns candidate matches

4. **Result Ranking**

   - Applies ranking strategy (similarity, confidence, hybrid, military)
   - Calculates confidence scores
   - Filters results based on thresholds

5. **Result Formatting**
   - Organizes results with metadata
   - Applies formatting based on output requirements
   - Generates comprehensive query result

### Mission Execution

The mission execution workflow (`scripts/run_mission.py`) provides a command-line interface for running complete IR classification missions.

#### Process Flow

1. **Configuration**

   - Parses command-line arguments
   - Applies configuration presets (military, development, etc.)
   - Validates inputs and parameters

2. **System Initialization**

   - Initializes query processor with configuration
   - Sets up model and database connections
   - Configures ranking and confidence strategies

3. **Query Execution**

   - Processes query image
   - Applies specified strategies and thresholds
   - Collects and organizes results

4. **Result Reporting**

   - Formats results based on output format (table, JSON, detailed, military)
   - Generates comprehensive reports
   - Saves results to file if requested

5. **Performance Validation**
   - Validates system performance against requirements
   - Reports performance metrics
   - Identifies potential issues

## Configuration

The system uses a centralized configuration system (`config/settings.py`) with settings for:

1. **Database Configuration**

   - Database path and collection name
   - Distance metric settings
   - Connection parameters

2. **Model Configuration**

   - Embedding dimensions
   - Batch size settings
   - Image size requirements
   - Supported image formats

3. **Training Configuration**

   - Learning rate and epochs
   - Data split ratios
   - Augmentation settings
   - Target accuracy thresholds

4. **Query Configuration**
   - Maximum query time limits
   - Result count settings
   - Confidence thresholds
   - Caching policies

## Scripts

The system includes a comprehensive set of scripts for various operations:

### Operational Scripts

1. **run_mission.py**

   - Comprehensive command-line interface for mission execution
   - Multiple configuration presets
   - Flexible ranking strategies
   - Advanced confidence scoring
   - Multiple output formats

2. **populate_database.py**
   - Populates vector database with embeddings
   - Configurable sampling strategies
   - Verification capabilities
   - Performance monitoring

### Performance Improvement Scripts

1. **quick_improvements.py**

   - Immediate performance boost without retraining
   - IR-specific normalization enhancements
   - Similarity score calculation improvements
   - Threshold optimizations

2. **enhanced_fine_tuning.py**

   - Comprehensive model fine-tuning
   - Advanced learning techniques
   - IR-specific optimizations
   - Confidence calibration

3. **fix_similarity_confidence.py**

   - Targeted fixes for similarity and confidence issues
   - IR-specific similarity boosting
   - Confidence score recalibration
   - Distance metric optimization

4. **train_improved_model.py**
   - Advanced training pipeline
   - Weighted sampling for imbalanced classes
   - One-cycle learning rate scheduling
   - Advanced augmentation for IR images

### Command Examples

Below are detailed examples of how to use the main scripts with various command-line options:

#### Run Mission Script Examples

1. **Basic Object Identification**:

   ```bash
   python scripts/run_mission.py --image examples/tank.png --database data/vector_db
   ```

2. **Military Intelligence Operation**:

   ```bash
   python scripts/run_mission.py --image surveillance_image.png --database data/vector_db \
     --preset military --strategy military_priority \
     --confidence-strategy military_calibrated --format military \
     --operator "Analyst_Alpha" --classification SECRET \
     --mission-id "OP_EAGLE_EYE_001"
   ```

3. **Development Testing with Debug Output**:

   ```bash
   python scripts/run_mission.py --image test_image.png --database data/vector_db \
     --preset development --debug --output test_results.json \
     --save-metadata
   ```

4. **High-Precision Analysis**:

   ```bash
   python scripts/run_mission.py --image critical_target.png --database data/vector_db \
     --confidence-threshold 0.9 --similarity-threshold 0.8 \
     --validation-mode strict --max-results 3 \
     --format detailed
   ```

5. **Using Custom Model**:
   ```bash
   python scripts/run_mission.py --image query.png --database data/vector_db \
     --model checkpoints/fine_tuned_resnet50.pth \
     --collection custom_embeddings
   ```

#### Database Population Examples

1. **Basic Database Population**:

   ```bash
   python scripts/populate_database.py --database-path data/vector_db \
     --processed-dir data/processed
   ```

2. **Limited Sampling for Testing**:

   ```bash
   python scripts/populate_database.py --database-path data/chroma_db_test \
     --processed-dir data/processed --max-per-class 3 --max-total 30
   ```

3. **Process All Available Images**:

   ```bash
   python scripts/populate_database.py --database-path data/chroma_db_full \
     --processed-dir data/processed --all-images
   ```

4. **Verify Existing Database**:

   ```bash
   python scripts/populate_database.py --database-path data/vector_db \
     --verify-only
   ```

5. **Dry Run (No Database Changes)**:
   ```bash
   python scripts/populate_database.py --database-path data/vector_db \
     --processed-dir data/processed --dry-run
   ```

#### Performance Improvement Examples

1. **Quick Improvements**:

   ```bash
   python scripts/quick_improvements.py --database data/vector_db \
     --test-image examples/test_image.png
   ```

2. **Apply All Similarity and Confidence Fixes**:

   ```bash
   python scripts/fix_similarity_confidence.py --database data/vector_db \
     --apply-all --test-image examples/test_image.png
   ```

3. **Enhanced Fine-tuning**:

   ```bash
   python scripts/enhanced_fine_tuning.py --train-data data/processed \
     --database data/vector_db --model-type resnet50 \
     --epochs 50 --batch-size 16
   ```

4. **Advanced Training Pipeline**:
   ```bash
   python scripts/train_improved_model.py --data-dir data/processed \
     --model-type resnet50 --epochs 100 --learning-rate 1e-4
   ```

## Performance Optimization

The system includes several performance optimization features:

1. **Embedding Caching**

   - Memory and disk-based caching
   - Thread-safe implementation
   - Cache statistics and management
   - Configurable cache size

2. **GPU Acceleration**

   - CUDA support for embedding extraction
   - Mixed precision for improved performance
   - Memory optimization for batch processing
   - Adaptive batch sizing

3. **Query Optimization**

   - Query result caching
   - Approximate search for faster results
   - Parallel preprocessing
   - Early termination for time-constrained queries

4. **Database Optimization**
   - Efficient vector indexing
   - Metadata-based filtering
   - Connection pooling
   - Query planning

## Military-Specific Features

The system includes several features specifically designed for military applications:

1. **Military Classification**

   - Comprehensive object class registry with military assets
   - Threat level assessment
   - Critical asset identification
   - Military category organization

2. **Military Ranking Strategies**

   - Military priority ranking
   - Confidence calibration for military applications
   - Threat-based result ordering
   - Military-specific thresholds

3. **Military Reporting**

   - Military intelligence report format
   - Classification levels (UNCLASSIFIED to TOP_SECRET)
   - Operator tracking
   - Mission ID management

4. **Military Validation**
   - Strict validation modes for critical applications
   - Performance requirement validation
   - Confidence calibration for military standards
   - Audit trail capabilities

## Troubleshooting

This section provides solutions to common issues encountered when working with the IR Image Classification System.

### Database Connection Issues

1. **ChromaDB Connection Failures**

   - **Symptom**: Error messages like "Failed to connect to database" or "Collection not found"
   - **Solution**:

     ```bash
     # Verify database exists
     ls -la data/vector_db

     # Check database integrity
     python scripts/populate_database.py --database-path data/vector_db --verify-only

     # Recreate database if needed
     python scripts/populate_database.py --database-path data/vector_db --processed-dir data/processed --all-images
     ```

2. **Collection Not Found**

   - **Symptom**: "Collection 'ir_embeddings' not found" error
   - **Solution**: Ensure you're using the correct collection name or create a new collection:

     ```bash
     # Check available collections
     python -c "import chromadb; client = chromadb.PersistentClient('data/vector_db'); print(client.list_collections())"

     # Use correct collection name in commands
     python scripts/run_mission.py --image query.png --database data/vector_db --collection correct_collection_name
     ```

### Image Processing Issues

1. **Unsupported Image Format**

   - **Symptom**: "Unsupported image format" or "Failed to load image" errors
   - **Solution**: Convert image to a supported format:

     ```bash
     # Convert to PNG using ImageMagick
     convert input_image.tiff output_image.png

     # Then run the mission
     python scripts/run_mission.py --image output_image.png --database data/vector_db
     ```

2. **IR Format Validation Failures**
   - **Symptom**: "Image does not meet IR format requirements" error
   - **Solution**: Use relaxed validation mode:
     ```bash
     python scripts/run_mission.py --image query.png --database data/vector_db --validation-mode relaxed
     ```

### Performance Issues

1. **Slow Query Processing**

   - **Symptom**: Queries taking longer than 2 seconds
   - **Solution**: Enable GPU acceleration and optimize search parameters:

     ```bash
     # Enable GPU and optimize search
     python scripts/run_mission.py --image query.png --database data/vector_db --max-query-time 5.0

     # Apply quick improvements
     python scripts/quick_improvements.py --database data/vector_db
     ```

2. **Out of Memory Errors**
   - **Symptom**: "CUDA out of memory" or similar errors
   - **Solution**: Reduce batch size or disable GPU:
     ```bash
     # Disable GPU
     python scripts/run_mission.py --image query.png --database data/vector_db --disable-gpu
     ```

### Model Issues

1. **Model Loading Failures**

   - **Symptom**: "Failed to load model" or "Model file not found" errors
   - **Solution**: Verify model path or use default model:

     ```bash
     # Check if model exists
     ls -la checkpoints/your_model.pth

     # Run without specifying model (uses default)
     python scripts/run_mission.py --image query.png --database data/vector_db
     ```

2. **Low Confidence Results**

   - **Symptom**: All results have very low confidence scores
   - **Solution**: Adjust confidence threshold or apply confidence fixes:

     ```bash
     # Lower confidence threshold
     python scripts/run_mission.py --image query.png --database data/vector_db --confidence-threshold 0.3

     # Apply confidence fixes
     python scripts/fix_similarity_confidence.py --database data/vector_db --apply-all
     ```

## Advanced Usage

This section covers advanced usage scenarios and techniques for the IR Image Classification System.

### Custom Model Integration

To integrate a custom-trained model with the system:

1. **Train or fine-tune your model** using the provided scripts:

   ```bash
   python scripts/enhanced_fine_tuning.py --train-data data/processed --epochs 100 --model-type resnet50 --batch-size 16
   ```

2. **Save the model** to the checkpoints directory:

   ```bash
   # Model will be saved automatically to checkpoints/fine_tuning
   ```

3. **Use the custom model** for populating the database:

   ```bash
   python scripts/populate_database.py --database-path data/vector_db \
     --processed-dir data/processed --model-path path/to/your_model.pth
   ```

4. **Use the custom model** in queries:
   ```bash
   python scripts/run_mission.py --image query.png --database data/vector_db --model path/to/your_model.pth
   ```

### Database Management

For advanced database management:

1. **Create multiple specialized databases**:

   ```bash
   # Create database for tanks only
   python scripts/populate_database.py --database-path data/tanks_db \
     --processed-dir data/processed/tanks --all-images

   # Create database for missile systems
   python scripts/populate_database.py --database-path data/missiles_db \
     --processed-dir data/processed/missiles --all-images
   ```

2. **Merge databases** (requires custom script):

   ```bash
   # Example of merging databases (conceptual)
   python scripts/merge_databases.py --source data/tanks_db data/missiles_db --target data/combined_db
   ```

3. **Database backup and restoration**:

   ```bash
   # Backup
   cp -r data/vector_db data/chroma_db_backup

   # Restore
   rm -rf data/vector_db
   cp -r data/chroma_db_backup data/vector_db
   ```
