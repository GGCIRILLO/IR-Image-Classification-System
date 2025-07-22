# IR Image Classification System - Project Structure

This document provides a comprehensive overview of the project structure for the IR Image Classification System.

## Directory Structure

```bash
ir-image-classification/
├── .git/                  # Git repository data
├── .gitignore             # Git ignore patterns
├── .pre-commit-config.yaml # Pre-commit hook configuration
├── .pytest_cache/         # Pytest cache data
├── .venv/                 # Python virtual environment
├── .vscode/               # VS Code configuration
├── Makefile               # Build automation
├── README.md              # Project overview
├── cache/                 # Caching directory
│   └── embeddings/        # Cached embedding vectors
├── checkpoints/           # Model checkpoints
│   ├── fine_tuning/       # Fine-tuned model weights
│   └── logs/              # Training logs
├── config/                # Configuration files
│   ├── settings.py        # Main configuration settings
│   ├── test_config.json   # Test configuration
│   └── vector_db_config.json # Vector database configuration
├── data/                  # Data storage
│   ├── processed/         # Processed IR images
│   ├── processed.zip      # Compressed dataset
│   └── vector_db/         # Vector database storage
├── docs/                  # Documentation files
├── examples/              # Example query images
├── pyproject.toml         # Python project configuration
├── pytest.ini             # Pytest configuration
├── requirements.txt       # Python dependencies
├── scripts/               # Utility scripts
│   ├── augmentation.py    # Data augmentation script
│   ├── enhanced_fine_tuning.py # Fine-tuning script
│   ├── fix_similarity_confidence.py # Similarity fixing script
│   ├── init_database.py   # Database initialization
│   ├── populate_database.py # Database population
│   ├── quick_improvements.py # Quick improvement script
│   ├── run_mission.py     # Mission execution script
│   └── train_improved_model.py # Model training script
├── src/                   # Core source code
│   ├── data/              # Data processing modules
│   ├── database/          # Vector database interface
│   ├── embedding/         # Embedding extraction
│   ├── models/            # Data models and interfaces
│   ├── query/             # Query processing
│   └── training/          # Model training utilities
└── tests/                 # Test suite
```

## Key Components

### Core Components

1. **Query Processing (`src/query/processor.py`)**

   - Central component orchestrating the query workflow
   - Handles image validation, preprocessing, embedding extraction, and result ranking
   - Implements performance tracking and caching

2. **Embedding Extraction (`src/embedding/extractor.py`)**

   - Extracts feature embeddings from IR images
   - Supports multiple model architectures
   - Implements caching and batch processing
   - Validates embedding quality

3. **Similarity Search (`src/database/similarity_searcher.py`)**

   - Performs efficient vector similarity searches
   - Implements exact, approximate, and hybrid search modes
   - Enhances similarity calculation for IR images
   - Integrates with ChromaDB

4. **Object Classification (`src/models/object_classes.py`)**

   - Defines comprehensive object classification registry
   - Organizes objects into categories (military vehicles, air defense, etc.)
   - Provides military intelligence features
   - Implements lookup capabilities

5. **Data Models (`src/models/data_models.py`)**
   - Defines core data structures (IRImage, Embedding, SimilarityResult, QueryResult)
   - Implements validation and utility methods
   - Provides military-specific assessment methods

### Operational Scripts

1. **Mission Runner (`scripts/run_mission.py`)**

   - Command-line interface for executing classification missions
   - Supports multiple configuration presets and output formats
   - Implements comprehensive parameter handling
   - Provides detailed result reporting

2. **Database Population (`scripts/populate_database.py`)**
   - Populates vector database with embeddings
   - Implements sampling strategies and verification
   - Monitors performance and database integrity

### Performance Improvement Scripts

1. **Quick Improvements (`scripts/quick_improvements.py`)**

   - Immediate performance improvements without retraining
   - Enhances embedding extraction and similarity calculation
   - Optimizes search parameters

2. **Enhanced Fine-tuning (`scripts/enhanced_fine_tuning.py`)**
   - Comprehensive model fine-tuning
   - Implements contrastive learning and advanced loss functions
   - Provides IR-specific optimizations

## Key Features

1. **High Precision Classification**

   - 95%+ accuracy on IR object classification
   - Specialized for military object identification
   - Comprehensive object class registry with 80+ classes

2. **Fast Query Processing**

   - Sub-2 second response times
   - Efficient vector similarity search
   - Multi-level caching system

3. **Local Deployment**

   - Uses local vector database (ChromaDB)
   - No external API dependencies
   - Secure operation for sensitive data

4. **Military-Specific Features**

   - Threat level assessment
   - Military asset identification
   - Classification levels for reports
   - Military-optimized ranking strategies

5. **Performance Optimization**
   - GPU acceleration
   - Batch processing
   - Embedding caching
   - Approximate search for large datasets
