# Configuration Directory

This directory contains configuration files and settings for the IR Image Classification System.

## Files

### `__init__.py`

Python package initialization file that makes the configuration directory a proper Python package.

### `settings.py`

The main configuration settings file for the IR Image Classification System. It contains:

- Database configuration (path, collection name, distance metrics)
- Model configuration (embedding dimensions, batch size, image requirements)
- Query processing parameters (thresholds, caching policies, performance limits)
- Training configuration (learning rates, data splits, augmentation settings)

### `test_config.json`

JSON configuration file used for testing purposes. Contains test-specific settings that can be loaded during test execution.

### `vector_db_config.json`

Configuration file for the vector database (ChromaDB) with settings for:

- Database connection parameters
- Collection configuration
- Distance metrics and indexing options
- Performance optimization settings

## Usage

Configuration settings are loaded at runtime by various components of the system:

```python
from config.settings import DATABASE_PATH, EMBEDDING_DIMENSION

# Use configuration in your code
db_path = DATABASE_PATH
embedding_dim = EMBEDDING_DIMENSION
```

## Environment Variables

The system supports overriding configuration settings using environment variables. Create a `.env` file in the project root to set environment-specific configurations:

```
DATABASE_PATH=/custom/path/to/database
EMBEDDING_DIMENSION=512
ENABLE_GPU=true
```

## Adding New Configuration

When adding new configuration parameters:

1. Add the parameter to `settings.py` with a default value
2. Add documentation for the parameter in comments
3. Update this README if the parameter is significant
