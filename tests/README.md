# Tests Directory

This directory contains the test suite for the IR Image Classification System, organized by test type and component.

## Test Structure

### Unit Tests

Tests for individual components and functions:

- `test_embedding_extractor.py`: Tests for embedding extraction functionality
- `test_similarity_searcher.py`: Tests for similarity search operations
- `test_query_processor.py`: Tests for query processing pipeline
- `test_data_models.py`: Tests for core data models and validation
- `test_object_classes.py`: Tests for object classification system

### Integration Tests

Tests for component interactions:

- `test_database_integration.py`: Tests for database operations and queries
- `test_query_pipeline.py`: Tests for the complete query pipeline
- `test_military_features.py`: Tests for military-specific functionality
- `test_performance_metrics.py`: Tests for performance monitoring and validation

### End-to-End Tests

Tests for complete system workflows:

- `test_mission_execution.py`: Tests for end-to-end mission execution
- `test_database_population.py`: Tests for database population workflow
- `test_fine_tuning.py`: Tests for model fine-tuning process

### Performance Tests

Tests for system performance and optimization:

- `test_query_performance.py`: Tests for query processing speed
- `test_batch_processing.py`: Tests for batch embedding extraction
- `test_caching.py`: Tests for caching mechanisms
- `test_gpu_acceleration.py`: Tests for GPU acceleration

## Test Data

The `test_data/` subdirectory contains:

- Sample IR images for testing
- Mock embeddings and results
- Test configuration files
- Expected output files for comparison

## Running Tests

### Run All Tests

```bash
pytest
```

### Run with Coverage

```bash
pytest --cov=src
```

### Run Specific Test Categories

```bash
# Run unit tests
pytest -m unit

# Run integration tests
pytest -m integration

# Run performance tests
pytest -m performance
```

### Run Specific Test File

```bash
pytest tests/test_query_processor.py
```

## Test Configuration

The test suite uses configuration from:

- `pytest.ini`: Main pytest configuration
- `conftest.py`: Test fixtures and shared utilities
- `config/test_config.json`: Test-specific configuration values

## Adding New Tests

When adding new tests:

1. Follow the naming convention: `test_*.py` for files, `test_*` for functions
2. Use appropriate markers for test categorization
3. Create fixtures for reusable test components
4. Include docstrings explaining test purpose and expectations
5. Ensure tests are isolated and don't depend on external state
