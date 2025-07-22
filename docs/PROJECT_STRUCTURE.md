# Project Structure Documentation

This document provides an overview of the IR Image Classification project structure, explaining the purpose and organization of each directory.

## Root Directory

The root directory contains the main application files, configuration files, and directories for various components of the system.

- `app.py`: The main Streamlit application for interacting with the IR image classification system
- `.env.example`: Example environment variables configuration
- `Makefile`: Contains commands for building, testing, and running the project
- `pyproject.toml`: Python project configuration
- `pytest.ini`: Configuration for pytest
- `README.md`: Main project documentation
- `requirements.txt`: Python dependencies

## Source Code Structure

The `src` directory contains the core implementation of the IR image classification system, organized into several modules:

### `data` Module

Contains functionality for processing IR images, including loading, validation, preprocessing, and enhancement.

[Detailed Documentation](data_module.md)

### `database` Module

Provides functionality for storing, indexing, and searching vector embeddings of IR images using ChromaDB.

[Detailed Documentation](database_module.md)

### `embedding` Module

Responsible for extracting feature embeddings from IR images using various deep learning models.

[Detailed Documentation](embedding_module.md)

### `fine_tuning` Module

Provides tools for optimizing pre-trained models to improve their performance on IR image classification tasks.

[Detailed Documentation](fine_tuning_module.md)

### `models` Module

Defines core data structures, interfaces, and object classification taxonomy used throughout the system.

[Detailed Documentation](models_module.md)

### `query` Module

Implements functionality for processing IR image queries, ranking results, and calculating confidence scores.

[Detailed Documentation](query_module.md)

### `training` Module

Provides tools for training, fine-tuning, and validating deep learning models for IR image classification.

[Detailed Documentation](training_module.md)

## Supporting Directories

### `cache`

Stores cached data to improve performance, including:

- Embedding cache for avoiding redundant embedding extraction
- Search result cache for frequently performed queries
- Model optimization cache

[Detailed Documentation](../cache/README.md)

### `checkpoints`

Stores model checkpoints and training logs:

- Fine-tuning checkpoints
- Training progress logs
- Best model snapshots

[Detailed Documentation](../checkpoints/README.md)

### `config`

Contains configuration files for various components of the system:

- Database configuration
- Model configuration
- Training configuration
- Query processing configuration
- Test configuration

### `data`

Stores data files used by the system:

- Processed IR images
- Vector database files
- Test and validation datasets

### `docs`

Contains detailed documentation for the project:

- Module documentation
- Technical specifications
- User guides
- API documentation

### `examples`

Contains example code and notebooks demonstrating the use of the system.

### `results`

Stores results from system runs:

- Query results
- Evaluation metrics
- Performance benchmarks
- Validation reports

### `scripts`

Contains utility scripts for various tasks:

- Database population
- Model fine-tuning
- System testing
- Batch processing

[Detailed Documentation](scripts.md)

### `tests`

Contains unit tests and integration tests for the system.

## Development and Build Directories

- `.git`: Git version control directory
- `.venv`: Python virtual environment
- `.pytest_cache`: Pytest cache directory
- `.vscode`: VS Code configuration

## Application Interface

The Streamlit application (`app.py`) provides a user-friendly web interface for interacting with the IR image classification system.

[Detailed Documentation](streamlit_app.md)

## Workflow

The typical workflow for using this system involves:

1. **Data Preparation**: Processing IR images using the `data` module
2. **Embedding Extraction**: Extracting embeddings using the `embedding` module
3. **Database Population**: Storing embeddings using the `database` module
4. **Model Fine-Tuning**: Optimizing models using the `fine_tuning` module
5. **Query Processing**: Processing queries using the `query` module
6. **Result Analysis**: Analyzing results through the Streamlit application

Each of these steps is supported by the corresponding module in the `src` directory and can be performed through the Streamlit application or using the utility scripts in the `scripts` directory.
