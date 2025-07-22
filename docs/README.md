# IR Image Classification Documentation

Welcome to the documentation for the IR Image Classification system. This documentation provides comprehensive information about the system's architecture, components, and usage.

## Overview

The IR Image Classification system is designed for processing, analyzing, and classifying infrared (IR) images. It uses deep learning models to extract embeddings from IR images and performs similarity-based search and classification.

## Documentation Structure

### Module Documentation

- [Data Module](data_module.md): Documentation for the IR image processing module
- [Database Module](database_module.md): Documentation for the vector database module
- [Embedding Module](embedding_module.md): Documentation for the embedding extraction module
- [Fine-Tuning Module](fine_tuning_module.md): Documentation for the model fine-tuning module
- [Models Module](models_module.md): Documentation for the core data models and interfaces
- [Query Module](query_module.md): Documentation for the query processing module
- [Training Module](training_module.md): Documentation for the model training module

### Additional Documentation

- [Project Structure](PROJECT_STRUCTURE.md): Overview of the project's directory structure
- [Technical Documentation](TECHNICAL_DOCUMENTATION.md): Detailed technical information about the system
- [Scripts](scripts.md): Documentation for utility scripts
- [Streamlit App](streamlit_app.md): Documentation for the web-based user interface

## Getting Started

To get started with the IR Image Classification system:

1. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Set up the configuration:

   - Copy `.env.example` to `.env` and adjust the settings
   - Review and modify configuration files in the `config` directory

> Actually, every param in the config can be overwritten by command line arguments. 

3. Run the Streamlit application:

   ```bash
   streamlit run app.py
   ```

4. Access the application at `http://localhost:8501`

## Development

For development purposes:

1. Set up a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. Install development dependencies:

   ```bash
   pip install -e ".[dev]"
   ```

3. Run tests:
   ```bash
   pytest
   ```
