# IR Image Classification System

A high-precision, locally-deployed solution for military object identification using deep learning embeddings on infrared imagery.

## Features

- **High Precision**: 95%+ accuracy on IR object classification
- **Fast Query**: Sub-2 second response times for similarity searches
- **Local Deployment**: Uses local vector database for storage
- **Scalable**: Handles datasets from 3k to 18k+ images through augmentation

## Architecture

The system processes infrared imagery through fine-tuned neural networks (ResNet50 or Qwen VLM), stores embeddings in a local vector database (ChromaDB), and performs fast similarity searches to identify objects.

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (optional, but recommended)

### Setup

1. Clone the repository:

```bash
git clone <repository-url>
cd ir-image-classification
```

2. Create virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
# Or using pip with pyproject.toml
pip install -e .
```

4. Install development dependencies:

```bash
pip install -e ".[dev]"
```

5. Set up pre-commit hooks:

```bash
pre-commit install
```

## Project Structure

```
ir-image-classification/
├── src/
│   ├── data/           # Data processing and augmentation
│   ├── database/       # Vector database operations
│   ├── embedding/      # Embedding extraction and model inference
│   ├── models/         # Data models and interfaces
│   ├── query/          # Query processing and similarity search
│   ├── training/       # Model training and fine-tuning
│   └── ui/             # User interface components
├── tests/              # Test suite
├── scripts/            # Utility scripts
├── config/             # Configuration management
├── data/               # Data storage (created at runtime)
├── models/             # Trained model storage (created at runtime)
└── logs/               # Application logs (created at runtime)
```

## Usage

### Training a Model

```bash
python scripts/train_model.py --config config/training_config.yaml
```

### Running the Web Interface

```bash
# Install Streamlit dependencies
pip install -r requirements_streamlit.txt

# Run the Streamlit app
streamlit run app.py
```

The web interface allows you to:

- Upload your own IR images or select from examples
- Configure classification parameters
- View results with confidence scores and object identification
- Save results to JSON files

### Command Line Query

```bash
python scripts/query_image.py --image path/to/ir_image.png
```

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test categories
pytest -m unit
pytest -m integration
```

### Code Quality

```bash
# Format code
black src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/
```

## Configuration

The system uses configuration files in the `config/` directory:

- `settings.py`: Main configuration settings
- Environment variables can override settings using `.env` file

## License

MIT License

## Contributing

[Add contribution guidelines if applicable]
