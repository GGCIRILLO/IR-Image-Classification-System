"""Configuration settings for IR Image Classification System."""

import os
from pathlib import Path
from typing import Dict, Any

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Database settings
DATABASE_CONFIG = {
    "chroma_db_path": PROJECT_ROOT / "chroma_db",
    "collection_name": "ir_embeddings",
    "distance_metric": "cosine",
}

# Model settings
MODEL_CONFIG = {
    "embedding_dim": 512,
    "batch_size": 32,
    "max_image_size": (224, 224),
    "supported_formats": [".png", ".jpg", ".jpeg", ".tiff", ".bmp"],
}

# Training settings
TRAINING_CONFIG = {
    "learning_rate": 1e-4,
    "epochs": 100,
    "validation_split": 0.15,
    "test_split": 0.15,
    "augmentation_factor": 6,  
    "target_accuracy": 0.95,
}

# Query settings
QUERY_CONFIG = {
    "max_query_time": 2.0,  # seconds
    "top_k_results": 5,
    "confidence_threshold": 0.7,
}

def get_config() -> Dict[str, Any]:
    """Get complete configuration dictionary."""
    return {
        "database": DATABASE_CONFIG,
        "model": MODEL_CONFIG,
        "training": TRAINING_CONFIG,
        "query": QUERY_CONFIG,
        "paths": {
            "data": DATA_DIR,
            "raw_data": RAW_DATA_DIR,
            "processed_data": PROCESSED_DATA_DIR,
            "models": MODELS_DIR,
            "logs": LOGS_DIR,
        }
    }