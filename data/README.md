# Data Directory

This directory contains data storage for the IR Image Classification System.

## Structure

### `processed/`

Contains processed infrared (IR) images organized by object class. These images have been:

- Resized to 224x224 pixels
- Converted to grayscale
- Normalized for IR-specific characteristics
- Organized into folders by object class

### `processed.zip`

Compressed archive of the processed image dataset, useful for distribution or backup.

### `vector_db/`

Contains the ChromaDB vector database files with:

- Embedding vectors extracted from processed images
- Metadata for each embedding (object class, image ID, etc.)
- Index structures for efficient similarity search

## Data Format

The IR images follow specific format requirements:

- 224x224 pixel resolution
- Grayscale (single channel)
- Normalized pixel values (0.0-1.0)
- IR-specific format with white/bright objects on black/dark background

## Usage

The data directory is used by:

1. **Database Population Scripts**:

   ```bash
   python scripts/populate_database.py --database-path data/vector_db --processed-dir data/processed
   ```

2. **Query Processing**:
   ```bash
   python scripts/run_mission.py --image query.png --database data/vector_db
   ```

## Adding New Data

To add new data to the system:

1. Place new IR images in the appropriate class folder under `data/processed/`
2. Run the database population script to extract embeddings and update the vector database:
   ```bash
   python scripts/populate_database.py --database-path data/vector_db --processed-dir data/processed
   ```

## Data Augmentation

The system supports data augmentation to increase the dataset size. Augmented images are generated during training or database population with techniques like:

- Rotation
- Flipping
- Brightness/contrast adjustments
- Noise addition
- Slight perspective changes

These augmentations help improve model robustness for IR image classification.
