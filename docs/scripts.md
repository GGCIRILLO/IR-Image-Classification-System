# Scripts Documentation

The `scripts` directory contains utility scripts for various tasks related to the IR image classification system. These scripts provide command-line interfaces for common operations such as database population, model fine-tuning, and system testing.

## Script Files

### `populate_database.py`

This script populates the vector database with embeddings extracted from IR images.

**Functionality:**

- Scans directories for IR images
- Processes images using the IR image processor
- Extracts embeddings using the embedding extractor
- Stores embeddings in the vector database
- Supports batch processing for efficiency
- Provides progress reporting and error handling

**Usage:**

```bash
python scripts/populate_database.py --data-dir /path/to/images --db-path /path/to/database --batch-size 32
```

### `enhanced_fine_tuning.py`

This script performs enhanced fine-tuning of models for IR image classification.

**Functionality:**

- Loads a pre-trained model
- Prepares training, validation, and test datasets
- Implements advanced fine-tuning techniques
- Supports contrastive learning and triplet loss
- Provides checkpointing and early stopping
- Generates performance reports and visualizations

**Usage:**

```bash
python scripts/enhanced_fine_tuning.py --model-path /path/to/model --data-dir /path/to/data --epochs 50 --learning-rate 1e-4
```

### `fix_similarity_confidence.py`

This script fixes and calibrates similarity confidence scores in the system.

**Functionality:**

- Analyzes existing similarity results
- Identifies issues with confidence calculation
- Applies fixes to the confidence calculation algorithm
- Calibrates confidence scores using ground truth data
- Updates the confidence configuration
- Validates the improvements

**Usage:**

```bash
python scripts/fix_similarity_confidence.py --db-path /path/to/database --validation-data /path/to/validation/data
```

### `run_mission.py`

This script simulates a mission scenario for testing the IR image classification system.

**Functionality:**

- Loads mission parameters and target information
- Processes a sequence of IR images
- Performs real-time classification and identification
- Evaluates system performance under mission conditions
- Generates mission reports and analytics
- Supports different mission profiles and scenarios

**Usage:**

```bash
python scripts/run_mission.py --mission-config /path/to/config --output-dir /path/to/output
```

### `augmentation.py`

Initializes a set of image augmentation transformations to enhance dataset variability during training.
The augmentations applied include:

- RandomRotation: Rotates the image by a random angle within ±25 degrees.
- RandomAffine: Applies random affine transformations with up to 10% translation and scaling between 0.9 and 1.1.
- RandomPerspective: Randomly distorts the perspective of the image with a distortion scale of 0.3 and a 50% chance of application.
- RandomApply (GaussianBlur): Applies Gaussian blur with a kernel size of 3 to the image with a 50% probability.
- RandomApply (RandomErasing): Randomly erases a rectangular region of the image with a 50% probability, where the erased area covers 2% to 20% of the image and has an aspect ratio between 0.3 and 3.3, filled with zeros.
- RandomAdjustSharpness: Randomly adjusts the sharpness of the image by a factor of 2 with a 30% probability.

These scripts provide convenient command-line interfaces for common operations, making it easier to work with the IR image classification system without having to write custom code for routine tasks.
