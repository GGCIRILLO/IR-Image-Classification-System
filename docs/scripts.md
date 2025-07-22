# Scripts Directory

This directory contains operational and performance improvement scripts for the IR Image Classification System.

## Operational Scripts

### `run_mission.py`

The main command-line interface for executing IR image classification missions end-to-end.

**Features:**

- Multiple configuration presets (military, development, production, testing)
- Flexible ranking strategies (similarity_only, confidence_weighted, hybrid_score, military_priority)
- Advanced confidence scoring (similarity_based, statistical, ensemble, military_calibrated)
- Multiple output formats (table, json, detailed, military report)
- Performance monitoring and validation

**Usage:**

```bash
# Basic query
python scripts/run_mission.py --image query.png --database data/chroma_db_final

# Military deployment
python scripts/run_mission.py --image target.png --database data/chroma_db_final \
  --preset military --strategy military_priority \
  --confidence-strategy military_calibrated --format military
```

### `populate_database.py`

Populates the vector database with embeddings extracted from processed IR images.

**Features:**

- Configurable sampling strategies (max per class, max total)
- Embedding extraction with quality validation
- Database integrity verification
- Performance monitoring

**Usage:**

```bash
# Basic database population
python scripts/populate_database.py --database-path data/vector_db \
  --processed-dir data/processed

# Limited sampling for testing
python scripts/populate_database.py --database-path data/vector_db_test \
  --processed-dir data/processed --max-per-class 3 --max-total 30
```

### `init_database.py`

Initializes a new vector database with the required structure and settings.

**Usage:**

```bash
python scripts/init_database.py --database-path data/new_vector_db
```

## Performance Improvement Scripts

### `quick_improvements.py`

Provides immediate performance improvements without retraining models.

**Features:**

- Enhanced embedding extraction with IR-specific normalization
- Improved similarity score calculation for thermal images
- Reduced confidence thresholds for better recall
- Optimized search parameters

**Usage:**

```bash
python scripts/quick_improvements.py --database data/vector_db \
  --test-image examples/test_image.png
```

### `enhanced_fine_tuning.py`

Offers comprehensive fine-tuning with advanced techniques.

**Features:**

- Contrastive learning with hard negative mining
- Advanced loss functions (Triplet + Center Loss)
- IR-specific data augmentation
- Confidence calibration
- Automatic hyperparameter optimization

**Usage:**

```bash
python scripts/enhanced_fine_tuning.py --train-data data/processed \
  --database data/chroma_db_final --model-type resnet50 --epochs 50
```

### `fix_similarity_confidence.py`

Provides targeted fixes for similarity and confidence issues.

**Features:**

- IR-specific similarity boosting
- Confidence score recalibration
- Distance metric optimization
- Threshold adjustments

**Usage:**

```bash
python scripts/fix_similarity_confidence.py --database data/chroma_db_final \
  --apply-all --test-image examples/test_image.png
```

### `train_improved_model.py`

Advanced training pipeline for high-accuracy models.

**Features:**

- Weighted sampling for imbalanced classes
- One-cycle learning rate scheduling
- Advanced augmentation for IR images
- Automatic early stopping at 95% accuracy

**Usage:**

```bash
python scripts/train_improved_model.py --data-dir data/processed \
  --model-type resnet50 --epochs 100
```

## Augmentation Scripts

### `augmentation.py`

Initializes a set of image augmentation transformations to enhance dataset variability during training.
The augmentations applied include:

- RandomRotation: Rotates the image by a random angle within ±25 degrees.
- RandomAffine: Applies random affine transformations with up to 10% translation and scaling between 0.9 and 1.1.
- RandomPerspective: Randomly distorts the perspective of the image with a distortion scale of 0.3 and a 50% chance of application.
- RandomApply (GaussianBlur): Applies Gaussian blur with a kernel size of 3 to the image with a 50% probability.
- RandomApply (RandomErasing): Randomly erases a rectangular region of the image with a 50% probability, where the erased area covers 2% to 20% of the image and has an aspect ratio between 0.3 and 3.3, filled with zeros.
- RandomAdjustSharpness: Randomly adjusts the sharpness of the image by a factor of 2 with a 30% probability.

These augmentations are composed sequentially to increase the diversity of training samples and improve model generalization.
