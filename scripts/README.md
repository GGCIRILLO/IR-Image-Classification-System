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
python scripts/populate_database.py --database-path data/chroma_db_final \
  --processed-dir data/processed

# Limited sampling for testing
python scripts/populate_database.py --database-path data/chroma_db_test \
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

## Testing Scripts

### `test_query_processor.py`

Tests the query processor functionality.

**Usage:**

```bash
python scripts/test_query_processor.py
```

### `test_ranking_confidence.py`

Tests ranking and confidence calculation components.

**Usage:**

```bash
python scripts/test_ranking_confidence.py
```

### `test_military_pipeline.py`

Tests military-specific features and workflows.

**Usage:**

```bash
python scripts/test_military_pipeline.py
```

### `test_different_queries.py`

Tests system performance with various query types.

**Usage:**

```bash
python scripts/test_different_queries.py
```

### `test_improvements.py`

Tests the effectiveness of performance improvements.

**Usage:**

```bash
python scripts/test_improvements.py
```

### `diagnose_score_consistency.py`

Diagnoses issues with similarity and confidence score consistency.

**Usage:**

```bash
python scripts/diagnose_score_consistency.py --database data/chroma_db_final
```
