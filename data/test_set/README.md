# Test Set Documentation

## Overview

This directory contains the test set created by randomly selecting 5 images per class from the processed dataset.

## Split Details

- **Total Classes**: 83
- **Images per Class**: 5 (target)
- **Total Test Images**: 415
- **Total Training Images**: 8,158 (remaining in `data/processed/`)
- **Split Ratio**: 4.8% test / 95.2% training
- **Split Date**: 2025-09-04T10:18:28.363016
- **Random Seed**: 42 (for reproducibility)

## Structure

The test set maintains the same folder structure as the original processed data:

```
data/test_set/
├── AAV Tank/
├── Airfield/
├── Ariane-LP/
├── ...
└── ZSU-23-4 Anti Aircraft Artillery Tank/
```

Each class folder contains exactly 5 randomly selected images from the original class.

## Metadata

Complete metadata about the split is available in `data/test_set_metadata.json`, including:

- List of specific files moved to test set for each class
- Original image counts per class
- Individual class split ratios
- Timestamp of the split operation

## Usage

This test set should be used for:

- Final model evaluation
- Performance benchmarking
- Validation of model generalization

**Important**: Do not use these images for training or hyperparameter tuning to avoid data leakage.

## Validation

The split has been validated using `scripts/validate_split.py` to ensure:

- Correct number of images per class
- Proper file movement
- Metadata accuracy
- No duplicate or missing files
