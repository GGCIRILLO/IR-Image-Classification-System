# Test Set Management Utilities

This directory contains comprehensive utilities for managing the test set and preventing training data leakage.

## Scripts Overview

### 1. `test_set_manager.py` - Main Test Set Management

Provides comprehensive management of the test set including tracking, rollback functionality, and validation.

**Commands:**

```bash
# Check current status
python scripts/test_set_manager.py status

# Validate no overlap between test and training sets
python scripts/test_set_manager.py validate

# Rollback test set (move images back to training)
python scripts/test_set_manager.py rollback

# Track image sources and locations
python scripts/test_set_manager.py track

# Export detailed tracking report
python scripts/test_set_manager.py export --output-file data/detailed_tracking.json
```

### 2. `training_validator.py` - Training Data Validation

Ensures test set images are not accidentally included in training processes.

**Commands:**

```bash
# Validate a directory for test image contamination
python scripts/training_validator.py validate-dir --directory data/processed

# Validate specific files
python scripts/training_validator.py validate-files --files image1.png image2.png

# Create training manifest with validation
python scripts/training_validator.py manifest --directory data/processed --output data/training_manifest.json
```

### 3. `split_dataset.py` - Dataset Splitting

Creates the initial test/training split (already executed).

### 4. `validate_split.py` - Split Validation

Validates the dataset split against metadata.

## Key Features

### 🔒 **Data Leakage Prevention**

- Validates training directories to ensure no test images are included
- Provides hooks for training scripts to automatically validate batches
- Creates manifests of safe training files

### 📊 **Comprehensive Tracking**

- Tracks source locations of all moved images
- Maintains detailed metadata about the split
- Exports tracking reports for audit purposes

### 🔄 **Rollback Functionality**

- Complete rollback capability to restore test images to training set
- Automatic backup of metadata before rollback
- Validation of rollback operations

### ✅ **Validation & Monitoring**

- Real-time validation of test/training separation
- Status monitoring of test set integrity
- Automated checks for data consistency

## Usage Examples

### For Training Scripts

```python
# Import validation in your training script
from scripts.training_validator import TrainingValidator

validator = TrainingValidator()

# Validate training directory before starting
is_valid, violations = validator.validate_directory("data/processed")
if not is_valid:
    print(f"Found {len(violations)} test images in training data!")
    exit(1)

# Get safe training files (excluding any test images)
safe_files = validator.get_safe_training_files("data/processed")
```

### For Data Pipeline Validation

```python
from scripts.test_set_manager import TestSetManager

manager = TestSetManager()

# Check system status
status = manager.get_status()
if not status['validation_passed']:
    print("Data leakage detected!")

# Get tracking information
tracking = manager.track_image_sources()
print(f"Tracking {len(tracking)} test images")
```

## File Structure After Split

```
data/
├── processed/                    # Training images (8,158 images)
│   ├── AAV Tank/                # 123 training images
│   ├── Airfield/                # 86 training images
│   └── ...                      # 81 more classes
├── test_set/                    # Test images (415 images)
│   ├── AAV Tank/                # 5 test images
│   ├── Airfield/                # 5 test images
│   └── ...                      # 81 more classes
├── test_set_metadata.json       # Complete split documentation
├── test_set_tracking.json       # Detailed tracking report
└── training_manifest.json       # Validated training file list
```

## Safety Guarantees

1. **No Data Leakage**: Test images are completely separated from training data
2. **Reproducible Split**: Random seed (42) ensures consistent results
3. **Audit Trail**: Complete tracking of all moved images
4. **Rollback Safety**: Metadata backup before any destructive operations
5. **Validation**: Continuous monitoring of data separation integrity

## Integration with Training Pipeline

### Recommended Workflow

1. **Before Training**: Validate training directory with `training_validator.py`
2. **During Training**: Use safe file lists from validation utilities
3. **After Training**: Validate model with test set (separate from training pipeline)
4. **Monitoring**: Regular status checks with `test_set_manager.py status`

### Preventing Accidents

- All training scripts should validate data before processing
- Use provided manifests instead of directory scanning
- Regular validation checks in CI/CD pipelines
- Automated alerts for data integrity issues

## Troubleshooting

### If Test Images Found in Training Data

```bash
# Check what's wrong
python scripts/test_set_manager.py validate

# Get detailed report
python scripts/test_set_manager.py export

# If needed, rollback and re-split
python scripts/test_set_manager.py rollback --confirm
python scripts/split_dataset.py --seed 42
```

### If Rollback Needed

```bash
# Check current status first
python scripts/test_set_manager.py status

# Perform rollback (with confirmation)
python scripts/test_set_manager.py rollback

# Or skip confirmation
python scripts/test_set_manager.py rollback --confirm
```

## Requirements Met

✅ **3.1**: Functions to track moved images and their source locations  
✅ **3.2**: Rollback functionality to restore test images to training set  
✅ **3.3**: Validation to ensure test set images are not accidentally included in training
