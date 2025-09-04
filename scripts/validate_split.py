#!/usr/bin/env python3
"""
Dataset Split Validation Script

Validates the dataset split by checking the test set structure and metadata.
"""

import json
from pathlib import Path
from typing import Dict
import argparse


def get_image_count(directory: Path) -> int:
    """Count image files in a directory."""
    if not directory.exists():
        return 0
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    return len([
        f for f in directory.iterdir() 
        if f.is_file() and f.suffix.lower() in image_extensions
    ])


def validate_split(processed_dir: Path, test_dir: Path, metadata_file: Path):
    """Validate the dataset split against metadata."""
    
    # Load metadata
    if not metadata_file.exists():
        print(f"Error: Metadata file '{metadata_file}' not found")
        return False
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    print("Validating dataset split...")
    print(f"Expected test images: {metadata['total_test_images']}")
    print(f"Expected training images: {metadata['total_training_images']}")
    
    validation_errors = []
    actual_test_total = 0
    actual_training_total = 0
    
    for class_name, class_data in metadata['classes'].items():
        # Check test set
        test_class_dir = test_dir / class_name
        actual_test_count = get_image_count(test_class_dir)
        expected_test_count = class_data['test_images']
        
        if actual_test_count != expected_test_count:
            validation_errors.append(
                f"Test set mismatch for {class_name}: "
                f"expected {expected_test_count}, found {actual_test_count}"
            )
        
        # Check training set (remaining in processed)
        training_class_dir = processed_dir / class_name
        actual_training_count = get_image_count(training_class_dir)
        expected_training_count = class_data['training_images']
        
        if actual_training_count != expected_training_count:
            validation_errors.append(
                f"Training set mismatch for {class_name}: "
                f"expected {expected_training_count}, found {actual_training_count}"
            )
        
        actual_test_total += actual_test_count
        actual_training_total += actual_training_count
    
    # Check totals
    if actual_test_total != metadata['total_test_images']:
        validation_errors.append(
            f"Total test images mismatch: "
            f"expected {metadata['total_test_images']}, found {actual_test_total}"
        )
    
    if actual_training_total != metadata['total_training_images']:
        validation_errors.append(
            f"Total training images mismatch: "
            f"expected {metadata['total_training_images']}, found {actual_training_total}"
        )
    
    # Report results
    if validation_errors:
        print("\nValidation FAILED:")
        for error in validation_errors:
            print(f"  ❌ {error}")
        return False
    else:
        print("\n✅ Validation PASSED")
        print(f"Test set: {actual_test_total} images across {len(metadata['classes'])} classes")
        print(f"Training set: {actual_training_total} images")
        print(f"Split ratio: {metadata['split_ratio']['test_percentage']:.1f}% test / "
              f"{metadata['split_ratio']['training_percentage']:.1f}% training")
        return True


def main():
    parser = argparse.ArgumentParser(description='Validate dataset split')
    parser.add_argument('--processed-dir', type=str, default='data/processed',
                       help='Path to processed data directory')
    parser.add_argument('--test-dir', type=str, default='data/test_set',
                       help='Path to test set directory')
    parser.add_argument('--metadata-file', type=str, default='data/test_set_metadata.json',
                       help='Path to metadata file')
    
    args = parser.parse_args()
    
    processed_dir = Path(args.processed_dir)
    test_dir = Path(args.test_dir)
    metadata_file = Path(args.metadata_file)
    
    success = validate_split(processed_dir, test_dir, metadata_file)
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())