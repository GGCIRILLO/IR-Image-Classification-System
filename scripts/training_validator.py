#!/usr/bin/env python3
"""
Training Data Validation Utilities

Ensures test set images are not accidentally included in training processes.
"""

import os
import json
from pathlib import Path
from typing import List, Set, Dict, Tuple, Optional
from datetime import datetime
import argparse


class TrainingValidator:
    """Validates training data to prevent test set leakage."""
    
    def __init__(self, test_metadata_file: str = "data/test_set_metadata.json"):
        self.test_metadata_file = Path(test_metadata_file)
        self._test_filenames = None
    
    def _load_test_filenames(self) -> Set[str]:
        """Load test image filenames from metadata."""
        if self._test_filenames is not None:
            return self._test_filenames
        
        if not self.test_metadata_file.exists():
            print(f"Warning: Test metadata file not found: {self.test_metadata_file}")
            self._test_filenames = set()
            return self._test_filenames
        
        try:
            with open(self.test_metadata_file, 'r') as f:
                metadata = json.load(f)
            
            test_filenames = set()
            for class_data in metadata['classes'].values():
                test_filenames.update(class_data['test_files'])
            
            self._test_filenames = test_filenames
            return self._test_filenames
            
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error loading test metadata: {e}")
            self._test_filenames = set()
            return self._test_filenames
    
    def validate_file_list(self, file_paths: List[str]) -> Tuple[bool, List[str]]:
        """Validate a list of file paths to ensure no test images are included."""
        test_filenames = self._load_test_filenames()
        violations = []
        
        for file_path in file_paths:
            filename = Path(file_path).name
            if filename in test_filenames:
                violations.append(file_path)
        
        return len(violations) == 0, violations
    
    def validate_directory(self, directory: str, recursive: bool = True) -> Tuple[bool, List[str]]:
        """Validate all images in a directory."""
        dir_path = Path(directory)
        if not dir_path.exists():
            return True, []
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        if recursive:
            image_files = [
                str(f) for f in dir_path.rglob('*') 
                if f.is_file() and f.suffix.lower() in image_extensions
            ]
        else:
            image_files = [
                str(f) for f in dir_path.iterdir() 
                if f.is_file() and f.suffix.lower() in image_extensions
            ]
        
        return self.validate_file_list(image_files)
    
    def create_training_manifest(self, training_dir: str, output_file: str = "data/training_manifest.json") -> str:
        """Create a manifest of all training images with validation."""
        is_valid, violations = self.validate_directory(training_dir)
        
        dir_path = Path(training_dir)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        training_images = []
        class_counts = {}
        
        for class_dir in dir_path.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name
                class_images = [
                    str(f) for f in class_dir.iterdir()
                    if f.is_file() and f.suffix.lower() in image_extensions
                ]
                
                training_images.extend(class_images)
                class_counts[class_name] = len(class_images)
        
        manifest = {
            'created_date': datetime.now().isoformat(),
            'training_directory': str(dir_path.resolve()),
            'validation_passed': is_valid,
            'total_images': len(training_images),
            'total_classes': len(class_counts),
            'class_counts': class_counts,
            'violations': violations
        }
        
        with open(output_file, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        print(f"Training manifest created: {output_file}")
        print(f"Total images: {len(training_images)}")
        print(f"Validation: {'✅ PASSED' if is_valid else '❌ FAILED'}")
        
        if violations:
            print(f"Found {len(violations)} test images in training data")
        
        return output_file


def main():
    parser = argparse.ArgumentParser(description='Training Data Validation')
    parser.add_argument('command', choices=['validate-dir', 'validate-files', 'manifest'],
                       help='Command to execute')
    parser.add_argument('--directory', type=str, help='Directory to validate')
    parser.add_argument('--files', nargs='+', help='Files to validate')
    parser.add_argument('--output', type=str, help='Output file path')
    parser.add_argument('--metadata', type=str, default='data/test_set_metadata.json',
                       help='Test set metadata file')
    
    args = parser.parse_args()
    
    validator = TrainingValidator(args.metadata)
    
    if args.command == 'validate-dir':
        if not args.directory:
            print("Error: --directory required")
            return 1
        
        is_valid, violations = validator.validate_directory(args.directory)
        
        if is_valid:
            print("✅ Validation PASSED")
        else:
            print(f"❌ Validation FAILED: Found {len(violations)} test images")
            for violation in violations:
                print(f"  - {violation}")
        
        return 0 if is_valid else 1
    
    elif args.command == 'validate-files':
        if not args.files:
            print("Error: --files required")
            return 1
        
        is_valid, violations = validator.validate_file_list(args.files)
        
        if is_valid:
            print("✅ Validation PASSED")
        else:
            print(f"❌ Validation FAILED: Found {len(violations)} test images")
        
        return 0 if is_valid else 1
    
    elif args.command == 'manifest':
        if not args.directory:
            print("Error: --directory required")
            return 1
        
        output_file = args.output or "data/training_manifest.json"
        validator.create_training_manifest(args.directory, output_file)
    
    return 0


if __name__ == "__main__":
    exit(main())