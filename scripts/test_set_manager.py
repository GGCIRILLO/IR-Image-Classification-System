#!/usr/bin/env python3
"""
Test Set Management Utilities

Provides comprehensive management of the test set including tracking,
rollback functionality, and validation to prevent data leakage.
"""

import os
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Tuple, Optional
import argparse


class TestSetManager:
    """Manages test set operations with tracking and validation."""
    
    def __init__(self, processed_dir: str = "data/processed", 
                 test_dir: str = "data/test_set",
                 metadata_file: str = "data/test_set_metadata.json"):
        self.processed_dir = Path(processed_dir)
        self.test_dir = Path(test_dir)
        self.metadata_file = Path(metadata_file)
        
    def load_metadata(self) -> Optional[Dict]:
        """Load test set metadata if it exists."""
        if not self.metadata_file.exists():
            return None
        
        try:
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading metadata: {e}")
            return None
    
    def get_test_image_paths(self) -> Set[str]:
        """Get set of all test image file paths for validation."""
        metadata = self.load_metadata()
        if not metadata:
            return set()
        
        test_paths = set()
        for class_name, class_data in metadata['classes'].items():
            for filename in class_data['test_files']:
                test_paths.add(str(self.test_dir / class_name / filename))
        
        return test_paths
    
    def get_training_image_paths(self) -> Set[str]:
        """Get set of all training image file paths."""
        training_paths = set()
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        for class_dir in self.processed_dir.iterdir():
            if class_dir.is_dir():
                for img_file in class_dir.iterdir():
                    if img_file.is_file() and img_file.suffix.lower() in image_extensions:
                        training_paths.add(str(img_file))
        
        return training_paths
    
    def validate_no_overlap(self) -> Tuple[bool, List[str]]:
        """
        Validate that no test images are present in training set.
        
        Returns:
            Tuple of (is_valid, list_of_overlapping_files)
        """
        test_paths = self.get_test_image_paths()
        training_paths = self.get_training_image_paths()
        
        # Check for filename overlaps (not full paths since they're in different dirs)
        test_filenames = {Path(p).name for p in test_paths}
        training_filenames = {Path(p).name for p in training_paths}
        
        overlapping_files = list(test_filenames.intersection(training_filenames))
        
        return len(overlapping_files) == 0, overlapping_files
    
    def create_backup_metadata(self) -> Path:
        """Create a backup of current metadata before operations."""
        if not self.metadata_file.exists():
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.metadata_file.with_suffix(f'.backup_{timestamp}.json')
        shutil.copy2(self.metadata_file, backup_path)
        
        print(f"Metadata backed up to: {backup_path}")
        return backup_path
    
    def rollback_test_set(self, confirm: bool = False) -> bool:
        """
        Rollback test set by moving all test images back to training set.
        
        Args:
            confirm: If True, skip confirmation prompt
            
        Returns:
            True if rollback successful, False otherwise
        """
        metadata = self.load_metadata()
        if not metadata:
            print("No metadata found. Cannot perform rollback.")
            return False
        
        if not confirm:
            print(f"This will move {metadata['total_test_images']} images back to training set.")
            print(f"Test set directory '{self.test_dir}' will be removed.")
            response = input("Proceed with rollback? (y/N): ")
            if response.lower() != 'y':
                print("Rollback cancelled.")
                return False
        
        # Create backup before rollback
        backup_path = self.create_backup_metadata()
        
        try:
            moved_count = 0
            errors = []
            
            for class_name, class_data in metadata['classes'].items():
                test_class_dir = self.test_dir / class_name
                training_class_dir = self.processed_dir / class_name
                
                # Ensure training class directory exists
                training_class_dir.mkdir(parents=True, exist_ok=True)
                
                # Move each test file back to training
                for filename in class_data['test_files']:
                    test_file = test_class_dir / filename
                    training_file = training_class_dir / filename
                    
                    if test_file.exists():
                        try:
                            shutil.move(str(test_file), str(training_file))
                            moved_count += 1
                        except Exception as e:
                            errors.append(f"Failed to move {test_file}: {e}")
                    else:
                        errors.append(f"Test file not found: {test_file}")
            
            # Remove empty test directories
            if self.test_dir.exists():
                try:
                    shutil.rmtree(self.test_dir)
                    print(f"Removed test set directory: {self.test_dir}")
                except Exception as e:
                    errors.append(f"Failed to remove test directory: {e}")
            
            # Remove metadata file
            if self.metadata_file.exists():
                try:
                    self.metadata_file.unlink()
                    print(f"Removed metadata file: {self.metadata_file}")
                except Exception as e:
                    errors.append(f"Failed to remove metadata file: {e}")
            
            # Report results
            if errors:
                print(f"\nRollback completed with {len(errors)} errors:")
                for error in errors:
                    print(f"  ❌ {error}")
                return False
            else:
                print(f"\n✅ Rollback successful!")
                print(f"Moved {moved_count} images back to training set")
                if backup_path:
                    print(f"Original metadata backed up to: {backup_path}")
                return True
                
        except Exception as e:
            print(f"Rollback failed: {e}")
            return False
    
    def get_status(self) -> Dict:
        """Get current status of test set and training set."""
        metadata = self.load_metadata()
        
        status = {
            'test_set_exists': self.test_dir.exists(),
            'metadata_exists': self.metadata_file.exists(),
            'test_images': 0,
            'training_images': 0,
            'classes': 0,
            'validation_passed': False,
            'overlapping_files': []
        }
        
        if metadata:
            status['test_images'] = metadata['total_test_images']
            status['training_images'] = metadata['total_training_images']
            status['classes'] = len(metadata['classes'])
            status['split_date'] = metadata.get('split_date', 'Unknown')
        
        # Count actual files
        if self.test_dir.exists():
            actual_test_count = len(list(self.test_dir.rglob('*.png'))) + \
                              len(list(self.test_dir.rglob('*.jpg'))) + \
                              len(list(self.test_dir.rglob('*.jpeg')))
            status['actual_test_images'] = actual_test_count
        
        if self.processed_dir.exists():
            actual_training_count = len(list(self.processed_dir.rglob('*.png'))) + \
                                  len(list(self.processed_dir.rglob('*.jpg'))) + \
                                  len(list(self.processed_dir.rglob('*.jpeg')))
            status['actual_training_images'] = actual_training_count
        
        # Validate no overlap
        is_valid, overlapping = self.validate_no_overlap()
        status['validation_passed'] = is_valid
        status['overlapping_files'] = overlapping
        
        return status
    
    def print_status(self):
        """Print formatted status information."""
        status = self.get_status()
        
        print("="*60)
        print("TEST SET STATUS")
        print("="*60)
        
        print(f"Test set exists: {'✅' if status['test_set_exists'] else '❌'}")
        print(f"Metadata exists: {'✅' if status['metadata_exists'] else '❌'}")
        
        if status['metadata_exists']:
            print(f"Split date: {status.get('split_date', 'Unknown')}")
            print(f"Classes: {status['classes']}")
            print(f"Test images (metadata): {status['test_images']}")
            print(f"Training images (metadata): {status['training_images']}")
        
        if 'actual_test_images' in status:
            print(f"Test images (actual): {status['actual_test_images']}")
        
        if 'actual_training_images' in status:
            print(f"Training images (actual): {status['actual_training_images']}")
        
        print(f"Validation passed: {'✅' if status['validation_passed'] else '❌'}")
        
        if status['overlapping_files']:
            print(f"\n⚠️  Found {len(status['overlapping_files'])} overlapping files:")
            for filename in status['overlapping_files'][:10]:  # Show first 10
                print(f"  - {filename}")
            if len(status['overlapping_files']) > 10:
                print(f"  ... and {len(status['overlapping_files']) - 10} more")
    
    def track_image_sources(self) -> Dict[str, Dict]:
        """
        Create detailed tracking of image sources and locations.
        
        Returns:
            Dictionary mapping image filenames to their source information
        """
        metadata = self.load_metadata()
        if not metadata:
            return {}
        
        tracking = {}
        
        for class_name, class_data in metadata['classes'].items():
            for filename in class_data['test_files']:
                tracking[filename] = {
                    'class': class_name,
                    'original_location': str(self.processed_dir / class_name / filename),
                    'current_location': str(self.test_dir / class_name / filename),
                    'moved_date': metadata['split_date'],
                    'in_test_set': True
                }
        
        return tracking
    
    def export_tracking_report(self, output_file: str = "data/test_set_tracking.json"):
        """Export detailed tracking report to file."""
        tracking = self.track_image_sources()
        status = self.get_status()
        
        report = {
            'generated_date': datetime.now().isoformat(),
            'status': status,
            'image_tracking': tracking,
            'summary': {
                'total_tracked_images': len(tracking),
                'validation_status': 'PASSED' if status['validation_passed'] else 'FAILED'
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Tracking report exported to: {output_file}")
        return output_file


def main():
    parser = argparse.ArgumentParser(description='Test Set Management Utilities')
    parser.add_argument('command', choices=['status', 'validate', 'rollback', 'track', 'export'],
                       help='Command to execute')
    parser.add_argument('--processed-dir', type=str, default='data/processed',
                       help='Path to processed data directory')
    parser.add_argument('--test-dir', type=str, default='data/test_set',
                       help='Path to test set directory')
    parser.add_argument('--metadata-file', type=str, default='data/test_set_metadata.json',
                       help='Path to metadata file')
    parser.add_argument('--output-file', type=str, default='data/test_set_tracking.json',
                       help='Output file for tracking report')
    parser.add_argument('--confirm', action='store_true',
                       help='Skip confirmation prompts')
    
    args = parser.parse_args()
    
    manager = TestSetManager(args.processed_dir, args.test_dir, args.metadata_file)
    
    if args.command == 'status':
        manager.print_status()
        
    elif args.command == 'validate':
        is_valid, overlapping = manager.validate_no_overlap()
        if is_valid:
            print("✅ Validation PASSED: No overlap between test and training sets")
        else:
            print(f"❌ Validation FAILED: Found {len(overlapping)} overlapping files")
            for filename in overlapping[:10]:
                print(f"  - {filename}")
            if len(overlapping) > 10:
                print(f"  ... and {len(overlapping) - 10} more")
        return 0 if is_valid else 1
        
    elif args.command == 'rollback':
        success = manager.rollback_test_set(args.confirm)
        return 0 if success else 1
        
    elif args.command == 'track':
        tracking = manager.track_image_sources()
        print(f"Tracking {len(tracking)} test images:")
        for filename, info in list(tracking.items())[:5]:
            print(f"  {filename} -> {info['class']}")
        if len(tracking) > 5:
            print(f"  ... and {len(tracking) - 5} more")
            
    elif args.command == 'export':
        manager.export_tracking_report(args.output_file)
    
    return 0


if __name__ == "__main__":
    exit(main())