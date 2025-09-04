#!/usr/bin/env python3
"""
Dataset Splitting Utility

This script randomly selects 5 images per class from the processed data folders
and moves them to a test set directory while maintaining folder structure.
Generates metadata documenting the split for reproducibility.
"""

import os
import random
import shutil
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import argparse


def get_image_files(directory: Path) -> List[Path]:
    """Get all image files from a directory."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    return [
        f for f in directory.iterdir() 
        if f.is_file() and f.suffix.lower() in image_extensions
    ]


def create_test_set_structure(processed_dir: Path, test_dir: Path, images_per_class: int = 5) -> Dict:
    """
    Create test set by randomly selecting images from each class.
    
    Args:
        processed_dir: Path to processed data directory
        test_dir: Path to test set directory
        images_per_class: Number of images to select per class
        
    Returns:
        Dictionary containing metadata about the split
    """
    metadata = {
        'split_date': datetime.now().isoformat(),
        'images_per_class': images_per_class,
        'classes': {},
        'total_test_images': 0,
        'total_training_images': 0,
        'split_ratio': {}
    }
    
    # Create test directory if it doesn't exist
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all class directories
    class_dirs = [d for d in processed_dir.iterdir() if d.is_dir()]
    
    print(f"Found {len(class_dirs)} classes")
    print(f"Selecting {images_per_class} images per class for test set...")
    
    for class_dir in class_dirs:
        class_name = class_dir.name
        print(f"Processing class: {class_name}")
        
        # Get all images in this class
        images = get_image_files(class_dir)
        
        if len(images) < images_per_class:
            print(f"  Warning: {class_name} has only {len(images)} images, selecting all")
            selected_images = images
        else:
            # Randomly select images for test set
            selected_images = random.sample(images, images_per_class)
        
        # Create class directory in test set
        test_class_dir = test_dir / class_name
        test_class_dir.mkdir(exist_ok=True)
        
        # Move selected images to test set
        moved_files = []
        for img in selected_images:
            dest_path = test_class_dir / img.name
            shutil.move(str(img), str(dest_path))
            moved_files.append(img.name)
        
        # Update metadata
        remaining_images = len(images) - len(selected_images)
        test_images = len(selected_images)
        
        metadata['classes'][class_name] = {
            'total_original': len(images),
            'test_images': test_images,
            'training_images': remaining_images,
            'test_files': moved_files,
            'test_ratio': test_images / len(images) if len(images) > 0 else 0
        }
        
        metadata['total_test_images'] += test_images
        metadata['total_training_images'] += remaining_images
        
        print(f"  Moved {test_images} images to test set, {remaining_images} remain for training")
    
    # Calculate overall split ratio
    total_images = metadata['total_test_images'] + metadata['total_training_images']
    metadata['split_ratio'] = {
        'test_percentage': (metadata['total_test_images'] / total_images * 100) if total_images > 0 else 0,
        'training_percentage': (metadata['total_training_images'] / total_images * 100) if total_images > 0 else 0
    }
    
    return metadata


def save_metadata(metadata: Dict, output_path: Path):
    """Save metadata to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)


def print_summary(metadata: Dict):
    """Print summary of the dataset split."""
    print("\n" + "="*60)
    print("DATASET SPLIT SUMMARY")
    print("="*60)
    print(f"Total classes: {len(metadata['classes'])}")
    print(f"Images per class (target): {metadata['images_per_class']}")
    print(f"Total test images: {metadata['total_test_images']}")
    print(f"Total training images: {metadata['total_training_images']}")
    print(f"Test set percentage: {metadata['split_ratio']['test_percentage']:.1f}%")
    print(f"Training set percentage: {metadata['split_ratio']['training_percentage']:.1f}%")
    print(f"Split date: {metadata['split_date']}")
    
    # Show classes with fewer than target images
    insufficient_classes = [
        (name, data['total_original']) 
        for name, data in metadata['classes'].items() 
        if data['total_original'] < metadata['images_per_class']
    ]
    
    if insufficient_classes:
        print(f"\nClasses with fewer than {metadata['images_per_class']} images:")
        for class_name, count in insufficient_classes:
            print(f"  {class_name}: {count} images")


def main():
    parser = argparse.ArgumentParser(description='Split dataset into training and test sets')
    parser.add_argument('--processed-dir', type=str, default='data/processed',
                       help='Path to processed data directory')
    parser.add_argument('--test-dir', type=str, default='data/test_set',
                       help='Path to test set directory')
    parser.add_argument('--images-per-class', type=int, default=5,
                       help='Number of images per class for test set')
    parser.add_argument('--metadata-file', type=str, default='data/test_set_metadata.json',
                       help='Path to save metadata file')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    # Convert to Path objects
    processed_dir = Path(args.processed_dir)
    test_dir = Path(args.test_dir)
    metadata_file = Path(args.metadata_file)
    
    # Validate input directory
    if not processed_dir.exists():
        print(f"Error: Processed data directory '{processed_dir}' does not exist")
        return 1
    
    # Confirm operation
    print(f"This will move {args.images_per_class} images per class from:")
    print(f"  Source: {processed_dir}")
    print(f"  Destination: {test_dir}")
    print(f"  Metadata: {metadata_file}")
    
    response = input("\nProceed with dataset split? (y/N): ")
    if response.lower() != 'y':
        print("Operation cancelled")
        return 0
    
    # Perform the split
    try:
        metadata = create_test_set_structure(processed_dir, test_dir, args.images_per_class)
        save_metadata(metadata, metadata_file)
        print_summary(metadata)
        
        print(f"\nDataset split completed successfully!")
        print(f"Metadata saved to: {metadata_file}")
        
    except Exception as e:
        print(f"Error during dataset split: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())