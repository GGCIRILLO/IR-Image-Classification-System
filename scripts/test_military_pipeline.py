#!/usr/bin/env python3
"""
Test script for military augmentation pipeline on real IR dataset.

This script loads the pickup and tank images, applies the military-specific
augmentation pipeline, and saves the results for inspection.
"""

import os
import sys
import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple, Dict
import time

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.data.augmentation import DataAugmentationEngine, AugmentationConfig


def load_dataset_images(data_dir: str = "data/raw2.0") -> Dict[str, List[np.ndarray]]:
    """
    Load all images from the dataset organized by class.
    
    Args:
        data_dir: Directory containing class folders
        
    Returns:
        Dict mapping class names to lists of images
    """
    dataset = {}
    data_path = Path(data_dir)
    
    print(f"Loading dataset from: {data_path.absolute()}")
    
    if not data_path.exists():
        print(f"Warning: Dataset directory {data_path} does not exist!")
        print("Falling back to original raw dataset...")
        data_path = Path("data/raw")
        if not data_path.exists():
            print(f"Error: Neither {data_dir} nor data/raw exists!")
            return {}
    
    for class_dir in data_path.iterdir():
        if class_dir.is_dir():
            class_name = class_dir.name
            print(f"\nLoading class: {class_name}")
            
            images = []
            
            # Look for WebP files first (optimized dataset), then BMP files (original dataset)
            webp_files = list(class_dir.glob("*.webp"))
            bmp_files = list(class_dir.glob("*.bmp"))
            
            if webp_files:
                image_files = webp_files
                file_format = "WebP"
                print(f"Found {len(image_files)} WebP files (optimized dataset)")
            elif bmp_files:
                image_files = bmp_files
                file_format = "BMP"
                print(f"Found {len(image_files)} BMP files (original dataset)")
            else:
                print(f"No supported image files found in {class_dir}")
                continue
            
            for img_file in image_files:
                try:
                    # Load image using OpenCV
                    img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
                    
                    if img is None:
                        print(f"Warning: Could not load {img_file}")
                        continue
                    
                    # Convert to float32 and normalize to [0, 1]
                    img = img.astype(np.float32) / 255.0
                    
                    # Handle different input sizes
                    current_height, current_width = img.shape
                    
                    if file_format == "WebP":
                        # WebP images are already 256x256, resize to 224x224 for processing
                        if current_height != 224 or current_width != 224:
                            img_resized = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
                        else:
                            img_resized = img
                    else:
                        # BMP images are large (4096x4096), resize to 224x224
                        img_resized = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
                    
                    images.append(img_resized)
                    
                except Exception as e:
                    print(f"Error loading {img_file}: {e}")
            
            dataset[class_name] = images
            print(f"Successfully loaded {len(images)} images for {class_name} ({file_format} format)")
    
    return dataset


def analyze_ir_characteristics(images: List[np.ndarray], class_name: str) -> Dict:
    """
    Analyze IR characteristics of the loaded images.
    
    Args:
        images: List of IR images
        class_name: Name of the class
        
    Returns:
        Dict with analysis results
    """
    if not images:
        return {}
    
    print(f"\n=== IR Analysis for {class_name} ===")
    
    # Calculate statistics
    all_pixels = np.concatenate([img.flatten() for img in images])
    
    stats = {
        'num_images': len(images),
        'image_shape': images[0].shape,
        'mean_intensity': np.mean(all_pixels),
        'std_intensity': np.std(all_pixels),
        'min_intensity': np.min(all_pixels),
        'max_intensity': np.max(all_pixels),
        'bright_pixel_ratio': np.sum(all_pixels > 0.5) / len(all_pixels),
        'dark_pixel_ratio': np.sum(all_pixels < 0.3) / len(all_pixels)
    }
    
    print(f"Number of images: {stats['num_images']}")
    print(f"Image shape: {stats['image_shape']}")
    print(f"Mean intensity: {stats['mean_intensity']:.3f}")
    print(f"Std intensity: {stats['std_intensity']:.3f}")
    print(f"Intensity range: [{stats['min_intensity']:.3f}, {stats['max_intensity']:.3f}]")
    print(f"Bright pixels (>0.5): {stats['bright_pixel_ratio']:.1%}")
    print(f"Dark pixels (<0.3): {stats['dark_pixel_ratio']:.1%}")
    
    # Check IR characteristics
    is_ir_like = (
        stats['dark_pixel_ratio'] > 0.6 and  # Mostly dark background
        stats['bright_pixel_ratio'] < 0.4 and  # Some bright objects
        stats['mean_intensity'] < 0.4  # Overall dark image
    )
    
    print(f"IR-like characteristics: {'✓' if is_ir_like else '✗'}")
    
    return stats


def test_military_augmentation_pipeline(dataset: Dict[str, List[np.ndarray]]) -> Dict:
    """
    Test the military augmentation pipeline on the dataset.
    
    Args:
        dataset: Dictionary mapping class names to image lists
        
    Returns:
        Dict with augmentation results
    """
    print(f"\n{'='*60}")
    print("TESTING MILITARY AUGMENTATION PIPELINE")
    print(f"{'='*60}")
    
    # Configure augmentation for military IR
    config = AugmentationConfig(
        rotation_range=(-15.0, 15.0),
        scale_range=(0.8, 1.2),
        noise_range=(0.02, 0.08),
        brightness_range=(0.7, 1.3),
        contrast_range=(0.8, 1.2),
        flip_probability=0.2,  # Lower for military context
        preserve_ir_properties=True,
        target_size=(224, 224)
    )
    
    # Initialize augmentation engine
    engine = DataAugmentationEngine(config)
    engine.set_random_seed(42)  # For reproducible results
    
    results = {}
    
    for class_name, images in dataset.items():
        print(f"\n--- Processing {class_name} ---")
        print(f"Base images: {len(images)}")
        
        # Calculate target count (5x expansion)
        target_count = len(images) * 5
        print(f"Target count: {target_count}")
        
        # Apply military augmentation pipeline
        start_time = time.time()
        augmented_images = engine.create_military_augmentation_pipeline(images, target_count)
        end_time = time.time()
        
        print(f"Augmentation completed in {end_time - start_time:.2f} seconds")
        print(f"Generated {len(augmented_images)} total images")
        
        # Analyze augmented results
        augmented_stats = analyze_ir_characteristics(augmented_images, f"{class_name} (Augmented)")
        
        results[class_name] = {
            'original_count': len(images),
            'augmented_count': len(augmented_images),
            'expansion_ratio': len(augmented_images) / len(images),
            'processing_time': end_time - start_time,
            'augmented_images': augmented_images,
            'stats': augmented_stats
        }
    
    return results


def save_sample_augmentations(results: Dict, output_dir: str = "data/processed") -> None:
    """
    Save sample augmented images for visual inspection.
    
    Args:
        results: Results from augmentation pipeline
        output_dir: Directory to save samples
    """
    print(f"\n{'='*60}")
    print("SAVING SAMPLE AUGMENTATIONS")
    print(f"{'='*60}")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for class_name, result in results.items():
        class_dir = output_path / class_name
        class_dir.mkdir(exist_ok=True)
        
        augmented_images = result['augmented_images']
        original_count = result['original_count']
        
        print(f"\nSaving samples for {class_name}:")
        
        # Save all augmented samples
        augmented_samples = augmented_images[original_count:]
        for i, img in enumerate(augmented_samples):
            img_uint8 = (img * 255).astype(np.uint8)
            filename = class_dir / f"augmented_{i+1:02d}.png"
            cv2.imwrite(str(filename), img_uint8)
        
        print(f"  Saved {len(augmented_samples)} augmented samples")


def print_pipeline_summary(results: Dict) -> None:
    """
    Print a summary of the augmentation pipeline results.
    
    Args:
        results: Results from augmentation pipeline
    """
    print(f"\n{'='*60}")
    print("MILITARY AUGMENTATION PIPELINE SUMMARY")
    print(f"{'='*60}")
    
    total_original = sum(r['original_count'] for r in results.values())
    total_augmented = sum(r['augmented_count'] for r in results.values())
    total_time = sum(r['processing_time'] for r in results.values())
    
    print(f"\nOverall Results:")
    print(f"  Total original images: {total_original}")
    print(f"  Total augmented images: {total_augmented}")
    print(f"  Overall expansion ratio: {total_augmented / total_original:.1f}x")
    print(f"  Total processing time: {total_time:.2f} seconds")
    print(f"  Average time per image: {total_time / total_original:.3f} seconds")
    
    print(f"\nPer-Class Results:")
    for class_name, result in results.items():
        print(f"  {class_name}:")
        print(f"    {result['original_count']} → {result['augmented_count']} images")
        print(f"    {result['expansion_ratio']:.1f}x expansion")
        print(f"    {result['processing_time']:.2f}s processing time")


def main():
    """Main function to run the military augmentation pipeline test."""
    print("Military IR Augmentation Pipeline Test")
    print("=" * 60)
    
    try:
        # Load dataset
        dataset = load_dataset_images()
        
        if not dataset:
            print("No images found in dataset!")
            return

        # Analyze original images
        print(f"\n{'='*60}")
        print("ORIGINAL DATASET ANALYSIS")
        print(f"{'='*60}")
        
        for class_name, images in dataset.items():
            analyze_ir_characteristics(images, class_name)
        
        # Test augmentation pipeline
        results = test_military_augmentation_pipeline(dataset)
        
        # Save sample results
        save_sample_augmentations(results)
        
        # Print summary
        print_pipeline_summary(results)
        
        print(f"\n{'='*60}")
        print("PIPELINE TEST COMPLETED SUCCESSFULLY!")
        print(f"{'='*60}")
        print("\nCheck 'data/processed/' for sample augmented images.")
        print("\nTo process all classes, modify the main() function to remove the 3-class limit.")
        
    except Exception as e:
        print(f"Error during pipeline test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()