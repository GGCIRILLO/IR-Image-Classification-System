"""
Image optimization module for IR dataset preprocessing.

This module provides functionality to optimize large IR image datasets by:
- Resizing images from large formats (e.g., 4096x4096) to standard sizes (e.g., 256x256)
- Converting between image formats (BMP to WebP for space efficiency)
- Batch processing with progress tracking
- Preserving IR image characteristics during optimization
"""

import numpy as np
from PIL import Image, ImageOps
import cv2
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import logging
from datetime import datetime
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from ..models.data_models import IRImage


class ImageOptimizationError(Exception):
    """Custom exception for image optimization errors."""
    pass


class IRImageOptimizer:
    """
    Optimizes IR images for efficient storage and processing.
    
    Handles batch conversion from large BMP files to smaller WebP format
    while preserving IR image characteristics and quality.
    """
    
    def __init__(self, 
                 target_size: Tuple[int, int] = (256, 256),
                 output_format: str = 'webp',
                 quality: int = 85,
                 preserve_aspect_ratio: bool = False,
                 max_workers: int = 4):
        """
        Initialize the image optimizer.
        
        Args:
            target_size: Target dimensions (width, height) for resized images
            output_format: Output format ('webp', 'png', 'jpeg')
            quality: Compression quality (1-100, only for lossy formats)
            preserve_aspect_ratio: Whether to maintain aspect ratio during resize
            max_workers: Number of parallel workers for batch processing
        """
        self.target_size = target_size
        self.output_format = output_format.lower()
        self.quality = quality
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.max_workers = max_workers
        
        self.logger = logging.getLogger(__name__)
        
        # Supported input formats
        self.supported_input_formats = {'.bmp', '.png', '.jpg', '.jpeg', '.tiff', '.tif'}
        
        # Output format configurations
        self.format_configs = {
            'webp': {'extension': '.webp', 'pil_format': 'WebP', 'supports_quality': True},
            'png': {'extension': '.png', 'pil_format': 'PNG', 'supports_quality': False},
            'jpeg': {'extension': '.jpg', 'pil_format': 'JPEG', 'supports_quality': True},
            'jpg': {'extension': '.jpg', 'pil_format': 'JPEG', 'supports_quality': True}
        }
        
        if self.output_format not in self.format_configs:
            raise ValueError(f"Unsupported output format: {output_format}")
        
        self.logger.info(f"IRImageOptimizer initialized: {target_size} -> {output_format}")
    
    def optimize_single_image(self, 
                            input_path: Union[str, Path], 
                            output_path: Optional[Union[str, Path]] = None,
                            preserve_ir_characteristics: bool = True) -> Dict[str, Any]:
        """
        Optimize a single image file.
        
        Args:
            input_path: Path to input image file
            output_path: Path for output file (auto-generated if None)
            preserve_ir_characteristics: Whether to preserve IR image properties
            
        Returns:
            Dict[str, Any]: Optimization results and statistics
        """
        input_path = Path(input_path)
        
        # Validate input file
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        if input_path.suffix.lower() not in self.supported_input_formats:
            raise ImageOptimizationError(f"Unsupported input format: {input_path.suffix}")
        
        # Generate output path if not provided
        if output_path is None:
            output_path = self._generate_output_path(input_path)
        else:
            output_path = Path(output_path)
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        start_time = datetime.now()
        
        try:
            # Load and process image
            with Image.open(input_path) as img:
                # Convert to grayscale if needed (IR images are typically grayscale)
                if img.mode != 'L':
                    if img.mode == 'RGBA':
                        # Handle transparency by compositing on black background
                        background = Image.new('RGB', img.size, (0, 0, 0))
                        background.paste(img, mask=img.split()[-1])
                        img = background.convert('L')
                    else:
                        img = img.convert('L')
                
                # Store original dimensions
                original_size = img.size
                original_file_size = input_path.stat().st_size
                
                # Resize image
                resized_img = self._resize_image(img)
                
                # Apply IR-specific enhancements if requested
                if preserve_ir_characteristics:
                    resized_img = self._enhance_ir_characteristics(resized_img)
                
                # Save optimized image
                self._save_optimized_image(resized_img, output_path)
                
                # Calculate statistics
                processing_time = (datetime.now() - start_time).total_seconds()
                output_file_size = output_path.stat().st_size
                compression_ratio = original_file_size / output_file_size if output_file_size > 0 else 0
                
                result = {
                    'input_path': str(input_path),
                    'output_path': str(output_path),
                    'original_size': original_size,
                    'target_size': self.target_size,
                    'original_file_size': original_file_size,
                    'output_file_size': output_file_size,
                    'compression_ratio': compression_ratio,
                    'space_saved_bytes': original_file_size - output_file_size,
                    'space_saved_percent': ((original_file_size - output_file_size) / original_file_size) * 100,
                    'processing_time_seconds': processing_time,
                    'success': True,
                    'error': None
                }
                
                self.logger.info(f"Optimized {input_path.name}: "
                               f"{original_size} -> {self.target_size}, "
                               f"{original_file_size:,} -> {output_file_size:,} bytes "
                               f"({compression_ratio:.1f}x compression)")
                
                return result
                
        except Exception as e:
            error_msg = f"Failed to optimize {input_path}: {str(e)}"
            self.logger.error(error_msg)
            
            return {
                'input_path': str(input_path),
                'output_path': str(output_path) if output_path else None,
                'success': False,
                'error': error_msg,
                'processing_time_seconds': (datetime.now() - start_time).total_seconds()
            }
    
    def optimize_batch(self, 
                      input_directory: Union[str, Path],
                      output_directory: Union[str, Path],
                      recursive: bool = True,
                      preserve_structure: bool = True) -> Dict[str, Any]:
        """
        Optimize all images in a directory.
        
        Args:
            input_directory: Directory containing input images
            output_directory: Directory for optimized images
            recursive: Whether to process subdirectories
            preserve_structure: Whether to maintain directory structure
            
        Returns:
            Dict[str, Any]: Batch processing results and statistics
        """
        input_dir = Path(input_directory)
        output_dir = Path(output_directory)
        
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        # Find all image files
        image_files = self._find_image_files(input_dir, recursive)
        
        if not image_files:
            self.logger.warning(f"No supported image files found in {input_dir}")
            return {
                'total_files': 0,
                'successful': 0,
                'failed': 0,
                'total_space_saved': 0,
                'processing_time': 0,
                'results': []
            }
        
        self.logger.info(f"Found {len(image_files)} images to optimize")
        
        # Process images in parallel
        results = []
        successful = 0
        failed = 0
        total_space_saved = 0
        
        start_time = datetime.now()
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_file = {}
            
            for input_file in image_files:
                # Calculate output path
                if preserve_structure:
                    relative_path = input_file.relative_to(input_dir)
                    output_file = output_dir / relative_path.with_suffix(
                        self.format_configs[self.output_format]['extension']
                    )
                else:
                    output_file = output_dir / (input_file.stem + 
                                              self.format_configs[self.output_format]['extension'])
                
                future = executor.submit(self.optimize_single_image, input_file, output_file)
                future_to_file[future] = input_file
            
            # Process completed tasks with progress bar
            with tqdm(total=len(image_files), desc="Optimizing images") as pbar:
                for future in as_completed(future_to_file):
                    result = future.result()
                    results.append(result)
                    
                    if result['success']:
                        successful += 1
                        total_space_saved += result.get('space_saved_bytes', 0)
                    else:
                        failed += 1
                    
                    pbar.update(1)
                    pbar.set_postfix({
                        'Success': successful,
                        'Failed': failed,
                        'Space Saved': f"{total_space_saved / (1024*1024):.1f}MB"
                    })
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Calculate summary statistics
        if successful > 0:
            avg_compression = np.mean([r.get('compression_ratio', 0) for r in results if r['success']])
            avg_processing_time = np.mean([r.get('processing_time_seconds', 0) for r in results if r['success']])
        else:
            avg_compression = 0
            avg_processing_time = 0
        
        summary = {
            'total_files': len(image_files),
            'successful': successful,
            'failed': failed,
            'success_rate': (successful / len(image_files)) * 100,
            'total_space_saved_bytes': total_space_saved,
            'total_space_saved_mb': total_space_saved / (1024 * 1024),
            'total_space_saved_gb': total_space_saved / (1024 * 1024 * 1024),
            'average_compression_ratio': avg_compression,
            'average_processing_time': avg_processing_time,
            'total_processing_time': processing_time,
            'throughput_files_per_second': len(image_files) / processing_time if processing_time > 0 else 0,
            'results': results
        }
        
        self.logger.info(f"Batch optimization complete: {successful}/{len(image_files)} successful, "
                        f"{total_space_saved / (1024*1024):.1f}MB saved, "
                        f"{processing_time:.1f}s total time")
        
        return summary
    
    def estimate_optimization_savings(self, 
                                    input_directory: Union[str, Path],
                                    sample_size: int = 10) -> Dict[str, Any]:
        """
        Estimate space savings from optimization without actually processing all files.
        
        Args:
            input_directory: Directory containing input images
            sample_size: Number of files to sample for estimation
            
        Returns:
            Dict[str, Any]: Estimated savings and statistics
        """
        input_dir = Path(input_directory)
        
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        # Find all image files
        image_files = self._find_image_files(input_dir, recursive=True)
        
        if not image_files:
            return {'error': 'No image files found'}
        
        # Sample files for estimation
        sample_files = np.random.choice(image_files, min(sample_size, len(image_files)), replace=False)
        
        # Process sample files to temporary location
        import tempfile
        sample_results = []
        
        with tempfile.TemporaryDirectory() as temp_dir:
            for sample_file in sample_files:
                temp_output = Path(temp_dir) / (sample_file.stem + 
                                              self.format_configs[self.output_format]['extension'])
                result = self.optimize_single_image(sample_file, temp_output)
                if result['success']:
                    sample_results.append(result)
        
        if not sample_results:
            return {'error': 'Failed to process any sample files'}
        
        # Calculate statistics from sample
        avg_compression_ratio = np.mean([r['compression_ratio'] for r in sample_results])
        avg_space_saved_percent = np.mean([r['space_saved_percent'] for r in sample_results])
        
        # Calculate total current size
        total_current_size = sum(f.stat().st_size for f in image_files)
        
        # Estimate total savings
        estimated_total_saved = total_current_size * (avg_space_saved_percent / 100)
        estimated_final_size = total_current_size - estimated_total_saved
        
        return {
            'total_files': len(image_files),
            'sample_size': len(sample_results),
            'current_total_size_bytes': total_current_size,
            'current_total_size_gb': total_current_size / (1024**3),
            'estimated_compression_ratio': avg_compression_ratio,
            'estimated_space_saved_percent': avg_space_saved_percent,
            'estimated_space_saved_bytes': estimated_total_saved,
            'estimated_space_saved_gb': estimated_total_saved / (1024**3),
            'estimated_final_size_bytes': estimated_final_size,
            'estimated_final_size_gb': estimated_final_size / (1024**3),
            'sample_results': sample_results
        }
    
    def _resize_image(self, img: Image.Image) -> Image.Image:
        """Resize image to target dimensions."""
        if self.preserve_aspect_ratio:
            # Resize maintaining aspect ratio, then pad
            img.thumbnail(self.target_size, Image.Resampling.LANCZOS)
            
            # Create new image with target size and black background
            new_img = Image.new('L', self.target_size, 0)
            
            # Calculate position to center the resized image
            paste_x = (self.target_size[0] - img.width) // 2
            paste_y = (self.target_size[1] - img.height) // 2
            
            new_img.paste(img, (paste_x, paste_y))
            return new_img
        else:
            # Direct resize to target dimensions
            return img.resize(self.target_size, Image.Resampling.LANCZOS)
    
    def _enhance_ir_characteristics(self, img: Image.Image) -> Image.Image:
        """Apply IR-specific enhancements to preserve image characteristics."""
        # Convert to numpy for processing
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        # Enhance contrast for IR imagery
        # Use histogram equalization to improve contrast
        img_uint8 = (img_array * 255).astype(np.uint8)
        equalized = cv2.equalizeHist(img_uint8)
        enhanced_array = equalized.astype(np.float32) / 255.0
        
        # Blend with original to avoid over-enhancement
        blend_factor = 0.3
        final_array = (1 - blend_factor) * img_array + blend_factor * enhanced_array
        
        # Convert back to PIL Image
        final_uint8 = (np.clip(final_array, 0, 1) * 255).astype(np.uint8)
        return Image.fromarray(final_uint8, mode='L')
    
    def _save_optimized_image(self, img: Image.Image, output_path: Path) -> None:
        """Save image in optimized format."""
        format_config = self.format_configs[self.output_format]
        
        save_kwargs = {'format': format_config['pil_format']}
        
        # Add quality parameter for formats that support it
        if format_config['supports_quality']:
            save_kwargs['quality'] = self.quality
            
            # WebP-specific optimizations
            if self.output_format == 'webp':
                save_kwargs['method'] = 6  # Better compression
                save_kwargs['lossless'] = False
        
        img.save(output_path, **save_kwargs)
    
    def _generate_output_path(self, input_path: Path) -> Path:
        """Generate output path based on input path."""
        extension = self.format_configs[self.output_format]['extension']
        return input_path.with_suffix(extension)
    
    def _find_image_files(self, directory: Path, recursive: bool = True) -> List[Path]:
        """Find all supported image files in directory."""
        image_files = []
        
        if recursive:
            pattern = "**/*"
        else:
            pattern = "*"
        
        for file_path in directory.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_input_formats:
                image_files.append(file_path)
        
        return sorted(image_files)


# Convenience functions

def optimize_ir_dataset(input_directory: Union[str, Path],
                       output_directory: Union[str, Path],
                       target_size: Tuple[int, int] = (256, 256),
                       output_format: str = 'webp',
                       quality: int = 85,
                       max_workers: int = 4) -> Dict[str, Any]:
    """
    Convenience function to optimize an entire IR dataset.
    
    Args:
        input_directory: Directory containing BMP files
        output_directory: Directory for optimized WebP files
        target_size: Target image dimensions
        output_format: Output format ('webp', 'png', 'jpeg')
        quality: Compression quality (1-100)
        max_workers: Number of parallel workers
        
    Returns:
        Dict[str, Any]: Optimization results
    """
    optimizer = IRImageOptimizer(
        target_size=target_size,
        output_format=output_format,
        quality=quality,
        max_workers=max_workers
    )
    
    return optimizer.optimize_batch(
        input_directory=input_directory,
        output_directory=output_directory,
        recursive=True,
        preserve_structure=True
    )


def estimate_dataset_savings(input_directory: Union[str, Path],
                           target_size: Tuple[int, int] = (256, 256),
                           output_format: str = 'webp',
                           quality: int = 85,
                           sample_size: int = 20) -> Dict[str, Any]:
    """
    Estimate space savings for dataset optimization.
    
    Args:
        input_directory: Directory containing images to analyze
        target_size: Target image dimensions
        output_format: Output format for estimation
        quality: Compression quality
        sample_size: Number of files to sample
        
    Returns:
        Dict[str, Any]: Estimated savings
    """
    optimizer = IRImageOptimizer(
        target_size=target_size,
        output_format=output_format,
        quality=quality
    )
    
    return optimizer.estimate_optimization_savings(
        input_directory=input_directory,
        sample_size=sample_size
    )