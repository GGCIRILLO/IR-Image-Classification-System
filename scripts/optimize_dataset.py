#!/usr/bin/env python3
"""
Dataset optimization script for IR image classification.

This script optimizes large IR image datasets by:
- Converting BMP files to WebP format for better compression
- Resizing images from 4096x4096 to 256x256 for efficient processing
- Preserving IR image characteristics during optimization
- Providing detailed progress tracking and statistics

Usage:
    python scripts/optimize_dataset.py --input data/raw --output data/processed --estimate-only
    python scripts/optimize_dataset.py --input data/raw --output data/processed --format webp --size 256 256
"""

import argparse
import sys
from pathlib import Path
import logging
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.image_optimizer import IRImageOptimizer, optimize_ir_dataset, estimate_dataset_savings


def setup_logging(log_level: str = 'INFO') -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'logs/dataset_optimization_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Optimize IR image dataset for efficient processing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Estimate space savings without processing
  python scripts/optimize_dataset.py --input data/raw --estimate-only
  
  # Convert BMP to WebP with 256x256 size
  python scripts/optimize_dataset.py --input data/raw --output data/processed
  
  # Custom format and size
  python scripts/optimize_dataset.py --input data/raw --output data/processed --format png --size 512 512
  
  # High quality WebP conversion
  python scripts/optimize_dataset.py --input data/raw --output data/processed --quality 95
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Input directory containing raw images (e.g., data/raw)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output directory for optimized images (e.g., data/processed)'
    )
    
    parser.add_argument(
        '--format', '-f',
        type=str,
        choices=['webp', 'png', 'jpeg', 'jpg'],
        default='webp',
        help='Output image format (default: webp)'
    )
    
    parser.add_argument(
        '--size', '-s',
        type=int,
        nargs=2,
        default=[256, 256],
        metavar=('WIDTH', 'HEIGHT'),
        help='Target image size in pixels (default: 256 256)'
    )
    
    parser.add_argument(
        '--quality', '-q',
        type=int,
        default=85,
        help='Compression quality 1-100 (default: 85, only for lossy formats)'
    )
    
    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=4,
        help='Number of parallel workers (default: 4)'
    )
    
    parser.add_argument(
        '--estimate-only',
        action='store_true',
        help='Only estimate space savings without processing files'
    )
    
    parser.add_argument(
        '--sample-size',
        type=int,
        default=20,
        help='Number of files to sample for estimation (default: 20)'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--preserve-aspect-ratio',
        action='store_true',
        help='Preserve aspect ratio during resize (pad with black)'
    )
    
    return parser.parse_args()


def print_estimation_results(results: dict) -> None:
    """Print estimation results in a formatted way."""
    print("\n" + "="*60)
    print("DATASET OPTIMIZATION ESTIMATION")
    print("="*60)
    
    if 'error' in results:
        print(f"❌ Error: {results['error']}")
        return
    
    print(f"📁 Total files found: {results['total_files']:,}")
    print(f"🔍 Sample size: {results['sample_size']}")
    print(f"📊 Current dataset size: {results['current_total_size_gb']:.2f} GB")
    print(f"🗜️  Estimated compression ratio: {results['estimated_compression_ratio']:.1f}x")
    print(f"💾 Estimated space saved: {results['estimated_space_saved_gb']:.2f} GB ({results['estimated_space_saved_percent']:.1f}%)")
    print(f"📦 Estimated final size: {results['estimated_final_size_gb']:.2f} GB")
    
    print(f"\n💡 Sample results:")
    for i, sample in enumerate(results['sample_results'][:5], 1):
        original_mb = sample['original_file_size'] / (1024*1024)
        output_mb = sample['output_file_size'] / (1024*1024)
        print(f"   {i}. {Path(sample['input_path']).name}")
        print(f"      {sample['original_size']} -> {sample['target_size']}")
        print(f"      {original_mb:.1f}MB -> {output_mb:.1f}MB ({sample['compression_ratio']:.1f}x)")
    
    if len(results['sample_results']) > 5:
        print(f"   ... and {len(results['sample_results']) - 5} more samples")


def print_optimization_results(results: dict) -> None:
    """Print optimization results in a formatted way."""
    print("\n" + "="*60)
    print("DATASET OPTIMIZATION RESULTS")
    print("="*60)
    
    print(f"📁 Total files processed: {results['total_files']:,}")
    print(f"✅ Successful: {results['successful']:,}")
    print(f"❌ Failed: {results['failed']:,}")
    print(f"📊 Success rate: {results['success_rate']:.1f}%")
    
    print(f"\n💾 Space savings:")
    print(f"   Total saved: {results['total_space_saved_gb']:.2f} GB")
    print(f"   Average compression: {results['average_compression_ratio']:.1f}x")
    
    print(f"\n⏱️  Performance:")
    print(f"   Total time: {results['total_processing_time']:.1f} seconds")
    print(f"   Average per file: {results['average_processing_time']:.2f} seconds")
    print(f"   Throughput: {results['throughput_files_per_second']:.1f} files/second")
    
    if results['failed'] > 0:
        print(f"\n❌ Failed files:")
        failed_results = [r for r in results['results'] if not r['success']]
        for i, failed in enumerate(failed_results[:10], 1):
            print(f"   {i}. {Path(failed['input_path']).name}: {failed['error']}")
        
        if len(failed_results) > 10:
            print(f"   ... and {len(failed_results) - 10} more failures")


def main():
    """Main function."""
    args = parse_arguments()
    
    # Set up logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Validate arguments
    input_dir = Path(args.input)
    if not input_dir.exists():
        print(f"❌ Error: Input directory '{input_dir}' does not exist")
        sys.exit(1)
    
    if not args.estimate_only and not args.output:
        print("❌ Error: --output is required unless using --estimate-only")
        sys.exit(1)
    
    # Ensure logs directory exists
    Path('logs').mkdir(exist_ok=True)
    
    target_size = tuple(args.size)
    
    print(f"🚀 Starting IR dataset optimization...")
    print(f"📂 Input directory: {input_dir}")
    print(f"🎯 Target size: {target_size[0]}x{target_size[1]}")
    print(f"🗜️  Output format: {args.format}")
    print(f"⚙️  Quality: {args.quality}")
    
    if args.estimate_only:
        print(f"🔍 Running estimation with {args.sample_size} samples...")
        
        try:
            results = estimate_dataset_savings(
                input_directory=input_dir,
                target_size=target_size,
                output_format=args.format,
                quality=args.quality,
                sample_size=args.sample_size
            )
            
            print_estimation_results(results)
            
        except Exception as e:
            logger.error(f"Estimation failed: {str(e)}")
            print(f"❌ Estimation failed: {str(e)}")
            sys.exit(1)
    
    else:
        output_dir = Path(args.output)
        print(f"📁 Output directory: {output_dir}")
        print(f"👥 Workers: {args.workers}")
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            print(f"\n🔄 Processing images...")
            
            results = optimize_ir_dataset(
                input_directory=input_dir,
                output_directory=output_dir,
                target_size=target_size,
                output_format=args.format,
                quality=args.quality,
                max_workers=args.workers
            )
            
            print_optimization_results(results)
            
            if results['successful'] > 0:
                print(f"\n🎉 Optimization completed successfully!")
                print(f"💾 Saved {results['total_space_saved_gb']:.2f} GB of storage space")
            else:
                print(f"\n❌ No files were successfully optimized")
                sys.exit(1)
                
        except Exception as e:
            logger.error(f"Optimization failed: {str(e)}")
            print(f"❌ Optimization failed: {str(e)}")
            sys.exit(1)


if __name__ == '__main__':
    main()