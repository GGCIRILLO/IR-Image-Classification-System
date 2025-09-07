#!/usr/bin/env python3
"""
Batch Test Set Runner for IR Image Classification System

This script runs the mission runner on all images in the test set,
collects results, and generates summary reports and visualizations.

Usage:
    python scripts/batch_test_set.py
"""

import os
import sys
import json
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import re

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np
    PLOTTING_AVAILABLE = True
except ImportError:
    print("⚠️  Warning: Plotting libraries not available. Install with: pip install matplotlib seaborn pandas numpy")
    PLOTTING_AVAILABLE = False


class BatchTestRunner:
    """
    Batch runner for testing all images in the test set.
    """

    def __init__(self, test_set_path: str = "data/test_set",
                 output_dir: str = "results/batch_test",
                 database: str = "data/vector_db_resnet18",
                 model_type: str = "resnet18",
                 model_path: Optional[str] = None,
                 max_results: int = 5,
                 confidence_threshold: float = 0.1,
                 similarity_threshold: float = 0.1):
        """
        Initialize batch test runner.

        Args:
            test_set_path: Path to test set directory
            output_dir: Output directory for results
            database: Vector database path
            model_type: Model type to use
            model_path: Path to fine-tuned model weights
            max_results: Maximum results per query
            confidence_threshold: Confidence threshold
            similarity_threshold: Similarity threshold
        """
        self.test_set_path = Path(test_set_path)
        self.output_dir = Path(output_dir)
        self.database = database
        self.model_type = model_type
        self.model_path = model_path
        self.max_results = max_results
        self.confidence_threshold = confidence_threshold
        self.similarity_threshold = similarity_threshold

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Results storage
        self.results = []
        self.summary_stats = {
            'total_queries': 0,
            'successful_queries': 0,
            'total_results': 0,
            'avg_similarity': 0.0,
            'avg_confidence': 0.0,
            'class_distribution': {},
            'success_rate_by_class': {}
        }

    def extract_class_from_path(self, image_path: Path) -> str:
        """
        Extract object class from image path.

        Args:
            image_path: Path to image file

        Returns:
            str: Object class name
        """
        # Get the parent directory name as class
        class_name = image_path.parent.name

        # Clean up class name (remove extra spaces, normalize)
        class_name = re.sub(r'\s+', ' ', class_name.strip())

        return class_name

    def parse_mission_output(self, output: str) -> List[Dict[str, Any]]:
        """
        Parse the output from run_mission.py to extract results.

        Args:
            output: Raw output string from mission runner

        Returns:
            List[Dict]: Parsed results
        """
        results = []

        # Split output into lines
        lines = output.split('\n')

        # Find the results table
        in_results_table = False
        for line in lines:
            if 'Rank' in line and 'Object Class' in line:
                in_results_table = True
                continue

            if in_results_table and line.strip() and not line.startswith('-'):
                # Parse result line
                # Format: Rank Object_Class Category Similarity Confidence Threat
                parts = line.split()
                if len(parts) >= 6:
                    try:
                        rank = int(parts[0])
                        # Object class is from position 1 to -4 (before Category)
                        object_class_parts = []
                        i = 1
                        while i < len(parts) and not any(cat in ' '.join(parts[i:i+2]) for cat in ['Military', 'Civilian']):
                            object_class_parts.append(parts[i])
                            i += 1

                        object_class = ' '.join(object_class_parts)

                        # Find category (Military Vehicle, Civilian Vehicle, etc.)
                        category = ""
                        if i < len(parts):
                            if parts[i] == 'Military':
                                category = 'Military Vehicle' if i+1 < len(parts) and parts[i+1] == 'Vehicle' else 'Military'
                                i += 1 if category == 'Military' else 2
                            elif parts[i] == 'Civilian':
                                category = 'Civilian Vehicle' if i+1 < len(parts) and parts[i+1] == 'Vehicle' else 'Civilian'
                                i += 1 if category == 'Civilian' else 2
                            else:
                                category = parts[i]
                                i += 1

                        # Get similarity and confidence
                        similarity = float(parts[-3]) if len(parts) > 2 else 0.0
                        confidence = float(parts[-2]) if len(parts) > 1 else 0.0

                        result = {
                            'rank': rank,
                            'object_class': object_class,
                            'category': category,
                            'similarity': similarity,
                            'confidence': confidence
                        }

                        results.append(result)

                    except (ValueError, IndexError) as e:
                        print(f"Warning: Could not parse result line: {line}")
                        continue

        return results

    def run_single_query(self, image_path: Path) -> Dict[str, Any]:
        """
        Run a single query for the given image.

        Args:
            image_path: Path to query image

        Returns:
            Dict: Query result
        """
        query_class = self.extract_class_from_path(image_path)

        print(f"🔍 Testing: {image_path.name} (Class: {query_class})")

        # Build command
        cmd = [
            sys.executable, 'scripts/run_mission.py',
            '--image', str(image_path),
            '--database', self.database,
            '--model-type', self.model_type,
            '--max-results', str(self.max_results),
            '--confidence-threshold', str(self.confidence_threshold),
            '--similarity-threshold', str(self.similarity_threshold),
            '--quiet'  # Suppress banner output
        ]
        
        # Add model path if specified
        if self.model_path:
            cmd.extend(['--model', self.model_path])

        try:
            # Run command
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=project_root,
                timeout=30
            )

            if result.returncode != 0:
                print(f"❌ Query failed for {image_path.name}: {result.stderr}")
                return {
                    'image_path': str(image_path),
                    'query_class': query_class,
                    'success': False,
                    'error': result.stderr,
                    'results': []
                }

            # Parse results
            parsed_results = self.parse_mission_output(result.stdout)

            # Check if query was successful (top result matches query class)
            success = False
            if parsed_results:
                top_result = parsed_results[0]
                # Simple class matching (case-insensitive, partial match)
                success = query_class.lower() in top_result['object_class'].lower() or \
                         top_result['object_class'].lower() in query_class.lower()

            query_result = {
                'image_path': str(image_path),
                'query_class': query_class,
                'success': success,
                'results': parsed_results
            }

            print(f"   ✅ Success: {success} | Top result: {parsed_results[0]['object_class'] if parsed_results else 'None'}")

            return query_result

        except subprocess.TimeoutExpired:
            print(f"⏰ Query timed out for {image_path.name}")
            return {
                'image_path': str(image_path),
                'query_class': query_class,
                'success': False,
                'error': 'Timeout',
                'results': []
            }
        except Exception as e:
            print(f"❌ Query error for {image_path.name}: {str(e)}")
            return {
                'image_path': str(image_path),
                'query_class': query_class,
                'success': False,
                'error': str(e),
                'results': []
            }

    def run_batch_test(self) -> None:
        """
        Run batch test on all images in test set.
        """
        print("🚀 Starting Batch Test Set Runner")
        print(f"   Test Set: {self.test_set_path}")
        print(f"   Database: {self.database}")
        print(f"   Model: {self.model_type}")
        print(f"   Output: {self.output_dir}")
        print()

        # Find all images in test set
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
        test_images = []

        for root, dirs, files in os.walk(self.test_set_path):
            for file in files:
                if Path(file).suffix.lower() in image_extensions:
                    test_images.append(Path(root) / file)

        print(f"📁 Found {len(test_images)} images to test")
        print()

        # Run queries
        successful_queries = 0
        total_similarity = 0.0
        total_confidence = 0.0
        class_success_count = {}
        class_total_count = {}

        for i, image_path in enumerate(test_images, 1):
            print(f"[{i}/{len(test_images)}] ", end="")

            query_result = self.run_single_query(image_path)
            self.results.append(query_result)

            # Update statistics
            query_class = query_result['query_class']

            if query_class not in class_total_count:
                class_total_count[query_class] = 0
                class_success_count[query_class] = 0

            class_total_count[query_class] += 1

            if query_result['success']:
                successful_queries += 1
                class_success_count[query_class] += 1

            # Calculate averages from results
            if query_result['results']:
                for result in query_result['results']:
                    total_similarity += result['similarity']
                    total_confidence += result['confidence']
                    self.summary_stats['total_results'] += 1

        # Calculate final statistics
        self.summary_stats['total_queries'] = len(test_images)
        self.summary_stats['successful_queries'] = successful_queries

        if self.summary_stats['total_results'] > 0:
            self.summary_stats['avg_similarity'] = total_similarity / self.summary_stats['total_results']
            self.summary_stats['avg_confidence'] = total_confidence / self.summary_stats['total_results']

        # Calculate success rate by class
        for class_name in class_total_count:
            total = class_total_count[class_name]
            success = class_success_count[class_name]
            self.summary_stats['success_rate_by_class'][class_name] = {
                'success_rate': success / total if total > 0 else 0,
                'total_queries': total,
                'successful_queries': success
            }

        print()
        print("✅ Batch test completed!")
        print(f"   Total queries: {self.summary_stats['total_queries']}")
        print(f"   Successful: {self.summary_stats['successful_queries']}")
        print(".1f")
        print(".3f")
        print(".3f")

    def save_results(self) -> None:
        """
        Save results to JSON file.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.output_dir / f"batch_test_results_{timestamp}.json"

        output_data = {
            'metadata': {
                'timestamp': timestamp,
                'test_set_path': str(self.test_set_path),
                'database': self.database,
                'model_type': self.model_type,
                'max_results': self.max_results,
                'confidence_threshold': self.confidence_threshold,
                'similarity_threshold': self.similarity_threshold
            },
            'summary': self.summary_stats,
            'results': self.results
        }

        with open(results_file, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"💾 Results saved to: {results_file}")

    def generate_plots(self) -> None:
        """
        Generate summary plots and visualizations.
        """
        if not PLOTTING_AVAILABLE:
            print("⚠️  Plotting libraries not available, skipping plots")
            return

        try:
            # Set style
            plt.style.use('default')
            sns.set_palette("husl")

            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Batch Test Set Results Summary', fontsize=16, fontweight='bold')

            # 1. Success Rate by Class
            if self.summary_stats['success_rate_by_class']:
                classes = list(self.summary_stats['success_rate_by_class'].keys())
                success_rates = [self.summary_stats['success_rate_by_class'][cls]['success_rate'] * 100
                               for cls in classes]

                # Sort by success rate
                sorted_indices = np.argsort(success_rates)[::-1]
                classes = [classes[i] for i in sorted_indices]
                success_rates = [success_rates[i] for i in sorted_indices]

                axes[0, 0].bar(range(len(classes)), success_rates)
                axes[0, 0].set_xticks(range(len(classes)))
                axes[0, 0].set_xticklabels(classes, rotation=45, ha='right')
                axes[0, 0].set_ylabel('Success Rate (%)')
                axes[0, 0].set_title('Success Rate by Object Class')
                axes[0, 0].grid(True, alpha=0.3)

            # 2. Overall Statistics
            stats_labels = ['Total Queries', 'Successful', 'Success Rate', 'Avg Similarity', 'Avg Confidence']
            stats_values = [
                self.summary_stats['total_queries'],
                self.summary_stats['successful_queries'],
                self.summary_stats['successful_queries'] / self.summary_stats['total_queries'] * 100,
                self.summary_stats['avg_similarity'] * 100,
                self.summary_stats['avg_confidence'] * 100
            ]

            bars = axes[0, 1].bar(stats_labels, stats_values)
            axes[0, 1].set_ylabel('Value (%)')
            axes[0, 1].set_title('Overall Test Statistics')
            axes[0, 1].grid(True, alpha=0.3)

            # Add value labels on bars
            for bar, value in zip(bars, stats_values):
                axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                               '.1f', ha='center', va='bottom', fontweight='bold')

            # 3. Similarity vs Confidence Scatter
            similarities = []
            confidences = []
            for query in self.results:
                for result in query['results']:
                    similarities.append(result['similarity'])
                    confidences.append(result['confidence'])

            if similarities and confidences:
                axes[1, 0].scatter(confidences, similarities, alpha=0.6, s=50)
                axes[1, 0].set_xlabel('Confidence')
                axes[1, 0].set_ylabel('Similarity')
                axes[1, 0].set_title('Similarity vs Confidence Distribution')
                axes[1, 0].grid(True, alpha=0.3)

                # Add trend line
                if len(similarities) > 1:
                    z = np.polyfit(confidences, similarities, 1)
                    p = np.poly1d(z)
                    axes[1, 0].plot(sorted(confidences), p(sorted(confidences)),
                                   "r--", alpha=0.8, linewidth=2)

            # 4. Results Distribution
            success_counts = [self.summary_stats['successful_queries'],
                            self.summary_stats['total_queries'] - self.summary_stats['successful_queries']]
            labels = ['Successful', 'Failed']

            axes[1, 1].pie(success_counts, labels=labels, autopct='%1.1f%%',
                          startangle=90, colors=['#4CAF50', '#F44336'])
            axes[1, 1].set_title('Query Success Distribution')

            plt.tight_layout()

            # Save plot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_file = self.output_dir / f"batch_test_summary_{timestamp}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"📊 Summary plot saved to: {plot_file}")

            plt.show()

        except Exception as e:
            print(f"❌ Error generating plots: {str(e)}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Batch Test Set Runner")
    parser.add_argument('--test-set', default='data/test_set',
                       help='Path to test set directory')
    parser.add_argument('--output-dir', default='results/batch_test',
                       help='Output directory for results')
    parser.add_argument('--database', default='data/vector_db_resnet18',
                       help='Vector database path')
    parser.add_argument('--model-type', default='resnet18',
                       help='Model type')
    parser.add_argument('--model-path', default=None,
                       help='Path to fine-tuned model weights (.pth file)')
    parser.add_argument('--max-results', type=int, default=5,
                       help='Maximum results per query')
    parser.add_argument('--confidence-threshold', type=float, default=0.1,
                       help='Confidence threshold')
    parser.add_argument('--similarity-threshold', type=float, default=0.1,
                       help='Similarity threshold')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip plot generation')

    args = parser.parse_args()

    # Create runner
    runner = BatchTestRunner(
        test_set_path=args.test_set,
        output_dir=args.output_dir,
        database=args.database,
        model_type=args.model_type,
        model_path=args.model_path,
        max_results=args.max_results,
        confidence_threshold=args.confidence_threshold,
        similarity_threshold=args.similarity_threshold
    )

    # Run batch test
    runner.run_batch_test()

    # Save results
    runner.save_results()

    # Generate plots (unless disabled)
    if not args.no_plots:
        runner.generate_plots()


if __name__ == "__main__":
    main()
