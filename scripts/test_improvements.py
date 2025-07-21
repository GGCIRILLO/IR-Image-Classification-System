#!/usr/bin/env python3
"""
Test script to demonstrate the improvements in similarity and confidence scores.

This script compares performance before and after applying improvements.
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.query.processor import QueryProcessor
from src.database.similarity_searcher import SimilaritySearcher
from src.embedding.extractor import EmbeddingExtractor
from scripts.quick_improvements import QuickImprovements

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_with_sample_data(database_path: str, collection_name: str = "ir_embeddings") -> Dict[str, Any]:
    """Test improvements with sample data from the database."""
    logger.info("Testing improvements with sample data from database...")
    
    try:
        # Initialize searcher to get sample embeddings
        searcher = SimilaritySearcher(database_path, collection_name)
        searcher.initialize()
        
        # Get a sample embedding from the database
        if not searcher.collection:
            return {'error': 'No collection available'}
        
        # Get some sample data
        try:
            sample_results = searcher.collection.peek(limit=1)
            logger.info(f"Sample results keys: {sample_results.keys()}")
            
            embeddings_list = sample_results.get('embeddings', [])
            if len(embeddings_list) == 0:
                return {'error': 'No embeddings found in database'}
            
            # Use the first embedding as a test query
            test_embedding = np.array(embeddings_list[0])
            
        except Exception as e:
            logger.error(f"Error getting sample data: {e}")
            # Create a dummy test embedding if we can't get real data
            test_embedding = np.random.rand(512).astype(np.float32)
            test_embedding = test_embedding / np.linalg.norm(test_embedding)  # Normalize
            logger.info("Using random test embedding")
        
        logger.info(f"Using sample embedding with shape: {test_embedding.shape}")
        
        # Test BEFORE improvements
        logger.info("Testing BEFORE improvements...")
        searcher_before = SimilaritySearcher(database_path, collection_name)
        searcher_before.initialize()
        
        results_before, metrics_before = searcher_before.search_similar(test_embedding, k=10)
        
        before_stats = {
            'num_results': len(results_before),
            'avg_similarity': np.mean([r.similarity_score for r in results_before]) if results_before else 0.0,
            'max_similarity': np.max([r.similarity_score for r in results_before]) if results_before else 0.0,
            'avg_confidence': np.mean([r.confidence for r in results_before]) if results_before else 0.0,
            'max_confidence': np.max([r.confidence for r in results_before]) if results_before else 0.0,
            'high_confidence_count': len([r for r in results_before if r.confidence >= 0.7]),
            'search_time_ms': metrics_before.search_time_ms
        }
        
        logger.info(f"BEFORE stats: {before_stats}")
        
        # Apply improvements
        logger.info("Applying improvements...")
        improver = QuickImprovements(database_path, collection_name)
        improver.initialize_components()
        
        # Apply all improvements
        improvements = improver.apply_all_improvements()
        
        # Test AFTER improvements
        logger.info("Testing AFTER improvements...")
        results_after, metrics_after = improver.searcher.search_similar(test_embedding, k=10)
        
        after_stats = {
            'num_results': len(results_after),
            'avg_similarity': np.mean([r.similarity_score for r in results_after]) if results_after else 0.0,
            'max_similarity': np.max([r.similarity_score for r in results_after]) if results_after else 0.0,
            'avg_confidence': np.mean([r.confidence for r in results_after]) if results_after else 0.0,
            'max_confidence': np.max([r.confidence for r in results_after]) if results_after else 0.0,
            'high_confidence_count': len([r for r in results_after if r.confidence >= 0.7]),
            'search_time_ms': metrics_after.search_time_ms
        }
        
        logger.info(f"AFTER stats: {after_stats}")
        
        # Calculate improvements
        improvements_achieved = {
            'similarity_improvement': after_stats['avg_similarity'] - before_stats['avg_similarity'],
            'confidence_improvement': after_stats['avg_confidence'] - before_stats['avg_confidence'],
            'max_similarity_improvement': after_stats['max_similarity'] - before_stats['max_similarity'],
            'max_confidence_improvement': after_stats['max_confidence'] - before_stats['max_confidence'],
            'high_confidence_improvement': after_stats['high_confidence_count'] - before_stats['high_confidence_count'],
            'results_improvement': after_stats['num_results'] - before_stats['num_results']
        }
        
        return {
            'before_stats': before_stats,
            'after_stats': after_stats,
            'improvements_achieved': improvements_achieved,
            'improvements_applied': improvements
        }
        
    except Exception as e:
        logger.error(f"Error testing improvements: {e}")
        return {'error': str(e)}


def print_comparison_report(results: Dict[str, Any]):
    """Print a detailed comparison report."""
    if 'error' in results:
        print(f"❌ Error: {results['error']}")
        return
    
    print("\n" + "="*80)
    print("🔬 IR IMAGE CLASSIFICATION - IMPROVEMENT TEST RESULTS")
    print("="*80)
    
    before = results['before_stats']
    after = results['after_stats']
    improvements = results['improvements_achieved']
    
    print("\n📊 PERFORMANCE COMPARISON:")
    print("-" * 50)
    
    metrics = [
        ('Results Found', 'num_results', ''),
        ('Avg Similarity', 'avg_similarity', '.4f'),
        ('Max Similarity', 'max_similarity', '.4f'),
        ('Avg Confidence', 'avg_confidence', '.4f'),
        ('Max Confidence', 'max_confidence', '.4f'),
        ('High Confidence Results', 'high_confidence_count', ''),
        ('Search Time (ms)', 'search_time_ms', '.2f')
    ]
    
    for name, key, fmt in metrics:
        before_val = before[key]
        after_val = after[key]
        
        if fmt:
            before_str = f"{before_val:{fmt}}"
            after_str = f"{after_val:{fmt}}"
        else:
            before_str = str(before_val)
            after_str = str(after_val)
        
        # Calculate change
        if isinstance(before_val, (int, float)) and before_val != 0:
            change_pct = ((after_val - before_val) / before_val) * 100
            change_str = f"({change_pct:+.1f}%)"
        else:
            change_str = ""
        
        print(f"  {name:25} | Before: {before_str:>8} | After: {after_str:>8} {change_str}")
    
    print("\n🎯 KEY IMPROVEMENTS:")
    print("-" * 50)
    
    key_improvements = [
        ('Similarity Score', 'similarity_improvement', '+{:.4f}'),
        ('Confidence Score', 'confidence_improvement', '+{:.4f}'),
        ('Max Similarity', 'max_similarity_improvement', '+{:.4f}'),
        ('Max Confidence', 'max_confidence_improvement', '+{:.4f}'),
        ('High Confidence Results', 'high_confidence_improvement', '+{}')
    ]
    
    for name, key, fmt in key_improvements:
        improvement = improvements[key]
        if improvement > 0:
            print(f"  ✅ {name:25} | {fmt.format(improvement)}")
        elif improvement < 0:
            print(f"  ⚠️  {name:25} | {improvement:.4f}")
        else:
            print(f"  ➖ {name:25} | No change")
    
    print("\n🔧 APPLIED IMPROVEMENTS:")
    print("-" * 50)
    for improvement, description in results['improvements_applied'].items():
        print(f"  ✅ {improvement}: {description}")
    
    print("\n📈 SUMMARY:")
    print("-" * 50)
    
    # Overall assessment
    similarity_gain = improvements['similarity_improvement']
    confidence_gain = improvements['confidence_improvement']
    
    if similarity_gain > 0.1 or confidence_gain > 0.1:
        print("  🎉 EXCELLENT: Significant improvements achieved!")
    elif similarity_gain > 0.05 or confidence_gain > 0.05:
        print("  ✅ GOOD: Noticeable improvements achieved!")
    elif similarity_gain > 0.01 or confidence_gain > 0.01:
        print("  👍 MODERATE: Some improvements achieved!")
    else:
        print("  ℹ️  MINIMAL: Small or no improvements (may need fine-tuning)")
    
    print(f"  • Similarity improved by: {similarity_gain:+.4f}")
    print(f"  • Confidence improved by: {confidence_gain:+.4f}")
    print(f"  • High confidence results: {improvements['high_confidence_improvement']:+d}")


def main():
    """Main function for testing improvements."""
    parser = argparse.ArgumentParser(description="Test IR image classification improvements")
    parser.add_argument('--database', '-d', required=True, help='Database path')
    parser.add_argument('--collection', default='ir_embeddings', help='Collection name')
    
    args = parser.parse_args()
    
    print("🧪 Testing IR Image Classification Improvements")
    print("=" * 60)
    
    try:
        # Run the test
        results = test_with_sample_data(args.database, args.collection)
        
        # Print the report
        print_comparison_report(results)
        
        print("\n✅ Testing completed!")
        
    except Exception as e:
        logger.error(f"Testing failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()