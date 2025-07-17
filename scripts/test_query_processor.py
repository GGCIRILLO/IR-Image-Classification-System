"""
Test script for QueryProcessor implementation.

This script demonstrates how to use the QueryProcessor for end-to-end 
image similarity queries in the IR Image Classification System.
"""

import sys
import os
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.query import QueryProcessor, QueryValidationError, QueryProcessingError
from src.models.data_models import IRImage


def test_query_processor():
    """Test basic QueryProcessor functionality."""
    
    # Configuration
    database_path = "data/chroma_db_final"  # Path to your vector database
    model_path = None  # Path to your fine-tuned model (optional for testing)
    
    # Initialize query processor
    processor = QueryProcessor(
        database_path=database_path,
        model_path=model_path,
        collection_name="ir_embeddings",
        config={
            'max_query_time': 2.0,
            'min_confidence_threshold': 0.7,
            'top_k_results': 5,
            'enable_result_reranking': True,
            'cache_queries': True
        }
    )
    
    try:
        # Initialize the processor
        print("Initializing QueryProcessor...")
        if processor.initialize():
            print("✓ QueryProcessor initialized successfully")
        else:
            print("✗ Failed to initialize QueryProcessor")
            return
            
    except QueryProcessingError as e:
        print(f"✗ Initialization failed: {e}")
        return
    
    # Test with sample image path (adjust path as needed)
    sample_image_paths = [
        "data/processed/M-1A1 Abrams Tank/image_001.png",  # Example path
        "data/processed/M2 Bradley Apc Tank/image_001.png",  # Example path
    ]
    
    for image_path in sample_image_paths:
        if not Path(image_path).exists():
            print(f"Sample image not found: {image_path}")
            continue
            
        try:
            print(f"\nProcessing query for: {image_path}")
            
            # Process the query
            start_time = datetime.now()
            result = processor.process_query(
                image_input=image_path,
                query_id=f"test_query_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                options={
                    'strict_validation': False,  # Allow non-perfect IR format
                    'confidence_threshold': 0.6,  # Lower threshold for testing
                    'max_results': 3
                }
            )
            end_time = datetime.now()
            
            # Display results
            print(f"✓ Query completed in {result.processing_time:.3f} seconds")
            print(f"✓ Found {len(result.results)} similar images")
            print(f"✓ Model version: {result.model_version}")
            
            for i, similarity_result in enumerate(result.results, 1):
                print(f"  {i}. Image ID: {similarity_result.image_id}")
                print(f"     Similarity: {similarity_result.similarity_score:.3f}")
                print(f"     Confidence: {similarity_result.confidence:.3f}")
                print(f"     Class: {similarity_result.object_class}")
                print(f"     Confidence Level: {similarity_result.metadata.get('confidence_level', 'Unknown')}")
                print()
                
        except QueryValidationError as e:
            print(f"✗ Query validation failed: {e}")
        except QueryProcessingError as e:
            print(f"✗ Query processing failed: {e}")
        except Exception as e:
            print(f"✗ Unexpected error: {e}")
    
    # Display performance statistics
    print("\nPerformance Statistics:")
    stats = processor.get_performance_stats()
    for metric_name, metric_stats in stats.items():
        if metric_stats['count'] > 0:
            print(f"  {metric_name}:")
            print(f"    Average: {metric_stats['average']:.3f}s")
            print(f"    Min: {metric_stats['min']:.3f}s")
            print(f"    Max: {metric_stats['max']:.3f}s")
            print(f"    Count: {metric_stats['count']}")
    
    # Validate system performance requirements
    print("\nSystem Performance Validation:")
    validation_results = processor.validate_system_performance()
    for requirement, passed in validation_results.items():
        status = "✓" if passed else "✗"
        print(f"  {status} {requirement}: {'PASS' if passed else 'FAIL'}")


def test_query_processor_with_numpy_array():
    """Test QueryProcessor with numpy array input."""
    import numpy as np
    
    # Create a synthetic IR image (white rectangle on black background)
    test_image = np.zeros((224, 224), dtype=np.float32)
    test_image[50:150, 75:175] = 1.0  # White rectangle
    
    processor = QueryProcessor(
        database_path="data/chroma_db_final",
        model_path=None,
        config={'strict_validation': False}
    )
    
    try:
        if processor.initialize():
            print("\nTesting with synthetic numpy array...")
            
            result = processor.process_query(
                image_input=test_image,
                query_id="synthetic_test",
                options={'strict_validation': False}
            )
            
            print(f"✓ Synthetic image query completed in {result.processing_time:.3f}s")
            print(f"✓ Found {len(result.results)} similar images")
            
    except Exception as e:
        print(f"✗ Synthetic image test failed: {e}")


if __name__ == "__main__":
    print("IR Image Classification System - QueryProcessor Test")
    print("=" * 60)
    
    # Run basic test
    test_query_processor()
    
    # Run numpy array test
    test_query_processor_with_numpy_array()
    
    print("\nTest completed!")
