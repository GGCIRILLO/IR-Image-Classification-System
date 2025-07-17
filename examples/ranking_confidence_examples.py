"""
Examples demonstrating advanced result ranking and confidence scoring.

This module provides practical examples of how to use the new ResultRanker
and ConfidenceCalculator components in different scenarios.
"""

import os
from pathlib import Path
import sys
import numpy as np
from typing import List, Dict, Any

# Add the src directory to the path
# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.data_models import SimilarityResult, Embedding
from src.query import (
    QueryProcessor, ResultRanker, ConfidenceCalculator,
    RankingConfig, ConfidenceConfig, RankingStrategy, ConfidenceStrategy
)


def example_1_basic_ranking():
    """Example 1: Basic result ranking with different strategies."""
    print("Example 1: Basic Result Ranking")
    print("-" * 40)
    
    # Create some mock results
    results = [
        SimilarityResult(
            image_id="tank_001",
            similarity_score=0.92,
            confidence=0.85,
            object_class="Tank",
            metadata={}
        ),
        SimilarityResult(
            image_id="vehicle_001",
            similarity_score=0.88,
            confidence=0.78,
            object_class="Vehicle",
            metadata={}
        ),
        SimilarityResult(
            image_id="aircraft_001",
            similarity_score=0.82,
            confidence=0.90,
            object_class="Aircraft",
            metadata={}
        )
    ]
    
    # Configure ranker for military priority
    config = RankingConfig(
        strategy=RankingStrategy.MILITARY_PRIORITY,
        min_confidence=0.7,
        max_results=5,
        boost_critical_objects=1.2
    )
    
    ranker = ResultRanker(config)
    ranked_results, metrics = ranker.rank_results(results)
    
    print(f"Results ranked using {config.strategy.value}:")
    for i, result in enumerate(ranked_results):
        print(f"{i+1}. {result.object_class} - Similarity: {result.similarity_score:.3f}, "
              f"Confidence: {result.confidence:.3f}")
    
    print(f"Ranking metrics: {metrics.final_results} results in {metrics.ranking_time_ms:.1f}ms")
    print()


def example_2_enhanced_confidence():
    """Example 2: Enhanced confidence calculation."""
    print("Example 2: Enhanced Confidence Calculation")
    print("-" * 45)
    
    # Create a result that needs confidence enhancement
    result = SimilarityResult(
        image_id="uncertain_001",
        similarity_score=0.75,
        confidence=0.60,  # Low initial confidence
        object_class="Tank",
        metadata={}
    )
    
    # Create a mock query embedding
    query_embedding = Embedding(
        id="query_001",
        vector=np.random.normal(0, 1, 512).astype(np.float32),
        image_id="query_image",
        model_version="resnet50_v1.0"
    )
    
    # Configure confidence calculator for military applications
    config = ConfidenceConfig(
        strategy=ConfidenceStrategy.MILITARY_CALIBRATED,
        enable_adaptive_calibration=True,
        enable_uncertainty_quantification=True
    )
    
    calculator = ConfidenceCalculator(config)
    analysis = calculator.calculate_confidence(result, None, query_embedding)
    
    print(f"Original confidence: {result.confidence:.3f}")
    print(f"Enhanced confidence: {analysis.final_confidence:.3f}")
    print(f"Uncertainty estimate: {analysis.uncertainty_estimate:.3f}")
    print(f"Explanation: {analysis.explanation}")
    print(f"Confidence factors: {analysis.confidence_factors}")
    print()


def example_3_integrated_query_processing():
    """Example 3: Integrated query processing with advanced features."""
    print("Example 3: Integrated Query Processing")
    print("-" * 38)
    
    # This example shows how to configure QueryProcessor with advanced features
    # Note: This requires actual database and model files to run
    
    # Configuration for military deployment
    config = {
        'ranking_strategy': 'military_priority',
        'confidence_strategy': 'military_calibrated',
        'min_confidence_threshold': 0.7,
        'top_k_results': 5,
        'enable_diversity_filtering': True,
        'enable_confidence_calibration': True,
        'boost_critical_objects': 1.3
    }
    
    print("Configuration for military deployment:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    print("\nTo use this configuration:")
    print("""
    processor = QueryProcessor(
        database_path="data/chroma_db_final",
        model_path="models/ir_model.pth",
        config=config
    )
    
    processor.initialize()
    result = processor.process_query("path/to/query_image.png")
    """)
    print()


def example_4_custom_ranking_strategy():
    """Example 4: Creating custom ranking configurations."""
    print("Example 4: Custom Ranking Configurations")
    print("-" * 42)
    
    # High-precision configuration for critical applications
    high_precision_config = RankingConfig(
        strategy=RankingStrategy.CONFIDENCE_WEIGHTED,
        min_confidence=0.9,  # Very high confidence required
        min_similarity=0.8,  # High similarity required
        max_results=3,       # Fewer, more reliable results
        confidence_weight=0.6,
        similarity_weight=0.4,
        enable_diversity_filtering=True
    )
    
    # Fast deployment configuration
    fast_deployment_config = RankingConfig(
        strategy=RankingStrategy.SIMILARITY_ONLY,
        min_confidence=0.5,  # Lower confidence acceptable
        min_similarity=0.3,  # Lower similarity acceptable
        max_results=10,      # More results for broader coverage
        enable_diversity_filtering=False  # Faster processing
    )
    
    print("High-precision configuration:")
    print(f"  Strategy: {high_precision_config.strategy.value}")
    print(f"  Min confidence: {high_precision_config.min_confidence}")
    print(f"  Min similarity: {high_precision_config.min_similarity}")
    print(f"  Max results: {high_precision_config.max_results}")
    
    print("\nFast deployment configuration:")
    print(f"  Strategy: {fast_deployment_config.strategy.value}")
    print(f"  Min confidence: {fast_deployment_config.min_confidence}")
    print(f"  Min similarity: {fast_deployment_config.min_similarity}")
    print(f"  Max results: {fast_deployment_config.max_results}")
    print()


def example_5_confidence_calibration():
    """Example 5: Confidence calibration with historical data."""
    print("Example 5: Confidence Calibration")
    print("-" * 35)
    
    # Create confidence calculator
    config = ConfidenceConfig(
        strategy=ConfidenceStrategy.ENSEMBLE,
        enable_adaptive_calibration=True
    )
    
    calculator = ConfidenceCalculator(config)
    
    # Simulate updating historical accuracy data
    historical_data = [
        (0.9, 0.95, "Tank"),      # High confidence, high accuracy
        (0.8, 0.85, "Vehicle"),  # Good confidence, good accuracy  
        (0.7, 0.60, "Aircraft"), # Medium confidence, lower accuracy
        (0.6, 0.40, "Building")  # Low confidence, low accuracy
    ]
    
    print("Updating confidence calibration with historical data:")
    for predicted_conf, actual_acc, obj_class in historical_data:
        calculator.update_historical_accuracy(predicted_conf, actual_acc, obj_class)
        print(f"  {obj_class}: predicted {predicted_conf:.2f}, actual {actual_acc:.2f}")
    
    # Get calibration metrics
    metrics = calculator.get_calibration_metrics()
    if metrics:
        print(f"\nCalibration metrics:")
        print(f"  Calibration error: {metrics['calibration_error']:.3f}")
        print(f"  Total samples: {metrics['total_samples']}")
        print(f"  Average predicted: {metrics['average_predicted_confidence']:.3f}")
        print(f"  Average actual: {metrics['average_actual_accuracy']:.3f}")
    print()


def example_6_performance_monitoring():
    """Example 6: Performance monitoring and optimization."""
    print("Example 6: Performance Monitoring")
    print("-" * 34)
    
    # Create ranker with performance monitoring
    config = RankingConfig(
        strategy=RankingStrategy.HYBRID_SCORE,
        add_ranking_metadata=True
    )
    
    ranker = ResultRanker(config)
    
    # Simulate multiple ranking operations
    results = [
        SimilarityResult("test_001", 0.9, 0.8, "Tank", {}),
        SimilarityResult("test_002", 0.7, 0.7, "Vehicle", {}),
        SimilarityResult("test_003", 0.6, 0.6, "Building", {})
    ]
    
    # Perform multiple rankings to build history
    for i in range(5):
        ranked_results, metrics = ranker.rank_results(results.copy())
    
    # Get performance statistics
    stats = ranker.get_ranking_statistics()
    
    print("Ranking performance statistics:")
    for metric, value in stats.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.3f}")
        else:
            print(f"  {metric}: {value}")
    print()


def main():
    """Run all examples."""
    print("Advanced Ranking and Confidence Scoring Examples")
    print("=" * 55)
    print()
    
    example_1_basic_ranking()
    example_2_enhanced_confidence()
    example_3_integrated_query_processing()
    example_4_custom_ranking_strategy()
    example_5_confidence_calibration()
    example_6_performance_monitoring()
    
    print("All examples completed!")
    print("\nFor more information, see:")
    print("- src/query/ranker.py for ResultRanker documentation")
    print("- src/query/confidence.py for ConfidenceCalculator documentation")
    print("- scripts/test_ranking_confidence.py for comprehensive tests")


if __name__ == "__main__":
    main()
