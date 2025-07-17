#!/usr/bin/env python3
"""
Comprehensive test script for ResultRanker and ConfidenceCalculator.

This script tests the advanced ranking and confidence scoring features
implemented in Task 8.2 of the IR Image Classification System.
"""

import sys
import os
import time
import logging
from datetime import datetime
from typing import List

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.models.data_models import SimilarityResult, Embedding
from src.query.ranker import ResultRanker, RankingConfig, RankingStrategy
from src.query.confidence import ConfidenceCalculator, ConfidenceConfig, ConfidenceStrategy

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_mock_similarity_results() -> List[SimilarityResult]:
    """Create mock similarity results for testing."""
    mock_results = [
        SimilarityResult(
            image_id="tank_001",
            similarity_score=0.92,
            confidence=0.85,
            object_class="M-1A1 Abrams Tank",
            metadata={"file_path": "data/processed/M-1A1 Abrams Tank/image_001.png"}
        ),
        SimilarityResult(
            image_id="tank_002", 
            similarity_score=0.88,
            confidence=0.78,
            object_class="Challenger Tank",
            metadata={"file_path": "data/processed/Challenger Tank/image_002.png"}
        ),
        SimilarityResult(
            image_id="vehicle_001",
            similarity_score=0.75,
            confidence=0.65,
            object_class="Humvee",
            metadata={"file_path": "data/processed/Humvee/image_003.png"}
        ),
        SimilarityResult(
            image_id="missile_001",
            similarity_score=0.82,
            confidence=0.88,
            object_class="BGM-109 Tomahawk Cruise",
            metadata={"file_path": "data/processed/BGM-109 Tomahawk Cruise/image_004.png"}
        ),
        SimilarityResult(
            image_id="building_001",
            similarity_score=0.58,
            confidence=0.45,
            object_class="Building",
            metadata={"file_path": "data/processed/Building/image_005.png"}
        )
    ]
    
    return mock_results


def create_mock_query_embedding() -> Embedding:
    """Create a mock query embedding for testing."""
    # Create a realistic embedding vector (512 dimensions)
    embedding_vector = np.random.normal(0, 1, 512).astype(np.float32)
    embedding_vector = embedding_vector / np.linalg.norm(embedding_vector)  # Normalize
    
    return Embedding(
        id="query_embedding_001",
        vector=embedding_vector,
        image_id="query_image",
        model_version="resnet50_v1.0"
    )


def test_result_ranker():
    """Test the ResultRanker with different strategies."""
    logger.info("Testing ResultRanker with different strategies...")
    
    # Create test data
    results = create_mock_similarity_results()
    
    # Test different ranking strategies
    strategies = [
        RankingStrategy.SIMILARITY_ONLY,
        RankingStrategy.CONFIDENCE_WEIGHTED,
        RankingStrategy.HYBRID_SCORE,
        RankingStrategy.MILITARY_PRIORITY
    ]
    
    for strategy in strategies:
        logger.info(f"\n--- Testing {strategy.value} strategy ---")
        
        config = RankingConfig(
            strategy=strategy,
            min_confidence=0.5,
            max_results=5,
            enable_diversity_filtering=False
        )
        
        ranker = ResultRanker(config)
        ranked_results, metrics = ranker.rank_results(results.copy())
        
        logger.info(f"Strategy: {strategy.value}")
        logger.info(f"Total candidates: {metrics.total_candidates}")
        logger.info(f"Final results: {metrics.final_results}")
        logger.info(f"Average confidence: {metrics.average_confidence:.3f}")
        logger.info(f"Ranking time: {metrics.ranking_time_ms:.1f}ms")
        
        logger.info("Top 3 ranked results:")
        for i, result in enumerate(ranked_results[:3]):
            logger.info(f"  {i+1}. {result.object_class} - "
                       f"Similarity: {result.similarity_score:.3f}, "
                       f"Confidence: {result.confidence:.3f}")
    
    logger.info("ResultRanker testing completed successfully!")


def test_confidence_calculator():
    """Test the ConfidenceCalculator with different strategies."""
    logger.info("\nTesting ConfidenceCalculator with different strategies...")
    
    # Create test data
    results = create_mock_similarity_results()
    query_embedding = create_mock_query_embedding()
    
    # Test different confidence strategies
    strategies = [
        ConfidenceStrategy.SIMILARITY_BASED,
        ConfidenceStrategy.STATISTICAL,
        ConfidenceStrategy.ENSEMBLE,
        ConfidenceStrategy.MILITARY_CALIBRATED
    ]
    
    for strategy in strategies:
        logger.info(f"\n--- Testing {strategy.value} strategy ---")
        
        config = ConfidenceConfig(
            strategy=strategy,
            min_confidence=0.0,
            max_confidence=1.0,
            enable_adaptive_calibration=True
        )
        
        calculator = ConfidenceCalculator(config)
        
        # Test batch confidence calculation
        analyses = calculator.calculate_batch_confidence(results.copy(), query_embedding)
        
        logger.info(f"Strategy: {strategy.value}")
        logger.info("Confidence analyses:")
        
        for result, analysis in zip(results, analyses):
            logger.info(f"  {result.object_class}:")
            logger.info(f"    Original confidence: {result.confidence:.3f}")
            logger.info(f"    Enhanced confidence: {analysis.final_confidence:.3f}")
            logger.info(f"    Uncertainty: {analysis.uncertainty_estimate:.3f}")
            logger.info(f"    Explanation: {analysis.explanation}")
    
    logger.info("ConfidenceCalculator testing completed successfully!")


def test_integrated_ranking_and_confidence():
    """Test integrated ranking and confidence scoring."""
    logger.info("\nTesting integrated ranking and confidence scoring...")
    
    # Create test data
    results = create_mock_similarity_results()
    query_embedding = create_mock_query_embedding()
    
    # Configure components for military deployment
    ranking_config = RankingConfig(
        strategy=RankingStrategy.MILITARY_PRIORITY,
        min_confidence=0.6,
        max_results=5,
        boost_critical_objects=1.3,
        enable_diversity_filtering=True
    )
    
    confidence_config = ConfidenceConfig(
        strategy=ConfidenceStrategy.MILITARY_CALIBRATED,
        min_confidence=0.0,
        max_confidence=1.0,
        enable_adaptive_calibration=True,
        enable_uncertainty_quantification=True
    )
    
    # Initialize components
    ranker = ResultRanker(ranking_config)
    calculator = ConfidenceCalculator(confidence_config)
    
    # Step 1: Enhanced confidence calculation
    logger.info("Step 1: Enhanced confidence calculation")
    confidence_analyses = calculator.calculate_batch_confidence(results, query_embedding)
    
    # Update results with enhanced confidence scores
    for result, analysis in zip(results, confidence_analyses):
        result.confidence = analysis.final_confidence
        result.metadata.update({
            'confidence_explanation': analysis.explanation,
            'confidence_factors': analysis.confidence_factors,
            'uncertainty_estimate': analysis.uncertainty_estimate
        })
    
    # Step 2: Military-priority ranking
    logger.info("Step 2: Military-priority ranking")
    query_context = {
        'query_embedding': query_embedding,
        'total_candidates': len(results)
    }
    
    ranked_results, ranking_metrics = ranker.rank_results(results, query_context)
    
    # Display final results
    logger.info(f"\nFinal integrated results (Top {len(ranked_results)}):")
    logger.info(f"Total processing - Ranking time: {ranking_metrics.ranking_time_ms:.1f}ms")
    logger.info(f"Average confidence: {ranking_metrics.average_confidence:.3f}")
    
    for i, result in enumerate(ranked_results):
        logger.info(f"\n{i+1}. {result.object_class}")
        logger.info(f"   Similarity: {result.similarity_score:.3f}")
        logger.info(f"   Enhanced Confidence: {result.confidence:.3f}")
        logger.info(f"   Uncertainty: {result.metadata.get('uncertainty_estimate', 'N/A'):.3f}")
        logger.info(f"   Explanation: {result.metadata.get('confidence_explanation', 'N/A')}")
        
        # Show confidence factors if available
        if 'confidence_factors' in result.metadata:
            factors = result.metadata['confidence_factors']
            logger.info(f"   Confidence factors: {factors}")
    
    logger.info("\nIntegrated testing completed successfully!")


def test_performance_benchmarks():
    """Test performance requirements compliance."""
    logger.info("\nTesting performance benchmarks...")
    
    # Create larger test dataset
    large_results = []
    for i in range(100):  # 100 results to simulate realistic search
        result = SimilarityResult(
            image_id=f"test_{i:03d}",
            similarity_score=np.random.uniform(0.4, 0.95),
            confidence=np.random.uniform(0.3, 0.9),
            object_class=np.random.choice([
                "Tank", "Aircraft", "Vehicle", "Building", "Missile", "Personnel"
            ]),
            metadata={"synthetic": True}
        )
        large_results.append(result)
    
    query_embedding = create_mock_query_embedding()
    
    # Test performance with default configuration
    config = RankingConfig(strategy=RankingStrategy.HYBRID_SCORE)
    ranker = ResultRanker(config)
    
    # Measure ranking performance
    start_time = datetime.now()
    ranked_results, metrics = ranker.rank_results(large_results, {'query_embedding': query_embedding})
    ranking_time = (datetime.now() - start_time).total_seconds() * 1000
    
    logger.info(f"Performance test results:")
    logger.info(f"  Input results: {len(large_results)}")
    logger.info(f"  Output results: {len(ranked_results)}")
    logger.info(f"  Ranking time: {ranking_time:.1f}ms")
    logger.info(f"  Results per second: {len(large_results) / (ranking_time / 1000):.0f}")
    
    # Test confidence calculation performance
    confidence_config = ConfidenceConfig(strategy=ConfidenceStrategy.ENSEMBLE)
    calculator = ConfidenceCalculator(confidence_config)
    
    start_time = datetime.now()
    analyses = calculator.calculate_batch_confidence(large_results[:10], query_embedding)
    confidence_time = (datetime.now() - start_time).total_seconds() * 1000
    
    logger.info(f"  Confidence calculation time (10 results): {confidence_time:.1f}ms")
    logger.info(f"  Average time per confidence calculation: {confidence_time / 10:.1f}ms")
    
    # Check if performance meets requirements (should be much faster than 2 seconds)
    if ranking_time < 100:  # 100ms threshold for ranking
        logger.info("✅ Ranking performance meets requirements")
    else:
        logger.warning("⚠️ Ranking performance may be slow")
    
    if confidence_time < 50:  # 50ms for 10 calculations
        logger.info("✅ Confidence calculation performance meets requirements")
    else:
        logger.warning("⚠️ Confidence calculation performance may be slow")


def main():
    """Main test function."""
    logger.info("Starting advanced ranking and confidence scoring tests...")
    logger.info("=" * 70)
    
    try:
        # Test individual components
        test_result_ranker()
        test_confidence_calculator()
        
        # Test integrated functionality
        test_integrated_ranking_and_confidence()
        
        # Test performance
        test_performance_benchmarks()
        
        logger.info("\n" + "=" * 70)
        logger.info("🎉 All tests completed successfully!")
        logger.info("Task 8.2 - Result ranking and confidence scoring implementation is ready!")
        
    except Exception as e:
        logger.error(f"❌ Test failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
