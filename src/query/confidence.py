"""
Confidence scoring implementation for IR Image Classification System.

This module provides the ConfidenceCalculator class for computing sophisticated
confidence scores for similarity search results. Implements multiple confidence
estimation strategies optimized for military IR image classification.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
from scipy import stats

from ..models.data_models import SimilarityResult, Embedding


# Configure logging
logger = logging.getLogger(__name__)


class ConfidenceStrategy(Enum):
    """Confidence calculation strategies."""
    SIMILARITY_BASED = "similarity_based"           # Based on similarity score distribution
    STATISTICAL = "statistical"                     # Statistical confidence intervals
    ENSEMBLE = "ensemble"                          # Multiple confidence metrics
    MILITARY_CALIBRATED = "military_calibrated"    # Military-specific calibration


class ConfidenceFactors(Enum):
    """Factors that influence confidence scoring."""
    SIMILARITY_SCORE = "similarity_score"
    SCORE_DISTRIBUTION = "score_distribution"
    EMBEDDING_QUALITY = "embedding_quality"
    HISTORICAL_ACCURACY = "historical_accuracy"
    OBJECT_CLASS_RELIABILITY = "object_class_reliability"
    SEARCH_CONTEXT = "search_context"


@dataclass
class ConfidenceConfig:
    """Configuration for confidence calculation."""
    
    # Confidence strategy
    strategy: ConfidenceStrategy = ConfidenceStrategy.ENSEMBLE
    
    # Base confidence parameters
    min_confidence: float = 0.0
    max_confidence: float = 1.0
    confidence_smoothing: float = 0.1  # Smoothing factor for confidence scores
    
    # Similarity-based confidence
    similarity_weight: float = 0.5
    similarity_threshold_high: float = 0.8
    similarity_threshold_low: float = 0.5
    
    # Statistical confidence
    confidence_interval: float = 0.95  # 95% confidence interval
    min_samples_for_stats: int = 10
    
    # Ensemble weights
    ensemble_weights: Dict[str, float] = field(default_factory=lambda: {
        'similarity': 0.4,
        'distribution': 0.3,
        'quality': 0.2,
        'historical': 0.1
    })
    
    # Military-specific parameters
    critical_similarity_threshold: float = 0.9  # High confidence threshold for critical objects
    uncertainty_penalty: float = 0.1  # Penalty for uncertain classifications
    
    # Advanced options
    enable_adaptive_calibration: bool = True
    enable_uncertainty_quantification: bool = True
    enable_confidence_explanation: bool = True
    
    def validate(self) -> bool:
        """Validate confidence configuration."""
        if not 0.0 <= self.min_confidence <= self.max_confidence <= 1.0:
            raise ValueError("Confidence bounds must be between 0.0 and 1.0")
        
        if not 0.0 <= self.confidence_interval <= 1.0:
            raise ValueError("Confidence interval must be between 0.0 and 1.0")
        
        # Validate ensemble weights sum to 1.0
        if abs(sum(self.ensemble_weights.values()) - 1.0) > 0.01:
            raise ValueError("Ensemble weights must sum to 1.0")
        
        return True


@dataclass
class ConfidenceAnalysis:
    """Detailed analysis of confidence computation."""
    final_confidence: float
    confidence_factors: Dict[str, float]
    uncertainty_estimate: float
    explanation: str
    calculation_method: ConfidenceStrategy
    computation_time_ms: float


@dataclass
class ConfidenceMetrics:
    """Metrics for confidence calculation performance."""
    total_calculations: int
    average_confidence: float
    confidence_distribution: Dict[str, int]
    average_computation_time_ms: float
    calibration_accuracy: Optional[float] = None


class ConfidenceCalculator:
    """
    Advanced confidence scoring for IR image similarity results.
    
    Provides multiple confidence calculation strategies:
    - Similarity-based confidence scoring
    - Statistical confidence intervals
    - Ensemble multi-factor confidence
    - Military-calibrated confidence scoring
    """
    
    def __init__(self, config: Optional[ConfidenceConfig] = None):
        """
        Initialize the ConfidenceCalculator.
        
        Args:
            config: Confidence calculation configuration
        """
        self.config = config or ConfidenceConfig()
        self.config.validate()
        
        # Historical data for adaptive calibration
        self.historical_results: List[Tuple[float, float]] = []  # (predicted_confidence, actual_accuracy)
        self.class_reliability: Dict[str, float] = {}  # Per-class reliability scores
        self.calculation_history: List[ConfidenceMetrics] = []
        
        # Initialize confidence calculation strategies
        self._confidence_strategies = {
            ConfidenceStrategy.SIMILARITY_BASED: self._calculate_similarity_confidence,
            ConfidenceStrategy.STATISTICAL: self._calculate_statistical_confidence,
            ConfidenceStrategy.ENSEMBLE: self._calculate_ensemble_confidence,
            ConfidenceStrategy.MILITARY_CALIBRATED: self._calculate_military_confidence
        }
        
        logger.info(f"ConfidenceCalculator initialized with strategy: {self.config.strategy}")
    
    def calculate_confidence(self, 
                           result: SimilarityResult,
                           all_results: Optional[List[SimilarityResult]] = None,
                           query_embedding: Optional[Embedding] = None,
                           context: Optional[Dict[str, Any]] = None) -> ConfidenceAnalysis:
        """
        Calculate confidence score for a similarity result.
        
        Args:
            result: The similarity result to score
            all_results: All results for context analysis
            query_embedding: Original query embedding for quality assessment
            context: Additional context for confidence calculation
            
        Returns:
            ConfidenceAnalysis: Detailed confidence analysis
        """
        start_time = datetime.now()
        context = context or {}
        
        # Select and apply confidence calculation strategy
        strategy_func = self._confidence_strategies[self.config.strategy]
        confidence_analysis = strategy_func(result, all_results, query_embedding, context)
        
        # Apply bounds and smoothing
        confidence_analysis.final_confidence = self._apply_confidence_bounds(
            confidence_analysis.final_confidence
        )
        
        # Add computation time
        computation_time = (datetime.now() - start_time).total_seconds() * 1000
        confidence_analysis.computation_time_ms = computation_time
        
        logger.debug(f"Confidence calculated: {confidence_analysis.final_confidence:.3f} "
                    f"for {result.object_class} in {computation_time:.1f}ms")
        
        return confidence_analysis
    
    def calculate_batch_confidence(self, 
                                 results: List[SimilarityResult],
                                 query_embedding: Optional[Embedding] = None,
                                 context: Optional[Dict[str, Any]] = None) -> List[ConfidenceAnalysis]:
        """
        Calculate confidence scores for multiple results efficiently.
        
        Args:
            results: List of similarity results to score
            query_embedding: Original query embedding
            context: Additional context information
            
        Returns:
            List[ConfidenceAnalysis]: Confidence analyses for all results
        """
        analyses = []
        
        for result in results:
            analysis = self.calculate_confidence(
                result, results, query_embedding, context
            )
            analyses.append(analysis)
        
        logger.debug(f"Batch confidence calculation completed for {len(results)} results")
        return analyses
    
    def _calculate_similarity_confidence(self, 
                                       result: SimilarityResult,
                                       all_results: Optional[List[SimilarityResult]],
                                       query_embedding: Optional[Embedding],
                                       context: Dict[str, Any]) -> ConfidenceAnalysis:
        """Calculate confidence based primarily on similarity score."""
        similarity = result.similarity_score
        
        # Base confidence from similarity score with sigmoid transformation
        base_confidence = 1 / (1 + np.exp(-10 * (similarity - 0.5)))
        
        # Adjust based on similarity thresholds
        if similarity >= self.config.similarity_threshold_high:
            confidence_boost = 0.1
        elif similarity <= self.config.similarity_threshold_low:
            confidence_boost = -0.2
        else:
            confidence_boost = 0.0
        
        final_confidence = base_confidence + confidence_boost
        
        factors = {
            'similarity_score': similarity,
            'base_confidence': base_confidence,
            'threshold_adjustment': confidence_boost
        }
        
        explanation = f"Confidence based on similarity score of {similarity:.3f}"
        
        return ConfidenceAnalysis(
            final_confidence=final_confidence,
            confidence_factors=factors,
            uncertainty_estimate=1 - final_confidence,
            explanation=explanation,
            calculation_method=ConfidenceStrategy.SIMILARITY_BASED,
            computation_time_ms=0.0  # Will be set by caller
        )
    
    def _calculate_statistical_confidence(self, 
                                        result: SimilarityResult,
                                        all_results: Optional[List[SimilarityResult]],
                                        query_embedding: Optional[Embedding],
                                        context: Dict[str, Any]) -> ConfidenceAnalysis:
        """Calculate confidence using statistical methods."""
        if not all_results or len(all_results) < self.config.min_samples_for_stats:
            # Fallback to similarity-based if insufficient data
            return self._calculate_similarity_confidence(result, all_results, query_embedding, context)
        
        # Extract similarity scores for statistical analysis
        similarities = [r.similarity_score for r in all_results]
        
        # Calculate statistical measures
        mean_similarity = float(np.mean(similarities))
        std_similarity = float(np.std(similarities))
        
        # Z-score for this result
        z_score = (result.similarity_score - mean_similarity) / max(float(std_similarity), 0.01)
        
        # Convert z-score to confidence using cumulative distribution
        statistical_confidence = float(stats.norm.cdf(z_score))
        
        # Confidence interval estimation
        confidence_interval = stats.norm.interval(
            self.config.confidence_interval,
            loc=result.similarity_score,
            scale=std_similarity
        )
        
        # Calculate uncertainty from confidence interval width
        uncertainty = (confidence_interval[1] - confidence_interval[0]) / 2
        
        factors = {
            'z_score': z_score,
            'mean_similarity': mean_similarity,
            'std_similarity': std_similarity,
            'confidence_interval_width': uncertainty * 2
        }
        
        explanation = f"Statistical confidence based on z-score {z_score:.2f} in distribution"
        
        return ConfidenceAnalysis(
            final_confidence=statistical_confidence,
            confidence_factors=factors,
            uncertainty_estimate=uncertainty,
            explanation=explanation,
            calculation_method=ConfidenceStrategy.STATISTICAL,
            computation_time_ms=0.0
        )
    
    def _calculate_ensemble_confidence(self, 
                                     result: SimilarityResult,
                                     all_results: Optional[List[SimilarityResult]],
                                     query_embedding: Optional[Embedding],
                                     context: Dict[str, Any]) -> ConfidenceAnalysis:
        """Calculate confidence using ensemble of multiple factors."""
        factors = {}
        
        # Factor 1: Similarity-based confidence
        similarity_conf = self._get_similarity_factor(result.similarity_score)
        factors['similarity'] = similarity_conf
        
        # Factor 2: Score distribution analysis
        distribution_conf = self._get_distribution_factor(result, all_results)
        factors['distribution'] = distribution_conf
        
        # Factor 3: Embedding quality (if available)
        quality_conf = self._get_quality_factor(query_embedding, context)
        factors['quality'] = quality_conf
        
        # Factor 4: Historical accuracy (if available)
        historical_conf = self._get_historical_factor(result.object_class)
        factors['historical'] = historical_conf
        
        # Weighted ensemble combination
        weights = self.config.ensemble_weights
        ensemble_confidence = (
            weights['similarity'] * similarity_conf +
            weights['distribution'] * distribution_conf +
            weights['quality'] * quality_conf +
            weights['historical'] * historical_conf
        )
        
        # Calculate uncertainty as weighted variance
        uncertainty = self._calculate_ensemble_uncertainty(factors, weights)
        
        explanation = f"Ensemble confidence from {len(factors)} factors"
        
        return ConfidenceAnalysis(
            final_confidence=ensemble_confidence,
            confidence_factors=factors,
            uncertainty_estimate=uncertainty,
            explanation=explanation,
            calculation_method=ConfidenceStrategy.ENSEMBLE,
            computation_time_ms=0.0
        )
    
    def _calculate_military_confidence(self, 
                                     result: SimilarityResult,
                                     all_results: Optional[List[SimilarityResult]],
                                     query_embedding: Optional[Embedding],
                                     context: Dict[str, Any]) -> ConfidenceAnalysis:
        """Calculate confidence with military-specific calibration."""
        # Start with ensemble confidence as base
        base_analysis = self._calculate_ensemble_confidence(result, all_results, query_embedding, context)
        
        # Military-specific adjustments
        military_factors = {}
        
        # Critical object detection boost
        if self._is_critical_object(result.object_class):
            if result.similarity_score >= self.config.critical_similarity_threshold:
                critical_boost = 0.1
            else:
                critical_boost = -0.05  # Penalty for uncertain critical object detection
            military_factors['critical_object_boost'] = critical_boost
        else:
            critical_boost = 0.0
            military_factors['critical_object_boost'] = 0.0
        
        # Uncertainty penalty for ambiguous classifications
        uncertainty_penalty = 0.0
        if all_results and len(all_results) > 1:
            # Check if there are other high-scoring results (ambiguous)
            top_scores = sorted([r.similarity_score for r in all_results], reverse=True)
            if len(top_scores) > 1 and top_scores[1] > 0.7:
                score_gap = top_scores[0] - top_scores[1]
                if score_gap < 0.1:  # Very close scores indicate uncertainty
                    uncertainty_penalty = self.config.uncertainty_penalty
        
        military_factors['uncertainty_penalty'] = uncertainty_penalty
        
        # Combine base confidence with military adjustments
        military_confidence = (base_analysis.final_confidence + 
                             critical_boost - 
                             uncertainty_penalty)
        
        # Update factors
        all_factors = {**base_analysis.confidence_factors, **military_factors}
        
        explanation = f"Military-calibrated confidence with critical object and uncertainty analysis"
        
        return ConfidenceAnalysis(
            final_confidence=military_confidence,
            confidence_factors=all_factors,
            uncertainty_estimate=base_analysis.uncertainty_estimate,
            explanation=explanation,
            calculation_method=ConfidenceStrategy.MILITARY_CALIBRATED,
            computation_time_ms=0.0
        )
    
    def _get_similarity_factor(self, similarity: float) -> float:
        """Get confidence factor from similarity score."""
        # Sigmoid transformation for smooth confidence mapping
        return 1 / (1 + np.exp(-8 * (similarity - 0.6)))
    
    def _get_distribution_factor(self, 
                               result: SimilarityResult,
                               all_results: Optional[List[SimilarityResult]]) -> float:
        """Get confidence factor from score distribution."""
        if not all_results or len(all_results) < 2:
            return 0.5  # Neutral factor if no distribution available
        
        similarities = [r.similarity_score for r in all_results]
        max_sim = max(similarities)
        
        # If this result is clearly the best, higher confidence
        if result.similarity_score == max_sim:
            # Check gap to second-best
            second_best = sorted(similarities, reverse=True)[1]
            gap = max_sim - second_best
            return min(0.8 + gap, 1.0)
        else:
            # Lower confidence for non-top results
            return result.similarity_score / max_sim * 0.6
    
    def _get_quality_factor(self, 
                          query_embedding: Optional[Embedding],
                          context: Dict[str, Any]) -> float:
        """Get confidence factor from embedding quality."""
        if not query_embedding:
            return 0.5  # Neutral factor if no embedding available
        
        # Check embedding quality indicators
        embedding_norm = np.linalg.norm(query_embedding.vector)
        
        # Well-normalized embeddings typically have norm close to 1
        norm_quality = 1.0 - abs(embedding_norm - 1.0)
        
        # Check for any quality indicators in context
        image_quality = context.get('image_quality', 0.5)
        preprocessing_quality = context.get('preprocessing_quality', 0.5)
        
        # Combine quality factors
        overall_quality = (norm_quality + image_quality + preprocessing_quality) / 3
        return max(0.1, min(0.9, overall_quality))
    
    def _get_historical_factor(self, object_class: str) -> float:
        """Get confidence factor from historical accuracy."""
        if object_class in self.class_reliability:
            return self.class_reliability[object_class]
        else:
            return 0.5  # Neutral factor for unknown classes
    
    def _calculate_ensemble_uncertainty(self, 
                                      factors: Dict[str, float],
                                      weights: Dict[str, float]) -> float:
        """Calculate uncertainty for ensemble confidence."""
        # Variance-based uncertainty estimation
        factor_values = list(factors.values())
        if len(factor_values) < 2:
            return 0.5
        
        weighted_variance = float(np.var(factor_values))
        return min(weighted_variance, 0.5)  # Cap uncertainty at 0.5
    
    def _is_critical_object(self, object_class: str) -> bool:
        """Check if object class is considered critical for military applications."""
        critical_classes = [
            'Tank', 'Aircraft', 'Missile', 'Artillery', 'Radar',
            'AAV Tank', 'M-1A1 Abrams Tank', 'Challenger Tank',
            'BGM-109 Tomahawk Cruise', 'Patriot', 'Hawk Air Defense'
        ]
        return object_class in critical_classes
    
    def _apply_confidence_bounds(self, confidence: float) -> float:
        """Apply confidence bounds and smoothing."""
        # Apply bounds
        bounded_confidence = max(self.config.min_confidence, 
                               min(self.config.max_confidence, confidence))
        
        # Apply smoothing if enabled
        if self.config.confidence_smoothing > 0:
            # Smooth towards 0.5 (neutral confidence)
            smoothing = self.config.confidence_smoothing
            bounded_confidence = (1 - smoothing) * bounded_confidence + smoothing * 0.5
        
        return bounded_confidence
    
    def update_historical_accuracy(self, 
                                 predicted_confidence: float,
                                 actual_accuracy: float,
                                 object_class: str) -> None:
        """Update historical accuracy data for calibration."""
        # Update overall historical data
        self.historical_results.append((predicted_confidence, actual_accuracy))
        
        # Keep only recent data (last 1000 results)
        if len(self.historical_results) > 1000:
            self.historical_results.pop(0)
        
        # Update class-specific reliability
        if object_class not in self.class_reliability:
            self.class_reliability[object_class] = actual_accuracy
        else:
            # Exponential moving average
            alpha = 0.1
            self.class_reliability[object_class] = (
                alpha * actual_accuracy + 
                (1 - alpha) * self.class_reliability[object_class]
            )
    
    def get_calibration_metrics(self) -> Dict[str, float]:
        """Get confidence calibration metrics."""
        if len(self.historical_results) < 10:
            return {}
        
        predicted, actual = zip(*self.historical_results)
        
        # Calculate calibration error (reliability diagram)
        bins = np.linspace(0, 1, 11)
        bin_indices = np.digitize(predicted, bins) - 1
        
        calibration_error = 0.0
        total_samples = 0
        
        for i in range(len(bins) - 1):
            mask = bin_indices == i
            if np.sum(mask) > 0:
                bin_predicted = np.mean([predicted[j] for j in range(len(predicted)) if mask[j]])
                bin_actual = np.mean([actual[j] for j in range(len(actual)) if mask[j]])
                bin_size = np.sum(mask)
                
                calibration_error += bin_size * abs(bin_predicted - bin_actual)
                total_samples += bin_size
        
        calibration_error = calibration_error / total_samples if total_samples > 0 else 0.0
        
        return {
            'calibration_error': calibration_error,
            'total_samples': len(self.historical_results),
            'average_predicted_confidence': float(np.mean(predicted)),
            'average_actual_accuracy': float(np.mean(actual))
        }
    
    def update_config(self, new_config: ConfidenceConfig) -> None:
        """Update confidence calculation configuration."""
        new_config.validate()
        self.config = new_config
        logger.info(f"Confidence configuration updated to strategy: {self.config.strategy}")
    
    def reset_historical_data(self) -> None:
        """Reset historical accuracy data."""
        self.historical_results.clear()
        self.class_reliability.clear()
        self.calculation_history.clear()
        logger.info("Historical confidence data reset")
