"""
Result ranking implementation for IR Image Classification System.

This module provides the ResultRanker class for intelligent ranking and
filtering of similarity search results. Implements multiple ranking strategies
optimized for military IR image classification.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional, Callable, Tuple
import numpy as np

from ..models.data_models import SimilarityResult


# Configure logging
logger = logging.getLogger(__name__)


class RankingStrategy(Enum):
    """Ranking strategies for similarity results."""
    SIMILARITY_ONLY = "similarity_only"           # Rank by similarity score only
    CONFIDENCE_WEIGHTED = "confidence_weighted"   # Weight similarity by confidence
    HYBRID_SCORE = "hybrid_score"                # Combine multiple factors
    MILITARY_PRIORITY = "military_priority"      # Military-specific ranking


class ConfidenceLevel(Enum):
    """Confidence level categories."""
    VERY_HIGH = "very_high"    # >= 0.9
    HIGH = "high"              # >= 0.8
    MEDIUM = "medium"          # >= 0.7
    LOW = "low"                # >= 0.6
    VERY_LOW = "very_low"      # < 0.6


@dataclass
class RankingConfig:
    """Configuration for result ranking behavior."""
    
    # Ranking strategy
    strategy: RankingStrategy = RankingStrategy.HYBRID_SCORE
    
    # Filtering thresholds
    min_confidence: float = 0.7
    min_similarity: float = 0.5
    max_results: int = 5
    
    # Weighting factors for hybrid scoring
    similarity_weight: float = 0.7
    confidence_weight: float = 0.3
    diversity_weight: float = 0.0  # For promoting diverse results
    
    # Military-specific parameters
    critical_object_classes: List[str] = field(default_factory=lambda: [
        "Tank", "Aircraft", "Vehicle", "Missile", "Artillery"
    ])
    boost_critical_objects: float = 1.2  # Boost factor for critical objects
    
    # Advanced filtering
    enable_diversity_filtering: bool = False
    diversity_threshold: float = 0.9  # Similarity threshold for diversity
    enable_confidence_calibration: bool = True
    
    # Metadata enrichment
    add_ranking_metadata: bool = True
    add_confidence_explanations: bool = True
    
    def validate(self) -> bool:
        """Validate ranking configuration."""
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0.0 and 1.0")
        
        if not 0.0 <= self.min_similarity <= 1.0:
            raise ValueError("min_similarity must be between 0.0 and 1.0")
        
        if self.max_results <= 0:
            raise ValueError("max_results must be positive")
        
        if not 0.0 <= self.similarity_weight <= 1.0:
            raise ValueError("similarity_weight must be between 0.0 and 1.0")
        
        if not 0.0 <= self.confidence_weight <= 1.0:
            raise ValueError("confidence_weight must be between 0.0 and 1.0")
        
        return True


@dataclass
class RankingMetrics:
    """Metrics for ranking performance analysis."""
    total_candidates: int
    filtered_results: int
    final_results: int
    ranking_time_ms: float
    average_confidence: float
    average_similarity: float
    confidence_distribution: Dict[str, int] = field(default_factory=dict)
    ranking_strategy_used: RankingStrategy = RankingStrategy.SIMILARITY_ONLY


class ResultRanker:
    """
    Advanced result ranking and filtering for IR image similarity search.
    
    Provides multiple ranking strategies optimized for military applications:
    - Similarity-based ranking
    - Confidence-weighted scoring
    - Hybrid multi-factor scoring
    - Military-priority ranking
    """
    
    def __init__(self, config: Optional[RankingConfig] = None):
        """
        Initialize the ResultRanker.
        
        Args:
            config: Ranking configuration (defaults to standard config)
        """
        self.config = config or RankingConfig()
        self.config.validate()
        
        self.ranking_history: List[RankingMetrics] = []
        
        # Initialize ranking strategies
        self._ranking_strategies = {
            RankingStrategy.SIMILARITY_ONLY: self._rank_by_similarity,
            RankingStrategy.CONFIDENCE_WEIGHTED: self._rank_by_confidence_weighted,
            RankingStrategy.HYBRID_SCORE: self._rank_by_hybrid_score,
            RankingStrategy.MILITARY_PRIORITY: self._rank_by_military_priority
        }
        
        logger.info(f"ResultRanker initialized with strategy: {self.config.strategy}")
    
    def rank_results(self, 
                    results: List[SimilarityResult],
                    query_context: Optional[Dict[str, Any]] = None) -> Tuple[List[SimilarityResult], RankingMetrics]:
        """
        Rank and filter similarity results according to configured strategy.
        
        Args:
            results: Raw similarity results to rank
            query_context: Additional context for ranking decisions
            
        Returns:
            Tuple[List[SimilarityResult], RankingMetrics]: Ranked results and metrics
        """
        start_time = datetime.now()
        query_context = query_context or {}
        
        logger.debug(f"Ranking {len(results)} results using strategy: {self.config.strategy}")
        
        # Initial filtering
        filtered_results = self._apply_initial_filters(results)
        
        # Apply ranking strategy
        ranking_func = self._ranking_strategies[self.config.strategy]
        ranked_results = ranking_func(filtered_results, query_context)
        
        # Apply diversity filtering if enabled
        if self.config.enable_diversity_filtering:
            ranked_results = self._apply_diversity_filtering(ranked_results)
        
        # Limit to max results
        final_results = ranked_results[:self.config.max_results]
        
        # Enhance with metadata
        if self.config.add_ranking_metadata:
            final_results = self._add_ranking_metadata(final_results)
        
        # Calculate metrics
        ranking_time = (datetime.now() - start_time).total_seconds() * 1000
        metrics = self._calculate_ranking_metrics(
            len(results), len(filtered_results), len(final_results), 
            ranking_time, final_results
        )
        
        self.ranking_history.append(metrics)
        
        logger.debug(f"Ranking complete: {len(final_results)} final results in {ranking_time:.1f}ms")
        return final_results, metrics
    
    def _apply_initial_filters(self, results: List[SimilarityResult]) -> List[SimilarityResult]:
        """Apply basic confidence and similarity thresholds."""
        filtered = []
        
        for result in results:
            # Check confidence threshold
            if result.confidence < self.config.min_confidence:
                continue
            
            # Check similarity threshold
            if result.similarity_score < self.config.min_similarity:
                continue
            
            filtered.append(result)
        
        logger.debug(f"Initial filtering: {len(results)} -> {len(filtered)} results")
        return filtered
    
    def _rank_by_similarity(self, 
                           results: List[SimilarityResult],
                           context: Dict[str, Any]) -> List[SimilarityResult]:
        """Rank results by similarity score only."""
        return sorted(results, key=lambda x: x.similarity_score, reverse=True)
    
    def _rank_by_confidence_weighted(self, 
                                   results: List[SimilarityResult],
                                   context: Dict[str, Any]) -> List[SimilarityResult]:
        """Rank results by confidence-weighted similarity score."""
        def weighted_score(result: SimilarityResult) -> float:
            return (result.similarity_score * self.config.similarity_weight + 
                   result.confidence * self.config.confidence_weight)
        
        return sorted(results, key=weighted_score, reverse=True)
    
    def _rank_by_hybrid_score(self, 
                             results: List[SimilarityResult],
                             context: Dict[str, Any]) -> List[SimilarityResult]:
        """Rank results using hybrid multi-factor scoring."""
        def hybrid_score(result: SimilarityResult) -> float:
            # Base score from similarity and confidence
            base_score = (result.similarity_score * self.config.similarity_weight + 
                         result.confidence * self.config.confidence_weight)
            
            # Apply confidence calibration
            if self.config.enable_confidence_calibration:
                confidence_boost = self._calculate_confidence_boost(result.confidence)
                base_score *= confidence_boost
            
            # Apply object class boosting
            if result.object_class in self.config.critical_object_classes:
                base_score *= self.config.boost_critical_objects
            
            return base_score
        
        return sorted(results, key=hybrid_score, reverse=True)
    
    def _rank_by_military_priority(self, 
                                  results: List[SimilarityResult],
                                  context: Dict[str, Any]) -> List[SimilarityResult]:
        """Rank results with military-specific priority rules."""
        # Define military priority tiers
        priority_tiers = {
            'critical': ['Tank', 'Missile', 'Aircraft', 'Artillery'],
            'high': ['Vehicle', 'APC', 'Aircraft', 'Radar'],
            'medium': ['Building', 'Infrastructure', 'Personnel'],
            'low': ['Unknown', 'Other']
        }
        
        def military_priority_score(result: SimilarityResult) -> Tuple[int, float]:
            # Determine priority tier
            priority_level = 3  # low priority by default
            for tier_level, classes in enumerate(priority_tiers.values()):
                if result.object_class in classes:
                    priority_level = tier_level
                    break
            
            # Calculate weighted score within tier
            weighted_score = (result.similarity_score * 0.8 + result.confidence * 0.2)
            
            # Return as tuple (priority, score) for sorting
            return (priority_level, weighted_score)
        
        return sorted(results, key=military_priority_score, reverse=True)
    
    def _apply_diversity_filtering(self, results: List[SimilarityResult]) -> List[SimilarityResult]:
        """Remove very similar results to promote diversity."""
        if len(results) <= 1:
            return results
        
        diverse_results = [results[0]]  # Always include top result
        
        for result in results[1:]:
            # Check if this result is too similar to existing ones
            is_diverse = True
            for existing in diverse_results:
                if result.similarity_score > self.config.diversity_threshold:
                    # Check if object classes are the same (indicating potential duplicates)
                    if result.object_class == existing.object_class:
                        is_diverse = False
                        break
            
            if is_diverse:
                diverse_results.append(result)
        
        logger.debug(f"Diversity filtering: {len(results)} -> {len(diverse_results)} results")
        return diverse_results
    
    def _calculate_confidence_boost(self, confidence: float) -> float:
        """Calculate confidence-based score boost."""
        if confidence >= 0.9:
            return 1.1  # 10% boost for very high confidence
        elif confidence >= 0.8:
            return 1.05  # 5% boost for high confidence
        elif confidence >= 0.7:
            return 1.0   # No boost for medium confidence
        else:
            return 0.95  # 5% penalty for low confidence
    
    def _add_ranking_metadata(self, results: List[SimilarityResult]) -> List[SimilarityResult]:
        """Add ranking-specific metadata to results."""
        for i, result in enumerate(results):
            # Add ranking information
            result.metadata.update({
                'rank': i + 1,
                'ranking_strategy': self.config.strategy.value,
                'confidence_level': self._categorize_confidence(result.confidence),
                'ranking_timestamp': datetime.now().isoformat()
            })
            
            # Add confidence explanation if enabled
            if self.config.add_confidence_explanations:
                result.metadata['confidence_explanation'] = self._generate_confidence_explanation(result)
        
        return results
    
    def _categorize_confidence(self, confidence: float) -> str:
        """Categorize confidence score into human-readable levels."""
        if confidence >= 0.9:
            return ConfidenceLevel.VERY_HIGH.value
        elif confidence >= 0.8:
            return ConfidenceLevel.HIGH.value
        elif confidence >= 0.7:
            return ConfidenceLevel.MEDIUM.value
        elif confidence >= 0.6:
            return ConfidenceLevel.LOW.value
        else:
            return ConfidenceLevel.VERY_LOW.value
    
    def _generate_confidence_explanation(self, result: SimilarityResult) -> str:
        """Generate human-readable confidence explanation."""
        confidence_level = self._categorize_confidence(result.confidence)
        similarity_desc = "high" if result.similarity_score > 0.8 else "moderate" if result.similarity_score > 0.6 else "low"
        
        explanations = {
            ConfidenceLevel.VERY_HIGH.value: f"Very high confidence match with {similarity_desc} similarity to known {result.object_class}",
            ConfidenceLevel.HIGH.value: f"High confidence match with {similarity_desc} similarity",
            ConfidenceLevel.MEDIUM.value: f"Moderate confidence match - consider additional verification",
            ConfidenceLevel.LOW.value: f"Low confidence match - manual verification recommended",
            ConfidenceLevel.VERY_LOW.value: f"Very low confidence - result may not be reliable"
        }
        
        return explanations.get(confidence_level, "Unknown confidence level")
    
    def _calculate_ranking_metrics(self, 
                                  total_candidates: int,
                                  filtered_results: int,
                                  final_results: int,
                                  ranking_time: float,
                                  results: List[SimilarityResult]) -> RankingMetrics:
        """Calculate metrics for ranking performance analysis."""
        avg_confidence = float(np.mean([r.confidence for r in results])) if results else 0.0
        avg_similarity = float(np.mean([r.similarity_score for r in results])) if results else 0.0
        
        # Calculate confidence distribution
        confidence_dist = {level.value: 0 for level in ConfidenceLevel}
        for result in results:
            level = self._categorize_confidence(result.confidence)
            confidence_dist[level] += 1
        
        return RankingMetrics(
            total_candidates=total_candidates,
            filtered_results=filtered_results,
            final_results=final_results,
            ranking_time_ms=ranking_time,
            average_confidence=avg_confidence,
            average_similarity=avg_similarity,
            confidence_distribution=confidence_dist,
            ranking_strategy_used=self.config.strategy
        )
    
    def get_ranking_statistics(self) -> Dict[str, Any]:
        """Get statistics about ranking performance over time."""
        if not self.ranking_history:
            return {}
        
        recent_metrics = self.ranking_history[-100:]  # Last 100 rankings
        
        stats = {
            'total_rankings': len(self.ranking_history),
            'average_ranking_time_ms': np.mean([m.ranking_time_ms for m in recent_metrics]),
            'average_results_returned': np.mean([m.final_results for m in recent_metrics]),
            'average_confidence': np.mean([m.average_confidence for m in recent_metrics]),
            'average_similarity': np.mean([m.average_similarity for m in recent_metrics]),
            'filter_efficiency': np.mean([m.final_results / max(m.total_candidates, 1) for m in recent_metrics])
        }
        
        return stats
    
    def update_config(self, new_config: RankingConfig) -> None:
        """Update ranking configuration."""
        new_config.validate()
        self.config = new_config
        logger.info(f"Ranking configuration updated to strategy: {self.config.strategy}")
    
    def reset_metrics(self) -> None:
        """Reset ranking history and metrics."""
        self.ranking_history.clear()
        logger.info("Ranking metrics reset")
