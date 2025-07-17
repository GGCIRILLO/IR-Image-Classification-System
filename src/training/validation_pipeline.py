"""
Validation pipeline for IR Image Classification System.

This module provides comprehensive validation functionality to ensure
the model meets the 95% accuracy target and performs well on military IR imagery.
"""

import numpy as np
import torch
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import logging
from dataclasses import dataclass
import json
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from .trainer import TrainingMetrics
from .model_adapters import BaseModelAdapter
from src.models.data_models import IRImage, SimilarityResult

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """
    Comprehensive validation result with detailed metrics.
    """
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    embedding_quality_mean: float
    embedding_quality_std: float
    confusion_matrix: np.ndarray
    class_report: Dict[str, Any]
    processing_time_per_image: float
    meets_95_percent_target: bool
    total_samples: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'embedding_quality_mean': self.embedding_quality_mean,
            'embedding_quality_std': self.embedding_quality_std,
            'confusion_matrix': self.confusion_matrix.tolist(),
            'class_report': self.class_report,
            'processing_time_per_image': self.processing_time_per_image,
            'meets_95_percent_target': self.meets_95_percent_target,
            'total_samples': self.total_samples
        }


class ValidationPipeline:
    """
    Comprehensive validation pipeline for IR image classification models.
    
    Provides thorough evaluation including accuracy, precision, recall,
    embedding quality assessment, and performance benchmarking.
    """
    
    def __init__(self, model_adapter: BaseModelAdapter, target_accuracy: float = 0.95):
        """
        Initialize validation pipeline.
        
        Args:
            model_adapter: Model adapter to validate
            target_accuracy: Target accuracy threshold (default 95%)
        """
        self.model_adapter = model_adapter
        self.target_accuracy = target_accuracy
        self.device = model_adapter.device
        
        logger.info(f"ValidationPipeline initialized with target accuracy: {target_accuracy:.1%}")
    
    def validate_model(self, test_images: List[IRImage], 
                      class_labels: Optional[List[str]] = None) -> ValidationResult:
        """
        Perform comprehensive model validation.
        
        Args:
            test_images: Test images for validation
            class_labels: Optional list of class labels for detailed reporting
            
        Returns:
            ValidationResult: Comprehensive validation results
        """
        logger.info(f"Starting validation on {len(test_images)} test images")
        
        if not self.model_adapter.is_loaded:
            raise RuntimeError("Model must be loaded before validation")
        
        # Extract embeddings and measure performance
        embeddings, processing_times = self._extract_embeddings_with_timing(test_images)
        
        # Calculate embedding quality metrics
        embedding_qualities = [
            self.model_adapter.validate_embedding_quality(emb) for emb in embeddings
        ]
        
        # Perform similarity-based classification
        predictions, true_labels = self._classify_by_similarity(
            embeddings, test_images, class_labels
        )
        
        # Calculate comprehensive metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions, average='weighted', zero_division=0)
        recall = recall_score(true_labels, predictions, average='weighted', zero_division=0)
        f1 = f1_score(true_labels, predictions, average='weighted', zero_division=0)
        
        # Generate confusion matrix and classification report
        cm = confusion_matrix(true_labels, predictions)
        class_report = classification_report(
            true_labels, predictions, output_dict=True, zero_division=0
        )
        
        # Create validation result
        result = ValidationResult(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            embedding_quality_mean=np.mean(embedding_qualities),
            embedding_quality_std=np.std(embedding_qualities),
            confusion_matrix=cm,
            class_report=class_report,
            processing_time_per_image=np.mean(processing_times),
            meets_95_percent_target=accuracy >= self.target_accuracy,
            total_samples=len(test_images)
        )
        
        # Log results
        self._log_validation_results(result)
        
        return result
    
    def _extract_embeddings_with_timing(self, images: List[IRImage]) -> Tuple[List[np.ndarray], List[float]]:
        """
        Extract embeddings while measuring processing time.
        
        Args:
            images: List of IR images
            
        Returns:
            Tuple[List[np.ndarray], List[float]]: (embeddings, processing_times)
        """
        embeddings = []
        processing_times = []
        
        self.model_adapter.model.eval()
        
        with torch.no_grad():
            for image in tqdm(images, desc="Extracting embeddings"):
                start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                
                if start_time:
                    start_time.record()
                else:
                    import time
                    start_cpu_time = time.time()
                
                # Extract embedding
                embedding = self.model_adapter.extract_embedding(image.image_data)
                embeddings.append(embedding)
                
                if end_time:
                    end_time.record()
                    torch.cuda.synchronize()
                    processing_time = start_time.elapsed_time(end_time) / 1000.0  # Convert to seconds
                else:
                    processing_time = time.time() - start_cpu_time
                
                processing_times.append(processing_time)
        
        return embeddings, processing_times
    
    def _classify_by_similarity(self, embeddings: List[np.ndarray], 
                               images: List[IRImage],
                               class_labels: Optional[List[str]] = None) -> Tuple[List[str], List[str]]:
        """
        Classify images based on embedding similarity.
        
        Args:
            embeddings: List of embedding vectors
            images: List of IR images
            class_labels: Optional class labels for mapping
            
        Returns:
            Tuple[List[str], List[str]]: (predictions, true_labels)
        """
        predictions = []
        true_labels = []
        
        # Build class-to-embedding mapping for reference
        class_embeddings = {}
        for i, (embedding, image) in enumerate(zip(embeddings, images)):
            if image.object_class not in class_embeddings:
                class_embeddings[image.object_class] = []
            class_embeddings[image.object_class].append(embedding)
        
        # Calculate class centroids
        class_centroids = {}
        for class_name, class_embs in class_embeddings.items():
            class_centroids[class_name] = np.mean(class_embs, axis=0)
        
        # Classify each image
        for embedding, image in zip(embeddings, images):
            true_labels.append(image.object_class)
            
            # Find most similar class centroid
            best_similarity = -1
            best_class = image.object_class  # Default fallback
            
            for class_name, centroid in class_centroids.items():
                similarity = self._cosine_similarity(embedding, centroid)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_class = class_name
            
            predictions.append(best_class)
        
        return predictions, true_labels
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return np.dot(vec1, vec2) / (norm1 * norm2)
    
    def _log_validation_results(self, result: ValidationResult) -> None:
        """Log comprehensive validation results."""
        logger.info("=" * 50)
        logger.info("VALIDATION RESULTS")
        logger.info("=" * 50)
        logger.info(f"Total Samples: {result.total_samples}")
        logger.info(f"Accuracy: {result.accuracy:.4f} ({result.accuracy:.1%})")
        logger.info(f"Precision: {result.precision:.4f}")
        logger.info(f"Recall: {result.recall:.4f}")
        logger.info(f"F1-Score: {result.f1_score:.4f}")
        logger.info(f"Embedding Quality: {result.embedding_quality_mean:.4f} ± {result.embedding_quality_std:.4f}")
        logger.info(f"Processing Time: {result.processing_time_per_image:.4f}s per image")
        logger.info(f"95% Target Met: {'✅ YES' if result.meets_95_percent_target else '❌ NO'}")
        logger.info("=" * 50)
    
    def validate_embedding_quality(self, embeddings: List[np.ndarray]) -> Dict[str, float]:
        """
        Validate embedding quality with detailed analysis.
        
        Args:
            embeddings: List of embedding vectors
            
        Returns:
            Dict[str, float]: Embedding quality metrics
        """
        if not embeddings:
            return {'error': 'No embeddings provided'}
        
        qualities = []
        norms = []
        diversities = []
        
        for embedding in embeddings:
            # Individual quality score
            quality = self.model_adapter.validate_embedding_quality(embedding)
            qualities.append(quality)
            
            # L2 norm
            norm = np.linalg.norm(embedding)
            norms.append(norm)
            
            # Diversity (standard deviation)
            diversity = np.std(embedding)
            diversities.append(diversity)
        
        # Calculate pairwise similarities for clustering analysis
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = self._cosine_similarity(embeddings[i], embeddings[j])
                similarities.append(sim)
        
        return {
            'mean_quality': np.mean(qualities),
            'std_quality': np.std(qualities),
            'min_quality': np.min(qualities),
            'max_quality': np.max(qualities),
            'mean_norm': np.mean(norms),
            'std_norm': np.std(norms),
            'mean_diversity': np.mean(diversities),
            'std_diversity': np.std(diversities),
            'mean_pairwise_similarity': np.mean(similarities) if similarities else 0.0,
            'std_pairwise_similarity': np.std(similarities) if similarities else 0.0,
            'total_embeddings': len(embeddings)
        }
    
    def benchmark_performance(self, test_images: List[IRImage], 
                            target_time_per_image: float = 2.0) -> Dict[str, Any]:
        """
        Benchmark model performance against requirements.
        
        Args:
            test_images: Test images for benchmarking
            target_time_per_image: Target processing time per image (seconds)
            
        Returns:
            Dict[str, Any]: Performance benchmark results
        """
        logger.info(f"Benchmarking performance on {len(test_images)} images")
        
        # Validate model
        validation_result = self.validate_model(test_images)
        
        # Performance benchmarks
        benchmarks = {
            'accuracy_benchmark': {
                'target': self.target_accuracy,
                'achieved': validation_result.accuracy,
                'meets_target': validation_result.meets_95_percent_target,
                'margin': validation_result.accuracy - self.target_accuracy
            },
            'speed_benchmark': {
                'target_time_per_image': target_time_per_image,
                'achieved_time_per_image': validation_result.processing_time_per_image,
                'meets_target': validation_result.processing_time_per_image <= target_time_per_image,
                'speedup_factor': target_time_per_image / validation_result.processing_time_per_image
            },
            'embedding_quality_benchmark': {
                'target_quality': 0.8,  # Reasonable threshold
                'achieved_quality': validation_result.embedding_quality_mean,
                'meets_target': validation_result.embedding_quality_mean >= 0.8,
                'quality_std': validation_result.embedding_quality_std
            },
            'overall_performance': {
                'all_targets_met': (
                    validation_result.meets_95_percent_target and
                    validation_result.processing_time_per_image <= target_time_per_image and
                    validation_result.embedding_quality_mean >= 0.8
                )
            }
        }
        
        # Log benchmark results
        logger.info("PERFORMANCE BENCHMARKS:")
        logger.info(f"Accuracy: {'✅' if benchmarks['accuracy_benchmark']['meets_target'] else '❌'} "
                   f"{validation_result.accuracy:.1%} (target: {self.target_accuracy:.1%})")
        logger.info(f"Speed: {'✅' if benchmarks['speed_benchmark']['meets_target'] else '❌'} "
                   f"{validation_result.processing_time_per_image:.3f}s (target: {target_time_per_image}s)")
        logger.info(f"Quality: {'✅' if benchmarks['embedding_quality_benchmark']['meets_target'] else '❌'} "
                   f"{validation_result.embedding_quality_mean:.3f} (target: 0.8)")
        logger.info(f"Overall: {'✅ ALL TARGETS MET' if benchmarks['overall_performance']['all_targets_met'] else '❌ SOME TARGETS MISSED'}")
        
        return benchmarks
    
    def plot_validation_results(self, result: ValidationResult, save_path: Optional[str] = None) -> None:
        """
        Plot comprehensive validation results.
        
        Args:
            result: Validation result to plot
            save_path: Optional path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Confusion Matrix
        sns.heatmap(result.confusion_matrix, annot=True, fmt='d', ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('True')
        
        # Metrics Bar Chart
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        values = [result.accuracy, result.precision, result.recall, result.f1_score]
        colors = ['green' if v >= self.target_accuracy else 'red' for v in values]
        
        bars = axes[0, 1].bar(metrics, values, color=colors, alpha=0.7)
        axes[0, 1].axhline(y=self.target_accuracy, color='blue', linestyle='--', 
                          label=f'Target ({self.target_accuracy:.1%})')
        axes[0, 1].set_title('Performance Metrics')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].legend()
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        # Embedding Quality Distribution (placeholder)
        axes[1, 0].hist([result.embedding_quality_mean], bins=1, alpha=0.7, color='purple')
        axes[1, 0].axvline(x=0.8, color='blue', linestyle='--', label='Quality Target (0.8)')
        axes[1, 0].set_title('Embedding Quality')
        axes[1, 0].set_xlabel('Quality Score')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        
        # Performance Summary
        axes[1, 1].axis('off')
        summary_text = f"""
        VALIDATION SUMMARY
        
        Total Samples: {result.total_samples}
        Accuracy: {result.accuracy:.1%}
        Target Met: {'✅ YES' if result.meets_95_percent_target else '❌ NO'}
        
        Processing Time: {result.processing_time_per_image:.3f}s/image
        Embedding Quality: {result.embedding_quality_mean:.3f} ± {result.embedding_quality_std:.3f}
        
        Precision: {result.precision:.3f}
        Recall: {result.recall:.3f}
        F1-Score: {result.f1_score:.3f}
        """
        axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes,
                        fontsize=12, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Validation results plot saved to {save_path}")
        
        plt.show()
    
    def save_validation_report(self, result: ValidationResult, report_path: str) -> None:
        """
        Save comprehensive validation report to file.
        
        Args:
            result: Validation result to save
            report_path: Path to save the report
        """
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        report = {
            'validation_summary': {
                'timestamp': str(result.__dict__.get('timestamp', 'N/A')),
                'model_info': self.model_adapter.get_model_info(),
                'target_accuracy': self.target_accuracy,
                'validation_passed': result.meets_95_percent_target
            },
            'metrics': result.to_dict(),
            'recommendations': self._generate_recommendations(result)
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Validation report saved to {report_path}")
    
    def _generate_recommendations(self, result: ValidationResult) -> List[str]:
        """
        Generate recommendations based on validation results.
        
        Args:
            result: Validation result
            
        Returns:
            List[str]: List of recommendations
        """
        recommendations = []
        
        if not result.meets_95_percent_target:
            recommendations.append(
                f"Accuracy ({result.accuracy:.1%}) is below target ({self.target_accuracy:.1%}). "
                "Consider: 1) More training data, 2) Data augmentation, 3) Model architecture changes, "
                "4) Hyperparameter tuning."
            )
        
        if result.precision < 0.9:
            recommendations.append(
                f"Precision ({result.precision:.3f}) could be improved. "
                "Consider: 1) Better negative sampling, 2) Class balancing, 3) Threshold tuning."
            )
        
        if result.recall < 0.9:
            recommendations.append(
                f"Recall ({result.recall:.3f}) could be improved. "
                "Consider: 1) More diverse training data, 2) Reduced class imbalance, 3) Model capacity increase."
            )
        
        if result.embedding_quality_mean < 0.8:
            recommendations.append(
                f"Embedding quality ({result.embedding_quality_mean:.3f}) is below optimal. "
                "Consider: 1) Longer training, 2) Better loss function, 3) Regularization techniques."
            )
        
        if result.processing_time_per_image > 2.0:
            recommendations.append(
                f"Processing time ({result.processing_time_per_image:.3f}s) exceeds 2s target. "
                "Consider: 1) Model optimization, 2) Quantization, 3) Hardware acceleration."
            )
        
        if not recommendations:
            recommendations.append("All validation targets met! Model is ready for deployment.")
        
        return recommendations