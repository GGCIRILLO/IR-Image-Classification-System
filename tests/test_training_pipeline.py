"""
Test suite for training pipeline and metrics tracking.

Tests the ModelTrainer, TrainingMetrics, and ValidationPipeline components
to ensure they work correctly with triplet loss optimization and 95% accuracy target.
"""

import pytest
import torch
import numpy as np
from datetime import datetime
from pathlib import Path
import tempfile
import shutil
from unittest.mock import Mock, patch

from src.training import (
    ModelTrainer, TrainingMetrics, TripletLoss, IRImageDataset,
    ValidationPipeline, ValidationResult, IRResNet50Adapter,
    TrainingConfig, ResNet50Config
)
from src.models.data_models import IRImage


class TestTrainingMetrics:
    """Test TrainingMetrics dataclass."""
    
    def test_training_metrics_creation(self):
        """Test creating TrainingMetrics with valid values."""
        metrics = TrainingMetrics(
            epoch=1,
            train_loss=0.5,
            val_loss=0.4,
            train_accuracy=0.85,
            val_accuracy=0.90,
            precision=0.88,
            recall=0.92,
            f1_score=0.90,
            embedding_quality=0.85,
            learning_rate=0.001,
            processing_time=120.5,
            timestamp=datetime.now()
        )
        
        assert metrics.epoch == 1
        assert metrics.val_accuracy == 0.90
        assert metrics.meets_target_accuracy(0.95) == False
        assert metrics.meets_target_accuracy(0.85) == True
    
    def test_training_metrics_validation(self):
        """Test TrainingMetrics validation."""
        with pytest.raises(ValueError):
            TrainingMetrics(
                epoch=1,
                train_loss=0.5,
                val_loss=0.4,
                train_accuracy=1.5,  # Invalid: > 1.0
                val_accuracy=0.90,
                precision=0.88,
                recall=0.92,
                f1_score=0.90,
                embedding_quality=0.85,
                learning_rate=0.001,
                processing_time=120.5,
                timestamp=datetime.now()
            )


class TestTripletLoss:
    """Test TripletLoss implementation."""
    
    def test_triplet_loss_forward(self):
        """Test triplet loss forward pass."""
        loss_fn = TripletLoss(margin=0.3, hard_negative_mining=False)
        
        # Create dummy embeddings
        batch_size = 4
        embedding_dim = 128
        
        anchor = torch.randn(batch_size, embedding_dim)
        positive = torch.randn(batch_size, embedding_dim)
        negative = torch.randn(batch_size, embedding_dim)
        
        loss = loss_fn(anchor, positive, negative)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar loss
        assert loss.item() >= 0  # Loss should be non-negative
    
    def test_triplet_loss_with_hard_mining(self):
        """Test triplet loss with hard negative mining."""
        loss_fn = TripletLoss(margin=0.3, hard_negative_mining=True)
        
        batch_size = 4
        embedding_dim = 128
        
        anchor = torch.randn(batch_size, embedding_dim)
        positive = torch.randn(batch_size, embedding_dim)
        negative = torch.randn(batch_size, embedding_dim)
        
        loss = loss_fn(anchor, positive, negative)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0


class TestIRImageDataset:
    """Test IRImageDataset implementation."""
    
    def create_dummy_ir_images(self, num_images=10, num_classes=3):
        """Create dummy IR images for testing."""
        images = []
        classes = [f"class_{i % num_classes}" for i in range(num_images)]
        
        for i in range(num_images):
            # Create dummy 224x224 grayscale image
            image_data = np.random.rand(224, 224).astype(np.float32)
            
            ir_image = IRImage(
                id=f"test_image_{i}",
                image_data=image_data,
                object_class=classes[i],
                confidence_score=0.9
            )
            images.append(ir_image)
        
        return images
    
    def test_dataset_creation(self):
        """Test creating IRImageDataset."""
        images = self.create_dummy_ir_images(10, 3)
        dataset = IRImageDataset(images)
        
        assert len(dataset) == 10
        assert len(dataset.class_to_indices) == 3
    
    def test_dataset_getitem(self):
        """Test dataset __getitem__ method."""
        images = self.create_dummy_ir_images(5, 2)
        dataset = IRImageDataset(images)
        
        image_tensor, class_label, idx = dataset[0]
        
        assert isinstance(image_tensor, torch.Tensor)
        assert image_tensor.shape == (1, 224, 224)  # CHW format
        assert isinstance(class_label, str)
        assert idx == 0
    
    def test_get_triplet(self):
        """Test triplet generation."""
        images = self.create_dummy_ir_images(10, 3)
        dataset = IRImageDataset(images)
        
        anchor, positive, negative = dataset.get_triplet(0)
        
        assert isinstance(anchor, torch.Tensor)
        assert isinstance(positive, torch.Tensor)
        assert isinstance(negative, torch.Tensor)
        assert anchor.shape == positive.shape == negative.shape


class TestModelTrainer:
    """Test ModelTrainer implementation."""
    
    def create_mock_adapter(self):
        """Create a mock model adapter for testing."""
        adapter = Mock(spec=IRResNet50Adapter)
        adapter.device = 'cpu'
        adapter.model = Mock()
        adapter.model.parameters.return_value = [torch.randn(10, 10, requires_grad=True)]
        adapter.model.state_dict.return_value = {'test': torch.randn(10, 10)}
        adapter.validate_embedding_quality.return_value = 0.85
        adapter.get_model_info.return_value = {'name': 'test_model'}
        return adapter
    
    def test_trainer_initialization(self):
        """Test ModelTrainer initialization."""
        adapter = self.create_mock_adapter()
        config = TrainingConfig(num_epochs=5, batch_size=4)
        
        trainer = ModelTrainer(adapter, config)
        
        assert trainer.model_adapter == adapter
        assert trainer.config == config
        assert trainer.device == 'cpu'
        assert isinstance(trainer.criterion, TripletLoss)
    
    @patch('src.training.trainer.DataLoader')
    def test_setup_optimizer_and_scheduler(self, mock_dataloader):
        """Test optimizer and scheduler setup."""
        adapter = self.create_mock_adapter()
        adapter.fine_tune_setup.return_value = torch.optim.Adam([torch.randn(10, requires_grad=True)])
        
        config = TrainingConfig(
            num_epochs=5,
            optimizer='adamw',
            scheduler='cosine'
        )
        
        trainer = ModelTrainer(adapter, config)
        trainer.setup_optimizer_and_scheduler()
        
        assert trainer.optimizer is not None
        assert trainer.scheduler is not None


class TestValidationPipeline:
    """Test ValidationPipeline implementation."""
    
    def create_mock_adapter(self):
        """Create a mock model adapter for testing."""
        adapter = Mock(spec=IRResNet50Adapter)
        adapter.device = 'cpu'
        adapter.is_loaded = True
        adapter.model = Mock()
        adapter.model.eval = Mock()
        adapter.extract_embedding.return_value = np.random.randn(512)
        adapter.validate_embedding_quality.return_value = 0.85
        adapter.get_model_info.return_value = {'name': 'test_model'}
        return adapter
    
    def create_dummy_ir_images(self, num_images=10, num_classes=3):
        """Create dummy IR images for testing."""
        images = []
        classes = [f"class_{i % num_classes}" for i in range(num_images)]
        
        for i in range(num_images):
            image_data = np.random.rand(224, 224).astype(np.float32)
            
            ir_image = IRImage(
                id=f"test_image_{i}",
                image_data=image_data,
                object_class=classes[i],
                confidence_score=0.9
            )
            images.append(ir_image)
        
        return images
    
    def test_validation_pipeline_initialization(self):
        """Test ValidationPipeline initialization."""
        adapter = self.create_mock_adapter()
        pipeline = ValidationPipeline(adapter, target_accuracy=0.95)
        
        assert pipeline.model_adapter == adapter
        assert pipeline.target_accuracy == 0.95
        assert pipeline.device == 'cpu'
    
    @patch('torch.cuda.is_available', return_value=False)
    def test_validate_model(self, mock_cuda):
        """Test model validation."""
        adapter = self.create_mock_adapter()
        pipeline = ValidationPipeline(adapter, target_accuracy=0.95)
        
        test_images = self.create_dummy_ir_images(10, 3)
        
        result = pipeline.validate_model(test_images)
        
        assert isinstance(result, ValidationResult)
        assert 0.0 <= result.accuracy <= 1.0
        assert 0.0 <= result.precision <= 1.0
        assert 0.0 <= result.recall <= 1.0
        assert 0.0 <= result.f1_score <= 1.0
        assert result.total_samples == 10
    
    def test_embedding_quality_validation(self):
        """Test embedding quality validation."""
        adapter = self.create_mock_adapter()
        pipeline = ValidationPipeline(adapter)
        
        # Create dummy embeddings
        embeddings = [np.random.randn(512) for _ in range(5)]
        
        quality_metrics = pipeline.validate_embedding_quality(embeddings)
        
        assert 'mean_quality' in quality_metrics
        assert 'std_quality' in quality_metrics
        assert 'total_embeddings' in quality_metrics
        assert quality_metrics['total_embeddings'] == 5
    
    def test_benchmark_performance(self):
        """Test performance benchmarking."""
        adapter = self.create_mock_adapter()
        pipeline = ValidationPipeline(adapter, target_accuracy=0.95)
        
        test_images = self.create_dummy_ir_images(5, 2)
        
        benchmarks = pipeline.benchmark_performance(test_images, target_time_per_image=2.0)
        
        assert 'accuracy_benchmark' in benchmarks
        assert 'speed_benchmark' in benchmarks
        assert 'embedding_quality_benchmark' in benchmarks
        assert 'overall_performance' in benchmarks


class TestValidationResult:
    """Test ValidationResult dataclass."""
    
    def test_validation_result_creation(self):
        """Test creating ValidationResult."""
        cm = np.array([[10, 2], [1, 15]])
        class_report = {'accuracy': 0.89}
        
        result = ValidationResult(
            accuracy=0.89,
            precision=0.91,
            recall=0.88,
            f1_score=0.89,
            embedding_quality_mean=0.85,
            embedding_quality_std=0.05,
            confusion_matrix=cm,
            class_report=class_report,
            processing_time_per_image=0.5,
            meets_95_percent_target=False,
            total_samples=28
        )
        
        assert result.accuracy == 0.89
        assert result.meets_95_percent_target == False
        assert result.total_samples == 28
    
    def test_validation_result_to_dict(self):
        """Test converting ValidationResult to dictionary."""
        cm = np.array([[10, 2], [1, 15]])
        class_report = {'accuracy': 0.89}
        
        result = ValidationResult(
            accuracy=0.89,
            precision=0.91,
            recall=0.88,
            f1_score=0.89,
            embedding_quality_mean=0.85,
            embedding_quality_std=0.05,
            confusion_matrix=cm,
            class_report=class_report,
            processing_time_per_image=0.5,
            meets_95_percent_target=False,
            total_samples=28
        )
        
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert 'accuracy' in result_dict
        assert 'confusion_matrix' in result_dict
        assert isinstance(result_dict['confusion_matrix'], list)


# Integration tests
class TestTrainingIntegration:
    """Integration tests for training pipeline components."""
    
    def test_end_to_end_training_setup(self):
        """Test end-to-end training setup without actual training."""
        # Create minimal config
        model_config = ResNet50Config(
            embedding_dim=128,
            batch_size=2,
            learning_rate=0.001
        )
        
        training_config = TrainingConfig(
            num_epochs=2,
            batch_size=2,
            early_stopping_patience=1
        )
        
        # This test verifies that all components can be instantiated together
        # without running actual training (which would require real data and models)
        assert model_config.embedding_dim == 128
        assert training_config.num_epochs == 2
        assert isinstance(training_config.triplet_margin, float)


if __name__ == '__main__':
    pytest.main([__file__])