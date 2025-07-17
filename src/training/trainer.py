"""
Training pipeline for IR Image Classification System.

This module implements the ModelTrainer class with triplet loss optimization,
training metrics tracking, and validation pipeline for achieving 95% accuracy target.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
import logging
from pathlib import Path
import time
from datetime import datetime
import json
from dataclasses import dataclass, asdict
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from tqdm import tqdm

from .model_adapters import BaseModelAdapter, IRResNet50Adapter, QwenVLMAdapter
from .model_config import ModelConfig, TrainingConfig
from src.models.data_models import IRImage, Embedding

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """
    Comprehensive metrics tracking for training and validation.
    
    Tracks precision, recall, embedding quality, and other performance indicators
    to ensure 95% accuracy target is met.
    """
    epoch: int
    train_loss: float
    val_loss: float
    train_accuracy: float
    val_accuracy: float
    precision: float
    recall: float
    f1_score: float
    embedding_quality: float
    learning_rate: float
    processing_time: float
    timestamp: datetime
    
    def __post_init__(self):
        """Validate metrics after initialization."""
        self.validate()
    
    def validate(self) -> bool:
        """Validate metric values are within expected ranges."""
        if not 0.0 <= self.train_accuracy <= 1.0:
            raise ValueError(f"Train accuracy must be 0-1, got {self.train_accuracy}")
        if not 0.0 <= self.val_accuracy <= 1.0:
            raise ValueError(f"Validation accuracy must be 0-1, got {self.val_accuracy}")
        if not 0.0 <= self.precision <= 1.0:
            raise ValueError(f"Precision must be 0-1, got {self.precision}")
        if not 0.0 <= self.recall <= 1.0:
            raise ValueError(f"Recall must be 0-1, got {self.recall}")
        if not 0.0 <= self.embedding_quality <= 1.0:
            raise ValueError(f"Embedding quality must be 0-1, got {self.embedding_quality}")
        return True
    
    def meets_target_accuracy(self, target: float = 0.95) -> bool:
        """Check if validation accuracy meets target threshold."""
        return self.val_accuracy >= target
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for logging."""
        return asdict(self)


class TripletLoss(nn.Module):
    """
    Triplet loss implementation for embedding optimization.
    
    Optimizes embeddings to minimize distance between similar images (anchor-positive)
    and maximize distance between dissimilar images (anchor-negative).
    """
    
    def __init__(self, margin: float = 0.3, hard_negative_mining: bool = True):
        """
        Initialize triplet loss.
        
        Args:
            margin: Margin for triplet loss
            hard_negative_mining: Whether to use hard negative mining
        """
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.hard_negative_mining = hard_negative_mining
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
    
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, 
                negative: torch.Tensor) -> torch.Tensor:
        """
        Compute triplet loss.
        
        Args:
            anchor: Anchor embeddings
            positive: Positive embeddings (same class as anchor)
            negative: Negative embeddings (different class from anchor)
            
        Returns:
            torch.Tensor: Triplet loss value
        """
        # Calculate distances
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)
        
        if self.hard_negative_mining:
            # Select hard negatives (closest negative samples)
            hard_negatives = self._select_hard_negatives(anchor, negative)
            neg_dist = F.pairwise_distance(anchor, hard_negatives, p=2)
        
        # Compute triplet loss
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()
    
    def _select_hard_negatives(self, anchor: torch.Tensor, 
                              negatives: torch.Tensor) -> torch.Tensor:
        """
        Select hard negative samples for each anchor.
        
        Args:
            anchor: Anchor embeddings
            negatives: Negative embeddings
            
        Returns:
            torch.Tensor: Hard negative embeddings
        """
        batch_size = anchor.size(0)
        hard_negatives = []
        
        for i in range(batch_size):
            # Calculate distances from anchor to all negatives
            distances = F.pairwise_distance(
                anchor[i].unsqueeze(0).expand_as(negatives), 
                negatives, p=2
            )
            # Select the closest negative (hardest)
            hard_idx = torch.argmin(distances)
            hard_negatives.append(negatives[hard_idx])
        
        return torch.stack(hard_negatives)


class IRImageDataset(Dataset):
    """
    Dataset class for IR images with triplet sampling support.
    """
    
    def __init__(self, images: List[IRImage], transform: Optional[Callable] = None):
        """
        Initialize dataset.
        
        Args:
            images: List of IR images
            transform: Optional transform function
        """
        self.images = images
        self.transform = transform
        self.class_to_indices = self._build_class_indices()
    
    def _build_class_indices(self) -> Dict[str, List[int]]:
        """Build mapping from class to image indices."""
        class_indices = {}
        for idx, image in enumerate(self.images):
            if image.object_class not in class_indices:
                class_indices[image.object_class] = []
            class_indices[image.object_class].append(idx)
        return class_indices
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, int]:
        """
        Get image, class, and index.
        
        Returns:
            Tuple[torch.Tensor, str, int]: (image_tensor, class_label, index)
        """
        image = self.images[idx]
        image_tensor = torch.from_numpy(image.image_data).float()
        
        # Add channel dimension if needed
        if len(image_tensor.shape) == 2:
            image_tensor = image_tensor.unsqueeze(0)
        
        if self.transform:
            image_tensor = self.transform(image_tensor)
        
        return image_tensor, image.object_class, idx
    
    def get_triplet(self, anchor_idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get triplet (anchor, positive, negative) for training.
        
        Args:
            anchor_idx: Index of anchor image
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (anchor, positive, negative)
        """
        anchor_image = self.images[anchor_idx]
        anchor_class = anchor_image.object_class
        
        # Get positive sample (same class)
        positive_indices = [i for i in self.class_to_indices[anchor_class] if i != anchor_idx]
        if not positive_indices:
            # If no other samples in class, use the same image
            positive_idx = anchor_idx
        else:
            positive_idx = np.random.choice(positive_indices)
        
        # Get negative sample (different class)
        negative_classes = [cls for cls in self.class_to_indices.keys() if cls != anchor_class]
        if not negative_classes:
            raise ValueError("Need at least 2 classes for triplet training")
        
        negative_class = np.random.choice(negative_classes)
        negative_idx = np.random.choice(self.class_to_indices[negative_class])
        
        # Get tensors
        anchor_tensor, _, _ = self[anchor_idx]
        positive_tensor, _, _ = self[positive_idx]
        negative_tensor, _, _ = self[negative_idx]
        
        return anchor_tensor, positive_tensor, negative_tensor


class ModelTrainer:
    """
    Main training class with triplet loss optimization and comprehensive metrics tracking.
    
    Implements training pipeline with validation to achieve 95% accuracy target.
    """
    
    def __init__(self, model_adapter: BaseModelAdapter, config: TrainingConfig):
        """
        Initialize trainer.
        
        Args:
            model_adapter: Model adapter for training
            config: Training configuration
        """
        self.model_adapter = model_adapter
        self.config = config
        self.device = model_adapter.device
        
        # Training components
        self.criterion = TripletLoss(
            margin=config.triplet_margin,
            hard_negative_mining=config.hard_negative_mining
        )
        self.optimizer = None
        self.scheduler = None
        
        # Metrics tracking
        self.training_history: List[TrainingMetrics] = []
        self.best_val_accuracy = 0.0
        self.best_model_state = None
        self.early_stopping_counter = 0
        
        # Setup logging
        self.setup_logging()
        
        logger.info(f"ModelTrainer initialized with {type(model_adapter).__name__}")
    
    def setup_logging(self) -> None:
        """Setup training logging."""
        log_dir = Path(self.config.checkpoint_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create training log file
        log_file = log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    def setup_optimizer_and_scheduler(self) -> None:
        """Setup optimizer and learning rate scheduler."""
        # Get trainable parameters
        if hasattr(self.model_adapter, 'fine_tune_setup'):
            self.optimizer = self.model_adapter.fine_tune_setup(
                learning_rate=self.config.learning_rate
            )
        else:
            # Fallback optimizer setup
            trainable_params = filter(lambda p: p.requires_grad, self.model_adapter.model.parameters())
            
            if self.config.optimizer == 'adamw':
                self.optimizer = torch.optim.AdamW(
                    trainable_params,
                    lr=self.config.learning_rate,
                    weight_decay=self.config.weight_decay
                )
            elif self.config.optimizer == 'adam':
                self.optimizer = torch.optim.Adam(
                    trainable_params,
                    lr=self.config.learning_rate,
                    weight_decay=self.config.weight_decay
                )
            elif self.config.optimizer == 'sgd':
                self.optimizer = torch.optim.SGD(
                    trainable_params,
                    lr=self.config.learning_rate,
                    momentum=0.9,
                    weight_decay=self.config.weight_decay
                )
        
        # Setup scheduler
        if self.config.scheduler == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.config.num_epochs
            )
        elif self.config.scheduler == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=10, gamma=0.1
            )
        elif self.config.scheduler == 'plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='max', patience=5, factor=0.5
            )
    
    def train(self, train_images: List[IRImage], val_images: List[IRImage]) -> List[TrainingMetrics]:
        """
        Train the model with comprehensive metrics tracking.
        
        Args:
            train_images: Training images
            val_images: Validation images
            
        Returns:
            List[TrainingMetrics]: Training history
        """
        logger.info(f"Starting training with {len(train_images)} train, {len(val_images)} val images")
        
        # Setup datasets
        train_dataset = IRImageDataset(train_images)
        val_dataset = IRImageDataset(val_images)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=2
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=2
        )
        
        # Setup optimizer and scheduler
        self.setup_optimizer_and_scheduler()
        
        # Training loop
        for epoch in range(self.config.num_epochs):
            epoch_start_time = time.time()
            
            # Training phase
            train_metrics = self._train_epoch(train_loader, epoch)
            
            # Validation phase
            val_metrics = self._validate_epoch(val_loader, epoch)
            
            # Combine metrics
            epoch_time = time.time() - epoch_start_time
            metrics = TrainingMetrics(
                epoch=epoch,
                train_loss=train_metrics['loss'],
                val_loss=val_metrics['loss'],
                train_accuracy=train_metrics['accuracy'],
                val_accuracy=val_metrics['accuracy'],
                precision=val_metrics['precision'],
                recall=val_metrics['recall'],
                f1_score=val_metrics['f1_score'],
                embedding_quality=val_metrics['embedding_quality'],
                learning_rate=self.optimizer.param_groups[0]['lr'],
                processing_time=epoch_time,
                timestamp=datetime.now()
            )
            
            self.training_history.append(metrics)
            
            # Log progress
            if epoch % self.config.log_interval == 0:
                self._log_epoch_progress(metrics)
            
            # Check for improvement
            if metrics.val_accuracy > self.best_val_accuracy:
                self.best_val_accuracy = metrics.val_accuracy
                self.best_model_state = self.model_adapter.model.state_dict().copy()
                self.early_stopping_counter = 0
                
                # Save best model
                if self.config.save_best_only:
                    self._save_checkpoint(epoch, metrics, is_best=True)
            else:
                self.early_stopping_counter += 1
            
            # Early stopping check
            if self.early_stopping_counter >= self.config.early_stopping_patience:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break
            
            # Check if target accuracy reached
            if metrics.meets_target_accuracy(0.95):
                logger.info(f"Target accuracy of 95% reached at epoch {epoch}!")
                break
            
            # Update scheduler
            if self.scheduler:
                if self.config.scheduler == 'plateau':
                    self.scheduler.step(metrics.val_accuracy)
                else:
                    self.scheduler.step()
            
            # Save periodic checkpoint
            if epoch % self.config.save_interval == 0:
                self._save_checkpoint(epoch, metrics, is_best=False)
        
        # Load best model
        if self.best_model_state:
            self.model_adapter.model.load_state_dict(self.best_model_state)
            logger.info(f"Loaded best model with validation accuracy: {self.best_val_accuracy:.4f}")
        
        return self.training_history    
    
    def _train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
            
        Returns:
            Dict[str, float]: Training metrics for the epoch
        """
        self.model_adapter.model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch} Training")
        
        for batch_idx, (images, classes, indices) in enumerate(progress_bar):
            images = images.to(self.device)
            batch_size = images.size(0)
            
            # Generate triplets for this batch
            anchors, positives, negatives = [], [], []
            
            for i in range(batch_size):
                try:
                    anchor, positive, negative = train_loader.dataset.get_triplet(indices[i].item())
                    anchors.append(anchor)
                    positives.append(positive)
                    negatives.append(negative)
                except ValueError:
                    # Skip if can't form triplet
                    continue
            
            if not anchors:
                continue
            
            # Stack triplets
            anchor_batch = torch.stack(anchors).to(self.device)
            positive_batch = torch.stack(positives).to(self.device)
            negative_batch = torch.stack(negatives).to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            anchor_embeddings = self.model_adapter.model(anchor_batch)
            positive_embeddings = self.model_adapter.model(positive_batch)
            negative_embeddings = self.model_adapter.model(negative_batch)
            
            # Compute triplet loss
            loss = self.criterion(anchor_embeddings, positive_embeddings, negative_embeddings)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model_adapter.model.parameters(),
                    self.config.gradient_clip_norm
                )
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            
            # Calculate accuracy based on triplet distances
            with torch.no_grad():
                pos_distances = F.pairwise_distance(anchor_embeddings, positive_embeddings)
                neg_distances = F.pairwise_distance(anchor_embeddings, negative_embeddings)
                correct = (pos_distances < neg_distances).sum().item()
                correct_predictions += correct
                total_samples += len(anchors)
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{correct_predictions/max(total_samples, 1):.4f}'
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct_predictions / max(total_samples, 1)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }
    
    def _validate_epoch(self, val_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        Validate for one epoch with comprehensive metrics.
        
        Args:
            val_loader: Validation data loader
            epoch: Current epoch number
            
        Returns:
            Dict[str, float]: Validation metrics
        """
        self.model_adapter.model.eval()
        total_loss = 0.0
        all_embeddings = []
        all_classes = []
        embedding_qualities = []
        
        with torch.no_grad():
            for images, classes, indices in tqdm(val_loader, desc=f"Epoch {epoch} Validation"):
                images = images.to(self.device)
                
                # Extract embeddings
                embeddings = self.model_adapter.model(images)
                all_embeddings.extend(embeddings.cpu().numpy())
                all_classes.extend(classes)
                
                # Calculate embedding quality
                for embedding in embeddings.cpu().numpy():
                    quality = self.model_adapter.validate_embedding_quality(embedding)
                    embedding_qualities.append(quality)
                
                # Calculate validation loss using triplets
                batch_size = images.size(0)
                if batch_size >= 3:  # Need at least 3 samples for triplets
                    try:
                        # Simple triplet formation for validation
                        anchor_idx = 0
                        positive_idx = 1 if classes[0] == classes[1] else 2
                        negative_idx = 2 if classes[0] != classes[2] else 1
                        
                        if positive_idx < batch_size and negative_idx < batch_size:
                            loss = self.criterion(
                                embeddings[anchor_idx:anchor_idx+1],
                                embeddings[positive_idx:positive_idx+1],
                                embeddings[negative_idx:negative_idx+1]
                            )
                            total_loss += loss.item()
                    except:
                        pass  # Skip if can't form valid triplet
        
        # Calculate comprehensive metrics
        metrics = self._calculate_validation_metrics(all_embeddings, all_classes)
        metrics['loss'] = total_loss / len(val_loader)
        metrics['embedding_quality'] = np.mean(embedding_qualities) if embedding_qualities else 0.0
        
        return metrics
    
    def _calculate_validation_metrics(self, embeddings: List[np.ndarray], 
                                    classes: List[str]) -> Dict[str, float]:
        """
        Calculate comprehensive validation metrics including precision, recall, F1.
        
        Args:
            embeddings: List of embedding vectors
            classes: List of class labels
            
        Returns:
            Dict[str, float]: Validation metrics
        """
        if len(embeddings) < 2:
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0
            }
        
        # Convert to numpy arrays
        embeddings_array = np.array(embeddings)
        
        # Calculate pairwise similarities and predict classes
        predictions = []
        true_labels = []
        
        for i in range(len(embeddings)):
            query_embedding = embeddings_array[i]
            query_class = classes[i]
            
            # Find most similar embedding (excluding self)
            similarities = []
            for j in range(len(embeddings)):
                if i != j:
                    similarity = np.dot(query_embedding, embeddings_array[j]) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(embeddings_array[j])
                    )
                    similarities.append((similarity, classes[j]))
            
            if similarities:
                # Get most similar class
                similarities.sort(reverse=True)
                predicted_class = similarities[0][1]
                predictions.append(predicted_class)
                true_labels.append(query_class)
        
        if not predictions:
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0
            }
        
        # Calculate metrics
        unique_classes = list(set(true_labels))
        
        if len(unique_classes) == 1:
            # Only one class, use simple accuracy
            accuracy = sum(p == t for p, t in zip(predictions, true_labels)) / len(predictions)
            return {
                'accuracy': accuracy,
                'precision': accuracy,
                'recall': accuracy,
                'f1_score': accuracy
            }
        
        # Multi-class metrics
        accuracy = sum(p == t for p, t in zip(predictions, true_labels)) / len(predictions)
        
        try:
            precision = precision_score(true_labels, predictions, average='weighted', zero_division=0)
            recall = recall_score(true_labels, predictions, average='weighted', zero_division=0)
            f1 = f1_score(true_labels, predictions, average='weighted', zero_division=0)
        except:
            # Fallback if sklearn metrics fail
            precision = accuracy
            recall = accuracy
            f1 = accuracy
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    def _log_epoch_progress(self, metrics: TrainingMetrics) -> None:
        """Log epoch progress with comprehensive metrics."""
        logger.info(
            f"Epoch {metrics.epoch}: "
            f"Train Loss: {metrics.train_loss:.4f}, "
            f"Val Loss: {metrics.val_loss:.4f}, "
            f"Train Acc: {metrics.train_accuracy:.4f}, "
            f"Val Acc: {metrics.val_accuracy:.4f}, "
            f"Precision: {metrics.precision:.4f}, "
            f"Recall: {metrics.recall:.4f}, "
            f"F1: {metrics.f1_score:.4f}, "
            f"Embedding Quality: {metrics.embedding_quality:.4f}, "
            f"LR: {metrics.learning_rate:.6f}, "
            f"Time: {metrics.processing_time:.2f}s"
        )
        
        # Check target accuracy
        if metrics.meets_target_accuracy(0.95):
            logger.info("🎯 TARGET ACCURACY OF 95% ACHIEVED!")
    
    def _save_checkpoint(self, epoch: int, metrics: TrainingMetrics, is_best: bool = False) -> None:
        """
        Save model checkpoint with metrics.
        
        Args:
            epoch: Current epoch
            metrics: Training metrics
            is_best: Whether this is the best model so far
        """
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model_adapter.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': metrics.to_dict(),
            'best_val_accuracy': self.best_val_accuracy,
            'model_config': self.model_adapter.get_model_info(),
            'training_config': asdict(self.config)
        }
        
        # Save checkpoint
        if is_best:
            checkpoint_path = checkpoint_dir / 'best_model.pth'
            logger.info(f"Saving best model with val accuracy: {metrics.val_accuracy:.4f}")
        else:
            checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
        
        torch.save(checkpoint, checkpoint_path)
        
        # Save metrics history
        metrics_path = checkpoint_dir / 'training_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump([m.to_dict() for m in self.training_history], f, indent=2, default=str)
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Load model checkpoint and resume training.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Dict[str, Any]: Checkpoint information
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model_adapter.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state if available
        if self.optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state if available
        if self.scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load training state
        self.best_val_accuracy = checkpoint.get('best_val_accuracy', 0.0)
        
        logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint
    
    def evaluate_model(self, test_images: List[IRImage]) -> Dict[str, float]:
        """
        Comprehensive model evaluation on test set.
        
        Args:
            test_images: Test images for evaluation
            
        Returns:
            Dict[str, float]: Comprehensive evaluation metrics
        """
        logger.info(f"Evaluating model on {len(test_images)} test images")
        
        test_dataset = IRImageDataset(test_images)
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=2
        )
        
        self.model_adapter.model.eval()
        all_embeddings = []
        all_classes = []
        embedding_qualities = []
        processing_times = []
        
        with torch.no_grad():
            for images, classes, _ in tqdm(test_loader, desc="Evaluating"):
                start_time = time.time()
                
                images = images.to(self.device)
                embeddings = self.model_adapter.model(images)
                
                processing_time = time.time() - start_time
                processing_times.extend([processing_time / len(images)] * len(images))
                
                # Collect embeddings and classes
                for embedding, class_label in zip(embeddings.cpu().numpy(), classes):
                    all_embeddings.append(embedding)
                    all_classes.append(class_label)
                    
                    # Calculate embedding quality
                    quality = self.model_adapter.validate_embedding_quality(embedding)
                    embedding_qualities.append(quality)
        
        # Calculate comprehensive metrics using validation pipeline approach
        metrics = self._calculate_validation_metrics(all_embeddings, all_classes)
        
        # Add additional evaluation metrics
        evaluation_results = {
            'test_accuracy': metrics['accuracy'],
            'test_precision': metrics['precision'],
            'test_recall': metrics['recall'],
            'test_f1_score': metrics['f1_score'],
            'embedding_quality_mean': np.mean(embedding_qualities),
            'embedding_quality_std': np.std(embedding_qualities),
            'processing_time_per_image': np.mean(processing_times),
            'total_test_samples': len(test_images),
            'meets_95_percent_target': metrics['accuracy'] >= 0.95,
            'embedding_dimension': len(all_embeddings[0]) if all_embeddings else 0
        }
        
        # Log evaluation results
        logger.info("=" * 50)
        logger.info("MODEL EVALUATION RESULTS")
        logger.info("=" * 50)
        logger.info(f"Test Accuracy: {evaluation_results['test_accuracy']:.4f} ({evaluation_results['test_accuracy']:.1%})")
        logger.info(f"Test Precision: {evaluation_results['test_precision']:.4f}")
        logger.info(f"Test Recall: {evaluation_results['test_recall']:.4f}")
        logger.info(f"Test F1-Score: {evaluation_results['test_f1_score']:.4f}")
        logger.info(f"Embedding Quality: {evaluation_results['embedding_quality_mean']:.4f} ± {evaluation_results['embedding_quality_std']:.4f}")
        logger.info(f"Processing Time: {evaluation_results['processing_time_per_image']:.4f}s per image")
        logger.info(f"95% Target Met: {'✅ YES' if evaluation_results['meets_95_percent_target'] else '❌ NO'}")
        logger.info("=" * 50)
        
        return evaluation_results
    
    def plot_training_history(self, save_path: Optional[str] = None) -> None:
        """
        Plot training history with comprehensive metrics.
        
        Args:
            save_path: Optional path to save the plot
        """
        if not self.training_history:
            logger.warning("No training history to plot")
            return
        
        epochs = [m.epoch for m in self.training_history]
        train_losses = [m.train_loss for m in self.training_history]
        val_losses = [m.val_loss for m in self.training_history]
        train_accs = [m.train_accuracy for m in self.training_history]
        val_accs = [m.val_accuracy for m in self.training_history]
        precisions = [m.precision for m in self.training_history]
        recalls = [m.recall for m in self.training_history]
        f1_scores = [m.f1_score for m in self.training_history]
        embedding_qualities = [m.embedding_quality for m in self.training_history]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        axes[0, 0].plot(epochs, train_losses, label='Train Loss', color='blue')
        axes[0, 0].plot(epochs, val_losses, label='Val Loss', color='red')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy plot
        axes[0, 1].plot(epochs, train_accs, label='Train Accuracy', color='blue')
        axes[0, 1].plot(epochs, val_accs, label='Val Accuracy', color='red')
        axes[0, 1].axhline(y=0.95, color='green', linestyle='--', label='95% Target')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Precision, Recall, F1 plot
        axes[1, 0].plot(epochs, precisions, label='Precision', color='green')
        axes[1, 0].plot(epochs, recalls, label='Recall', color='orange')
        axes[1, 0].plot(epochs, f1_scores, label='F1-Score', color='purple')
        axes[1, 0].set_title('Precision, Recall, and F1-Score')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Embedding Quality plot
        axes[1, 1].plot(epochs, embedding_qualities, label='Embedding Quality', color='brown')
        axes[1, 1].set_title('Embedding Quality Over Time')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Quality Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history plot saved to {save_path}")
        
        plt.show()
    
    def get_training_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive training summary.
        
        Returns:
            Dict[str, Any]: Training summary with key metrics
        """
        if not self.training_history:
            return {"error": "No training history available"}
        
        best_metrics = max(self.training_history, key=lambda x: x.val_accuracy)
        final_metrics = self.training_history[-1]
        
        return {
            'total_epochs': len(self.training_history),
            'best_validation_accuracy': best_metrics.val_accuracy,
            'best_epoch': best_metrics.epoch,
            'final_validation_accuracy': final_metrics.val_accuracy,
            'target_95_percent_achieved': best_metrics.val_accuracy >= 0.95,
            'best_precision': best_metrics.precision,
            'best_recall': best_metrics.recall,
            'best_f1_score': best_metrics.f1_score,
            'best_embedding_quality': best_metrics.embedding_quality,
            'total_training_time': sum(m.processing_time for m in self.training_history),
            'average_epoch_time': np.mean([m.processing_time for m in self.training_history]),
            'early_stopping_triggered': self.early_stopping_counter >= self.config.early_stopping_patience
        }    
    def get_training_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive training summary.
        
        Returns:
            Dict[str, Any]: Training summary with key metrics
        """
        if not self.training_history:
            return {"error": "No training history available"}
        
        best_metrics = max(self.training_history, key=lambda x: x.val_accuracy)
        final_metrics = self.training_history[-1]
        
        return {
            'total_epochs': len(self.training_history),
            'best_validation_accuracy': best_metrics.val_accuracy,
            'best_epoch': best_metrics.epoch,
            'final_validation_accuracy': final_metrics.val_accuracy,
            'target_95_percent_achieved': best_metrics.val_accuracy >= 0.95,
            'best_precision': best_metrics.precision,
            'best_recall': best_metrics.recall,
            'best_f1_score': best_metrics.f1_score,
            'best_embedding_quality': best_metrics.embedding_quality,
            'total_training_time': sum(m.processing_time for m in self.training_history),
            'average_epoch_time': np.mean([m.processing_time for m in self.training_history]),
            'early_stopping_triggered': self.early_stopping_counter >= self.config.early_stopping_patience
        }

    def plot_training_history(self, save_path: Optional[str] = None) -> None:
        """
        Plot training history with comprehensive metrics.
        
        Args:
            save_path: Optional path to save the plot
        """
        if not self.training_history:
            logger.warning("No training history to plot")
            return
        
        epochs = [m.epoch for m in self.training_history]
        train_losses = [m.train_loss for m in self.training_history]
        val_losses = [m.val_loss for m in self.training_history]
        train_accs = [m.train_accuracy for m in self.training_history]
        val_accs = [m.val_accuracy for m in self.training_history]
        precisions = [m.precision for m in self.training_history]
        recalls = [m.recall for m in self.training_history]
        f1_scores = [m.f1_score for m in self.training_history]
        embedding_qualities = [m.embedding_quality for m in self.training_history]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        axes[0, 0].plot(epochs, train_losses, label='Train Loss', color='blue')
        axes[0, 0].plot(epochs, val_losses, label='Val Loss', color='red')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy plot
        axes[0, 1].plot(epochs, train_accs, label='Train Accuracy', color='blue')
        axes[0, 1].plot(epochs, val_accs, label='Val Accuracy', color='red')
        axes[0, 1].axhline(y=0.95, color='green', linestyle='--', label='95% Target')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Precision, Recall, F1 plot
        axes[1, 0].plot(epochs, precisions, label='Precision', color='green')
        axes[1, 0].plot(epochs, recalls, label='Recall', color='orange')
        axes[1, 0].plot(epochs, f1_scores, label='F1-Score', color='purple')
        axes[1, 0].set_title('Precision, Recall, and F1-Score')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Embedding Quality plot
        axes[1, 1].plot(epochs, embedding_qualities, label='Embedding Quality', color='brown')
        axes[1, 1].axhline(y=0.8, color='red', linestyle='--', label='Quality Target')
        axes[1, 1].set_title('Embedding Quality Over Time')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Quality Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history plot saved to {save_path}")
        
        plt.show()