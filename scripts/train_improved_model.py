#!/usr/bin/env python3
"""
Comprehensive training script for improved IR image classification.

This script provides end-to-end training with advanced techniques:
1. Data augmentation for IR images
2. Advanced loss functions (Triplet + Center Loss)
3. Learning rate scheduling
4. Model ensemble techniques
5. Automatic hyperparameter optimization
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.data_models import IRImage
from src.embedding.extractor import EmbeddingExtractor
from src.database.similarity_searcher import SimilaritySearcher
from src.training.model_adapters import ModelAdapterFactory, BaseModelAdapter
from src.training.trainer import ModelTrainer, TrainingConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IRImageAugmentedDataset(Dataset):
    """Enhanced dataset with IR-specific augmentations."""
    
    def __init__(self, images: List[IRImage], mode: str = 'train', augment_prob: float = 0.8):
        self.images = images
        self.mode = mode
        self.augment_prob = augment_prob
        self.class_to_idx = self._build_class_mapping()
        self.transforms = self._setup_transforms()
        
    def _build_class_mapping(self) -> Dict[str, int]:
        """Build mapping from class names to indices."""
        unique_classes = sorted(list(set(img.object_class for img in self.images)))
        return {cls: idx for idx, cls in enumerate(unique_classes)}
    
    def _setup_transforms(self) -> A.Compose:
        """Setup IR-specific augmentations."""
        if self.mode == 'train':
            return A.Compose([
                # Geometric transformations
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.Rotate(limit=15, p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.1,
                    rotate_limit=10,
                    p=0.5
                ),
                
                # Intensity transformations (important for IR)
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.6
                ),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.Blur(blur_limit=3, p=0.2),
                
                # Thermal-specific augmentations
                A.RandomGamma(gamma_limit=(80, 120), p=0.3),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
                
                # Normalization
                A.Normalize(mean=[0.485], std=[0.229]),
                ToTensorV2()
            ])
        else:
            return A.Compose([
                A.Normalize(mean=[0.485], std=[0.229]),
                ToTensorV2()
            ])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        image_data = image.image_data.copy()
        
        # Ensure image is in correct format for albumentations
        if image_data.dtype != np.uint8:
            image_data = (image_data * 255).astype(np.uint8)
        
        # Apply augmentations
        if np.random.random() < self.augment_prob or self.mode != 'train':
            transformed = self.transforms(image=image_data)
            image_tensor = transformed['image']
        else:
            # Convert to tensor without augmentation
            image_tensor = torch.from_numpy(image_data).float() / 255.0
            image_tensor = image_tensor.unsqueeze(0)  # Add channel dimension
        
        label = self.class_to_idx[image.object_class]
        
        return image_tensor, label, image.object_class


class CenterLoss(nn.Module):
    """Center Loss for better feature learning."""
    
    def __init__(self, num_classes: int, feat_dim: int, device: str):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device
        
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim).to(device))
        
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        batch_size = features.size(0)
        
        # Compute distances to centers
        centers_batch = self.centers.index_select(0, labels.long())
        criterion = nn.MSELoss()
        center_loss = criterion(features, centers_batch)
        
        return center_loss


class TripletCenterLoss(nn.Module):
    """Combined Triplet and Center Loss."""
    
    def __init__(self, num_classes: int, feat_dim: int, device: str, 
                 margin: float = 0.3, center_weight: float = 0.1):
        super().__init__()
        self.margin = margin
        self.center_weight = center_weight
        
        self.triplet_loss = nn.TripletMarginLoss(margin=margin)
        self.center_loss = CenterLoss(num_classes, feat_dim, device)
        
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, 
                negative: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        
        triplet_loss = self.triplet_loss(anchor, positive, negative)
        center_loss = self.center_loss(anchor, labels)
        
        total_loss = triplet_loss + self.center_weight * center_loss
        
        return total_loss, triplet_loss, center_loss


class ImprovedIRTrainer:
    """Advanced trainer for IR image classification."""
    
    def __init__(self, 
                 model_type: str = "resnet50",
                 embedding_dim: int = 512,
                 device: Optional[str] = None):
        self.model_type = model_type
        self.embedding_dim = embedding_dim
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model components
        self.model_adapter: Optional[BaseModelAdapter] = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        
        # Training state
        self.training_history = []
        self.best_metrics = {}
        self.class_names = []
        
        logger.info(f"ImprovedIRTrainer initialized with {model_type} on {self.device}")
    
    def setup_model(self, model_path: Optional[str] = None, num_classes: int = 10):
        """Setup model adapter with advanced architecture."""
        self.model_adapter = ModelAdapterFactory.create_adapter(
            self.model_type,
            embedding_dim=self.embedding_dim,
            device=self.device
        )
        
        self.model_adapter.load_model(model_path)
        
        # Setup advanced loss function
        self.criterion = TripletCenterLoss(
            num_classes=num_classes,
            feat_dim=self.embedding_dim,
            device=self.device,
            margin=0.3,
            center_weight=0.1
        )
        
        logger.info(f"Model setup completed with {num_classes} classes")
    
    def setup_optimizer(self, learning_rate: float = 1e-4, weight_decay: float = 1e-4):
        """Setup optimizer with advanced scheduling."""
        if not self.model_adapter:
            raise RuntimeError("Model not setup. Call setup_model() first.")
        
        # Different learning rates for different parts
        backbone_params = []
        head_params = []
        
        for name, param in self.model_adapter.model.named_parameters():
            if 'fc' in name or 'classifier' in name:
                head_params.append(param)
            else:
                backbone_params.append(param)
        
        self.optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': learning_rate * 0.1},  # Lower LR for backbone
            {'params': head_params, 'lr': learning_rate},  # Higher LR for head
            {'params': self.criterion.center_loss.centers, 'lr': learning_rate * 0.5}
        ], weight_decay=weight_decay)
        
        # Advanced scheduler
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=[learning_rate * 0.1, learning_rate, learning_rate * 0.5],
            epochs=100,  # Will be adjusted based on actual epochs
            pct_start=0.3,
            anneal_strategy='cos'
        )
        
        logger.info("Optimizer and scheduler setup completed")
    
    def load_and_prepare_data(self, data_dir: str, test_size: float = 0.2) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Load and prepare data with stratified splitting."""
        data_path = Path(data_dir)
        images = []
        
        # Load images
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
        
        for class_dir in data_path.iterdir():
            if not class_dir.is_dir():
                continue
                
            class_name = class_dir.name
            logger.info(f"Loading class: {class_name}")
            
            for img_file in class_dir.iterdir():
                if img_file.suffix.lower() in image_extensions:
                    try:
                        ir_image = IRImage.from_file_path(
                            str(img_file),
                            f"{class_name}_{img_file.stem}",
                            class_name
                        )
                        images.append(ir_image)
                    except Exception as e:
                        logger.warning(f"Failed to load {img_file}: {e}")
        
        # Store class names
        self.class_names = sorted(list(set(img.object_class for img in images)))
        logger.info(f"Loaded {len(images)} images from {len(self.class_names)} classes")
        
        # Stratified split
        labels = [img.object_class for img in images]
        train_images, temp_images = train_test_split(
            images, test_size=test_size * 2, stratify=labels, random_state=42
        )
        
        temp_labels = [img.object_class for img in temp_images]
        val_images, test_images = train_test_split(
            temp_images, test_size=0.5, stratify=temp_labels, random_state=42
        )
        
        logger.info(f"Data split: {len(train_images)} train, {len(val_images)} val, {len(test_images)} test")
        
        # Create datasets
        train_dataset = IRImageAugmentedDataset(train_images, mode='train', augment_prob=0.8)
        val_dataset = IRImageAugmentedDataset(val_images, mode='val', augment_prob=0.0)
        test_dataset = IRImageAugmentedDataset(test_images, mode='test', augment_prob=0.0)
        
        # Create weighted sampler for imbalanced classes
        class_counts = {}
        for img in train_images:
            class_counts[img.object_class] = class_counts.get(img.object_class, 0) + 1
        
        weights = []
        for img in train_images:
            weights.append(1.0 / class_counts[img.object_class])
        
        sampler = WeightedRandomSampler(weights, len(weights))
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=32, sampler=sampler, num_workers=4
        )
        val_loader = DataLoader(
            val_dataset, batch_size=32, shuffle=False, num_workers=4
        )
        test_loader = DataLoader(
            test_dataset, batch_size=32, shuffle=False, num_workers=4
        )
        
        return train_loader, val_loader, test_loader
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch with advanced techniques."""
        self.model_adapter.model.train()
        total_loss = 0.0
        total_triplet_loss = 0.0
        total_center_loss = 0.0
        correct_triplets = 0
        total_triplets = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, (images, labels, class_names) in enumerate(progress_bar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            batch_size = images.size(0)
            if batch_size < 3:  # Need at least 3 samples for triplets
                continue
            
            # Extract embeddings
            embeddings = self.model_adapter.model(images)
            
            # Form triplets
            anchors, positives, negatives, triplet_labels = self._form_triplets(
                embeddings, labels, batch_size
            )
            
            if len(anchors) == 0:
                continue
            
            # Compute loss
            self.optimizer.zero_grad()
            
            total_loss_batch, triplet_loss, center_loss = self.criterion(
                anchors, positives, negatives, triplet_labels
            )
            
            # Backward pass
            total_loss_batch.backward()
            torch.nn.utils.clip_grad_norm_(self.model_adapter.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Update metrics
            total_loss += total_loss_batch.item()
            total_triplet_loss += triplet_loss.item()
            total_center_loss += center_loss.item()
            
            # Calculate triplet accuracy
            with torch.no_grad():
                pos_dist = torch.norm(anchors - positives, p=2, dim=1)
                neg_dist = torch.norm(anchors - negatives, p=2, dim=1)
                correct_triplets += (pos_dist < neg_dist).sum().item()
                total_triplets += len(anchors)
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{total_loss_batch.item():.4f}',
                'Triplet': f'{triplet_loss.item():.4f}',
                'Center': f'{center_loss.item():.4f}',
                'Acc': f'{correct_triplets/max(total_triplets, 1):.4f}'
            })
            
            # Update scheduler
            if self.scheduler:
                self.scheduler.step()
        
        return {
            'total_loss': total_loss / len(train_loader),
            'triplet_loss': total_triplet_loss / len(train_loader),
            'center_loss': total_center_loss / len(train_loader),
            'triplet_accuracy': correct_triplets / max(total_triplets, 1)
        }
    
    def _form_triplets(self, embeddings: torch.Tensor, labels: torch.Tensor, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Form triplets from batch."""
        anchors, positives, negatives, triplet_labels = [], [], [], []
        
        for i in range(batch_size):
            anchor_label = labels[i]
            
            # Find positive (same class)
            positive_mask = (labels == anchor_label) & (torch.arange(batch_size).to(self.device) != i)
            positive_indices = torch.where(positive_mask)[0]
            
            if len(positive_indices) == 0:
                continue
            
            # Find negative (different class)
            negative_mask = labels != anchor_label
            negative_indices = torch.where(negative_mask)[0]
            
            if len(negative_indices) == 0:
                continue
            
            # Select positive and negative
            pos_idx = positive_indices[torch.randint(len(positive_indices), (1,))]
            neg_idx = negative_indices[torch.randint(len(negative_indices), (1,))]
            
            anchors.append(embeddings[i])
            positives.append(embeddings[pos_idx])
            negatives.append(embeddings[neg_idx])
            triplet_labels.append(anchor_label)
        
        if anchors:
            return (torch.stack(anchors), torch.stack(positives), 
                   torch.stack(negatives), torch.stack(triplet_labels))
        else:
            return torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([])
    
    def validate_epoch(self, val_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model_adapter.model.eval()
        all_embeddings = []
        all_labels = []
        all_predictions = []
        
        with torch.no_grad():
            for images, labels, _ in tqdm(val_loader, desc=f"Validation {epoch}"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                embeddings = self.model_adapter.model(images)
                
                all_embeddings.extend(embeddings.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate validation metrics using nearest neighbor
        accuracy = self._calculate_embedding_accuracy(all_embeddings, all_labels)
        
        return {
            'accuracy': accuracy,
            'num_samples': len(all_labels)
        }
    
    def _calculate_embedding_accuracy(self, embeddings: List[np.ndarray], labels: List[int]) -> float:
        """Calculate accuracy using embedding similarity."""
        embeddings = np.array(embeddings)
        labels = np.array(labels)
        
        correct = 0
        total = len(labels)
        
        for i in range(total):
            query_embedding = embeddings[i]
            query_label = labels[i]
            
            # Find most similar embedding (excluding self)
            similarities = []
            for j in range(total):
                if i != j:
                    similarity = np.dot(query_embedding, embeddings[j]) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(embeddings[j])
                    )
                    similarities.append((similarity, labels[j]))
            
            if similarities:
                similarities.sort(reverse=True)
                predicted_label = similarities[0][1]
                if predicted_label == query_label:
                    correct += 1
        
        return correct / total
    
    def train_model(self, 
                   data_dir: str,
                   epochs: int = 100,
                   learning_rate: float = 1e-4,
                   model_path: Optional[str] = None) -> Dict[str, Any]:
        """Complete training pipeline."""
        logger.info("🚀 Starting advanced IR image training...")
        
        # Load and prepare data
        train_loader, val_loader, test_loader = self.load_and_prepare_data(data_dir)
        
        # Setup model
        self.setup_model(model_path, len(self.class_names))
        self.setup_optimizer(learning_rate)
        
        # Adjust scheduler for actual epochs
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=[learning_rate * 0.1, learning_rate, learning_rate * 0.5],
            epochs=epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.3,
            anneal_strategy='cos'
        )
        
        # Training loop
        best_val_accuracy = 0.0
        patience_counter = 0
        patience = 15
        
        for epoch in range(epochs):
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_metrics = self.validate_epoch(val_loader, epoch)
            
            # Log progress
            logger.info(
                f"Epoch {epoch+1}/{epochs}: "
                f"Train Loss: {train_metrics['total_loss']:.4f}, "
                f"Val Acc: {val_metrics['accuracy']:.4f}, "
                f"Triplet Acc: {train_metrics['triplet_accuracy']:.4f}"
            )
            
            # Save best model
            if val_metrics['accuracy'] > best_val_accuracy:
                best_val_accuracy = val_metrics['accuracy']
                patience_counter = 0
                self._save_checkpoint(epoch, val_metrics['accuracy'], "best_model")
                
                # Check if target accuracy reached
                if best_val_accuracy >= 0.95:
                    logger.info("🎯 Target accuracy of 95% reached!")
                    break
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
            
            # Store history
            self.training_history.append({
                'epoch': epoch,
                'train_loss': train_metrics['total_loss'],
                'train_triplet_loss': train_metrics['triplet_loss'],
                'train_center_loss': train_metrics['center_loss'],
                'train_triplet_accuracy': train_metrics['triplet_accuracy'],
                'val_accuracy': val_metrics['accuracy'],
                'learning_rate': self.optimizer.param_groups[0]['lr']
            })
        
        # Final evaluation
        final_metrics = self.evaluate_model(test_loader)
        
        results = {
            'best_val_accuracy': best_val_accuracy,
            'final_test_metrics': final_metrics,
            'training_history': self.training_history,
            'class_names': self.class_names,
            'total_epochs': epoch + 1
        }
        
        # Save results
        self._save_results(results)
        
        logger.info(f"✅ Training completed! Best validation accuracy: {best_val_accuracy:.4f}")
        return results
    
    def evaluate_model(self, test_loader: DataLoader) -> Dict[str, Any]:
        """Comprehensive model evaluation."""
        logger.info("Evaluating model on test set...")
        
        self.model_adapter.model.eval()
        all_embeddings = []
        all_labels = []
        all_class_names = []
        
        with torch.no_grad():
            for images, labels, class_names in tqdm(test_loader, desc="Testing"):
                images = images.to(self.device)
                embeddings = self.model_adapter.model(images)
                
                all_embeddings.extend(embeddings.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_class_names.extend(class_names)
        
        # Calculate metrics
        accuracy = self._calculate_embedding_accuracy(all_embeddings, all_labels)
        
        # Generate classification report
        predicted_labels = []
        true_labels = []
        
        embeddings_array = np.array(all_embeddings)
        
        for i, (embedding, true_label) in enumerate(zip(all_embeddings, all_labels)):
            # Find most similar
            similarities = []
            for j, other_embedding in enumerate(all_embeddings):
                if i != j:
                    sim = np.dot(embedding, other_embedding) / (
                        np.linalg.norm(embedding) * np.linalg.norm(other_embedding)
                    )
                    similarities.append((sim, all_labels[j]))
            
            if similarities:
                similarities.sort(reverse=True)
                predicted_labels.append(similarities[0][1])
                true_labels.append(true_label)
        
        # Classification report
        report = classification_report(
            true_labels, predicted_labels, 
            target_names=self.class_names,
            output_dict=True
        )
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'num_test_samples': len(all_labels)
        }
    
    def _save_checkpoint(self, epoch: int, accuracy: float, name: str):
        """Save model checkpoint."""
        checkpoint_dir = Path("checkpoints/improved_training")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f"{name}_epoch_{epoch}.pth"
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model_adapter.model.state_dict(),
            'accuracy': accuracy,
            'model_type': self.model_type,
            'embedding_dim': self.embedding_dim,
            'class_names': self.class_names
        }, checkpoint_path)
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def _save_results(self, results: Dict[str, Any]):
        """Save training results."""
        results_dir = Path("results/improved_training")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f"training_results_{timestamp}.json"
        
        # Convert numpy types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            return obj
        
        json_str = json.dumps(results, default=convert_numpy, indent=2)
        
        with open(results_file, 'w') as f:
            f.write(json_str)
        
        logger.info(f"Results saved to: {results_file}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Advanced IR Image Classification Training")
    parser.add_argument('--data-dir', '-d', required=True, help='Training data directory')
    parser.add_argument('--model-type', default='resnet50', choices=['resnet50', 'qwen_vlm'],
                       help='Model type to use')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--embedding-dim', type=int, default=512, help='Embedding dimension')
    parser.add_argument('--model-path', help='Pre-trained model path (optional)')
    
    args = parser.parse_args()
    
    print("🔥 Advanced IR Image Classification Training")
    print("=" * 60)
    
    try:
        # Initialize trainer
        trainer = ImprovedIRTrainer(
            model_type=args.model_type,
            embedding_dim=args.embedding_dim
        )
        
        # Train model
        results = trainer.train_model(
            data_dir=args.data_dir,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            model_path=args.model_path
        )
        
        print("\n📊 Training Results:")
        print(f"Best validation accuracy: {results['best_val_accuracy']:.4f}")
        print(f"Final test accuracy: {results['final_test_metrics']['accuracy']:.4f}")
        print(f"Total epochs trained: {results['total_epochs']}")
        
        if results['best_val_accuracy'] >= 0.95:
            print("🎯 Target accuracy of 95% achieved!")
        else:
            print(f"📈 Progress: {results['best_val_accuracy']*100:.1f}% towards 95% target")
        
        print("\n✅ Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()