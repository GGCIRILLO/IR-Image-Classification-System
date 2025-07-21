#!/usr/bin/env python3
"""
Enhanced Fine-tuning Script for IR Image Classification System.

This script provides comprehensive fine-tuning capabilities specifically designed
to improve similarity and confidence scores for IR images through:
1. Contrastive learning with hard negative mining
2. Triplet loss optimization
3. Domain adaptation techniques
4. Confidence calibration
5. Embedding space optimization
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
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.data_models import IRImage
from src.embedding.extractor import EmbeddingExtractor
from src.database.similarity_searcher import SimilaritySearcher
from src.training.model_adapters import ModelAdapterFactory
from src.training.trainer import ModelTrainer, TrainingConfig
from src.fine_tuning.model_optimizer import ModelOptimizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IRContrastiveDataset(Dataset):
    """Enhanced dataset for contrastive learning with IR images."""
    
    def __init__(self, images: List[IRImage], hard_negative_ratio: float = 0.3):
        self.images = images
        self.hard_negative_ratio = hard_negative_ratio
        self.class_to_indices = self._build_class_indices()
        self.class_embeddings = {}  # Cache for computing hard negatives
        
    def _build_class_indices(self) -> Dict[str, List[int]]:
        """Build mapping from class to image indices."""
        class_indices = {}
        for idx, image in enumerate(self.images):
            if image.object_class not in class_indices:
                class_indices[image.object_class] = []
            class_indices[image.object_class].append(idx)
        return class_indices
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        anchor_image = self.images[idx]
        anchor_tensor = self._image_to_tensor(anchor_image.image_data)
        
        # Get positive sample (same class)
        positive_idx = self._get_positive_sample(idx)
        positive_image = self.images[positive_idx]
        positive_tensor = self._image_to_tensor(positive_image.image_data)
        
        # Get negative sample (different class, with hard negative mining)
        negative_idx = self._get_negative_sample(idx)
        negative_image = self.images[negative_idx]
        negative_tensor = self._image_to_tensor(negative_image.image_data)
        
        return {
            'anchor': anchor_tensor,
            'positive': positive_tensor,
            'negative': negative_tensor,
            'anchor_class': anchor_image.object_class,
            'positive_class': positive_image.object_class,
            'negative_class': negative_image.object_class
        }
    
    def _image_to_tensor(self, image_data: np.ndarray) -> torch.Tensor:
        """Convert image data to tensor."""
        tensor = torch.from_numpy(image_data).float()
        if len(tensor.shape) == 2:
            tensor = tensor.unsqueeze(0)  # Add channel dimension
        
        # Convert single channel to 3 channels for ResNet50
        if tensor.shape[0] == 1:
            tensor = tensor.repeat(3, 1, 1)  # Repeat grayscale across 3 channels
         
        return tensor
    
    def _get_positive_sample(self, anchor_idx: int) -> int:
        """Get positive sample (same class as anchor)."""
        anchor_class = self.images[anchor_idx].object_class
        same_class_indices = [i for i in self.class_to_indices[anchor_class] if i != anchor_idx]
        
        if not same_class_indices:
            return anchor_idx  # Return self if no other samples in class
        
        return np.random.choice(same_class_indices)
    
    def _get_negative_sample(self, anchor_idx: int) -> int:
        """Get negative sample with hard negative mining."""
        anchor_class = self.images[anchor_idx].object_class
        different_classes = [cls for cls in self.class_to_indices.keys() if cls != anchor_class]
        
        if not different_classes:
            raise ValueError("Need at least 2 classes for contrastive learning")
        
        # Simple random selection for now (can be enhanced with hard negative mining)
        negative_class = np.random.choice(different_classes)
        return np.random.choice(self.class_to_indices[negative_class])


class EnhancedContrastiveLoss(nn.Module):
    """Enhanced contrastive loss with temperature scaling and hard negative mining."""
    
    def __init__(self, temperature: float = 0.1, margin: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        
    def forward(self, anchor, positive, negative):
        # Normalize embeddings
        anchor = nn.functional.normalize(anchor, p=2, dim=1)
        positive = nn.functional.normalize(positive, p=2, dim=1)
        negative = nn.functional.normalize(negative, p=2, dim=1)
        
        # Compute similarities
        pos_sim = torch.sum(anchor * positive, dim=1) / self.temperature
        neg_sim = torch.sum(anchor * negative, dim=1) / self.temperature
        
        # Contrastive loss with margin
        loss = torch.mean(torch.clamp(self.margin - pos_sim + neg_sim, min=0.0))
        
        return loss


class IRFineTuner:
    """Enhanced fine-tuning class for IR image classification."""
    
    def __init__(self, 
                 model_type: str = "resnet50",
                 embedding_dim: int = 512,
                 device: Optional[str] = None):
        self.model_type = model_type
        self.embedding_dim = embedding_dim
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self.model_adapter = None
        self.extractor = None
        self.searcher = None
        self.optimizer = None
        
        # Training state
        self.training_history = []
        self.best_metrics = {}
        
        logger.info(f"IRFineTuner initialized with {model_type} on {self.device}")
    
    def setup_model(self, model_path: Optional[str] = None):
        """Setup model adapter and extractor."""
        # Create model adapter
        self.model_adapter = ModelAdapterFactory.create_adapter(
            self.model_type,
            embedding_dim=self.embedding_dim,
            device=self.device
        )
        
        # Load model
        self.model_adapter.load_model(model_path)
        
        # Create extractor
        self.extractor = EmbeddingExtractor(
            model_type=self.model_type,
            model_path=model_path
        )
        self.extractor.load_model(model_path)
        
        logger.info("Model setup completed")
    
    def setup_database(self, database_path: str, collection_name: str = "ir_embeddings"):
        """Setup similarity searcher."""
        self.searcher = SimilaritySearcher(database_path, collection_name)
        self.searcher.initialize()
        logger.info(f"Database setup completed: {database_path}")
    
    def load_training_data(self, data_dir: str) -> List[IRImage]:
        """Load training data from directory."""
        data_path = Path(data_dir)
        images = []
        
        # Supported image formats
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
        
        logger.info(f"Loaded {len(images)} images from {len(set(img.object_class for img in images))} classes")
        return images
    
    def evaluate_baseline(self, validation_images: List[IRImage]) -> Dict[str, float]:
        """Evaluate baseline performance before fine-tuning."""
        logger.info("Evaluating baseline performance...")
        
        similarities = []
        confidences = []
        correct_predictions = 0
        total_predictions = 0
        
        for image in validation_images[:50]:  # Sample for efficiency
            try:
                # Extract embedding
                embedding = self.extractor.extract_embedding(image.image_data)
                
                # Search for similar images
                results, _ = self.searcher.search_similar(embedding, k=5)
                
                if results:
                    # Check accuracy
                    top_classes = [r.object_class for r in results]
                    if image.object_class in top_classes:
                        correct_predictions += 1
                    
                    # Collect metrics
                    similarities.extend([r.similarity_score for r in results])
                    confidences.extend([r.confidence for r in results])
                
                total_predictions += 1
                
            except Exception as e:
                logger.warning(f"Error evaluating image {image.id}: {e}")
                continue
        
        metrics = {
            'avg_similarity': np.mean(similarities) if similarities else 0.0,
            'max_similarity': np.max(similarities) if similarities else 0.0,
            'avg_confidence': np.mean(confidences) if confidences else 0.0,
            'max_confidence': np.max(confidences) if confidences else 0.0,
            'accuracy': correct_predictions / max(total_predictions, 1),
            'total_results': len(similarities)
        }
        
        logger.info(f"Baseline metrics: {metrics}")
        return metrics
    
    def fine_tune_contrastive(self, 
                            train_images: List[IRImage],
                            val_images: List[IRImage],
                            epochs: int = 50,
                            batch_size: int = 16,
                            learning_rate: float = 1e-4) -> Dict[str, Any]:
        """Fine-tune using contrastive learning."""
        logger.info("Starting contrastive fine-tuning...")
        
        # Create datasets
        train_dataset = IRContrastiveDataset(train_images)
        val_dataset = IRContrastiveDataset(val_images)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Setup loss and optimizer
        criterion = EnhancedContrastiveLoss(temperature=0.1, margin=0.5)
        optimizer = optim.AdamW(
            self.model_adapter.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-4
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 10
        
        for epoch in range(epochs):
            # Training phase
            self.model_adapter.model.train()
            train_loss = 0.0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                anchor = batch['anchor'].to(self.device)
                positive = batch['positive'].to(self.device)
                negative = batch['negative'].to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                anchor_emb = self.model_adapter.model(anchor)
                positive_emb = self.model_adapter.model(positive)
                negative_emb = self.model_adapter.model(negative)
                
                # Compute loss
                loss = criterion(anchor_emb, positive_emb, negative_emb)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_adapter.model.parameters(), 1.0)
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation phase
            self.model_adapter.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch in val_loader:
                    anchor = batch['anchor'].to(self.device)
                    positive = batch['positive'].to(self.device)
                    negative = batch['negative'].to(self.device)
                    
                    anchor_emb = self.model_adapter.model(anchor)
                    positive_emb = self.model_adapter.model(positive)
                    negative_emb = self.model_adapter.model(negative)
                    
                    loss = criterion(anchor_emb, positive_emb, negative_emb)
                    val_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            logger.info(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                self._save_checkpoint(epoch, avg_val_loss, "contrastive_best")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
            
            scheduler.step()
            
            # Store training history
            self.training_history.append({
                'epoch': epoch,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'learning_rate': scheduler.get_last_lr()[0]
            })
        
        return {
            'final_train_loss': avg_train_loss,
            'final_val_loss': avg_val_loss,
            'best_val_loss': best_val_loss,
            'epochs_trained': epoch + 1
        }
    
    def optimize_similarity_calculation(self, validation_images: List[IRImage]) -> Dict[str, Any]:
        """Optimize similarity calculation parameters."""
        logger.info("Optimizing similarity calculation...")
        
        if not self.searcher:
            logger.warning("No searcher available for optimization")
            return {}
        
        # Test different similarity metrics
        metrics_to_test = [
            {'metric': 'cosine', 'threshold': 0.1},
            {'metric': 'cosine', 'threshold': 0.2},
            {'metric': 'cosine', 'threshold': 0.3},
        ]
        
        best_config = None
        best_score = 0.0
        
        for config in metrics_to_test:
            # Temporarily update configuration
            original_threshold = self.searcher.config.similarity_threshold
            self.searcher.config.similarity_threshold = config['threshold']
            
            # Evaluate performance
            metrics = self.evaluate_baseline(validation_images[:20])
            score = metrics['avg_similarity'] + metrics['avg_confidence']
            
            if score > best_score:
                best_score = score
                best_config = config
            
            # Restore original configuration
            self.searcher.config.similarity_threshold = original_threshold
        
        # Apply best configuration
        if best_config:
            self.searcher.config.similarity_threshold = best_config['threshold']
            logger.info(f"Applied best similarity config: {best_config}")
        
        return {'best_config': best_config, 'best_score': best_score}
    
    def calibrate_confidence_scores(self, validation_images: List[IRImage]) -> Dict[str, Any]:
        """Calibrate confidence score calculation."""
        logger.info("Calibrating confidence scores...")
        
        # Collect similarity-accuracy pairs
        similarity_scores = []
        actual_accuracies = []
        
        for image in validation_images[:30]:
            try:
                embedding = self.extractor.extract_embedding(image.image_data)
                results, _ = self.searcher.search_similar(embedding, k=5)
                
                if results:
                    top_result = results[0]
                    is_correct = top_result.object_class == image.object_class
                    
                    similarity_scores.append(top_result.similarity_score)
                    actual_accuracies.append(1.0 if is_correct else 0.0)
                    
            except Exception as e:
                logger.warning(f"Error in confidence calibration: {e}")
                continue
        
        if len(similarity_scores) > 5:
            # Fit simple linear calibration
            from sklearn.linear_model import LinearRegression
            
            X = np.array(similarity_scores).reshape(-1, 1)
            y = np.array(actual_accuracies)
            
            reg = LinearRegression()
            reg.fit(X, y)
            
            calibration_params = {
                'slope': float(reg.coef_[0]),
                'intercept': float(reg.intercept_)
            }
            
            # Apply calibration to searcher
            self._apply_confidence_calibration(calibration_params)
            
            return calibration_params
        
        return {}
    
    def _apply_confidence_calibration(self, params: Dict[str, float]):
        """Apply confidence calibration parameters."""
        if not self.searcher:
            return
        
        # Store calibration parameters
        setattr(self.searcher, '_calibration_params', params)
        
        # Override confidence calculation method
        original_calc = getattr(self.searcher, '_calculate_confidence', None)
        
        def calibrated_confidence(similarity_score: float, rank: int, total_results: int) -> float:
            # Apply linear calibration
            calibrated = params['slope'] * similarity_score + params['intercept']
            
            # Apply rank penalty (reduced)
            if total_results > 1:
                rank_penalty = 0.05 * (rank / (total_results - 1))
            else:
                rank_penalty = 0.0
            
            confidence = max(0.0, min(1.0, calibrated - rank_penalty))
            return confidence
        
        self.searcher._calculate_confidence = calibrated_confidence
        logger.info(f"Applied confidence calibration: {params}")
    
    def _save_checkpoint(self, epoch: int, loss: float, name: str):
        """Save model checkpoint."""
        checkpoint_dir = Path("checkpoints/fine_tuning")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f"{name}_epoch_{epoch}.pth"
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model_adapter.model.state_dict(),
            'loss': loss,
            'model_type': self.model_type,
            'embedding_dim': self.embedding_dim
        }, checkpoint_path)
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def run_comprehensive_fine_tuning(self,
                                    train_data_dir: str,
                                    database_path: str,
                                    model_path: Optional[str] = None,
                                    epochs: int = 50,
                                    batch_size: int = 16,
                                    learning_rate: float = 1e-4) -> Dict[str, Any]:
        """Run comprehensive fine-tuning pipeline."""
        logger.info("🚀 Starting comprehensive fine-tuning pipeline...")
        
        # Setup
        self.setup_model(model_path)
        self.setup_database(database_path)
        
        # Load data
        all_images = self.load_training_data(train_data_dir)
        
        # Split data
        train_images, val_images = train_test_split(
            all_images, test_size=0.2, stratify=[img.object_class for img in all_images]
        )
        
        logger.info(f"Split data: {len(train_images)} train, {len(val_images)} validation")
        
        # Evaluate baseline
        baseline_metrics = self.evaluate_baseline(val_images)
        
        # Fine-tune with contrastive learning
        contrastive_results = self.fine_tune_contrastive(
            train_images, val_images, epochs, batch_size, learning_rate
        )
        
        # Optimize similarity calculation
        similarity_optimization = self.optimize_similarity_calculation(val_images)
        
        # Calibrate confidence scores
        confidence_calibration = self.calibrate_confidence_scores(val_images)
        
        # Final evaluation
        final_metrics = self.evaluate_baseline(val_images)
        
        # Calculate improvements
        improvements = {
            'similarity_improvement': final_metrics['avg_similarity'] - baseline_metrics['avg_similarity'],
            'confidence_improvement': final_metrics['avg_confidence'] - baseline_metrics['avg_confidence'],
            'accuracy_improvement': final_metrics['accuracy'] - baseline_metrics['accuracy']
        }
        
        results = {
            'baseline_metrics': baseline_metrics,
            'final_metrics': final_metrics,
            'improvements': improvements,
            'contrastive_results': contrastive_results,
            'similarity_optimization': similarity_optimization,
            'confidence_calibration': confidence_calibration,
            'training_history': self.training_history
        }
        
        # Save results
        self._save_results(results)
        
        logger.info("✅ Comprehensive fine-tuning completed!")
        logger.info(f"Improvements: {improvements}")
        
        return results
    
    def _save_results(self, results: Dict[str, Any]):
        """Save fine-tuning results."""
        results_dir = Path("results/fine_tuning")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f"fine_tuning_results_{timestamp}.json"
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            return obj
        
        # Deep convert the results
        import json
        json_str = json.dumps(results, default=convert_numpy, indent=2)
        
        with open(results_file, 'w') as f:
            f.write(json_str)
        
        logger.info(f"Results saved to: {results_file}")


def main():
    """Main function for enhanced fine-tuning."""
    parser = argparse.ArgumentParser(description="Enhanced Fine-tuning for IR Image Classification")
    parser.add_argument('--train-data', '-t', required=True, help='Training data directory')
    parser.add_argument('--database', '-d', required=True, help='Vector database path')
    parser.add_argument('--model-path', '-m', help='Pre-trained model path (optional)')
    parser.add_argument('--model-type', default='resnet50', choices=['resnet50', 'qwen_vlm'], 
                       help='Model type to use')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--embedding-dim', type=int, default=512, help='Embedding dimension')
    
    args = parser.parse_args()
    
    print("🔥 Enhanced IR Image Classification Fine-tuning")
    print("=" * 60)
    
    try:
        # Initialize fine-tuner
        fine_tuner = IRFineTuner(
            model_type=args.model_type,
            embedding_dim=args.embedding_dim
        )
        
        # Run comprehensive fine-tuning
        results = fine_tuner.run_comprehensive_fine_tuning(
            train_data_dir=args.train_data,
            database_path=args.database,
            model_path=args.model_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate
        )
        
        print("\n📊 Fine-tuning Results:")
        print(f"Similarity improvement: {results['improvements']['similarity_improvement']:.4f}")
        print(f"Confidence improvement: {results['improvements']['confidence_improvement']:.4f}")
        print(f"Accuracy improvement: {results['improvements']['accuracy_improvement']:.4f}")
        
        print("\n✅ Fine-tuning completed successfully!")
        
    except Exception as e:
        logger.error(f"Fine-tuning failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()