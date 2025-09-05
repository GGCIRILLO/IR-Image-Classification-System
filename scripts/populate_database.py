#!/usr/bin/env python3
"""
Script to populate the database with embeddings from processed images.

Enhanced with ResNet18 support and multi-database configuration.
Ensures only training images are processed (excludes test set).
"""

import argparse
import sys
import os
from pathlib import Path
import logging
from typing import List, Optional, Set
import json
from datetime import datetime
import numpy as np
from PIL import Image


# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.embedding.extractor import EmbeddingExtractor
from src.database.vector_store import ChromaVectorStore
from src.database.db_manager import DatabaseManager, ModelType
from src.data.ir_processor import IRImageProcessor
from src.models.data_models import Embedding

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DatabasePopulator:
    """
    Populates the database with embeddings from processed images.
    
    Enhanced with ResNet18 support and multi-database configuration.
    Ensures only training images are processed (excludes test set).
    """
    
    def __init__(self, model_type: ModelType = ModelType.RESNET50, 
                 model_path: Optional[str] = None,
                 database_path: Optional[str] = None,
                 collection_name: Optional[str] = None):
        """
        Initialize the populator with model type and optional overrides.
        
        Args:
            model_type: Model type (ResNet18 or ResNet50)
            model_path: Optional path to fine-tuned model
            database_path: Optional database path override
            collection_name: Optional collection name override
        """
        self.model_type = model_type
        self.model_path = model_path
        
        # Initialize database manager for multi-database support
        self.db_manager = DatabaseManager(model_type=model_type)
        
        # Use provided overrides or get from database manager
        self.database_path = database_path or self.db_manager.database_path
        self.collection_name = collection_name or self.db_manager.collection_name
        
        # Initialize components
        self.embedding_extractor = EmbeddingExtractor(
            model_type=model_type.value,
            model_path=model_path
        )
        
        self.vector_store = ChromaVectorStore(
            db_path=self.database_path,
            collection_name=self.collection_name
        )
        self.ir_processor = IRImageProcessor()
        
        # Load test set filenames to exclude them
        self.test_set_filenames = self._load_test_set_filenames()
        
        logger.info(f"✅ Initialized database populator for {model_type.value}")
        logger.info(f"   📁 Database: {self.database_path}")
        logger.info(f"   📊 Collection: {self.collection_name}")
        logger.info(f"   🚫 Test images excluded: {len(self.test_set_filenames)}")
        if model_path:
            logger.info(f"   🤖 Fine-tuned model: {model_path}")
    
    def _load_test_set_filenames(self) -> Set[str]:
        """
        Load test set filenames to exclude them from training database.
        
        Returns:
            Set[str]: Set of test image filenames
        """
        test_filenames = set()
        metadata_file = Path("data/test_set_metadata.json")
        
        try:
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                for class_data in metadata.get('classes', {}).values():
                    test_filenames.update(class_data.get('test_files', []))
                
                logger.info(f"📋 Loaded {len(test_filenames)} test image filenames to exclude")
            else:
                logger.warning("⚠️ Test set metadata not found - no images will be excluded")
                
        except Exception as e:
            logger.error(f"❌ Error loading test set metadata: {e}")
            logger.warning("⚠️ Continuing without test set exclusion")
        
        return test_filenames

    def get_training_images(self, processed_dir: str, max_per_class: int = 5, 
                           max_total: int = 50) -> List[tuple]:
        """
        Get training images from the processed dataset, excluding test set images.
        
        Args:
            processed_dir: Directory containing processed images
            max_per_class: Maximum images per class (0 for all)
            max_total: Maximum total images (0 for all)
            
        Returns:
            List[tuple]: List of (image_path, class_name) tuples
        """
        processed_path = Path(processed_dir)
        if not processed_path.exists():
            raise FileNotFoundError(f"Directory not found: {processed_dir}")

        image_files = []
        excluded_count = 0
        
        for class_dir in processed_path.iterdir():
            if not class_dir.is_dir():
                continue
                
            class_name = class_dir.name
            class_images = []
            class_excluded = 0
            
            # Get images from this class, excluding test set images
            for img_file in class_dir.iterdir():
                if img_file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                    # Check if this image is in the test set
                    if img_file.name in self.test_set_filenames:
                        class_excluded += 1
                        excluded_count += 1
                        continue
                    
                    class_images.append((str(img_file), class_name))
                    # Only break if we're limiting per class and have reached the limit
                    if max_per_class > 0 and len(class_images) >= max_per_class:
                        break
            
            image_files.extend(class_images)
            logger.info(f"📁 Class '{class_name}': {len(class_images)} training images "
                       f"({class_excluded} test images excluded)")
            
            # Only break if we're limiting total and have reached the limit
            if max_total > 0 and len(image_files) >= max_total:
                break
        
        logger.info(f"🚫 Total test images excluded: {excluded_count}")
        
        # Only limit if max_total is positive
        return image_files if max_total <= 0 else image_files[:max_total]
    
    def populate_database(self, processed_dir: str, max_per_class: int = 5, 
                         max_total: int = 50, dry_run: bool = False):
        """
        Populate the database with embeddings from training images only.
        
        Args:
            processed_dir: Directory containing processed images
            max_per_class: Maximum images per class (0 for all)
            max_total: Maximum total images (0 for all)
            dry_run: If True, only show what would be processed
        """
        
        logger.info(f"🔄 Starting database population for {self.model_type.value}...")
        logger.info(f"   📊 Max per class: {max_per_class}")
        logger.info(f"   📊 Max total: {max_total}")
        logger.info(f"   🧪 Dry run: {dry_run}")
        logger.info(f"   📁 Database: {self.database_path}")
        logger.info(f"   📊 Collection: {self.collection_name}")
        
        # Get training images (excluding test set)
        image_files = self.get_training_images(processed_dir, max_per_class, max_total)
        logger.info(f"📸 Found {len(image_files)} training images to process")
        
        if dry_run:
            logger.info("🧪 DRY RUN - No embeddings will be saved")
            for img_path, class_name in image_files:
                logger.info(f"   📸 {class_name}: {Path(img_path).name}")
            return
        
        # Initialize components
        logger.info(f"🤖 Initializing {self.model_type.value} embedding extractor...")
        self.embedding_extractor.load_model(self.model_path)

        logger.info("🗄️ Initializing database...")
        # Get model-specific configuration
        db_config = self.db_manager._get_model_database_config(self.model_type)
        config = {
            'embedding_dimension': db_config.get('embedding_dimension', 512),
            'metric': db_config.get('distance_metric', 'cosine')
        }
        self.vector_store.initialize_database(config)
        
        successful = 0
        failed = 0
        
        for i, (img_path, class_name) in enumerate(image_files, 1):
            try:
                logger.info(f"[{i}/{len(image_files)}] Processing {Path(img_path).name} ({class_name})")

                # Load and process image (simplified for processed images)
                try:                    
                    # Load image
                    pil_image = Image.open(img_path).convert('L')  # Convert to grayscale
                    logger.info(f"   📸 Loaded {Path(img_path).name}: {pil_image.size}")
                    
                    # Convert to numpy array and normalize to 0-1 range
                    image_array = np.array(pil_image, dtype=np.float32) / 255.0
                    
                    # Use the same IR processor as in query processing
                    from src.data.ir_processor import IRImageProcessor
                    ir_processor = IRImageProcessor(target_size=(224, 224))
                    
                    # Apply the same preprocessing pipeline
                    processed_image = ir_processor.preprocess_ir_image(image_array)
                    
                except Exception as load_error:
                    raise ValueError(f"Impossible to load image: {load_error}")

                # Extract embedding
                embedding_vector = self.embedding_extractor.extract_embedding(processed_image)
                
                # Generate unique ID
                image_id = f"{class_name}_{Path(img_path).stem}"
                embedding_id = f"emb_{image_id}"
                
                # Create Embedding object
                embedding = Embedding(
                    id=embedding_id,
                    image_id=image_id,
                    vector=embedding_vector,
                    model_version=self.model_type.value,
                    extraction_timestamp=datetime.now()
                )
                
                # Store in database
                success = self.vector_store.store_embedding(embedding)
                
                if success:
                    successful += 1
                    logger.info(f"✅ Saved embedding for {image_id}")
                else:
                    failed += 1
                    logger.error(f"❌ Failed to save embedding for {image_id}")

            except Exception as e:
                failed += 1
                logger.error(f"❌ Error processing {img_path}: {e}")

        # Summary
        logger.info(f"📊 Seeding completed:")
        logger.info(f"   ✅ Success: {successful}")
        logger.info(f"   ❌ Failed: {failed}")
        logger.info(f"   📝 Total: {len(image_files)}")

    def verify_database(self):
        """Verify the database by checking the number of embeddings."""
        logger.info("🔍 Verifying database...")
        
        try:
            config = {'embedding_dimension': 512, 'metric': 'cosine'}
            self.vector_store.initialize_database(config)
            
            # Try to get count using ChromaDB directly
            if self.vector_store._collection:
                count = self.vector_store._collection.count()
                logger.info(f"📊 Database: {self.database_path}")
                logger.info(f"   📁 ir_embeddings: {count} items")
            else:
                logger.warning("❌ Collection not available")

            return True
            
        except Exception as e:
            logger.error(f"❌ Error verifying database: {e}")
            return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Populates the database with embeddings from processed images"
    )
    
    parser.add_argument(
        "--database-path",
        default="data/vector_db",
        help="Path to the vector database (default: data/vector_db)"
    )
    
    parser.add_argument(
        "--processed-dir",
        default="data/processed",
        help="Directory with processed images (default: data/processed)"
    )
    
    parser.add_argument(
        "--model",
        default="resnet50",
        choices=["resnet18", "resnet50"],
        help="Model for embedding (default: resnet50, choices: resnet18, resnet50)"
    )
    
    parser.add_argument(
        "--model-path",
        help="Path to the fine-tuned model (optional, default: None)"
    )
    
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=5,
        help="Maximum number of images per class (default: 5, 0 for all images per class)"
    )
    
    parser.add_argument(
        "--max-total",
        type=int,
        default=50,
        help="Maximum total number of images (default: 50, 0 for all)"
    )
    
    parser.add_argument(
        "--all-images",
        action="store_true",
        help="Process all images (equivalent to --max-per-class 0 --max-total 0)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without saving (only shows what it would do)"
    )
    
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify the existing database (do not populate)"
    )
    
    args = parser.parse_args()
    
    try:
        # Convert model string to ModelType enum
        model_type = ModelType.RESNET18 if args.model == "resnet18" else ModelType.RESNET50
        
        # Initialize populator
        populator = DatabasePopulator(
            model_type=model_type,
            model_path=args.model_path,
            database_path=args.database_path if args.database_path != "data/vector_db" else None,
            collection_name=None  # Let DatabaseManager handle model-specific naming
        )
        
        if args.verify_only:
            populator.verify_database()
        else:
            # If all-images flag is set, use 0 for both limits to process all images
            max_per_class = 0 if args.all_images else args.max_per_class
            max_total = 0 if args.all_images else args.max_total
            
            populator.populate_database(
                processed_dir=args.processed_dir,
                max_per_class=max_per_class,
                max_total=max_total,
                dry_run=args.dry_run
            )
        
    except KeyboardInterrupt:
        logger.info("❌ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
