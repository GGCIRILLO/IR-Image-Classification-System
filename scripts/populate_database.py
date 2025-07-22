#!/usr/bin/env python3
"""
Script per popolare il database con embedding dalle immagini processate.
"""

import argparse
import sys
import os
from pathlib import Path
import logging
from typing import List, Optional
import json
from datetime import datetime
import numpy as np
from PIL import Image


# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.embedding.extractor import EmbeddingExtractor
from src.database.vector_store import ChromaVectorStore
from src.data.ir_processor import IRImageProcessor
from src.models.data_models import Embedding

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DatabasePopulator:
    """Popola il database con embedding dalle immagini processate."""
    
    def __init__(self, database_path: str, model_type: str = "resnet50", model_path: str = None):
        """Inizializza il popolatore del database."""
        self.database_path = database_path
        self.model_type = model_type
        self.model_path = model_path
        
        # Initialize components
        self.embedding_extractor = EmbeddingExtractor(
            model_type=model_type,
            model_path=model_path  # Use custom model if provided
        )
        
        self.vector_store = ChromaVectorStore(
            db_path=database_path,
            collection_name="ir_embeddings"
        )
        self.ir_processor = IRImageProcessor()
        
        logger.info(f"✅ Inizializzato popolatore database: {database_path}")
        if model_path:
            logger.info(f"✅ Utilizzo modello fine-tuned: {model_path}")
    
    def get_sample_images(self, processed_dir: str, max_per_class: int = 5, 
                         max_total: int = 50) -> List[tuple]:
        """Ottieni immagini di esempio dal dataset processato."""
        processed_path = Path(processed_dir)
        if not processed_path.exists():
            raise FileNotFoundError(f"Directory processata non trovata: {processed_dir}")
        
        image_files = []
        
        for class_dir in processed_path.iterdir():
            if not class_dir.is_dir():
                continue
                
            class_name = class_dir.name
            class_images = []
            
            # Get images from this class
            for img_file in class_dir.iterdir():
                if img_file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                    class_images.append((str(img_file), class_name))
                    # Only break if we're limiting per class and have reached the limit
                    if max_per_class > 0 and len(class_images) >= max_per_class:
                        break
            
            image_files.extend(class_images)
            logger.info(f"📁 Classe '{class_name}': {len(class_images)} immagini")
            
            # Only break if we're limiting total and have reached the limit
            if max_total > 0 and len(image_files) >= max_total:
                break
        
        # Only limit if max_total is positive
        return image_files if max_total <= 0 else image_files[:max_total]
    
    def populate_database(self, processed_dir: str, max_per_class: int = 5, 
                         max_total: int = 50, dry_run: bool = False):
        """Popola il database con embedding dalle immagini."""
        
        logger.info(f"🔄 Inizio popolamento database...")
        logger.info(f"   📊 Max per classe: {max_per_class}")
        logger.info(f"   📊 Max totale: {max_total}")
        logger.info(f"   🧪 Dry run: {dry_run}")
        
        # Get sample images
        image_files = self.get_sample_images(processed_dir, max_per_class, max_total)
        logger.info(f"📸 Trovate {len(image_files)} immagini da processare")
        
        if dry_run:
            logger.info("🧪 DRY RUN - Nessun embedding sarà salvato")
            for img_path, class_name in image_files:
                logger.info(f"   📸 {class_name}: {Path(img_path).name}")
            return
        
        # Initialize components
        logger.info("🤖 Inizializzazione estrattore embedding...")
        self.embedding_extractor.load_model(self.model_path)
        
        logger.info("🗄️ Inizializzazione database...")
        config = {
            'embedding_dimension': 512,  # ResNet50 final embedding dimension
            'metric': 'cosine'
        }
        self.vector_store.initialize_database(config)
        
        successful = 0
        failed = 0
        
        for i, (img_path, class_name) in enumerate(image_files, 1):
            try:
                logger.info(f"[{i}/{len(image_files)}] Processando {Path(img_path).name} ({class_name})")
                
                # Load and process image (simplified for processed images)
                try:                    
                    # Load image
                    pil_image = Image.open(img_path).convert('L')  # Convert to grayscale
                    logger.info(f"   📸 Caricato {Path(img_path).name}: {pil_image.size}")
                    
                    # Convert to numpy array and normalize to 0-1 range
                    image_array = np.array(pil_image, dtype=np.float32) / 255.0
                    
                    # Use the same IR processor as in query processing
                    from src.data.ir_processor import IRImageProcessor
                    ir_processor = IRImageProcessor(target_size=(224, 224))
                    
                    # Apply the same preprocessing pipeline
                    processed_image = ir_processor.preprocess_ir_image(image_array)
                    
                except Exception as load_error:
                    raise ValueError(f"Impossibile caricare l'immagine: {load_error}")
                
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
                    model_version=self.model_type,
                    extraction_timestamp=datetime.now()
                )
                
                # Store in database
                success = self.vector_store.store_embedding(embedding)
                
                if success:
                    successful += 1
                    logger.info(f"✅ Salvato embedding per {image_id}")
                else:
                    failed += 1
                    logger.error(f"❌ Fallito salvataggio embedding per {image_id}")
                
            except Exception as e:
                failed += 1
                logger.error(f"❌ Errore processando {img_path}: {e}")
        
        # Summary
        logger.info(f"📊 Popolamento completato:")
        logger.info(f"   ✅ Successo: {successful}")
        logger.info(f"   ❌ Falliti: {failed}")
        logger.info(f"   📝 Totale: {len(image_files)}")
    
    def verify_database(self):
        """Verifica il contenuto del database."""
        logger.info("� Verifica database...")
        
        try:
            config = {'embedding_dimension': 512, 'metric': 'cosine'}
            self.vector_store.initialize_database(config)
            
            # Try to get count using ChromaDB directly
            if self.vector_store._collection:
                count = self.vector_store._collection.count()
                logger.info(f"📊 Database: {self.database_path}")
                logger.info(f"   📁 ir_embeddings: {count} items")
            else:
                logger.warning("❌ Collection non disponibile")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Errore verifica database: {e}")
            return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Popola il database con embedding dalle immagini processate"
    )
    
    parser.add_argument(
        "--database-path",
        default="data/vector_db",
        help="Path al database vector (default: data/chroma_db_final)"
    )
    
    parser.add_argument(
        "--processed-dir",
        default="data/processed",
        help="Directory con immagini processate (default: data/processed)"
    )
    
    parser.add_argument(
        "--model",
        default="resnet50",
        help="Modello per embedding (default: resnet50)"
    )
    
    parser.add_argument(
        "--model-path",
        help="Path al modello fine-tuned (opzionale)"
    )
    
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=5,
        help="Massimo numero di immagini per classe (default: 5, 0 per tutte)"
    )
    
    parser.add_argument(
        "--max-total",
        type=int,
        default=50,
        help="Massimo numero totale di immagini (default: 50, 0 per tutte)"
    )
    
    parser.add_argument(
        "--all-images",
        action="store_true",
        help="Processa tutte le immagini (equivalente a --max-per-class 0 --max-total 0)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Esegui senza salvare (solo visualizza quello che farebbe)"
    )
    
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Solo verifica il database esistente"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize populator
        populator = DatabasePopulator(
            database_path=args.database_path,
            model_type=args.model,
            model_path=args.model_path
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
        logger.info("❌ Interrotto dall'utente")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Errore: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
